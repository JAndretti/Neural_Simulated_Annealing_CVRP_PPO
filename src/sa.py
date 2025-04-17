from typing import Dict
import torch
from model import SAModel
from problem import Problem
from replay import ReplayBuffer
from utils import extend_to
from scheduler import Scheduler


def p_accept(gain: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
    """
    Compute the Metropolis acceptance probability for a proposed move.

    Args:
        gain: Energy difference (current_cost - proposed_cost)
        temp: Current temperature

    Returns:
        Probability to accept the move, following exp(gain/temp) with clipping to [0,1]
    """
    return torch.minimum(torch.exp(gain / temp), torch.ones_like(gain))


def sa(
    actor: SAModel,
    problem: Problem,
    initial_solution: torch.Tensor,
    config: dict,
    baseline: bool = False,
    random_std: float = 0.2,
    greedy: bool = False,
    record_state: bool = False,
    replay_buffer: ReplayBuffer = None,
) -> Dict[str, torch.Tensor]:
    """
    Perform Simulated Annealing optimization for combinatorial problems.

    Args:
        actor: Policy network proposing solution modifications
        problem: Problem definition (e.g., CVRP)
        initial_solution: Starting solution tensor
        config: Dictionary containing hyperparameters:
               - INIT_TEMP: Starting temperature
               - STOP_TEMP: Final temperature
               - OUTER_STEPS: Number of temperature steps
               - INNER_STEPS: Steps per temperature
               - SCHEDULER: Cooling schedule type
               - METHOD: Optimization method
               - REWARD: Reward calculation method
        baseline: Whether to use random actions instead of actor
        random_std: Standard deviation for random actions
        greedy: Whether to use greedy action selection
        record_state: Whether to track optimization states
        replay_buffer: Experience replay buffer for RL training

    Returns:
        Dictionary containing optimization results:
        - best_x: Best solution found
        - min_cost: Cost of best solution
        - primal: Cumulative best costs
        - ngain: Negative improvement from initial cost
        - n_acc/n_rej: Count of accepted/rejected moves
        - distributions: Action probabilities (if recorded)
        - states: State history (if recorded)
        - actions: Action history (if recorded)
        - acceptance: Move acceptance (if recorded)
        - costs: Cost history (if recorded)
        - init_cost: Initial solution cost
        - reward: Computed rewards
        - temperature: Temperature schedule
    """
    device = initial_solution.device

    # Initialize temperature parameters
    current_temp = torch.tensor([config["INIT_TEMP"]], device=device)
    next_temp = current_temp

    # Set up cooling schedule
    scheduler = Scheduler(
        config["SCHEDULER"],
        T_max=config["INIT_TEMP"],
        T_min=config["STOP_TEMP"],
        step_max=config["OUTER_STEPS"],
    )

    # Initialize optimization tracking variables
    best_solution = current_solution = initial_solution
    best_cost = problem.cost(best_solution)
    initial_cost = best_cost.clone()
    cumulative_cost = best_cost
    current_cost = best_cost

    # Tracking counters and histories
    accepted_moves = 0
    rejected_moves = 0
    action_distributions = []
    state_history = []
    action_history = []
    temperature = [current_temp]
    acceptance_history = []
    cost_history = [current_cost]
    reward_signal = None
    ratio = 0.0

    # Convert initial solution to state representation
    current_state = problem.to_state(
        current_solution,
        current_temp,
        torch.tensor(1 - (1 / config["OUTER_STEPS"]), device=device),
    ).to(device)

    # Main optimization loop over temperature steps
    for step in range(config["OUTER_STEPS"]):
        # Inner loop at fixed temperature
        for inner_step in range(config["INNER_STEPS"]):
            if record_state:
                state_history.append(current_state)

            # Generate action from policy or baseline
            if baseline:
                action, action_log_prob = actor.baseline_sample(
                    current_state, random_std=random_std, problem=problem
                )
            else:
                action, action_log_prob, validity_mask = actor.sample(
                    current_state, greedy=greedy, problem=problem
                )
            if config["HEURISTIC"] == "mix":
                heuristic_action = sum(action[:, 2])
                ratio += heuristic_action / len(action[:, 2])

            # Record action information if needed
            if record_state:
                action_distributions.append(
                    actor.get_logits(current_state, action, problem=problem)
                )
                action_history.append(action)

            # Generate proposed solution
            solution_components, *_ = problem.from_state(current_state)
            proposed_solution = problem.update(solution_components, action)
            proposed_cost = problem.cost(proposed_solution)

            # Calculate improvement
            cost_improvement = current_cost - proposed_cost

            # Metropolis acceptance criterion
            acceptance_prob = p_accept(cost_improvement, current_temp)
            random_sample = torch.rand(acceptance_prob.shape, device=device)
            is_accepted = (random_sample < acceptance_prob).long()
            actual_improvement = cost_improvement * is_accepted

            # Update counters
            accepted_moves += is_accepted
            rejected_moves += 1 - is_accepted

            if record_state:
                acceptance_history.append(is_accepted)

            # Update current solution
            current_cost = (
                is_accepted * proposed_cost + (1 - is_accepted) * current_cost
            )
            is_accepted_expanded = extend_to(is_accepted, solution_components)
            current_solution = (
                is_accepted_expanded * proposed_solution
                + (1 - is_accepted_expanded) * solution_components
            ).long()

            # Update best solution tracking
            if record_state:
                cost_history.append(current_cost)

            is_improvement = (current_cost < best_cost).long()
            is_improvement_expanded = extend_to(is_improvement, solution_components)
            best_solution = (
                is_improvement_expanded * current_solution
                + (1 - is_improvement_expanded) * best_solution
            )
            best_cost = torch.minimum(current_cost, best_cost)
            cumulative_cost += best_cost

            # Temperature update at end of inner steps
            if inner_step == config["INNER_STEPS"] - 1:
                next_temp = scheduler.step(step).to(device)

                # # Adaptive cooling schedule adjustment
                # if (
                #     max(10, config["OUTER_STEPS"] * 0.1)
                #     == config["OUTER_STEPS"] - step + 1
                # ):
                #     scheduler = Scheduler(
                #         "lam",
                #         T_max=1.0,
                #         T_min=0.01,
                #         step_max=config["OUTER_STEPS"] - step + 1,
                #     )
                #     next_temp = torch.tensor([1], device=device)
                temperature.append(next_temp)
                model_next_temp = next_temp / config["INIT_TEMP"]

            # Prepare next state
            next_state = problem.to_state(
                current_solution,
                model_next_temp,
                torch.tensor(1 - (step / config["OUTER_STEPS"]), device=device),
            )

            # Reward calculation for reinforcement learning
            if config["METHOD"] == "ppo":
                if config["REWARD"] == "immediate":
                    reward_signal = actual_improvement.unsqueeze(1)
                elif config["REWARD"] == "min_cost":
                    reward_signal = -best_cost.view(-1, 1)
                elif config["REWARD"] == "primal":
                    reward_signal = -cumulative_cost.view(-1, 1)

            # Store experience in replay buffer
            if replay_buffer is not None:
                replay_buffer.push(
                    current_state,
                    action,
                    next_state,
                    reward_signal,
                    action_log_prob,
                    validity_mask,
                    config["GAMMA"],
                )

        # Update state and temperature for next iteration
        current_state = next_state.clone()
        current_temp = next_temp

    # Finalize results
    negative_improvement = -(initial_cost - current_cost)

    # Handle final transition in replay buffer
    if replay_buffer is not None and len(replay_buffer) > 0:
        final_transition = replay_buffer.pop()
        replay_buffer.push(*(list(final_transition[:-1]) + [0.0]))

    dict = {
        "best_x": best_solution,
        "min_cost": best_cost,
        "primal": cumulative_cost,
        "ngain": negative_improvement,
        "n_acc": accepted_moves.float(),
        "n_rej": rejected_moves.float(),
        "distributions": action_distributions,
        "states": state_history,
        "actions": action_history,
        "acceptance": acceptance_history,
        "costs": cost_history,
        "init_cost": initial_cost,
        "reward": reward_signal,
        "temperature": temperature,
    }
    if config["HEURISTIC"] == "mix":
        dict["ratio"] = ratio / config["OUTER_STEPS"]
    return dict
