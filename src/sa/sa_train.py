from typing import Dict
import torch
from model import SAModel
from problem import Problem
from utils import extend_to
from scheduler import Scheduler
from tqdm import tqdm


def metropolis_accept(
    cost_improvement: torch.Tensor, current_temp: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the Metropolis acceptance criterion.

    Args:
        cost_improvement: The difference between the current cost and the proposed cost.
        current_temp: The current temperature.
        device: The device to perform the calculations on.

    Returns:
        A tuple containing:
        - is_accepted: A Long tensor indicating acceptance (1) or rejection (0).
        - actual_improvement: The cost improvement achieved,
          considering only the accepted moves.
    """
    # Metropolis acceptance criterion
    acceptance_prob = p_accept(cost_improvement, current_temp)
    random_sample = torch.rand(acceptance_prob.shape, device=device)
    is_accepted = (random_sample < acceptance_prob).long()
    actual_improvement = cost_improvement * is_accepted
    return is_accepted, actual_improvement


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


def scale_between(value: float, min_value: float, max_value: float) -> float:
    """
    Scale a value in [0, 1] to the range [min_value, max_value].

    Args:
        value: Input value in [0, 1]
        min_value: Minimum of target range
        max_value: Maximum of target range

    Returns:
        Scaled value in [min_value, max_value]
    """
    return min_value + (max_value - min_value) * value


def scale_to_unit(value: float, min_value: float, max_value: float) -> float:
    """
    Scale a value in [min_value, max_value] to the range [0, 1].

    Args:
        value: Input value in [min_value, max_value]
        min_value: Minimum of original range
        max_value: Maximum of original range

    Returns:
        Scaled value in [0, 1]
    """
    return (value - min_value) / (max_value - min_value)


def sa_train(
    actor: SAModel,
    problem: Problem,
    initial_solution: torch.Tensor,
    config: dict,
    baseline: bool = False,
    random_std: float = 0.2,
    greedy: bool = False,
    record_state: bool = False,
    replay_buffer=None,
    train: bool = False,
    device: str = None,
    desc_tqdm: str = "Simulated Annealing Progress",
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
    if device is None:
        device = initial_solution.device

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
    capacity_left = problem.capacity_utilization(best_solution)
    initial_cost = best_cost.clone()
    cumulative_cost = best_cost.clone()
    current_cost = best_cost.clone()

    # --------------------------------
    # Track optimization progress
    # --------------------------------
    # Record the step at which the best cost is found for each instance in the batch
    best_cost_step = torch.zeros_like(best_cost, dtype=torch.long)

    # --------------------------------
    # Set initial temperature
    # --------------------------------

    # Use default initial temperature (will be scaled below)
    current_temp = torch.tensor([1], device=device).repeat(best_cost.shape[0])

    # Scale temperature to the configured range [STOP_TEMP, INIT_TEMP]
    current_temp = scale_between(current_temp, config["STOP_TEMP"], config["INIT_TEMP"])
    next_temp = current_temp.clone()  # Initialize next_temp for first iteration

    # --------------------------------
    # Initialize tracking variables
    # --------------------------------
    # Move statistics
    accepted_moves = 0  # Count of accepted moves
    rejected_moves = 0  # Count of rejected moves

    # History containers for analysis and visualization
    action_distributions = []  # Store policy distributions
    state_history = []  # Store states visited during optimization
    action_history = []  # Store actions taken
    is_valid_history = []  # Track validity of actions if needed
    temperature = [current_temp.clone()]  # Store temperature schedule
    heuristic_choice = []  # Track heuristic choices if using mixed approach
    acceptance_history = []  # Track which moves were accepted
    cost_history = [current_cost.clone()]  # Track cost evolution
    reward_signal = None  # Will store rewards for RL training
    ratio = 0.0  # Track ratio of heuristic actions if using mixed approach

    # --------------------------------
    # Prepare initial state representation
    # --------------------------------
    # Convert the initial solution to the state representation expected by the policy
    # Normalize temperature to [0,1] for the state representation
    normalized_temp = scale_to_unit(
        current_temp, config["STOP_TEMP"], config["INIT_TEMP"]
    )

    # Create state components using problem-specific function
    # This typically includes solution encoding, temperature, and progress information
    components = problem.build_state_components(
        current_solution,
        normalized_temp,
        torch.tensor((1), device=device),  # Initial progress (full time remaining)
    )

    current_state = problem.to_state(*components).to(device)
    desc = "Train/ " if train else "Test/ "
    desc += desc_tqdm
    # Main optimization loop over temperature steps
    for step in tqdm(
        range(config["OUTER_STEPS"]),
        desc=desc,
        colour="green",
        unit="step",
        leave=False,
    ):
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
                action, action_log_prob = actor.sample(
                    current_state, greedy=greedy, problem=problem
                )
            if config["HEURISTIC"] == "mix":
                heuristic_action = sum(action[:, 2])
                ratio += heuristic_action / len(action[:, 2])
                heuristic_choice.append(heuristic_action)

            # Record action information if needed
            if record_state:
                action_distributions.append(
                    actor.get_logits(current_state, action, problem=problem)
                )
                action_history.append(action)

            # Generate proposed solution
            solution_components, *_ = problem.from_state(current_state)
            proposed_solution, is_valid = problem.update(solution_components, action)
            proposed_cost = problem.cost(proposed_solution)

            # Metrics is valid
            is_valid_history.append(is_valid.to(torch.float32).mean().item())
            # Calculate improvement
            cost_improvement = current_cost - proposed_cost

            if config["METROPOLIS"]:
                is_accepted, actual_improvement = metropolis_accept(
                    cost_improvement, current_temp, device
                )
            else:
                is_accepted = torch.ones_like(cost_improvement)
                actual_improvement = cost_improvement
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

            if config["CAPACITY_REWARD"]:
                new_capacity_left = problem.capacity_utilization(current_solution)
                diff_capacity = config["CAPACITY_REWARD_FACTOR"] * (
                    new_capacity_left - capacity_left
                )
                actual_improvement -= diff_capacity
                capacity_left = new_capacity_left
            # Update best solution tracking
            if record_state:
                cost_history.append(current_cost)

            is_improvement = (current_cost < best_cost).long()
            is_improvement_expanded = extend_to(is_improvement, solution_components)
            best_solution = (
                is_improvement_expanded * current_solution
                + (1 - is_improvement_expanded) * best_solution
            )
            best_cost_step = torch.max(is_improvement * (step + 1), best_cost_step)
            best_cost = torch.minimum(current_cost, best_cost)
            cumulative_cost += best_cost

            # Temperature update at end of inner steps
            if inner_step == config["INNER_STEPS"] - 1:
                # Calculate advancement ratio (decreases from 1 to 0)
                adv = torch.tensor(1 - (step / config["OUTER_STEPS"]), device=device)
                # Use predefined cooling schedule
                next_temp = scheduler.step(step).to(device).repeat(best_cost.shape[0])
                temperature.append(next_temp.detach())

                # Normalize temperature for state representation
                model_next_temp = scale_to_unit(
                    next_temp, config["STOP_TEMP"], config["INIT_TEMP"]
                )

                # Build next state for the algorithm
                next_state = problem.to_state(
                    *problem.build_state_components(
                        current_solution, model_next_temp, adv
                    )
                ).to(device)

            # Reward calculation for reinforcement learning
            if config["METHOD"] == "ppo":
                if config["REWARD"] == "immediate":
                    reward_signal = actual_improvement.unsqueeze(1)
                    if config["NEG_REWARD"] != 0:
                        # Apply negative reward for invalid actions
                        reward_signal = (
                            reward_signal * is_valid
                            + (1 - is_valid) * -config["NEG_REWARD"]
                        )
                    if config["NORMALIZE_REWARD"]:
                        # Normalize rewards to [-1, 1] range
                        # Find the max absolute value for scaling
                        max_abs_reward = torch.max(torch.abs(reward_signal))
                        # Avoid division by zero
                        if max_abs_reward > 0:
                            reward_signal = reward_signal / max_abs_reward
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
                    config["GAMMA"],
                )

        # Update state and temperature for next iteration
        current_state = next_state.clone()
        current_temp = next_temp.clone()

    # Finalize results
    negative_improvement = -(initial_cost - current_cost)

    # Handle final transition in replay buffer
    if replay_buffer is not None and len(replay_buffer) > 0:
        final_transition = replay_buffer.pop()
        replay_buffer.push(*(list(final_transition[:-1]) + [0.0]))

    final_capacity_left = problem.capacity_utilization(best_solution)

    dict = {
        "best_x": best_solution,
        "min_cost": best_cost,
        "primal": cumulative_cost,
        "ngain": negative_improvement,
        "n_acc": accepted_moves.float(),
        "n_rej": rejected_moves.float(),
        "distributions": action_distributions,
        "is_valid": torch.tensor(is_valid_history),
        "states": state_history,
        "actions": action_history,
        "acceptance": acceptance_history,
        "costs": cost_history,
        "init_cost": initial_cost,
        "reward": reward_signal,
        "temperature": temperature,
        "best_step": best_cost_step.float(),
        "capacity_left": final_capacity_left,
    }
    if config["HEURISTIC"] == "mix":
        dict["ratio"] = ratio / config["OUTER_STEPS"]
        dict["heuristic"] = heuristic_choice
    return dict
