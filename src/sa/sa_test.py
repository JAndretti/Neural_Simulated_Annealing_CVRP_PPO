from typing import Dict
import torch
from model import SAModel
from problem import Problem
from utils import extend_to
from .scheduler import Scheduler
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


# ================================
# ACTION GENERATION AND PROCESSING
# ================================


def generate_action(
    actor: SAModel,
    current_state: torch.Tensor,
    baseline: bool,
    greedy: bool,
    random_std: float,
    problem: Problem,
    device: torch.device,
):
    """Generate action from policy or baseline."""
    with torch.no_grad():
        with torch.amp.autocast(device.type):
            if baseline:
                res = actor.baseline_sample(
                    current_state, random_std=random_std, problem=problem
                )
            else:
                res = actor.sample(current_state, greedy=greedy, problem=problem)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


def sa_test(
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
    current_cost = best_cost.clone()

    # Use default initial temperature (will be scaled below)
    current_temp = torch.tensor([1], device=device).repeat(best_cost.shape[0])

    # Scale temperature to the configured range [STOP_TEMP, INIT_TEMP]
    current_temp = scale_between(current_temp, config["STOP_TEMP"], config["INIT_TEMP"])
    next_temp = current_temp.clone()  # Initialize next_temp for first iteration

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
    # Main optimization loop over temperature steps
    for step in tqdm(
        range(config["OUTER_STEPS"]),
        desc=desc_tqdm,
        colour="green",
        unit="step",
        leave=False,
    ):
        # Inner loop at fixed temperature
        for inner_step in range(config["INNER_STEPS"]):
            # Generate action from policy
            action, action_log_prob = generate_action(
                actor, current_state, baseline, greedy, random_std, problem, device
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            # Generate proposed solution
            solution_components, *_ = problem.from_state(current_state)
            proposed_solution, is_valid = problem.update(solution_components, action)
            proposed_cost = problem.cost(proposed_solution)

            # Calculate improvement
            cost_improvement = current_cost - proposed_cost
            is_accepted, actual_improvement = metropolis_accept(
                cost_improvement, current_temp, device
            )
            # Update current solution
            current_cost = (
                is_accepted * proposed_cost + (1 - is_accepted) * current_cost
            )
            is_accepted_expanded = extend_to(is_accepted, solution_components)
            current_solution = (
                is_accepted_expanded * proposed_solution
                + (1 - is_accepted_expanded) * solution_components
            ).long()

            best_cost = torch.minimum(current_cost, best_cost)

            # Temperature update at end of inner steps
            if inner_step == config["INNER_STEPS"] - 1:
                # Calculate advancement ratio (decreases from 1 to 0)
                adv = torch.tensor(1 - (step / config["OUTER_STEPS"]), device=device)
                # Use predefined cooling schedule
                next_temp = scheduler.step(step).to(device).repeat(best_cost.shape[0])

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

        # Update state and temperature for next iteration
        current_state = next_state.clone()
        current_temp = next_temp.clone()

    dict_solution = {
        "min_cost": best_cost,
    }
    return dict_solution
