from typing import Dict
import torch
from scheduler import Scheduler
from problem import Problem
from utils import extend_to
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


def sa_baseline(
    problem: Problem,
    initial_solution: torch.Tensor,
    config: dict,
    desc_tqdm: str = "Simulated Annealing Progress",
) -> Dict[str, torch.Tensor]:
    """Simulated Annealing baseline for solving the CVRP problem."""
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

    # Action
    if config["UPDATE_METHOD"] == "free":
        shape_action = best_solution.shape[1]
    elif config["UPDATE_METHOD"] == "rm_depot":
        shape_action = problem.dim

    # Init temperature
    # Use default initial temperature (will be scaled below)
    current_temp = torch.tensor([1], device=device).repeat(best_cost.shape[0])

    for step in tqdm(
        range(config["OUTER_STEPS"]),
        desc=desc_tqdm,
        colour="green",
        unit="step",
        leave=False,
    ):
        # Inner loop at fixed temperature
        for inner_step in range(config["INNER_STEPS"]):
            action = torch.randint(
                0, shape_action, (best_solution.shape[0], 2), device=device
            )
            proposed_solution, is_valid = problem.update(current_solution, action)
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
            is_accepted_expanded = extend_to(is_accepted, proposed_solution)
            current_solution = (
                is_accepted_expanded * proposed_solution
                + (1 - is_accepted_expanded) * current_solution
            ).long()
            is_improvement = (current_cost < best_cost).long()
            is_improvement_expanded = extend_to(is_improvement, current_solution)
            best_solution = (
                is_improvement_expanded * current_solution
                + (1 - is_improvement_expanded) * best_solution
            )
            best_cost = torch.minimum(current_cost, best_cost)

            # Temperature update at end of inner steps
            if inner_step == config["INNER_STEPS"] - 1:
                # Use predefined cooling schedule
                current_temp = (
                    scheduler.step(step).to(device).repeat(best_cost.shape[0])
                )
    dict_result = {"min_cost": best_cost, "best_x": best_solution}
    return dict_result
