import torch

from problem import Problem
from utils import extend_to


def hill_climbing(problem: Problem, init_x: torch.Tensor, cfg: dict):
    """Runs Hill Climbing optimisation."""
    batch_size, dim, _ = init_x.shape
    best_solutions = init_x.clone()
    best_costs = problem.cost(init_x)
    init_costs = best_costs.clone()  # Initial costs
    primal = best_costs.clone()  # Primal loss

    for _ in range(cfg["OUTER_STEPS"]):
        # Generate all possible pairs of indices for swaps
        all_permutations = torch.combinations(
            torch.arange(dim, device=cfg["DEVICE"]), r=2
        )
        all_permutations = all_permutations.repeat(batch_size, 1, 1)  # Match batch size

        for i in range(all_permutations.size(1)):  # Iterate over each swap pair
            actions = all_permutations[:, i]  # Current action pair
            new_solutions = problem.update(best_solutions, actions)
            new_costs = problem.cost(new_solutions)

            # Keep the best solutions
            improved = new_costs < best_costs
            # best_costs[improved] = new_costs[improved]
            # best_solutions[improved] = new_solutions[improved]
            new_best = 1 * improved
            new_best = extend_to(new_best, best_solutions)
            best_solutions = new_best * new_solutions + (1 - new_best) * best_solutions
            best_costs = torch.minimum(new_costs, best_costs)
            primal = primal + best_costs
    # Calculate negative gain
    ngain = -(init_costs - new_costs)
    return best_solutions, best_costs, ngain, primal
