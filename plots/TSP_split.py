import torch
import matplotlib.pyplot as plt
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from problem import CVRP
from heur_init import (
    construct_cvrp_solution,
    generate_Clark_and_Wright,
    generate_sweep_solution,
    generate_nearest_neighbor,
    vrp_optimal_split,
)


def calculate_tour_distances(solutions, distance_matrices):
    """
    Calculate the total distance for each tour (vectorized version).

    Args:
        solutions: tensor of shape [n_problems, n_nodes] with node indices
        distance_matrices: tensor of shape [n_problems, n_nodes, n_nodes]

    Returns:
        distances: tensor of shape [n_problems] with total distances
    """
    n_problems = solutions.shape[0]

    # Roll the solutions to get the next node in each tour
    next_nodes = torch.roll(solutions, shifts=-1, dims=1)

    # Create a batch index to select the correct distance matrix for each problem
    # Shape: [n_problems, 1] to enable broadcasting
    batch_idx = torch.arange(n_problems, device=solutions.device).unsqueeze(1)

    # Use advanced indexing to get the distances for each leg of the tour.
    # This is equivalent to:
    # tour_leg_distances[p, i] = distance_matrices[p, solutions[p, i], next_nodes[p, i]]
    tour_leg_distances = distance_matrices[batch_idx, solutions, next_nodes]

    # Sum the distances of all legs for each tour to get the total distance
    return tour_leg_distances.sum(dim=1)


def two_opt(tours, i, j):
    """
    Applique un échange 2-opt à plusieurs tournées de manière vectorisée.
    Inverse le segment de tournée entre les indices i et j (inclus).

    Args:
        tours: Tenseur de forme [n_problems, n_nodes] représentant les tournées.
        i, j: Tenseurs d'indices pour l'échange 2-opt, de forme [n_problems].

    Returns:
        new_tours: Tenseur avec les échanges 2-opt appliqués.
    """
    # Obtenir les dimensions et le device depuis le tenseur d'entrée
    n_problems, n_nodes = tours.shape
    device = tours.device

    # Déterminer les bornes gauche (l) et droite (r) pour chaque segment à inverser
    left = torch.min(i, j)
    right = torch.max(i, j)

    # 1. Créer une matrice d'indices de base [0, 1, ..., n_nodes-1] pour chaque tournée
    # Shape : [n_problems, n_nodes]
    fwd_indices = (
        torch.arange(n_nodes, device=device).unsqueeze(0).expand(n_problems, -1)
    )

    # 2. Créer la matrice des indices inversés
    # La formule pour inverser un indice k dans un segment [left, right]
    # est left + right - k
    offset = (left + right).unsqueeze(1)
    rev_indices = offset - fwd_indices

    # 3. Créer un masque booléen pour le segment à inverser (bornes incluses)
    # Le masque est True là où left <= index <= right
    mask = (fwd_indices >= left.unsqueeze(1)) & (fwd_indices <= right.unsqueeze(1))

    # 4. Construire l'index final pour l'opération gather
    # On utilise les indices normaux là où le masque est False
    # et les indices inversés là où le masque est True.
    final_indices = torch.where(mask, rev_indices, fwd_indices)

    # 5. Utiliser torch.gather pour appliquer l'échange 2-opt de manière vectorisée
    new_tours = torch.gather(tours, 1, final_indices)

    return new_tours


def SA_TSP(
    init_solutions,
    distance_matrices,
    initial_temp=1000,
    cooling_rate=0.995,
    min_temp=1,
    max_iter=10000,
):
    """
    Vectorized Simulated Annealing with 2-opt operator for TSP.

    Args:
        init_solutions: tensor of shape [n_problems, n_nodes, 1] with initial solutions
        distance_matrices: tensor of shape [n_problems, n_nodes, n_nodes]
        initial_temp: initial temperature for SA
        cooling_rate: cooling rate (should be < 1)
        min_temp: minimum temperature to stop
        max_iter: maximum number of iterations

    Returns:
        best_solutions: tensor of shape [n_problems, n_nodes, 1]
        with best solutions found
    """
    n_problems, n_nodes, _ = init_solutions.shape
    device = init_solutions.device

    current_solutions = init_solutions.squeeze(-1).clone()  # [n_problems, n_nodes]
    best_solutions = current_solutions.clone()

    current_distances = calculate_tour_distances(current_solutions, distance_matrices)
    best_distances = current_distances.clone()

    temperature = initial_temp

    for iteration in range(max_iter):
        if temperature < min_temp:
            break

        # Generate random 2-opt moves for all problems
        i = torch.randint(0, n_nodes - 1, (n_problems,), device=device)
        j = torch.randint(1, n_nodes, (n_problems,), device=device)
        # j = torch.where(j > i, j, i + 1)  # Ensure j > i

        # Apply 2-opt swaps (vectorized)
        new_tours = two_opt(current_solutions, i, j)

        # Calculate new distances (vectorized)
        new_distances = calculate_tour_distances(new_tours, distance_matrices)

        # Calculate deltas
        deltas = new_distances - current_distances

        # Acceptance probabilities
        accept_probs = torch.exp(-deltas / temperature)
        accept = (deltas < 0) | (torch.rand(n_problems, device=device) < accept_probs)

        # Update current solutions
        current_solutions[accept] = new_tours[accept]
        current_distances[accept] = new_distances[accept]

        # Update best solutions
        improved = new_distances < best_distances
        best_solutions[improved] = new_tours[improved]
        best_distances[improved] = new_distances[improved]

        # Cool down
        temperature *= cooling_rate

        if iteration % 1000 == 0:
            print(
                f"Iteration {iteration}, Temperature: {temperature:.4f}, "
                f"Best avg distance: {best_distances.mean():.2f}"
            )

    return best_solutions.unsqueeze(-1)


def load_problem_data(dim):
    """Load problem data from file."""
    BDD_PATH = f"generated_problem/gen{dim}.pt"
    bdd = torch.load(BDD_PATH, map_location="cpu")
    return bdd["node_coords"], bdd["demands"], bdd["capacity"]


def setup_config(coords, capacities):
    """Setup configuration parameters."""
    return {
        "PROBLEM_DIM": coords.shape[1] - 1,
        "N_PROBLEMS": capacities.shape[0],
        "OUTER_STEPS": 10000,
        "MAX_LOAD": 50,
        "DEVICE": "cpu",
        "SEED": 0,
        "LOAD_PB": True,
        "CLUSTERING": False,
    }


def initialize_problem(cfg, coords, demands, capacities):
    """Initialize CVRP problem and generate initial solutions."""
    problem = CVRP(
        cfg["PROBLEM_DIM"],
        cfg["N_PROBLEMS"],
        capacities if capacities is not None else cfg["MAX_LOAD"],
        device=cfg["DEVICE"],
        params=cfg,
    )

    problem.manual_seed(cfg["SEED"])
    params = problem.generate_params("test", cfg["LOAD_PB"], coords, demands)
    problem.set_params(params)
    init_x = problem.generate_init_x("random")

    # Filter out zero entries
    mask = init_x.squeeze(-1) != 0
    init_x = init_x[mask].view(cfg["N_PROBLEMS"], -1, init_x.size(-1))

    return problem, init_x


def run_simulated_annealing(init_x, problem, sa_params):
    """Run simulated annealing optimization."""
    print("Starting Simulated Annealing...")

    # Calculate initial distances
    initial_distances = calculate_tour_distances(init_x.squeeze(-1), problem.matrix)
    print(f"Initial average distance: {initial_distances.mean():.2f}")

    # Run SA
    best_solutions = SA_TSP(init_x, problem.matrix, **sa_params)

    # Calculate final distances
    final_distances = calculate_tour_distances(
        best_solutions.squeeze(-1), problem.matrix
    )

    # Print results
    print(f"Final average distance: {final_distances.mean():.2f}")
    improvement = (
        (initial_distances.mean() - final_distances.mean())
        / initial_distances.mean()
        * 100
    )
    print(f"Improvement: {improvement:.2f}%")

    return best_solutions


if __name__ == "__main__":
    # Problem parameters
    dim = 100

    # SA parameters
    sa_params = {
        "initial_temp": 10_000,
        "cooling_rate": 0.999,
        "min_temp": 0.00001,
        "max_iter": 1_000_000,
    }

    # Load problem data
    coords, demands, capacities = load_problem_data(dim)

    # Setup configuration
    cfg = setup_config(coords, capacities)

    # Initialize problem
    problem, init_x = initialize_problem(cfg, coords, demands, capacities)

    # Run simulated annealing
    start_time = time.time()

    best_solutions = run_simulated_annealing(init_x, problem, sa_params)
    time_TSP = time.time() - start_time

    start_time = time.time()
    sol_split = construct_cvrp_solution(
        best_solutions, problem.demands, problem.capacity
    )
    time_split = time.time() - start_time + time_TSP

    start_time = time.time()
    sol_opt_split = vrp_optimal_split(
        problem.coords, problem.demands, problem.capacity, best_solutions.squeeze(-1)
    ).to(problem.device)
    time_opt_split = time.time() - start_time + time_TSP

    start_time = time.time()
    sol_CW = generate_Clark_and_Wright(problem).to(problem.device)
    time_CW = time.time() - start_time
    start_time = time.time()
    sol_sweep = generate_sweep_solution(problem).to(problem.device)
    time_sweep = time.time() - start_time
    start_time = time.time()
    sol_nn = generate_nearest_neighbor(problem).to(problem.device)
    time_nn = time.time() - start_time

    cost_split = problem.cost(sol_split)
    cost_opt_split = problem.cost(sol_opt_split)
    cost_CW = problem.cost(sol_CW)
    cost_sweep = problem.cost(sol_sweep)
    cost_nn = problem.cost(sol_nn)

    print(f"Cost of split solution: {cost_split.mean().item()}")
    print(f"Cost of optimal split solution: {cost_opt_split.mean().item()}")
    print(f"Cost of initial solution: {cost_CW.mean().item()}")
    print(f"Cost of sweep solution: {cost_sweep.mean().item()}")
    print(f"Cost of nearest neighbor solution: {cost_nn.mean().item()}")

    # Prepare data for the bar plot
    methods = [
        "Split TSP",
        "Optimal Split TSP",
        "Clark & Wright",
        "Sweep",
        "Nearest Neighbor",
    ]
    costs = [
        cost_split.mean().item(),
        cost_opt_split.mean().item(),
        cost_CW.mean().item(),
        cost_sweep.mean().item(),
        cost_nn.mean().item(),
    ]

    # Create the bar plot
    plt.figure(figsize=(8, 6))
    plt.bar(
        methods, costs, color=["blue", "purple", "orange", "green", "red"], alpha=0.7
    )

    # Add labels and title
    plt.xlabel("Methods", fontsize=14)
    plt.ylabel("Average Cost", fontsize=14)
    plt.title("Comparison of CVRP Solution Costs", fontsize=16)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Add value annotations on top of the bars
    for i, t in enumerate([time_split, time_opt_split, time_CW, time_sweep, time_nn]):
        plt.text(i, costs[i] + 0.25, f"{t:.2f}s", ha="center", fontsize=11)

    # Show the plot
    plt.tight_layout()
    plt.savefig("plots/TSP_split_solution_comparison.png")
    plt.show()
