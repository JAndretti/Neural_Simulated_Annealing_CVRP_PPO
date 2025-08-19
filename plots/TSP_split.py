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
from TSP import get_features, TSP_actor


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


def two_opt(x: torch.Tensor, a: torch.Tensor):
    """Swap cities a[0] <-> a[1].

    Args:
        s: perm vector [batch size, coords]
        a: cities to swap ([batch size], [batch size])
    """
    n_problems, n_nodes = x.shape
    # Two-opt moves invert a section of a tour. If we cut a tour into
    # segments a and b then we can choose to invert either a or b. Due
    # to the linear representation of a tour, we choose always to invert
    # the segment that is stored contiguously.
    left = torch.minimum(a[:, 0], a[:, 1])
    right = torch.maximum(a[:, 0], a[:, 1])
    ones = torch.ones((n_problems, 1), dtype=torch.long, device=x.device)
    fidx = torch.arange(n_nodes, device=x.device) * ones
    # Reversed indices
    offset = left + right - 1
    ridx = torch.arange(0, -n_nodes, -1, device=x.device) + offset[:, None]
    # Set flipped section to all True
    flip = torch.ge(fidx, left[:, None]) * torch.lt(fidx, right[:, None])
    # Set indices to replace flipped section with
    idx = (~flip) * fidx + flip * ridx
    # Perform 2-opt move
    return torch.gather(x, 1, idx)


def SA_TSP(
    init_solutions,
    problem,
    actor,
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
    distance_matrices = problem.matrix
    coords = problem.coords
    current_solutions = init_solutions.squeeze(-1).clone()  # [n_problems, n_nodes]
    best_solutions = current_solutions.clone()

    current_distances = calculate_tour_distances(current_solutions, distance_matrices)
    best_distances = current_distances.clone()

    temperature = initial_temp

    for iteration in range(max_iter):
        if temperature < min_temp:
            break

        if actor is not None:
            with torch.no_grad():
                features = get_features(current_solutions, coords, temperature)
                action, _ = actor.sample(features)
            # i, j = action[:, 0], action[:, 1]
        else:
            # Generate random 2-opt moves for all problems
            i = torch.randint(0, n_nodes - 1, (n_problems,), device=device)  # 0,98
            j = torch.randint(1, n_nodes, (n_problems,), device=device)  # 1,99
            action = torch.stack((i, j), dim=1)
            # j = torch.where(j > i, j, i + 1)  # Ensure j > i

        # Apply 2-opt swaps (vectorized)
        new_tours = two_opt(current_solutions, action)

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

        if iteration % 100 == 0:
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
        "DEVICE": (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        ),
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


def run_simulated_annealing(init_x, problem, sa_params, actor=None):
    """Run simulated annealing optimization."""
    print("Starting Simulated Annealing...")

    # Calculate initial distances
    initial_distances = calculate_tour_distances(init_x.squeeze(-1), problem.matrix)
    print(f"Initial average distance: {initial_distances.mean():.2f}")

    # Run SA
    best_solutions = SA_TSP(init_x, problem, actor, **sa_params)

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

    sa_params_model = {
        "initial_temp": 1,
        "cooling_rate": 0.995,
        "min_temp": 0.01,
        "max_iter": 1_000_000,
    }

    # Load problem data
    coords, demands, capacities = load_problem_data(dim)

    # Setup configuration
    cfg = setup_config(coords, capacities)
    TSP_actor.to(cfg["DEVICE"])
    # Initialize problem
    problem, init_x = initialize_problem(cfg, coords, demands, capacities)

    # Run simulated annealing
    start_time = time.time()

    best_solutions = run_simulated_annealing(init_x, problem, sa_params)
    time_TSP = time.time() - start_time
    print(f"SA time : {time_TSP:.2f} seconds")

    start_time = time.time()
    sol_split = construct_cvrp_solution(
        best_solutions, problem.demands, problem.capacity
    )
    time_split = time.time() - start_time + time_TSP

    start_time = time.time()
    sol_opt_split = vrp_optimal_split(
        problem.coords.to("cpu"),
        problem.demands.to("cpu"),
        problem.capacity.to("cpu"),
        best_solutions.squeeze(-1).to("cpu"),
    ).to(cfg["DEVICE"])
    time_opt_split = time.time() - start_time + time_TSP

    # Run simulated annealing
    start_time = time.time()

    best_solutions_actor = run_simulated_annealing(
        init_x, problem, sa_params_model, actor=TSP_actor
    )
    time_TSP_actor = time.time() - start_time

    print(f"NSA time : {time_TSP_actor:.2f} seconds")

    start_time = time.time()
    sol_split_model = construct_cvrp_solution(
        best_solutions_actor, problem.demands, problem.capacity
    )
    time_split_model = time.time() - start_time + time_TSP_actor

    start_time = time.time()
    sol_opt_split_model = vrp_optimal_split(
        problem.coords.to("cpu"),
        problem.demands.to("cpu"),
        problem.capacity.to("cpu"),
        best_solutions_actor.squeeze(-1).to("cpu"),
    ).to(cfg["DEVICE"])
    time_opt_split_model = time.time() - start_time + time_TSP_actor

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
    cost_split_model = problem.cost(sol_split_model)
    cost_opt_split_model = problem.cost(sol_opt_split_model)
    cost_CW = problem.cost(sol_CW)
    cost_sweep = problem.cost(sol_sweep)
    cost_nn = problem.cost(sol_nn)

    print(f"Cost of split solution: {cost_split.mean().item()}")
    print(f"Cost of optimal split solution: {cost_opt_split.mean().item()}")
    print(f"Cost of split model solution: {cost_split_model.mean().item()}")
    print(f"Cost of optimal split model solution: {cost_opt_split_model.mean().item()}")
    print(f"Cost of initial solution: {cost_CW.mean().item()}")
    print(f"Cost of sweep solution: {cost_sweep.mean().item()}")
    print(f"Cost of nearest neighbor solution: {cost_nn.mean().item()}")

    # Prepare data for the bar plot
    methods = [
        "Split TSP",
        "Optimal Split TSP",
        "Split Model TSP",
        "Optimal Split Model TSP",
        "Clark & Wright",
        "Sweep",
        "Nearest Neighbor",
    ]
    costs = [
        cost_split.mean().item(),
        cost_opt_split.mean().item(),
        cost_split_model.mean().item(),
        cost_opt_split_model.mean().item(),
        cost_CW.mean().item(),
        cost_sweep.mean().item(),
        cost_nn.mean().item(),
    ]

    # Create the bar plot
    plt.figure(figsize=(8, 6))
    plt.bar(
        methods,
        costs,
        color=["blue", "purple", "yellow", "brown", "orange", "green", "red"],
        alpha=0.7,
    )

    # Add labels and title
    plt.xlabel("Methods", fontsize=14)
    plt.ylabel("Average Cost", fontsize=14)
    plt.title("Comparison of CVRP Solution Costs", fontsize=16)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Add value annotations on top of the bars
    for i, t in enumerate(costs):
        plt.text(i, t + 0.25, f"{t:.2f}", ha="center", fontsize=11)

    # Add value annotations on bottom of the bars
    for i, t in enumerate(
        [
            time_split,
            time_opt_split,
            time_split_model,
            time_opt_split_model,
            time_CW,
            time_sweep,
            time_nn,
        ]
    ):
        plt.text(
            i, costs[i] - 1.25, f"{t:.2f}s", ha="center", color="grey", fontsize=11
        )

    # Rotate x-axis labels by 45 degrees
    plt.xticks(rotation=45)
    # Show the plot
    plt.tight_layout()
    plt.savefig("plots/TSP_split_solution_comparison.png")
    plt.show()
