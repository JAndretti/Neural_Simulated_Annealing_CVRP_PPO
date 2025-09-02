import os
import sys
from glob import glob
import matplotlib.pyplot as plt
import time
import torch
from rich import print


from func import (
    init_problem_parameters,
    set_seed,
    load_model,
    plot_cvrp_solution,
    init_pb,
)

# Add src path to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from problem import CVRP
from sa import sa_train
from model import CVRPActorPairs, CVRPActor
from algo import test_or_tools

# --- Configurations ---

MODEL_NAME = "20250827_172131_57xbb0z1"
MODEL_DIR = glob(
    os.path.join("wandb", "Neural_Simulated_Annealing", "*", "models", MODEL_NAME)
)[0]
SEED = 0

cfg = {
    "PROBLEM_DIM": 100,
    "MAX_LOAD": 50,
    "N_PROBLEMS": 1,
    "OUTER_STEPS": 10000,
    "DEVICE": "cpu",
    "INIT": "isolate",
    "SEED": 0,
    "LOAD_PB": False,
    "STOP_TEMP": 0.01,
}


def cvrp_2opt_vectorized(
    solution, coordinates, max_iterations=10000, improvement_threshold=0.0001
):
    """
    Apply 2-opt optimization within each route of a CVRP solution.

    Args:
        solution: Tensor of shape [batch_size, route_length, 1] representing the
                 solution
        coordinates: Tensor of shape [batch_size, num_nodes, 2] with node coordinates
        max_iterations: Maximum number of improvement iterations
        improvement_threshold: Minimum improvement to continue optimization

    Returns:
        Tensor of shape [batch_size, route_length, 1] with the optimized solution
    """
    solution = solution.clone().squeeze(-1)
    coords = coordinates.squeeze(0)
    device = solution.device

    # Extract individual routes
    routes = []
    current_route = []

    for node in solution.squeeze(0):
        node_idx = node.item()
        if node_idx == 0:  # Depot
            if current_route:  # End of a route
                routes.append(torch.tensor([0] + current_route + [0], device=device))
                current_route = []
        else:
            current_route.append(node_idx)

    # Add last route if it exists
    if current_route:
        routes.append(torch.tensor([0] + current_route + [0], device=device))

    # Optimize each route independently
    optimized_routes = []

    for route in routes:
        # Skip routes with 3 or fewer nodes (depot-client-depot)
        if len(route) <= 3:
            optimized_routes.append(route)
            continue

        # Apply 2-opt on this route
        best_route = route.clone()
        best_distance = calculate_route_distance(best_route, coords)
        improved = True
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            # Try all possible 2-opt swaps
            for i in range(1, len(route) - 2):  # Skip depot
                for j in range(i + 1, len(route) - 1):  # Skip depot at end
                    # Create new route with 2-opt swap (reverse segment)
                    new_route = torch.cat(
                        [
                            best_route[:i],
                            best_route[i : j + 1].flip(0),
                            best_route[j + 1 :],
                        ]
                    )

                    # Calculate new distance
                    new_distance = calculate_route_distance(new_route, coords)

                    # If better, keep it
                    if new_distance < best_distance - improvement_threshold:
                        best_distance = new_distance
                        best_route = new_route
                        improved = True

        optimized_routes.append(best_route)

    # Rebuild complete solution
    final_solution = torch.cat(optimized_routes)

    # Pad to original length if needed
    if len(final_solution) < solution.size(1):
        padding = torch.zeros(
            solution.size(1) - len(final_solution), dtype=torch.long, device=device
        )
        final_solution = torch.cat([final_solution, padding])

    return final_solution.unsqueeze(0).unsqueeze(-1)


def calculate_route_distance(route, coords):
    """Calculate the total distance of a single route"""
    total = 0
    for i in range(len(route) - 1):
        total += torch.norm(coords[route[i]] - coords[route[i + 1]])
    return total.item()


def main():
    set_seed(SEED)  # For reproducibility
    CFG = init_problem_parameters(MODEL_DIR, cfg)
    # Print the configuration dictionary
    print("Configuration parameters:")
    for key, value in CFG.items():
        print(f"  {key}: {value}")
    print()  # Add an empty line for better readability
    # Initialize problem
    problem = CVRP(
        CFG["PROBLEM_DIM"],
        CFG["N_PROBLEMS"],
        CFG["MAX_LOAD"],
        device=CFG["DEVICE"],
        params=CFG,
    )
    problem.manual_seed(SEED)
    problem, init_x, initial_cost = init_pb(CFG)

    # Initialize actor
    if CFG["PAIRS"]:
        actor = CVRPActorPairs(
            CFG["EMBEDDING_DIM"],
            num_hidden_layers=CFG["NUM_H_LAYERS"],
            device=CFG["DEVICE"],
            mixed_heuristic=True if CFG["HEURISTIC"] == "mix" else False,
        )
    else:
        actor = CVRPActor(
            CFG["EMBEDDING_DIM"],
            CFG["ENTRY"],
            num_hidden_layers=CFG["NUM_H_LAYERS"],
            device=CFG["DEVICE"],
            mixed_heuristic=True if CFG["HEURISTIC"] == "mix" else False,
            method=CFG["UPDATE_METHOD"],
        )
    actor = load_model(actor, MODEL_DIR, "actor")

    # apply 2-opt intra-route optimization
    # init_x = problem.generate_init_x(CFG["INIT"])
    coordinates = problem.state_encoding
    init_x_2opt = cvrp_2opt_vectorized(init_x, coordinates, max_iterations=10000)

    print(("SA started"))
    # Measure execution time
    start_time = time.time()
    init_x = problem.generate_init_solution(CFG["INIT"])
    # Run experiment with record_state=True
    result = sa_train(actor, problem, init_x, CFG, record_state=True, baseline=False)

    # apply 2-opt intra-route optimization
    coordinates = problem.state_encoding
    result_2opt = cvrp_2opt_vectorized(
        result["best_x"], coordinates, max_iterations=10000
    )
    costs = [c.item() for c in result["costs"]]
    temperatures = [t.item() for t in result["temperature"]]
    acceptance = [
        a.float().mean().item() if hasattr(a, "float") else float(a)
        for a in result["acceptance"]
    ]
    invalid_moves = CFG["OUTER_STEPS"] - result["is_valid"].sum().item()

    del result

    # Calculate and print execution time
    execution_time = time.time() - start_time
    print(f"Execution time model: {execution_time:.2f} seconds")
    min_cost_2opt = problem.cost(result_2opt).item()
    print(f"Solution found is feasible: {problem.is_feasible(result_2opt).item()}")

    start_time = time.time()
    init_x = problem.generate_init_solution(CFG["INIT"])
    result_baseline = sa_train(
        actor, problem, init_x, CFG, baseline=True, record_state=True
    )

    # apply 2-opt intra-route optimization
    coordinates = problem.state_encoding
    result_baseline_2_opt = cvrp_2opt_vectorized(
        result_baseline["best_x"], coordinates, max_iterations=10000
    )

    # Calculate and print execution time
    execution_time_baseline = time.time() - start_time
    print(f"Execution time baseline: {execution_time_baseline:.2f} seconds")
    min_cost_2opt_baseline = problem.cost(result_baseline_2_opt).item()
    print(
        f"Solution found is feasible: "
        f"{problem.is_feasible(result_baseline_2_opt).item()}"
    )
    # Configure OR-Tools parameters
    or_tools_cfg = {
        "OR_TOOLS_TIME": int(
            execution_time + 0.999
        ),  # Time limit in seconds rounded up
    }
    params = {
        "coords": problem.coords.cpu(),
        "demands": problem.demands.cpu(),
        "MAX_LOAD": [CFG["MAX_LOAD"]],
        "OR_DIM": [CFG["PROBLEM_DIM"]],
        "names": ["Test_Instance"],
    }
    print(
        f"Testing with OR-Tools with time limit of {or_tools_cfg['OR_TOOLS_TIME']} ",
        "seconds",
    )
    or_tools_solution = test_or_tools(params, or_tools_cfg)
    if or_tools_solution is not None:
        or_tools_cost = problem.cost(or_tools_solution)
        or_tools_sol = or_tools_solution[0, :, 0].cpu().numpy()
        print(f"OR-Tools cost: {or_tools_cost.item():.4f}")
    else:
        print("OR-Tools could not find a solution.")
        or_tools_cost = torch.tensor([float("inf")])
        or_tools_sol = None
    # Retrieve cost, temperature, and acceptance evolution
    costs_baseline = [c.item() for c in result_baseline["costs"]]

    # Compute cumulative acceptance (running mean)
    acceptance_cum = []
    count_ones = 0
    for i, a in enumerate(acceptance):
        if a == 1:
            count_ones += 1
        acceptance_cum.append(count_ones / (i + 1))

    # Find the first index where the minimum cost is reached
    min_cost = min(costs)
    min_idx = costs.index(min_cost)
    print(f"Initial cost: {problem.cost(init_x).item():.4f}")
    print(f"Initial cost after 2-opt: {problem.cost(init_x_2opt).item():.4f}")
    print(f"Minimum cost: {min_cost:.4f} at step {min_idx}")
    print(f"Minimum cost after 2-opt: {min_cost_2opt:.4f}")
    print(
        f"Number of invalid moves (refused): "
        f"{invalid_moves} / {CFG['OUTER_STEPS']}"
        f" ({invalid_moves * 100 / CFG['OUTER_STEPS']:.2f} %)"
    )
    min_cost_baseline = min(costs_baseline)
    min_idx_baseline = costs_baseline.index(min_cost_baseline)
    print(f"Minimal cost baseline: {min_cost_baseline:.4f} at step {min_idx_baseline}")
    print(f"Minimal cost baseline after 2-opt: {min_cost_2opt_baseline:.4f}")
    invalid_moves_baseline = (
        CFG["OUTER_STEPS"] - result_baseline["is_valid"].sum().item()
    )
    print(
        f"Number of invalid moves (refused) for baseline: "
        f"{invalid_moves_baseline} / {CFG['OUTER_STEPS']}"
        f" ({invalid_moves_baseline * 100 / CFG['OUTER_STEPS']:.2f} %)"
    )
    # Plot cost, temperature, cumulative acceptance, and ratio if mix
    fig, ax1 = plt.subplots(figsize=(30, 6))
    ax1.plot(costs, color="tab:blue", label="Cost")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Cost", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.axvline(
        min_idx, color="tab:purple", linestyle="--", label="min cost", alpha=0.7
    )

    ax2 = ax1.twinx()
    ax2.plot(temperatures, color="tab:red", label="Temperature", alpha=0.7)
    ax2.set_ylabel("Temperature", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # Third axis for cumulative acceptance
    ax3 = ax1.twinx()
    ax3.spines.right.set_position(("axes", 1.22))
    ax3.plot(
        acceptance_cum, color="tab:orange", label="Cumulative Acceptance", alpha=0.8
    )
    ax3.set_ylabel("Cumulative Acceptance", color="tab:orange")
    ax3.tick_params(axis="y", labelcolor="tab:orange")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    all_lines = lines1 + lines2 + lines3
    all_labels = labels1 + labels2 + labels3

    ax1.legend(
        all_lines,
        all_labels,
        loc="upper right",
    )

    plt.title("Evolution of Cost, Temperature, Acceptance during SA")
    fig.tight_layout()
    plt.grid()
    plt.show()

    print("Printing best solution...")
    # Print solution route by route
    solution = result_2opt.squeeze(0).squeeze(-1)
    demands = problem.demands.squeeze(0)
    capacity = CFG["MAX_LOAD"]
    coords = problem.state_encoding.squeeze(0)
    routes = []
    current_route = []
    ratio_mean = []

    print("Best solution routes:")
    for idx, node in enumerate(solution):
        node_idx = node.item()
        if node_idx == 0:  # Depot
            if current_route:  # End of a route
                routes.append([0] + current_route + [0])
                # Calculate total demand for this route
                route_demand = sum(demands[node_id].item() for node_id in current_route)
                demand_ratio = route_demand / capacity
                ratio_mean.append(demand_ratio)
                # Calculate route distance
                route_distance = 0
                route_nodes = [0] + current_route + [0]  # Add depot at start and end
                for i in range(len(route_nodes) - 1):
                    route_distance += torch.norm(
                        coords[route_nodes[i]] - coords[route_nodes[i + 1]]
                    ).item()
                print(
                    f"Route {len(routes)}: {routes[-1]} - "
                    f"Demand: {route_demand}/{capacity} ({demand_ratio:.2f}) - "
                    f"Distance: {route_distance:.2f}"
                )
                current_route = []
            elif idx > 0:  # Skip consecutive depots, but include the first depot
                pass
        elif node_idx > 0:  # Skip padding zeros
            current_route.append(node_idx)

    # Add last route if it exists
    if current_route:
        routes.append([0] + current_route + [0])
        route_demand = sum(demands[node_id].item() for node_id in current_route)
        demand_ratio = route_demand / capacity
        # Calculate route distance
        route_distance = 0
        route_nodes = [0] + current_route + [0]  # Add depot at start and end
        for i in range(len(route_nodes) - 1):
            route_distance += torch.norm(
                coords[route_nodes[i]] - coords[route_nodes[i + 1]]
            ).item()
        print(
            f"Route {len(routes)}: {routes[-1]} - "
            f"Demand: {route_demand}/{capacity} ({demand_ratio:.2f}) - "
            f"Distance: {route_distance:.2f}"
        )
    print(f"Mean demand ratio: {sum(ratio_mean) / len(ratio_mean):.2f}")
    print(f"Total routes: {len(routes)}")

    if or_tools_solution is not None:
        print("Printing OR solution...")
        # Print solution route by route
        solution = or_tools_solution.squeeze(0).squeeze(-1)
        demands = problem.demands.squeeze(0)
        capacity = cfg["MAX_LOAD"]
        coords = problem.coords.squeeze(0)
        routes = []
        current_route = []
        ratio_mean = []

        print("Best solution routes:")
        for idx, node in enumerate(solution):
            node_idx = node.item()
            if node_idx == 0:  # Depot
                if current_route:  # End of a route
                    routes.append([0] + current_route + [0])
                    # Calculate total demand for this route
                    route_demand = sum(
                        demands[node_id].item() for node_id in current_route
                    )
                    demand_ratio = route_demand / capacity
                    # Calculate route distance
                    route_distance = 0
                    route_nodes = (
                        [0] + current_route + [0]
                    )  # Add depot at start and end
                    for i in range(len(route_nodes) - 1):
                        route_distance += torch.norm(
                            coords[route_nodes[i]] - coords[route_nodes[i + 1]]
                        ).item()
                    print(
                        f"Route {len(routes)}: {routes[-1]} - "
                        f"Demand: {route_demand}/{capacity} ({demand_ratio:.2f}) - "
                        f"Distance: {route_distance:.2f}"
                    )
                    ratio_mean.append(demand_ratio)
                    current_route = []
                elif idx > 0:  # Skip consecutive depots, but include the first depot
                    pass
            elif node_idx > 0:  # Skip padding zeros
                current_route.append(node_idx)

        # Add last route if it exists
        if current_route:
            routes.append([0] + current_route + [0])
            route_demand = sum(demands[node_id].item() for node_id in current_route)
            demand_ratio = route_demand / capacity
            # Calculate route distance
            route_distance = 0
            route_nodes = [0] + current_route + [0]  # Add depot at start and end
            for i in range(len(route_nodes) - 1):
                route_distance += torch.norm(
                    coords[route_nodes[i]] - coords[route_nodes[i + 1]]
                ).item()
            print(
                f"Route {len(routes)}: {routes[-1]} - "
                f"Demand: {route_demand}/{capacity} ({demand_ratio:.2f}) - "
                f"Distance: {route_distance:.2f}"
            )
        print(f"Mean demand ratio: {sum(ratio_mean) / len(ratio_mean):.2f}")
        print(f"Total routes: {len(routes)}")
    else:
        print("No OR-Tools solution found.")
    # Plot the best solution (routes)
    plt0 = plot_cvrp_solution(
        problem.state_encoding,
        None,
        title="CVRP Problem (No Solution)",
        show_routes=False,  # Ensure no routes are displayed, only points
    )

    plt1 = plot_cvrp_solution(
        problem.state_encoding,
        result_2opt,
        title=f"Best CVRP Solution, cost: {min_cost_2opt:.4f}",
    )
    plt2 = plot_cvrp_solution(
        problem.state_encoding,
        init_x,
        title=f"Initial CVRP Solution, cost: {problem.cost(init_x).item():.4f}",
    )  # Plot initial solution
    if or_tools_sol is not None:
        plt3 = plot_cvrp_solution(
            problem.coords,
            or_tools_solution,
            title=f"CVRP OR-Tools Solution, cost: {or_tools_cost.item():.4f}",
        )
    plt4 = plot_cvrp_solution(
        problem.state_encoding,
        result_baseline_2_opt,
        title=f"Baseline CVRP Solution, cost: {min_cost_2opt_baseline:.4f}",
    )  # Plot baseline solution
    plt.figure(plt0.number)
    plt.show()

    # Display both solution plots
    plt.figure(plt1.number)
    plt.show()

    plt.figure(plt2.number)
    plt.show()
    if or_tools_sol is not None:
        # Display OR-Tools solution plot
        plt.figure(plt3.number)
        plt.show()
    plt.figure(plt4.number)
    plt.show()
    plt.close("all")  # Close all plots to free memory


if __name__ == "__main__":
    main()
