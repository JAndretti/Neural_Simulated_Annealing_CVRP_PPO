import os
import sys
import torch
import matplotlib.pyplot as plt
import time


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from problem import CVRP


def init_pb(
    cfg: dict,
):
    """Initialize the CVRP problem and generate initial solutions."""
    problem = CVRP(
        cfg["PROBLEM_DIM"],
        cfg["N_PROBLEMS"],
        cfg["CAPACITIES"],
        device=cfg["DEVICE"],
        params=cfg,
    )
    problem.manual_seed(cfg["SEED"])
    params = problem.generate_params("test")
    problem.set_params(params)

    return problem


if __name__ == "__main__":

    cfg = {
        "PROBLEM_DIM": 100,
        "N_PROBLEMS": 10000,
        "CAPACITIES": 50,
        "DEVICE": "cpu",
        "SEED": 0,
        "HEURISTIC": "insertion",
        "MIX1": "",
        "MIX2": "",
        "LOAD_PB": False,
        "CLUSTERING": False,
        "NB_CLUSTERS_MAX": 0,
    }
    methods = [
        "sweep",
        "random",
        "isolate",
        "nearest_neighbor",
        "Clark_and_Wright",
        "cheapest_insertion",
        "path_cheapest_arc",
    ]
    res = []
    times = []
    problem = init_pb(cfg)
    for init in methods:
        print(f"Using init method: {init}")

        # Measure execution time
        start_time = time.time()
        init_x = problem.generate_init_x(init)
        end_time = time.time()

        # Record results
        res.append(torch.mean(problem.cost(init_x)).item())
        times.append(end_time - start_time)

    # Plot the results
    colors = [
        "lightsalmon",
        "skyblue",
        "lightgreen",
        "salmon",
        "lightcoral",
        "pink",
        "lightyellow",
        "lightgray",
    ]
    plt.figure(figsize=(10, 6))
    plt.bar(methods, res, color=colors, edgecolor="black", alpha=0.7)
    for i, method in enumerate(methods):
        plt.text(
            i, res[i] + 0.5, f"{res[i]:.2f}", ha="center", fontsize=12, color="black"
        )
        plt.text(
            i, res[i] - 4, f"{times[i]:.2f}s", ha="center", fontsize=10, color="gray"
        )
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12, rotation=45, ha="right")
    plt.title("Comparison of Initialization Methods", fontsize=16, fontweight="bold")
    plt.xlabel("Initialization Method", fontsize=14)
    plt.ylabel("Average Cost", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("plots/init_methods_comparison.png", dpi=300)
    plt.show()
