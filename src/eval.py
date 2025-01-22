import torch
import os
import random
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor


from rich import print
from rich.console import Console
from rich.table import Table

from model import TSPActor
from problem import TSP
from sa import sa
from hill_climbing import hill_climbing

from HP import _HP, get_script_arguments

cfg = _HP("src/test.yaml")
cfg.update(get_script_arguments(cfg.keys()))


# Function to set the seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Function to extract the loss value from a filename
def extract_loss(filename):
    try:
        # Identify and extract the "_loss_X" portion
        return float((filename.split("_")[-1])[:-3])
    except ValueError:
        pass
    return float("inf")  # Return a large number if extraction fails


# Function to load the model weights from wandb
def load_model(model, folder):
    # List all files in the directory
    files = [f for f in os.listdir(folder) if f.endswith(".pt")]
    if files:
        # Find the file with the smallest loss
        file_with_min_loss = min(files, key=extract_loss)
    model.load_state_dict(torch.load(folder + file_with_min_loss, weights_only=True))
    print(f"Loaded model from {file_with_min_loss}")
    return model


def evaluate_problem(pb_dim, steps, cfg, actor):
    print(f"Thread started for problem dimension {pb_dim} and steps {steps}")
    dict = {}
    cfg["OUTER_STEPS"] = steps
    # Initialize the problem
    problem = TSP(pb_dim, cfg["N_PROBLEMS"], device=cfg["DEVICE"])
    # Create random instances
    params = problem.generate_params()
    params = {k: v.to(cfg["DEVICE"]) for k, v in params.items()}
    problem.set_params(**params)
    # Find initial solutions
    init_x = problem.generate_init_x()

    if cfg["HILL"]:
        start_time = time.time()
        _, hill_cost, hill_ngain, hill_primal = hill_climbing(problem, init_x, cfg)
        hill_time = time.time() - start_time
        print(f"Hill climbing done for problem dimension {pb_dim} and k = {steps}")

    # Define temperature decay parameter as a function of the number of steps
    alpha = np.log(cfg["STOP_TEMP_BASE"]) - np.log(cfg["INIT_TEMP_BASE"])
    cfg["ALPHA"] = np.exp(alpha / cfg["OUTER_STEPS"]).item()

    start_time = time.time()
    baseline = sa(actor, problem, init_x, cfg, replay=None, baseline=True, greedy=False)
    baseline_time = time.time() - start_time
    print(f"Baseline done for problem dimension {pb_dim} and k = {steps}")

    # Define temperature decay parameter as a function of the number of steps
    alpha = np.log(cfg["STOP_TEMP"]) - np.log(cfg["INIT_TEMP"])
    cfg["ALPHA"] = np.exp(alpha / cfg["OUTER_STEPS"]).item()

    with torch.no_grad():
        actor.eval()
        start_time = time.time()
        greedy = sa(
            actor, problem, init_x, cfg, replay=None, baseline=False, greedy=True
        )
        greedy_time = time.time() - start_time
        print(f"Greedy done for problem dimension {pb_dim} and k = {steps}")

        start_time = time.time()
        train = sa(
            actor, problem, init_x, cfg, replay=None, baseline=False, greedy=False
        )
        train_time = time.time() - start_time
        print(f"Training done for problem dimension {pb_dim} and k = {steps}")

    metrics = {
        "avg_min_cost_greedy": torch.mean(greedy["min_cost"]).item(),
        "avg_min_cost_baseline": torch.mean(baseline["min_cost"]).item(),
        "avg_min_cost_train": torch.mean(train["min_cost"]).item(),
        "avg_ngain_greedy": torch.mean(greedy["ngain"]).item(),
        "avg_ngain_baseline": torch.mean(baseline["ngain"]).item(),
        "avg_ngain_train": torch.mean(train["ngain"]).item(),
        "avg_primal_greedy": torch.mean(greedy["primal"]).item(),
        "avg_primal_baseline": torch.mean(baseline["primal"]).item(),
        "avg_primal_train": torch.mean(train["primal"]).item(),
        "time_greedy": greedy_time,
        "time_baseline": baseline_time,
        "time_train": train_time,
    }

    if cfg["HILL"]:
        metrics.update(
            {
                "avg_min_cost_hill": torch.mean(hill_cost).item(),
                "avg_ngain_hill": torch.mean(hill_ngain).item(),
                "avg_primal_hill": torch.mean(hill_primal).item(),
                "time_hill": hill_time,
            }
        )

    dict[pb_dim] = metrics
    print(f"For problem dimension {pb_dim} and k = {steps}: DONE")

    # Create a table
    table = Table(title="Results Summary")

    # Add columns
    table.add_column("Problem Dimension", style="cyan", justify="center")
    table.add_column("Metric", style="magenta", justify="center")
    table.add_column("Value", style="green", justify="center")

    # Extract and sort data
    sorted_data = []
    for pb_dim, metrics in dict.items():
        for metric_name, value in metrics.items():
            sorted_data.append((pb_dim, metric_name, value))

    # Sort by metric (prioritize min_cost) and then by dimension
    sorted_data.sort(key=lambda x: (x[1] != "min_cost", x[0], x[1]))
    thresh = 3
    if cfg["HILL"]:
        thresh = 4
    # Add rows to the table with separators every three values
    for i, (pb_dim, metric_name, value) in enumerate(sorted_data):
        table.add_row(str(pb_dim), metric_name, f"{value:.4f}")
        if (i + 1) % thresh == 0:
            table.add_row("------", "------", "------")  # Add a separator row
    # print(table)
    path = os.path.join(cfg["MODEL_DIR"], f"dim_{pb_dim}_K_{steps}_res.txt")
    with open(path, "w") as file:
        console = Console(file=file)
        console.print(table)
    print(f"Thread finished for problem dimension {pb_dim} and steps {steps}")

    return metrics


# def main():
#     # Initialize the actor model
#     actor = TSPActor(cfg["EMBEDDING_DIM"], device=cfg["DEVICE"])
#     actor = load_model(actor, cfg["MODEL_DIR"])
#     set_seed(cfg["SEED"])

#     for pb_dim in cfg["PROBLEM_DIM"]:
#         for steps in [
#             pb_dim**2,
#             2 * pb_dim**2,
#             5 * pb_dim**2,
#             10 * pb_dim**2,
#         ]:
#             evaluate_problem(pb_dim, steps, cfg, actor)


def main():
    # Initialize the actor model
    actor = TSPActor(cfg["EMBEDDING_DIM"], device=cfg["DEVICE"])
    actor = load_model(actor, cfg["MODEL_DIR"])
    set_seed(cfg["SEED"])

    # Create a list of tasks
    tasks = [
        (pb_dim, steps, cfg, actor)
        for pb_dim in cfg["PROBLEM_DIM"]
        for steps in [pb_dim**2, 2 * pb_dim**2, 5 * pb_dim**2, 10 * pb_dim**2]
    ]

    # Use ThreadPoolExecutor to run tasks in parallel
    with ThreadPoolExecutor(max_workers=cfg["N_THREADS"]) as executor:
        results = list(executor.map(lambda p: evaluate_problem(*p), tasks))

    # Process results
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
