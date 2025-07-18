import vrplib
import pandas as pd
import torch
import numpy as np
from loguru import logger  # Enhanced logging capabilities

from func import init_problem_parameters, set_seed, load_model, init_pb

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sa import sa
from model import CVRPActorPairs, CVRPActor

# Remove default logger
logger.remove()
# Add custom logger with colored output
logger.add(
    lambda msg: print(msg, end=""),
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "  # Timestamp in green
        "<blue>{file}:{line}</blue> | "  # File and line in blue
        "<yellow>{message}</yellow>"  # Message in yellow
    ),
    colorize=True,
)

PATH = "wandb/Neural_Simulated_Annealing/"
FOLDER = "baseline_methods_2" + "/models/"
MODEL = "20250630_225812_yhajndhh"
MODEL_PATH = PATH + FOLDER + MODEL

cfg = {
    "PROBLEM_DIM": 100,
    "N_PROBLEMS": 10000,
    "OUTER_STEPS": 20000,
    "DEVICE": (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    ),
    "INIT": "sweep",
    "SEED": 0,
    "LOAD_PB": True,
}


def init_res(path: str, names: list[str] = None):
    """Initialize results DataFrame."""
    logger.info(f"Initializing results DataFrame at path: {path}")
    if not os.path.exists(path):
        df = pd.DataFrame(
            columns=[
                "name",
            ]
        )
        if names is not None:
            df["name"] = names
        logger.info("Results DataFrame created.")
    else:
        df = pd.read_csv(path)
        logger.info("Results DataFrame loaded from existing file.")
    return df


def calculate_distance(
    solutions: torch.Tensor, node_coords_tensor: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the total distance for each solution based on the node coordinates.

    Args:
        solutions (torch.Tensor): A tensor containing the solutions (routes).
        node_coords_tensor (torch.Tensor): A tensor containing the non-normalized
        node coordinates.

    Returns:
        torch.Tensor: A tensor containing the total distance for each solution.
    """
    logger.info("Calculating distances for solutions...")
    total_distances = torch.zeros(solutions.size(0), device=solutions.device)

    for i, solution in enumerate(solutions):
        distance = 0.0
        route_coords = node_coords_tensor[i]  # Index using batch dimension
        for j in range(len(solution) - 1):
            distance += torch.sqrt(
                torch.sum(
                    (route_coords[solution[j]] - route_coords[solution[j + 1]]) ** 2
                )
            )
        total_distances[i] = distance

    logger.info("Distances calculated successfully.")
    return total_distances


if __name__ == "__main__":
    logger.info("Starting evaluation script...")
    # Set seed for reproducibility
    set_seed(cfg["SEED"])
    logger.info("Seed set for reproducibility.")

    path_file = "bdd/Vrp-Set-XML100/instances/"
    path_baseline = "res/Vrp-Set-XML100_res.csv"
    df = pd.read_csv(path_baseline)
    logger.info(f"Loaded baseline data from {path_baseline}.")
    instances = [path_file + name + ".vrp" for name in df["name"]]

    names = []
    dimensions = []
    capacities = []
    node_coords = []
    demands = []
    depots = []
    opt_costs = []

    logger.info("Reading VRP instances...")
    for instance_path in instances:
        data = vrplib.read_instance(instance_path)
        name = data["name"]
        names.append(name)
        dimensions.append(data["dimension"])
        capacities.append(data["capacity"])
        node_coords.append(data["node_coord"])
        demands.append(data["demand"])
        depots.append(data["depot"])
    logger.info("Finished reading VRP instances.")

    # Create a tensor for each attribute
    logger.info("Creating tensors for problem attributes...")
    dimension_tensor = torch.tensor(dimensions)
    capacity_tensor = torch.tensor(capacities).unsqueeze(-1)
    node_coords_tensor = torch.tensor(np.array(node_coords))
    demands_tensor = torch.tensor(np.array(demands))
    depots_tensor = torch.tensor(np.array(depots))

    # Normalize node_coords_tensor between 0 and 1 for each row
    logger.info("Normalizing node coordinates...")
    min_coords = torch.min(node_coords_tensor, dim=1, keepdim=True).values
    max_coords = torch.max(node_coords_tensor, dim=1, keepdim=True).values
    node_coords_tensor_n = (node_coords_tensor - min_coords) / (max_coords - min_coords)
    logger.info("Node coordinates normalized.")

    CFG = init_problem_parameters(MODEL_PATH, cfg)
    problem, init_x, initial_cost = init_pb(
        CFG, node_coords_tensor_n, demands_tensor, capacity_tensor
    )
    # Initialize models
    logger.info("Initializing models...")
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
    logger.info("Models initialized.")

    # Load model
    logger.info(f"Loading model from {MODEL_PATH}...")
    actor = load_model(actor, MODEL_PATH, "actor")
    logger.info("Model loaded successfully.")

    logger.info("Starting simulated annealing...")
    test = sa(
        actor,
        problem,
        init_x,
        CFG,
        replay_buffer=None,
        baseline=False,
        greedy=False,
    )
    final_cost = torch.mean(test["min_cost"])
    logger.info(f"Simulated annealing completed. Final cost: {final_cost:.4f}")

    solutions = test["best_x"].cpu().detach()

    # Calculate distances using the solutions and non-normalized node coordinates
    distances = calculate_distance(solutions, node_coords_tensor)
    logger.info(f"Calculated distances: {distances}")

    tmp_df = pd.DataFrame(
        {
            "name": names,
            FOLDER + MODEL: distances,
        }
    )

    logger.info("Saving results...")
    df = init_res("res/models_res_on_bdd.csv", names=names)
    df = pd.merge(df, tmp_df, on="name", how="outer")
    df.to_csv("res/models_res_on_bdd.csv", index=False)
    logger.info("Results saved to res/models_res_on_bdd.csv.")
    print("Results saved to res/models_res_on_bdd.csv")
