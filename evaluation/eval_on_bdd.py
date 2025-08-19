import vrplib
import pandas as pd
import torch
import numpy as np
import tqdm
from loguru import logger  # Enhanced logging capabilities
from func import init_problem_parameters, set_seed, load_model, init_pb

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sa import sa_train
from model import CVRPActorPairs, CVRPActor
from algo import compute_euclidean_distance_matrix

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

# TO FILL
###########################################################################
MODEL = "20250630_165952_txdtlez8"
cfg = {
    "PROBLEM_DIM": 100,
    "N_PROBLEMS": 10000,
    "OUTER_STEPS": 20000,
    "DEVICE": (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    ),
    "INIT": "Clark_and_Wright",
    "SEED": 0,
    "LOAD_PB": True,
}
BASELINE = False  # If True, eval on baseline method, else on the model
###########################################################################

if BASELINE:
    logger.info("Running in baseline mode.")

PATH = "wandb/Neural_Simulated_Annealing/"

# Find the path of the folder dynamically
MODEL_PATH = next(
    (
        os.path.join(PATH, folder, "models", MODEL)
        for folder in os.listdir(PATH)
        if MODEL in os.listdir(os.path.join(PATH, folder, "models"))
    ),
    None,
)

if MODEL_PATH is None:
    raise FileNotFoundError(f"Model folder containing {MODEL} not found in {PATH}")
FOLDER = MODEL_PATH.split("/")[-3] + "/" + MODEL_PATH.split("/")[-2] + "/"


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


def calculate_dist_route(route, distance_matrix):
    """Calculate the total distance of a given route."""
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i]][route[i + 1]]
    return total_distance


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
    matrixes = []

    logger.info("Reading VRP instances...")
    for instance_path in tqdm.tqdm(instances, leave=False, desc="Reading instances"):
        data = vrplib.read_instance(instance_path)
        name = data["name"]
        names.append(name)
        dimensions.append(data["dimension"])
        capacities.append(data["capacity"])
        node_coords.append(data["node_coord"])
        matrix = compute_euclidean_distance_matrix(
            [tuple(coord) for coord in data["node_coord"]]
        )
        matrixes.append(matrix)
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
    test = sa_train(
        actor,
        problem,
        init_x,
        CFG,
        replay_buffer=None,
        baseline=True if BASELINE else False,
        greedy=False,
        device=CFG["DEVICE"],
    )
    final_cost = torch.mean(test["min_cost"])
    logger.info(f"Simulated annealing completed. Final cost: {final_cost:.4f}")

    solutions = test["best_x"].cpu().detach()

    # Calculate distances using the solutions and non-normalized node coordinates
    distances = []
    for sol, dist_mat in zip(solutions, matrixes):
        distances.append(calculate_dist_route(sol.squeeze().numpy(), dist_mat))

    tmp_df = pd.DataFrame(
        {
            "name": names,
            FOLDER + MODEL if not BASELINE else "Baseline": distances,
        }
    )

    logger.info("Saving results...")
    df = init_res("res/models_res_on_bdd.csv", names=names)
    df = pd.merge(df, tmp_df, on="name", how="outer")
    df.to_csv("res/models_res_on_bdd.csv", index=False)
    logger.info("Results saved to res/models_res_on_bdd.csv.")
    print("Results saved to res/models_res_on_bdd.csv")
