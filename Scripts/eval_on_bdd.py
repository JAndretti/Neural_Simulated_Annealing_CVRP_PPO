import pandas as pd
import torch
import os
import numpy as np
import random
from tqdm import tqdm
from multiprocessing import Pool
import warnings

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from problem import CVRP
from sa import sa
from model import CVRPActorPairs

# Suppress warnings if needed
warnings.filterwarnings("ignore")

# Constants
PATH = "wandb/Neural_Simulated_Annealing/models/"
GROUP = "bench_model_temp"
MODEL_NAME = "20250402_171304_a3bnyrfg"
FULL_PATH = os.path.join(PATH, GROUP, MODEL_NAME)
RESULTS_FILE = "res/model_on_bdd_swap_vs_2opt.csv"
HEURISTIC = "mix"
BATCH_SIZE = 20  # Number of parallel processes

# Configuration
CFG = {
    "N_PROBLEMS": 1,
    "DEVICE": "cpu",
    "EMBEDDING_DIM": 32,
    "INIT_TEMP": 1.0,
    "STOP_TEMP": 0.01,
    "INNER_STEPS": 1,
    "OUTER_STEPS": 1000,
    "SCHEDULER": "lam",
    "METHOD": "ppo",
    "REWARD": "immediate",
    "GAMMA": 0.9,
}


def create_distance_matrix(coords):
    """Vectorized distance matrix calculation."""
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt((diff**2).sum(axis=2))


def calculate_route_cost(route, distance_matrix):
    """Calculate total distance of a route."""
    return distance_matrix[route[:-1], route[1:]].sum()


def extract_loss(filename):
    """Extract loss value from filename."""
    try:
        return float((filename.split("_")[-1])[:-3])
    except ValueError:
        return float("inf")


def load_model(model, folder):
    """Load model with smallest loss."""
    files = [f for f in os.listdir(folder) if f.endswith(".pt")]
    if files:
        best_file = min(files, key=extract_loss)
        model.load_state_dict(
            torch.load(os.path.join(folder, best_file), weights_only=True)
        )
    return model


def initialize_results_df():
    """Initialize or load results DataFrame."""
    if os.path.exists(RESULTS_FILE):
        print(f"Loading existing results file from {RESULTS_FILE}")
        return pd.read_csv(RESULTS_FILE)
    return pd.DataFrame(
        columns=[
            "model",
            "problem_name",
            "initial_cost",
            "final_cost",
            "best_cost",
            "best_real_cost",
            "Diff_bestCost_vs_realBestCost",
            "Diff_Fcost_vs_BestCost",
        ]
    )


def process_single_row(row_dict):
    """Process a single row - modified for parallel processing."""
    try:
        # Set seeds for each process (important for reproducibility)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        # Prepare data
        coord = row_dict["node_coord"]
        distance_matrix = create_distance_matrix(coord)
        best_real_cost = row_dict["cost"]

        # Calculate best cost from solution
        solution = np.concatenate(
            [[0]] + [np.array(lst) for lst in row_dict["solution"]] + [[0]]
        )
        best_cost = calculate_route_cost(solution, distance_matrix)

        # Initialize problem
        row_dict["HEURISTIC"] = HEURISTIC
        problem = CVRP(device="cpu", params=row_dict)
        problem.manual_seed(0)
        params = problem.generate_params("test", True)
        params = {k: v.to("cpu") for k, v in params.items()}
        problem.set_params(params)

        # Generate initial solution and calculate cost
        init_x = problem.generate_init_x()
        init_route = init_x.squeeze(-1).squeeze(0).numpy()
        init_cost = calculate_route_cost(init_route, distance_matrix)

        # Load model inside each process (important for multiprocessing)
        actor = CVRPActorPairs(
            device="cpu", mixed_heuristic=True if HEURISTIC == "mix" else False
        )
        actor = load_model(actor, FULL_PATH)

        # Run simulated annealing
        test = sa(
            actor,
            problem,
            init_x,
            CFG,
            replay_buffer=None,
            baseline=False,
            greedy=False,
        )
        final_route = test["best_x"].squeeze(-1).squeeze(0).numpy().astype(int)
        final_cost = calculate_route_cost(final_route, distance_matrix)

        return {
            "model": MODEL_NAME,
            "problem_name": row_dict["name"],
            "initial_cost": init_cost,
            "final_cost": final_cost,
            "best_cost": best_cost,
            "best_real_cost": best_real_cost,
            "Diff_bestCost_vs_realBestCost": (best_cost - best_real_cost),
            "Diff_Fcost_vs_BestCost": (final_cost - best_cost),
        }
    except Exception as e:
        print(f"Error processing {row_dict.get('name', 'unknown')}: {str(e)}")
        return None


def process_batch(rows):
    """Process a batch of rows in parallel."""
    with Pool(processes=BATCH_SIZE) as pool:
        results = list(tqdm(pool.imap(process_single_row, rows), total=len(rows)))
    return [r for r in results if r is not None]  # Filter out failed cases


def main():
    """Main execution function."""
    # Initialize results DataFrame
    new_df = initialize_results_df()

    # Load problem data
    df = pd.read_pickle("bdd/bdd.pkl")
    df = df[df["dimension"] <= 500]

    # Convert to list of dicts for parallel processing
    rows = [row.to_dict() for _, row in df.iterrows()]

    # Process in batches to avoid memory issues
    batch_results = []
    for i in tqdm(range(0, len(rows), BATCH_SIZE), desc="Processing batches"):
        batch = rows[i : i + BATCH_SIZE]
        batch_results.extend(process_batch(batch))

    # Combine results
    if batch_results:
        new_df = pd.concat([new_df, pd.DataFrame(batch_results)], ignore_index=True)

    # Save results
    new_df.to_csv(RESULTS_FILE, index=False)
    print(f"Saved results file at {RESULTS_FILE}")


if __name__ == "__main__":
    # # Important for Windows compatibility
    # if os.name == "nt":
    #     from multiprocessing import freeze_support

    #     freeze_support()
    main()
