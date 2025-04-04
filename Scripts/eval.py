import pandas as pd
import torch
import os
import numpy as np
import random
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
MODEL_NAME = "20250331_195027_m0824xzz"
FULL_PATH = os.path.join(PATH, MODEL_NAME)
RESULTS_FILE = "res/res_model.csv"
HEURISTIC = "two_opt"

# Configuration
CFG = {
    "PROBLEM_DIM": 20,
    "N_PROBLEMS": 500,
    "MAX_LOAD": 30,
    "CLUSTERING": False,
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
    "HEURISTIC": HEURISTIC,
    "name": None,
}


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
            "initial_cost",
            "final_cost",
        ]
    )


def set_seed(seed=0):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    """Main execution function."""

    set_seed()
    # Initialize results DataFrame
    new_df = initialize_results_df()

    # Load model
    actor = CVRPActorPairs(
        device="cpu", mixed_heuristic=True if HEURISTIC == "mix" else False
    )
    actor = load_model(actor, FULL_PATH)

    # Initialize CVRP problem environment
    problem = CVRP(
        CFG["PROBLEM_DIM"],
        CFG["N_PROBLEMS"],
        CFG["MAX_LOAD"],
        device=CFG["DEVICE"],
        params=CFG,
    )
    problem.manual_seed(0)

    # Generate new problem instances
    params = problem.generate_params()
    params = {k: v.to(CFG["DEVICE"]) for k, v in params.items()}
    problem.set_params(params)

    # Get initial solutions
    init_x = problem.generate_init_x()
    initial_cost = torch.mean(problem.cost(init_x))
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
    final_cost = torch.mean(test["min_cost"])

    # Add results to DataFrame
    new_df = pd.concat(
        [
            new_df,
            pd.DataFrame(
                {
                    "model": [MODEL_NAME],
                    "initial_cost": [initial_cost.item()],
                    "final_cost": [final_cost.item()],
                }
            ),
        ],
        ignore_index=True,
    )
    # Save updated results
    new_df.to_csv(RESULTS_FILE, index=False)
    print("Results saved to", RESULTS_FILE)


if __name__ == "__main__":

    main()
