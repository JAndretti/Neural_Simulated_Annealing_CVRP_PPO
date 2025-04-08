import pandas as pd
import torch
import os
import numpy as np
import random
import warnings
import yaml

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from problem import CVRP
from sa import sa
from model import CVRPActorPairs
from tqdm import tqdm

# Suppress warnings if needed
warnings.filterwarnings("ignore")

# Constants
PATH = "wandb/Neural_Simulated_Annealing/models/"
RESULTS_FILE = "res/res_model.csv"
MODEL_NAMES = [
    "20250404_025649_rkxgs6ga",
    "20250403_200902_28z10n0a",
    "20250405_053104_yh9z35z2",
    "20250405_044605_jh94xgch",
    "20250404_090907_zi1rvs8g",
    "20250404_223057_pkcua1in",
    "20250403_200902_hwzuk154",
    "20250404_160429_qfifvsbc",
    "20250404_154956_wmuzlbls",
    "20250404_092956_lqq1yf4w",
    "20250404_221246_wvrdjdcv",
    "20250404_023801_x70b5bv2",
]

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
    "name": None,
    "HEURISTIC": "mix",
}


def get_heuristic_for_model(model_name):
    """Extract heuristic from HP.yaml file."""
    hp_file = os.path.join(PATH, model_name, "HP.yaml")
    try:
        with open(hp_file, "r") as file:
            content = file.read()
            content_clean = content.replace("!!python/object:HP._HP", "")
            hp_data = yaml.unsafe_load(content_clean)
            return (
                hp_data.get("config", {}).get("HEURISTIC")
                if isinstance(hp_data, dict)
                else None
            )
    except Exception as e:
        print(f"Error for reading HP.yaml file for mdoel : {model_name}: {e}")
        return None


def init_heuristic(model_name):
    """Initialize heuristic based on model names."""
    # Get heuristic from model name
    heuristic = get_heuristic_for_model(model_name)
    if heuristic is None:
        heuristic = "mix"
    CFG["HEURISTIC"] = heuristic


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


def init_pb():
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
    return problem, init_x, initial_cost


def perform_test(
    model_name: str,
    problem: CVRP,
    init_x: torch.Tensor,
):
    """Main execution function."""

    # Initialize heuristic
    init_heuristic(model_name)
    # Set heuristic in problem
    problem.set_heuristic(CFG["HEURISTIC"])

    # Load model
    actor = CVRPActorPairs(
        device="cpu", mixed_heuristic=True if CFG["HEURISTIC"] == "mix" else False
    )
    FULL_PATH = os.path.join(PATH, model_name)

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
    final_cost = torch.mean(test["min_cost"])
    return final_cost


if __name__ == "__main__":
    set_seed()
    # Initialize results DataFrame
    new_df = initialize_results_df()
    problem, init_x, init_cost = init_pb()
    for model_name in tqdm(MODEL_NAMES, desc="Processing models"):
        final_cost = perform_test(model_name, problem, init_x)
        # Add results to DataFrame
        new_df = pd.concat(
            [
                new_df,
                pd.DataFrame(
                    {
                        "model": [model_name],
                        "initial_cost": [init_cost.item()],
                        "final_cost": [final_cost.item()],
                    }
                ),
            ],
            ignore_index=True,
        )
    # Save updated results
    new_df.to_csv(RESULTS_FILE, index=False)
    print("Results saved to", RESULTS_FILE)
