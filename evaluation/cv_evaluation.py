import pandas as pd
import torch
import os
import warnings
import glob2
import sys
import time  # Add import for time measurement

from func import (
    set_seed,
    load_model,
    init_problem_parameters,
    init_pb,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from sa import sa_test, sa_baseline
from model import CVRPActorPairs, CVRPActor
from tqdm import tqdm
from loguru import logger  # Enhanced logging capabilities

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

# Suppress warnings if needed
warnings.filterwarnings("ignore")

# TO FILL
###########################################################################
FOLDER = "INIT_METHODS"
rapid = False  # Set to True for faster execution, False for full evaluation
dim = 100  # Problem dimension used for BDD if rapid is False [50, 100, 500, 1000]
cv_key = "INIT"
cv_val = ["random", "isolate", "sweep", "Clark_and_Wright", "nearest_neighbor"]
###########################################################################

if dim not in [50, 100, 500, 1000]:
    raise ValueError("Invalid problem dimension. Choose from [50, 100, 500, 1000].")

if rapid:
    logger.info("Running in rapid mode for quick evaluation.")
else:
    BDD_PATH = f"generated_problem/gen{dim}.pt"
    logger.info(f"Running evaluation on {BDD_PATH}")

# Constants
PATH = "wandb/Neural_Simulated_Annealing/"
RESULTS_FILE_ALL_MODEL = (
    f"res/{FOLDER}/cv_{cv_key}_all_model_rapid.csv"
    if rapid
    else f"res/{FOLDER}/cv_{cv_key}_all_model_{dim}.csv"
)
# Ensure the results folder exists
os.makedirs(f"res/{FOLDER}/", exist_ok=True)
RESULTS_FILE = (
    PATH + FOLDER + "/cv_res_model_rapid.csv"
    if rapid
    else PATH + FOLDER + f"/cv_res_model_{dim}.csv"
)
MODEL_NAMES = glob2.glob(os.path.join(PATH, FOLDER, "models", "*"))

if not rapid:
    bdd = torch.load(BDD_PATH, map_location="cpu")
    coords = bdd["node_coords"]
    demands = bdd["demands"]
    capacities = bdd["capacity"]
else:
    coords = None
    demands = None
    capacities = None

# Configuration
cfg = {
    "PROBLEM_DIM": 100 if rapid else coords.shape[1] - 1,
    "N_PROBLEMS": 1000 if rapid else capacities.shape[0],
    "OUTER_STEPS": 2000 if rapid else 10000,
    "MAX_LOAD": 50,
    "DEVICE": (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    ),
    "INIT": "nearest_neighbor",
    "CLUSTERING": False,
    "SEED": 0,
    "LOAD_PB": False if rapid else True,
}


logger.info(f"Device set to {cfg['DEVICE']}")


def initialize_results_df(columns: list):
    """Initialize or load results DataFrame."""
    new_file = RESULTS_FILE
    if os.path.exists(RESULTS_FILE):
        logger.info(
            f"Existing results file found at {RESULTS_FILE}, creating a new file."
        )
        base, ext = os.path.splitext(RESULTS_FILE)
        i = 1
        new_file = f"{base}_{i}{ext}"
        while os.path.exists(new_file):
            i += 1
            new_file = f"{base}_{i}{ext}"
    return pd.DataFrame(columns=columns), new_file


def load_results_models():
    """Load results from all models into a DataFrame."""
    if os.path.exists(RESULTS_FILE_ALL_MODEL):
        df = pd.read_csv(RESULTS_FILE_ALL_MODEL)
        logger.info(f"Loaded existing results from {RESULTS_FILE_ALL_MODEL}")
    else:
        df = pd.DataFrame(
            columns=[
                "model",
                "final_cost",
                "cv_key",
                "train_param",
                "test_param",
            ]
        )
        logger.info(f"Created new DataFrame for results at {RESULTS_FILE_ALL_MODEL}")
    return df


def perform_test(
    model_name: str,
):
    """Main execution function."""
    logger.info(f"Processing model: {model_name}")
    # init HP
    HP = init_problem_parameters(model_name, cfg)

    problem, _, _ = init_pb(HP, coords, demands, capacities)

    # Initialize models
    if HP["PAIRS"]:
        actor = CVRPActorPairs(
            HP["EMBEDDING_DIM"],
            num_hidden_layers=HP["NUM_H_LAYERS"],
            device=HP["DEVICE"],
            mixed_heuristic=True if HP["HEURISTIC"] == "mix" else False,
        )
    else:
        actor = CVRPActor(
            HP["EMBEDDING_DIM"],
            HP["ENTRY"],
            num_hidden_layers=HP["NUM_H_LAYERS"],
            device=HP["DEVICE"],
            mixed_heuristic=True if HP["HEURISTIC"] == "mix" else False,
            method=HP["UPDATE_METHOD"],
        )
    # Load model
    actor = load_model(actor, model_name, "actor")

    # Run simulated annealing with time measurement
    start_time = time.time()
    init_x = problem.generate_init_x(HP["INIT"])
    test = sa_test(
        actor,
        problem,
        init_x,
        HP,
        replay_buffer=None,
        baseline=False,
        greedy=False,
        desc_tqdm="NSA Model",
    )
    execution_time = time.time() - start_time
    init_cost = torch.mean(problem.cost(init_x))
    final_cost = torch.mean(test["min_cost"])
    test_baseline = sa_baseline(
        problem,
        init_x,
        HP,
        desc_tqdm="SA Baseline",
    )
    final_cost_baseline = torch.mean(test_baseline["min_cost"])
    return init_cost, final_cost, final_cost_baseline, execution_time


if __name__ == "__main__":
    set_seed(cfg["SEED"])  # Set random seed for reproducibility

    # Initialize results DataFrame with dynamic columns
    columns = [
        "model",
        "initial_cost",
        "final_cost",
        "final_cost_baseline",
        "execution_time",
        "cv_key",
        "train_param",
        "test_param",
    ]
    new_df, RESULTS_FILE = initialize_results_df(columns)
    all_models_results_df = load_results_models()

    for model_name in tqdm(MODEL_NAMES, desc="Processing models"):
        for cv_v in tqdm(cv_val, desc="Processing CV values", leave=False):
            cfg[cv_key] = cv_v
            init_cost, final_cost, final_cost_baseline, execution_time = perform_test(
                model_name
            )
            cv_train = init_problem_parameters(model_name, {})[cv_key]
            # Add results to DataFrame
            new_df = pd.concat(
                [
                    new_df,
                    pd.DataFrame(
                        {
                            "model": [model_name.split("/")[-1]],
                            "initial_cost": [init_cost.item()],
                            "final_cost": [final_cost.item()],
                            "final_cost_baseline": [final_cost_baseline.item()],
                            "execution_time": [execution_time],
                            "cv_key": [cv_key],
                            "train_param": [cv_train],
                            "test_param": [cv_v],
                        }
                    ),
                ],
                ignore_index=True,
            )
            # Save results for all models
            all_models_results_df = pd.concat(
                [
                    all_models_results_df,
                    pd.DataFrame(
                        {
                            "model": [FOLDER + "/" + model_name.split("/")[-1]],
                            "final_cost": [final_cost.item()],
                            "cv_key": [cv_key],
                            "train_param": [cv_train],
                            "test_param": [cv_v],
                        }
                    ),
                ],
                ignore_index=True,
            )

    # Save updated results
    new_df.to_csv(RESULTS_FILE, index=False)
    logger.info(f"Results saved to {RESULTS_FILE}")
    logger.info("Results DataFrame:")
    print(new_df.head())

    all_models_results_df.to_csv(RESULTS_FILE_ALL_MODEL, index=False)
    logger.info("All models results DataFrame:")
    print(all_models_results_df.head())
