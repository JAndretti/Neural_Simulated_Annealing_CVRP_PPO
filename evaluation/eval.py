import pandas as pd
import torch
import os
import warnings
import glob2
import sys
import time  # Add import for time measurement

from func import (
    get_HP_for_model,
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
FOLDER = "baseline_methods_2"
rapid = False  # Set to True for faster execution, False for full evaluation
dim = 500  # Problem dimension used for BDD if rapid is False [50, 100, 500, 1000]
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
    "res/res_all_model_rapid.csv" if rapid else f"res/res_all_model_{dim}.csv"
)
RESULTS_FILE = (
    PATH + FOLDER + "/res_model_rapid.csv"
    if rapid
    else PATH + FOLDER + f"/res_model_{dim}.csv"
)
MODEL_NAMES = glob2.glob(os.path.join(PATH, FOLDER, "models", "*"))
# BDD_OR_TOOLS_PATH =
# "generated_problem/problem_data_dim100_load50_solution_or_tools.pt"
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
    "OUTER_STEPS": 2000 if rapid else 20000,
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
        print(f"Existing results file found at {RESULTS_FILE}, creating a new file.")
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
        print(f"Loaded existing results from {RESULTS_FILE_ALL_MODEL}")
    else:
        df = pd.DataFrame(
            columns=[
                "model",
                "final_cost",
            ]
        )
        print(f"Created new DataFrame for results at {RESULTS_FILE_ALL_MODEL}")
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


def extract_differing_keys(model_names):
    """Extract keys with differing values across HP.yaml files."""
    all_hp_data = []
    for model_name in model_names:
        HP = get_HP_for_model(model_name)
        if HP:
            all_hp_data.append(HP)

    differing_keys = set()
    if all_hp_data:
        keys = all_hp_data[0].keys()
        for key in keys:
            values = {
                tuple(hp.get(key)) if isinstance(hp.get(key), list) else hp.get(key)
                for hp in all_hp_data
            }
            if len(values) > 1:  # Key has differing values
                differing_keys.add(key)
    return differing_keys


def add_hp_to_results(model_name, differing_keys):
    """Extract HP values for the given model and return as a dictionary."""
    HP = get_HP_for_model(model_name)
    if not HP:
        return {key: None for key in differing_keys}
    return {key: HP.get(key, None) for key in differing_keys}


if __name__ == "__main__":
    set_seed(cfg["SEED"])  # Set random seed for reproducibility

    # Extract keys with differing values across HP.yaml files
    differing_keys = extract_differing_keys(MODEL_NAMES)

    # Initialize results DataFrame with dynamic columns
    columns = [
        "model",
        "initial_cost",
        "final_cost",
        "final_cost_baseline",
        "execution_time",
    ] + list(differing_keys)
    new_df, RESULTS_FILE = initialize_results_df(columns)
    all_models_results_df = load_results_models()

    for model_name in tqdm(MODEL_NAMES, desc="Processing models"):
        init_cost, final_cost, final_cost_baseline, execution_time = perform_test(
            model_name
        )
        # Extract HP values for the current model
        hp_values = add_hp_to_results(model_name, differing_keys)
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
                        **hp_values,
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
                    }
                ),
            ],
            ignore_index=True,
        )
    # # Load OR-Tools solution mean distance and add to DataFrame
    # if os.path.exists(BDD_OR_TOOLS_PATH):
    #     or_tools_data = torch.load(BDD_OR_TOOLS_PATH, map_location="cpu")
    #     mean_or_distances = or_tools_data.get("mean_or_distances", float("nan"))
    # else:
    #     mean_or_distances = float("nan")

    # # Prepare a row for OR-Tools
    # or_tools_row = {
    #     "model": "or_tools",
    #     "initial_cost": float("nan"),
    #     "final_cost": mean_or_distances.item(),
    #     "execution_time": 60,
    # }
    # for key in differing_keys:
    #     or_tools_row[key] = float("nan")

    # new_df = pd.concat(
    #     [new_df, pd.DataFrame([or_tools_row])],
    #     ignore_index=True,
    # )

    # Remove duplicate rows based on the 'model' column
    new_df = new_df.drop_duplicates(subset=["model"], keep="first")
    all_models_results_df = all_models_results_df.drop_duplicates(
        subset=["model"], keep="first"
    )

    # Save updated results
    new_df.to_csv(RESULTS_FILE, index=False)
    print("Results saved to", RESULTS_FILE)
    logger.info("Results DataFrame:")
    print(new_df.head())

    all_models_results_df.to_csv(RESULTS_FILE_ALL_MODEL, index=False)
    logger.info("All models results DataFrame:")
    print(all_models_results_df.head())
