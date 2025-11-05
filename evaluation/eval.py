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
from sa import sa_train
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
FOLDER = "dd"
rapid = False  # Set to True for faster execution, False for full evaluation
dim = 100  # Problem dimension used for BDD if rapid is False [50, 100, 500, 1000]
###########################################################################

if dim not in [50, 100, 500, 1000]:
    raise ValueError("Invalid problem dimension. Choose from [50, 100, 500, 1000].")

if rapid:
    logger.info("Running in rapid mode for quick evaluation.")
else:
    BDD_PATH = f"generated_problem/gen{dim}.pt"
    logger.info(f"Running evaluation on {BDD_PATH}")

# Constants
BASE_PATH = "res/" + FOLDER + "/"
if not os.path.exists(BASE_PATH):
    os.makedirs(BASE_PATH, exist_ok=True)
PATH = "wandb/Neural_Simulated_Annealing/"
RESULTS_FILE_ALL_MODEL = BASE_PATH + (
    "res_all_model_rapid.csv" if rapid else f"res_all_model_{dim}.csv"
)
RESULTS_FILE = (
    BASE_PATH + "res_model_rapid.csv" if rapid else BASE_PATH + f"res_model_{dim}.csv"
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
    "PROBLEM_DIM": 50 if rapid else 1000,
    "N_PROBLEMS": 100 if rapid else 1000,
    "OUTER_STEPS": 100 if rapid else 10000,
    "DEVICE": (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    ),
    "SEED": 0,
    "LOAD_PB": False if rapid else True,
    "INIT": "random",
    "MULTI_INIT": False,
}

tmp_dict = {100: 50, 50: 40, 20: 30, 10: 20}
cfg["MAX_LOAD"] = tmp_dict.get(cfg["PROBLEM_DIM"], 40)


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
    problem: object,
    init_x: torch.Tensor,
):
    """Main execution function."""
    logger.info(f"Processing model: {model_name}")
    # init HP
    HP = init_problem_parameters(model_name, cfg)

    problem.set_heuristic(HP["HEURISTIC"])
    problem.params = HP
    # Initialize models
    if HP["MODEL"] == "pairs":
        actor = CVRPActorPairs(
            embed_dim=HP["EMBEDDING_DIM"],
            c=HP["ENTRY"],
            num_hidden_layers=HP["NUM_H_LAYERS"],
            device=HP["DEVICE"],
            mixed_heuristic=True if len(HP["HEURISTIC"]) > 1 else False,
            method=HP["UPDATE_METHOD"],
        )
    elif HP["MODEL"] == "seq":
        actor = CVRPActor(
            embed_dim=HP["EMBEDDING_DIM"],
            c=HP["ENTRY"],
            num_hidden_layers=HP["NUM_H_LAYERS"],
            device=HP["DEVICE"],
            mixed_heuristic=True if len(HP["HEURISTIC"]) > 1 else False,
            method=HP["UPDATE_METHOD"],
        )
    # Load model
    actor = load_model(actor, model_name, "actor")

    # Run simulated annealing with time measurement
    start_time = time.time()

    test = sa_train(
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

    step = HP["OUTER_STEPS"]
    HP["OUTER_STEPS"] *= 20
    step_baseline = HP["OUTER_STEPS"]
    start_time = time.time()
    test_baseline = sa_train(
        actor,
        problem,
        init_x,
        HP,
        replay_buffer=None,
        baseline=True,
        greedy=False,
        desc_tqdm="NSA Model",
    )
    execution_time_baseline = time.time() - start_time
    HP["OUTER_STEPS"] = step
    final_cost_baseline = torch.mean(test_baseline["min_cost"])
    # Clear CUDA cache if using GPU
    if cfg["DEVICE"] == "cuda":
        torch.cuda.empty_cache()
    return (
        init_cost,
        final_cost,
        final_cost_baseline,
        execution_time,
        execution_time_baseline,
        step,
        step_baseline,
    )


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
        "execution_time_baseline",
        "NSA_steps",
        "SA_steps",
    ] + list(differing_keys)
    new_df, RESULTS_FILE = initialize_results_df(columns)
    all_models_results_df = load_results_models()
    problem, _, _ = init_pb(cfg, coords, demands, capacities)
    init_x = problem.generate_init_state(cfg["INIT"])
    init_cost = torch.mean(problem.cost(init_x))
    logger.info(f"CVRP problem initialized. Initial cost: {init_cost:.4f}")

    for model_name in tqdm(MODEL_NAMES, desc="Processing models"):

        (
            init_cost,
            final_cost,
            final_cost_baseline,
            execution_time,
            execution_time_baseline,
            step,
            step_baseline,
        ) = perform_test(model_name, problem, init_x)
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
                        "execution_time_baseline": [execution_time_baseline],
                        "NSA_steps": [step],
                        "SA_steps": [step_baseline],
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
