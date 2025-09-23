import pandas as pd
import torch
import os
import warnings
import glob2
import sys
import time  # Add import for time measurement
import numpy as np  # Add for random problem generation

from func import (
    get_HP_for_model,
    set_seed,
    load_model,
    init_problem_parameters,
    init_pb,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from sa import sa_test
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
FOLDER = "ARCH_NEW_2"
N_PROBLEMS = 1000  # Number of random problems to generate for each configuration
# VRP configurations: (name, customers, capacity)
VRP_CONFIGS = [
    ("VRP10", 10, 20),
    ("VRP20", 20, 30),
    ("VRP50", 50, 40),
    ("VRP100", 100, 50),
]
###########################################################################

logger.info(
    f"Testing {len(VRP_CONFIGS)} VRP configurations with {N_PROBLEMS} problems each"
)

# Constants
BASE_PATH = "res/" + FOLDER + "/"
if not os.path.exists(BASE_PATH):
    os.makedirs(BASE_PATH, exist_ok=True)
PATH = "wandb/Neural_Simulated_Annealing/"
RESULTS_FILE_ALL_MODEL = BASE_PATH + "res_all_model_multi_vrp.csv"
RESULTS_FILE = BASE_PATH + "res_model_multi_vrp.csv"
MODEL_NAMES = glob2.glob(os.path.join(PATH, FOLDER, "models", "*"))


# Generate random problems
def generate_random_cvrp_problems(n_problems, problem_size, capacity, device, seed=42):
    """Generate random CVRP problems."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate coordinates (depot + customers)
    coords = torch.rand(n_problems, problem_size + 1, 2, device=device)

    # Generate demands (depot has 0 demand, customers have demand 1-9)
    demands = torch.randint(
        1, 10, (n_problems, problem_size + 1), device=device, generator=generator
    )
    demands[:, 0] = 0
    # Set capacity for all problems
    capacities = torch.full((n_problems, 1), capacity, device=device)

    return coords, demands, capacities


# Configuration
cfg = {
    "DEVICE": (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    ),
    "SEED": 0,
    "LOAD_PB": True,
    "INIT": "isolate",
    "MULTI_INIT": False,
}

generator = torch.Generator(device=cfg["DEVICE"])
generator.manual_seed(cfg["SEED"])

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


def perform_test(model_name: str, vrp_config):
    """Main execution function for a specific VRP configuration."""
    config_name, problem_size, capacity = vrp_config
    logger.info(f"Processing model: {model_name} on {config_name}")

    # Update configuration for this VRP size
    current_cfg = cfg.copy()
    current_cfg.update(
        {
            "PROBLEM_DIM": problem_size,
            "N_PROBLEMS": N_PROBLEMS,
            "OUTER_STEPS": problem_size * 100,
            "MAX_LOAD": capacity,
        }
    )

    # Generate problems for this configuration
    coords, demands, capacities = generate_random_cvrp_problems(
        N_PROBLEMS, problem_size, capacity, cfg["DEVICE"]
    )

    # init HP
    HP = init_problem_parameters(model_name, current_cfg)

    problem, _, _ = init_pb(HP, coords, demands, capacities)

    # Initialize models
    if HP["PAIRS"]:
        actor = CVRPActorPairs(
            embed_dim=HP["EMBEDDING_DIM"],
            c=HP["ENTRY"],
            num_hidden_layers=HP["NUM_H_LAYERS"],
            device=HP["DEVICE"],
            mixed_heuristic=True if HP["HEURISTIC"] == "mix" else False,
            method=HP["UPDATE_METHOD"],
        )
    else:
        actor = CVRPActor(
            embed_dim=HP["EMBEDDING_DIM"],
            c=HP["ENTRY"],
            num_hidden_layers=HP["NUM_H_LAYERS"],
            device=HP["DEVICE"],
            mixed_heuristic=True if HP["HEURISTIC"] == "mix" else False,
            method=HP["UPDATE_METHOD"],
        )
    # Load model
    actor = load_model(actor, model_name, "actor")

    # Run simulated annealing with time measurement
    start_time = time.time()
    init_x = problem.generate_init_solution(HP["INIT"])
    test = sa_test(
        actor,
        problem,
        init_x,
        HP,
        replay_buffer=None,
        baseline=False,
        greedy=False,
        desc_tqdm=f"NSA Model {config_name}",
    )
    execution_time = time.time() - start_time
    init_cost = torch.mean(problem.cost(init_x))
    final_cost = torch.mean(test["min_cost"])

    # Clear CUDA cache if using GPU
    if cfg["DEVICE"] == "cuda":
        torch.cuda.empty_cache()
    return init_cost, final_cost, execution_time


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
        "vrp_config",
        "problem_size",
        "capacity",
        "initial_cost",
        "final_cost",
        "execution_time",
    ] + list(differing_keys)
    new_df, RESULTS_FILE = initialize_results_df(columns)
    all_models_results_df = load_results_models()

    for model_name in tqdm(MODEL_NAMES, desc="Processing models"):
        for vrp_config in VRP_CONFIGS:
            config_name, problem_size, capacity = vrp_config
            init_cost, final_cost, execution_time = perform_test(model_name, vrp_config)

            # Extract HP values for the current model
            hp_values = add_hp_to_results(model_name, differing_keys)

            # Add results to DataFrame
            new_df = pd.concat(
                [
                    new_df,
                    pd.DataFrame(
                        {
                            "model": [model_name.split("/")[-1]],
                            "vrp_config": [config_name],
                            "problem_size": [problem_size],
                            "capacity": [capacity],
                            "initial_cost": [init_cost.item()],
                            "final_cost": [final_cost.item()],
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
                            "model": [
                                f"{FOLDER}/{model_name.split('/')[-1]}_{config_name}"
                            ],
                            "final_cost": [final_cost.item()],
                        }
                    ),
                ],
                ignore_index=True,
            )

    # Remove duplicate rows based on the 'model' column
    # new_df = new_df.drop_duplicates(subset=["model"], keep="first")
    # all_models_results_df = all_models_results_df.drop_duplicates(
    #     subset=["model"], keep="first"
    # )

    # Save updated results
    new_df.to_csv(RESULTS_FILE, index=False)
    print("Results saved to", RESULTS_FILE)
    logger.info("Results DataFrame:")
    print(new_df.head())

    all_models_results_df.to_csv(RESULTS_FILE_ALL_MODEL, index=False)
    logger.info("All models results DataFrame:")
    print(all_models_results_df.head())
    #     or_tools_row[key] = float("nan")

    # new_df = pd.concat(
    #     [new_df, pd.DataFrame([or_tools_row])],
    #     ignore_index=True,
    # )

    # Save updated results
    new_df.to_csv(RESULTS_FILE, index=False)
    print("Results saved to", RESULTS_FILE)
    logger.info("Results DataFrame:")
    print(new_df.head())

    all_models_results_df.to_csv(RESULTS_FILE_ALL_MODEL, index=False)
    logger.info("All models results DataFrame:")
    print(all_models_results_df.head())
