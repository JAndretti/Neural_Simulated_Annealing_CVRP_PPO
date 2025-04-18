import pandas as pd
import torch
import os
import numpy as np
import random
import warnings
import yaml
import glob

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from problem import CVRP
from sa import sa
from model import CVRPActorPairs, CVRPActor
from tqdm import tqdm

# Suppress warnings if needed
warnings.filterwarnings("ignore")

# Constants
PATH = "wandb/Neural_Simulated_Annealing/"
FOLDER = "init_temp_benchmark"
RESULTS_FILE = f"res/models_res/res_model_{FOLDER}.csv"
MODEL_NAMES = glob.glob(os.path.join(PATH, FOLDER, "models", "*"))


# Configuration
CFG = {
    "PROBLEM_DIM": 20,
    "N_PROBLEMS": 500,
    "MAX_LOAD": 30,
    "CLUSTERING": False,
    "DEVICE": "cpu",
    "INNER_STEPS": 1,
    "OUTER_STEPS": 100,
    "SCHEDULER": "lam",  # unused but mandatory, model value will be used
    "HEURISTIC": "mix",  # unused but mandatory, model value will be used
    "name": None,  # unused but mandatory, model value will be used
}


def get_HP_for_model(model_name):
    """Extract heuristic from HP.yaml file."""
    hp_file = os.path.join(model_name, "HP.yaml")
    try:
        with open(hp_file, "r") as file:
            content = file.read()
            content_clean = content.replace("!!python/object:HP._HP", "")
            hp_data = yaml.unsafe_load(content_clean)
            if isinstance(hp_data, dict):
                for key in hp_data.get("config", {}):
                    if key in CFG and key not in ["SCHEDULER", "HEURISTIC"]:
                        hp_data["config"][key] = CFG[key]
                return hp_data.get("config", {})
            else:
                return None
    except Exception as e:
        print(f"Error for reading HP.yaml file for mdoel : {model_name}: {e}")
        return None


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

    # init HP
    HP = get_HP_for_model(model_name)

    # Set heuristic in problem
    problem.set_heuristic(HP["HEURISTIC"])

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
            num_hidden_layers=HP["NUM_H_LAYERS"],
            device=HP["DEVICE"],
            mixed_heuristic=True if HP["HEURISTIC"] == "mix" else False,
        )
    # Load model
    actor = load_model(actor, model_name)

    # Run simulated annealing
    test = sa(
        actor,
        problem,
        init_x,
        HP,
        replay_buffer=None,
        baseline=False,
        greedy=False,
    )
    final_cost = torch.mean(test["min_cost"])
    return final_cost


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
            values = {hp.get(key) for hp in all_hp_data}
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
    set_seed()
    problem, init_x, init_cost = init_pb()

    # Extract keys with differing values across HP.yaml files
    differing_keys = extract_differing_keys(MODEL_NAMES)

    # Initialize results DataFrame with dynamic columns
    columns = ["model", "initial_cost", "final_cost"] + list(differing_keys)
    new_df, RESULTS_FILE = initialize_results_df(columns)

    for model_name in tqdm(MODEL_NAMES, desc="Processing models"):
        final_cost = perform_test(model_name, problem, init_x)
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
                        **hp_values,
                    }
                ),
            ],
            ignore_index=True,
        )
    # Save updated results
    new_df.to_csv(RESULTS_FILE, index=False)
    print("Results saved to", RESULTS_FILE)
