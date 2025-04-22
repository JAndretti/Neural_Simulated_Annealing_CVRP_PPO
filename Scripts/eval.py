import pandas as pd
import torch
import os
import warnings
import glob
import sys

from func import get_HP_for_model, set_seed, load_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from problem import CVRP
from sa import sa
from model import CVRPActorPairs, CVRPActor
from tqdm import tqdm

# Suppress warnings if needed
warnings.filterwarnings("ignore")

# Constants
PATH = "wandb/Neural_Simulated_Annealing/"
FOLDER = "init_temp_outer_steps_benchmark"
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
    "name": None,  # unused but mandatory
}


def init_problem_parameters(model_path: str):
    HP = get_HP_for_model(model_path)
    CFG = {
        "PROBLEM_DIM": HP.get("PROBLEM_DIM", 20),
        "N_PROBLEMS": 500,
        "MAX_LOAD": HP.get("MAX_LOAD", 30),
        "DEVICE": HP.get("DEVICE", "cpu"),
        "INNER_STEPS": HP.get("INNER_STEPS", 1),
        "OUTER_STEPS": 1000,
        "SCHEDULER": HP.get("SCHEDULER", "lam"),
        "HEURISTIC": HP.get("HEURISTIC", "mix"),
        "INIT_TEMP": HP.get("INIT_TEMP", 1.0),
        "STOP_TEMP": HP.get("STOP_TEMP", 0.1),
        "METHOD": HP.get("METHOD", "ppo"),
        "REWARD": HP.get("REWARD", "immediate"),
        "PAIRS": HP.get("PAIRS", False),
        "EMBEDDING_DIM": HP.get("EMBEDDING_DIM", 32),
        "NUM_H_LAYERS": HP.get("NUM_H_LAYERS", 1),
        "GAMMA": HP.get("GAMMA", 0.99),
        "name": None,  # unused but mandatory
        "CLUSTERING": False,
    }
    return CFG


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


def init_pb():
    """Initialize the CVRP problem and generate initial solutions."""
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
    HP = init_problem_parameters(model_name)

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
