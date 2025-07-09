import pandas as pd
import torch
import os
import warnings
import glob2
import sys
import time  # Add import for time measurement

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
FOLDER = "res_model_Capacity_reward_swap"
RESULTS_FILE_ALL_MODEL = "res/models_res/res_all_model.csv"
RESULTS_FILE = PATH + FOLDER + "/res_model.csv"
MODEL_NAMES = glob2.glob(os.path.join(PATH, FOLDER, "models", "*"))

# Configuration
CFG = {
    "PROBLEM_DIM": 100,
    "N_PROBLEMS": 1000,
    "MAX_LOAD": 50,
    "CLUSTERING": False,
    "DEVICE": "cpu",
    "INNER_STEPS": 1,
    "OUTER_STEPS": 20000,  # 1000
    "SCHEDULER": "lam",  # unused but mandatory, model value will be used
    "HEURISTIC": "mix",  # unused but mandatory, model value will be used
    "name": None,  # unused but mandatory
    "MIX1": "swap",  # unused but mandatory, model value will be used
    "MIX2": "insertion",  # unused but mandatory, model value will be used
    "NB_CLUSTERS_MAX": 5,  # unused but mandatory, model value will be used
}


def init_problem_parameters(model_path: str):
    HP = get_HP_for_model(model_path)
    CFG = {
        "PROBLEM_DIM": 100,
        "N_PROBLEMS": 2000,
        "MAX_LOAD": 50,
        "LEARN_TEMP": HP.get("LEARN_TEMP", False),
        "DEVICE": HP.get("DEVICE", "cpu"),
        "INNER_STEPS": HP.get("INNER_STEPS", 1),
        "OUTER_STEPS": 4000,
        "SCHEDULER": HP.get("SCHEDULER", "lam"),
        # "INIT": HP.get("INIT", "sweep"),
        "INIT": "sweep",
        "HEURISTIC": HP.get("HEURISTIC", "mix"),
        "MIX1": HP.get("MIX1", "swap"),
        "MIX2": HP.get("MIX2", "insertion"),
        "INIT_TEMP": HP.get("INIT_TEMP", 1.0),
        "STOP_TEMP": HP.get("STOP_TEMP", 0.1),
        "METHOD": HP.get("METHOD", "ppo"),
        "REWARD": HP.get("REWARD", "immediate"),
        "PAIRS": HP.get("PAIRS", False),
        "ENTRY": HP.get("ENTRY", 13),
        "EMBEDDING_DIM": HP.get("EMBEDDING_DIM", 32),
        "NUM_H_LAYERS": HP.get("NUM_H_LAYERS", 1),
        "GAMMA": HP.get("GAMMA", 0.99),
        "name": None,  # unused but mandatory
        "CLUSTERING": False,
        "METROPOLIS": HP.get("METROPOLIS", True),
        "NEG_REWARD": HP.get("NEG_REWARD", 0.0),
        "CAPACITY_REWARD": HP.get("CAPACITY_REWARD", False),
        "NORMALIZE_REWARD": HP.get("NORMALIZE_REWARD", False),
        "CAPACITY_REWARD_FACTOR": HP.get("CAPACITY_REWARD_FACTOR", 1.0),
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
    path = f"pb/problem_data_dim{CFG.get('PROBLEM_DIM')}_load{CFG.get('MAX_LOAD')}.pt"

    init_x = problem.load_from_pt(path)
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
    problem.set_heuristic(HP["HEURISTIC"], HP["MIX1"], HP["MIX2"])

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
        )
    # Load model
    actor = load_model(actor, model_name, "actor")

    # Run simulated annealing with time measurement
    start_time = time.time()
    init_x = problem.generate_init_x(HP["INIT"])
    test = sa(
        actor,
        problem,
        init_x,
        HP,
        replay_buffer=None,
        baseline=False,
        greedy=False,
    )
    execution_time = time.time() - start_time
    init_cost = torch.mean(problem.cost(init_x))
    final_cost = torch.mean(test["min_cost"])
    test_baseline = sa(
        actor,
        problem,
        init_x,
        HP,
        replay_buffer=None,
        baseline=True,
        greedy=False,
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
    problem, init_x, initial_cost = init_pb()

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
            model_name, problem, init_x
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
                        "model": FOLDER + "/" + model_name.split("/")[-1],
                        "final_cost": final_cost.item(),
                    }
                ),
            ],
            ignore_index=True,
        )
    # Load OR-Tools solution mean distance and add to DataFrame
    or_tools_path = (
        f"pb/problem_data_dim{CFG.get('PROBLEM_DIM')}_load{CFG.get('MAX_LOAD')}"
        f"_solution_or_tools.pt"
    )
    if os.path.exists(or_tools_path):
        or_tools_data = torch.load(or_tools_path, map_location="cpu")
        mean_or_distances = or_tools_data.get("mean_or_distances", float("nan"))
    else:
        mean_or_distances = float("nan")

    # Prepare a row for OR-Tools
    or_tools_row = {
        "model": "or_tools",
        "initial_cost": float("nan"),
        "final_cost": mean_or_distances.item(),
        "execution_time": 60,
    }
    for key in differing_keys:
        or_tools_row[key] = float("nan")

    new_df = pd.concat(
        [new_df, pd.DataFrame([or_tools_row])],
        ignore_index=True,
    )
    # Save updated results
    new_df.to_csv(RESULTS_FILE, index=False)
    print("Results saved to", RESULTS_FILE)

    all_models_results_df.to_csv(RESULTS_FILE_ALL_MODEL, index=False)
    print("All models results saved to", RESULTS_FILE_ALL_MODEL)
