import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from rich import print
import pandas as pd
import yaml


from model import CVRPActor
from problem import CVRP
from sa import sa
from utils import plot_CVRP
from or_tools import test_or_tools

from HP import _HP, get_script_arguments


cfg = _HP("src/test.yaml")
cfg.update(get_script_arguments(cfg.keys()))

# Check if the results file exists, if not create it
results_file = "src/res_tmp.csv"
if not os.path.exists(results_file):
    # Create a DataFrame with the necessary columns
    df = pd.DataFrame(
        columns=[
            "Model",
            "Type",
            "Train Init Temp",
            "Train Steps",
            "Train Dim",
            "Demand",
            "Dimension",
            "Step",
            "Initial Temp",
            "Initial Cost",
            "Final Cost",
            "Gain",
        ]
    )
else:
    # Load the existing results file
    df = pd.read_csv(results_file)


# Function to set the seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Function to extract the loss value from a filename
def extract_loss(filename):
    try:
        # Identify and extract the "_loss_X" portion
        return float((filename.split("_")[-1])[:-3])
    except ValueError:
        pass
    return float("inf")  # Return a large number if extraction fails


# Function to load the model weights from wandb
def load_model(model, folder):
    # List all files in the directory
    files = [f for f in os.listdir(folder) if f.endswith(".pt")]
    if files:
        # Find the file with the smallest loss
        file_with_min_loss = min(files, key=extract_loss)
    model.load_state_dict(torch.load(folder + file_with_min_loss, weights_only=True))
    print(f"Loaded model from {file_with_min_loss}")
    with open(folder + "HP.yaml", "r") as f:
        config = yaml.unsafe_load(f)
        config.update(get_script_arguments(config.keys()))
    return model, config


def main():
    cfg = _HP("src/test.yaml")
    cfg.update(get_script_arguments(cfg.keys()))
    # Initialize the actor model
    actor = CVRPActor(
        cfg["EMBEDDING_DIM"],
        cfg["C1"],
        cfg["C2"],
    )
    actor, config = load_model(actor, cfg["MODEL_DIR"])
    demands = config["DEMANDS"]
    train_init_temp = config["INIT_TEMP"]
    train_steps = config["OUTER_STEPS"]
    train_dim = config["PROBLEM_DIM"]
    cfg = _HP("src/test.yaml")
    cfg.update(get_script_arguments(cfg.keys()))
    problem = CVRP(
        cfg["VISU_DIM"], 256, cfg["MAX_LOAD"], cfg["DEMANDS"], device=cfg["DEVICE"]
    )
    params = problem.generate_params()
    params = {k: v.to(cfg["DEVICE"]) for k, v in params.items()}
    problem.set_params(params)
    # Find initial solutions
    init_x = problem.generate_init_x()
    init_cost = problem.cost(init_x)
    print(f"Cost of the initial solution: {torch.mean(init_cost).item()}")

    cfg["OUTER_STEPS"] = cfg["VISU_STEPS"]
    alpha = np.log(cfg["STOP_TEMP"]) - np.log(cfg["INIT_TEMP"])
    cfg["ALPHA"] = np.exp(alpha / cfg["OUTER_STEPS"]).item()
    with torch.no_grad():
        actor.eval()
        train = sa(
            actor,
            problem,
            init_x,
            cfg,
            replay=None,
            baseline=False,
            greedy=False,
            record_state=False,
        )
    min_cost_train = train["min_cost"]
    print(
        f"Cost of the trained solution ({cfg["VISU_STEPS"]} steps): {torch.mean(min_cost_train).item()}"
    )
    df.loc[len(df)] = [
        cfg["MODEL_DIR"],
        "Trained",
        train_init_temp,
        train_steps,
        train_dim,
        demands,
        cfg["VISU_DIM"],
        cfg["VISU_STEPS"],
        cfg["INIT_TEMP"],
        torch.mean(init_cost).item(),
        torch.mean(min_cost_train).item(),
        torch.mean(init_cost).item() - torch.mean(min_cost_train).item(),
    ]

    with torch.no_grad():
        actor.eval()
        greedy = sa(
            actor,
            problem,
            init_x,
            cfg,
            replay=None,
            baseline=False,
            greedy=True,
            record_state=False,
        )
    min_cost_greedy = greedy["min_cost"]
    print(
        f"Cost of the greedy solution ({cfg["VISU_STEPS"]} steps): {torch.mean(min_cost_greedy).item()}"
    )
    df.loc[len(df)] = [
        cfg["MODEL_DIR"],
        "Greedy",
        train_init_temp,
        train_steps,
        train_dim,
        demands,
        cfg["VISU_DIM"],
        cfg["VISU_STEPS"],
        cfg["INIT_TEMP"],
        torch.mean(init_cost).item(),
        torch.mean(min_cost_greedy).item(),
        torch.mean(init_cost).item() - torch.mean(min_cost_greedy).item(),
    ]

    with torch.no_grad():
        actor.eval()
        baseline = sa(
            actor,
            problem,
            init_x,
            cfg,
            replay=None,
            baseline=True,
            greedy=False,
            record_state=True,
        )
    min_cost_baseline_short = baseline["min_cost"]
    print(
        f"Cost of the baseline solution ({cfg["VISU_STEPS"]} steps): {torch.mean(min_cost_baseline_short).item()}"
    )
    df.loc[len(df)] = [
        cfg["MODEL_DIR"],
        "Baseline",
        train_init_temp,
        train_steps,
        train_dim,
        demands,
        cfg["VISU_DIM"],
        cfg["VISU_STEPS"],
        cfg["INIT_TEMP"],
        torch.mean(init_cost).item(),
        torch.mean(min_cost_baseline_short).item(),
        torch.mean(init_cost).item() - torch.mean(min_cost_baseline_short).item(),
    ]

    cfg["OUTER_STEPS"] = 10 * (cfg["VISU_DIM"] ** 2)
    alpha = np.log(cfg["STOP_TEMP"]) - np.log(cfg["INIT_TEMP"])
    cfg["ALPHA"] = np.exp(alpha / cfg["OUTER_STEPS"]).item()

    with torch.no_grad():
        actor.eval()
        big_train = sa(
            actor,
            problem,
            init_x,
            cfg,
            replay=None,
            baseline=False,
            greedy=False,
            record_state=False,
        )
    min_cost_big_train = big_train["min_cost"]
    print(
        f"Cost of the trained solution ({10 * (cfg["VISU_DIM"] ** 2)} steps): {torch.mean(min_cost_big_train).item()}"
    )
    df.loc[len(df)] = [
        cfg["MODEL_DIR"],
        "Trained",
        train_init_temp,
        train_steps,
        train_dim,
        demands,
        cfg["VISU_DIM"],
        10 * (cfg["VISU_DIM"] ** 2),
        cfg["INIT_TEMP"],
        torch.mean(init_cost).item(),
        torch.mean(min_cost_big_train).item(),
        torch.mean(init_cost).item() - torch.mean(min_cost_big_train).item(),
    ]

    with torch.no_grad():
        actor.eval()
        greedy = sa(
            actor,
            problem,
            init_x,
            cfg,
            replay=None,
            baseline=False,
            greedy=True,
            record_state=False,
        )
    min_cost_greedy_big = greedy["min_cost"]
    print(
        f"Cost of the big greedy solution ({10 * (cfg["VISU_DIM"] ** 2)} steps): {torch.mean(min_cost_greedy_big).item()}"
    )
    df.loc[len(df)] = [
        cfg["MODEL_DIR"],
        "Greedy",
        train_init_temp,
        train_steps,
        train_dim,
        demands,
        cfg["VISU_DIM"],
        10 * (cfg["VISU_DIM"] ** 2),
        cfg["INIT_TEMP"],
        torch.mean(init_cost).item(),
        torch.mean(min_cost_greedy_big).item(),
        torch.mean(init_cost).item() - torch.mean(min_cost_greedy_big).item(),
    ]

    with torch.no_grad():
        actor.eval()
        baseline = sa(
            actor,
            problem,
            init_x,
            cfg,
            replay=None,
            baseline=True,
            greedy=False,
            record_state=False,
        )
    min_cost_baseline_big = baseline["min_cost"]
    print(
        f"Cost of the big baseline solution ({10 * (cfg["VISU_DIM"] ** 2)} steps): {torch.mean(min_cost_baseline_big).item()}"
    )
    df.loc[len(df)] = [
        cfg["MODEL_DIR"],
        "Baseline",
        train_init_temp,
        train_steps,
        train_dim,
        demands,
        cfg["VISU_DIM"],
        10 * (cfg["VISU_DIM"] ** 2),
        cfg["INIT_TEMP"],
        torch.mean(init_cost).item(),
        torch.mean(min_cost_baseline_big).item(),
        torch.mean(init_cost).item() - torch.mean(min_cost_baseline_big).item(),
    ]

    solution_or_tools = test_or_tools(params, cfg)
    cost_or_tools = problem.cost(solution_or_tools)
    print(
        f"Cost of the OR-Tools solution ({cfg["OR_TOOLS_TIME"]} sec): {torch.mean(cost_or_tools).item()}"
    )
    df.loc[len(df)] = [
        cfg["MODEL_DIR"],
        "OR-Tools",
        train_init_temp,
        train_steps,
        train_dim,
        demands,
        cfg["VISU_DIM"],
        cfg["OR_TOOLS_TIME"],
        cfg["INIT_TEMP"],
        torch.mean(init_cost).item(),
        torch.mean(cost_or_tools).item(),
        torch.mean(init_cost).item() - torch.mean(cost_or_tools).item(),
    ]

    # Save the DataFrame to a CSV file
    df.to_csv(results_file, index=False)
    print(f"Saved results file at {results_file}")


if __name__ == "__main__":
    main()
