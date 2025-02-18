import os
import pandas as pd
import random
import numpy as np
import torch

from problem import CVRP
from or_tools import test_or_tools
from model import CVRPActor
from sa import sa


# Check if the results file exists, if not create it
results_file = "src/res_baseline.csv"
if not os.path.exists(results_file):
    # Create a DataFrame with the necessary columns
    baseline_df = pd.DataFrame(
        columns=[
            "Model",
            "Type",
            "Train Init Temp",
            "Train Steps",
            "Train Dimension",
            "Train Demand",
            "Train Scheduler",
            "Train Clustering",
            "Dimension",
            "Step",
            "Scheduler",
            "Clustering",
            "Initial Temp",
            "Initial Cost",
            "Final Cost",
            "Gain",
        ]
    )
else:
    # Load the existing results file
    baseline_df = pd.read_csv(results_file)


# Function to set the seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


cfg = {
    "DEMANDS": True,
    "NB_CLUSTERS_MAX": 5,
    "N_PROBLEMS": 100,
    "DEVICE": "cpu",
    "EMBEDDING_DIM": 16,
    "C1": 10,
    "C2": 16,
    "OR_TOOLS_TIME": 1,
    "STOP_TEMP": 0.01,
    "INNER_STEPS": 1,
    "METHOD": "ppo",
    "REWARD": "immediate",
    "GAMMA": 0.9,
}

set_seed(0)

actor = CVRPActor(
    cfg["EMBEDDING_DIM"],
    cfg["C1"],
    cfg["C2"],
)
for dim, load in zip([100], [50]):
    cfg["OR_DIM"] = dim
    cfg["MAX_LOAD"] = load
    for clustering in [True, False]:
        cfg["CLUSTERING"] = clustering
        problem = CVRP(
            dim,
            100,
            load,
            device=cfg["DEVICE"],
            params=cfg,
        )
        params = problem.generate_params(mode="test")
        params = {k: v.to(cfg["DEVICE"]) for k, v in params.items()}
        problem.set_params(params)
        # Find initial solutions
        init_x = problem.generate_init_x()
        init_cost = problem.cost(init_x)
        solution_or_tools = test_or_tools(params, cfg)
        if solution_or_tools is not None:
            cost_or_tools = problem.cost(solution_or_tools)
        else:
            cost_or_tools = torch.tensor([np.nan])
        baseline_df.loc[len(baseline_df)] = [
            "/",
            f"OR_TOOLS {cfg["OR_TOOLS_TIME"]} sec",
            "/",
            "/",
            "/",
            "/",
            "/",
            "/",
            dim,
            "/",
            "/",
            clustering,
            "/",
            torch.mean(init_cost).item(),
            torch.mean(cost_or_tools).item(),
            torch.mean(init_cost).item() - torch.mean(cost_or_tools).item(),
        ]
        print(
            f"Cost of OR_Tools ({cfg["OR_TOOLS_TIME"]} sec, "
            f"{dim} pb size): {torch.mean(cost_or_tools).item()}"
        )
        for init_temp in [1, 100, 1000]:
            for step in [100, 1000, 10 * (dim**2)]:
                for scheduler in ["cyclic", "lam", "step"]:
                    cfg["SCHEDULER"] = scheduler
                    cfg["OUTER_STEPS"] = step
                    cfg["INIT_TEMP"] = init_temp
                    test_baseline = sa(
                        actor,
                        problem,
                        init_x,
                        cfg,
                        replay=None,
                        baseline=True,
                        greedy=False,
                    )
                    min_cost_baseline = test_baseline["min_cost"]
                    print(
                        f"Cost of the Baseline ({step} steps, {init_temp} temp, "
                        f"{dim} pb size): {torch.mean(min_cost_baseline).item()}"
                    )
                    baseline_df.loc[len(baseline_df)] = [
                        f"{dim}_{step}_{scheduler}_{clustering}_{init_temp}",
                        "Baseline",
                        "/",
                        "/",
                        "/",
                        "/",
                        "/",
                        "/",
                        dim,
                        step,
                        scheduler,
                        clustering,
                        init_temp,
                        torch.mean(init_cost).item(),
                        torch.mean(min_cost_baseline).item(),
                        torch.mean(init_cost).item()
                        - torch.mean(min_cost_baseline).item(),
                    ]
# Save the DataFrame to a CSV file
baseline_df.to_csv(results_file, index=False)
print(f"Saved results file at {results_file}")
