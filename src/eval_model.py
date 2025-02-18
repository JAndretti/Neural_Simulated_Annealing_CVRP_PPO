import pandas as pd
import os
import torch
import random
import numpy as np

from problem import CVRP
from model import CVRPActor
from sa import sa
from tqdm import tqdm

df = pd.read_csv("src/res.csv")

# Check if the results file exists, if not create it
results_file = "src/res_model.csv"
if not os.path.exists(results_file):
    # Create a DataFrame with the necessary columns
    new_df = pd.DataFrame(
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
    new_df = pd.read_csv(results_file)


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
    # print(f"Loaded model from {file_with_min_loss}")
    return model


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


already_tested = []

# Count the number of unique models to be tested
unique_models = df["Model"].nunique()
print(f"Number of unique models to be tested: {unique_models}")
tested = 0
for row in tqdm(df.iterrows(), total=len(df)):
    row = row[1]
    if row["Model"] in already_tested or row["Type"] != "Trained":
        continue
    already_tested.append(row["Model"])
    actor = CVRPActor(
        cfg["EMBEDDING_DIM"],
        cfg["C1"],
        cfg["C2"],
    )
    actor = load_model(actor, row["Model"].replace("HP.yaml", ""))
    train_path = row["Model"]
    train_init_temp = row["Train Init Temp"]
    train_steps = row["Train Steps"]
    train_dimension = row["Train Dimension"]
    train_demand = row["Train Demand"]
    train_scheduler = row["Train Scheduler"]
    train_clustering = row["Train Clustering"]
    for dim, load in zip([20, 50, 100], [30, 40, 50]):
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
            for init_temp in [1, 100, 1000]:
                for step in [100, 1000, 10 * (dim**2)]:
                    for scheduler in ["cyclic", "lam", "step"]:
                        cfg["SCHEDULER"] = scheduler
                        cfg["OUTER_STEPS"] = step
                        cfg["INIT_TEMP"] = init_temp

                        test = sa(
                            actor,
                            problem,
                            init_x,
                            cfg,
                            replay=None,
                            baseline=False,
                            greedy=False,
                        )
                        min_cost_train = test["min_cost"]
                        # print(
                        #    f"Cost of the trained solution ({step} steps, {init_temp} "
                        #   f"temp, {dim} pb size): {torch.mean(min_cost_train).item()}"
                        # )
                        new_df.loc[len(new_df)] = [
                            train_path,
                            "Trained",
                            train_init_temp,
                            train_steps,
                            train_dimension,
                            train_demand,
                            train_scheduler,
                            train_clustering,
                            dim,
                            step,
                            scheduler,
                            clustering,
                            init_temp,
                            torch.mean(init_cost).item(),
                            torch.mean(min_cost_train).item(),
                            torch.mean(init_cost).item()
                            - torch.mean(min_cost_train).item(),
                        ]
    tested += 1
    tqdm.write(f"Tested {tested}/{unique_models} models")
# Save the DataFrame to a CSV file
new_df.to_csv(results_file, index=False)
print(f"Saved results file at {results_file}")
