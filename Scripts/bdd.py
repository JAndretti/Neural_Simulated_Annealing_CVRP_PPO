import vrplib
import pandas as pd
import glob
from tqdm import tqdm
import numpy as np

df = pd.DataFrame(
    columns=[
        "path",
        "name",
        "dimension",
        "capacity",
        "node_coord",
        "demand",
        "depot",
        "solution",
        "cost",
    ]
)

paths = glob.glob("bdd/**/*.vrp", recursive=True)
error = 0
data_list = []
for path in tqdm(paths, desc="Processing VRP files"):
    try:
        instance = vrplib.read_instance(path)
        if "XML100_" in instance["name"]:
            sol_path = path.replace(
                f"instances/{instance["name"]}.vrp", f"solutions/{instance["name"]}.sol"
            )
        else:
            sol_path = path.replace(".vrp", ".sol")
        solution = vrplib.read_solution(sol_path)
        data_list.append(
            {
                "path": path,
                "name": instance["name"],
                "dimension": instance["dimension"],
                "capacity": instance["capacity"],
                "node_coord": instance["node_coord"],
                "demand": instance["demand"],
                "depot": instance["depot"],
                "solution": solution["routes"],
                "cost": solution["cost"],
            }
        )
    except Exception as e:
        print(f"Error on file {path}: {e}")
        error += 1
        continue
df = pd.concat([df, pd.DataFrame(data_list)], ignore_index=True)
print(error)


df["node_coord_normalized"] = df["node_coord"].apply(lambda x: x / np.max(x))
df["cost_normalized"] = df["cost"] / df["node_coord"].apply(lambda x: np.max(x))

df.to_pickle("bdd/bdd.pkl")
