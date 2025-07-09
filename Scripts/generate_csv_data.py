import os
import sys
import glob2
import vrplib
import pandas as pd

# Add src directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from or_tools import or_tools


def get_opt_cost(sol_file):
    with open(sol_file, "r") as f:
        for line in f:
            if line.startswith("Cost"):
                return int(line.strip().split()[1])
    return None


if __name__ == "__main__":
    path_opt_cost = "bdd/Vrp-Set-XML100/solutions/"
    instances = glob2.glob("bdd/Vrp-Set-XML100/instances/*.vrp")

    names = []
    dimensions = []
    capacities = []
    node_coords = []
    demands = []
    depots = []
    opt_costs = []

    for instance_path in instances:
        data = vrplib.read_instance(instance_path)
        name = data["name"]
        names.append(name)
        sol_file = os.path.join(path_opt_cost, f"{name}.sol")
        opt_costs.append(get_opt_cost(sol_file))
        dimensions.append(data["dimension"])
        capacities.append(data["capacity"])
        node_coords.append(data["node_coord"])
        demands.append(data["demand"])
        depots.append(data["depot"])

    cfg = {
        "OR_TOOLS_TIME": 60,  # Time limit in seconds
    }

    params = {
        "coords": node_coords,
        "demands": demands,
        "MAX_LOAD": capacities,
        "OR_DIM": dimensions,
    }

    solutions, distances = or_tools(params, cfg)
    # Divide all or_tools distances by 100 to match original scale
    # distances = [d / 100 if d is not None else None for d in distances]
    df = pd.DataFrame(
        {
            "name": names,
            "opt_sol": opt_costs,
            "or_tools_cost": distances,
        }
    )
    output_path = "bdd/Vrp-Set-XML100_res.csv"
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        print(
            f"Warning: Output directory '{output_dir}' does not exist. ",
            "Saving to current directory.",
        )
        output_path = "Vrp-Set-XML100_res.csv"
    df.to_csv(output_path, index=False)

    print("Moyenne or_tools_cost:", df["or_tools_cost"].mean())
    print("Moyenne opt_sol:", df["opt_sol"].mean())
    print(
        "Nombre de fois or_tools_cost < opt_sol:",
        (df["or_tools_cost"] < df["opt_sol"]).sum(),
    )
