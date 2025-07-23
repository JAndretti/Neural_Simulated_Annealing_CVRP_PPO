import os
import sys
import glob2
import vrplib
import pandas as pd

# Add src directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from or_tools import or_tools
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


def get_opt_cost(sol_file):
    with open(sol_file, "r") as f:
        for line in f:
            if line.startswith("Cost"):
                return int(line.strip().split()[1])
    return None


if __name__ == "__main__":

    cfg = {
        "OR_TOOLS_TIME": 30,  # Time limit in seconds
    }

    path_opt_cost = "bdd/Vrp-Set-XML100/solutions/"
    instances = glob2.glob("bdd/Vrp-Set-XML100/instances/*.vrp")

    names = []
    dimensions = []
    capacities = []
    node_coords = []
    demands = []
    depots = []
    opt_costs = []

    for instance_path in tqdm(instances, desc="Getting data", leave=False):
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

    params = {
        "coords": node_coords,
        "demands": demands,
        "MAX_LOAD": capacities,
        "OR_DIM": dimensions,
    }
    logger.info("Starting OR-Tools solver...")
    solutions, distances = or_tools(params, cfg, mult_thread=True)
    df = pd.DataFrame(
        {
            "name": names,
            "opt_sol": opt_costs,
            "or_tools_cost": distances,
        }
    )
    output_path = f"res/Vrp-Set-XML100_res_{cfg['OR_TOOLS_TIME']}.csv"
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        logger.warning(
            f"Output directory '{output_dir}' does not exist. "
            "Saving to current directory."
        )
        output_path = f"Vrp-Set-XML100_res_{cfg['OR_TOOLS_TIME']}.csv"
    df.to_csv(output_path, index=False)

    logger.info(f"Results saved to {output_path}")
    logger.info(f"Average or_tools_cost: {df['or_tools_cost'].mean()}")
    logger.info(f"Average opt_sol: {df['opt_sol'].mean()}")
    logger.info(
        "Number of times or_tools_cost < opt_sol: "
        f"{(df['or_tools_cost'] < df['opt_sol']).sum()}"
    )
