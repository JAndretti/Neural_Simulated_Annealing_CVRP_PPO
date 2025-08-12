import os
import sys
import glob2
import vrplib
import pandas as pd
import tqdm

# Add src directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from or_tools import or_tools

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
cfg = {
    "OR_TOOLS_TIME": 1,  # Time limit in seconds
}

if __name__ == "__main__":
    logger.info("Starting the VRP processing script.")
    vrp_dir = os.listdir("generated_problem/bdd_generated/")
    logger.info(f"Found {len(vrp_dir)} directories in 'bdd_generated'.")

    for dir in tqdm.tqdm(vrp_dir, leave=True, desc="Processing directories"):
        number = int(dir.split("gen")[1])
        instances = glob2.glob(f"generated_problem/bdd_generated/{dir}/*.vrp")
        logger.info(f"Found {len(instances)} instances in directory '{dir}'.")

        names = []
        dimensions = []
        capacities = []
        node_coords = []
        demands = []
        depots = []

        for instance_path in tqdm.tqdm(instances, leave=False, desc="Getting data"):
            data = vrplib.read_instance(instance_path)
            name = data["name"]
            names.append(name)
            dimensions.append(data["dimension"])
            capacities.append(data["capacity"])
            node_coords.append(data["node_coord"])
            demands.append(data["demand"])
            depots.append(data["depot"])

        logger.info("Configuration for OR-Tools set.")

        params = {
            "coords": node_coords,
            "demands": demands,
            "MAX_LOAD": capacities,
            "OR_DIM": dimensions,
        }
        logger.info("Parameters for OR-Tools prepared.")

        logger.info("Running OR-Tools solver...")
        solutions, distances = or_tools(params, cfg, mult_thread=True)
        logger.info("OR-Tools solver completed.")

        df = pd.DataFrame(
            {
                "name": names,
                "or_tools_cost": distances,
            }
        )
        logger.info("Dataframe created with solution results.")

        output_path = (
            f"generated_problem/or_tools_gen{number}_{cfg['OR_TOOLS_TIME']}.csv"
        )
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            logger.warning(
                f"Output directory '{output_dir}' does not exist. "
                "Saving to current directory."
            )
            output_path = f"or_tools_gen{number}.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to '{output_path}'.")
