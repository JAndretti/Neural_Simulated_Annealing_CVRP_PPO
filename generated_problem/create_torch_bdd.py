import os
import glob2
import tqdm
import vrplib

import torch
import numpy as np

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


PATH = "generated_problem/bdd_generated/"
dir = os.listdir(PATH)
number = [int(d.split("gen")[1]) for d in dir if "gen" in d]

logger.info(f"Found {len(number)} generated problems to process.")

for num in tqdm.tqdm(number, leave=True, desc="Processing generated problems"):
    logger.info(f"Processing problem gen{num}...")
    dimensions = []
    capacities = []
    node_coords = []
    demands = []
    vrp_file = glob2.glob(PATH + "gen" + str(num) + "/*.vrp")
    logger.info(f"Found {len(vrp_file)} VRP files for gen{num}.")

    for file in tqdm.tqdm(vrp_file, leave=False, desc=f"Processing gen{num}"):
        data = vrplib.read_instance(file)
        capacities.append(data["capacity"])
        node_coords.append(data["node_coord"])
        demands.append(data["demand"])

    logger.info(f"Converting data to tensors for gen{num}...")
    capacity_tensor = torch.tensor(capacities).unsqueeze(-1)
    node_coords_tensor = torch.tensor(np.array(node_coords))
    demands_tensor = torch.tensor(np.array(demands))

    # Normalize node_coords_tensor between 0 and 1 for each row
    logger.info("Normalizing node coordinates...")
    min_coords = torch.min(node_coords_tensor, dim=1, keepdim=True).values
    max_coords = torch.max(node_coords_tensor, dim=1, keepdim=True).values
    node_coords_tensor_n = (node_coords_tensor - min_coords) / (max_coords - min_coords)
    logger.info("Node coordinates normalized.")

    output_path = "generated_problem/" + f"gen{num}.pt"
    logger.info(f"Saving tensors to {output_path}...")
    torch.save(
        {
            "capacity": capacity_tensor,
            "node_coords": node_coords_tensor_n,
            "demands": demands_tensor,
        },
        output_path,
    )
    logger.info(f"Finished processing gen{num}.")
