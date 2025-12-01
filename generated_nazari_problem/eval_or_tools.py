import torch
import pandas as pd
import os
import sys

# Add src directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from algo import or_tools
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OR-Tools on generated Nazari problem."
    )
    parser.add_argument(
        "--dim", type=int, default=50, help="Dimension of the problem (default: 50)"
    )
    parser.add_argument(
        "--time",
        type=float,
        default=1.0,
        help="Time limit for OR-Tools in seconds (default: 1)",
    )

    args = parser.parse_args()

    DIM = args.dim
    cfg = {
        "OR_TOOLS_TIME": args.time,
    }

    PATH = f"generated_nazari_problem/gen_nazari_{DIM}.pt"

    bdd = torch.load(PATH, map_location="cpu")
    coords = bdd["node_coords"]
    demands = bdd["demands"]
    capacities = bdd["capacity"]

    params = {
        "coords": coords,
        "demands": demands,
        "MAX_LOAD": capacities,
        "OR_DIM": torch.full((len(coords),), DIM, dtype=torch.int),
        "names": [f"nazari_{DIM}"] * len(coords),
    }
    solutions, distances, names = or_tools(params, cfg, mult_thread=True)

    # Prepare data for CSV
    data = {
        "Distance": [distance / 100 for distance in distances],
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Define the output file name
    output_file = (
        f"generated_nazari_problem/results_dim{DIM}_time{cfg['OR_TOOLS_TIME']}.csv"
    )

    # Save to CSV
    df.to_csv(output_file, index=False)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
