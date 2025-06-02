import torch
import os
from loguru import logger
import sys

# --------------------------------
# Configure logger formatting
# --------------------------------
logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<blue>{file}:{line}</blue> | "
        "<yellow>{message}</yellow>"
    ),
    colorize=True,
)


def setup_imports():
    """Add src directory to path to allow importing project modules."""
    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    )


def merge_problem_data_files():
    """Merge problem data files (without OR-Tools solutions)."""
    logger.info("Merging problem data files...")

    # Define file paths
    base_path = os.path.dirname(__file__)
    file_true = os.path.join(base_path, "problem_data_dim100_load50_clusteringTrue.pt")
    file_false = os.path.join(
        base_path, "problem_data_dim100_load50_clusteringFalse.pt"
    )
    output_file = os.path.join(base_path, "problem_data_dim100_load50.pt")

    # Load data
    try:
        data_true = torch.load(file_true)
        logger.info(f"Loaded data from {file_true}")
        data_false = torch.load(file_false)
        logger.info(f"Loaded data from {file_false}")
    except Exception as e:
        logger.error(f"Error loading files: {e}")
        return False

    # Concatenate data
    merged_data = {
        "coords": torch.cat([data_true["coords"], data_false["coords"]], dim=0),
        "demands": torch.cat([data_true["demands"], data_false["demands"]], dim=0),
        "init_x": torch.cat([data_true["init_x"], data_false["init_x"]], dim=0),
    }

    logger.info(
        f"Merged data shapes: coords={merged_data['coords'].shape}, "
        f"demands={merged_data['demands'].shape}, init_x={merged_data['init_x'].shape}"
    )

    # Save merged data
    try:
        torch.save(merged_data, output_file)
        logger.success(f"Successfully saved merged problem data to {output_file}")
    except Exception as e:
        logger.error(f"Error saving merged problem data: {e}")
        return False

    return True


def merge_or_tools_files():
    """Merge OR-Tools solution files."""
    logger.info("Merging OR-Tools solution files...")

    # Define file paths
    base_path = os.path.dirname(__file__)
    file_true = os.path.join(
        base_path, "problem_data_dim100_load50_clusteringTrue_solution_or_tools.pt"
    )
    file_false = os.path.join(
        base_path, "problem_data_dim100_load50_clusteringFalse_solution_or_tools.pt"
    )
    output_file = os.path.join(
        base_path, "problem_data_dim100_load50_solution_or_tools.pt"
    )

    # Load data
    try:
        data_true = torch.load(file_true)
        logger.info(f"Loaded OR-Tools data from {file_true}")
        data_false = torch.load(file_false)
        logger.info(f"Loaded OR-Tools data from {file_false}")
    except Exception as e:
        logger.error(f"Error loading OR-Tools files: {e}")
        return False

    # Handle or_solutions (list of tensors)
    # Combine the solution lists
    if isinstance(data_true["or_solutions"], list) and isinstance(
        data_false["or_solutions"], list
    ):
        merged_or_solutions = data_true["or_solutions"] + data_false["or_solutions"]
    else:
        logger.warning(
            "or_solutions format unexpected, attempting to concatenate directly"
        )
        try:
            merged_or_solutions = torch.cat(
                [data_true["or_solutions"], data_false["or_solutions"]], dim=0
            )
        except Exception as e:
            logger.error(f"Failed to merge or_solutions: {e}")
            merged_or_solutions = []

    # Merge distance tensors
    merged_or_distances = torch.cat(
        [data_true["or_distances"], data_false["or_distances"]], dim=0
    )

    # Calculate new mean
    total_elements = len(merged_or_distances)
    merged_mean = (
        torch.mean(merged_or_distances)
        if total_elements > 0
        else torch.tensor(float("nan"))
    )

    # Create merged data structure
    merged_data = {
        "or_solutions": merged_or_solutions,
        "or_distances": merged_or_distances,
        "mean_or_distances": merged_mean,
    }

    logger.info(
        f"Merged OR-Tools data: {len(merged_or_solutions)} solutions, "
        f"distances shape={merged_or_distances.shape}, mean={merged_mean.item()}"
    )

    # Save merged data
    try:
        torch.save(merged_data, output_file)
        logger.success(f"Successfully saved merged OR-Tools data to {output_file}")
    except Exception as e:
        logger.error(f"Error saving merged OR-Tools data: {e}")
        return False

    return True


def main():
    """Main function to execute the merge operations."""
    setup_imports()

    # Merge problem data files
    problem_success = merge_problem_data_files()

    # Merge OR-Tools solution files
    or_tools_success = merge_or_tools_files()

    if problem_success and or_tools_success:
        logger.success("All files merged successfully!")
    else:
        logger.warning("Some merges failed. See logs above.")


if __name__ == "__main__":
    main()
