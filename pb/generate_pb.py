import torch
import sys
import os
from loguru import logger


# --------------------------------
# Configure logger formatting
# --------------------------------
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


def setup_imports():
    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    )
    try:
        from problem import CVRP
        from or_tools import or_tools

        return CVRP, or_tools
    except ImportError as e:
        logger.error(f"Failed to import modules: {e}")
        sys.exit(1)


def get_config():
    return {
        "PROBLEM_DIM": 100,
        "N_PROBLEMS": 1000,
        "MAX_LOAD": 50,
        "CLUSTERING": True,
        "NB_CLUSTERS_MAX": 5,
        "DEVICE": "cpu",
        "HEURISTIC": "mix",  # unused but mandatory, model value will be used
        "MIX1": "swap",  # unused but mandatory, model value will be used
        "MIX2": "insertion",  # unused but mandatory, model value will be used
        "name": None,  # unused but mandatory
        "OR_TOOLS_TIME": 60,  # Time limit in seconds for OR-Tools per problem
        "INIT": "sweep",
    }


def initialize_problem(cfg, CVRP):
    logger.info("Initializing CVRP problem...")
    problem = CVRP(
        cfg["PROBLEM_DIM"],
        cfg["N_PROBLEMS"],
        cfg["MAX_LOAD"],
        device=cfg["DEVICE"],
        params=cfg,
    )
    problem.manual_seed(0)
    logger.info("CVRP problem initialized and seed set.")
    return problem


def generate_problem_instances(problem, cfg):
    logger.info("Generating new problem instances...")
    params = problem.generate_params()
    params = {k: v.to(cfg["DEVICE"]) for k, v in params.items()}
    problem.set_params(params)
    logger.info("Problem instances generated and parameters set.")
    return params


def solve_with_or_tools(params, cfg, or_tools):
    logger.info("Solving with OR-Tools...")
    cfg_for_or_tools = cfg.copy()
    cfg_for_or_tools["OR_DIM"] = cfg["PROBLEM_DIM"]
    or_solutions, _ = or_tools(params, cfg_for_or_tools)
    logger.info("OR-Tools solving complete.")
    return or_solutions


def process_or_solutions(or_solutions, problem, cfg):
    valid_or_solutions = [sol for sol in or_solutions if sol is not None]
    if not valid_or_solutions:
        logger.warning("No valid solutions found by OR-Tools.")
        return torch.tensor([]), torch.tensor(float("nan"))

    or_solutions_tensor = torch.tensor(valid_or_solutions, device=cfg["DEVICE"])

    if or_solutions_tensor.ndim == 2:
        or_solutions_tensor = or_solutions_tensor.unsqueeze(-1)
    elif or_solutions_tensor.ndim == 1 and cfg["N_PROBLEMS"] == 1:
        or_solutions_tensor = or_solutions_tensor.unsqueeze(0).unsqueeze(-1)

    if or_solutions_tensor.numel() > 0:
        or_distances = problem.cost(or_solutions_tensor)
        mean_or_distances = torch.mean(or_distances.float())
        logger.info(
            f"OR-Tools distances calculated. Mean distance: {mean_or_distances.item()}"
        )
        return or_distances, mean_or_distances
    else:
        logger.warning(
            "OR-Tools solutions tensor is empty after filtering. "
            "Cannot calculate distances."
        )
        return torch.tensor([]), torch.tensor(float("nan"))


def save_data(
    problem, params, init_x, or_solutions, or_distances, mean_or_distances, cfg
):
    coords = params["coords"]
    demands = params["demands"]

    data_to_save = {
        "init_x": init_x,
        "coords": coords,
        "demands": demands,
    }
    problem_dim = cfg["PROBLEM_DIM"]
    max_load = cfg["MAX_LOAD"]
    clustering = cfg["CLUSTERING"]
    filename = f"problem_data_dim{problem_dim}_load{max_load}_clustering{clustering}.pt"
    save_path = os.path.join(os.path.dirname(__file__), filename)
    torch.save(data_to_save, save_path)
    logger.info(f"Problem data saved to {save_path}")

    or_tools_data_to_save = {
        "or_solutions": or_solutions,
        "or_distances": or_distances,
        "mean_or_distances": mean_or_distances,
    }
    or_tools_filename = (
        f"problem_data_dim{problem_dim}_load{max_load}"
        f"_clustering{clustering}_solution_or_tools.pt"
    )
    or_tools_save_path = os.path.join(os.path.dirname(__file__), or_tools_filename)
    torch.save(or_tools_data_to_save, or_tools_save_path)
    logger.info(f"OR-Tools solution data saved to {or_tools_save_path}")


def main():
    # Setup imports
    CVRP, or_tools = setup_imports()

    # Get configuration
    cfg = get_config()
    logger.info(f"Configuration loaded: {cfg}")

    # Initialize problem
    problem = initialize_problem(cfg, CVRP)

    # Generate problem instances
    params = generate_problem_instances(problem, cfg)

    # Solve with OR-Tools
    or_solutions = solve_with_or_tools(params, cfg, or_tools)

    # Process OR-Tools solutions
    or_distances, mean_or_distances = process_or_solutions(or_solutions, problem, cfg)

    # Generate initial solutions
    logger.info("Generating initial solutions...")
    init_x = problem.generate_init_x()
    initial_cost = torch.mean(problem.cost(init_x))
    logger.info(
        f"Initial solutions generated. Mean initial cost: {initial_cost.item()}"
    )

    # Save data
    save_data(
        problem, params, init_x, or_solutions, or_distances, mean_or_distances, cfg
    )

    logger.info("Script finished successfully.")


if __name__ == "__main__":
    main()
