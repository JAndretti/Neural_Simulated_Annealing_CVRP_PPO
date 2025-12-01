import torch
import random
import numpy as np
import math
import os


from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# from problem import CVRP

# from setup import _HP, get_script_arguments
from tqdm import tqdm
import multiprocessing

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


# Function to calculate Euclidean distance between two points
def distance(x1, y1, x2, y2):
    """Calculate the Euclidean distance between two points."""
    return np.linalg.norm([x2 - x1, y2 - y1])


# Function to calculate the distance matrix
def compute_euclidean_distance_matrix(locations):
    """Creates callback to return distance between points."""
    if all(0 <= coord <= 1 for loc in locations for coord in loc):
        MULT = 100
    else:
        MULT = 1
    distances = {}
    for from_counter, from_node in enumerate(locations):
        distances[from_counter] = {}
        for to_counter, to_node in enumerate(locations):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                # Euclidean distance
                distances[from_counter][to_counter] = int(
                    (
                        math.hypot(
                            (from_node[0] - to_node[0]), (from_node[1] - to_node[1])
                        )
                    )
                    * MULT
                )

    return distances


def create_data_model(coord, demand, max_load):
    """Stores the data for the problem."""
    data = {}
    # Locations in block units
    data["locations"] = coord
    data["demands"] = demand
    data["num_vehicles"] = 50
    data["vehicle_capacities"] = [max_load for _ in range(data["num_vehicles"])]
    data["depot"] = 0
    return data


def distance_solution(distance_matrix, solution):
    """Returns the distance of the solution."""
    distance = 0
    for i in range(len(solution) - 1):
        distance += distance_matrix[solution[i]][solution[i + 1]]
    return distance


def get_solution_nodes(data, manager, routing, solution, dim):
    """Returns a list of routes in the solution."""
    routes = []
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))
        # routes.append(route)
        if route != [0, 0]:  # Vérifie si la route ne contient pas uniquement des 0
            routes.append(route)
        flattened_list = []
        for sublist in routes:
            if flattened_list and sublist[0] == 0 and flattened_list[-1] == 0:
                flattened_list.extend(sublist[1:])  # Éviter d'ajouter un 0 doublon
            else:
                flattened_list.extend(sublist)
    while len(flattened_list) < dim + int(dim * 0.4):
        flattened_list.append(0)
    return flattened_list


# Function to set the seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def solve_w_ortools(coord, distance_matrix, demand, or_tools_time, max_load, dim):
    """Entry point of the program."""
    # Instantiate the data problem.
    data = create_data_model(coord, demand, max_load)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["locations"]), data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data["demands"][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data["vehicle_capacities"],  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity",
    )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    if or_tools_time > 0:
        seconds = int(or_tools_time)
        nanos = int((or_tools_time - seconds) * 1e9)
        search_parameters.time_limit.seconds = seconds
        search_parameters.time_limit.nanos = nanos

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        solution = get_solution_nodes(data, manager, routing, solution, dim)
        distance = distance_solution(distance_matrix, solution)
        return solution, distance
    return None, None


def solve_instance(args):
    """Helper function to solve a single instance, for parallel processing."""
    (
        coord_tuple,
        demand_tensor,
        cfg_or_tools_time,
        cfg_max_load,
        cfg_or_dim,
        instance_name,
    ) = args
    coord_list = [tuple(row) for row in coord_tuple.tolist()]
    dist_matrix = compute_euclidean_distance_matrix(coord_list)
    solution, distance = solve_w_ortools(
        coord_list,
        dist_matrix,
        demand_tensor.tolist(),
        cfg_or_tools_time,
        cfg_max_load,
        cfg_or_dim,
    )
    return solution, distance, instance_name


def or_tools(params, cfg, mult_thread=False):
    # to do in if __name__ == '__main__':

    # Prepare arguments for each call to solve_instance
    args_list = [
        (coord, demand, cfg["OR_TOOLS_TIME"], max_load, dimension, name)
        for coord, demand, max_load, dimension, name in zip(
            params["coords"],
            params["demands"],
            params["MAX_LOAD"],
            params["OR_DIM"],
            params["names"],
        )
    ]
    logger.info(f"Number of instances to solve: {len(args_list)}")
    num_cores = min(50, os.cpu_count(), len(args_list))
    logger.info(f"Number of cores used for parallel processing: {num_cores}")
    if mult_thread:
        with multiprocessing.Pool(processes=num_cores) as pool:
            results = list(
                tqdm(
                    pool.imap(solve_instance, args_list),
                    total=len(params["coords"]),
                    leave=False,
                    desc="Solving instances with OR-Tools",
                )
            )

        # Unzip results
        solutions, distances, names = zip(*results)
    else:
        solutions, distances, names = zip(*[solve_instance(args) for args in args_list])

    return list(solutions), list(distances), list(names)


# def main():
#     cfg = _HP("src/test.yaml")
#     cfg.update(get_script_arguments(cfg.keys()))

#     set_seed(cfg["SEED"])
#     problem = CVRP(cfg["OR_DIM"], cfg["N_PROBLEMS"], device="cpu")
#     params = problem.generate_params()
#     params = {k: v.to("cpu") for k, v in params.items()}
#     problem.set_params(params)
#     init_x = problem.generate_init_x()
#     solutions, distance = or_tools(params, cfg)
#     solutions = torch.tensor(solutions).unsqueeze(-1)
#     cost_init = problem.cost(init_x)
#     cost = problem.cost(solutions)
#     print((torch.mean(cost_init)).item())
#     print((torch.mean(cost)).item())


def test_or_tools(params, cfg):
    solutions, _, _ = or_tools(params, cfg)
    tensor = filter_and_convert_solutions(solutions)
    return tensor


def filter_and_convert_solutions(solutions):
    """Filter out None values and convert the list to a tensor."""
    filtered_solutions = [sol for sol in solutions if sol is not None]
    if not filtered_solutions:
        return None
    return torch.tensor(filtered_solutions).unsqueeze(-1)


# if __name__ == "__main__":
#     main()
