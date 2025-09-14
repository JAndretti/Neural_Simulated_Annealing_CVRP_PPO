import random
import math
import torch
from typing import Tuple, Optional
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def generate_cvrp_instance(
    n: int,
    depot_pos: int,
    cust_pos: int,
    demand_type: int,
    avg_route_size: int,
    rand_seed: Optional[int] = None,
    instance_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Generate a CVRP instance with PyTorch tensors.

    Args:
        n: Number of customers
        depot_pos: Depot positioning (1=Random, 2=Centered, 3=Cornered)
        cust_pos: Customer positioning (1=Random, 2=Clustered, 3=Random-clustered)
        demand_type: Demand distribution (1-7)
        avg_route_size: Average route size (1-6)
        rand_seed: Random seed for reproducibility
        instance_id: Instance identifier

    Returns:
        Tuple of (coordinates_tensor, demands_tensor, capacity_tensor, metadata)
    """

    def distance(x, y):
        return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

    # Constants
    maxCoord = 1000
    decay = 40

    # Set random seed if provided
    if rand_seed is not None:
        random.seed(rand_seed)

    # Validate inputs
    if demand_type > 7 or demand_type < 1:
        raise ValueError("Demand type must be between 1 and 7")

    if avg_route_size > 6 or avg_route_size < 1:
        raise ValueError("Average route size must be between 1 and 6")

    if depot_pos > 3 or depot_pos < 1:
        raise ValueError("Depot position must be between 1 and 3")

    if cust_pos > 3 or cust_pos < 1:
        raise ValueError("Customer position must be between 1 and 3")

    # Determine number of seeds for clustering
    nSeeds = random.randint(2, 6) if cust_pos in [2, 3] else 0

    # Route size parameters
    In = {1: (3, 5), 2: (5, 8), 3: (8, 12), 4: (12, 16), 5: (16, 25), 6: (25, 50)}
    r = random.uniform(In[avg_route_size][0], In[avg_route_size][1])

    # Generate instance name
    instanceName = (
        f"XML{n}_{depot_pos}{cust_pos}{demand_type}{avg_route_size}_{instance_id:02d}"
    )

    depot = (-1, -1)
    S = set()  # set of coordinates for the customers

    # Depot positioning
    if depot_pos == 1:
        x_ = random.randint(0, maxCoord)
        y_ = random.randint(0, maxCoord)
    elif depot_pos == 2:
        x_ = y_ = int(maxCoord / 2.0)
    elif depot_pos == 3:
        x_ = y_ = 0
    depot = (x_, y_)

    # Customer positioning
    if cust_pos == 3:
        nRandCust = int(n / 2.0)
    elif cust_pos == 2:
        nRandCust = 0
    elif cust_pos == 1:
        nRandCust = n
        nSeeds = 0

    nClustCust = n - nRandCust

    # Generating random customers
    for i in range(nRandCust):
        x_ = random.randint(0, maxCoord)
        y_ = random.randint(0, maxCoord)
        while (x_, y_) in S or (x_, y_) == depot:
            x_ = random.randint(0, maxCoord)
            y_ = random.randint(0, maxCoord)
        S.add((x_, y_))

    nS = nRandCust
    seeds = []

    # Generation of the clustered customers
    if nClustCust > 0:
        if nClustCust < nSeeds:
            nSeeds = nClustCust  # Adjust seeds to avoid error

        # Generate the seeds
        for i in range(nSeeds):
            x_ = random.randint(0, maxCoord)
            y_ = random.randint(0, maxCoord)
            while (x_, y_) in S or (x_, y_) == depot:
                x_ = random.randint(0, maxCoord)
                y_ = random.randint(0, maxCoord)
            S.add((x_, y_))
            seeds.append((x_, y_))
        nS = nS + nSeeds

        # Determine the seed with maximum sum of weights
        maxWeight = 0.0
        for seed in seeds:
            w_ij = 0.0
            for other_seed in seeds:
                w_ij += 2 ** (-distance(seed, other_seed) / decay)
            if w_ij > maxWeight:
                maxWeight = w_ij

        norm_factor = 1.0 / maxWeight if maxWeight > 0 else 1.0

        # Generate remaining customers using Accept-reject method
        while nS < n:
            x_ = random.randint(0, maxCoord)
            y_ = random.randint(0, maxCoord)
            while (x_, y_) in S or (x_, y_) == depot:
                x_ = random.randint(0, maxCoord)
                y_ = random.randint(0, maxCoord)

            weight = 0.0
            for seed in seeds:
                weight += 2 ** (-distance((x_, y_), seed) / decay)
            weight *= norm_factor
            rand = random.uniform(0, 1)

            if rand <= weight:  # Accept the customer
                S.add((x_, y_))
                nS = nS + 1

    V = [depot] + list(S)  # set of vertices

    # Demands
    demandMinValues = [1, 1, 5, 1, 50, 1, 51, 50, 1]
    demandMaxValues = [1, 10, 10, 100, 100, 50, 100, 100, 10]
    demandMin = demandMinValues[demand_type - 1]
    demandMax = demandMaxValues[demand_type - 1]
    demandMinEvenQuadrant = 51
    demandMaxEvenQuadrant = 100
    demandMinLarge = 50
    demandMaxLarge = 100
    largePerRoute = 1.5
    demandMinSmall = 1
    demandMaxSmall = 10

    D = []  # demands
    sumDemands = 0
    maxDemand = 0

    for i in range(1, n + 1):  # Skip depot (index 0)
        j = int((demandMax - demandMin + 1) * random.uniform(0, 1) + demandMin)

        if demand_type == 6:
            if (V[i][0] < maxCoord / 2.0 and V[i][1] < maxCoord / 2.0) or (
                V[i][0] >= maxCoord / 2.0 and V[i][1] >= maxCoord / 2.0
            ):
                j = int(
                    (demandMaxEvenQuadrant - demandMinEvenQuadrant + 1)
                    * random.uniform(0, 1)
                    + demandMinEvenQuadrant
                )

        if demand_type == 7:
            if i < (n / r) * largePerRoute:
                j = int(
                    (demandMaxLarge - demandMinLarge + 1) * random.uniform(0, 1)
                    + demandMinLarge
                )
            else:
                j = int(
                    (demandMaxSmall - demandMinSmall + 1) * random.uniform(0, 1)
                    + demandMinSmall
                )

        D.append(j)
        if j > maxDemand:
            maxDemand = j
        sumDemands = sumDemands + j

    # Generate capacity
    if sumDemands == n:
        capacity = math.floor(r)
    else:
        capacity = max(maxDemand, math.ceil(r * sumDemands / n))

    k = math.ceil(sumDemands / float(capacity))

    # Create PyTorch tensors
    coords_tensor = torch.tensor(V, dtype=torch.float32)
    demands_tensor = torch.tensor([0] + D, dtype=torch.float32)  # depot demand = 0
    capacity_tensor = torch.tensor([capacity], dtype=torch.float32)

    # Metadata
    metadata = {
        "instance_name": instanceName,
        "n_customers": n,
        "capacity": capacity,
        "total_demand": sumDemands,
        "estimated_vehicles": k,
        "depot_position": depot_pos,
        "customer_position": cust_pos,
        "demand_type": demand_type,
        "avg_route_size": avg_route_size,
        "random_seed": rand_seed,
    }

    return coords_tensor, demands_tensor, capacity_tensor, metadata


def generate_single_instance(args):
    """Function to generate a single instance, compatible with multiprocessing"""
    i, base_seed, n_customers = args
    # Generate random parameters
    depot_pos = random.randint(1, 3)
    cust_pos = random.randint(1, 3)
    demand_type = random.randint(1, 7)
    avg_route_size = random.randint(1, 6)

    # Generate the instance
    coords, demands, capacity, metadata = generate_cvrp_instance(
        n=n_customers,
        depot_pos=depot_pos,
        cust_pos=cust_pos,
        demand_type=demand_type,
        avg_route_size=avg_route_size,
        rand_seed=base_seed + i,  # Different seed for each instance
        instance_id=i + 1,
    )
    return coords, demands, capacity, metadata


def P_generate_instances(n_instances, base_seed, customers):
    """
    Generate multiple CVRP instances in parallel and display statistics.
    Returns separate lists for coords, demands, capacity, and metadata.
    """

    with Pool(processes=cpu_count() - 1) as pool:
        # Create arguments for each instance
        args_list = [(i, base_seed, customers) for i in range(n_instances)]

        # Use tqdm to display progress
        results = list(
            tqdm(
                pool.imap(generate_single_instance, args_list),
                total=n_instances,
                desc="Generating CVRP instances",
                leave=False,
            )
        )

        coords, demands, capacity, metadata = zip(*results)
    return coords, demands, capacity, metadata


def stack_res(coords_list, demands_list, capacity_list):
    coords_tensor = torch.stack(coords_list)
    min_coords = torch.min(coords_tensor, dim=1, keepdim=True).values
    max_coords = torch.max(coords_tensor, dim=1, keepdim=True).values
    coords_tensor = (coords_tensor - min_coords) / (max_coords - min_coords)
    demands_tensor = torch.stack(demands_list)
    capacity_tensor = torch.stack(capacity_list)
    return coords_tensor, demands_tensor, capacity_tensor


# if __name__ == "__main__":

#     # Parameters for random generation
#     n_instances = 10000
#     customers = 100

#     print(f"Generating {n_instances} random CVRP instances...")
#     print(f"Available CPU cores: {cpu_count()-1}")

#     # Use multiprocessing to generate instances in parallel
#     base_seed = random.randint(0, 1000000)  # Base seed for reproducibility

#     coords_list, demands_list, capacity_list, metadata_list = P_generate_instances(
#         n_instances, base_seed, customers
#     )

#     # Stack the results into tensors
#     coords_tensor, demands_tensor, capacity_tensor = stack_res(
#         coords_list, demands_list, capacity_list
#     )

#     print(f"Generated {n_instances} instances")
#     print(f"Coordinates tensor shape: {coords_tensor.shape}")
#     print(f"Demands tensor shape: {demands_tensor.shape}")
#     print(f"Capacity tensor shape: {capacity_tensor.shape}")
