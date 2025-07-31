import torch
from tqdm import tqdm

# Padding multiplier for route tensors (60% extra space for depot visits)
MULT = 0.6


def generate_isolate_solution(cvrp_instance):
    """
    Generate initial CVRP solutions using the Isolate Algorithm.

    This creates a solution where each client is visited individually
    with depot returns between each visit: [0,1,0,2,0,3,0,4,...,0,N,0]

    Args:
        cvrp_instance: An instance of the CVRP class containing problem data

    Returns:
        torch.Tensor: Solution tensor of shape [batch_size, route_length, 1]
    """
    batch_size = cvrp_instance.n_problems
    dim = cvrp_instance.dim  # Number of clients (excluding depot)
    device = "cpu"

    # Pattern: [0, 1, 0, 2, ..., 0, N, 0]
    route_length = 2 * dim + 1  # Start depot, depot after each client, end depot

    # Build pattern for a single batch
    pattern = torch.zeros(route_length, dtype=torch.long, device=device)
    pattern[0] = 0  # Start at depot
    pattern[-1] = 0  # End at depot
    pattern[1:-1:2] = torch.arange(1, dim + 1, device=device)  # Clients
    pattern[2:-1:2] = 0  # Depot after each client

    # Repeat for batch
    routes = pattern.unsqueeze(0).repeat(batch_size, 1)
    return routes.unsqueeze(-1)


def generate_sweep_solution(cvrp_instance):
    """
    Generate initial CVRP solutions using the Sweep Algorithm.

    This classical heuristic constructs routes by:
    1. Sorting clients by polar angle around the depot
    2. Sweeping through clients in angular order
    3. Starting new routes when vehicle capacity would be exceeded

    Args:
        cvrp_instance: An instance of the CVRP class containing problem data

    Returns:
        torch.Tensor: Solution tensor of shape [batch_size, max_route_length, 1]
    """
    # --------------------------------
    # Extract problem parameters
    # --------------------------------
    batch_size = cvrp_instance.n_problems  # Number of problem instances in batch
    dim = cvrp_instance.dim  # Number of clients (excluding depot)
    num_total_nodes = dim + 1  # Total nodes including depot
    device = "cpu"  # Computation device (CPU/GPU)
    coords = cvrp_instance.coords.cpu()  # Node coordinates [batch, nodes, 2]
    demands = cvrp_instance.demands.cpu()  # Node demands [batch, nodes]
    capacity = cvrp_instance.capacity.squeeze(-1).cpu()  # Vehicle capacity constraint

    # --------------------------------
    # Initialize solution tensor
    # --------------------------------
    # Add padding for depot returns (60% extra space)
    max_route_len = num_total_nodes + int(num_total_nodes * MULT)
    routes = torch.zeros(batch_size, max_route_len, dtype=torch.long, device=device)

    # --------------------------------
    # Prepare coordinate calculations
    # --------------------------------
    # Extract depot coordinates (first node in each problem)
    depot_coords = coords[:, 0:1, :]  # [batch_size, 1, 2]
    # Extract client coordinates (all nodes except depot)
    client_coords = coords[:, 1:, :]  # [batch_size, dim, 2]

    # Calculate vectors from depot to each client
    delta_coords = client_coords - depot_coords  # [batch_size, dim, 2]

    # Calculate polar angles (in radians) for each client relative to depot
    # atan2(y, x) gives angle in radians in range [-π, π]
    angles = torch.atan2(
        delta_coords[:, :, 1], delta_coords[:, :, 0]
    )  # [batch_size, dim]

    # --------------------------------
    # Sort clients by polar angle
    # --------------------------------
    # Create client indices (1 to dim) for each problem
    client_indices = torch.arange(1, dim + 1, device=device)
    client_indices = client_indices.expand(batch_size, -1)  # [batch_size, dim]

    # Sort clients by their polar angles for each problem
    sorted_indices = []
    sorted_demands = []

    for b in range(batch_size):
        # Get sorted indices for this problem
        _, indices = torch.sort(angles[b])
        sorted_indices.append(client_indices[b, indices])
        # Get corresponding demands
        sorted_demands.append(demands[b, client_indices[b, indices]])

    # Stack tensors for batch processing
    sorted_indices = torch.stack(sorted_indices)  # [batch_size, dim]
    sorted_demands = torch.stack(sorted_demands)  # [batch_size, dim]

    # --------------------------------
    # Construct routes using sorted clients
    # --------------------------------
    # Initialize variables for route construction
    current_loads = torch.zeros(batch_size, device=device)
    route_pos = torch.ones(
        batch_size, dtype=torch.long, device=device
    )  # Start at pos 1 (after depot)

    # All routes start at depot
    routes[:, 0] = 0

    # Sweep through sorted clients
    for i in range(dim):
        client_ids = sorted_indices[:, i]
        client_demands = sorted_demands[:, i]

        # Check if adding client would exceed capacity
        capacity_check = current_loads + client_demands <= capacity

        # For problems where capacity would be exceeded:
        # 1. Return to depot (already filled with 0s)
        # 2. Reset current load to 0
        route_pos = torch.where(capacity_check, route_pos, route_pos + 1)
        current_loads = torch.where(
            capacity_check, current_loads + client_demands, client_demands
        )

        # Add client to route
        for b in range(batch_size):
            if route_pos[b] < max_route_len:
                routes[b, route_pos[b]] = client_ids[b]

        # Move position pointer
        route_pos += 1

        # Check for route_pos overflow
        if (route_pos >= max_route_len).any():
            print("Warning: Some routes may be truncated")

    # Return to depot at the end of each route if not already there
    for b in range(batch_size):
        if route_pos[b] < max_route_len and routes[b, route_pos[b] - 1] != 0:
            routes[b, route_pos[b]] = 0

    # Add final dimension to match expected shape
    return routes.unsqueeze(-1)


def random_init_batch(cvrp_instance) -> torch.Tensor:
    """
    Vectorized random algorithm for initial CVRP solution generation.

    Args:
        demands: Tensor of shape [batch_size, num_nodes] representing client demands
        capacity: Vehicle capacity constraint

    Returns:
        Tensor of shape [batch_size, max_route_length, 1] containing initial routes
        with depot visits inserted to respect capacity constraints
    """
    demands = cvrp_instance.demands.cpu()  # Ensure demands are on CPU
    capacity = cvrp_instance.capacity.squeeze(-1).cpu()  # Ensure capacity is a scalar
    batch_size, num_nodes = demands.size()
    device = demands.device

    # Initialize solution tensor with 40% padding for depot insertions
    max_route_length = num_nodes + int(num_nodes * MULT)
    routes = torch.zeros(batch_size, max_route_length, dtype=torch.long, device=device)

    # Prepare client indices (excluding depot 0)
    clients = torch.arange(1, num_nodes, device=device)
    clients = clients.repeat(batch_size, 1)  # [batch_size, num_nodes-1]

    # Shuffle clients differently for each problem in batch
    clients = torch.stack([row[torch.randperm(row.size(0))] for row in clients])

    # Add depot (0) at start of each route
    clients = torch.cat(
        [
            torch.zeros(clients.size(0), 1, dtype=clients.dtype, device=device),
            clients,
        ],
        dim=1,
    )

    # Initialize tracking variables
    remaining_capacity = capacity
    route_positions = torch.zeros(batch_size, dtype=torch.long, device=device)
    vehicle_count = torch.ones(batch_size, device=device)

    # Build routes by sequentially assigning clients
    while (clients >= 0).any():  # While clients remain unassigned
        current_client = clients[:, 0]

        # Check if client demand can be satisfied with current capacity
        can_serve = (
            demands[torch.arange(batch_size), current_client] <= remaining_capacity
        )

        # Update route with client or depot (0)
        routes[torch.arange(batch_size), route_positions] = torch.where(
            can_serve, current_client, 0
        )
        route_positions += 1

        # Update remaining capacity
        remaining_capacity = torch.where(
            can_serve,
            remaining_capacity - demands[torch.arange(batch_size), current_client],
            remaining_capacity,
        )

        # If can't serve, return to depot and dispatch new vehicle
        remaining_capacity = torch.where(can_serve, remaining_capacity, capacity)
        vehicle_count = torch.where(can_serve, vehicle_count, vehicle_count + 1)

        # Remove assigned client from consideration
        clients = torch.where(
            can_serve.unsqueeze(1),
            torch.cat(
                [
                    clients[:, 1:],
                    torch.full((batch_size, 1), -1, device=clients.device),
                ],
                dim=1,
            ),
            clients,
        )

    # Ensure all negative values (padding) are set to depot (0)
    routes = torch.where(routes < 0, torch.tensor(0, device=routes.device), routes)

    return routes.unsqueeze(-1)  # Add dimension for compatibility


def construct_cvrp_solution(
    x: torch.Tensor, demands: torch.Tensor, capacity: int = 30
) -> torch.Tensor:
    """
    Constructs a valid CVRP solution from node sequence while respecting
    capacity constraints.

    Args:
        x: Tensor of shape [batch_size, num_nodes] representing node visit sequence
        demands: Tensor of shape [batch_size, num_nodes+1] containing client demands
        capacity: Maximum vehicle capacity

    Returns:
        Tensor of shape [batch_size, max_route_length, 1] with depot visits inserted
        to maintain feasible solutions
    """
    x = x.squeeze(-1)  # Remove trailing dimension if present
    batch_size, num_nodes = demands.shape
    device = x.device

    # Initialize solution with padding for depot insertions
    max_route_length = num_nodes + int(num_nodes * MULT)
    routes = torch.zeros(batch_size, max_route_length, dtype=torch.long, device=device)

    # Add depot at start of each route
    x = torch.cat(
        [torch.zeros(batch_size, 1, dtype=torch.long, device=device), x], dim=1
    )

    # Get demands in current route order
    ordered_demands = torch.gather(demands, 1, x)

    # Initialize tracking variables
    capacity = capacity.squeeze()
    remaining_capacity = capacity.long()
    route_pos = torch.zeros(batch_size, dtype=torch.long, device=device)

    for i in range(num_nodes):
        current_client = x[:, i]
        can_serve = ordered_demands[:, i] <= remaining_capacity

        # Insert client or depot based on capacity
        routes[torch.arange(batch_size), route_pos] = torch.where(
            can_serve, current_client, 0
        )
        route_pos += 1

        # Update capacity
        remaining_capacity = torch.where(
            can_serve, remaining_capacity - ordered_demands[:, i], remaining_capacity
        )

        # If capacity exceeded, return to depot and reset
        remaining_capacity = torch.where(
            can_serve, remaining_capacity, capacity - ordered_demands[:, i]
        )

        # Insert depot if needed
        routes[torch.arange(batch_size), route_pos] = torch.where(
            can_serve, 0, current_client
        )
        route_pos += (~can_serve).long()

    return routes.unsqueeze(-1)


def generate_nearest_neighbor(cvrp_instance):
    """
    Generate initial CVRP solutions using the Nearest Neighbor Heuristic.

    This heuristic starts at the depot and iteratively visits the closest unvisited
    customer,
    returning to the depot when vehicle capacity would be exceeded.

    Args:
        cvrp_instance: An instance of the CVRP class containing problem data

    Returns:
        torch.Tensor: Solution tensor of shape [batch_size, max_route_length, 1]
    """
    # Extract problem parameters
    batch_size = cvrp_instance.n_problems
    dim = cvrp_instance.dim  # Number of clients (excluding depot)
    device = "cpu"
    coords = cvrp_instance.coords.cpu()
    demands = cvrp_instance.demands.cpu()
    capacity = cvrp_instance.capacity.squeeze(-1).cpu()

    # Compute distance matrix
    dist_matrix = torch.cdist(coords, coords, p=2)

    # Initialize routes tensor with padding
    max_route_len = dim + 1 + int((dim + 1) * MULT)
    routes = torch.zeros(batch_size, max_route_len, dtype=torch.long, device=device)

    # Track visited clients (True for visited)
    visited = torch.zeros(batch_size, dim + 1, dtype=torch.bool, device=device)
    visited[:, 0] = True  # Depot is not a client to be visited

    # Initialize state for each problem in the batch
    current_node = torch.zeros(batch_size, dtype=torch.long, device=device)
    current_load = torch.zeros(batch_size, device=device)
    route_pos = torch.ones(batch_size, dtype=torch.long, device=device)

    # All routes start at the depot
    routes[:, 0] = 0

    # Loop until all clients are visited
    while not visited[:, 1:].all():
        # Create a mask for unvisited clients
        unvisited_mask = ~visited

        # Set distances to visited nodes to infinity
        dist_from_current = dist_matrix[torch.arange(batch_size), current_node].clone()
        dist_from_current[unvisited_mask] = dist_from_current[unvisited_mask]
        dist_from_current[~unvisited_mask] = float("inf")
        dist_from_current[:, 0] = float("inf")  # Don't go back to depot unless needed

        # Find the nearest unvisited neighbor for each problem
        nearest_neighbor = torch.argmin(dist_from_current, dim=1)

        # For nodes where all clients are visited, nearest_neighbor might be invalid
        # Set to depot for those cases
        all_clients_visited = visited[:, 1:].all(dim=1)
        nearest_neighbor = torch.where(
            all_clients_visited, torch.zeros_like(nearest_neighbor), nearest_neighbor
        )

        # Get demands of nearest neighbors
        neighbor_demands = demands[torch.arange(batch_size), nearest_neighbor]

        # Check if adding the neighbor exceeds capacity
        over_capacity = (current_load + neighbor_demands) > capacity

        # For problems where capacity is exceeded or all clients visited:
        # 1. Return to depot (add 0 to route)
        # 2. Reset load
        # 3. Update current node to depot
        routes[torch.arange(batch_size), route_pos] = torch.where(
            over_capacity | all_clients_visited,
            torch.zeros_like(nearest_neighbor),
            nearest_neighbor,
        )

        # Update route position for all problems
        route_pos += 1

        # For problems that returned to depot, add another position
        current_node = torch.where(
            over_capacity | all_clients_visited,
            torch.zeros_like(current_node),  # Return to depot
            nearest_neighbor,  # Go to nearest neighbor
        )

        # Update current load based on action taken
        current_load = torch.where(
            over_capacity | all_clients_visited,
            torch.zeros_like(current_load),  # Reset if returned to depot
            current_load + neighbor_demands,  # Add demand if visited client
        )

        # Mark nearest neighbors as visited (except depot)
        for b in range(batch_size):
            if not all_clients_visited[b] and not over_capacity[b]:
                visited[b, nearest_neighbor[b]] = True

    # Ensure final return to depot
    for b in range(batch_size):
        if route_pos[b] < max_route_len and routes[b, route_pos[b] - 1] != 0:
            routes[b, route_pos[b]] = 0
            route_pos[b] += 1

    return routes.unsqueeze(-1)


def generate_Clark_and_Wright(cvrp_instance):
    """
    Generate initial CVRP solutions using the Clarke and Wright Savings Algorithm.

    This algorithm works by:
    1. Initially creating N separate routes (depot-client-depot) for each client
    2. Computing savings from merging pairs of routes
    3. Iteratively merging routes with highest savings that maintain feasibility

    Args:
        cvrp_instance: An instance of the CVRP class containing problem data

    Returns:
        torch.Tensor: Solution tensor of shape [batch_size, max_route_length, 1]
    """
    batch_size = cvrp_instance.n_problems
    dim = cvrp_instance.dim  # Number of clients (excluding depot)
    num_total_nodes = dim + 1
    device = "cpu"
    coords = cvrp_instance.coords.cpu()
    demands = cvrp_instance.demands.cpu()
    capacity = cvrp_instance.capacity.squeeze(-1).cpu()

    # Compute distance matrix [batch, num_nodes, num_nodes]
    coords_i = coords.unsqueeze(2)
    coords_j = coords.unsqueeze(1)
    dist = torch.norm(coords_i - coords_j, dim=3)

    # Compute savings matrix for each batch [batch, dim, dim]
    # Savings S_ij = d_{0i} + d_{0j} - d_{ij}
    d0i = dist[:, 0, 1:]  # [batch, dim]
    d0j = dist[:, 0, 1:]  # [batch, dim]
    dij = dist[:, 1:, 1:]  # [batch, dim, dim]
    savings = d0i.unsqueeze(2) + d0j.unsqueeze(1) - dij  # [batch, dim, dim]
    # Set diagonal to -inf to avoid merging a node with itself
    savings = savings + torch.diag_embed(
        torch.full((batch_size, dim), float("-inf"), device=device)
    )

    # For each batch, get sorted savings indices (descending)
    savings_flat = savings.view(batch_size, -1)
    sorted_savings, sorted_idx = torch.sort(savings_flat, descending=True, dim=1)
    # Map flat indices back to (i, j)
    i_idx = sorted_idx // dim
    j_idx = sorted_idx % dim
    # Convert to global node indices (clients are 1..dim)
    i_idx = i_idx + 1
    j_idx = j_idx + 1

    # Initialize routes: each client in its own route [0, i, 0]
    routes = [[[0, i + 1, 0] for i in range(dim)] for _ in range(batch_size)]
    route_demands = [
        [demands[b, i + 1].item() for i in range(dim)] for b in range(batch_size)
    ]
    # Track which route each client is in
    client_route = [{i + 1: i for i in range(dim)} for _ in range(batch_size)]

    # Try to merge routes according to savings
    for b in tqdm(range(batch_size), desc="Clark and Wright Init", leave=False):
        for k in range(dim * dim):
            i = i_idx[b, k].item()
            j = j_idx[b, k].item()
            if i == j:
                continue
            route_i = client_route[b].get(i, None)
            route_j = client_route[b].get(j, None)
            if route_i is None or route_j is None or route_i == route_j:
                continue
            # Check if i is at the end of its route and j at the start of its route
            route_i_seq = routes[b][route_i]
            route_j_seq = routes[b][route_j]
            if route_i_seq[-2] == i and route_j_seq[1] == j:
                total_demand = route_demands[b][route_i] + route_demands[b][route_j]
                if total_demand <= capacity[b].item():
                    # Merge route_i and route_j
                    new_route = route_i_seq[:-1] + route_j_seq[1:]
                    routes[b][route_i] = new_route
                    route_demands[b][route_i] = total_demand
                    # Remove route_j
                    routes[b][route_j] = []
                    route_demands[b][route_j] = 0
                    # Update client_route mapping
                    for node in route_j_seq[1:-1]:
                        client_route[b][node] = route_i
                    continue
            # Also check the reverse: j at end, i at start
            if route_j_seq[-2] == j and route_i_seq[1] == i:
                total_demand = route_demands[b][route_i] + route_demands[b][route_j]
                if total_demand <= capacity[b].item():
                    new_route = route_j_seq[:-1] + route_i_seq[1:]
                    routes[b][route_j] = new_route
                    route_demands[b][route_j] = total_demand
                    routes[b][route_i] = []
                    route_demands[b][route_i] = 0
                    for node in route_i_seq[1:-1]:
                        client_route[b][node] = route_j
                    continue

    # Collect non-empty routes and flatten to padded tensor
    max_route_len = num_total_nodes + int(num_total_nodes * MULT)
    batch_routes = torch.zeros(
        batch_size, max_route_len, dtype=torch.long, device=device
    )
    for b in range(batch_size):
        flat = []
        for r in routes[b]:
            if r:
                if flat and flat[-1] != 0:
                    flat.append(0)
                flat += r[1:] if flat else r
        # Ensure route starts at depot
        if not flat or flat[0] != 0:
            flat = [0] + flat
        # Pad to max_route_len
        flat = flat[:max_route_len] + [0] * (max_route_len - len(flat))
        batch_routes[b, : len(flat)] = torch.tensor(
            flat[:max_route_len], dtype=torch.long
        )
    return batch_routes.unsqueeze(-1)


def cheapest_insertion(cvrp_instance):
    """
    Vectorized implementation of the Cheapest Insertion algorithm for the CVRP.

    Args:
        cvrp_instance: An instance of the CVRP class containing problem data

    Returns:
        routes: tensor of shape [batch_size, solution_length] containing the routes
    """

    batch_size = cvrp_instance.n_problems
    n_clients = cvrp_instance.dim
    n_total = n_clients + 1  # Including depot
    distance_matrix = cvrp_instance.matrix.cpu()
    demand = cvrp_instance.demands.cpu()
    vehicle_capacity = cvrp_instance.capacity.squeeze(-1).cpu()
    device = "cpu"

    # Initialization: for each problem, start with an empty route [0, 0]
    routes = [
        torch.tensor([0, 0], dtype=torch.long, device=device) for _ in range(batch_size)
    ]
    remaining_clients = (
        torch.arange(1, n_clients + 1, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    current_load = torch.zeros(batch_size, device=device)

    # Mask to track which clients have been inserted
    inserted = torch.zeros((batch_size, n_total), dtype=torch.bool, device=device)
    inserted[:, 0] = True  # The depot is always inserted

    while True:
        # Find clients not yet inserted
        not_inserted = ~inserted
        not_inserted[:, 0] = False  # Ignore the depot

        # If all clients are inserted, we are done
        if not_inserted.sum().item() == 0:
            break

        # For each problem in the batch
        for b in range(batch_size):
            if not_inserted[b].sum().item() == 0:
                continue  # All clients are already inserted for this problem

            best_cost = float("inf")
            best_position = -1
            best_client = -1

            # Find all possible positions for insertion
            depot_positions = (routes[b] == 0).nonzero().flatten()

            for i in range(len(depot_positions) - 1):
                start_idx = depot_positions[i]
                end_idx = depot_positions[i + 1]
                subroute = routes[b][start_idx : end_idx + 1]

                # Calculate the current demand of this subroute
                route_demand = (
                    demand[b, subroute[1:-1]].sum() if len(subroute) > 2 else 0
                )
                remaining_capacity = vehicle_capacity[b] - route_demand

                # Find eligible clients for this route
                eligible_clients = remaining_clients[b][
                    (demand[b, remaining_clients[b]] <= remaining_capacity)
                    & not_inserted[b, remaining_clients[b]]
                ]

                if len(eligible_clients) == 0:
                    continue

                # For each pair of consecutive nodes in the subroute
                for k in range(len(subroute) - 1):
                    u = subroute[k]
                    v = subroute[k + 1]

                    # Calculate the insertion cost for each eligible client
                    delta = (
                        distance_matrix[b, u, eligible_clients]
                        + distance_matrix[b, eligible_clients, v]
                        - distance_matrix[b, u, v]
                    )

                    # Find the best client for this insertion
                    min_delta, min_idx = delta.min(dim=0)
                    if min_delta < best_cost:
                        best_cost = min_delta.item()
                        best_client = eligible_clients[min_idx].item()
                        best_position = start_idx + k + 1

            # If no insertion is possible in existing routes, create a new route
            if best_cost == float("inf"):
                # Find the closest uninserted client to the depot
                eligible = remaining_clients[b][not_inserted[b, remaining_clients[b]]]
                if len(eligible) == 0:
                    continue

                depot_dist = distance_matrix[b, 0, eligible]
                _, closest_idx = depot_dist.min(dim=0)
                best_client = eligible[closest_idx].item()

                # Find the position after the last 0
                last_zero_pos = (routes[b] == 0).nonzero()[-1].item()
                routes[b] = torch.cat(
                    [
                        routes[b][: last_zero_pos + 1],
                        torch.tensor([best_client, 0], device=device),
                        routes[b][last_zero_pos + 1 :],
                    ]
                )
                inserted[b, best_client] = True
                current_load[b] = demand[b, best_client]
            else:
                # Insert the best client found
                routes[b] = torch.cat(
                    [
                        routes[b][:best_position],
                        torch.tensor([best_client], device=device),
                        routes[b][best_position:],
                    ]
                )
                inserted[b, best_client] = True
                current_load[b] += demand[b, best_client]

    # Convert the list of routes into a single tensor with padding if necessary
    max_len = max(len(r) for r in routes) + int(n_clients * MULT)
    padded_routes = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    for b in range(batch_size):
        padded_routes[b, : len(routes[b])] = routes[b]

    return padded_routes.unsqueeze(-1)


def path_cheapest_arc(cvrp_instance):
    """
    Implementation of the Path Cheapest Arc for the CVRP with strict constraints.

    Args:
        cvrp_instance: An instance of the CVRP class containing problem data

    Returns:
        routes: tensor [batch_size, max_len] (sequence of type [0,1,2,0,3,4,0,...])
    """
    batch_size = cvrp_instance.n_problems
    n_clients = cvrp_instance.dim
    n_total = n_clients + 1  # Including depot
    distance_matrix = cvrp_instance.matrix.cpu()
    demand = cvrp_instance.demands.cpu()
    vehicle_capacity = cvrp_instance.capacity.squeeze(-1).cpu()
    device = "cpu"

    # Initialization
    routes = [
        torch.tensor([0], dtype=torch.long, device=device) for _ in range(batch_size)
    ]
    remaining_clients = (
        torch.arange(1, n_clients + 1, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    inserted = torch.zeros((batch_size, n_total), dtype=torch.bool, device=device)
    inserted[:, 0] = True

    while True:
        # Check if all clients are inserted
        not_inserted = ~inserted
        not_inserted[:, 0] = False  # Ignore the depot
        if not_inserted.sum() == 0:
            break

        for b in range(batch_size):
            if not_inserted[b].sum() == 0:
                continue

            best_cost = float("inf")
            best_client = -1
            best_pos = -1
            new_route_needed = True

            # Calculate the used capacity in each existing sub-route
            route = routes[b]
            depot_positions = (route == 0).nonzero().flatten()

            for i in range(len(depot_positions) - 1):
                start, end = depot_positions[i], depot_positions[i + 1]
                subroute = route[start : end + 1]

                # Calculate the current demand
                current_demand = (
                    demand[b, subroute[1:-1]].sum() if len(subroute) > 2 else 0
                )
                remaining_cap = vehicle_capacity[b] - current_demand

                # Eligible clients for this sub-route
                eligible = remaining_clients[b][
                    (demand[b, remaining_clients[b]] <= remaining_cap)
                    & not_inserted[b, remaining_clients[b]]
                ]

                if len(eligible) == 0:
                    continue

                # Find the best arc for insertion
                for j in range(len(subroute) - 1):
                    u, v = subroute[j], subroute[j + 1]
                    delta = (
                        distance_matrix[b, u, eligible]
                        + distance_matrix[b, eligible, v]
                        - distance_matrix[b, u, v]
                    )
                    min_delta, min_idx = delta.min(dim=0)

                    if min_delta < best_cost:
                        best_cost = min_delta.item()
                        best_client = eligible[min_idx].item()
                        best_pos = start + j + 1
                        new_route_needed = False

            # Handle new routes if necessary
            if new_route_needed:
                eligible = remaining_clients[b][not_inserted[b, remaining_clients[b]]]
                if len(eligible) == 0:
                    continue

                # Choose the closest client to the depot
                depot_dists = distance_matrix[b, 0, eligible]
                _, closest_idx = depot_dists.min(dim=0)
                best_client = eligible[closest_idx].item()

                # Add new route [0, client, 0]
                if routes[b][-1] != 0:
                    routes[b] = torch.cat([routes[b], torch.tensor([0], device=device)])
                routes[b] = torch.cat(
                    [routes[b], torch.tensor([best_client, 0], device=device)]
                )
            else:
                # Insert the client at the optimal position
                routes[b] = torch.cat(
                    [
                        routes[b][:best_pos],
                        torch.tensor([best_client], device=device),
                        routes[b][best_pos:],
                    ]
                )

            inserted[b, best_client] = True

    # Padding for uniform output format
    max_len = max(len(r) for r in routes) + int(n_clients * MULT)
    padded_routes = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    for b in range(batch_size):
        padded_routes[b, : len(routes[b])] = routes[b]

    return padded_routes.unsqueeze(-1)  # Add dimension for compatibility
