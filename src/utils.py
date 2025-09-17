import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


def extend(tensor: torch.Tensor, dims: int) -> torch.Tensor:
    """Extend tensor to match dimensions 'dims'."""
    return tensor[(...,) + (None,) * dims]


def extend_to(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """Extend tensor1 to have same number of dims as tensor2."""
    return extend(tensor1, len(tensor2.shape) - len(tensor1.shape))


def repeat_to(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """Repeat tensor1 to match the shape of tensor2."""
    tensor1 = extend_to(tensor1, tensor2)
    ones = torch.ones(tensor2.shape[:-1] + (1,), device=tensor1.device)
    return tensor1 * ones


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array."""
    return tensor.detach().cpu().numpy()


def convert_tensor(a: torch.Tensor) -> torch.Tensor:
    """Convert a tensor to a tensor of indices where values change."""
    # Create a mask to identify indices where values change
    diff = (a[:, 1:] != a[:, :-1]) & (a[:, 1:] != 0)

    # Initialize a result tensor with zeros
    result = torch.zeros_like(a, dtype=torch.int)

    # Fill the first column of result with zeros
    result[:, 0] = 0

    # Use a tensor to increment indices where there is a non-zero value change
    indices = torch.cumsum(diff, dim=1)
    # Add the last value of each row to indices
    indices = torch.cat((indices, indices[:, -1].unsqueeze(1)), dim=1)
    return indices


# Code inspired by Google OR Tools plot:
# https://github.com/google/or-tools/blob/fb12c5ded7423d524fc6c95656a9bdc290a81d4d/examples/python/cvrptw_plot.py


def discrete_cmap(N, base_cmap=None):
    """
    Create an N-bin discrete colormap from the specified input map
    """
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def prepare_plot(pb, init):
    depot = pb.coords[:, 0, :].squeeze()
    loc = pb.coords[:, 1:, :].squeeze()
    demand = pb.demands[:, 1:].squeeze()
    data = {"depot": depot, "loc": loc, "demand": demand}
    init = init.squeeze()
    return data, init


def plot_vehicle_routes(
    data,
    route,
    ax1,
    markersize=5,
    visualize_demands=False,
    capacity=1,
    round_demand=False,
    title="None",
):
    """
    Plot the vehicle routes on matplotlib axis ax1.
    data: dict with 'depot', 'loc', 'demand'
    route: LongTensor [route_length] with 0 for depot and 1..N
    ax1: matplotlib axis
    """

    # route is one sequence, separating different routes with 0 (depot)
    routes = [
        r[r != 0]
        for r in np.split(route.cpu().numpy(), np.where(route == 0)[0])
        if (r != 0).any()
    ]
    depot = data["depot"].cpu().numpy()
    locs = data["loc"].cpu().numpy()
    demands = data["demand"].cpu().numpy()  # * demand_scale

    x_dep, y_dep = depot
    ax1.plot(x_dep, y_dep, "sk", markersize=markersize * 4)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # ax1.legend(loc="upper center")

    cmap = discrete_cmap(len(routes) + 2, "nipy_spectral")
    dem_rects = []
    used_rects = []
    cap_rects = []
    qvs = []
    total_dist = 0
    for veh_number, r in enumerate(routes):
        color = cmap(len(routes) - veh_number)  # Invert to have in rainbow order

        route_demands = demands[r - 1]
        coords = locs[r - 1, :]
        xs, ys = coords.transpose()

        total_route_demand = sum(route_demands)
        assert total_route_demand <= capacity
        if not visualize_demands:
            ax1.plot(xs, ys, "o", mfc=color, markersize=markersize, markeredgewidth=0.0)

        dist = 0
        x_prev, y_prev = x_dep, y_dep
        cum_demand = 0
        for (x, y), d in zip(coords, route_demands):
            dist += np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2)

            cap_rects.append(Rectangle((x, y), 0.01, 0.1))
            used_rects.append(
                Rectangle((x, y), 0.01, 0.1 * total_route_demand / capacity)
            )
            dem_rects.append(
                Rectangle(
                    (x, y + 0.1 * cum_demand / capacity), 0.01, 0.1 * d / capacity
                )
            )

            x_prev, y_prev = x, y
            cum_demand += d

        dist += np.sqrt((x_dep - x_prev) ** 2 + (y_dep - y_prev) ** 2)
        total_dist += dist
        qv = ax1.quiver(
            xs[:-1],
            ys[:-1],
            xs[1:] - xs[:-1],
            ys[1:] - ys[:-1],
            scale_units="xy",
            angles="xy",
            scale=1,
            color=color,
            label="R{}, # {}, c {} / {}, d {:.2f}".format(
                veh_number,
                len(r),
                int(total_route_demand) if round_demand else total_route_demand,
                int(capacity) if round_demand else capacity,
                dist,
            ),
        )

        qvs.append(qv)
    title = (
        f"{title or ''} {len(routes)} routes, total distance {total_dist:.4f}".strip()
    )

    ax1.set_title(title)
    ax1.legend(handles=qvs)

    pc_cap = PatchCollection(
        cap_rects, facecolor="whitesmoke", alpha=1.0, edgecolor="lightgray"
    )
    pc_used = PatchCollection(
        used_rects, facecolor="lightgray", alpha=1.0, edgecolor="lightgray"
    )
    pc_dem = PatchCollection(dem_rects, facecolor="black", alpha=1.0, edgecolor="black")

    if visualize_demands:
        ax1.add_collection(pc_cap)
        ax1.add_collection(pc_used)
        ax1.add_collection(pc_dem)


def calculate_client_angles(coords: torch.Tensor) -> torch.Tensor:
    """
    coords: [B, N, 2], coords[:,0] = depot
    returns [B, N, 1] with normalized angle [0,1] (depot=0)
    """
    depot = coords[:, :1]
    clients = coords[:, 1:]
    delta = clients - depot
    ang = torch.atan2(delta[..., 1], delta[..., 0])  # [-π,π]
    norm = ang.div(2 * torch.pi).add(0.5)  # [0,1]
    all_ang = torch.cat(
        [torch.zeros(coords.size(0), 1, device=coords.device), norm], dim=1
    )
    return all_ang.unsqueeze(-1)


def calculate_distance_matrix(coords: torch.Tensor) -> torch.Tensor:
    """
    coords: [B, N, 2] → Euclidean distances [B, N, N]
    """
    return torch.cdist(coords, coords, p=2)


def is_feasible(
    solution: torch.Tensor, demands: torch.Tensor, capacity: torch.Tensor
) -> torch.Tensor:
    """
    Check if the solution respects capacity constraints for all routes.

    Args:
        solution: Tensor [batch, route_length, 1] representing routes
        demands: Tensor [batch, route_length] with demands for each client (0 for depot)
        capacity: Tensor [batch] with vehicle capacities for each problem instance

    Returns:
        Tensor [batch] of boolean values indicating feasibility for each problem
    """
    batch_size = solution.size(0)
    device = solution.device

    mask = solution.squeeze(-1) != 0  # [batch, route_length]

    # Identify route starts: True where a route starts (depot->client)
    route_starts = torch.cat(
        [
            torch.ones(batch_size, 1, dtype=torch.bool, device=device),
            (~mask[:, :-1]) & mask[:, 1:],
        ],
        dim=1,
    )  # [batch, route_length]

    # Assign a route id to each position
    route_ids = torch.cumsum(route_starts, dim=1) - 1  # [batch, route_length]

    # Set depot positions to -1 so they don't contribute to route sums
    route_ids[~mask] = -1  # depot positions

    # Compute route demand sums (scatter_add)
    max_routes = route_ids.max().item() + 1 if route_ids.numel() > 0 else 0
    max_routes = max(max_routes, solution.size(1))  # ensure enough space
    route_demands = torch.zeros(
        batch_size, max_routes, dtype=torch.int64, device=device
    )

    route_demands.scatter_add_(1, torch.clamp(route_ids, min=0), demands * mask)

    # Check if any route demand exceeds capacity
    feasible = (route_demands <= capacity).all(dim=1)
    return feasible


def capacity_utilization(
    solution: torch.Tensor, demands: torch.Tensor, capacity: torch.Tensor
) -> torch.Tensor:
    """
    Calculate capacity utilization score for each solution.

    For each route in a solution, computes the ratio of total demand
    to vehicle capacity, then averages these ratios across all routes.
    Higher scores indicate better capacity utilization.

    Args:
        solution: Tensor [batch_size, route_length, 1] representing routes
        demands: Tensor [batch_size, route_length] with demands for each client
        (0 for depot)
        capacity: Tensor [batch_size] with vehicle capacities for each problem instance

    Returns:
        Tensor [batch_size] with utilization score for each solution
    """
    batch_size = solution.size(0)
    device = solution.device

    mask = solution.squeeze(-1) != 0  # [batch, route_length] - True for clients

    # Identify route starts: depot followed by client
    route_starts = torch.cat(
        [
            torch.ones(batch_size, 1, dtype=torch.bool, device=device),
            (~mask[:, :-1]) & mask[:, 1:],
        ],
        dim=1,
    )  # [batch, route_length]

    # Count number of routes per solution
    routes_count = route_starts.sum(dim=1).float()  # [batch_size]

    # Assign route IDs to each position
    route_ids = torch.cumsum(route_starts, dim=1) - 1  # [batch, route_length]

    # Mask depot positions so they don't contribute to demand sums
    route_ids[~mask] = -1

    # Compute total demand per route
    max_routes = route_ids.max().item() + 1 if route_ids.numel() > 0 else 0
    max_routes = max(max_routes, 1)  # ensure at least one route
    route_demands = torch.zeros(
        batch_size, max_routes, dtype=torch.float, device=device
    )

    route_demands.scatter_add_(
        1, torch.clamp(route_ids, min=0), demands.float() * mask.float()
    )

    # Calculate utilization ratio for each route (demand / capacity)
    route_utilization = route_demands / capacity

    # Get sum of utilization across all routes
    total_utilization = torch.sum(route_utilization, dim=1)

    # Compute average utilization per route
    avg_utilization = total_utilization / routes_count

    return 1 - avg_utilization
