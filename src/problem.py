from abc import ABC, abstractmethod
from typing import Dict, Tuple


import torch
import torch.nn.functional as F


from utils import repeat_to, convert_tensor


def greedy_init_batch(demands, capacity):
    """Vectorized greedy algorithm for initial solution generation.

    Args:
        demands: Tensor of shape [batch_size, pb_size] representing client demands.
        capacity: Scalar representing the vehicle capacity.

    Returns:
        routes: Tensor of shape [batch_size, max_route_length, 1] with initial routes.
    """
    batch_size, pb_size = demands.size()

    # Initialize tensors
    routes = torch.zeros(
        batch_size,
        pb_size + int(pb_size * 0.40),  # Add 20% to the maximum route length
        # to avoid trimming because we add 0 many times
        # for depot
        dtype=torch.long,
        device=demands.device,
    )  # Maximum route length
    clients = torch.arange(1, pb_size, device=demands.device).repeat(
        batch_size, 1
    )  # [batch_size, pb_size-1]
    # Shuffle the clients within each batch
    clients = torch.stack([row[torch.randperm(row.size(0))] for row in clients])
    clients = torch.cat(
        [
            torch.zeros(clients.size(0), 1, dtype=clients.dtype, device=demands.device),
            clients,
        ],
        dim=1,
    )
    capacity_left = torch.full(
        (batch_size,), capacity, device=demands.device
    )  # Remaining capacity per truck
    truck_counts = torch.ones(
        batch_size, device=demands.device
    )  # Truck counter per batch
    route_lengths = torch.zeros(
        batch_size, dtype=torch.long, device=demands.device
    )  # Route length tracker

    # Loop over clients (vectorized step by step)
    while (clients >= 0).any():  # While there are unvisited clients
        # Get the current client for each batch
        current_client = clients[:, 0]

        # Check which demands can be satisfied
        satisfiable = demands[torch.arange(batch_size), current_client] <= capacity_left

        # Update routes for satisfiable demands
        routes[torch.arange(batch_size), route_lengths] = torch.where(
            satisfiable, current_client, 0
        )
        route_lengths += 1  # Increment route length

        # Deduct satisfied demands from capacity
        capacity_left = torch.where(
            satisfiable,
            capacity_left - demands[torch.arange(batch_size), current_client],
            capacity_left,
        )

        # Reset capacity and add depot if not satisfiable
        capacity_left = torch.where(satisfiable, capacity_left, capacity)
        truck_counts = torch.where(satisfiable, truck_counts, truck_counts + 1)

        # Remove the client from the list of unvisited clients
        clients = torch.where(
            satisfiable.unsqueeze(1),
            torch.cat(
                [
                    clients[:, 1:],
                    torch.full((batch_size, 1), -1, device=clients.device),
                ],
                dim=1,
            ),
            clients,
        )
    # Replace all negative numbers in routes with 0
    routes = torch.where(routes < 0, torch.tensor(0, device=routes.device), routes)
    return routes.unsqueeze(-1)  # Add trailing dimension for compatibility


class Problem(ABC):
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.generator = torch.Generator(device=device)

    def gain(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.cost(s) - self.cost(self.update(s, a))

    def manual_seed(self, seed: int) -> None:
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

    @abstractmethod
    def cost(self, s: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def update(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def set_params(self, **kwargs) -> None:
        pass

    @abstractmethod
    def generate_params(self) -> Dict[str, torch.Tensor]:
        pass

    @property
    def state_encoding(self) -> torch.Tensor:
        pass

    @abstractmethod
    def generate_init_state(self) -> torch.Tensor:
        pass

    def to_state(
        self, x: torch.Tensor, temp: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate state encoding with x and repeat temp dynamically."""
        padding = max(0, x.size(1) - self.state_encoding.size(1))
        self_state_encoding = F.pad(self.state_encoding, (0, 0, 0, padding))
        components = [x, self_state_encoding]

        components.extend(self.get_percentage_demands(x))
        components.extend([self.cost_per_route(x)])
        components.extend([repeat_to(temp, x), repeat_to(time, x)])
        # components.extend(repeat_to(c, temp.size(1)) for c in components)
        return torch.cat(components, -1)

    def from_state(self, state: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Split state into components dynamically."""
        num_extra_features = state.shape[-1] - 3  # Adjusting for variable dimensions
        split_sizes = [1, 2] + [1] * num_extra_features
        return tuple(torch.split(state, split_sizes, dim=-1))


class CVRP(Problem):
    x_dim = 1

    def __init__(
        self,
        dim: int = 50,
        n_problems: int = 256,
        capacity: int = 40,
        device: str = "cpu",
        params: str = {},
    ):
        """Initialize CVRP.

        Args:
            dim: num items
            n_problems: batch size
            params: {'weight': torch.Tensor}
        """
        super().__init__(device)
        self.dim = dim
        self.n_problems = n_problems
        self.capacity = capacity
        self.params = params
        self.clustering = self.params["CLUSTERING"]
        self.nb_clusters_max = self.params["NB_CLUSTERS_MAX"]

    def set_params(self, params: dict = None) -> None:
        """Set params.

        Args:
            params: Dictionary containing 'coords' and 'demands'
        """
        if params["coords"] is not None:
            self.coords = params["coords"]
        if params["demands"] is not None:
            self.demands = params["demands"]

    def generate_params(self, mode: str = "train") -> Dict[str, torch.Tensor]:
        """Generate random coordinates in the unit square.

        Returns:
            coords [batch size, num problems, 2]
        """
        if mode == "test":
            self.manual_seed(0)
        if self.clustering:
            # Generate 5 centers for each problem
            cluster_centers = torch.rand(
                self.n_problems,
                self.nb_clusters_max,
                2,
                device=self.device,
                generator=self.generator,
            )  # Size (n_problems, self.nb_clusters_max, 2)

            # Assigne every node to a cluster
            cluster_assignments = torch.randint(
                0,
                self.nb_clusters_max,
                (self.n_problems, self.dim + 1),
                device=self.device,
                generator=self.generator,
            )  # size (n_problems, dim + 1)

            # Add some noise to the cluster centers
            coords = cluster_centers[
                torch.arange(self.n_problems)[:, None], cluster_assignments
            ] + 0.05 * torch.randn(
                self.n_problems,
                self.dim + 1,
                2,
                device=self.device,
                generator=self.generator,
            )

            # Clamp the coordinates to the unit square
            coords = torch.clamp(coords, 0, 1)
        else:
            coords = torch.rand(
                self.n_problems,
                self.dim + 1,
                2,
                device=self.device,
                generator=self.generator,
            )
        demands = torch.randint(
            1,
            10,
            (self.n_problems, self.dim + 1),
            device=self.device,
            generator=self.generator,
        )
        demands[:, 0] = 0
        return {"coords": coords, "demands": demands}

    def cost(self, s: torch.Tensor) -> torch.Tensor:
        """Compute Euclidean tour lengths from city permutations

        Args:
            s: [batch size, dim]
        """
        # Edge lengths
        edge_lengths = self.get_edge_lengths_in_tour(s)
        return torch.sum(edge_lengths, -1)

    def cost_per_route(self, s: torch.Tensor) -> torch.Tensor:
        """
        Calculate the cost per route for a given tensor of routes.

        This function computes the cost per route by first calculating the edge lengths
        in the tour, then summing these lengths. It creates a mask to identify non-zero
        elements in the input tensor `s`, detects changes in groups, and assigns unique
        group identifiers. It then calculates the sum of edge lengths per group and
        propagates these sums to the corresponding indices. Finally, it normalizes the
        output by the total cost.

        Args:
            s (torch.Tensor): A tensor representing the routes.

        Returns:
            torch.Tensor: A tensor containing the normalized cost per route.
        """
        edge_lengths = self.get_edge_lengths_in_tour(s)
        total_cost = torch.sum(edge_lengths, -1)
        # Create a mask for non-zero elements in s
        mask = s.squeeze(-1) != 0

        # Detect group changes
        shift = torch.cat(
            [torch.zeros((mask.shape[0], 1), dtype=torch.bool), mask[:, :-1]], dim=1
        )
        group_change = mask & ~shift  # Detect the start of new groups

        # Create unique group identifiers
        group_ids = torch.cumsum(group_change, dim=1) * mask

        # Calculate the sum of values per group
        sums = torch.zeros_like(edge_lengths)
        sums.scatter_add_(1, group_ids, edge_lengths)

        # Propagate the sums to the corresponding indices
        output = sums.gather(1, group_ids) * mask

        return ((output) / total_cost.view(-1, 1)).unsqueeze(-1)

    def cost_demands(
        self, s: torch.Tensor, demands: torch.Tensor
    ) -> torch.Tensor:  # TODO to long
        """Compute sum of demands for tours

        Args:
            s: [batch size, dim]
        """
        # Detect non-zero values
        mask = demands != 0

        # Create a marker for group breaks (when the difference
        # between adjacent indices > 1)
        shift = torch.cat(
            [
                torch.zeros(
                    (demands.shape[0], 1), dtype=torch.bool, device=self.device
                ),
                mask[:, :-1],
            ],
            dim=1,
        )
        group_change = mask & ~shift  # Start of new groups

        # Assign a unique group identifier (cumsum creates group labels)
        group_ids = torch.cumsum(group_change, dim=1) * mask

        # Sum by group
        sums = torch.zeros_like(demands)
        sums.scatter_add_(1, group_ids, demands)

        # Propagate summed values within their respective groups
        output = sums.gather(1, group_ids)

        return output

    def update(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Silly city swap

        Args:
            s: perm vector [batch size, coords]
            a: cities to swap ([batch size], [batch size])
        """
        return self.swap(s, a).long()

    def swap(self, x: torch.Tensor, a: torch.Tensor):
        """
        Vectorizes the inversion of values in `x` at the indices specified in `a`
        row by row.

        Args:
            x (torch.Tensor): A tensor of shape [n, dim, 1].
            a (torch.Tensor): A tensor of shape [n, 2] containing the indices to swap
            for each row of `x`.

        Returns:
            torch.Tensor: The tensor `x` with the specified swaps applied.
        """
        new_x = x.clone()  # To avoid modifying the original tensor

        # Row indices
        rows = torch.arange(new_x.size(0)).unsqueeze(1)  # Shape [n, 1]

        # Column indices specified by `a`
        idx1 = a[:, 0].unsqueeze(1)  # Shape [n]
        idx2 = a[:, 1].unsqueeze(1)  # Shape [n]

        # Extract values at the indices to swap
        temp = new_x[rows, idx1].clone()  # Shape [n, 1]
        new_x[rows, idx1] = new_x[rows, idx2]
        new_x[rows, idx2] = temp

        return new_x

    @property
    def state_encoding(self) -> torch.Tensor:
        return self.coords

    def get_coords(self, s: torch.Tensor) -> torch.Tensor:
        """Get coords from tour permutation."""
        s_expanded = s.unsqueeze(-1).expand(-1, -1, self.coords.size(-1))
        return torch.gather(self.coords, 1, s_expanded)

    def get_demands(self, s: torch.Tensor) -> torch.Tensor:
        """Get coords from tour permutation."""
        return torch.gather(self.demands, 1, s)

    def generate_init_x(self):
        """Generate initial greedy solutions for all batch demands."""
        perm = greedy_init_batch(self.demands, self.capacity).to(self.device)
        return perm

    def generate_init_state(self) -> torch.Tensor:
        """State encoding has dims

        [state enc] = [batch size, num items, concat]
        """
        perm = self.generate_init_x()
        return torch.cat([perm, self.state_encoding], -1)

    def get_edge_offsets_in_tour(self, s: torch.Tensor) -> torch.Tensor:
        """Compute vector to right city in tour

        Args:
            s: [batch size, dim]
        Returns:
            [batch size, dim, 2]
        """
        # Gather dataset in order of tour
        d = self.get_coords(s[..., 0])
        d_roll = torch.roll(d, -1, 1)
        # Edge lengths
        return d_roll - d

    def get_edge_lengths_in_tour(self, s: torch.Tensor) -> torch.Tensor:
        """Compute distance to right city in tour

        Args:
            s: [batch size, dim, 1]
        Returns:
            [batch size, dim]
        """
        # Edge offsets
        offset = self.get_edge_offsets_in_tour(s)
        # Edge lengths
        return torch.sqrt(torch.sum(offset**2, -1))

    def get_neighbors_in_tour(self, s: torch.Tensor) -> torch.Tensor:
        """Return distances to neighbors in tour.

        Args:
            s: [batch size, dim, 1] vector
        """
        right_distance = self.get_edge_lengths_in_tour(s)
        left_distance = torch.roll(right_distance, 1, 1)
        return torch.stack([right_distance, left_distance], -1)

    def get_arg_demands(self, x: torch.Tensor) -> torch.Tensor:
        """Return demands, grp, max route"""
        dem = self.get_demands(x[..., 0])
        agg = self.cost_demands(x, dem)
        grp = convert_tensor(agg)
        return dem, agg, grp

    def get_percentage_demands(self, x: torch.Tensor) -> torch.Tensor:
        """Return the percentage of the demands per nodes."""
        self.dem, self.agg, self.grp = self.get_arg_demands(x)

        per_nodes = torch.nan_to_num(self.dem / self.agg, nan=0.0)
        per_routes = self.agg / self.capacity
        per_nodes_routes = self.dem / self.capacity
        return (
            per_nodes.unsqueeze(-1),
            per_routes.unsqueeze(-1),
            per_nodes_routes.unsqueeze(-1),
        )

    def allowed_permutations(self, x: torch.Tensor, node: torch.Tensor) -> torch.Tensor:
        """Return all allowed permutations from a node."""
        # Create a mask for the conditions
        mask1 = (
            (self.dem > 0)
            # & (
            #     torch.arange(dem.shape[1]).repeat(dem.shape[0], 1).to(self.device)
            #     != node
            # )
            & (self.grp == self.grp.gather(1, node))
        )
        mask2 = (
            (self.dem > 0)
            # & (
            #     torch.arange(dem.shape[1]).repeat(dem.shape[0], 1).to(self.device)
            #     != node
            # )
            & (self.agg + self.dem.gather(1, node) - self.dem - self.capacity <= 0)
            & (
                self.agg.gather(1, node)
                + self.dem
                - self.dem.gather(1, node)
                - self.capacity
                <= 0
            )
        )

        # Combine the masks
        return mask1 | mask2

    def get_permutable_pairs(self, x: torch.Tensor, indices: Tuple) -> torch.Tensor:
        """
        Returns a boolean tensor indicating which pairs of nodes are swappable.

        Args:
            x (torch.Tensor): Tensor representing the solutions of the current problems.

        Returns:
            torch.Tensor: A boolean tensor of size (B, N, N) where
            True indicates that (i, j) is swappable.
        """
        batch_size = x.shape[0]

        # Initialize an output tensor (B, N+1, N+1)
        permutable = torch.zeros(
            (batch_size, indices[0].size(0)),
            dtype=torch.bool,
            device=x.device,
        )
        colonne = 0
        # Iterate over all pairs of nodes
        for i, j in zip(indices[0], indices[1]):

            # Condition 1: Both nodes must have a demand > 0
            valid_demand = (self.dem[:, i] > 0) & (self.dem[:, j] > 0)

            # Condition 2: Nodes must belong to the same group
            same_group = self.grp[:, i] == self.grp[:, j]

            # Condition 3: Check the capacity constraint after swapping
            valid_capacity = (
                self.agg[:, i] + self.dem[:, j] - self.dem[:, i] <= self.capacity
            ) & (self.agg[:, j] + self.dem[:, i] - self.dem[:, j] <= self.capacity)

            # A swap is possible if all conditions are met
            can_swap = valid_demand & (same_group | valid_capacity)

            # Append the result to the permutable tensor
            permutable[:, colonne] = can_swap
            colonne += 1

        return permutable
