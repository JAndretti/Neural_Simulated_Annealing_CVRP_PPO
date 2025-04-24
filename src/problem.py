from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import torch
import torch.nn.functional as F
from utils import repeat_to


def greedy_init_batch(demands: torch.Tensor, capacity: int) -> torch.Tensor:
    """
    Vectorized greedy algorithm for initial CVRP solution generation.

    Args:
        demands: Tensor of shape [batch_size, num_nodes] representing client demands
        capacity: Vehicle capacity constraint

    Returns:
        Tensor of shape [batch_size, max_route_length, 1] containing initial routes
        with depot visits inserted to respect capacity constraints
    """
    batch_size, num_nodes = demands.size()
    device = demands.device

    # Initialize solution tensor with 40% padding for depot insertions
    max_route_length = num_nodes + int(num_nodes * 0.40)
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
    remaining_capacity = torch.full((batch_size,), capacity, device=device)
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
    max_route_length = num_nodes + int(num_nodes * 0.40)
    routes = torch.zeros(batch_size, max_route_length, dtype=torch.long, device=device)

    # Add depot at start of each route
    x = torch.cat(
        [torch.zeros(batch_size, 1, dtype=torch.long, device=device), x], dim=1
    )

    # Get demands in current route order
    ordered_demands = torch.gather(demands, 1, x)

    # Initialize tracking variables
    remaining_capacity = torch.full((batch_size,), capacity, device=device)
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
        """
        Construct full state representation by combining:
        - Current solution (x)
        - Problem features (state_encoding)
        - Demand percentages
        - Route costs
        - Temperature and time parameters
        """
        padding = max(0, x.size(1) - self.state_encoding.size(1))
        padded_features = F.pad(self.state_encoding, (0, 0, 0, padding))

        components = [
            x,
            padded_features,
            *self.get_percentage_demands(x),
            self.cost_per_route(x),
            repeat_to(temp, x),
            repeat_to(time, x),
        ]

        return torch.cat(components, -1)

    def from_state(self, state: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Split state into components dynamically."""
        num_extra_features = state.shape[-1] - 3  # Adjusting for variable dimensions
        split_sizes = [1, 2] + [1] * num_extra_features
        return tuple(torch.split(state, split_sizes, dim=-1))


class CVRP(Problem):
    """Capacitated Vehicle Routing Problem implementation."""

    x_dim = 1  # Dimension for solution representation

    def __init__(
        self,
        dim: int = 50,
        n_problems: int = 256,
        capacity: int = 40,
        device: str = "cpu",
        params: Optional[Dict] = None,
    ):
        """
        Initialize CVRP instance.

        Args:
            dim: Number of client nodes (excluding depot)
            n_problems: Batch size for parallel processing
            capacity: Vehicle capacity constraint
            device: Computation device (cpu/cuda)
            params: Configuration parameters including:
                   - HEURISTIC: 'swap' or 'two_opt'
                   - CLUSTERING: Whether to use clustered instances
                   - NB_CLUSTERS_MAX: Max clusters if clustering enabled
        """
        super().__init__(device)
        self.params = params or {}
        self.set_heuristic(self.params["HEURISTIC"])
        self._init_problem_parameters(dim, n_problems, capacity)

    def set_heuristic(self, heuristic: str) -> None:
        """Set the heuristic for modifying solutions."""
        if heuristic == "swap":
            self.heuristic = self.swap
        elif heuristic == "two_opt":
            self.heuristic = self.two_opt
        elif heuristic == "insertion":
            self.heuristic = self.insertion
        elif heuristic == "mix":
            self.heuristic = self.mixed_heuristic
        else:
            raise ValueError(f"Unsupported heuristic: {heuristic}")

    def _init_problem_parameters(self, dim, n_problems, capacity):
        """Initialize problem size and constraints."""
        if self.params["name"] is not None:
            # Load from predefined problem
            self.n_problems = 1
            self.dim = self.params["dimension"] - 1
            self.n_problems = 1
            self.capacity = self.params["capacity"]
            self.clustering = False
        else:
            # Initialize random problem
            self.dim = dim
            self.n_problems = n_problems
            self.capacity = capacity
            self.clustering = self.params["CLUSTERING"]
            if self.clustering:
                self.nb_clusters_max = self.params["NB_CLUSTERS_MAX"]

    def set_params(self, params: Dict) -> None:
        """Update problem coordinates and demands."""
        if "coords" in params:
            self.coords = params["coords"]
        if "demands" in params:
            self.demands = params["demands"]

    def generate_params(
        self, mode: str = "train", pb: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Generate problem instances with optional clustering.

        Args:
            mode: 'train' or 'test' (affects random seed)
            pb: Whether to load predefined problem

        Returns:
            Dictionary containing:
            - coords: Node coordinates [batch, num_nodes+1, 2]
            - demands: Node demands [batch, num_nodes+1] (depot demand=0)
        """
        if mode == "test":
            self.manual_seed(0)  # Fixed seed for reproducibility

        if pb:
            return self._load_predefined_problem()

        if self.clustering:
            return self._generate_clustered_instances()
        else:
            return self._generate_random_instances()

    def _load_predefined_problem(self) -> Dict[str, torch.Tensor]:
        """Load problem from predefined parameters."""
        coords = (
            torch.from_numpy(self.params["node_coord_normalized"])
            .unsqueeze(0)
            .to(self.device, dtype=torch.float32)
        )
        demands = torch.from_numpy(self.params["demand"]).unsqueeze(0).to(self.device)
        demands[:, 0] = 0  # Depot has no demand
        return {"coords": coords, "demands": demands}

    def _generate_clustered_instances(self) -> Dict[str, torch.Tensor]:
        """Generate problems with clustered customer locations."""
        # Create cluster centers
        centers = torch.rand(
            self.n_problems,
            self.nb_clusters_max,
            2,
            device=self.device,
            generator=self.generator,
        )

        # Assign nodes to clusters
        cluster_assign = torch.randint(
            0,
            self.nb_clusters_max,
            (self.n_problems, self.dim + 1),
            device=self.device,
            generator=self.generator,
        )

        # Generate coordinates with cluster-based noise
        coords = centers[torch.arange(self.n_problems)[:, None], cluster_assign]
        coords += 0.05 * torch.randn(
            self.n_problems,
            self.dim + 1,
            2,
            device=self.device,
            generator=self.generator,
        )
        coords = torch.clamp(coords, 0, 1)  # Keep in unit square

        # Generate demands
        demands = self._generate_demands()
        return {"coords": coords, "demands": demands}

    def _generate_random_instances(self) -> Dict[str, torch.Tensor]:
        """Generate completely random problem instances."""
        coords = torch.rand(
            self.n_problems,
            self.dim + 1,
            2,
            device=self.device,
            generator=self.generator,
        )
        demands = self._generate_demands()
        return {"coords": coords, "demands": demands}

    def _generate_demands(self) -> torch.Tensor:
        """Generate random customer demands (depot demand=0)."""
        demands = torch.randint(
            1,
            10,
            (self.n_problems, self.dim + 1),
            device=self.device,
            generator=self.generator,
        )
        demands[:, 0] = 0  # Depot has no demand
        return demands

    def cost(self, solution: torch.Tensor) -> torch.Tensor:
        """
        Compute total route length for given solution.

        Args:
            solution: Tensor [batch, route_length, 1] representing routes

        Returns:
            Tensor [batch] containing total distance for each solution
        """
        return torch.sum(self.get_edge_lengths_in_tour(solution), -1)

    def cost_per_route(self, solution: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized cost contribution per route segment.

        Args:
            solution: Tensor [batch, route_length, 1] representing routes

        Returns:
            Tensor [batch, route_length, 1] with normalized route costs
        """
        edge_lengths = self.get_edge_lengths_in_tour(solution)
        total_cost = torch.sum(edge_lengths, -1, keepdim=True)

        # Create route segment masks
        mask = solution.squeeze(-1) != 0
        segment_start = mask & ~torch.cat(
            [torch.zeros_like(mask[:, :1]), mask[:, :-1]], dim=1
        )

        # Compute segment costs
        segment_ids = torch.cumsum(segment_start, 1) * mask
        segment_sums = torch.zeros_like(edge_lengths)
        segment_sums.scatter_add_(1, segment_ids, edge_lengths)
        route_costs = segment_sums.gather(1, segment_ids) * mask

        return (route_costs / total_cost).unsqueeze(-1)

    def update(self, solution: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Modify solution by applying heuristic action.

        Args:
            solution: Current solution tensor [batch, route_length, 1]
            action: Pair of indices to modify [batch, 2]

        Returns:
            New solution after applying heuristic
        """
        # Remove depot visits for processing
        mask = solution.squeeze(-1) != 0
        compact_sol = solution[mask].view(solution.size(0), -1, solution.size(-1))

        # Apply selected heuristic
        modified_sol = self.heuristic(compact_sol, action).long()

        # Rebuild valid CVRP solution with depot visits
        return construct_cvrp_solution(modified_sol, self.demands, self.capacity)

    def swap(self, solution: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Swap two nodes in the solution.

        Args:
            solution: Tensor [batch, num_nodes, 1]
            indices: Tensor [batch, 2] containing swap indices

        Returns:
            Solution with swapped nodes
        """
        new_solution = solution.clone()
        batch_idx = torch.arange(solution.size(0))[:, None]
        idx1, idx2 = indices[:, 0, None].to(torch.int64), indices[:, 1, None].to(
            torch.int64
        )

        # Perform swap
        temp = new_solution[batch_idx, idx1].clone()
        new_solution[batch_idx, idx1] = new_solution[batch_idx, idx2]
        new_solution[batch_idx, idx2] = temp

        return new_solution

    def two_opt(self, solution: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Perform 2-opt move by reversing segment between indices.

        Args:
            solution: Tensor [batch, num_nodes, 1]
            indices: Tensor [batch, 2] containing segment endpoints

        Returns:
            Solution with reversed segment
        """
        left = torch.minimum(indices[:, 0], indices[:, 1])
        right = torch.maximum(indices[:, 0], indices[:, 1])

        # Create reversed indices for the segment
        idx = torch.arange(solution.size(1), device=solution.device)
        idx = idx.repeat(solution.size(0), 1)
        reverse_mask = (idx >= left[:, None]) & (idx < right[:, None])
        reversed_idx = left[:, None] + right[:, None] - 1 - idx
        idx = torch.where(reverse_mask, reversed_idx, idx).to(torch.int64)

        return torch.gather(solution, 1, idx.unsqueeze(-1))

    def insertion(self, solution: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Vectorized version of the insertion operation.

        solution: Tensor [batch, route_length, 1]
        indices: Tensor [batch, 2] -> (node_position, new_position)
        Returns: Tensor [batch, route_length, 1]
        """
        batch_size, route_length, _ = solution.shape
        device = solution.device

        # Remove last dimension and clamp new_pos
        solution = solution.squeeze(-1)  # Ensure float dtype
        node_pos = indices[:, 0]
        new_pos = indices[:, 1].clamp(0, route_length - 1)

        # Create batch indices [0, 1, ..., batch_size-1]
        batch_idx = torch.arange(batch_size, device=device)

        # Create mask for all elements except the ones to move
        mask = torch.ones_like(solution, dtype=torch.bool)
        mask[batch_idx, node_pos] = False

        # Get remaining nodes after removing the moved ones
        remaining_nodes = solution[mask].view(batch_size, route_length - 1)

        # Create new solution tensor with proper dtype
        new_solution = torch.zeros(
            batch_size, route_length, dtype=solution.dtype, device=device
        )

        # Create position ranges
        pos_range = torch.arange(route_length, device=device).expand(batch_size, -1)

        # Build the new solution by scattering
        # For positions before new_pos, take from remaining_nodes
        new_solution[pos_range < new_pos.unsqueeze(1)] = remaining_nodes[
            :, : route_length - 1
        ][pos_range[:, : route_length - 1] < new_pos.unsqueeze(1)]

        # For positions after new_pos, take from remaining_nodes offset by 1
        new_solution[pos_range > new_pos.unsqueeze(1)] = remaining_nodes[
            :, : route_length - 1
        ][pos_range[:, : route_length - 1] >= new_pos.unsqueeze(1)]

        # Insert the moved nodes at their new positions
        new_solution[batch_idx, new_pos] = solution[batch_idx, node_pos]

        return new_solution.unsqueeze(-1)

    def mixed_heuristic(
        self, solution: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply either swap or two_opt based on the third value in action.

        Args:
            solution: Tensor [batch, num_nodes, 1]
            action: Tensor [batch, 3] where:
                   - action[:, 0:2] are the indices
                   - action[:, 2] determines heuristic (swap if <0.5, two_opt otherwise)

        Returns:
            Modified solution
        """
        indices = action[:, :2]
        heuristic_choice = action[:, 2] >= 0.5  # Convert to boolean

        # Apply heur_1 where choice < 0.5
        heur_1 = self.swap(solution, indices)
        # Apply heur_2 where choice >= 0.5
        heur_2 = self.insertion(solution, indices)

        # Combine results based on heuristic choice
        return torch.where(
            heuristic_choice.unsqueeze(-1).unsqueeze(-1),
            heur_2,
            heur_1,
        )

    @property
    def state_encoding(self) -> torch.Tensor:
        """Node coordinates as static problem features."""
        return self.coords

    def get_coords(self, solution: torch.Tensor) -> torch.Tensor:
        """Get coordinates in solution order."""
        return torch.gather(
            self.coords, 1, solution.expand(-1, -1, self.coords.size(-1))
        )

    def get_demands(self, solution: torch.Tensor) -> torch.Tensor:
        """Get demands in solution order."""
        return torch.gather(self.demands, 1, solution.squeeze(-1))

    def generate_init_x(self) -> torch.Tensor:
        """Generate initial solutions using greedy algorithm."""
        return greedy_init_batch(self.demands, self.capacity).to(self.device)

    def generate_init_state(self) -> torch.Tensor:
        """Generate initial state including coordinates."""
        solution = self.generate_init_x()
        return torch.cat([solution, self.state_encoding], -1)

    def get_edge_lengths_in_tour(self, solution: torch.Tensor) -> torch.Tensor:
        """
        Compute Euclidean distances between consecutive nodes in solution.

        Args:
            solution: Tensor [batch, num_nodes, 1]

        Returns:
            Tensor [batch, num_nodes] of inter-node distances
        """
        coords = self.get_coords(solution)
        next_coords = torch.cat([coords[:, 1:, :], coords[:, :1, :]], dim=1)
        return (coords - next_coords).norm(p=2, dim=-1)

    def _compute_route_demands(
        self, demands: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute total demands per route segment."""
        # Identify route segments
        mask = demands != 0
        segment_start = mask & ~torch.cat(
            [torch.zeros_like(mask[:, :1]), mask[:, :-1]], dim=1
        )

        # Compute segment demands
        segment_ids = torch.cumsum(segment_start, 1) * mask
        segment_demands = torch.zeros_like(demands)
        segment_demands.scatter_add_(1, segment_ids, demands)

        return segment_demands.gather(1, segment_ids), segment_ids

    def get_percentage_demands(
        self, solution: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        Compute demand-related percentages:
        1. Node demand as fraction of route demand
        2. Route demand as fraction of vehicle capacity
        3. Node demand as fraction of vehicle capacity
        """
        demands = self.get_demands(solution)
        route_demands, route_ids = self._compute_route_demands(demands)

        node_pct = torch.nan_to_num(demands / route_demands, nan=0.0)
        route_pct = route_demands / self.capacity
        node_cap_pct = route_ids / self.capacity
        return (
            node_pct.unsqueeze(-1),
            route_pct.unsqueeze(-1),
            node_cap_pct.unsqueeze(-1),
        )

    # def allowed_permutations(self, x: torch.Tensor, node: torch.Tensor)
    # -> torch.Tensor:
    #     """Return all allowed permutations from a node."""
    #     # Create a mask for the conditions
    #     mask1 = (
    #         (self.dem > 0)
    #         # & (
    #         #     torch.arange(dem.shape[1]).repeat(dem.shape[0], 1).to(self.device)
    #         #     != node
    #         # )
    #         & (self.grp == self.grp.gather(1, node))
    #     )
    #     mask2 = (
    #         (self.dem > 0)
    #         # & (
    #         #     torch.arange(dem.shape[1]).repeat(dem.shape[0], 1).to(self.device)
    #         #     != node
    #         # )
    #         & (self.agg + self.dem.gather(1, node) - self.dem - self.capacity <= 0)
    #         & (
    #             self.agg.gather(1, node)
    #             + self.dem
    #             - self.dem.gather(1, node)
    #             - self.capacity
    #             <= 0
    #         )
    #     )

    #     # Combine the masks
    #     return mask1 | mask2

    # def get_permutable_pairs(self, x: torch.Tensor, indices: Tuple) -> torch.Tensor:
    #     """
    #     Returns a boolean tensor indicating which pairs of nodes are swappable.

    #     Args:
    #         x (torch.Tensor): Tensor representing the solutions of the current
    #   problems.

    #     Returns:
    #         torch.Tensor: A boolean tensor of size (B, N, N) where
    #         True indicates that (i, j) is swappable.
    #     """
    #     dem_i = self.dem.gather(1, indices[0].unsqueeze(0))
    #     dem_j = self.dem.gather(1, indices[1].unsqueeze(0))
    #     grp_i = self.grp.gather(1, indices[0].unsqueeze(0))
    #     grp_j = self.grp.gather(1, indices[1].unsqueeze(0))

    #     # Calcul vectorisé
    #     valid_demand = (dem_i > 0) & (dem_j > 0)
    #     same_group = grp_i == grp_j
    #     valid_capacity = ((self.agg + dem_j - dem_i) <= self.capacity) & (
    #         (self.agg.gather(1, indices[1].unsqueeze(0)) + dem_i - dem_j)
    #         <= self.capacity
    #     )

    #     return valid_demand & (same_group | valid_capacity)
