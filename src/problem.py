# --------------------------------
# Import required libraries
# --------------------------------
from abc import ABC, abstractmethod  # Abstract base classes for problem definition
from typing import Dict, Tuple, Optional  # Type hints for better code readability
import torch  # PyTorch for tensor operations
import torch.nn.functional as F  # PyTorch functional API for operations
from utils import repeat_to  # Utility function for tensor expansion
from loguru import logger  # Advanced logging utility
from typing import Union

from algo import (
    generate_isolate_solution,
    generate_sweep_solution,
    random_init_batch,
    generate_Clark_and_Wright,
    generate_nearest_neighbor,
    construct_cvrp_solution,
    cheapest_insertion,
    path_cheapest_arc,
    farthest_insertion,
)


def calculate_client_angles(coords: torch.Tensor) -> torch.Tensor:
    """
    Calculate polar angles of clients with respect to the depot for each problem.

    The depot is assumed to be at position 0 for each problem in the batch.
    Angles are calculated in radians in the range [-π, π] using the arctangent function.

    Args:
        coords: Tensor of shape [batch_size, num_nodes, 2] containing coordinates
               where coords[:, 0, :] represents depot coordinates

    Returns:
        Tensor of shape [batch_size, num_nodes-1] containing angles in radians
        for each client (excluding depot) relative to the depot
    """
    batch_size, num_nodes, _ = coords.shape
    device = coords.device

    # Extract depot coordinates (first node in each problem)
    depot_coords = coords[:, 0:1, :]  # [batch_size, 1, 2]

    # Extract client coordinates (all nodes except depot)
    client_coords = coords[:, 1:, :]  # [batch_size, num_nodes-1, 2]

    # Calculate vectors from depot to each client
    delta_coords = client_coords - depot_coords  # [batch_size, num_nodes-1, 2]

    # Calculate polar angles (in radians) for each client relative to depot
    # atan2(y, x) gives angle in radians in range [-π, π] and then normalize to [0, 1]
    angles = (
        torch.atan2(delta_coords[:, :, 1], delta_coords[:, :, 0]) / (2 * torch.pi) + 0.5
    ).to(device)
    # Concatenate a column of zeros at the beginning to represent the depot's angle (0)
    angles = torch.cat(
        (torch.zeros(batch_size, 1, device=device), angles), dim=1
    )  # Shape: [batch_size, num_nodes]
    return angles.unsqueeze(-1)  # Shape: [batch_size, num_nodes, 1]


def calculate_distance_matrix(coords: torch.Tensor) -> torch.Tensor:
    """
    Calculate the distance matrix between all nodes for each problem in the batch.

    Args:
        coords: Tensor of shape [batch_size, num_nodes, 2] containing coordinates
               of nodes for each problem in the batch

    Returns:
        Tensor of shape [batch_size, num_nodes, num_nodes] containing the Euclidean
        distances between all pairs of nodes for each problem
    """
    batch_size, num_nodes, _ = coords.shape
    device = coords.device

    # Expand coordinates for broadcasting
    # [batch_size, num_nodes, 1, 2]
    coords_expanded_1 = coords.unsqueeze(2)
    # [batch_size, 1, num_nodes, 2]
    coords_expanded_2 = coords.unsqueeze(1)

    # Calculate squared Euclidean distances between all pairs of nodes
    # Result shape: [batch_size, num_nodes, num_nodes]
    distances = torch.sqrt(
        torch.sum((coords_expanded_1 - coords_expanded_2) ** 2, dim=-1)
    )

    return distances.to(device)


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
        self,
        *components: torch.Tensor,
    ) -> torch.Tensor:
        """
        Concatenates multiple state components into a single state tensor.

        Args:
            *components (torch.Tensor): Variable number of tensors representing
            different components of the state. Each tensor should have the same shape
            except for the last dimension, which will be concatenated.

        Returns:
            torch.Tensor: A single tensor resulting from concatenating all input
            components along  the last dimension.


        Note:
            The input tensors are expected to be the output of the
            `build_state_components` function, which returns a tuple of tensors
            representing different aspects of the problem state.
        """

        return torch.cat(components, dim=-1)

    def from_state(self, state: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Split state into components dynamically."""
        num_extra_features = state.shape[-1] - 3  # Adjusting for variable dimensions
        split_sizes = [1, 2] + [1] * num_extra_features
        return tuple(torch.split(state, split_sizes, dim=-1))  # TODO Verify function


class CVRP(Problem):
    """Capacitated Vehicle Routing Problem implementation."""

    x_dim = 1  # Dimension for solution representation

    def __init__(
        self,
        dim: int = 50,
        n_problems: int = 256,
        capacities: Union[int, torch.Tensor] = 30,
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
        self._init_problem_parameters(dim, n_problems, capacities)

    def set_heuristic(self, heuristic: str, mix1: str = None, mix2: str = None) -> None:
        """Set the heuristic for modifying solutions."""
        if heuristic == "swap":
            self.heuristic = self.swap
        elif heuristic == "two_opt":
            self.heuristic = self.two_opt
        elif heuristic == "insertion":
            self.heuristic = self.insertion
        elif heuristic == "mix":
            self.heuristic = self.mixed_heuristic
            heuristics = {
                "swap": self.swap,
                "two_opt": self.two_opt,
                "insertion": self.insertion,
            }
            self.heuristic_1 = heuristics.get(mix1)
            self.heuristic_2 = heuristics.get(mix2)
            if self.heuristic_1 is None or self.heuristic_2 is None:
                raise ValueError(f"Unsupported mix heuristics: {mix1}, {mix2}")
        else:
            raise ValueError(f"Unsupported heuristic: {heuristic}")

    def _init_problem_parameters(self, dim, n_problems, capacities):
        """Initialize problem size and constraints."""
        self.n_problems = n_problems
        self.dim = dim
        if isinstance(capacities, int):
            self.capacity = torch.full(
                (self.n_problems, 1),
                capacities,
                device=self.device,
                dtype=torch.float32,
            )
        elif isinstance(capacities, torch.Tensor):
            self.capacity = capacities.to(self.device)
        else:
            raise ValueError("capacities must be either an int or a torch.Tensor")
        self.clustering = self.params["CLUSTERING"]
        self.nb_clusters_max = self.params.get("NB_CLUSTERS_MAX", 3)

    def _set_demands_coords(self, coords: torch.Tensor, demands: torch.Tensor) -> None:
        """Set coordinates and demands for the problem if loaded problem."""
        if coords.shape[0] != self.n_problems:
            raise ValueError(
                f"Expected {self.n_problems} problems, got {coords.shape[0]}"
            )
        if demands.shape[0] != self.n_problems:
            raise ValueError(
                f"Expected {self.n_problems} problems, got {demands.shape[0]}"
            )
        return {"coords": coords, "demands": demands}

    def set_params(self, params: Dict) -> None:
        """Update problem coordinates and demands."""
        if "coords" in params:
            self.coords = params["coords"].to(self.device)
        if "demands" in params:
            self.demands = params["demands"].to(self.device)
        self._init_some_features()

    def _init_some_features(self):
        """Initialize some derived features for the problem."""
        self.angles = calculate_client_angles(self.coords)
        self.matrix = calculate_distance_matrix(self.coords)
        # Calculate distances from depot to clients and normalize row-wise to [0,1]
        self.dist_to_depot = self.matrix[:, 0, 0:]  # Distances from depot to clients
        # Find min and max values per batch for normalization
        min_dist = torch.min(self.dist_to_depot, dim=1, keepdim=True)[0]
        max_dist = torch.max(self.dist_to_depot, dim=1, keepdim=True)[0]
        # Avoid division by zero
        divisor = torch.clamp(max_dist - min_dist, min=1e-10)
        # Normalize each row to [0,1] range
        self.dist_to_depot = ((self.dist_to_depot - min_dist) / divisor).unsqueeze(-1)

    def generate_params(
        self,
        mode: str = "train",
        pb: bool = False,
        coords: torch.Tensor = None,
        demands: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate problem instances with optional clustering.

        Args:
            mode: 'train' or 'test' (affects random seed)
            pb: Whether to load predefined problem
            coords: Optional coordinates tensor
            demands: Optional demands tensor

        Returns:
            Dictionary containing:
            - coords: Node coordinates [batch, num_nodes+1, 2]
            - demands: Node demands [batch, num_nodes+1] (depot demand=0)
        """
        if mode == "test":
            self.manual_seed(0)  # Fixed seed for reproducibility

        if pb:
            return self._set_demands_coords(coords, demands)

        if self.clustering:
            return self._generate_clustered_instances()
        else:
            return self._generate_random_instances()

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

    def build_state_components(self, x, temp, time):
        """
        Build state components for the model.
        """
        padding = max(0, x.size(1) - self.state_encoding.size(1))
        padded_coords = F.pad(self.state_encoding, (0, 0, 0, padding))
        is_depot = (x == 0).long()  # Identify depot visits
        # Ensure padded_coords and padded_angles have the same shape
        return [
            x,
            padded_coords,
            is_depot,
            self.angles.gather(1, x),
            self.dist_to_depot.gather(1, x),
            *self.get_percentage_demands(x),
            self.cost_per_route(x),
            repeat_to(temp, x),
            repeat_to(time, x),
        ]

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
            Tuple containing:
            - New solution after applying heuristic (or original if infeasible)
            - Validity flag for each solution in batch
        """
        # Apply the selected update method
        if self.params["UPDATE_METHOD"] == "rm_depot":
            # First approach: Remove depot visits to simplify operations
            # 1. Create mask identifying non-depot nodes
            mask = solution.squeeze(-1) != 0

            # 2. Extract only client nodes, reshaping to maintain batch dimension
            compact_sol = solution[mask].view(solution.size(0), -1, solution.size(-1))

            # 3. Apply heuristic on client-only solution
            modified_sol = self.heuristic(compact_sol, action).long()

            # 4. Rebuild valid CVRP solution with depot visits inserted where needed
            sol = construct_cvrp_solution(modified_sol, self.demands, self.capacity)
            # Add padding to match the shape of the original solution
            padding_size = solution.size(1) - sol.size(1)
            if padding_size > 0:
                padding = torch.zeros(
                    sol.size(0),
                    padding_size,
                    sol.size(2),
                    dtype=sol.dtype,
                    device=sol.device,
                )
                sol = torch.cat([sol, padding], dim=1)

        elif self.params["UPDATE_METHOD"] == "free":
            # Second approach: Apply heuristic directly on full solution
            # (including depot visits)
            sol = self.heuristic(solution, action).long()
            # Note: This approach may require post-processing to ensure feasibility

        # Check if the modified solutions are feasible (respect capacity constraints)
        valid = self.is_feasible(sol).unsqueeze(-1).long()

        # Return original solution if modified solution is infeasible
        sol = torch.where(valid.unsqueeze(-1) == 1, sol, solution).to(torch.int64)

        return sol, valid

    def swap(self, solution: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Swap two nodes in the solution.

        Args:
            solution: Tensor [batch, num_nodes, 1]
            indices: Tensor [batch, 2] containing swap indices

        Returns:
            Solution with swapped nodes
        """
        batch_idx = torch.arange(solution.size(0))[:, None]
        idx1, idx2 = indices[:, 0, None].to(torch.int64), indices[:, 1, None].to(
            torch.int64
        )

        # Perform swap
        temp = solution[batch_idx, idx1]
        solution[batch_idx, idx1] = solution[batch_idx, idx2]
        solution[batch_idx, idx2] = temp

        return solution

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
        heur_1 = self.heuristic_1(solution, indices)
        # Apply heur_2 where choice >= 0.5
        heur_2 = self.heuristic_2(solution, indices)

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

    def generate_init_x(self, param: str = None) -> torch.Tensor:
        """Generate initial solutions using specified algorithm."""
        if param is not None:
            self.params["INIT"] = param
        if self.params["INIT"] == "random":
            sol = random_init_batch(self).to(self.device)
        elif self.params["INIT"] == "sweep":
            sol = generate_sweep_solution(self).to(self.device)
        elif self.params["INIT"] == "isolate":
            sol = generate_isolate_solution(self).to(self.device)
        elif self.params["INIT"] == "Clark_and_Wright":
            sol = generate_Clark_and_Wright(self).to(self.device)
        elif self.params["INIT"] == "nearest_neighbor":
            sol = generate_nearest_neighbor(self).to(self.device)
        elif self.params["INIT"] == "cheapest_insertion":
            sol = cheapest_insertion(self).to(self.device)
        elif self.params["INIT"] == "path_cheapest_arc":
            sol = path_cheapest_arc(self).to(self.device)
        elif self.params["INIT"] == "farthest_insertion":
            sol = farthest_insertion(self).to(self.device)
        else:
            raise ValueError(
                f"Unsupported initialization method: {self.params['INIT']}"
            )
        valid = self.is_feasible(sol).all()
        if valid is False:
            logger.warning(
                "Generated initial solution is not feasible. "
                "Consider using a different initialization method."
            )
        return sol

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

    def capacity_utilization(self, solution: torch.Tensor) -> torch.Tensor:
        """
        Calculate capacity utilization score for each solution.

        For each route in a solution, computes the ratio of total demand
        to vehicle capacity,
        then averages these ratios across all routes to get a score between 0 and 1.
        Higher scores indicate better capacity utilization.

        Args:
            solution: Tensor [batch_size, route_length, 1] representing routes

        Returns:
            Tensor [batch_size] with utilization score for each solution
        """
        batch_size = solution.size(0)
        device = solution.device

        demands = self.get_demands(solution)  # [batch, route_length]
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
        route_utilization = route_demands / self.capacity

        # Get sum of utilization across all routes
        total_utilization = torch.sum(route_utilization, dim=1)

        # Compute average utilization per route
        avg_utilization = total_utilization / routes_count

        return 1 - avg_utilization

    def is_feasible(self, solution: torch.Tensor) -> torch.Tensor:
        """
        Vectorized check if the solution respects capacity constraints for all routes.

        Args:
            solution: Tensor [batch, route_length, 1] representing routes

        Returns:
            Tensor [batch] of boolean values indicating feasibility for each problem
        """
        batch_size = solution.size(0)
        device = solution.device

        demands = self.get_demands(solution)  # [batch, route_length]
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
        feasible = (route_demands <= self.capacity).all(dim=1)
        return feasible
