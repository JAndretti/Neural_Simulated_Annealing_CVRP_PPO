# --------------------------------
# Import required libraries
# --------------------------------
from abc import ABC, abstractmethod  # Abstract base classes for problem definition
from typing import (
    Dict,
    Tuple,
    Optional,
    Union,
)  # Type hints for better code readability
import torch  # PyTorch for tensor operations
import torch.nn.functional as F
from loguru import logger  # Advanced logging utility

from utils import (
    repeat_to,
    calculate_client_angles,
    calculate_distance_matrix,
)
from algo import (
    generate_isolate_solution,
    generate_sweep_solution,
    random_init_batch,
    generate_Clark_and_Wright,
    generate_nearest_neighbor,
    construct_cvrp_solution,
    # vrp_optimal_split,
    cheapest_insertion,
    path_cheapest_arc,
    farthest_insertion,
)

init_methods = {
    "random": random_init_batch,
    "sweep": generate_sweep_solution,
    "isolate": generate_isolate_solution,
    "Clark_and_Wright": generate_Clark_and_Wright,
    "nearest_neighbor": generate_nearest_neighbor,
    "cheapest_insertion": cheapest_insertion,
    "path_cheapest_arc": path_cheapest_arc,
    "farthest_insertion": farthest_insertion,
}


class Problem(ABC):
    """
    Abstract base class defining the interface for optimization problems.

    This class provides the foundation for implementing various optimization problems
    by defining common operations and required abstract methods.
    """

    def __init__(self, device: str = "cpu") -> None:
        """
        Initialize the problem.

        Args:
            device: Computation device (cpu/cuda)
        """
        self.device = device
        self.generator = torch.Generator(device=device)

    def gain(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Calculate improvement gained by applying action to solution.

        Args:
            s: Current solution
            a: Action to apply

        Returns:
            Cost difference between current and updated solution
        """
        return self.cost(s) - self.cost(self.update(s, a))

    def manual_seed(self, seed: int) -> None:
        """
        Set random generator seed for reproducibility.

        Args:
            seed: Random seed value
        """
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

    @abstractmethod
    def cost(self, s: torch.Tensor) -> torch.Tensor:
        """
        Calculate cost of a solution.

        Args:
            s: Solution tensor

        Returns:
            Cost tensor
        """
        pass

    @abstractmethod
    def update(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Apply an action to modify a solution.

        Args:
            s: Current solution tensor
            a: Action tensor

        Returns:
            Updated solution tensor
        """
        pass

    @abstractmethod
    def set_params(self, **kwargs) -> None:
        """
        Set problem parameters.

        Args:
            **kwargs: Problem-specific parameters
        """
        pass

    @abstractmethod
    def generate_params(self) -> Dict[str, torch.Tensor]:
        """
        Generate problem parameters.

        Returns:
            Dictionary of problem parameters
        """
        pass

    @property
    def state_encoding(self) -> torch.Tensor:
        """
        Get problem's state encoding.

        Returns:
            Tensor representation of the problem state
        """
        pass

    @abstractmethod
    def generate_init_state(self) -> torch.Tensor:
        """
        Generate initial state for the problem.

        Returns:
            Initial state tensor
        """
        pass

    def to_state(self, *components: torch.Tensor) -> torch.Tensor:
        """
        Concatenate multiple state components into a single state tensor.

        Args:
            *components: Variable number of tensors representing different components
                        of the state. Each tensor should have the same shape except
                        for the last dimension, which will be concatenated.

        Returns:
            A single tensor resulting from concatenating all input components
            along the last dimension.
        """
        return torch.cat(components, dim=-1)

    def from_state(self, state: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Split state into components dynamically.

        Args:
            state: Combined state tensor

        Returns:
            Tuple of component tensors
        """
        num_extra_features = state.shape[-1] - 3  # Adjusting for variable dimensions
        split_sizes = [1, 2] + [1] * num_extra_features
        return tuple(torch.split(state, split_sizes, dim=-1))


class CVRP(Problem):
    """
    Capacitated Vehicle Routing Problem implementation.

    This class implements the CVRP, where a fleet of vehicles with limited capacity
    must serve customer demands while minimizing total route distance.
    """

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
            capacities: Vehicle capacity constraint(s)
            device: Computation device (cpu/cuda)
            params: Configuration parameters including:
                   - HEURISTIC: 'swap' or 'two_opt', etc.
                   - CLUSTERING: Whether to use clustered instances
                   - NB_CLUSTERS_MAX: Max clusters if clustering enabled
                   - UPDATE_METHOD: How to apply heuristics
                   - INIT: Initial solution generation method
        """
        super().__init__(device)
        self.params = params or {}
        self._init_problem_parameters(dim, n_problems, capacities)

    # --------------------------------
    # Initialization and Configuration
    # --------------------------------

    def _init_problem_parameters(self, dim, n_problems, capacities):
        """
        Initialize problem size and constraints.

        Args:
            dim: Problem dimension (number of clients)
            n_problems: Number of problem instances in batch
            capacities: Vehicle capacity constraint(s)
        """
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
        self.clustering = self.params.get("CLUSTERING", False)
        self.nb_clusters_max = self.params.get("NB_CLUSTERS_MAX", 3)

    def set_heuristic(self, heuristic: str, mix1: str = None, mix2: str = None) -> None:
        """
        Configure the heuristic method for solution modification.

        Args:
            heuristic: Type of heuristic ('swap', 'two_opt', 'insertion', 'mix')
            mix1: First heuristic type if using mixed approach
            mix2: Second heuristic type if using mixed approach

        Raises:
            ValueError: If unsupported heuristic specified
        """
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

    def set_params(self, params: Dict) -> None:
        """
        Update problem coordinates and demands.

        Args:
            params: Dictionary containing problem parameters
        """
        if "coords" in params:
            self.coords = params["coords"].to(self.device)
        if "demands" in params:
            self.demands = params["demands"].to(self.device)
        self._init_derived_features()

    def _init_derived_features(self):
        """
        Initialize derived features from problem coordinates.

        Computes angles, distance matrix, and normalized distances from depot.
        """
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

    def _set_demands_coords(self, coords: torch.Tensor, demands: torch.Tensor) -> Dict:
        """
        Set coordinates and demands for the problem from loaded data.

        Args:
            coords: Node coordinates tensor
            demands: Demand values tensor

        Returns:
            Dictionary containing coordinates and demands

        Raises:
            ValueError: If dimensions don't match expected batch size
        """
        if coords.shape[0] != self.n_problems:
            raise ValueError(
                f"Expected {self.n_problems} problems, got {coords.shape[0]}"
            )
        if demands.shape[0] != self.n_problems:
            raise ValueError(
                f"Expected {self.n_problems} problems, got {demands.shape[0]}"
            )
        return {"coords": coords, "demands": demands}

    # --------------------------------
    # Problem Instance Generation
    # --------------------------------

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
            self.manual_seed(0)  # Fixed seed for reproducibility in test mode

        if pb:  # Use provided problem data
            return self._set_demands_coords(coords, demands)

        if self.clustering:
            return self._generate_clustered_instances()
        else:
            return self._generate_random_instances()

    def _generate_clustered_instances(self) -> Dict[str, torch.Tensor]:
        """
        Generate problems with clustered customer locations.

        Creates customer nodes grouped around cluster centers with small random offset.

        Returns:
            Dictionary with coordinates and demands
        """
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
        """
        Generate completely random problem instances.

        Coordinates are uniformly distributed in the unit square.

        Returns:
            Dictionary with coordinates and demands
        """
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
        """
        Generate random customer demands (depot demand=0).

        Returns:
            Tensor of demand values
        """
        demands = torch.randint(
            1,
            10,
            (self.n_problems, self.dim + 1),
            device=self.device,
            generator=self.generator,
        )
        demands[:, 0] = 0  # Depot has no demand
        return demands

    # --------------------------------
    # State and Solution Representation
    # --------------------------------

    def build_state_components(self, x, temp, time):
        """
        Build state components for the model.

        Combines solution representation with problem features and metadata.

        Args:
            x: Solution tensor
            temp: Temperature parameter
            time: Time step information

        Returns:
            List of state component tensors
        """
        padding = max(0, x.size(1) - self.state_encoding.size(1))
        padded_coords = F.pad(self.state_encoding, (0, 0, 0, padding))
        is_depot = (x == 0).long()  # Identify depot visits

        return [
            x,  # Current solution
            padded_coords,  # Node coordinates
            is_depot,  # Depot indicator
            self.angles.gather(1, x),  # Angles in solution order
            self.dist_to_depot.gather(1, x),  # Distance to depot
            *self.get_percentage_demands(x),  # Demand percentages
            self.cost_per_route(x),  # Route segment costs
            repeat_to(temp, x),  # Temperature parameter
            repeat_to(time, x),  # Time information
        ]

    @property
    def state_encoding(self) -> torch.Tensor:
        """
        Get node coordinates as static problem features.

        Returns:
            Coordinates tensor
        """
        return self.coords

    def get_coords(self, solution: torch.Tensor) -> torch.Tensor:
        """
        Get coordinates in solution order.

        Args:
            solution: Solution tensor

        Returns:
            Ordered coordinates tensor
        """
        return torch.gather(
            self.coords, 1, solution.expand(-1, -1, self.coords.size(-1))
        )

    def get_demands(self, solution: torch.Tensor) -> torch.Tensor:
        """
        Get demands in solution order.

        Args:
            solution: Solution tensor

        Returns:
            Ordered demands tensor
        """
        return torch.gather(self.demands, 1, solution.squeeze(-1))

    def generate_init_state(self) -> torch.Tensor:
        """
        Generate initial state including coordinates.

        Returns:
            Initial state tensor
        """
        solution = self.generate_init_solution()
        return torch.cat([solution, self.state_encoding], -1)

    def generate_init_solution(self, param: str = None) -> torch.Tensor:
        """
        Generate initial solutions using specified algorithm or multiple methods
        if MULTI_INIT is enabled.
        """
        if param is not None:
            self.params["INIT"] = param

        if self.params.get("MULTI_INIT", False):
            split_size = self.n_problems // len(self.params["INIT_LIST"])
            solutions = []
            for i, method in enumerate(self.params["INIT_LIST"]):
                sol = init_methods[method](self).to(self.device)
                if i == len(self.params["INIT_LIST"]) - 1:
                    solutions.append(sol[i * split_size :, :, :])
                else:
                    solutions.append(sol[i * split_size : (i + 1) * split_size, :, :])
            max_size = max(sol.shape[1] for sol in solutions)
            solutions_padded = [
                F.pad(sol, (0, 0, 0, max_size - sol.shape[1])) for sol in solutions
            ]
            sol = torch.cat(solutions_padded, dim=0)
        else:
            method = self.params["INIT"]
            if method not in init_methods:
                raise ValueError(f"Unsupported initialization method: {method}")

            sol = init_methods[method](self).to(self.device)

        valid = self.is_feasible(sol).all()
        if not valid:
            logger.warning(
                "Generated initial solution is not feasible. "
                "Consider using a different initialization method."
            )
        return sol

    # --------------------------------
    # Cost Calculation
    # --------------------------------

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

    # --------------------------------
    # Solution Modification Heuristics
    # --------------------------------

    def update(self, solution: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Modify solution by applying heuristic action.

        Args:
            solution: Current solution tensor [batch, route_length, 1]
            action: Indices to modify [batch, 2] or [batch, 3] for mixed heuristic

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
            # sol = vrp_optimal_split(
            #     self.coords, self.demands, self.capacity, modified_sol.squeeze(-1)
            # )
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
        temp = solution[batch_idx, idx1].clone()  # Clone to avoid in-place issues
        solution[batch_idx, idx1] = solution[batch_idx, idx2]
        solution[batch_idx, idx2] = temp

        return solution

    # def two_opt(self, solution: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    #     """
    #     Perform 2-opt move by reversing segment between indices.

    #     Args:
    #         solution: Tensor [batch, num_nodes, 1]
    #         indices: Tensor [batch, 2] containing segment endpoints

    #     Returns:
    #         Solution with reversed segment
    #     """
    #     left = torch.minimum(indices[:, 0], indices[:, 1])
    #     right = torch.maximum(indices[:, 0], indices[:, 1])

    #     # Create reversed indices for the segment
    #     idx = torch.arange(solution.size(1), device=solution.device)
    #     idx = idx.repeat(solution.size(0), 1)
    #     reverse_mask = (idx >= left[:, None]) & (idx < right[:, None])
    #     reversed_idx = left[:, None] + right[:, None] - 1 - idx
    #     idx = torch.where(reverse_mask, reversed_idx, idx).to(torch.int64)

    #     return torch.gather(solution, 1, idx.unsqueeze(-1))

    def two_opt(self, x: torch.Tensor, a: torch.Tensor):
        """
        Perform 2-opt move by reversing segment between indices.

        Args:
            solution: Tensor [batch, num_nodes, 1]
            indices: Tensor [batch, 2] containing segment endpoints

        Returns:
            Solution with reversed segment
        """
        # Two-opt moves invert a section of a tour. If we cut a tour into
        # segments a and b then we can choose to invert either a or b. Due
        # to the linear representation of a tour, we choose always to invert
        # the segment that is stored contiguously.
        left = torch.minimum(a[:, 0], a[:, 1])
        right = torch.maximum(a[:, 0], a[:, 1])
        ones = torch.ones((self.n_problems, 1), dtype=torch.long, device=self.device)
        fidx = torch.arange(self.dim, device=self.device) * ones
        # Reversed indices
        offset = left + right - 1
        ridx = torch.arange(0, -self.dim, -1, device=self.device) + offset[:, None]
        # Set flipped section to all True
        flip = torch.ge(fidx, left[:, None]) * torch.lt(fidx, right[:, None])
        # Set indices to replace flipped section with
        idx = (~flip) * fidx + flip * ridx
        # Perform 2-opt move
        return torch.gather(x, 1, idx.unsqueeze(-1))

    def insertion(self, solution: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Move a node to a new position in the solution.

        Args:
            solution: Tensor [batch, route_length, 1]
            indices: Tensor [batch, 2] -> (node_position, new_position)

        Returns:
            Modified solution tensor
        """
        batch_size, route_length, _ = solution.shape
        device = solution.device

        # Remove last dimension and ensure proper type
        solution = solution.squeeze(-1)
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
        Apply either the first or second heuristic based on action value.

        Args:
            solution: Tensor [batch, num_nodes, 1]
            action: Tensor [batch, 3] where:
                - action[:, 0:2] are the indices
                - action[:, 2] determines heuristic (heur_1 if <0.5,heur_2 otherwise)

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

    # --------------------------------
    # Demand and Capacity Analysis
    # --------------------------------

    def _compute_route_demands(
        self, demands: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute total demands per route segment.

        Args:
            demands: Demand values tensor

        Returns:
            Tuple of (segment_demands, segment_ids)
        """
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

        Args:
            solution: Solution tensor

        Returns:
            Tuple of three percentage tensors
        """
        demands = self.get_demands(solution)
        route_demands, route_ids = self._compute_route_demands(demands)

        node_pct = torch.nan_to_num(demands / route_demands, nan=0.0)
        route_pct = route_demands / self.capacity
        node_cap_pct = demands / self.capacity

        return (
            node_pct.unsqueeze(-1),
            route_pct.unsqueeze(-1),
            node_cap_pct.unsqueeze(-1),
        )

    def capacity_utilization(self, solution: torch.Tensor) -> torch.Tensor:
        """
        Calculate capacity utilization score for each solution.

        For each route in a solution, computes the ratio of total demand
        to vehicle capacity, then averages these ratios across all routes.
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
        Check if the solution respects capacity constraints for all routes.

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
