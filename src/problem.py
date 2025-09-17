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

from utils import (
    repeat_to,
    calculate_client_angles,
    calculate_distance_matrix,
    is_feasible,
)
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
    swap,
    two_opt,
    insertion,
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

    def set_heuristic(self, heuristic: list) -> None:
        """
        Configure the heuristic method for solution modification.

        Args:
            heuristic: Type of heuristic ('swap', 'two_opt', 'insertion')

        Raises:
            ValueError: If unsupported heuristic specified
        """
        self.heuristic = None
        self.heuristic_1 = None
        self.heuristic_2 = None
        if isinstance(heuristic, list):
            heuristics = {
                "swap": swap,
                "two_opt": two_opt,
                "insertion": insertion,
            }
            if len(heuristic) == 1:
                self.heuristic = heuristics.get(heuristic[0])
                if self.heuristic is None:
                    raise ValueError(f"Unsupported heuristic: {heuristic[0]}")
            elif len(heuristic) == 2:
                self.heuristic_1 = heuristics.get(heuristic[0])
                self.heuristic_2 = heuristics.get(heuristic[1])
                if self.heuristic_1 is None or self.heuristic_2 is None:
                    raise ValueError(f"Unsupported heuristics: {heuristic}")
            else:
                raise ValueError("Only up to 2 heuristics are supported.")
        else:
            raise ValueError("Heuristic must be provided as a list.")

    def apply_heuristic(
        self, solution: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply the selected heuristic to the given solution.

        Args:
            solution: Current solution tensor
            action: Action tensor indicating the modification to apply

        Returns:
            Modified solution tensor
        """
        if self.heuristic is not None:
            return self.heuristic(solution, action)
        elif self.heuristic_1 is not None and self.heuristic_2 is not None:
            idx = action[:, 2]
            sol1 = self.heuristic_1(solution, action[:, :2].long())
            sol2 = self.heuristic_2(solution, action[:, :2].long())
            return torch.where(idx.unsqueeze(-1).unsqueeze(-1) == 0, sol1, sol2)
        else:
            raise ValueError("Heuristic not properly configured.")

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
            if coords.shape[0] != self.n_problems:
                raise ValueError(
                    f"Expected {self.n_problems} problems, got {coords.shape[0]}"
                )
            if demands.shape[0] != self.n_problems:
                raise ValueError(
                    f"Expected {self.n_problems} problems, got {demands.shape[0]}"
                )
            params = {"coords": coords, "demands": demands}

        else:
            params = self._generate_random_instances()

        self.set_params(params)

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
            *self.get_percentage_demands(),  # Demand percentages
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

    def generate_init_state(self, param: str = None) -> torch.Tensor:
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
        self.ordered_demands = self.get_demands(sol)
        valid = is_feasible(sol, self.ordered_demands, self.capacity).all()
        if not valid:
            raise ValueError("Generated initial solution is not feasible.")
        # Identify route segments
        self.mask = self.ordered_demands != 0
        segment_start = self.mask & ~torch.cat(
            [torch.zeros_like(self.mask[:, :1]), self.mask[:, :-1]], dim=1
        )
        # Compute segment demands
        self.segment_ids = torch.cumsum(segment_start, 1) * self.mask

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

        segment_sums = torch.zeros_like(edge_lengths)
        segment_sums.scatter_add_(1, self.segment_ids, edge_lengths)
        route_costs = segment_sums.gather(1, self.segment_ids) * self.mask

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
    # Demand and Capacity Analysis
    # --------------------------------

    def get_percentage_demands(
        self,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Compute demand-related percentages:
        1. Node demand as a fraction of route demand
        represents the proportion of a node's demand relative to the total demand
        of the route.
        2. Route demand as a fraction of vehicle capacity
        represents how much the route is loaded relative to the vehicle's capacity.
        3. Node demand as a fraction of vehicle capacity
        represents the proportion of the node's demand relative to the vehicle's
        capacity.

        Returns:
            Tuple of three percentage tensors
        """
        demands = self.ordered_demands
        segment_demands = torch.zeros_like(demands)
        segment_demands.scatter_add_(1, self.segment_ids, demands)
        route_demands = segment_demands.gather(1, self.segment_ids)

        node_pct = torch.nan_to_num(demands / route_demands, nan=0.0)
        route_pct = route_demands / self.capacity
        node_cap_pct = demands / self.capacity

        return (
            node_pct.unsqueeze(-1),
            route_pct.unsqueeze(-1),
            node_cap_pct.unsqueeze(-1),
        )

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
            modified_sol = self.apply_heuristic(compact_sol, action).long()

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
            sol = self.apply_heuristic(solution, action).long()
            # Note: This approach may require post-processing to ensure feasibility

        new_ordered_demands = self.get_demands(sol)
        # Check if the modified solutions are feasible (respect capacity constraints)
        valid = (
            is_feasible(sol, new_ordered_demands, self.capacity).unsqueeze(-1).long()
        )
        # Update self.ordered_demands if the solution is valid
        self.ordered_demands = torch.where(
            valid == 1, new_ordered_demands, self.ordered_demands
        )
        # Identify route segments
        self.mask = self.ordered_demands != 0
        segment_start = self.mask & ~torch.cat(
            [torch.zeros_like(self.mask[:, :1]), self.mask[:, :-1]], dim=1
        )

        # Compute segment demands
        self.segment_ids = torch.cumsum(segment_start, 1) * self.mask

        # Return original solution if modified solution is infeasible
        sol = torch.where(valid.unsqueeze(-1) == 1, sol, solution).to(torch.int64)

        return sol, valid
