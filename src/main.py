"""
CVRP (Capacitated Vehicle Routing Problem) Solver using PPO and Simulated Annealing

This module implements a reinforcement learning approach to solve CVRP problems
using Proximal Policy Optimization (PPO) combined with Simulated Annealing for
solution refinement.

Main components:
- Actor-Critic neural networks for policy learning
- PPO for policy optimization
- Simulated Annealing for solution exploration
- WandB integration for experiment tracking
"""

# --------------------------------
# Import required libraries
# --------------------------------
import os
import random
from typing import Dict, Tuple, Optional, Any

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR

# --------------------------------
# Import custom modules
# --------------------------------
from setup import _HP, get_script_arguments, WandbLogger
from model import CVRPActor, CVRPActorPairs, CVRPCritic
from ppo import ppo, ReplayBuffer
from problem import CVRP
from sa import sa_train
from algo import P_generate_instances, stack_res

# --------------------------------
# Profiling tools (optional)
# --------------------------------
ENABLE_PROFILER = False
if ENABLE_PROFILER:
    import cProfile
    import io
    import pstats

# --------------------------------
# Configure logger formatting
# --------------------------------
logger.remove()  # Remove default logger
logger.add(
    lambda msg: print(msg, end=""),
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<blue>{file}:{line}</blue> | "
        "<red>WARNING</red> | "
        "<red>{message}</red>"
    ),
    colorize=True,
    level="WARNING",
)
logger.add(
    lambda msg: print(msg, end=""),
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "  # Timestamp in green
        "<blue>{file}:{line}</blue> | "  # File and line in blue
        "<yellow>{message}</yellow>"  # Message in yellow
    ),
    colorize=True,
    level="INFO",
)

import warnings

warnings.filterwarnings("ignore", message="Attempting to run cuBLAS")

# --------------------------------
# Load and prepare configuration
# --------------------------------
config = _HP("src/HyperParameters/HP.yaml")
config.update(get_script_arguments(config.keys()))
# --------------------------------
# Initialize experiment tracking
# --------------------------------
if config["LOG"]:
    WandbLogger.init(None, 3, config)
    logger.info(f"WandB model save directory: {WandbLogger._instance.model_dir}")


def log_training_and_test_metrics(
    actor_loss: Optional[float],
    critic_loss: Optional[float],
    avg_actor_grad: float,
    avg_critic_grad: float,
    lr_actor: float,
    beta_kl: float,
    entropy: float,
    early_stopping_counter: int,
    test_results: Optional[Dict[str, torch.Tensor]],
    epoch: int,
    config: Dict[str, Any],
) -> None:
    """
    Log training and testing metrics to WandB.

    Args:
        actor_loss: Actor network loss value
        critic_loss: Critic network loss value
        avg_actor_grad: Average gradient magnitude for actor
        avg_critic_grad: Average gradient magnitude for critic
        test_results: Dictionary containing test metrics
        epoch: Current training epoch
        config: Configuration dictionary
    """
    logs = {}

    # Log training metrics if available
    if actor_loss is not None:
        logs.update(
            {
                "Actor_loss": actor_loss,
                "Critic_loss": critic_loss,
                "Train_loss": actor_loss + 0.5 * critic_loss,
                "Avg_actor_grad": avg_actor_grad,
                "Avg_critic_grad": avg_critic_grad,
                "LR_actor": lr_actor,
                "Beta_KL": beta_kl,
                "Entropy": entropy,
                "early_stopping_counter": early_stopping_counter,
            }
        )

    # Log test metrics every 10 epochs
    if epoch % 10 == 0 and test_results is not None:
        test_logs = {
            "Min_cost": torch.mean(test_results["min_cost"]),
            "Gain": torch.mean(test_results["init_cost"] - test_results["min_cost"]),
            "Acceptance_rate": torch.mean(test_results["n_acc"])
            / config["TEST_OUTER_STEPS"],
            "Step_best_cost": torch.mean(test_results["best_step"])
            / config["TEST_OUTER_STEPS"],
            "Valid_percentage": torch.mean(test_results["is_valid"]),
            "Final_capacity_left": torch.mean(test_results["capacity_left"]),
        }
        logs.update(test_logs)

    if len(config["HEURISTIC"]) > 1:
        logs["ratio_heuristic"] = test_results["ratio"]

    WandbLogger.log(logs)


def save_model(path: str, actor_model: Optional[torch.nn.Module] = None) -> None:
    """
    Save actor model state dictionary to specified path.

    Args:
        path: Destination file path for model checkpoint
        actor_model: PyTorch model to be saved

    Raises:
        ValueError: If no model is provided for saving
    """
    if actor_model is not None:
        torch.save(actor_model.state_dict(), path)
        if config["VERBOSE"]:
            logger.info(f"Model saved to: {path}")
    else:
        raise ValueError("No model provided for saving")


def train_ppo(
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    critic_scheduler: torch.optim.lr_scheduler.ExponentialLR,  # Update scheduler type
    problem: CVRP,
    initial_solutions: torch.Tensor,
    config: Dict[str, Any],
    step: int = 0,
) -> Tuple[Dict[str, torch.Tensor], float, float, float, float]:
    """
    Execute one training cycle of PPO algorithm.

    Process:
    1. Collect experiences using Simulated Annealing
    2. Optimize policy using PPO
    3. Compute gradient statistics for monitoring

    Args:
        actor: Actor neural network
        critic: Critic neural network
        actor_optimizer: Optimizer for actor network
        critic_optimizer: Optimizer for critic network
        critic_scheduler: Learning rate scheduler for critic
        problem: CVRP problem instance
        initial_solutions: Initial solution tensor
        config: Configuration dictionary
        step: Current training step

    Returns:
        Tuple containing:
        - SA training results dictionary
        - Actor loss value
        - Critic loss value
        - Average actor gradient magnitude
        - Average critic gradient magnitude
    """
    # Clear GPU cache if using CUDA
    if problem.device == "cuda":
        torch.cuda.init()
        torch.cuda.empty_cache()

    # Initialize experience replay buffer
    buffer_size = config["OUTER_STEPS"] * config["INNER_STEPS"]
    replay_buffer = ReplayBuffer(buffer_size)

    # Collect experiences through Simulated Annealing
    sa_results = sa_train(
        actor=actor,
        problem=problem,
        initial_solution=initial_solutions,
        config=config,
        replay_buffer=replay_buffer,
        baseline=False,
        greedy=False,
        train=True,
    )

    # Optimize policy using PPO
    train_stats = ppo(
        actor=actor,
        critic=critic,
        pb_dim=initial_solutions.shape[1],
        replay=replay_buffer,
        actor_opt=actor_optimizer,
        critic_opt=critic_optimizer,
        curr_epoch=step,
        cfg=config,
    )

    # Step the learning rate scheduler for the critic
    critic_scheduler.step()

    # Compute gradient statistics for monitoring
    actor_gradients = [
        param.grad.abs().mean().item()
        for param in actor.parameters()
        if param.grad is not None
    ]
    critic_gradients = [
        param.grad.abs().mean().item()
        for param in critic.parameters()
        if param.grad is not None
    ]

    avg_actor_grad = np.mean(actor_gradients) if actor_gradients else 0.0
    avg_critic_grad = np.mean(critic_gradients) if critic_gradients else 0.0

    # Clean up GPU memory
    if problem.device == "cuda":
        torch.cuda.empty_cache()

    return sa_results, train_stats, avg_actor_grad, avg_critic_grad


def test_model(
    actor: torch.nn.Module,
    problem: CVRP,
    initial_solutions: torch.Tensor,
    config: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """
    Test the trained model performance using Simulated Annealing.

    Args:
        actor: Trained actor network
        problem: CVRP problem instance for testing
        initial_solutions: Initial solution tensor
        config: Configuration dictionary

    Returns:
        Dictionary containing test results and metrics
    """
    # Clear GPU cache if using CUDA
    if problem.device == "cuda":
        torch.cuda.empty_cache()

    tmp = config["OUTER_STEPS"]
    config["OUTER_STEPS"] = config["TEST_OUTER_STEPS"]
    # Perform Simulated Annealing for testing
    test_results = sa_train(
        actor=actor,
        problem=problem,
        initial_solution=initial_solutions,
        config=config,
        replay_buffer=None,
        baseline=False,
        greedy=False,
        train=False,
    )

    config["OUTER_STEPS"] = tmp

    # Clean up GPU memory
    if problem.device == "cuda":
        torch.cuda.empty_cache()

    return test_results


def setup_device_and_logging(config: Dict[str, Any]) -> str:
    """
    Configure compute device and logging based on availability.

    Args:
        config: Configuration dictionary

    Returns:
        Selected device string ('cuda', 'mps', or 'cpu')
    """
    device = config["DEVICE"]

    # CUDA device configuration
    if "cuda" in device:
        if not torch.cuda.is_available():
            device = "cpu"
            logger.warning("CUDA device not available. Falling back to CPU.")
        else:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")

    # Apple Metal Performance Shaders (MPS) configuration
    elif "mps" in device and not torch.backends.mps.is_available():
        device = "cpu"
        logger.warning("MPS device not available. Falling back to CPU.")

    logger.info(f"Using device: {device}")
    return device


def setup_reproducibility(seed: int) -> None:
    """
    Set random seeds for reproducible results across different runs.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Random seeds set to: {seed}")


def initialize_training_problem(
    problem: CVRP, device: str, config: Dict[str, Any]
) -> CVRP:
    if config["DATA"] == "uchoa":
        # Load test problem parameters
        test_dim = config["PROBLEM_DIM"]
        n_test_problems = config["N_PROBLEMS"]
        base_seed = random.randint(0, 1000000)
        coords_list, demands_list, capacity_list, _ = P_generate_instances(
            n_test_problems, base_seed, test_dim
        )
        # Stack the results into tensors
        coords, demands, capacity = stack_res(coords_list, demands_list, capacity_list)
        # Generate and set problem parameters
        problem.generate_params("train", True, coords, demands.to(torch.int64))
        problem.capacity = capacity.to(device)
    elif config["DATA"] == "random":
        # Generate new training problem instances
        problem.generate_params()
    return problem


def initialize_test_problem(
    config: Dict[str, Any], device: str
) -> Tuple[CVRP, torch.Tensor]:
    """
    Initialize test problem instance with pre-generated data.

    Args:
        config: Configuration dictionary
        device: Compute device string

    Returns:
        Tuple of (test_problem_instance, initial_test_solutions)
    """
    # Load test problem parameters
    test_dim = config["TEST_DIMENSION"]
    n_test_problems = config["TEST_NB_PROBLEMS"]

    if config["NAZARI"]:
        path = f"generated_nazari_problem/gen_nazari_{test_dim}.pt"

        try:
            test_data = torch.load(path, map_location="cpu")
            logger.info(f"Loaded Nazari test data from: {path}")
        except FileNotFoundError:
            logger.error(f"Nazari test data file not found: {path}")
            raise
        coordinates = test_data["node_coords"][:n_test_problems].to(device)
        demands = test_data["demands"][:n_test_problems].to(device)
        capacities = test_data["capacity"][:n_test_problems].to(device)

    else:
        problem_path = f"generated_uchoa_problem/gen_uchoa_{test_dim}.pt"
        try:
            test_data = torch.load(problem_path, map_location="cpu")
            logger.info(f"Loaded test data from: {problem_path}")
        except FileNotFoundError:
            logger.error(f"Test data file not found: {problem_path}")
            raise

        # Randomly select test problem indices
        indices = torch.randperm(n_test_problems, generator=torch.Generator())
        coordinates = test_data["node_coords"][indices]
        demands = test_data["demands"][indices]
        capacities = test_data["capacity"][indices]

    if (coordinates.shape[0] * coordinates.shape[1] > 1000 * 161) and config["PAIRS"]:
        logger.warning(
            "Test problem size exceeds 1000x161, which may lead to performance issues."
        )

    # Initialize test problem instance
    test_problem = CVRP(
        dim=test_dim,
        n_problems=capacities.shape[0],
        capacities=capacities,
        device=device,
        params=config,
    )
    test_problem.manual_seed(0)

    # Generate and set problem parameters
    test_problem.generate_params("test", True, coordinates, demands)

    # Generate initial solutions
    init_method = config["TEST_INIT"]

    initial_test_solutions = test_problem.generate_init_state(init_method, False)

    # Set heuristic method
    test_problem.set_heuristic(config["HEURISTIC"])

    # Log test problem statistics
    initial_cost = torch.mean(test_problem.cost(initial_test_solutions))
    logger.info(
        f"Test problem initialized - Dimension: {test_dim}, "
        f"Problems: {n_test_problems}, Initial cost: {initial_cost.item():.3f}"
    )

    return test_problem, initial_test_solutions


def initialize_models(
    config: Dict[str, Any], device: str
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Initialize actor and critic neural networks.

    Args:
        config: Configuration dictionary
        device: Compute device string

    Returns:
        Tuple of (actor_model, critic_model)
    """
    # Determine if mixed heuristic is used
    if isinstance(config["HEURISTIC"], str):
        config["HEURISTIC"] = [config["HEURISTIC"]]
    use_mixed_heuristic = len(config["HEURISTIC"]) > 1

    # Initialize actor model (with or without pairs)
    if config["MODEL"] == "pairs":
        actor = CVRPActorPairs(
            embed_dim=config["EMBEDDING_DIM"],
            c=config["ENTRY"],
            num_hidden_layers=config["NUM_H_LAYERS"],
            device=device,
            mixed_heuristic=use_mixed_heuristic,
            method=config["UPDATE_METHOD"],
        )
    elif config["MODEL"] == "seq":
        actor = CVRPActor(
            embed_dim=config["EMBEDDING_DIM"],
            c=config["ENTRY"],
            num_hidden_layers=config["NUM_H_LAYERS"],
            device=device,
            mixed_heuristic=use_mixed_heuristic,
            method=config["UPDATE_METHOD"],
        )
    else:
        raise ValueError(f"Unknown model type specified: {config['MODEL']}")

    actor.manual_seed(config["SEED"])
    logger.info(f"Actor model initialized: {actor.__class__.__name__}")

    # Initialize critic model
    if config["CRITIC_MODEL"] == "ff":
        critic = CVRPCritic(
            embed_dim=config["EMBEDDING_DIM"],
            c=config["ENTRY"],
            num_hidden_layers=config["NUM_H_LAYERS"],
            device=device,
        )
    else:
        raise ValueError(
            f"Unknown critic model type specified: {config['CRITIC_MODEL']}"
        )
    logger.info("Critic model initialized")

    return actor, critic


def main(config: Dict[str, Any]) -> None:
    """
    Main training loop for CVRP optimization using PPO and Simulated Annealing.

    Args:
        config: Configuration dictionary containing all hyperparameters
    """
    # Setup device and reproducibility
    device = setup_device_and_logging(config)
    config["DEVICE"] = device
    setup_reproducibility(config["SEED"])

    # Initialize training problem environment
    training_problem = CVRP(
        dim=config["PROBLEM_DIM"],
        n_problems=config["N_PROBLEMS"],
        capacities=config["MAX_LOAD"],
        device=device,
        params=config,
    )
    training_problem.manual_seed(config["SEED"])
    training_problem.set_heuristic(config["HEURISTIC"])

    if config["REWARD_LAST"]:
        config["REWARD_LAST_SCALE"] = 0.0

    # Initialize test problem environment
    test_problem, initial_test_solutions = initialize_test_problem(config, device)

    # Initialize neural network models
    actor, critic = initialize_models(config, device)

    # Initialize optimizers
    actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=config["LR_ACTOR"], weight_decay=config["WEIGHT_DECAY"]
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=config["LR_CRITIC"], weight_decay=config["WEIGHT_DECAY"]
    )

    # Initialize learning rate scheduler for the critic
    critic_scheduler = ExponentialLR(critic_optimizer, gamma=0.98)  # Smooth decay

    logger.info("Training initialization completed")

    # Perform initial test to establish baseline
    initial_test_results = test_model(
        actor, test_problem, initial_test_solutions, config
    )
    current_test_loss = torch.mean(initial_test_results["min_cost"])
    logger.info(
        f"Initial test completed with {config['TEST_INIT']}, "
        f"with loss: {current_test_loss:.4f}"
    )

    # Early stopping variables
    early_stopping_counter = 0
    best_loss_value = float("inf")

    logger.info("Starting training")

    # Clear GPU memory before training
    if training_problem.device == "cuda":
        torch.cuda.empty_cache()

    # Main training loop
    with tqdm(range(config["N_EPOCHS"]), unit="epoch", colour="blue") as progress_bar:
        for epoch in progress_bar:
            # Generate new training problem instances
            training_problem = initialize_training_problem(
                training_problem, device, config
            )

            # Generate initial solutions for training
            initial_training_solutions = training_problem.generate_init_state(
                config["INIT"], config["MULTI_INIT"]
            )

            # Execute training step
            training_results = train_ppo(
                actor=actor,
                critic=critic,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                critic_scheduler=critic_scheduler,  # Pass scheduler
                problem=training_problem,
                initial_solutions=initial_training_solutions,
                config=config,
                step=epoch + 1,
            )

            # Unpack training results
            (
                sa_results,
                train_stats,
                avg_actor_grad,
                avg_critic_grad,
            ) = training_results

            actor_loss, critic_loss, avg_entropy, beta_kl = train_stats

            config["BETA_KL"] = beta_kl  # Update beta_kl from PPO

            # Periodic testing and evaluation
            test_results = None
            if epoch % 10 == 0 and epoch != 0:
                test_results = test_model(
                    actor, test_problem, initial_test_solutions, config
                )
                current_test_loss = torch.mean(test_results["min_cost"])

                if config["REWARD_LAST"]:
                    config["REWARD_LAST_SCALE"] = min(
                        config["REWARD_LAST_SCALE"] + config["REWARD_LAST_ADD"], 100
                    )

            # Early stopping logic
            if epoch % 10 == 0:
                if current_test_loss.item() >= best_loss_value:
                    early_stopping_counter += 1
                    if config["VERBOSE"]:
                        logger.info(f"Early stopping counter: {early_stopping_counter}")
                else:
                    early_stopping_counter = 0
                    best_loss_value = min(current_test_loss.item(), best_loss_value)

            # Log metrics to WandB
            if config["LOG"]:
                lr_actor = actor_optimizer.param_groups[0]["lr"]
                log_training_and_test_metrics(
                    actor_loss=actor_loss,
                    critic_loss=critic_loss,
                    avg_actor_grad=avg_actor_grad,
                    avg_critic_grad=avg_critic_grad,
                    lr_actor=lr_actor,
                    beta_kl=beta_kl,
                    entropy=avg_entropy,
                    early_stopping_counter=early_stopping_counter,
                    test_results=(
                        test_results if (epoch >= 10) else initial_test_results
                    ),
                    epoch=epoch,
                    config=config,
                )

            # Model checkpointing
            if config["LOG"]:
                model_name = f"{config['PROJECT']}_{config['GROUP']}_actor"
                WandbLogger.log_model(
                    save_func=save_model,
                    model=actor,
                    val_loss=current_test_loss.item(),
                    epoch=epoch,
                    model_name=model_name,
                )

            # Trigger early stopping if no improvement for too long
            if early_stopping_counter > 10:
                logger.warning(
                    f"Early stopping triggered at epoch {epoch} "
                    f"with loss {best_loss_value:.4f}"
                )
                break

            # Update progress bar
            progress_bar.set_description(
                (
                    f"Test loss: {current_test_loss:.4f}, "
                    f"EarlyStop Counter: {early_stopping_counter}"
                )
            )

    logger.info("Training completed successfully")


if __name__ == "__main__":
    if ENABLE_PROFILER:
        # Profile the main function execution
        profiler = cProfile.Profile()
        profiler.enable()

        main(config)

        profiler.disable()
        profiler.dump_stats("profile_results.prof")

        # Display profiling summary
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.strip_dirs().sort_stats("cumtime").print_stats(20)
        print(stream.getvalue())

        logger.info("Profiled training completed successfully")
    else:
        main(config)
