# --------------------------------
# Import required libraries
# --------------------------------
import torch  # PyTorch for deep learning
import numpy as np  # NumPy for numerical operations
import random  # For random operations

from tqdm import tqdm  # Progress bar for iterations
from loguru import logger  # Enhanced logging capabilities

# --------------------------------
# Import custom modules
# --------------------------------
from model import (
    CVRPActor,
    CVRPActorPairs,
    CVRPCritic,
)  # Neural network models
from ppo import ppo  # Proximal Policy Optimization implementation
from replay import ReplayBuffer  # Experience replay for RL
from sa import sa_train  # Simulated Annealing implementation
from problem import CVRP  # CVRP problem definition

from Logger import WandbLogger  # Weights & Biases logging
from HP import _HP, get_script_arguments  # Hyperparameter management

# --------------------------------
# Profiling tools
# --------------------------------

profiler = False
if profiler:
    import cProfile  # For profiling code performance
    import pstats  # For statistics from profiling
    import io  # For in-memory stream handling

# --------------------------------
# Configure logger formatting
# --------------------------------
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

# --------------------------------
# Load and prepare configuration
# --------------------------------
# Load base configuration from YAML file
cfg = _HP("src/HyperParameters/HP.yaml")
# Override with command line arguments
cfg.update(get_script_arguments(cfg.keys()))

# --------------------------------
# Initialize experiment tracking
# --------------------------------
# Set up Weights & Biases logging if enabled
if cfg["LOG"]:
    # Initialize WandB with config
    WandbLogger.init(None, 3, cfg)
    # Log the model save directory for reference
    logger.info(f"WandB model save directory: {WandbLogger._instance.model_dir}")


def log_training_and_test_metrics(
    actor_loss, critic_loss, avg_actor_grad, avg_critic_grad, test, epoch, cfg
):
    logs = {}
    if actor_loss is not None:
        logs = {
            # Training metrics
            "Actor_loss": actor_loss,
            "Critic_loss": critic_loss,
            "Train_loss": actor_loss + 0.5 * critic_loss,
            # Gradient monitoring
            "Avg_actor_grad": avg_actor_grad,
            "Avg_critic_grad": avg_critic_grad,
        }
    if epoch % 10 == 0 and test is not None:
        # Additional logs every 10 epochs
        test_logs = {
            # Cost metrics
            "Min_cost": torch.mean(test["min_cost"]),
            # Improvement metrics
            "Gain": torch.mean(test["init_cost"] - test["min_cost"]),
            # Search statistics
            "Acceptance_rate": torch.mean(test["n_acc"]) / cfg["OUTER_STEPS"],
            "Step_best_cost": torch.mean(test["best_step"]) / cfg["OUTER_STEPS"],
            "Valid_percentage": torch.mean(test["is_valid"]),
            "Final_capacity_left": torch.mean(test["capacity_left"]),
        }
        logs.update(test_logs)
        if cfg["HEURISTIC"] == "mix":
            logs["ratio_heuristic"] = test["ratio"]
    WandbLogger.log(logs)


def save_model(path: str, actor_model: torch.nn.Module = None) -> None:
    """Saves actor model state to specified path.

    Args:
        path: Destination file path
        actor_model: Model to be saved
    Raises:
        ValueError: If no model is provided
    """
    if actor_model is not None:
        torch.save(actor_model.state_dict(), path)
    else:
        raise ValueError("No model provided for saving")


def train_ppo(
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    problem: CVRP,
    init_x: torch.Tensor,
    cfg: dict,
    step: int = 0,
) -> tuple:
    """Executes one training cycle of PPO.

    1. Runs Simulated Annealing to collect experiences
    2. Optimizes policy using PPO
    3. Computes gradient statistics

    Returns:
        Tuple containing:
        - SA training results
        - Actor loss
        - Critic loss
        - Average actor gradients
        - Average critic gradients
    """
    # Initialize experience replay buffer
    replay = ReplayBuffer(cfg["OUTER_STEPS"] * cfg["INNER_STEPS"])

    # Collect experiences through Simulated Annealing
    train_in = sa_train(
        actor,
        problem,
        init_x,
        cfg,
        replay_buffer=replay,
        baseline=False,
        greedy=False,
        train=True,
    )

    # Optimize policy with PPO
    actor_loss, critic_loss = ppo(
        actor, critic, init_x.shape[1], replay, actor_opt, critic_opt, step, cfg
    )

    # Compute gradient statistics for monitoring
    avg_actor_grad = np.mean(
        [p.grad.abs().mean().item() for p in actor.parameters() if p.grad is not None]
    )
    avg_critic_grad = np.mean(
        [p.grad.abs().mean().item() for p in critic.parameters() if p.grad is not None]
    )

    return (
        train_in,
        actor_loss,
        critic_loss,
        avg_actor_grad,
        avg_critic_grad,
    )


def test_model(
    actor: torch.nn.Module,
    problem: CVRP,
    init_x: torch.Tensor,
    cfg: dict,
):
    """Tests the trained model using Simulated Annealing."""

    # Perform Simulated Annealing
    test = sa_train(
        actor,
        problem,
        init_x,
        cfg,
        replay_buffer=None,
        baseline=False,
        greedy=False,
        train=False,
    )
    return test


def main(cfg: dict) -> None:
    """Main training loop for CVRP optimization."""

    # Device configuration
    if "cuda" in cfg["DEVICE"]:
        if not torch.cuda.is_available():
            cfg["DEVICE"] = "cpu"
            print("CUDA device not available. Falling back to CPU.")
        else:
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
    elif "mps" in cfg["DEVICE"] and not torch.backends.mps.is_available():
        cfg["DEVICE"] = "cpu"
        print("MPS device not available. Falling back to CPU.")

    logger.info(f"Using device: {cfg['DEVICE']}")

    # Set random seeds for reproducibility
    torch.manual_seed(cfg["SEED"])
    random.seed(cfg["SEED"])
    np.random.seed(cfg["SEED"])

    # Initialize CVRP problem environment
    problem = CVRP(
        cfg["PROBLEM_DIM"],
        cfg["N_PROBLEMS"],
        cfg["MAX_LOAD"],
        device=cfg["DEVICE"],
        params=cfg,
    )
    problem.manual_seed(cfg["SEED"])
    problem.set_heuristic(cfg["HEURISTIC"], cfg["MIX1"], cfg["MIX2"])

    dim_test = cfg["TEST_DIMENSION"]
    n_test_pb = cfg["TEST_NB_PROBLEMS"]  # Number of test problems
    path = f"generated_problem/gen{dim_test}.pt"
    bdd = torch.load(path, map_location="cpu")
    coords = bdd["node_coords"][:n_test_pb]
    demands = bdd["demands"][:n_test_pb]
    capacities = bdd["capacity"][:n_test_pb]
    problem_test = CVRP(
        dim_test,
        capacities.shape[0],  # Number of problems
        capacities,
        device=cfg["DEVICE"],
        params=cfg,
    )
    problem_test.manual_seed(0)
    # Generate new problem instances
    params = problem_test.generate_params("test", True, coords, demands)
    problem_test.set_params(params)
    if cfg["CHANGE_INIT_METHOD"]:
        init_test = "isolate"
    else:
        init_test = cfg["INIT"]
    init_x_test = problem_test.generate_init_x(init_test)
    problem_test.set_heuristic(cfg["HEURISTIC"], cfg["MIX1"], cfg["MIX2"])
    init_cost = torch.mean(problem_test.cost(init_x_test))
    logger.info(
        "Test problem initialized with params: "
        f"PROBLEM_DIM={dim_test}, "
        f"N_PROBLEMS={n_test_pb}, "
        f"Initial cost: {init_cost.item():.3f}"
    )

    # Initialize models
    if cfg["PAIRS"]:
        actor = CVRPActorPairs(
            cfg["EMBEDDING_DIM"],
            num_hidden_layers=cfg["NUM_H_LAYERS"],
            device=cfg["DEVICE"],
            mixed_heuristic=True if cfg["HEURISTIC"] == "mix" else False,
        )
    else:
        actor = CVRPActor(
            embed_dim=cfg["EMBEDDING_DIM"],
            c=cfg["ENTRY"],
            num_hidden_layers=cfg["NUM_H_LAYERS"],
            device=cfg["DEVICE"],
            mixed_heuristic=True if cfg["HEURISTIC"] == "mix" else False,
            method=cfg["UPDATE_METHOD"],
        )
    logger.info(f"Actor model initialized: {actor.__class__.__name__}")
    actor.manual_seed(cfg["SEED"])
    critic = CVRPCritic(
        embed_dim=cfg["EMBEDDING_DIM"],
        c=cfg["ENTRY"],
        num_hidden_layers=cfg["NUM_H_LAYERS"],
        device=cfg["DEVICE"],
    )
    # Initialize optimizers
    actor_opt = torch.optim.Adam(
        actor.parameters(), lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"]
    )
    critic_opt = torch.optim.Adam(
        critic.parameters(), lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"]
    )
    logger.info("Training started.")
    # Training loop
    test_init = test_model(
        actor,
        problem_test,
        init_x_test,
        cfg,
    )
    train_loss = torch.mean(test_init["min_cost"])
    logger.info(f"Initial test loss: {train_loss:.4f}")

    early_stopping_counter = 0
    early_stop_value = float("inf")
    logger.info(
        f"Starting training loop with INIT method: {cfg['INIT']}, CLUSTERING: "
        f"{cfg['CLUSTERING']}",
    )
    with tqdm(range(cfg["N_EPOCHS"]), unit="epoch", colour="blue") as progress_bar:
        for epoch in progress_bar:
            # for epoch in range(cfg["N_EPOCHS"]):
            # Generate new problem instances
            params = problem.generate_params()
            params = {k: v.to(cfg["DEVICE"]) for k, v in params.items()}
            problem.set_params(params)

            # Get initial solutions
            init_x = problem.generate_init_x(cfg["INIT"])

            # Training phase
            train_results = train_ppo(
                actor,
                critic,
                actor_opt,
                critic_opt,
                problem,
                init_x,
                cfg,
                step=epoch + 1,
            )
            (
                train_in,
                actor_loss,
                critic_loss,
                avg_actor_grad,
                avg_critic_grad,
            ) = train_results
            # Test phase every 10 epochs
            if epoch % 10 == 0 and epoch != 0:
                test = test_model(
                    actor,
                    problem_test,
                    init_x_test,
                    cfg,
                )
                train_loss = torch.mean(test["min_cost"])
                # logger.info(f"Epoch {epoch}: Loss: {train_loss:.4f}")
                # Inverse clustering flag for next 10 epoch
                if cfg["ALT_CLUSTERING"]:
                    problem.clustering = not problem.clustering
                    if cfg["VERBOSE"]:
                        logger.info(
                            f"Clustering set to {problem.clustering} at epoch {epoch}"
                        )
                    if problem.clustering:
                        problem.nb_clusters_max = random.randint(
                            2, cfg["NB_CLUSTERS_MAX"]
                        )

            if cfg["CHANGE_INIT_METHOD"] and epoch % 5 == 0 and epoch != 0:
                cfg["INIT"] = cfg["INIT_LIST"][(epoch // 5) % len(cfg["INIT_LIST"])]
                if cfg["VERBOSE"]:
                    logger.info(
                        f"Changed INIT method to {cfg['INIT']} at epoch {epoch}"
                    )
            # Logging
            if cfg["LOG"]:
                log_training_and_test_metrics(
                    actor_loss,
                    critic_loss,
                    avg_actor_grad,
                    avg_critic_grad,
                    test if (epoch != 0 and epoch >= 10) else test_init,
                    epoch,
                    cfg,
                )

            # Update progress bar
            progress_bar.set_description(f"Training loss: {train_loss:.4f}")

            # Model checkpointing
            if cfg["LOG"]:
                model_name = f"{cfg['PROJECT']}_{cfg['GROUP']}_actor"
                saved, path = WandbLogger.log_model(
                    save_func=save_model,
                    model=actor,
                    val_loss=(train_loss.item()),
                    epoch=epoch,
                    model_name=model_name,
                )
            if epoch % 10 == 0:
                if train_loss.item() >= early_stop_value:
                    early_stopping_counter += 1
                    if cfg["VERBOSE"]:
                        logger.info(
                            f"Epoch {epoch}: Early stopping counter: "
                            f"{early_stopping_counter}"
                        )
                else:
                    early_stopping_counter = 0
                    early_stop_value = min(train_loss.item(), early_stop_value)

            if early_stopping_counter > 5:
                logger.warning(
                    f"Early stopping triggered at epoch {epoch} "
                    f"with loss {early_stop_value:.4f}"
                )
                break


if __name__ == "__main__":
    if profiler:
        # Launch profiling and save results to 'profile_results.prof'

        profiler = cProfile.Profile()
        profiler.enable()

        main(cfg)  # Main function call
        logger.info("Training completed successfully.")

        profiler.disable()

        # Save the results
        profiler.dump_stats("profile_results.prof")

        # Optional: Display a summary in the console
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.strip_dirs().sort_stats("cumtime").print_stats(20)
        print(stream.getvalue())
    else:
        main(cfg)  # Main function call
        logger.info("Training completed successfully.")
