import torch
import numpy as np
import random
from tqdm import tqdm
from loguru import logger

from model import CVRPActor, CVRPActorPairs, CVRPCritic
from ppo2 import ppo
from sa import sa
from problem import CVRP
from replay import ReplayBuffer
from Logger import WandbLogger
from HP import _HP, get_script_arguments

# import cProfile
# import io
# import pstats


# Load configuration from YAML and command line arguments
cfg = _HP("src/HP.yaml")
cfg.update(get_script_arguments(cfg.keys()))

# Initialize WandB logging if enabled
if cfg["LOG"]:
    WandbLogger.init(None, 3, cfg)


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
    train_in = sa(
        actor, problem, init_x, cfg, replay_buffer=replay, baseline=False, greedy=False
    )

    # Optimize policy with PPO
    actor_loss, critic_loss = ppo(
        actor, critic, replay, actor_opt, critic_opt, cfg, problem
    )

    # Compute gradient statistics for monitoring
    avg_actor_grad = np.mean(
        [p.grad.abs().mean().item() for p in actor.parameters() if p.grad is not None]
    )
    avg_critic_grad = np.mean(
        [p.grad.abs().mean().item() for p in critic.parameters() if p.grad is not None]
    )

    return (train_in, actor_loss, critic_loss, avg_actor_grad, avg_critic_grad)


def test_model(
    actor: torch.nn.Module,
    problem: CVRP,
    init_x: torch.Tensor,
    cfg: dict,
):
    """Tests the trained model using Simulated Annealing."""

    # Perform Simulated Annealing
    test = sa(
        actor,
        problem,
        init_x,
        cfg,
        replay_buffer=None,
        baseline=False,
        greedy=False,
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

    problem_test = CVRP(
        cfg["PROBLEM_DIM"],
        cfg["N_PROBLEMS"],
        cfg["MAX_LOAD"],
        device=cfg["DEVICE"],
        params=cfg,
    )
    problem_test.manual_seed(0)
    # Generate new problem instances
    params = problem_test.generate_params(mode="test")
    params = {k: v.to(cfg["DEVICE"]) for k, v in params.items()}
    problem_test.set_params(params)
    # Get initial solutions
    init_x_test = problem_test.generate_init_x()

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
            cfg["EMBEDDING_DIM"],
            num_hidden_layers=cfg["NUM_H_LAYERS"],
            device=cfg["DEVICE"],
            mixed_heuristic=True if cfg["HEURISTIC"] == "mix" else False,
        )
    actor.manual_seed(cfg["SEED"])
    critic = CVRPCritic(
        cfg["EMBEDDING_DIM"],
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

    # Training loop
    with tqdm(range(cfg["N_EPOCHS"])) as progress_bar:
        for epoch in progress_bar:
            # Generate new problem instances
            params = problem.generate_params()
            params = {k: v.to(cfg["DEVICE"]) for k, v in params.items()}
            problem.set_params(params)

            # Get initial solutions
            init_x = problem.generate_init_x()

            # Training phase
            train_results = train_ppo(
                actor, critic, actor_opt, critic_opt, problem, init_x, cfg
            )
            train_in, actor_loss, critic_loss, avg_actor_grad, avg_critic_grad = (
                train_results
            )

            # Test phase every 10 epochs
            if epoch % 10 == 0:
                test = test_model(
                    actor,
                    problem_test,
                    init_x_test,
                    cfg,
                )
                train_loss = torch.mean(test["min_cost"])

            # Logging
            if cfg["LOG"]:
                logs = {
                    # Training metrics
                    "Actor_loss": actor_loss,
                    "Critic_loss": critic_loss,
                    "Train_loss": actor_loss + 0.5 * critic_loss,
                    # Gradient monitoring
                    "Avg_actor_grad": avg_actor_grad,
                    "Avg_critic_grad": avg_critic_grad,
                }
                if epoch % 10 == 0:
                    # Additional logs every 10 epochs
                    logs.update(
                        {
                            # Cost metrics
                            "Min_cost": torch.mean(test["min_cost"]),
                            # Improvement metrics
                            "N_gain": torch.mean(test["ngain"]),
                            "Gain": torch.mean(test["init_cost"] - test["min_cost"]),
                            # Search statistics
                            "Acceptance_rate": torch.mean(test["n_acc"]),
                            "Rejection_rate": torch.mean(test["n_rej"]),
                        }
                    )
                    if cfg["HEURISTIC"] == "mix":
                        logs["ratio_heuristic"] = test["ratio"]
                WandbLogger.log(logs)

            # Update progress bar
            progress_bar.set_description(f"Training loss: {train_loss:.4f}")

            # Model checkpointing
            if cfg["LOG"]:
                model_name = f"{cfg['PROJECT']}_{cfg['GROUP']}"
                WandbLogger.log_model(
                    save_func=save_model,
                    model=actor,
                    val_loss=(train_loss.item()),
                    epoch=epoch,
                    model_name=model_name,
                )


if __name__ == "__main__":
    # # Lance le profiling et sauvegarde les résultats dans 'profile_results.prof'
    # profiler = cProfile.Profile()
    # profiler.enable()

    logger.info("Training started.")
    main(cfg)  # Main function call
    logger.success("Training completed successfully.")

    # profiler.disable()

    # # Sauvegarde les résultats
    # profiler.dump_stats("profile_results.prof")

    # # Optionnel: Affiche un résumé dans la console
    # stream = io.StringIO()
    # stats = pstats.Stats(profiler, stream=stream)
    # stats.strip_dirs().sort_stats("cumtime").print_stats(20)
    # print(stream.getvalue())
