import os
import torch
import numpy as np
import random
from tqdm import tqdm

from model import CVRPActor, CVRPCritic
from ppo import ppo
from sa import sa
from problem import CVRP
from replay import Replay
from Logger import WandbLogger
from HP import _HP, get_script_arguments

cfg = _HP("src/HP.yaml")
cfg.update(get_script_arguments(cfg.keys()))

if cfg["LOG"]:
    WandbLogger.init(None, 3, cfg)


def create_folder(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print(f"Created: {dirname}")


def save_model(path, actor_model=None):
    if actor_model is not None:
        torch.save(actor_model.state_dict(), path)
    else:
        raise ValueError("No model to save.")


def train_ppo(actor, critic, actor_opt, critic_opt, problem, init_x, cfg):
    # Create replay to store transitions
    replay = Replay(cfg["OUTER_STEPS"] * cfg["INNER_STEPS"])
    # Run SA and collect transitions
    train_in = sa(
        actor, problem, init_x, cfg, replay=replay, baseline=False, greedy=False
    )
    # Optimize the policy with PPO
    actor_loss, critic_loss = ppo(
        actor, critic, replay, actor_opt, critic_opt, cfg, problem
    )
    return train_in, actor_loss, critic_loss


def main(cfg) -> None:
    if "cuda" in cfg["DEVICE"] and not torch.cuda.is_available():
        cfg["DEVICE"] = "cpu"
        print("CUDA device not found. Running on cpu.")
    elif "mps" in cfg["DEVICE"] and not torch.backends.mps.is_available():
        cfg["DEVICE"] = "cpu"
        print("MPS device not found. Running on cpu.")

    # Define temperature decay parameter as a function of the number of steps
    alpha = np.log(cfg["STOP_TEMP"]) - np.log(cfg["INIT_TEMP"])
    cfg["ALPHA"] = np.exp(alpha / cfg["OUTER_STEPS"]).item()

    # Set seeds
    torch.manual_seed(cfg["SEED"])
    random.seed(cfg["SEED"])
    np.random.seed(cfg["SEED"])

    problem = CVRP(
        cfg["PROBLEM_DIM"], cfg["N_PROBLEMS"], cfg["MAX_LOAD"], device=cfg["DEVICE"]
    )
    actor = CVRPActor(cfg["EMBEDDING_DIM"], device=cfg["DEVICE"])
    critic = CVRPCritic(cfg["EMBEDDING_DIM"], device=cfg["DEVICE"])

    # Set problem seed
    problem.manual_seed(cfg["SEED"])

    actor_opt = torch.optim.Adam(
        actor.parameters(), lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"]
    )
    critic_opt = torch.optim.Adam(
        critic.parameters(), lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"]
    )
    with tqdm(range(cfg["N_EPOCHS"])) as t:
        for i in t:
            # Create random instances
            params = problem.generate_params()
            params = {k: v.to(cfg["DEVICE"]) for k, v in params.items()}
            problem.set_params(params)
            # Find initial solutions
            init_x = problem.generate_init_x()
            actor.manual_seed(cfg["SEED"])
            train_in, actor_loss, critic_loss = train_ppo(
                actor, critic, actor_opt, critic_opt, problem, init_x, cfg
            )
            # Rerun trained model
            train_out = sa(
                actor, problem, init_x, cfg, replay=None, baseline=False, greedy=False
            )
            # Base line
            baseline = sa(
                actor, problem, init_x, cfg, replay=None, baseline=True, greedy=False
            )
            # Greedy
            greedy = sa(
                actor, problem, init_x, cfg, replay=None, baseline=False, greedy=True
            )
            if cfg["LOG"]:
                logs = {
                    "Actor_loss": actor_loss,
                    "Critic_loss": critic_loss,
                    "Train_loss": actor_loss + 0.5 * critic_loss,
                    "Min_cost_before_train": torch.mean(train_in["min_cost"]),
                    "Min_cost_after_train": torch.mean(train_out["min_cost"]),
                    "Min_cost_baseline": torch.mean(baseline["min_cost"]),
                    "Min_cost_greedy": torch.mean(greedy["min_cost"]),
                    "N_gain_before_train": torch.mean(train_in["ngain"]),
                    "N_gain_after_train": torch.mean(train_out["ngain"]),
                    "N_gain_baseline": torch.mean(baseline["ngain"]),
                    "N_gain_greedy": torch.mean(greedy["ngain"]),
                    "Primal_before_train": torch.mean(train_in["primal"]),
                    "Primal_after_train": torch.mean(train_out["primal"]),
                    "Primal_baseline": torch.mean(baseline["primal"]),
                    "Primal_greedy": torch.mean(greedy["primal"]),
                    "Gain_before_train": torch.mean(
                        train_in["init_cost"] - train_in["min_cost"]
                    ),
                    "Gain_after_train": torch.mean(
                        train_out["init_cost"] - train_out["min_cost"]
                    ),
                    "Gain_baseline": torch.mean(
                        baseline["init_cost"] - baseline["min_cost"]
                    ),
                    "Gain_greedy": torch.mean(greedy["init_cost"] - greedy["min_cost"]),
                    "Acceptance_rate_before_train": torch.mean(train_in["n_acc"]),
                    "Acceptance_rate_after_train": torch.mean(train_out["n_acc"]),
                    "Acceptance_rate_baseline": torch.mean(baseline["n_acc"]),
                    "Acceptance_rate_greedy": torch.mean(greedy["n_acc"]),
                    "Rejection_rate_before_train": torch.mean(train_in["n_rej"]),
                    "Rejection_rate_after_train": torch.mean(train_out["n_rej"]),
                    "Rejection_rate_baseline": torch.mean(baseline["n_rej"]),
                    "Rejection_rate_greedy": torch.mean(greedy["n_rej"]),
                    "Reward": torch.mean(train_out["reward"]),
                }
                WandbLogger.log(logs)
            train_loss = torch.mean(train_out["min_cost"])

            t.set_description(f"Training loss: {train_loss:.4f}")

            # path = os.path.join(os.getcwd(), "models")
            name = cfg["PROJECT"] + "_" + str(cfg["PROBLEM_DIM"]) + "_" + cfg["METHOD"]
            # create_folder(path)
            # torch.save(actor.state_dict(), os.path.join(path, name))
            WandbLogger.log_model(
                save_func=save_model,
                model=actor,
                val_loss=train_loss.item(),
                epoch=i,
                model_name=name,
            )


if __name__ == "__main__":
    main(cfg)
