import os
import torch
import numpy as np
import random
from tqdm import tqdm
import pandas as pd

from model import CVRPActor, CVRPCritic
from ppo import ppo
from sa import sa
from problem import CVRP
from replay import Replay
from Logger import WandbLogger
from HP import _HP, get_script_arguments
from or_tools import test_or_tools

cfg = _HP("src/HP.yaml")
cfg.update(get_script_arguments(cfg.keys()))

# Check if the results file exists, if not create it
results_file = "src/res.csv"
if not os.path.exists(results_file):
    # Create a DataFrame with the necessary columns
    df = pd.DataFrame(
        columns=[
            "Model",
            "Type",
            "Train Init Temp",
            "Train Steps",
            "Train Dimension",
            "Train Demand",
            "Train Scheduler",
            "Train Clustering",
            "Dimension",
            "Step",
            "Initial Temp",
            "Initial Cost",
            "Final Cost",
            "Gain",
        ]
    )
else:
    # Load the existing results file
    df = pd.read_csv(results_file)

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

    # Set seeds
    torch.manual_seed(cfg["SEED"])
    random.seed(cfg["SEED"])
    np.random.seed(cfg["SEED"])

    problem = CVRP(
        cfg["PROBLEM_DIM"],
        cfg["N_PROBLEMS"],
        cfg["MAX_LOAD"],
        device=cfg["DEVICE"],
        params=cfg,
    )

    if cfg["DEMANDS"]:
        cfg["C1"] = cfg["C"] = 10
        cfg["C2"] = 16
    # Initialize the actor and critic models
    actor = CVRPActor(
        cfg["EMBEDDING_DIM"],
        cfg["C1"],
        cfg["C2"],
        device=cfg["DEVICE"],
    )
    critic = CVRPCritic(cfg["EMBEDDING_DIM"], cfg["C"], device=cfg["DEVICE"])

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
            # Create random instances``
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

            name = cfg["PROJECT"] + "_" + str(cfg["PROBLEM_DIM"]) + "_" + cfg["METHOD"]

            path = WandbLogger.log_model(
                save_func=save_model,
                model=actor,
                val_loss=train_loss.item(),
                epoch=i,
                model_name=name,
            )
    train_init_temp = cfg["INIT_TEMP"]
    train_outer_steps = cfg["OUTER_STEPS"]
    # Create random test instances
    for dim, load in zip([20, 50], [30, 40]):
        problem = CVRP(
            dim,
            100,
            load,
            device=cfg["DEVICE"],
            params=cfg,
        )
        params = problem.generate_params(mode="test")
        params = {k: v.to(cfg["DEVICE"]) for k, v in params.items()}
        problem.set_params(params)
        # Find initial solutions
        init_x = problem.generate_init_x()
        init_cost = problem.cost(init_x)
        cfg["VISU_DIM"] = dim
        cfg["MAX_LOAD"] = load
        solution_or_tools = test_or_tools(params, cfg)
        cost_or_tools = problem.cost(solution_or_tools)
        df.loc[len(df)] = [
            path,
            f"OR_TOOLS {cfg["OR_TOOLS_TIME"]} sec",
            "/",
            "/",
            "/",
            "/",
            "/",
            "/",
            dim,
            "/",
            "/",
            torch.mean(init_cost).item(),
            torch.mean(cost_or_tools).item(),
            torch.mean(init_cost).item() - torch.mean(cost_or_tools).item(),
        ]
        print(
            f"Cost of OR_Tools ({cfg["OR_TOOLS_TIME"]} sec, "
            f"{dim} pb size): {torch.mean(cost_or_tools).item()}"
        )
        for init_temp in [1, 100, 1000]:
            for step in [cfg["OUTER_STEPS"], 10 * (cfg["OUTER_STEPS"] ** 2)]:
                cfg["OUTER_STEPS"] = step
                cfg["INIT_TEMP"] = init_temp

                test = sa(
                    actor,
                    problem,
                    init_x,
                    cfg,
                    replay=None,
                    baseline=False,
                    greedy=False,
                    test=True,
                )
                min_cost_train = test["min_cost"]
                print(
                    f"Cost of the trained solution ({step} steps, {init_temp} temp, "
                    f"{dim} pb size): {torch.mean(min_cost_train).item()}"
                )
                df.loc[len(df)] = [
                    path,
                    "Trained",
                    train_init_temp,
                    train_outer_steps,
                    cfg["PROBLEM_DIM"],
                    cfg["DEMANDS"],
                    cfg["SCHEDULER"],
                    cfg["CLUSTERING"],
                    dim,
                    cfg["OUTER_STEPS"],
                    cfg["INIT_TEMP"],
                    torch.mean(init_cost).item(),
                    torch.mean(min_cost_train).item(),
                    torch.mean(init_cost).item() - torch.mean(min_cost_train).item(),
                ]
                test_baseline = sa(
                    actor,
                    problem,
                    init_x,
                    cfg,
                    replay=None,
                    baseline=True,
                    greedy=False,
                    test=True,
                )
                min_cost_baseline = test_baseline["min_cost"]
                print(
                    f"Cost of the Baseline ({step} steps, {init_temp} temp, "
                    f"{dim} pb size): {torch.mean(min_cost_baseline).item()}"
                )
                df.loc[len(df)] = [
                    path,
                    "Baseline",
                    train_init_temp,
                    train_outer_steps,
                    cfg["PROBLEM_DIM"],
                    cfg["DEMANDS"],
                    cfg["SCHEDULER"],
                    cfg["CLUSTERING"],
                    dim,
                    cfg["OUTER_STEPS"],
                    cfg["INIT_TEMP"],
                    torch.mean(init_cost).item(),
                    torch.mean(min_cost_baseline).item(),
                    torch.mean(init_cost).item() - torch.mean(min_cost_baseline).item(),
                ]

    # Save the DataFrame to a CSV file
    df.to_csv(results_file, index=False)
    print(f"Saved results file at {results_file}")


if __name__ == "__main__":
    main(cfg)
