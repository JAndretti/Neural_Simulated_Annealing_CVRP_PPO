import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from rich import print


from model import CVRPActor
from problem import CVRP
from sa import sa
from utils import plot_CVRP
from or_tools import test_or_tools

from HP import _HP, get_script_arguments

cfg = _HP("src/test.yaml")
cfg.update(get_script_arguments(cfg.keys()))


# Function to set the seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Function to extract the loss value from a filename
def extract_loss(filename):
    try:
        # Identify and extract the "_loss_X" portion
        return float((filename.split("_")[-1])[:-3])
    except ValueError:
        pass
    return float("inf")  # Return a large number if extraction fails


# Function to load the model weights from wandb
def load_model(model, folder):
    # List all files in the directory
    files = [f for f in os.listdir(folder) if f.endswith(".pt")]
    if files:
        # Find the file with the smallest loss
        file_with_min_loss = min(files, key=extract_loss)
    model.load_state_dict(torch.load(folder + file_with_min_loss, weights_only=True))
    print(f"Loaded model from {file_with_min_loss}")
    return model


def main():
    # Initialize the actor model
    actor = CVRPActor(
        cfg["EMBEDDING_DIM"],
        cfg["C1"],
        cfg["C2"],
    )
    actor = load_model(actor, cfg["MODEL_DIR"])
    base_step = cfg["OUTER_STEPS"]
    set_seed(0)
    problem = CVRP(
        cfg["PROBLEM_DIM"],
        cfg["N_PROBLEMS"],
        cfg["MAX_LOAD"],
        device=cfg["DEVICE"],
        params=cfg,
    )
    params = problem.generate_params(mode="test")
    params = {k: v.to(cfg["DEVICE"]) for k, v in params.items()}
    problem.set_params(params)
    # Find initial solutions
    init_x = problem.generate_init_x()

    with torch.no_grad():
        actor.eval()
        train = sa(
            actor,
            problem,
            init_x,
            cfg,
            replay=None,
            baseline=False,
            greedy=False,
            record_state=True,
        )
    logits = train[
        "distributions"
    ]  # list of distributions of size VISU_STEPS (dist actor1, dist actor2)
    states = train[
        "states"
    ]  # list of states of size VISU_STEPS (node_nbr, coord1, coord2, temp)
    actions = train["actions"]  # list of actions of size VISU_STEPS (actor1, actor2)
    acceptance = train["acceptance"]  # list of bool of size VISU_STEPS
    costs = train["costs"]  # list of costs of size VISU_STEPS
    min_cost = train["min_cost"]
    temp_tmp = train["temperature"]
    iter = 0
    if cfg["PLOT"]:
        for logit, state, action, acc, cost in zip(
            logits, states, actions, acceptance, costs
        ):
            if iter == cfg["OUTER_STEPS"] - 1:
                break
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(25, 10))

            prob1 = torch.exp(logit[0])
            prob2 = torch.exp(logit[1])

            # Premier graphique en bar pour logit[0]
            ax1.bar(range(len(prob1.squeeze())), prob1.squeeze())
            ax1.set_title("Distribution prob actor 1")

            # Second graphique en bar pour logit[1]
            ax2.bar(range(len(prob2.squeeze())), prob2.squeeze())
            ax2.set_title("Distribution prob actor 2")

            action_start = action[0][0].item()
            action_end = action[0][1].item()

            # Troisième graphique pour le chemin actuel du TSP
            x, coords, temp = (
                state[..., :1],
                state[..., 1:3],
                state[..., 3:4],
            )
            coords = coords.gather(1, x.long().expand_as(coords))
            x = x.squeeze().long().tolist()
            n1 = int(x[action_start])
            n2 = int(x[action_end])
            coords1 = coords[0, :, 0]
            coords2 = coords[0, :, 1]
            temperature = temp.squeeze().long().tolist()[0]

            plot_CVRP(ax3, x, coords1, coords2, title="Current Solution")

            next_state = states[iter + 1]
            # Quatrième graphique pour le chemin actuel du TSP
            x, coords, temp = (
                next_state[..., :1],
                next_state[..., 1:3],
                next_state[..., 3:4],
            )
            coords = coords.gather(1, x.long().expand_as(coords))
            x = x.squeeze().long().tolist()
            n1 = int(x[action_start])
            n2 = int(x[action_end])
            coords1 = coords[0, :, 0]
            coords2 = coords[0, :, 1]
            temperature = temp.squeeze().tolist()[0]

            plot_CVRP(ax4, x, coords1, coords2, title="After action")
            fig.suptitle(
                f"Action idx: {action.squeeze().tolist()}, Action node : {n1, n2}, "
                f"Acceptance : {acc.item()}, "
                f"Current Cost : {cost.item()}, next Cost : {costs[iter + 1].item()}, "
                f"Température : {temperature}, Iteration : {iter}",
            )
            plt.show()
            iter += 1

        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.plot(
            [cost.item() for cost in costs],
            label="Cost Evolution trained model",
            color="b",
        )
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Cost", color="b")
        ax1.tick_params(axis="y", labelcolor="b")

        ax2 = ax1.twinx()
        ax2.plot([t.item() for t in temp_tmp], label="Temperature", color="r")
        ax2.set_ylabel("Temperature", color="r")
        ax2.tick_params(axis="y", labelcolor="r")

        fig.suptitle("Evolution of Costs and Temperature Over Iterations")
        fig.legend(loc="upper right")
        fig.tight_layout()
        plt.grid(True)
        plt.show()

    # BASELINE

    with torch.no_grad():
        actor.eval()
        baseline = sa(
            actor,
            problem,
            init_x,
            cfg,
            replay=None,
            baseline=True,
            greedy=False,
            record_state=True,
        )
    min_cost_baseline_short = baseline["min_cost"]
    acceptance_baseline_short = baseline["acceptance"]
    costs = baseline["costs"]
    temp_tmp = baseline["temperature"]
    if cfg["PLOT"]:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.plot(
            [cost.item() for cost in costs], label="Cost Evolution Baseline", color="b"
        )
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Cost", color="b")
        ax1.tick_params(axis="y", labelcolor="b")

        ax2 = ax1.twinx()
        ax2.plot([t.item() for t in temp_tmp], label="Temperature", color="r")
        ax2.set_ylabel("Temperature", color="r")
        ax2.tick_params(axis="y", labelcolor="r")

        fig.suptitle("Evolution of Costs and Temperature Over Iterations")
        fig.legend(loc="upper right")
        fig.tight_layout()
        plt.grid(True)
        plt.show()

    cfg["OUTER_STEPS"] = 1000  # 10 * (cfg["PROBLEM_DIM"] ** 2)

    with torch.no_grad():
        actor.eval()
        big_train = sa(
            actor,
            problem,
            init_x,
            cfg,
            replay=None,
            baseline=False,
            greedy=False,
            record_state=True,
        )
    min_cost_big_train = big_train["min_cost"]
    acceptance_big_train = big_train["acceptance"]
    costs = big_train["costs"]
    temp_tmp = big_train["temperature"]
    if cfg["PLOT"]:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.plot(
            [cost.item() for cost in costs],
            label="Cost Evolution Trained model",
            color="b",
        )
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Cost", color="b")
        ax1.tick_params(axis="y", labelcolor="b")

        ax2 = ax1.twinx()
        ax2.plot([t.item() for t in temp_tmp], label="Temperature", color="r")
        ax2.set_ylabel("Temperature", color="r")
        ax2.tick_params(axis="y", labelcolor="r")

        fig.suptitle("Evolution of Costs and Temperature Over Iterations")
        fig.legend(loc="upper right")
        fig.tight_layout()
        plt.grid(True)
        plt.show()

        fig, ax1 = plt.subplots(figsize=(12, 8))
        states = big_train["states"]
        costs = big_train["costs"]
        min_cost_state = states[costs.index(min(costs))]
        x, coords, temp = (
            min_cost_state[..., :1],
            min_cost_state[..., 1:3],
            min_cost_state[..., 3:4],
        )
        coords = coords.gather(1, x.long().expand_as(coords))
        x = x.squeeze().long().tolist()
        coords1 = coords[0, :, 0]
        coords2 = coords[0, :, 1]
        plot_CVRP(ax1, x, coords1, coords2, title="State with Minimum Cost")
        plt.show()

    with torch.no_grad():
        actor.eval()
        baseline = sa(
            actor,
            problem,
            init_x,
            cfg,
            replay=None,
            baseline=True,
            greedy=False,
            record_state=True,
        )
    min_cost_baseline = baseline["min_cost"]
    acceptance_baseline = baseline["acceptance"]

    # OR TOOLS
    cfg["OR_DIM"] = cfg["PROBLEM_DIM"]
    solution_or_tools = test_or_tools(params, cfg)
    cost_or_tools = problem.cost(solution_or_tools)
    if cfg["PLOT"]:
        fig, ax = plt.subplots(figsize=(12, 8))
        state = problem.to_state(solution_or_tools, torch.tensor(1))
        x, coords, temp = (
            next_state[..., :1],
            next_state[..., 1:3],
            next_state[..., 3:4],
        )
        coords = coords.gather(1, x.long().expand_as(coords))
        x = x.squeeze().long().tolist()
        coords1 = coords[0, :, 0]
        coords2 = coords[0, :, 1]
        plot_CVRP(ax, x, coords1, coords2, title="OR Tools Solution :")
        plt.show()

    print("Statistics:")
    print(f"Initial cost: {costs[0].item():.4f}")
    print(f"Cost Baseline with {base_step}: {min_cost_baseline_short.item():.4f}")
    print(
        f"Acceptance Baseline with {base_step}: {sum(acceptance_baseline_short).item()} /{base_step} -> {sum(acceptance_baseline_short).item()/base_step*100:.1f}%"
    )
    print(f"Min Cost Model with {base_step}: {min_cost.item():.4f}")
    print(
        f"Acceptance Model with {base_step}: {sum(acceptance).item()} /{base_step} -> {sum(acceptance).item()/base_step*100:.1f}%"
    )
    print(f"Cost Baseline with {cfg['OUTER_STEPS']}: {min_cost_baseline.item():.4f}")
    print(
        f"Acceptance Baseline with {cfg['OUTER_STEPS']}: {sum(acceptance_baseline).item()} /{cfg['OUTER_STEPS']} -> {sum(acceptance_baseline).item()/cfg['OUTER_STEPS']*100:.1f}%"
    )
    print(f"Min Cost Model with {cfg['OUTER_STEPS']}: {min_cost_big_train.item():.4f}")
    print(
        f"Acceptance Model with {cfg['OUTER_STEPS']}: {sum(acceptance_big_train).item()} /{cfg['OUTER_STEPS']} -> {sum(acceptance_big_train).item()/cfg['OUTER_STEPS']*100:.1f}%"
    )
    print(f"Cost or tools for {cfg["OR_TOOLS_TIME"]} sec: {cost_or_tools.item():.4f}")


if __name__ == "__main__":
    main()
