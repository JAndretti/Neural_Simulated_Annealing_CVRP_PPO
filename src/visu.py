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
    actor = CVRPActor(cfg["EMBEDDING_DIM"], device=cfg["DEVICE"])
    actor = load_model(actor, cfg["MODEL_DIR"])
    set_seed(cfg["SEED"])
    problem = CVRP(cfg["VISU_DIM"], 1, cfg["MAX_LOAD"], device=cfg["DEVICE"])
    params = problem.generate_params()
    params = {k: v.to(cfg["DEVICE"]) for k, v in params.items()}
    problem.set_params(params)
    # Find initial solutions
    init_x = problem.generate_init_x()

    cfg["OUTER_STEPS"] = cfg["VISU_STEPS"]
    alpha = np.log(cfg["STOP_TEMP"]) - np.log(cfg["INIT_TEMP"])
    cfg["ALPHA"] = np.exp(alpha / cfg["OUTER_STEPS"]).item()
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
    iter = 0
    if cfg["PLOT"]:
        for logit, state, action, acc, cost in zip(
            logits, states, actions, acceptance, costs
        ):
            if iter == cfg["VISU_STEPS"] - 1:
                break
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(25, 15))

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
                state[..., 1:-1],
                state[..., [-1]],
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
                next_state[..., 1:-1],
                next_state[..., [-1]],
            )
            coords = coords.gather(1, x.long().expand_as(coords))
            x = x.squeeze().long().tolist()
            n1 = int(x[action_start])
            n2 = int(x[action_end])
            coords1 = coords[0, :, 0]
            coords2 = coords[0, :, 1]
            temperature = temp.squeeze().long().tolist()[0]

            plot_CVRP(ax4, x, coords1, coords2, title="After action")
            fig.suptitle(
                f"Action idx: {action.squeeze().tolist()}, Action node : {n1, n2}, "
                f"Acceptance : {acc.item()}, "
                f"Current Cost : {cost.item()}, next Cost : {costs[iter + 1].item()}, "
                f"Température : {temperature}, Iteration : {iter}",
            )
            plt.show()
            iter += 1
        # print("Visualisation done")
        plt.figure(figsize=(10, 6))
        plt.plot([cost.item() for cost in costs], label="Cost Evolution trained model")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.title("Evolution of Costs Over Iterations")
        plt.legend()
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
    if cfg["PLOT"]:
        plt.figure(figsize=(10, 6))
        plt.plot([cost.item() for cost in costs], label="Cost Evolution Baseline")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.title("Evolution of Costs Over Iterations")
        plt.legend()
        plt.grid(True)
        plt.show()

    cfg["OUTER_STEPS"] = 10 * (cfg["VISU_DIM"] ** 2)
    stop_temp = 0.01
    init_temp = 1
    alpha = np.log(stop_temp) - np.log(init_temp)
    cfg["ALPHA"] = np.exp(alpha / cfg["OUTER_STEPS"]).item()

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
    if cfg["PLOT"]:
        plt.figure(figsize=(10, 6))
        plt.plot([cost.item() for cost in costs], label="Cost Evolution Trained model")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.title("Evolution of Costs Over Iterations")
        plt.legend()
        plt.grid(True)
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
    solution_or_tools = test_or_tools(params, cfg)
    cost_or_tools = problem.cost(solution_or_tools)
    if cfg["PLOT"]:
        fig, ax = plt.subplots(figsize=(12, 8))
        state = problem.to_state(solution_or_tools, torch.tensor(1))
        x, coords, temp = (
            state[..., :1],
            state[..., 1:-1],
            state[..., [-1]],
        )
        coords = coords.gather(1, x.long().expand_as(coords))
        x = x.squeeze().long().tolist()
        coords1 = coords[0, :, 0]
        coords2 = coords[0, :, 1]
        plot_CVRP(ax, x, coords1, coords2, title="OR Tools Solution :")
        plt.show()

    print("Statistics:")
    print(f"Initial cost: {costs[0].item():.4f}")
    print(
        f"Cost Baseline with {cfg["VISU_STEPS"]}: {min_cost_baseline_short.item():.4f}"
    )
    print(
        f"Acceptance Baseline with {cfg["VISU_STEPS"]}: {sum(acceptance_baseline_short).item()} /{cfg['VISU_STEPS']} -> {sum(acceptance_baseline_short).item()/cfg['VISU_STEPS']*100:.4f}%"
    )
    print(f"Min Cost Model with {cfg["VISU_STEPS"]}: {min_cost.item()}")
    print(
        f"Acceptance Model with {cfg["VISU_STEPS"]}: {sum(acceptance).item()} /{cfg["VISU_STEPS"]} -> {sum(acceptance).item()/cfg['VISU_STEPS']*100:.4f}%"
    )
    print(f"Cost Baseline with {cfg['OUTER_STEPS']}: {min_cost_baseline.item():.4f}")
    print(
        f"Acceptance Baseline with {cfg['OUTER_STEPS']}: {sum(acceptance_baseline).item()} /{cfg['OUTER_STEPS']} -> {sum(acceptance_baseline).item()/cfg['OUTER_STEPS']*100:.4f}%"
    )
    print(f"Min Cost Model with {cfg['OUTER_STEPS']}: {min_cost_big_train.item()}")
    print(
        f"Acceptance Model with {cfg['OUTER_STEPS']}: {sum(acceptance_big_train).item()} /{cfg['OUTER_STEPS']} -> {sum(acceptance_big_train).item()/cfg['OUTER_STEPS']*100:.4f}%"
    )
    print(f"Cost or tools for {cfg["OR_TOOLS_TIME"]} sec: {cost_or_tools.item()}")


if __name__ == "__main__":
    main()
