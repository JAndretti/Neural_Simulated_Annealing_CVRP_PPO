import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt


from model import TSPActor
from problem import TSP
from sa import sa

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
    actor = TSPActor(cfg["EMBEDDING_DIM"], device=cfg["DEVICE"])
    actor = load_model(actor, cfg["MODEL_DIR"])
    set_seed(cfg["SEED"])
    problem = TSP(cfg["VISU_DIM"], 1, device=cfg["DEVICE"])
    params = problem.generate_params()
    params = {k: v.to(cfg["DEVICE"]) for k, v in params.items()}
    problem.set_params(**params)
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
    iter = 0
    for logit, state, action, acc, cost in zip(
        logits, states, actions, acceptance, costs
    ):
        if iter == cfg["VISU_STEPS"] - 1:
            break
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(25, 15))

        prob1 = torch.softmax(logit[0], dim=-1)
        prob2 = torch.softmax(logit[1], dim=-1)

        # Premier graphique en bar pour logit[0]
        ax1.bar(range(len(prob1.squeeze())), prob1.squeeze())
        ax1.set_title("Distribution prob actor 1")

        # Second graphique en bar pour logit[1]
        ax2.bar(range(len(prob2.squeeze())), prob2.squeeze())
        ax2.set_title("Distribution prob actor 2")

        action_start = action[0][0].item()
        action_end = action[0][1].item()

        if action_start > action_end:
            action_start, action_end = action_end, action_start

        # Troisième graphique pour le chemin actuel du TSP
        nodes = state[0, :, 0].tolist()
        n1 = int(nodes[action_start])
        n2 = int(nodes[action_end])
        coords1 = state[0, :, 1]
        coords2 = state[0, :, 2]
        temperature = state[0, 0, 3]

        ax3.plot(coords1, coords2, marker="o")
        for i, txt in enumerate(nodes):
            ax3.annotate(int(txt), (coords1[i], coords2[i]))
        ax3.plot(
            coords1[action_start - 1 : action_end + 1],
            coords2[action_start - 1 : action_end + 1],
            color="red",
            marker="o",
        )
        ax3.scatter(coords1[action_start], coords2[action_start], color="red")
        ax3.scatter(coords1[action_end], coords2[action_end], color="red")
        ax3.set_title(f"Current solution {list(map(int, nodes))}")

        next_state = states[iter + 1]
        # Quatrième graphique pour le chemin actuel du TSP
        nodes = next_state[0, :, 0].tolist()
        coords1 = next_state[0, :, 1]
        coords2 = next_state[0, :, 2]

        ax4.plot(coords1, coords2, marker="o")
        for i, txt in enumerate(nodes):
            ax4.annotate(int(txt), (coords1[i], coords2[i]))
        # Colorier les nœuds et le chemin entre action[0] et action[1] en rouge

        ax4.plot(
            coords1[action_start - 1 : action_end + 1],
            coords2[action_start - 1 : action_end + 1],
            color="red",
            marker="o",
        )
        ax4.scatter(coords1[action_start], coords2[action_start], color="red")
        ax4.scatter(coords1[action_end], coords2[action_end], color="red")
        ax4.set_title(f"Next solution {list(map(int, nodes))}")
        fig.suptitle(
            f"Action idx: {action.squeeze().tolist()}, Action node : {n1, n2}, "
            f"Acceptance : {acc.item()}, "
            f"Current Cost : {cost.item()}, next Cost : {costs[iter + 1].item()}, "
            f"Température : {temperature}, Iteration : {iter}",
        )

        plt.show()
        iter += 1


if __name__ == "__main__":
    main()
