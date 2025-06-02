import os
from glob import glob
import matplotlib.pyplot as plt
import torch
from func import get_HP_for_model, set_seed, load_model
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from model import CVRPActor


MODEL_NAME = "20250521_124544_qkjmcthi"
MODEL_DIR = glob(
    os.path.join("wandb", "Neural_Simulated_Annealing", "*", "models", MODEL_NAME)
)[0]
set_seed(42)  # For reproducibility


def init_problem_parameters():
    HP = get_HP_for_model(MODEL_DIR)
    CFG = {
        "DEVICE": HP.get("DEVICE", "cpu"),
        "HEURISTIC": HP.get("HEURISTIC", "mix"),
        "EMBEDDING_DIM": HP.get("EMBEDDING_DIM", 32),
        "NUM_H_LAYERS": HP.get("NUM_H_LAYERS", 1),
    }
    return CFG


def create_test_state(batch_size=1, node_count=1, device="cpu"):
    """
    Create a test state with default values for all parameters.

    Args:
        batch_size: Number of problems in the batch
        node_count: Number of nodes in each problem
        device: Device to create the tensor on

    Returns:
        A tensor of shape [batch_size, node_count, feature_dim]
    """
    # Default values (all 0.5 except for temperatures and time)
    default_val = 0.5

    # Initialize the state tensor
    # Format: [x, coords_x, coords_y, ...extra_features]
    feature_dim = 12  # coords(x,y) + prev(x,y) + next(x,y) + extra features
    state = torch.ones(batch_size, node_count, feature_dim, device=device) * default_val
    # Set specific coordinate values
    # Current coordinates
    state[:, :, 0] = 0.10  # coords_x
    state[:, :, 1] = 0.15  # coords_y

    # Previous coordinates
    state[:, :, 2] = 0.60  # prev_x
    state[:, :, 3] = 0.68  # prev_y

    # Next coordinates
    state[:, :, 4] = 0.40  # next_x
    state[:, :, 5] = 0.50  # next_y

    base = state[:, :, :-2]
    state2 = (
        torch.ones(batch_size, node_count, feature_dim, device=device) * default_val
    )
    # Set specific coordinate values
    # Current coordinates
    state2[:, :, 0] = 0.22  # coords_x
    state2[:, :, 1] = 0.28  # coords_y

    # Previous coordinates
    state2[:, :, 2] = 0.54  # prev_x
    state2[:, :, 3] = 0.69  # prev_y

    # Next coordinates
    state2[:, :, 4] = 0.74  # next_x
    state2[:, :, 5] = 0.29  # next_y
    state2 = torch.cat((base, state2), -1)

    return state, state2


def test_input_sensitivity(actor, param_name, feature_idx, device="cpu"):
    """
    Test the sensitivity of the model to changes in a specific input parameter.

    Args:
        actor: The actor model to test
        param_name: The name of the parameter being varied (for plotting)
        feature_idx: The index of the feature in the state tensor
        device: Device to create tensors on

    Returns:
        Two lists of values and corresponding outputs
    """
    test_values = torch.linspace(0, 1, 21, device=device)

    # Create a base test state
    state1, state2 = create_test_state(batch_size=1, device=device)
    state1 = state1.repeat((1, test_values.shape[0], 1))
    state1[:, :, feature_idx] = test_values

    state2 = state2.repeat((1, test_values.shape[0], 1))
    state2[:, :, feature_idx + 10] = test_values
    with torch.no_grad():

        # Process through city1_net
        c1_output = actor.city1_net(state1)[..., 0]
        c1_probs = torch.softmax(c1_output, dim=-1)

        # Process through city2_net
        c2_output = actor.city2_net(state2)[..., 0]
        c2_probs = torch.softmax(c2_output, dim=-1)

    return test_values, c1_probs.squeeze(), c2_probs.squeeze()


def plot_sensitivity(param_name, values, c1_outputs, c2_outputs):
    """
    Plot the results of a sensitivity test.

    Args:
        param_name: The name of the parameter that was varied
        values: The values that the parameter took
        c1_outputs: The outputs from city1_net
        c2_outputs: The outputs from city2_net
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(values, c1_outputs)
    plt.title(f"city1_net Response to {param_name}")
    plt.xlabel(f"{param_name} Value")
    plt.ylabel("Probability of first node")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(values, c2_outputs)
    plt.title(f"city2_net Response to {param_name}")
    plt.xlabel(f"{param_name} Value")
    plt.ylabel("Probability of first node")
    plt.grid(True)

    plt.tight_layout()

    # Create directory for saving plots if it doesn't exist
    os.makedirs("sensitivity_plots", exist_ok=True)
    plt.savefig(f"sensitivity_plots/{param_name}_sensitivity.png")
    plt.close()


def main():
    CFG = init_problem_parameters()

    actor = CVRPActor(
        CFG["EMBEDDING_DIM"],
        num_hidden_layers=CFG["NUM_H_LAYERS"],
        device=CFG["DEVICE"],
        mixed_heuristic=True if CFG["HEURISTIC"] == "mix" else False,
    )
    actor = load_model(actor, MODEL_DIR, "actor")

    # Define feature names and their indices in the state tensor
    city1_features = [
        ("coordx", 1),
        ("coordy", 2),
        ("coordprevx", 3),
        ("coordprevy", 4),
        ("coordnextx", 5),
        ("coordnexty", 6),
        ("nodedemand", 7),
        ("routedemand", 8),
        ("nodedemandcapacity", 9),
        ("noderoutecost", 10),
        ("temperature", 11),
        ("time", 12),
    ]

    print("Starting sensitivity analysis...")

    # Test each feature's sensitivity
    for param_name, feature_idx in city1_features:
        print(f"Testing parameter: {param_name}")
        values, c1_outputs, c2_outputs = test_input_sensitivity(
            actor, param_name, feature_idx - 1, device=CFG["DEVICE"]
        )
        plot_sensitivity(param_name, values, c1_outputs, c2_outputs)

    print(
        "Sensitivity analysis complete. Results saved in 'sensitivity_plots' directory."
    )

    return actor


if __name__ == "__main__":
    model = main()
