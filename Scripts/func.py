import os
import random
import numpy as np
import torch
import yaml


# Load model hyperparameters
def get_HP_for_model(model_name):
    """Load hyperparameters for a given model."""
    hp_file = os.path.join(model_name, "HP.yaml")
    with open(hp_file, "r") as file:
        content = file.read()
        content_clean = content.replace("!!python/object:HP._HP", "")
        hp_data = yaml.unsafe_load(content_clean)
        return hp_data.get("config", {})


def set_seed(seed=0):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Load model weights
def extract_loss(filename):
    """Extract loss value from filename."""
    try:
        return float((filename.split("_")[-1])[:-3])
    except ValueError:
        return float("inf")


# Load model with smallest loss
def load_model(model, folder):
    """Load model with smallest loss."""
    files = [f for f in os.listdir(folder) if f.endswith(".pt")]
    if files:
        best_file = min(files, key=extract_loss)
        model.load_state_dict(
            torch.load(os.path.join(folder, best_file), weights_only=True)
        )
    return model
