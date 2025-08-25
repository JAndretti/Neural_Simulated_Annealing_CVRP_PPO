from .model import get_features, load_model
import torch

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
TSP_actor = load_model("src/TSP/model_TSP.pt", device=device)

__all__ = ["get_features", "load_model", "TSP_actor"]

# print(f"TSP actor loaded on {device}")
