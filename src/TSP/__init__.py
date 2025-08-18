from .model import get_features, load_model
import torch

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
TSP_actor = load_model("src/TSP/model_TSP.pt", device=device)
print(f"TSP actor loaded on {device}")
