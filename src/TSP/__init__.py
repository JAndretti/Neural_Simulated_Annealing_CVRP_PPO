from .model import get_features, load_model

TSP_actor = load_model("src/TSP/model_TSP.pt", device="cpu")
