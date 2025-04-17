import os

for folder in ["wandb", "res", "bdd"]:
    os.makedirs(folder, exist_ok=True)
