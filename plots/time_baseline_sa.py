import os
import sys
import torch
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sa import sa_baseline, sa_test
from model import CVRPActor

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluation"))
)
from func import init_pb

from loguru import logger  # Enhanced logging capabilities

# Remove default logger
logger.remove()
# Add custom logger with colored output
logger.add(
    lambda msg: print(msg, end=""),
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "  # Timestamp in green
        "<blue>{file}:{line}</blue> | "  # File and line in blue
        "<yellow>{message}</yellow>"  # Message in yellow
    ),
    colorize=True,
)

if __name__ == "__main__":

    cfg = {
        "PROBLEM_DIM": 100,
        "N_PROBLEMS": 10000,
        "MAX_LOAD": 50,
        "OUTER_STEPS": 10000,
        "INNER_STEPS": 1,
        "METROPOLIS": True,
        "UPDATE_METHOD": "free",
        "DEVICE": "mps",
        "SEED": 0,
        "HEURISTIC": "insertion",
        "MIX1": "",
        "MIX2": "",
        "LOAD_PB": False,
        "CLUSTERING": False,
        "NB_CLUSTERS_MAX": 0,
        "INIT": "nearest_neighbor",
        "SCHEDULER": "step",
        "INIT_TEMP": 1,
        "STOP_TEMP": 0.01,
        "LOAD_PB": False,
        "NUM_H_LAYERS": 1,
        "ENTRY": 15,
        "EMBEDDING_DIM": 32,
        "CAPACITY_REWARD": True,
        "CAPACITY_REWARD_FACTOR": 0.1,
        "METHOD": "ppo",
        "REWARD": "immediate",
        "NEG_REWARD": 0,
        "NORMALIZE_REWARD": True,
    }

    problem, init_x, initial_cost = init_pb(cfg)

    OUTER_STEPS = [100, 1000, 5000, 10000, 20000, 50000]
    results_time = []
    results_cost = []
    results_time_model = []
    results_cost_model = []
    for outer_steps in tqdm(OUTER_STEPS, desc="Processing OUTER_STEPS \n", leave=True):
        cfg["OUTER_STEPS"] = outer_steps
        logger.info(f"Running SA baseline with OUTER_STEPS={outer_steps}")
        start_time = time.time()
        dict_solution = sa_baseline(problem, init_x, cfg)
        end_time = time.time()
        final_cost = torch.mean(dict_solution["min_cost"]).item()
        results_cost.append(final_cost)
        results_time.append(end_time - start_time)
        logger.info(
            f"OUTER_STEPS: {outer_steps}, Final Cost: {final_cost}, "
            f"Time: {end_time - start_time:.2f} seconds"
        )
        actor = CVRPActor(
            cfg["EMBEDDING_DIM"],
            cfg["ENTRY"],
            num_hidden_layers=cfg["NUM_H_LAYERS"],
            device=cfg["DEVICE"],
            mixed_heuristic=True if cfg["HEURISTIC"] == "mix" else False,
            method=cfg["UPDATE_METHOD"],
        )
        logger.info(f"Running NSA with OUTER_STEPS={outer_steps}")
        start_time = time.time()
        dict_solution = sa_test(
            actor,
            problem,
            init_x,
            cfg,
            replay_buffer=None,
            baseline=False,
            greedy=False,
            desc_tqdm="NSA Model",
        )
        end_time = time.time()
        final_cost = torch.mean(dict_solution["min_cost"]).item()
        results_time_model.append(end_time - start_time)
        results_cost_model.append(final_cost)
        logger.info(
            f"OUTER_STEPS: {outer_steps}, Final Cost: {final_cost}, "
            f"Time: {end_time - start_time:.2f} seconds"
        )
    # Plotting the results
    fig, ax1 = plt.subplots()

    ax1.set_xlabel("OUTER_STEPS")
    ax1.set_ylabel("Time (seconds)")
    ax1.plot(OUTER_STEPS, results_time, marker="o", color="tab:blue", label="SA Time")
    ax1.plot(
        OUTER_STEPS, results_time_model, marker="^", color="tab:green", label="NSA Time"
    )
    ax1.tick_params(axis="y")
    ax1.legend(loc="upper left")

    plt.title("Time vs STEPS")

    fig.tight_layout()  # to prevent overlap
    plt.title("Time vs STEPS")
    plt.savefig("plots/time_comparaison.png")
    plt.close()
    logger.info("Time comparison plot saved as 'plots/time_comparaison.png'.")

    # Adding a second plot for cost comparison
    fig, ax2 = plt.subplots()

    ax2.set_xlabel("OUTER_STEPS")
    ax2.set_ylabel("Cost")
    ax2.plot(OUTER_STEPS, results_cost, marker="o", color="tab:red", label="SA Cost")
    ax2.plot(
        OUTER_STEPS,
        results_cost_model,
        marker="^",
        color="tab:purple",
        label="NSA Cost",
    )
    ax2.tick_params(axis="y")
    ax2.legend(loc="upper right")

    plt.title("Cost vs STEPS")
    fig.tight_layout()  # to prevent overlap
    plt.savefig("plots/cost_comparaison.png")
    plt.close()
    logger.info("Cost comparison plot saved as 'plots/cost_comparaison.png'.")
