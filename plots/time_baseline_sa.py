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
    gpu = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else None
    )
    results_time_cpu = []
    results_cost_cpu = []
    results_time_model_cpu = []
    results_cost_model_cpu = []
    results_time_gpu = []
    results_cost_gpu = []
    results_time_model_gpu = []
    results_cost_model_gpu = []
    # for device in [gpu, "cpu"]:
    for device in [gpu]:
        if device is None:
            continue
        cfg["DEVICE"] = device
        logger.info(f"Using device: {cfg['DEVICE']}")
        problem, init_x, initial_cost = init_pb(cfg)

        OUTER_STEPS = [100, 1000, 5000, 10000, 20000, 50000]
        results_time = []
        results_cost = []
        results_time_model = []
        results_cost_model = []
        for outer_steps in tqdm(
            OUTER_STEPS,
            desc="Processing OUTER_STEPS for device: {}\n".format(device),
            leave=True,
        ):
            if device == "cpu" and outer_steps > 10000:
                logger.warning(
                    f"Skipping OUTER_STEPS={outer_steps} for CPU due to performance "
                    "concerns."
                )
                continue
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
        if cfg["DEVICE"] == "cpu":
            results_time_cpu = results_time
            results_cost_cpu = results_cost
            results_time_model_cpu = results_time_model
            results_cost_model_cpu = results_cost_model
        else:
            results_time_gpu = results_time
            results_cost_gpu = results_cost
            results_time_model_gpu = results_time_model
            results_cost_model_gpu = results_cost_model

    # Plotting the results
    fig, ax1 = plt.subplots()

    ax1.set_xlabel("OUTER_STEPS")
    ax1.set_ylabel("Time (seconds)")
    if results_time_cpu != [] and results_time_model_cpu != []:
        ax1.plot(
            OUTER_STEPS[:len(results_time_cpu)],
            results_time_cpu,
            marker="o",
            color="tab:blue",
            label="CPU SA Time",
        )
        ax1.plot(
            OUTER_STEPS[:len(results_time_model_cpu)],
            results_time_model_cpu,
            marker="^",
            color="tab:green",
            label="CPU NSA Time",
        )
    if results_time_gpu != [] and results_time_model_gpu != []:
        ax1.plot(
            OUTER_STEPS,
            results_time_gpu,
            marker="o",
            color="tab:orange",
            label="GPU SA Time",
        )
        ax1.plot(
            OUTER_STEPS,
            results_time_model_gpu,
            marker="^",
            color="tab:red",
            label="GPU NSA Time",
        )
    ax1.tick_params(axis="y")
    ax1.legend(loc="upper left")

    fig.tight_layout()  # to prevent overlap
    plt.title("Time vs STEPS")
    plt.savefig("plots/time_comparaison.png")
    plt.close()
    logger.info("Time comparison plot saved as 'plots/time_comparaison.png'.")
