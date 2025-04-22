import os
import sys
import matplotlib.pyplot as plt
from func import get_HP_for_model, set_seed, load_model

# Add src path to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from problem import CVRP
from sa import sa
from model import CVRPActorPairs, CVRPActor

# --- Configurations ---
MODEL_FOLDER = "heuristic_benchmark"
MODEL_PATH = os.path.join("wandb", "Neural_Simulated_Annealing", MODEL_FOLDER, "models")
MODEL_NAME = "20250418_161856_49z48neh"
MODEL_DIR = os.path.join(MODEL_PATH, MODEL_NAME)


def init_problem_parameters():
    HP = get_HP_for_model(MODEL_DIR)
    CFG = {
        "PROBLEM_DIM": HP.get("PROBLEM_DIM", 20),
        "N_PROBLEMS": 1,
        "MAX_LOAD": HP.get("MAX_LOAD", 30),
        "DEVICE": HP.get("DEVICE", "cpu"),
        "INNER_STEPS": HP.get("INNER_STEPS", 1),
        "OUTER_STEPS": 1000,
        "SCHEDULER": HP.get("SCHEDULER", "lam"),
        "HEURISTIC": HP.get("HEURISTIC", "mix"),
        "INIT_TEMP": HP.get("INIT_TEMP", 1.0),
        "STOP_TEMP": HP.get("STOP_TEMP", 0.1),
        "METHOD": HP.get("METHOD", "ppo"),
        "REWARD": HP.get("REWARD", "immediate"),
        "PAIRS": HP.get("PAIRS", False),
        "EMBEDDING_DIM": HP.get("EMBEDDING_DIM", 32),
        "NUM_H_LAYERS": HP.get("NUM_H_LAYERS", 1),
        "GAMMA": HP.get("GAMMA", 0.99),
        "name": None,
        "CLUSTERING": False,
    }
    return CFG


def main():
    set_seed(42)  # For reproducibility
    CFG = init_problem_parameters()

    # Initialize problem
    problem = CVRP(
        CFG["PROBLEM_DIM"],
        CFG["N_PROBLEMS"],
        CFG["MAX_LOAD"],
        device=CFG["DEVICE"],
        params=CFG,
    )
    problem.manual_seed(0)
    params = problem.generate_params()
    params = {k: v.to(CFG["DEVICE"]) for k, v in params.items()}
    problem.set_params(params)
    init_x = problem.generate_init_x()

    # Initialize actor
    if CFG["PAIRS"]:
        actor = CVRPActorPairs(
            CFG["EMBEDDING_DIM"],
            num_hidden_layers=CFG["NUM_H_LAYERS"],
            device=CFG["DEVICE"],
            mixed_heuristic=True if CFG["HEURISTIC"] == "mix" else False,
        )
    else:
        actor = CVRPActor(
            CFG["EMBEDDING_DIM"],
            num_hidden_layers=CFG["NUM_H_LAYERS"],
            device=CFG["DEVICE"],
            mixed_heuristic=True if CFG["HEURISTIC"] == "mix" else False,
        )
    actor = load_model(actor, MODEL_DIR)

    # Run experiment with record_state=True
    result = sa(
        actor,
        problem,
        init_x,
        CFG,
        record_state=True,
    )

    # Retrieve cost, temperature, and acceptance evolution
    costs = [c.item() for c in result["costs"]]
    temperatures = [t.item() for t in result["temperature"]]
    acceptance = [
        a.float().mean().item() if hasattr(a, "float") else float(a)
        for a in result["acceptance"]
    ]

    # Compute cumulative acceptance (running mean)
    acceptance_cum = []
    count_ones = 0
    for i, a in enumerate(acceptance):
        if a == 1:
            count_ones += 1
        acceptance_cum.append(count_ones / (i + 1))

    # If HEURISTIC == "mix", compute the heuristic usage ratio
    ratio_curve = None
    if CFG["HEURISTIC"] == "mix" and "heuristic" in result:
        h_choices = result["heuristic"]
        # Compute the running ratio of 1s (e.g., usage of one heuristic) over total
        # (1s + 0s)
        ratio_curve = []
        count_ones = 0
        for i, h in enumerate(h_choices):
            if h == 1:
                count_ones += 1
            ratio_curve.append(count_ones / (i + 1))

    # Find the first index where the minimum cost is reached
    min_cost = min(costs)
    min_idx = costs.index(min_cost)

    # Plot cost, temperature, cumulative acceptance, and ratio if mix
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(costs, color="tab:blue", label="Cost")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Cost", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.axvline(
        min_idx, color="tab:purple", linestyle="--", label="min cost", alpha=0.7
    )

    ax2 = ax1.twinx()
    ax2.plot(temperatures, color="tab:red", label="Temperature", alpha=0.7)
    ax2.set_ylabel("Temperature", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # Third axis for cumulative acceptance
    ax3 = ax1.twinx()
    ax3.spines.right.set_position(("axes", 1.22))
    ax3.plot(
        acceptance_cum, color="tab:orange", label="Cumulative Acceptance", alpha=0.8
    )
    ax3.set_ylabel("Cumulative Acceptance", color="tab:orange")
    ax3.tick_params(axis="y", labelcolor="tab:orange")

    # Fourth axis for heuristic ratio if mix
    if ratio_curve is not None:
        ax4 = ax1.twinx()
        ax4.spines.right.set_position(("axes", 1.34))
        ax4.plot(
            ratio_curve, color="tab:green", label="Heuristic Ratio (mix)", alpha=0.8
        )
        ax4.set_ylabel("Heuristic Ratio", color="tab:green")
        ax4.tick_params(axis="y", labelcolor="tab:green")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    all_lines = lines1 + lines2 + lines3
    all_labels = labels1 + labels2 + labels3
    if ratio_curve is not None:
        lines4, labels4 = ax4.get_legend_handles_labels()
        all_lines += lines4
        all_labels += labels4
    ax1.legend(
        all_lines,
        all_labels,
        loc="upper right",
    )

    plt.title("Evolution of Cost, Temperature, Acceptance during SA")
    fig.tight_layout()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
