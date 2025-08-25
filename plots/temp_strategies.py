import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sa import Scheduler
import matplotlib.pyplot as plt


def plot_temperature_strategies():
    T_max = 1
    T_min = 0.1
    step_max = 1000

    schedulers = {
        "cyclic": Scheduler(
            "cyclic", T_max=T_max, T_min=T_min, step_max=step_max, mode="triangular2"
        ),
        # "cyclic2": Scheduler(
        #     "cyclic", T_max=T_max, T_min=T_min, step_max=step_max, mode="triangular"
        # ),
        # "cosine": Scheduler("cosine", T_max=T_max, T_min=T_min, step_max=step_max),
        "lam": Scheduler("lam", T_max=T_max, T_min=T_min, step_max=step_max),
        "step": Scheduler("step", T_max=T_max, T_min=T_min, step_max=step_max),
    }

    steps = range(step_max)
    plt.figure(figsize=(10, 6))

    for name, scheduler in schedulers.items():
        values = [scheduler.step(step).item() for step in steps]
        plt.plot(steps, values, label=name)

    plt.xlabel("Steps")
    plt.ylabel("Temperature")
    plt.title("Temperature Schedulers (1 → 0.1, 1000 steps)")
    plt.legend()
    plt.grid(True)

    out_path = os.path.join(os.path.dirname(__file__), "temp_strategies.png")
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    plot_temperature_strategies()
