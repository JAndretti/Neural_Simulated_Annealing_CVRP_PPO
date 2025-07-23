import torch
import matplotlib.pyplot as plt


class Scheduler:
    def __init__(self, scheduler_type, **kwargs):
        self.scheduler_type = scheduler_type
        self.scheduler = None
        if scheduler_type == "cyclic":
            self.scheduler = CyclicLR(**kwargs)
        elif scheduler_type == "cosine":
            self.scheduler = CosineAnnealingWarmRestarts(**kwargs)
        elif scheduler_type == "lam":
            self.scheduler = LambdaLR(**kwargs)
        elif scheduler_type == "step":
            self.scheduler = Step(**kwargs)
        else:
            raise ValueError(f"Scheduler de type {scheduler_type} non supporté.")

    def step(self, step):
        return self.scheduler.step(step)


class CyclicLR:

    def __init__(self, T_max, T_min, step_max, step_size_up=20, mode="triangular2"):
        self.T_min = T_min
        self.T_max = T_max
        self.step_max = step_max
        self.step_size_up = step_size_up
        self.mode = mode

    def step(self, step):
        step = torch.tensor(step)
        cycle = torch.floor(1 + step / (2 * self.step_size_up))
        x = torch.abs(step / self.step_size_up - 2 * cycle + 1)

        if self.mode == "triangular":
            return (
                (
                    self.T_min
                    + (self.T_max - self.T_min)
                    * torch.maximum(torch.tensor(0.0), 1 - x)
                )
                .clone()
                .detach()
            )
        elif self.mode == "triangular2":
            return (
                (
                    self.T_min
                    + (self.T_max - self.T_min)
                    * torch.maximum(torch.tensor(0.0), 1 - x)
                    / (2 ** (cycle - 1))
                )
                .clone()
                .detach()
            )
        else:
            raise ValueError("Mode non supporté")


class CosineAnnealingWarmRestarts:

    def __init__(self, T_max, T_min, step_max, T_0=20, T_mult=1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.step_max = step_max
        self.T_min = T_min
        self.T_max = T_max

    def step(self, step):
        T_cur = step
        T_i = self.T_0
        while T_cur >= T_i:
            T_cur -= T_i
            T_i *= self.T_mult
        return (
            (
                self.T_min
                + (self.T_max - self.T_min)
                * (1 + torch.cos(torch.tensor(torch.pi * T_cur / T_i)))
                / 2
            )
            .clone()
            .detach()
        )


class LambdaLR:
    def __init__(self, T_max, T_min, step_max):
        self.T_max = T_max
        self.T_min = T_min
        self.step_max = step_max
        self.lambda_factor = (T_min / T_max) ** (1 / step_max)

    def step(self, epoch):
        return torch.tensor(self.T_max * (self.lambda_factor**epoch))


class Step:
    def __init__(self, T_max, T_min, step_max):
        self.T_max = T_max
        self.temp = T_max
        self.T_min = T_min
        self.step_max = step_max
        self.lambda_factor = (T_min / T_max) ** (1 / step_max)

    def step(self, epoch):
        if (
            epoch == int(self.step_max / 2)
            or epoch == int(2 * self.step_max / 3)
            or epoch == int(3 * self.step_max / 4)
            or epoch == int(6 * self.step_max / 7)
        ):
            if epoch == int(6 * self.step_max / 7):
                self.temp = self.T_min
            else:
                self.temp = self.temp / 2
        return torch.tensor(self.temp)


def main():
    T_max = 1
    T_min = 0.01
    step_max = 100

    schedulers = {
        "cyclic": CyclicLR(T_max=T_max, T_min=T_min, step_max=step_max),
        "cyclic2": CyclicLR(
            T_max=T_max, T_min=T_min, step_max=step_max, mode="triangular"
        ),
        "cosine": CosineAnnealingWarmRestarts(
            T_max=T_max, T_min=T_min, step_max=step_max
        ),
        "lam": LambdaLR(T_max=T_max, T_min=T_min, step_max=step_max),
        "step": Step(T_max=T_max, T_min=T_min, step_max=step_max),
    }

    steps = range(step_max)
    plt.figure(figsize=(10, 6))

    for name, scheduler in schedulers.items():
        values = [scheduler.step(step).item() for step in steps]
        plt.plot(steps, values, label=name)

        plt.xlabel("Steps")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedulers")
        plt.legend()
        plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
