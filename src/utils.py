import numpy as np
import torch
import matplotlib.pyplot as plt


def extend(tensor: torch.Tensor, dims: int) -> torch.Tensor:
    """Extend tensor to match dimensions 'dims'."""
    return tensor[(...,) + (None,) * dims]


def extend_to(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """Extend tensor1 to have same number of dims as tensor2."""
    return extend(tensor1, len(tensor2.shape) - len(tensor1.shape))


def repeat_to(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """Repeat tensor1 to match the shape of tensor2."""
    tensor1 = extend_to(tensor1, tensor2)
    ones = torch.ones(tensor2.shape[:-1] + (1,), device=tensor1.device)
    return tensor1 * ones


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array."""
    return tensor.detach().cpu().numpy()


def convert_tensor(a: torch.Tensor) -> torch.Tensor:
    """Convert a tensor to a tensor of indices where values change."""
    # Create a mask to identify indices where values change
    diff = (a[:, 1:] != a[:, :-1]) & (a[:, 1:] != 0)

    # Initialize a result tensor with zeros
    result = torch.zeros_like(a, dtype=torch.int)

    # Fill the first column of result with zeros
    result[:, 0] = 0

    # Use a tensor to increment indices where there is a non-zero value change
    indices = torch.cumsum(diff, dim=1)
    # Add the last value of each row to indices
    indices = torch.cat((indices, indices[:, -1].unsqueeze(1)), dim=1)
    return indices


def plot_CVRP(ax, nodes, x_coords, y_coords, title="Solution"):
    """Plot the CVRP solution."""
    zero_indices = [i for i, v in enumerate(nodes) if v == 0]
    colors = plt.get_cmap(None, len(zero_indices) - 1)
    for i in range(len(zero_indices) - 1):
        start = zero_indices[i]
        end = zero_indices[i + 1]

        segment_x = x_coords[start : end + 1]
        segment_y = y_coords[start : end + 1]
        segment_values = nodes[start : end + 1]

        ax.plot(
            segment_x,
            segment_y,
            color=colors(i),
            marker="o",
            label=f"Segment {i + 1}",
        )
        for x, y, val in zip(segment_x, segment_y, segment_values):
            ax.text(x, y, str(int(val)), fontsize=9, ha="center", va="bottom")
    ax.set_title(f"{title} : {list(map(int, nodes))}")
