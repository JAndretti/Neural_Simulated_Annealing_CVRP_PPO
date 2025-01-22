# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.


import numpy as np
import torch


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
