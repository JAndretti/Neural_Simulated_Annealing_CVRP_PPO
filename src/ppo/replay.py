import random
from collections import deque, namedtuple
from typing import Deque, Optional
import torch

Transition = namedtuple(
    "Transition",
    (
        "state",
        "mask",
        "action",
        "next_state",
        "reward",
        "old_log_probs",
        "gamma",
    ),
)


class ReplayBuffer:
    """
    Experience replay buffer for reinforcement learning algorithms.

    Stores and manages transitions observed during training with:
    - Batch sampling
    - Device management (CPU/GPU)
    - Capacity control
    - Type safety checks

    Args:
        capacity: Maximum number of transitions to store
        device: Target device for tensor storage (default: CPU)
    """

    def __init__(
        self, capacity: int, device: torch.device = torch.device("cpu")
    ) -> None:
        self.capacity = capacity
        self.memory: Deque[Transition] = deque([], maxlen=capacity)
        self.device = device

    def push(
        self,
        state: torch.Tensor,
        mask: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        reward: torch.Tensor,
        old_log_probs: torch.Tensor,
        gamma: float = 0.99,
    ) -> None:
        """
        Store a transition in the buffer.

        Args:
            state: Current state tensor
            mask: Validity mask tensor (e.g., for invalid actions)
            action: Action tensor
            next_state: Next state tensor
            reward: Reward tensor (shape: [batch_size, 1])
            old_log_probs: Log probabilities of actions
            mask: Optional validity mask (e.g., for invalid actions)
            gamma: Discount factor
        """
        # Detach and clone to prevent memory leaks
        transition = Transition(
            state.detach().clone().to(self.device),
            mask.detach().clone().to(self.device),
            action.detach().clone().to(self.device),
            next_state.detach().clone().to(self.device),
            reward.detach().clone().to(self.device),
            old_log_probs.detach().clone().to(self.device),
            gamma,
        )
        self.memory.append(transition)

    def sample(self, batch_size: int) -> Transition:
        """
        Randomly sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            A Transition tuple where each field contains stacked tensors
        """
        if len(self.memory) < batch_size:
            batch_size = len(self.memory)

        samples = random.sample(self.memory, batch_size)
        return Transition(
            *[
                torch.stack([getattr(t, field) for t in samples])
                for field in Transition._fields
            ]
        )

    def pop(self) -> Optional[Transition]:
        """
        Remove and return the last transition from the buffer.

        Returns:
            The last transition or None if the buffer is empty.
        """
        if len(self.memory) == 0:
            return None
        return self.memory.pop()

    def clear(self) -> None:
        """Clear all transitions from the buffer."""
        self.memory.clear()

    def get_stats(self) -> dict:
        """
        Compute buffer statistics for monitoring.

        Returns:
            Dictionary containing:
            - reward_mean: Average reward
            - reward_std: Reward standard deviation
            - size: Current buffer size
        """
        if len(self.memory) == 0:
            return {"reward_mean": 0, "reward_std": 0, "size": 0}

        rewards = torch.stack([t.reward for t in self.memory])
        return {
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "size": len(self.memory),
        }

    def __len__(self) -> int:
        """Current number of stored transitions."""
        return len(self.memory)

    def to(self, device: torch.device) -> None:
        """
        Move all tensors to the specified device.

        Args:
            device: Target device (e.g., 'cuda' or 'cpu')
        """
        self.device = device
        for i, transition in enumerate(self.memory):
            self.memory[i] = Transition(
                *[t.to(device) if torch.is_tensor(t) else t for t in transition]
            )
