from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from model import SAModel
from replay import Replay, Transition
from problem import CVRP


def ppo(
    actor: SAModel,
    critic: nn.Module,
    replay: Replay,
    actor_opt: Optimizer,
    critic_opt: Optimizer,
    cfg: dict,
    problem: CVRP,
    criterion=torch.nn.MSELoss(),
) -> Tuple[float, float]:
    """
    Optimizes the actor and critic in PPO for 'ppo_epochs' epochs using transitions
    recorded in the 'replay' memory.

    Parameters
    ----------
    actor, critic: nn.Module
        Actor and Critic models.
    replay: Replay object
        Memory storing transitions (see replay.py).
    actor_opt, critic_opt: torch.optim
        Optimizers for the actor and critic.
    cfg: dict
        Configuration containing PPO hyperparameters.
    criterion: torch loss
        Loss function for the critic.

    Returns
    -------
    actor_loss, critic_loss: float
        Actor and Critic losses.
    """

    # PPO Hyperparameters
    ppo_epochs = cfg["PPO_EPOCHS"]  # Number of optimization epochs
    trace_decay = cfg["TRACE_DECAY"]  # Decay for TD(λ)
    eps_clip = cfg["EPS_CLIP"]  # Clipping factor for PPO
    batch_size = cfg["BATCH_SIZE"]  # Batch size
    n_problems = cfg["N_PROBLEMS"]  # Number of problems
    problem_dim = int(
        cfg["PROBLEM_DIM"] + (cfg["PROBLEM_DIM"] * 0.40) + 1
    )  # Problem dimensionality 40% for match padding +1 to add depot
    device = cfg["DEVICE"]  # Device (CPU or GPU)
    if cfg["PAIRS"]:
        mask_dim = int((problem_dim * (problem_dim - 1)) / 2)
    else:
        mask_dim = problem_dim
    actor.train()
    critic.train()

    # **1. Extract transitions from memory**
    with torch.no_grad():
        transitions = replay.memory  # Transitions stored in the replay buffer
        nt = len(transitions)  # Total number of transitions
        batch = Transition(*zip(*transitions))  # Convert to structured batch

        # Format tensors for states, actions, etc.
        state = torch.stack(batch.state).view(nt * n_problems, problem_dim, -1)
        action = torch.stack(batch.action).detach().view(nt * n_problems, -1)
        mask = torch.stack(batch.mask).detach().view(nt * n_problems, mask_dim)
        next_state = (
            torch.stack(batch.next_state)
            .detach()
            .view(nt * n_problems, problem_dim, -1)
        )
        old_log_probs = torch.stack(batch.old_log_probs).view(nt * n_problems, -1)

        # Evaluate state values using the critic
        state_values = critic(state).view(nt, n_problems, 1)
        next_state_values = critic(next_state).view(nt, n_problems, 1)

        # **2. Compute rewards and advantages**
        rewards_to_go = torch.zeros(
            (nt, n_problems, 1), device=device, dtype=torch.float32
        )  # Cumulative rewards
        advantages = torch.zeros(
            (nt, n_problems, 1), device=device, dtype=torch.float32
        )  # Advantage estimates
        discounted_reward = torch.zeros((n_problems, 1), device=device)
        advantage = torch.zeros((n_problems, 1), device=device)

        # Reverse traversal of transitions to compute cumulative rewards
        for i, reward, gamma in zip(
            reversed(np.arange(len(transitions))),
            reversed(batch.reward),
            reversed(batch.gamma),
        ):
            if gamma == 0:
                # Reset at episode boundaries
                discounted_reward = torch.zeros((n_problems, 1), device=device)
                advantage = torch.zeros((n_problems, 1), device=device)

            # Compute cumulative reward:
            # \( R_t = r_t + \gamma R_{t+1} \)
            discounted_reward = reward + (gamma * discounted_reward)

            # Compute advantage:
            # \( A_t = \delta_t + \gamma \lambda A_{t+1} \),
            # where \( \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \) (TD error)
            td_error = reward + gamma * next_state_values[i, ...] - state_values[i, ...]
            advantage = td_error + gamma * trace_decay * advantage

            rewards_to_go[i, ...] = discounted_reward
            advantages[i, ...] = advantage

        # Normalize advantages:
        # \( \hat{A_t} = \frac{A_t - \mu(A)}{\sigma(A) + \epsilon} \)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Flatten for optimization
    advantages = advantages.view(n_problems * nt, -1)
    rewards_to_go = rewards_to_go.view(n_problems * nt, -1)

    actor_loss, critic_loss = None, None

    # **3. Optimization over multiple epochs**
    for _ in range(ppo_epochs):
        actor_opt.zero_grad()
        critic_opt.zero_grad()

        # Shuffle transitions to avoid training biases
        if nt > 1:
            perm = np.arange(state.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(device)
            state = state[perm, :].clone()
            mask = mask[perm, :].clone()
            action = action[perm, :].clone()
            rewards_to_go = rewards_to_go[perm, :].clone()
            advantages = advantages[perm, :].clone()
            old_log_probs = old_log_probs[perm, :].clone()

            # Batch training
            for j in range(nt * n_problems, 0, -batch_size):
                nb = min(j, batch_size)
                if nb <= 1:  # Avoid instabilities
                    continue

                # Select a batch
                batch_idx = np.arange(j - nb, j)
                batch_state = state[batch_idx, ...]
                batch_mask = mask[batch_idx, ...]
                batch_action = action[batch_idx, ...]
                batch_advantages = advantages[batch_idx, 0]
                batch_rewards_to_go = rewards_to_go[batch_idx, 0]
                batch_old_log_probs = old_log_probs[batch_idx, 0]

                # Evaluate critic and actor
                batch_state_values = critic(batch_state)
                batch_log_probs = actor.evaluate(batch_state, batch_action, batch_mask)

                # Critic loss:
                # \( \mathcal{L}_c = \frac{1}{2} \left(V(s_t) - R_t\right)^2 \)
                critic_loss = 0.5 * criterion(
                    batch_state_values.squeeze(), batch_rewards_to_go.detach()
                )

                # Actor loss:
                # \( \mathcal{L}_a = - \mathbb{E}_t \left[\min(r_t A_t,
                # \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)\right] \)
                ratios = torch.exp(batch_log_probs - batch_old_log_probs.detach())
                surr1 = ratios * batch_advantages.detach()
                surr2 = (
                    torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip)
                    * batch_advantages.detach()
                )
                actor_loss = -torch.min(surr1, surr2).mean()

                # Gradient optimization
                actor_loss.backward()
                critic_loss.backward()
                actor_opt.step()
                critic_opt.step()

    return actor_loss.item(), critic_loss.item()
