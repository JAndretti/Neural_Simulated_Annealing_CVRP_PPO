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
    Proximal Policy Optimization (PPO) implementation for CVRP.
    Optimizes both actor (policy) and critic (value function) networks.

    Args:
        actor: Policy network that selects actions
        critic: Value network that estimates state values
        replay: Experience replay buffer storing transitions
        actor_opt: Optimizer for the actor network
        critic_opt: Optimizer for the critic network
        cfg: Configuration dictionary with hyperparameters
        problem: CVRP problem environment
        criterion: Loss function (default: MSE)

    Returns:
        Tuple containing actor and critic losses
    """

    # Hyperparameters
    ppo_epochs = cfg["PPO_EPOCHS"]  # Number of PPO optimization epochs
    trace_decay = cfg["TRACE_DECAY"]  # λ for GAE (Generalized Advantage Estimation)
    eps_clip = cfg["EPS_CLIP"]  # Clipping range for policy ratio
    batch_size = cfg["BATCH_SIZE"]  # Mini-batch size
    n_problems = cfg["N_PROBLEMS"]  # Number of parallel problem instances
    problem_dim = int(
        cfg["PROBLEM_DIM"] + (cfg["PROBLEM_DIM"] * 0.40) + 1
    )  # State dimension
    device = cfg["DEVICE"]  # Device (CPU/GPU)

    # Set networks to training mode
    actor.train()
    critic.train()

    # 1. Extract and process transitions from replay buffer
    with torch.no_grad():
        transitions = replay.memory
        nt = len(transitions)  # Total transitions
        batch = Transition(*zip(*transitions))  # Convert to structured batch

        # Stack and reshape tensors
        state = torch.stack(batch.state).view(nt * n_problems, problem_dim, -1)
        action = torch.stack(batch.action).detach().view(nt * n_problems, -1)
        next_state = (
            torch.stack(batch.next_state)
            .detach()
            .view(nt * n_problems, problem_dim, -1)
        )
        old_log_probs = torch.stack(batch.old_log_probs).view(nt * n_problems, -1)

        # Compute state values V(s) and V(s')
        state_values = critic(state).view(nt, n_problems, 1)
        next_state_values = critic(next_state).view(nt, n_problems, 1)

        # 2. Compute returns and advantages using Generalized Advantage Estimation (GAE)
        rewards_to_go = torch.zeros(
            (nt, n_problems, 1), device=device
        )  # R_t = cumulative returns
        advantages = torch.zeros((nt, n_problems, 1), device=device)  # A_t = advantages
        discounted_reward = torch.zeros(
            (n_problems, 1), device=device
        )  # Temporary storage
        advantage = torch.zeros((n_problems, 1), device=device)  # Temporary storage

        # Backward pass through time to compute returns and advantages
        for i, reward, gamma in zip(
            reversed(np.arange(len(transitions))),
            reversed(batch.reward),
            reversed(batch.gamma),
        ):
            if gamma == 0:  # Reset at episode boundaries
                discounted_reward.zero_()
                advantage.zero_()

            # Compute return: R_t = r_t + γ * R_{t+1}
            discounted_reward = reward + (gamma * discounted_reward)

            # Compute TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            td_error = reward + gamma * next_state_values[i, ...] - state_values[i, ...]

            # Compute advantage: A_t = δ_t + γλ * A_{t+1}
            advantage = td_error + gamma * trace_decay * advantage

            rewards_to_go[i, ...] = discounted_reward
            advantages[i, ...] = advantage

        # Normalize advantages: A_t = (A_t - μ_A) / (σ_A + ε)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Flatten tensors for optimization
    advantages = advantages.view(n_problems * nt, -1)
    rewards_to_go = rewards_to_go.view(n_problems * nt, -1)

    # 3. PPO Optimization
    for _ in range(ppo_epochs):
        actor_opt.zero_grad()
        critic_opt.zero_grad()

        # Shuffle transitions for better training
        if nt > 1:
            perm = np.arange(state.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(device)
            state = state[perm, :].clone()
            action = action[perm, :].clone()
            rewards_to_go = rewards_to_go[perm, :].clone()
            advantages = advantages[perm, :].clone()
            old_log_probs = old_log_probs[perm, :].clone()

        # Mini-batch training
        for j in range(nt * n_problems, 0, -batch_size):
            nb = min(j, batch_size)
            if nb <= 1:  # Skip batches that are too small
                continue

            batch_idx = np.arange(j - nb, j)
            batch_state = state[batch_idx, ...]
            batch_action = action[batch_idx, ...]
            batch_advantages = advantages[batch_idx, 0]
            batch_rewards_to_go = rewards_to_go[batch_idx, 0]
            batch_old_log_probs = old_log_probs[batch_idx, 0]

            # Forward pass
            batch_state_values = critic(batch_state)
            batch_log_probs = actor.evaluate(batch_state, batch_action)

            # Check for numerical instability
            if (
                torch.isnan(batch_state_values).any()
                or torch.isnan(batch_log_probs).any()
            ):
                print("NaN detected in model outputs. Skipping batch.")
                continue

            # Critic loss: L^VF = 0.5 * (V_θ(s_t) - R_t)^2
            critic_loss = 0.5 * criterion(
                batch_state_values.squeeze(), batch_rewards_to_go.detach()
            )

            # Policy loss: L^CLIP = -E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
            # where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
            ratios = torch.exp(batch_log_probs - batch_old_log_probs.detach())
            surr1 = ratios * batch_advantages.detach()
            surr2 = (
                torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip)
                * batch_advantages.detach()
            )
            actor_loss = -torch.min(surr1, surr2).mean()

            if torch.isnan(actor_loss) or torch.isnan(critic_loss):
                print("NaN detected in loss calculations. Skipping batch.")
                continue

            # Backward pass with gradient clipping
            actor_loss.backward()
            critic_loss.backward()

            # Gradient clipping: ||∇θ|| ≤ clip_norm
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)

            # Check for NaN gradients
            actor_grad_nan = any(
                torch.isnan(p.grad).any()
                for p in actor.parameters()
                if p.grad is not None
            )
            critic_grad_nan = any(
                torch.isnan(p.grad).any()
                for p in critic.parameters()
                if p.grad is not None
            )

            if actor_grad_nan or critic_grad_nan:
                print("NaN detected in gradients. Skipping batch.")
                actor_opt.zero_grad()
                critic_opt.zero_grad()
                continue

            # Update parameters
            actor_opt.step()
            critic_opt.step()

    return actor_loss.item(), critic_loss.item()
