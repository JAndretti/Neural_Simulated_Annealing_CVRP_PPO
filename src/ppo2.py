from typing import Tuple
import torch
from torch import nn
from torch.optim import Optimizer
from model import SAModel
from replay import ReplayBuffer, Transition
from loguru import logger


def gradient_penalty(critic, states):
    """
    Calculates the gradient penalty for the critic network.
    This helps enforce Lipschitz continuity for more stable training.

    Args:
        critic: The critic network
        states: The input states to evaluate

    Returns:
        The gradient penalty loss term
    """
    states.requires_grad_(True)
    values = critic(states)
    gradients = torch.autograd.grad(
        outputs=values,
        inputs=states,
        grad_outputs=torch.ones_like(values),
        create_graph=True,
        retain_graph=True,
    )[0]
    return ((gradients.norm(2.0, dim=1) - 1) ** 2).mean()


def ppo(
    actor: SAModel,
    critic: nn.Module,
    pb_dim: int,
    replay: ReplayBuffer,
    actor_opt: Optimizer,
    critic_opt: Optimizer,
    cfg: dict,
) -> Tuple[float, float, float]:
    """
    Proximal Policy Optimization (PPO) implementation for CVRP.

    PPO is a policy gradient method that constrains policy updates to prevent
    large changes that could destabilize training. It uses a clipped surrogate
    objective and Generalized Advantage Estimation (GAE).

    Args:
        actor: Policy network that selects actions
        critic: Value network that estimates state values
        replay: Experience replay buffer storing transitions
        actor_opt: Optimizer for the actor network
        critic_opt: Optimizer for the critic network
        cfg: Configuration dictionary with hyperparameters

    Returns:
        Tuple containing actor and critic losses
    """

    # === Hyperparameters ===
    ppo_epochs = cfg["PPO_EPOCHS"]  # Number of optimization epochs per update
    trace_decay = cfg[
        "TRACE_DECAY"
    ]  # λ (lambda) for GAE - controls bias-variance tradeoff
    eps_clip = cfg[
        "EPS_CLIP"
    ]  # Clipping range for policy ratio (prevents too large updates)
    batch_size = cfg["BATCH_SIZE"]  # Mini-batch size for SGD
    n_problems = cfg["N_PROBLEMS"]  # Number of parallel problem instances
    ent_coef = cfg["ENT_COEF"]  # Entropy coefficient for exploration
    problem_dim = pb_dim
    # problem_dim = int(
    #     cfg["PROBLEM_DIM"] + (cfg["PROBLEM_DIM"] * 0.60) + 1
    # )  # State dimension with padding
    n_step = cfg["OUTER_STEPS"]  # Number of steps for n-step returns
    gp_lam = cfg["GP_LAMBDA"]  # Gradient penalty coefficient for critic
    target_KL = cfg["TARGET_KL"]  # Target KL divergence for adaptive clipping
    device = cfg["DEVICE"]  # Computation device (CPU/GPU)

    # Set networks to training mode
    actor.train()
    critic.train()

    # === 1. Extract and process transitions from replay buffer ===
    with torch.no_grad():  # No gradients needed for preparation
        transitions = replay.memory
        nt = len(transitions)  # Total number of transitions
        batch = Transition(*zip(*transitions))  # Convert to structured batch

        # Stack and reshape tensors to proper dimensions
        state = (
            torch.stack(batch.state).view(nt * n_problems, problem_dim, -1).to(device)
        )
        action = torch.stack(batch.action).detach().view(nt * n_problems, -1).to(device)
        gamma = (
            torch.tensor(batch.gamma, device=device)
            .view(nt, 1)
            .repeat(1, n_problems)
            .to(device)
        )
        next_state = (
            torch.stack(batch.next_state)
            .detach()
            .view(nt * n_problems, problem_dim, -1)
            .to(device)
        )
        old_log_probs = (
            torch.stack(batch.old_log_probs).view(nt * n_problems, -1).to(device)
        )

        # Compute state values for current and next states
        state_values = critic(state).view(nt, n_problems, 1)
        next_state_values = critic(next_state).view(nt, n_problems, 1)

        # === 2. Compute returns and advantages using GAE ===
        # GAE: Generalized Advantage Estimation combines
        # multiple n-step advantage estimates
        # to reduce variance while maintaining acceptable bias
        advantages = torch.zeros((nt, n_problems, 1), device=device)
        rewards_to_go = torch.zeros((nt, n_problems, 1), device=device)

        # Extract rewards for all problem instances
        problem_rewards_glob = torch.stack(
            [torch.stack([t.reward[i] for t in transitions]) for i in range(n_problems)]
        ).to(device)

        # Process each problem instance separately
        for problem_idx in range(n_problems):
            # Extract the sequence for this specific problem

            problem_rewards = problem_rewards_glob[problem_idx]
            problem_values = state_values[:, problem_idx]
            problem_next_values = next_state_values[:, problem_idx]
            problem_gammas = gamma[:, problem_idx]

            # Find episode boundaries (where gamma == 0 indicates episode end)
            episode_ends = (problem_gammas == 0).nonzero().view(-1).cpu().numpy()
            episode_starts = [0] + (episode_ends[:-1] + 1).tolist()

            # Process each episode separately to avoid contamination between episodes
            for start, end in zip(episode_starts, episode_ends):
                episode_len = end - start + 1
                ep_rewards = problem_rewards[start : end + 1]
                ep_values = problem_values[start : end + 1]
                ep_next_values = problem_next_values[start : end + 1]
                ep_gammas = problem_gammas[start]  # Use individual gamma values
                last_advantage = 0

                # Compute n-step returns and advantages (working backwards)
                for t in reversed(range(episode_len)):
                    end_idx = min(t + n_step, episode_len)

                    # Calculate n-step returns (sum of discounted rewards)
                    rewards_to_go[start + t, problem_idx] = sum(
                        ep_gammas**k * ep_rewards[t + k] for k in range(end_idx - t)
                    )

                    # Calculate TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
                    delta = ep_rewards[t] + ep_gammas * ep_next_values[t] - ep_values[t]

                    # GAE formula: A_t = δ_t + γλA_{t+1}
                    # Where λ is trace_decay controlling bias-variance tradeoff
                    last_advantage = delta + ep_gammas * trace_decay * last_advantage
                    advantages[start + t, problem_idx] = last_advantage

    # Reshape advantages and rewards for batch processing
    batches = [torch.randperm(state.size(0), device=device) for _ in range(ppo_epochs)]
    advantages = advantages.view(n_problems * nt, -1)
    if torch.isnan(advantages).any():
        logger.warning("NaN detected in advantages")
    rewards_to_go = rewards_to_go.view(n_problems * nt, -1)

    # === 3. PPO Optimization Loop ===
    # Perform multiple epochs of optimization on the collected data
    for epoch in range(ppo_epochs):
        actor_opt.zero_grad()
        critic_opt.zero_grad()

        total_actor_loss = 0
        total_critic_loss = 0
        num_batches = 0

        # Process data in mini-batches
        for batch_idx in torch.split(batches[epoch], batch_size):
            if len(batch_idx) <= 1:  # Skip batches that are too small
                continue

            # Get batch data using pre-shuffled indices
            batch_state = state[batch_idx]
            batch_action = action[batch_idx]
            batch_advantages = advantages[batch_idx, 0]
            batch_rewards_to_go = rewards_to_go[batch_idx, 0]
            batch_old_log_probs = old_log_probs[batch_idx, 0]

            # Forward passes through actor and critic
            batch_state_values = critic(batch_state)
            batch_log_probs = actor.evaluate(batch_state, batch_action)

            # Safety check for numerical stability
            if (
                torch.isnan(batch_state_values).any()
                or torch.isnan(batch_log_probs).any()
            ):
                logger.warning("NaN detected in model outputs. Skipping batch.")
                continue

            # === Critic Loss Calculation ===
            # Uses clipped value loss similar to PPO's policy clipping
            old_values = critic(batch_state.detach()).detach()
            v_loss_unclipped = (batch_state_values - batch_rewards_to_go) ** 2
            v_clipped = old_values + torch.clamp(
                batch_state_values - old_values, -eps_clip, eps_clip
            )
            v_loss_clipped = (v_clipped - batch_rewards_to_go) ** 2
            critic_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

            # Add gradient penalty to enforce Lipschitz continuity
            gp_loss = gradient_penalty(critic, batch_state)
            critic_loss += gp_lam * gp_loss

            # === Actor Loss Calculation ===
            # PPO's clipped surrogate objective:
            # L^CLIP = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
            # where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
            ratios = torch.exp(batch_log_probs - batch_old_log_probs.detach())

            # Adaptive clipping based on KL divergence
            kl_div = (batch_old_log_probs - batch_log_probs).mean()
            if kl_div > 2 * target_KL:
                eps_clip = cfg["EPS_CLIP"] * 1.5  # Increase clipping if KL too high
            elif kl_div < 0.5 * target_KL:
                eps_clip = cfg["EPS_CLIP"] * 0.5  # Decrease clipping if KL too low

            # Calculate surrogate objectives
            surr1 = ratios * batch_advantages.detach()
            surr2 = (
                torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip)
                * batch_advantages.detach()
            )

            # Calculate policy entropy for exploration bonus
            entropy = -(torch.exp(batch_log_probs) * batch_log_probs).mean()

            if torch.isnan(entropy):
                logger.warning("NaN in entropy calculation - using zero entropy bonus")
                entropy = torch.zeros_like(entropy)

            # Final actor loss (negative because we're maximizing)
            actor_loss = -torch.min(surr1, surr2).mean() - ent_coef * entropy

            # Safety check for loss calculations
            if torch.isnan(actor_loss) or torch.isnan(critic_loss):
                logger.warning("NaN detected in loss calculations. Skipping batch.")
                continue

            # Backward pass to calculate gradients
            actor_loss.backward()
            critic_loss.backward()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            num_batches += 1

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)

            # Safety check for NaN gradients
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
                logger.warning("NaN detected in gradients. Skipping batch.")
                actor_opt.zero_grad()
                critic_opt.zero_grad()
                continue

            # Update network parameters
            actor_opt.step()
            critic_opt.step()

    # Return average losses
    return (
        total_actor_loss / num_batches,
        total_critic_loss / num_batches,
    )
