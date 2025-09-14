from typing import Tuple
import torch
from torch import nn
from torch.optim import Optimizer
from loguru import logger
from tqdm import tqdm  # Progress bar for iterations

from model import SAModel
from .replay import ReplayBuffer, Transition

# Remove default logger
logger.remove()
# Add custom logger with colored output
logger.add(
    lambda msg: print(msg, end=""),
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<blue>{file}:{line}</blue> | "
        "<red>WARNING</red> | "
        "<red>{message}</red>"
    ),
    colorize=True,
    level="WARNING",
)
logger.add(
    lambda msg: print(msg, end=""),
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "  # Timestamp in green
        "<blue>{file}:{line}</blue> | "  # File and line in blue
        "<yellow>{message}</yellow>"  # Message in yellow
    ),
    colorize=True,
    level="INFO",
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
logger.info(f"PPO epochs will use device: {DEVICE}")


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


def run_ppo_training_epochs(
    actor,
    critic,
    actor_opt,
    critic_opt,
    state,
    action,
    old_log_probs,
    advantages,
    returns,
    cfg,
    curr_epoch,
):
    """
    Dedicated function for the PPO training loop for clarity.
    """
    # === Retrieve Hyperparameters ===
    ppo_epochs = cfg["PPO_EPOCHS"]
    batch_size = cfg["BATCH_SIZE"]
    eps_clip = cfg["EPS_CLIP"]
    ent_coef = cfg["ENT_COEF"]
    gp_lam = cfg["GP_LAMBDA"]
    target_KL = cfg["TARGET_KL"]  # Use .get for flexibility

    # === Preparation for PPO Epochs ===
    total_samples = state.size(0)

    # Create a single permutation of indices for all epochs
    indices = torch.randperm(total_samples)

    # Store the "old" values that will not change during epochs
    with torch.no_grad():
        old_state_values = critic(state)

    for epoch in tqdm(
        range(ppo_epochs),
        desc=f"Epoch {curr_epoch - 1} / PPO Training",
        leave=False,
        unit="epoch",
    ):
        total_actor_loss = 0
        total_critic_loss = 0
        approx_kl_divs = []
        num_batches = 0

        # Iterate over mini-batches
        for start in range(0, total_samples, batch_size):
            end = start + batch_size
            minibatch_indices = indices[start:end]

            if len(minibatch_indices) <= 1:
                continue

            # --- Retrieve mini-batch data ---
            batch_state = state[minibatch_indices]
            batch_action = action[minibatch_indices]
            batch_advantages = advantages[minibatch_indices]
            batch_returns = returns[minibatch_indices]
            batch_old_log_probs = old_log_probs[minibatch_indices].squeeze()
            batch_old_values = old_state_values[minibatch_indices].squeeze()

            # --- Current evaluation of actor and critic ---
            batch_state_values = critic(batch_state).squeeze()
            batch_log_probs = actor.evaluate(batch_state, batch_action)

            # Gradients must be zeroed for each mini-batch
            actor_opt.zero_grad()
            critic_opt.zero_grad()

            # === Critic Loss Calculation ===
            # Standard value loss (MSE)
            v_loss_unclipped = (batch_state_values - batch_returns) ** 2

            # Clipped value loss (PPO-style)
            v_clipped = batch_old_values + torch.clamp(
                batch_state_values - batch_old_values, -eps_clip, eps_clip
            )
            v_loss_clipped = (v_clipped - batch_returns) ** 2
            critic_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

            if gp_lam > 0:
                gp_loss = gradient_penalty(critic, batch_state)
                critic_loss += gp_lam * gp_loss

            # === Actor Loss Calculation ===
            ratios = torch.exp(batch_log_probs - batch_old_log_probs.detach())
            surr1 = ratios * batch_advantages.detach()
            surr2 = (
                torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip)
                * batch_advantages.detach()
            )

            # Clipped PPO loss
            actor_loss = -torch.min(surr1, surr2).mean()

            # Entropy bonus for exploration
            entropy = -(torch.exp(batch_log_probs) * batch_log_probs).mean()
            if torch.isnan(entropy):
                entropy = 0  # Safety check

            actor_loss -= ent_coef * entropy

            # --- Backpropagation and Update ---
            # Safety checks for NaN
            if torch.isnan(actor_loss) or torch.isnan(critic_loss):
                logger.warning("NaN detected in loss. Skipping batch.")
                continue

            actor_loss.backward()
            critic_loss.backward()

            # Gradient clipping to avoid explosion
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)

            actor_opt.step()
            critic_opt.step()

            # --- Metrics tracking ---
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            num_batches += 1
            with torch.no_grad():
                # Compute KL div for monitoring and early stopping
                kl_div = (batch_old_log_probs - batch_log_probs).mean().item()
                approx_kl_divs.append(kl_div)

        # Check KL Divergence at the end of the epoch for early stopping
        avg_kl = sum(approx_kl_divs) / len(approx_kl_divs)
        if avg_kl > 1.5 * target_KL:
            logger.warning(
                f"KL divergence ({avg_kl:.3f}) exceeded target. Stopping PPO "
                "training early."
            )
            break

    # Return average losses
    return (
        total_actor_loss / num_batches if num_batches > 0 else 0,
        total_critic_loss / num_batches if num_batches > 0 else 0,
    )


def ppo(
    actor: SAModel,
    critic: nn.Module,
    pb_dim: int,
    replay: ReplayBuffer,
    actor_opt: Optimizer,
    critic_opt: Optimizer,
    curr_epoch: int,
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
    trace_decay = cfg[
        "TRACE_DECAY"
    ]  # λ (lambda) for GAE - controls bias-variance tradeoff

    n_problems = cfg["N_PROBLEMS"]  # Number of parallel problem instances
    problem_dim = pb_dim
    gamma = cfg["GAMMA"]  # Discount factor for future rewards
    end_device = cfg["DEVICE"]  # Computation device (CPU/GPU)
    device = DEVICE

    # Set networks to training mode
    actor.to(device)
    critic.to(device)
    actor.train()
    critic.train()

    # === 1. Extract and process transitions from replay buffer ===
    with torch.no_grad():  # No gradients needed for preparation
        transitions = replay.memory
        nt = len(transitions)  # nt is the sequence length

        batch = Transition(*zip(*transitions))

        rewards = (
            torch.stack(batch.reward).to(device).squeeze()
        )  # Should be of size [nt, n_problems]

        # Create the 'dones' mask from 'gamma'. This is more idiomatic.
        # gamma == 0 means the episode is done.
        gammas_tensor = torch.tensor(batch.gamma, device=device)
        dones = (
            (gammas_tensor == 0)
            .float()
            .unsqueeze(-1)
            .repeat((1, n_problems))  # .view(nt, n_problems)
        )  # Size [nt, n_problems]

        # Prepare other tensors. The .view(nt, n_problems, ...) is more direct.
        # Note: The reshape to (nt * n_problems, ...) is done AFTER GAE computation.
        state = (
            torch.stack(batch.state).view(nt, n_problems, problem_dim, -1).to(device)
        )
        action = torch.stack(batch.action).view(nt, n_problems, -1).to(device)
        next_state = (
            torch.stack(batch.next_state)
            .view(nt, n_problems, problem_dim, -1)
            .to(device)
        )
        old_log_probs = (
            torch.stack(batch.old_log_probs).view(nt, n_problems, -1).to(device)
        )

        # Get V(s) and V(s_next)
        # We need to flatten the sequence and problem dimension for network input
        flat_state = state.view(nt * n_problems, problem_dim, -1)
        flat_next_state = next_state.view(nt * n_problems, problem_dim, -1)

        state_values = critic(flat_state).view(nt, n_problems)  # Size [nt, n_problems]
        next_state_values = critic(flat_next_state).view(
            nt, n_problems
        )  # Size [nt, n_problems]

        # === 2. Vectorized Computation of Advantages (GAE) and Returns ===
        advantages = torch.zeros_like(rewards)
        last_advantage = torch.zeros(n_problems, device=device)

        # Compute deltas (TD Error) in a fully vectorized way
        # δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
        deltas = rewards + gamma * next_state_values * (1.0 - dones) - state_values

        # Compute GAE in a single reversed loop (vectorized over problems)
        # A_t = δ_t + γ * λ * A_{t+1}
        for t in reversed(range(nt)):
            last_advantage = deltas[t] + gamma * trace_decay * last_advantage * (
                1.0 - dones[t]
            )
            advantages[t] = last_advantage

        # Compute "returns" (target for value function)
        # This is the standard target for the critic, replacing your 'rewards_to_go'.
        returns = advantages + state_values

    # === 3. Preparation for PPO Epochs ===
    # Now, we flatten everything to create mini-batches.
    # The final size for all tensors will be (nt * n_problems, ...)

    state = state.view(nt * n_problems, problem_dim, -1)
    action = action.view(nt * n_problems, -1)
    old_log_probs = old_log_probs.view(nt * n_problems, -1)
    advantages = advantages.view(nt * n_problems)
    returns = returns.view(nt * n_problems)

    # Advantage normalization (common and recommended practice)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    if torch.isnan(advantages).any():
        logger.warning("NaN detected in advantages")

    # === 4. PPO Optimization Loop ===
    # Perform multiple epochs of optimization on the collected data
    actor_loss, critic_loss = run_ppo_training_epochs(
        actor,
        critic,
        actor_opt,
        critic_opt,
        state,
        action,
        old_log_probs,
        advantages,
        returns,
        cfg,
        curr_epoch,
    )
    actor.to(end_device)
    critic.to(end_device)
    return (actor_loss, critic_loss)
