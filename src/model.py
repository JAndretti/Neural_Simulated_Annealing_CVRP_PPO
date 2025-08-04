from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import repeat_to


def create_network(input_dim, embed_dim, num_hidden_layers, device):
    layers = []
    # Entry layer
    layers.append(nn.Linear(input_dim, embed_dim, bias=True, device=device))
    layers.append(nn.LeakyReLU())

    # Hidden layers
    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(embed_dim, embed_dim, bias=True, device=device))
        layers.append(nn.LeakyReLU())

    # Output layer
    layers.append(nn.Linear(embed_dim, 1, bias=False, device=device))

    return nn.Sequential(*layers).to(device)


class SAModel(nn.Module):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self.device = device
        self.generator = torch.Generator(device=device)

    def manual_seed(self, seed: int) -> None:
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

    def get_logits(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sample(
        self, state: torch.Tensor, greedy: bool = False, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError

    def baseline_sample(self, state: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        if type(m) is nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)


class CVRPActorPairs(SAModel):
    """Actor network for CVRP that selects pairs of nodes to swap."""

    def __init__(
        self,
        embed_dim: int = 32,
        c: int = 22,
        num_hidden_layers: int = 2,
        device: str = "cpu",
        mixed_heuristic: bool = False,
        method: str = "free",
    ) -> None:
        super().__init__(device)
        self.mixed_heuristic = mixed_heuristic
        self.method = method

        self.input_dim = c * 2 if mixed_heuristic else c * 2 - 2
        self.net = create_network(
            self.input_dim,
            embed_dim,
            num_hidden_layers=num_hidden_layers,
            device=device,
        )

        self.net.apply(self.init_weights)

    def forward(self, state):
        """Forward pass computing logits for node pairs."""
        pair_logits = self.net(state)
        return pair_logits

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        """Initialize weights using Kaiming uniform initialization."""
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def sample_from_logits(
        self, logits: torch.Tensor, greedy: bool = False, one_hot: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample actions from logits using either greedy or multinomial sampling."""
        probs = torch.softmax(logits, dim=-1)

        if greedy:
            smpl = torch.argmax(probs, -1, keepdim=False)
        else:
            smpl = torch.multinomial(probs, 1, generator=self.generator)[..., 0]

        taken_probs = probs.gather(1, smpl.view(-1, 1))

        if one_hot:
            smpl = F.one_hot(smpl, num_classes=logits.shape[-1])[..., None]

        return smpl, torch.log(taken_probs)

    def get_logits(
        self, state: torch.Tensor, action: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute logits and log probabilities for given state and action."""
        pair_features, idx1, idx2, heuristic_indices = self._prepare_features_and_pairs(
            state
        )

        # Forward pass
        outputs = self.forward(pair_features)
        logits = outputs[..., 0]
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs)
        all_action = (
            (
                torch.cat(
                    [
                        idx1.unsqueeze(-1).repeat_interleave(2, dim=0),
                        idx2.unsqueeze(-1).repeat_interleave(2, dim=0),
                        heuristic_indices.unsqueeze(-1),
                    ],
                    dim=-1,
                )
                if heuristic_indices is not None
                else torch.cat([idx1.unsqueeze(-1), idx2.unsqueeze(-1)], dim=-1)
            ),
        )

        return log_probs, all_action

    def baseline_sample(
        self, state: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, None]:
        """Generate baseline sample using uniform probabilities."""
        n_problems, problem_dim, dim = state.shape
        state = state[:, :, :-2]  # Remove temperature and time

        x, _, *_ = torch.split(state, [1, 2] + [1] * (dim - 5), dim=-1)
        mask = x.squeeze(-1) != 0
        x = torch.stack([c[m] for c, m in zip(x, mask)], dim=0)

        idx1, idx2 = torch.triu_indices(x.shape[1], x.shape[1], offset=1)
        logits = torch.ones(n_problems, idx1.shape[0]).to(self.device)
        c, _ = self.sample_from_logits(logits, one_hot=False)

        if self.mixed_heuristic:
            pair_idx = c // 2
            heuristic_idx = c % 2
            action = torch.stack(
                [idx1[pair_idx], idx2[pair_idx], heuristic_idx], dim=-1
            )
        else:
            action = torch.stack([idx1[c], idx2[c]], dim=-1)

        return action, None

    def sample(
        self, state: torch.Tensor, greedy: bool = False, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action pair from the current state."""

        pair_features, idx1, idx2 = self._prepare_features_and_pairs(state)
        logits = self.forward(pair_features)[..., 0]  # Forward pass

        c, log_probs = self.sample_from_logits(logits, greedy=greedy, one_hot=False)

        if self.mixed_heuristic:
            pair_idx = c // 2
            heuristic_idx = c % 2
            action = torch.stack(
                [idx1[pair_idx], idx2[pair_idx], heuristic_idx], dim=-1
            )
        else:
            action = torch.stack([idx1[c], idx2[c]], dim=-1)

        return action, log_probs[..., 0]

    def evaluate(
        self, state: torch.Tensor, action: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Evaluate log probabilities of given actions."""
        pair_features, idx1, idx2 = self._prepare_features_and_pairs(state)
        if self.mixed_heuristic:
            # Find the pair index
            pair_mask = (idx1 == action[:, 0].unsqueeze(1)) & (
                idx2 == action[:, 1].unsqueeze(1)
            )
            pair_idx = pair_mask.nonzero(as_tuple=True)[1]
            # Calculate the full action index (pair_idx * 2 + heuristic_idx)
            action_idx = pair_idx * 2 + action[:, 2]
        else:
            action_idx = (idx1 == action[:, 0].unsqueeze(1)) & (
                idx2 == action[:, 1].unsqueeze(1)
            )
            action_idx = action_idx.nonzero(as_tuple=True)[1]

        # Forward pass
        logits = self.forward(pair_features)[..., 0]

        # Compute action probabilities
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs.gather(1, action_idx.view(-1, 1)))
        return log_probs[..., 0]

    def _prepare_features_and_pairs(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Helper method to prepare features and pairs from state."""
        n_problems, problem_dim, dim = state.shape
        temp, time = state[:, :, -2], state[:, :, -1]
        state = state[:, :, :-2]  # Remove temperature and time

        x, coords, *extra_features = torch.split(
            state, [1, 2] + [1] * (dim - 5), dim=-1
        )

        # Gather coordinate information
        coords = coords.gather(1, x.long().expand_as(coords))
        coords_prev = torch.cat([coords[:, -1:, :], coords[:, :-1, :]], dim=1)
        coords_next = torch.cat([coords[:, 1:, :], coords[:, :1, :]], dim=1)

        c_state = torch.cat([coords, coords_prev, coords_next] + extra_features, -1)

        if self.method == "rm_depot":
            mask = x.squeeze(-1) != 0
            c_state = c_state[mask].view(n_problems, -1, c_state.size(-1))

        # Get all possible pairs
        idx1, idx2 = torch.triu_indices(
            c_state.shape[1], c_state.shape[1], offset=1, device=c_state.device
        )
        x_pairs_1 = c_state[:, idx1, :]
        x_pairs_2 = c_state[:, idx2, :]

        # Combine pair features with temperature and time
        pair_features = torch.cat(
            [
                x_pairs_1,
                x_pairs_2,
                temp[:, idx1].unsqueeze(-1),
                time[:, idx1].unsqueeze(-1),
            ],
            dim=-1,
        )
        if self.mixed_heuristic:
            # Duplicate each pair for both heuristic options
            n_pairs = pair_features.shape[1]
            pair_features = pair_features.repeat_interleave(2, dim=1)

            # Add one-hot encoding for heuristic selection
            heuristic_indices = torch.arange(2, device=c_state.device).repeat(n_pairs)
            heuristic_one_hot = F.one_hot(heuristic_indices, num_classes=2).repeat(
                n_problems, 1, 1
            )

            # Concatenate the one-hot encoding to the features
            pair_features = torch.cat([pair_features, heuristic_one_hot], dim=-1)

        return pair_features, idx1, idx2


class CVRPActor(SAModel):

    def __init__(
        self,
        embed_dim: int = 32,
        c: int = 13,
        num_hidden_layers: int = 2,
        device: str = "cpu",
        mixed_heuristic: bool = False,
        method: str = "free",
    ) -> None:
        super().__init__(device)
        self.mixed_heuristic = mixed_heuristic
        self.c1_state_dim = c
        self.method = method

        # Mean and std computation
        self.city1_net = create_network(
            self.c1_state_dim,
            embed_dim,
            num_hidden_layers=num_hidden_layers,
            device=device,
        )
        self.c2_state_dim = c * 2 if mixed_heuristic else c * 2 - 2

        # Mean and std computation
        self.city2_net = create_network(
            self.c2_state_dim,
            embed_dim,
            num_hidden_layers=num_hidden_layers,
            device=device,
        )

        self.city1_net.apply(self.init_weights)
        self.city2_net.apply(self.init_weights)

    def sample_from_logits(
        self, logits: torch.Tensor, greedy: bool = False, one_hot: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample actions from logits using either greedy or multinomial sampling."""
        n_problems, problem_dim = logits.shape
        probs = torch.softmax(logits, dim=-1)

        if greedy:
            smpl = torch.argmax(probs, -1, keepdim=False)
        else:
            smpl = torch.multinomial(probs, 1, generator=self.generator)[..., 0]

        taken_probs = probs.gather(1, smpl.view(-1, 1))

        if one_hot:
            smpl = F.one_hot(smpl, num_classes=problem_dim)[..., None]

        return smpl, torch.log(taken_probs)

    def get_logits(
        self, state: torch.Tensor, action: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute logits and log probabilities for given state and action."""
        c1_state, n_problems = self._prepare_features_city1(state)

        c1 = action[:, 0]

        # City 1 net
        logits = self.city1_net(c1_state)[..., 0]
        probs = torch.softmax(logits, dim=-1)
        log_probs_c1 = torch.log(probs)

        c2_state = self._prepare_features_city2(
            c1_state, c1, n_problems
        )  # Second city encoding
        logits = self.city2_net(c2_state)[..., 0]

        probs = torch.softmax(logits, dim=-1)
        log_probs_c2 = torch.log(probs)

        return log_probs_c1, log_probs_c2

    def baseline_sample(
        self, state: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, None]:
        """Generate baseline sample using uniform probabilities."""
        n_problems, problem_dim, _ = state.shape
        x = state[:, :, 0]

        # Sample c1 at random
        if self.method == "rm_depot":
            mask = x.squeeze(-1) != 0
            x = x[mask].view(n_problems, -1)
        logits = torch.ones(n_problems, x.shape[1]).to(self.generator.device)
        c1, _ = self.sample_from_logits(logits, one_hot=False)

        # sample c2
        if self.mixed_heuristic:
            logits = torch.ones(n_problems, x.shape[1] * 2).to(self.generator.device)
            c2, _ = self.sample_from_logits(logits, one_hot=False)
            heuristic_idx = c2 % 2
            action = torch.cat(
                [
                    c1.view(-1, 1).long(),
                    c2.view(-1, 1).long(),
                    heuristic_idx.view(-1, 1).long(),
                ],
                dim=-1,
            )
        else:
            c2, _ = self.sample_from_logits(logits, one_hot=False)
            action = torch.cat([c1.view(-1, 1).long(), c2.view(-1, 1).long()], dim=-1)
        return action, None

    def sample(
        self, state: torch.Tensor, greedy: bool = False, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample an action pair from the current state."""
        c1_state, n_problems = self._prepare_features_city1(state)
        logits = self.city1_net(c1_state)[..., 0]
        # Mask logits
        c1, log_probs_c1 = self.sample_from_logits(logits, greedy=greedy, one_hot=False)

        c2_state = self._prepare_features_city2(
            c1_state, c1, n_problems
        )  # Second city encoding

        # City 2 net
        logits = self.city2_net(c2_state)[..., 0]
        c2, log_probs_c2 = self.sample_from_logits(logits, greedy=greedy, one_hot=False)

        # Construct action and log-probabilities
        if self.mixed_heuristic:
            c2_idx = c2 // 2  # Index of the pair
            heuristic_idx = c2 % 2  # Index of the heuristic
            action = torch.cat(
                [
                    c1.view(-1, 1).long(),
                    c2_idx.view(-1, 1).long(),
                    heuristic_idx.view(-1, 1).long(),
                ],
                dim=-1,
            )
        else:
            # Concatenate c1 and c2
            action = torch.cat([c1.view(-1, 1).long(), c2.view(-1, 1).long()], dim=-1)

        log_probs = log_probs_c1 + log_probs_c2
        return action, log_probs[..., 0]

    def evaluate(
        self, state: torch.Tensor, action: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Evaluate log probabilities of given actions."""
        c1_state, n_problems = self._prepare_features_city1(state)

        c1 = action[:, 0]
        if self.mixed_heuristic:
            # When mixed_heuristic is True, action contains [c1, c2_idx, heuristic_idx]
            c2_idx = action[:, 1]
            heuristic_idx = action[:, 2]
            # Reconstruct the composite c2 action
            c2 = c2_idx * 2 + heuristic_idx
        else:
            # Standard case: action contains [c1, c2]
            c2 = action[:, 1]

        # City 1 net
        logits = self.city1_net(c1_state)[..., 0]
        probs = torch.softmax(logits, dim=-1)
        log_probs_c1 = torch.log(probs.gather(1, c1.view(-1, 1)))

        c2_state = self._prepare_features_city2(
            c1_state, c1, n_problems
        )  # Second city encoding

        # City 2 net
        logits = self.city2_net(c2_state)[..., 0]
        probs = torch.softmax(logits, dim=-1)
        log_probs_c2 = torch.log(probs.gather(1, c2.view(-1, 1)))

        # Construct log-probabilities and return
        log_probs = log_probs_c1 + log_probs_c2
        return log_probs[..., 0]

    def _prepare_features_city1(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """Helper method to prepare features from state."""
        n_problems, problem_dim, dim = state.shape

        x, coords, *extra_features = torch.split(
            state, [1, 2] + [1] * (dim - 3), dim=-1
        )
        # Gather coordinate information
        coords = coords.gather(1, x.long().expand_as(coords))
        coords_prev = torch.cat([coords[:, -1:, :], coords[:, :-1, :]], dim=1)
        coords_next = torch.cat([coords[:, 1:, :], coords[:, :1, :]], dim=1)
        # Add temp and time to the concatenated state
        c_state = torch.cat(
            [
                coords,
                coords_prev,
                coords_next,
            ]
            + extra_features,
            -1,
        )
        if self.method == "rm_depot":
            mask = x.squeeze(-1) != 0
            c_state = c_state[mask].view(n_problems, -1, c_state.size(-1))
        return c_state, n_problems

    def _prepare_features_city2(
        self, c1_state: torch.Tensor, c1: torch.Tensor, n_problems: int
    ) -> Tuple[torch.Tensor]:
        """Helper method to prepare features from state."""
        # Second city encoding
        arange = torch.arange(n_problems)
        c1_val = c1_state[arange, c1]
        base = torch.cat([c1_val], -1)[:, None, :]

        base = repeat_to(base, c1_state)
        c1_state = c1_state[:, :, :-2]
        c2_state = torch.cat([base, c1_state], -1)
        if self.mixed_heuristic:
            # Duplicate each pair for both heuristic options
            n_row = c2_state.shape[1]
            c2_state = c2_state.repeat_interleave(2, dim=1)

            # Add one-hot encoding for heuristic selection
            heuristic_indices = torch.arange(2, device=c1_state.device).repeat(n_row)
            heuristic_one_hot = F.one_hot(heuristic_indices, num_classes=2).repeat(
                n_problems, 1, 1
            )
            # Concatenate the one-hot encoding to the features
            c2_state = torch.cat([c2_state, heuristic_one_hot], dim=-1)
        return c2_state


class CVRPCritic(nn.Module):
    """Critic network for CVRP that estimates state values."""

    def __init__(
        self,
        embed_dim: int,
        c: int = 13,
        num_hidden_layers: int = 2,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.q_func = create_network(
            c,
            embed_dim,
            num_hidden_layers=num_hidden_layers,
            device=device,
        )
        self.q_func.apply(self.init_weights)

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        """Initialize weights using Kaiming uniform initialization."""
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass computing state values."""
        n_problems, problem_dim, dim = state.shape
        num_extra_features = dim - 3
        x, coords, *extra_features = torch.split(
            state, [1, 2] + [1] * num_extra_features, dim=-1
        )

        # Gather current, previous and next coordinates
        coords = coords.gather(1, x.long().expand_as(coords))
        coords_prev = torch.cat([coords[:, -1:, :], coords[:, :-1, :]], dim=1)
        coords_next = torch.cat([coords[:, 1:, :], coords[:, :1, :]], dim=1)

        state = torch.cat([coords, coords_prev, coords_next] + extra_features, -1)
        q_values = self.q_func(state).view(n_problems, problem_dim)
        return q_values.mean(dim=-1)
