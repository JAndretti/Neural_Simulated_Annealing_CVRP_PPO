from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import repeat_to


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


class CVRPActor(SAModel):
    def __init__(
        self, embed_dim: int, c1: int = 7, c2: int = 13, device: str = "cpu"
    ) -> None:
        super().__init__(device)
        self.c1_state_dim = c1
        self.c2_state_dim = c2

        # Mean and std computation
        self.city1_net = nn.Sequential(
            nn.Linear(self.c1_state_dim, embed_dim, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, 1, bias=False, device=device),
        )
        # Mean and std computation
        self.city2_net = nn.Sequential(
            nn.Linear(self.c2_state_dim, embed_dim, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, 1, bias=False, device=device),
        )

        self.city1_net.apply(self.init_weights)
        self.city2_net.apply(self.init_weights)

    def sample_from_logits(
        self, logits: torch.Tensor, greedy: bool = False, one_hot: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        pb = kwargs["problem"]
        n_problems, problem_dim, dim = state.shape
        num_extra_features = dim - 3
        x, coords, *extra_features = torch.split(
            state, [1, 2] + [1] * num_extra_features, dim=-1
        )

        coords = coords.gather(1, x.long().expand_as(coords))
        coords_prev = torch.roll(coords, 1, 1)
        coords_next = torch.roll(coords, -1, 1)

        c1_state = torch.cat([coords, coords_prev, coords_next] + extra_features, -1)

        c1 = action[:, 0]
        # c2 = action[:, 1]

        # City 1 net
        mask = ~(state[:, :, 0] != 0)
        logits = self.city1_net(c1_state)[..., 0]
        logits[mask] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        log_probs_c1 = torch.log(probs)

        c1_prev = (c1 - 1) % problem_dim
        c1_next = (c1 + 1) % problem_dim

        # Second city encoding: [base.prev, base, base.next,
        #                        target.prev, target, target.next]
        arange = torch.arange(n_problems)
        c1_coords = coords[arange, c1]
        c1_prev_coords = coords[arange, c1_prev]
        c1_next_coords = coords[arange, c1_next]
        base = torch.cat([c1_coords, c1_prev_coords, c1_next_coords], -1)[:, None, :]
        base = repeat_to(base, c1_state)
        c2_state = torch.cat([base, c1_state], -1)

        # City 2 net
        mask = ~(
            pb.allowed_permutations(
                state[:, :, 0].unsqueeze(-1).long(), c1.unsqueeze(-1).long()
            )
        )
        logits = self.city2_net(c2_state)[..., 0]
        logits[mask] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        log_probs_c2 = torch.log(probs)

        return log_probs_c1, log_probs_c2

    def baseline_sample(
        self, state: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, None]:
        n_problems, problem_dim, _ = state.shape
        pb = kwargs["problem"]

        # Sample c1 at random
        mask = ~(state[:, :, 0] != 0)
        logits = torch.ones(mask.shape[0], mask.shape[1]).to(self.generator.device)
        logits[mask] = -float("inf")
        c1, _ = self.sample_from_logits(logits, one_hot=False)

        # Compute mask and sample c2
        mask = ~(
            pb.allowed_permutations(
                state[:, :, 0].unsqueeze(-1).long(), c1.unsqueeze(-1).long()
            )
        )
        logits = torch.ones(mask.shape[0], mask.shape[1]).to(self.generator.device)
        logits[mask] = -float("inf")
        c2, _ = self.sample_from_logits(logits, one_hot=False)

        # Construct action tensor and return
        action = torch.cat([c1.view(-1, 1).long(), c2.view(-1, 1).long()], dim=-1)
        return action, None

    def sample(
        self, state: torch.Tensor, greedy: bool = False, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pb = kwargs["problem"]
        n_problems, problem_dim, dim = state.shape

        num_extra_features = dim - 3
        x, coords, *extra_features = torch.split(
            state, [1, 2] + [1] * num_extra_features, dim=-1
        )

        coords = coords.gather(1, x.long().expand_as(coords))
        coords_prev = torch.roll(coords, 1, 1)
        coords_next = torch.roll(coords, -1, 1)

        c1_state = torch.cat([coords, coords_prev, coords_next] + extra_features, -1)

        # City 1 net
        mask = ~(state[:, :, 0] != 0)
        logits = self.city1_net(c1_state)[..., 0]
        logits[mask] = -float("inf")
        c1, log_probs_c1 = self.sample_from_logits(logits, greedy=greedy, one_hot=False)

        c1_prev = (c1 - 1) % problem_dim
        c1_next = (c1 + 1) % problem_dim

        # Second city encoding: [base.prev, base, base.next,
        #                        target.prev, target, target.next]
        arange = torch.arange(n_problems)
        c1_coords = coords[arange, c1]
        c1_prev_coords = coords[arange, c1_prev]
        c1_next_coords = coords[arange, c1_next]
        base = torch.cat([c1_coords, c1_prev_coords, c1_next_coords], -1)[:, None, :]

        base = repeat_to(base, c1_state)
        c2_state = torch.cat([base, c1_state], -1)

        # City 2 net
        mask = ~(
            pb.allowed_permutations(
                state[:, :, 0].unsqueeze(-1).long(), c1.unsqueeze(-1).long()
            )
        )
        logits = self.city2_net(c2_state)[..., 0]
        logits[mask] = -float("inf")
        c2, log_probs_c2 = self.sample_from_logits(logits, greedy=greedy, one_hot=False)

        # Construct action and log-probabilities
        action = torch.cat([c1.view(-1, 1).long(), c2.view(-1, 1).long()], dim=-1)
        log_probs = log_probs_c1 + log_probs_c2
        return action, log_probs[..., 0], mask

    def evaluate(
        self, state: torch.Tensor, action: torch.Tensor, mask2: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        n_problems, problem_dim, dim = state.shape

        num_extra_features = dim - 3  # Nombre de caractéristiques supplémentaires
        x, coords, *extra_features = torch.split(
            state, [1, 2] + [1] * num_extra_features, dim=-1
        )

        coords = coords.gather(1, x.long().expand_as(coords))
        coords_prev = torch.roll(coords, 1, 1)
        coords_next = torch.roll(coords, -1, 1)

        c1_state = torch.cat([coords, coords_prev, coords_next] + extra_features, -1)

        c1 = action[:, 0]
        c2 = action[:, 1]

        # City 1 net
        mask = ~(state[:, :, 0] != 0)
        logits = self.city1_net(c1_state)[..., 0]
        logits[mask] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        log_probs_c1 = torch.log(probs.gather(1, c1.view(-1, 1)))

        c1_prev = (c1 - 1) % problem_dim
        c1_next = (c1 + 1) % problem_dim

        # Second city encoding: [base.prev, base, base.next,
        #                        target.prev, target, target.next]
        arange = torch.arange(n_problems)
        c1_coords = coords[arange, c1]
        c1_prev_coords = coords[arange, c1_prev]
        c1_next_coords = coords[arange, c1_next]
        base = torch.cat([c1_coords, c1_prev_coords, c1_next_coords], -1)[:, None, :]
        base = repeat_to(base, c1_state)
        c2_state = torch.cat([base, c1_state], -1)

        # City 2 net
        logits = self.city2_net(c2_state)[..., 0]
        logits[mask2] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        log_probs_c2 = torch.log(probs.gather(1, c2.view(-1, 1)))

        # Construct log-probabilities and return
        log_probs = log_probs_c1 + log_probs_c2
        return log_probs[..., 0]


class CVRPCritic(nn.Module):
    def __init__(self, embed_dim: int, c: int = 7, device: str = "cpu") -> None:
        super().__init__()
        self.q_func = nn.Sequential(
            nn.Linear(c, embed_dim, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, 1, device=device),
        )

        self.q_func.apply(self.init_weights)

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        if type(m) is nn.Linear:
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        n_problems, problem_dim, dim = state.shape

        num_extra_features = dim - 3  # Nombre de caractéristiques supplémentaires
        x, coords, *extra_features = torch.split(
            state, [1, 2] + [1] * num_extra_features, dim=-1
        )

        coords = coords.gather(1, x.long().expand_as(coords))
        coords_prev = torch.roll(coords, 1, 1)
        coords_next = torch.roll(coords, -1, 1)

        state = torch.cat([coords, coords_prev, coords_next] + extra_features, -1)
        q_values = self.q_func(state).view(n_problems, problem_dim)
        return q_values.mean(dim=-1)


class CVRPActorPairs(SAModel):
    def __init__(self, embed_dim: int = 32, c: int = 20, device: str = "cpu") -> None:
        super().__init__(device)
        self.net = nn.Sequential(
            nn.Linear(c, embed_dim, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, 1, bias=False, device=device),
        )

        self.net.apply(self.init_weights)

    def sample_from_logits(
        self, logits: torch.Tensor, greedy: bool = False, one_hot: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        """
        Computes the logits and log probabilities for the given state and action.
        Args:
            state (torch.Tensor): The current state tensor with shape
            (n_problems, problem_dim, dim).
            action (torch.Tensor): The action tensor.
            **kwargs: Additional keyword arguments, including:
                - problem: An instance of the problem class providing the
                get_permutable_pairs method.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - logits (torch.Tensor): The computed logits tensor.
                - log_probs (torch.Tensor): The log probabilities tensor.
        """

        pb = kwargs["problem"]
        n_problems, problem_dim, dim = state.shape
        num_extra_features = dim - 3 - 2  # Number of extra features excluding
        # temperature and time
        temp, time = state[:, :, -2], state[:, :, -1]
        # Extract temperature and time features
        state = state[:, :, :-2]  # Remove temperature and time from state

        # Split state into different components
        x, coords, *extra_features = torch.split(
            state, [1, 2] + [1] * num_extra_features, dim=-1
        )

        # Gather coordinates based on the current position
        coords = coords.gather(1, x.long().expand_as(coords))
        coords_prev = torch.roll(coords, 1, 1)  # Previous coordinates
        coords_next = torch.roll(coords, -1, 1)  # Next coordinates

        # Concatenate coordinates and extra features to form the state
        c_state = torch.cat([coords, coords_prev, coords_next] + extra_features, -1)

        # Get upper triangular indices for pair combinations
        idx1, idx2 = torch.triu_indices(problem_dim, problem_dim, offset=1)

        # Extract pairs of features
        x_pairs_1 = c_state[:, idx1, :]
        x_pairs_2 = c_state[:, idx2, :]
        pair_features = torch.cat([x_pairs_1, x_pairs_2], dim=-1)

        # Add temperature and time features to the pairs
        temp = temp[:, idx1].unsqueeze(-1)
        time = time[:, idx1].unsqueeze(-1)
        pair_features = torch.cat(
            [pair_features, temp, time],
            dim=-1,
        )

        # Get mask for valid permutations
        mask = ~(pb.get_permutable_pairs(x.long(), (idx1, idx2)))

        # Compute logits and apply mask
        logits = self.net(pair_features)[..., 0]
        logits[mask] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        return torch.log(probs)

    def baseline_sample(
        self, state: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, None]:
        """
        Generates a baseline sample action based on the given state and problem.

        Args:
            state (torch.Tensor): The input state tensor with shape
            (n_problems, problem_dim, dim).
            **kwargs: Additional keyword arguments, including:
                - problem: The problem instance providing the method
                `get_permutable_pairs`.

        Returns:
            Tuple[torch.Tensor, None]: A tuple containing the action tensor and None.
                The action tensor has shape (n_problems, 2) representing
                the selected pair of indices.
        """

        pb = kwargs["problem"]
        n_problems, problem_dim, dim = state.shape
        num_extra_features = dim - 3 - 2  # Number of extra features excluding

        # Extract temperature and time features
        state = state[:, :, :-2]  # Remove temperature and time from state

        # Split state into different components
        x, coords, *extra_features = torch.split(
            state, [1, 2] + [1] * num_extra_features, dim=-1
        )
        # Get upper triangular indices for pair combinations
        idx1, idx2 = torch.triu_indices(problem_dim, problem_dim, offset=1)
        # Get mask for valid permutations
        mask = pb.get_permutable_pairs(x.long(), (idx1, idx2))
        logits = torch.ones(mask.shape[0], mask.shape[1]).to(self.generator.device)
        logits[~mask] = -float("inf")
        c, _ = self.sample_from_logits(logits, one_hot=False)
        action = torch.stack([idx1[c], idx2[c]], dim=-1)
        return action, None

    def sample(
        self, state: torch.Tensor, greedy: bool = False, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a sample action based on the given state and problem.

        Args:
            state (torch.Tensor): The input state tensor with shape
            (n_problems, problem_dim, dim).
            greedy (bool): A boolean flag indicating whether to use greedy sampling.
            **kwargs: Additional keyword arguments, including:
                - problem: The problem instance providing the method
                `get_permutable_pairs`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the action tensor
            and the log probabilities tensor. The action tensor has shape
            (n_problems, 2) representing the selected pair of indices.
        """
        pb = kwargs["problem"]
        n_problems, problem_dim, dim = state.shape
        num_extra_features = dim - 3 - 2
        # temperature and time
        temp, time = state[:, :, -2], state[:, :, -1]
        # Extract temperature and time features
        state = state[:, :, :-2]  # Remove temperature and time from state

        # Split state into different components
        x, coords, *extra_features = torch.split(
            state, [1, 2] + [1] * num_extra_features, dim=-1
        )

        # Gather coordinates based on the current position
        coords = coords.gather(1, x.long().expand_as(coords))
        coords_prev = torch.roll(coords, 1, 1)  # Previous coordinates
        coords_next = torch.roll(coords, -1, 1)  # Next coordinates

        # Concatenate coordinates and extra features to form the state
        c_state = torch.cat([coords, coords_prev, coords_next] + extra_features, -1)

        # Get upper triangular indices for pair combinations
        idx1, idx2 = torch.triu_indices(problem_dim, problem_dim, offset=1).to(
            self.device
        )

        # Extract pairs of features
        x_pairs_1 = c_state[:, idx1, :]
        x_pairs_2 = c_state[:, idx2, :]
        pair_features = torch.cat([x_pairs_1, x_pairs_2], dim=-1)

        # Add temperature and time features to the pairs
        temp = temp[:, idx1].unsqueeze(-1)
        time = time[:, idx1].unsqueeze(-1)
        pair_features = torch.cat(
            [pair_features, temp, time],
            dim=-1,
        )

        # Get mask for valid permutations
        mask = ~(pb.get_permutable_pairs(x.long(), (idx1, idx2)))

        # Compute logits and apply mask
        logits = self.net(pair_features)[..., 0]
        logits[mask] = -float("inf")
        c, log_probs = self.sample_from_logits(logits, greedy=greedy, one_hot=False)
        action = torch.stack([idx1[c], idx2[c]], dim=-1)
        return action, log_probs[..., 0], mask

    def evaluate(
        self, state: torch.Tensor, action: torch.Tensor, mask: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Evaluates the log probabilities of the given action based on the input state.

        Args:
            state (torch.Tensor): The input state tensor with shape
            (n_problems, problem_dim, dim).
            action (torch.Tensor): The action tensor with shape (n_problems, 2).
            mask2 (torch.Tensor): The mask tensor for the second city.

        Returns:
            torch.Tensor: The log probabilities tensor.
        """
        n_problems, problem_dim, dim = state.shape
        num_extra_features = dim - 3 - 2  # Number of extra features excluding
        # temperature and time
        temp, time = state[:, :, -2], state[:, :, -1]
        # Extract temperature and time features
        state = state[:, :, :-2]

        # Split state into different components
        x, coords, *extra_features = torch.split(
            state, [1, 2] + [1] * num_extra_features, dim=-1
        )

        # Gather coordinates based on the current position
        coords = coords.gather(1, x.long().expand_as(coords))
        coords_prev = torch.roll(coords, 1, 1)  # Previous coordinates
        coords_next = torch.roll(coords, -1, 1)  # Next coordinates

        # Concatenate coordinates and extra features to form the state
        c_state = torch.cat([coords, coords_prev, coords_next] + extra_features, -1)

        # Get upper triangular indices for pair combinations
        idx1, idx2 = torch.triu_indices(problem_dim, problem_dim, offset=1)
        # Find the indices where idx1 equals action[:, 0] and idx2 equals action[:, 1]
        action_idx = (idx1 == action[:, 0].unsqueeze(1)) & (
            idx2 == action[:, 1].unsqueeze(1)
        )
        action_idx = action_idx.nonzero(as_tuple=True)[1]
        # Extract pairs of features
        x_pairs_1 = c_state[:, idx1, :]
        x_pairs_2 = c_state[:, idx2, :]
        pair_features = torch.cat([x_pairs_1, x_pairs_2], dim=-1)

        # Add temperature and time features to the pairs
        temp = temp[:, idx1].unsqueeze(-1)
        time = time[:, idx1].unsqueeze(-1)
        pair_features = torch.cat(
            [pair_features, temp, time],
            dim=-1,
        )

        # Compute logits and apply mask
        logits = self.net(pair_features)[..., 0]
        logits[mask] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs[torch.arange(n_problems), action_idx])
        return log_probs[..., 0]
