# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.


from typing import Dict

import torch

from model import SAModel
from problem import Problem
from replay import Replay
from utils import extend_to
from scheduler import Scheduler


def p_accept(gain: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
    """
    Compute acceptance probability, at temperature temp, of a move leading
    to a change in the energy function of 'gain'.
    """
    return torch.minimum(torch.exp(gain / temp), torch.ones_like(gain))


def sa(
    actor: SAModel,
    problem: Problem,
    init_x: torch.Tensor,
    cfg: dict,
    baseline: bool = False,
    random_std: float = 0.2,
    greedy: bool = False,
    record_state: bool = False,
    test: bool = False,
    replay: Replay = None,
) -> Dict[str, torch.Tensor]:
    """
    Runs Simulated Annealing (SA) optimisation.

    Parameters
    ----------
    actor: SAModel (nn.Module)
        The network used to compute the proposal distribution.
    problem: Problem object (see problem.py)
        The problem to be optimised.
    init_x: torch tensor
        An initial solution to the problem.
    cfg: dict
        Dictionary containing SA hyperparameters.
    baseline: boolean
        If True, actions are sampled from a uniform proposal distribution.
        If False, actions are sampled from the proposal distribution defined by 'actor'.
    random_std: float
        The standard deviation of the proposal distribution
        (only used for Rosenbrock's function).
    greedy: boolean
        If True, actor will pick the most likely action.
        If False, actor will sample from the proposal distribution.
    record_state: boolean
        If True, save partial results (states, distributions).
    replay: Replay object

    Returns
    -------
    best_x: torch tensor
        Best solution found during the optimisation.
    min_cost: torch tensor
        Minimum solution found during the optimisation.
    primal: torch tensor
        The primal loss (sum of minimum costs over the optimisation process).
    n_acc, n_rej: torch tensor
        Number of accepted (rejected) moves.
    distributions: torch tensor
        The proposal distribution at each point of the optimisation process.
    states: torch tensor
        The states observed during the optimisation.
    acceptance: torch tensor
        Whether each move is accepted or rejected.
    costs: torch tensor
        The costs observed during the optimisation.
    """
    device = init_x.device

    # Init SA cfg
    temp = torch.tensor([cfg["INIT_TEMP"]], device=device)
    next_temp = temp
    if test:
        cfg["SCHEDULER"] = "step"
    scheduler = Scheduler(
        cfg["SCHEDULER"],
        T_max=cfg["INIT_TEMP"],
        T_min=cfg["STOP_TEMP"],
        step_max=cfg["OUTER_STEPS"],
    )

    # Init archive
    best_x = x = init_x
    min_cost = problem.cost(best_x)
    init_cost = min_cost.clone()
    primal = min_cost
    first_cost = cost = min_cost
    n_acc, n_rej = 0, 0
    distributions, states, actions = [], [], []
    acceptance = []
    costs = [min_cost]
    reward = None
    last_step = None

    # Map initial solution to state
    state = problem.to_state(x, temp).to(device)
    next_state = state

    # Loops through the different temperatures in the optimization.
    for step in range(cfg["OUTER_STEPS"]):
        # Try a number of actions at each temperature.
        for j in range(cfg["INNER_STEPS"]):
            if record_state:
                states.append(state)

            # Evaluate the actor and sample an action.
            if baseline:
                action, old_log_probs = actor.baseline_sample(
                    state, random_std=random_std, problem=problem
                )
            else:
                action, old_log_probs, mask = actor.sample(
                    state, greedy=greedy, problem=problem
                )
            if record_state:
                logits = actor.get_logits(state, action, problem=problem)
                distributions.append(logits)
                actions.append(action)

            # Compute proposal
            x, *_ = problem.from_state(state)
            proposal = problem.update(x, action)

            # Compute gain
            proposal_cost = problem.cost(proposal)
            gain = cost - proposal_cost

            # Accept--reject step
            p_acceptance = p_accept(gain, temp)
            u = torch.rand(p_acceptance.shape, device=device)
            accept = 1 * (u < p_acceptance)
            realized_gain = gain * accept

            # Records
            n_acc += accept
            n_rej += 1 - accept
            if record_state:
                acceptance.append(accept)

            # Update state and cost
            cost = accept * proposal_cost + (1 - accept) * cost
            accept = extend_to(accept, x)
            next_x = (accept * proposal + (1 - accept) * x).long()

            # Update archive
            if record_state:
                costs.append(cost)
            new_best = 1 * (cost < min_cost)
            new_best = extend_to(new_best, x)
            best_x = new_best * x + (1 - new_best) * best_x
            min_cost = torch.minimum(cost, min_cost)
            primal = primal + min_cost

            # Cool down if we completed the inner steps
            if j == cfg["INNER_STEPS"] - 1:
                if last_step is not None:
                    step -= last_step

                next_temp = scheduler.step(step)
                if min(10, cfg["OUTER_STEPS"] * 0.1) == cfg["OUTER_STEPS"] - step + 1:
                    scheduler = Scheduler(
                        "lam",
                        T_max=1,
                        T_min=0.01,
                        step_max=cfg["OUTER_STEPS"] - step + 1,
                    )
                    last_step = step

            else:
                next_temp = temp
            # Compute next state
            next_state = problem.to_state(next_x, next_temp)

            # Compute reward (only relevant when training with PPO)
            if cfg["METHOD"] == "ppo":
                if cfg["REWARD"] == "immediate":
                    # Rewards are given by the immediate change in the energy function
                    reward = realized_gain.unsqueeze(1)
                elif cfg["REWARD"] == "min_cost":
                    # Rewards are given by the minimum cost found
                    reward = -min_cost.view(-1, 1)
                elif cfg["REWARD"] == "primal":
                    # Rewards are given by the sum of minimum costs over
                    # the optimisation process
                    reward = -primal.view(-1, 1)
                else:
                    raise NotImplementedError

            # if step == cfg["OUTER_STEPS"] - 1:
            #     tmp = init_cost - min_cost
            #     reward += tmp

            if replay is not None:
                replay.push(
                    state,
                    action,
                    next_state,
                    reward,
                    old_log_probs,
                    mask,
                    cfg["GAMMA"],
                )

        # Reset state and temperature
        state = next_state.clone()
        temp = next_temp

    # Negative gain
    ngain = -(first_cost - cost)

    # If a Replay object is given, we update the last transition with gamma=0
    # (final state).
    if replay is not None:
        if len(replay) > 0:
            last_transition = replay.pop()
            replay.push(*(list(last_transition[:-1]) + [0.0]))

    return {
        "best_x": best_x,
        "min_cost": min_cost,
        "primal": primal,
        "ngain": ngain,
        "n_acc": n_acc.float(),
        "n_rej": n_rej.float(),
        "distributions": distributions,
        "states": states,
        "actions": actions,
        "acceptance": acceptance,
        "costs": costs,
        "init_cost": init_cost,
        "reward": reward,
    }
