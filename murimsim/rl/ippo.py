"""IPPO (Independent PPO) with shared parameters — Phase 3 of IPPO migration.

Single-file CleanRL-style implementation. Provides:
    * SharedActorCritic — MLP trunk → masked categorical actor + scalar critic.
    * RolloutBuffer     — (T, n_envs, n_agents, *) tensors with active-mask aware
                          GAE that respects per-slot life boundaries.
    * ppo_update        — standard PPO clip update with active-mask weighting.

Design constraints (from IPPO migration plan and rubber-duck review):
    * Parameter sharing: one network instance shared across all agents/slots.
    * Decentralized critic: V depends only on local obs (CTDE deferred).
    * Per-slot active mask: dead/inactive slots contribute zero to every loss
      term and zero gradient. Skipped entirely if a minibatch is all-inactive.
    * Pre-action masks are stored in the buffer and reused at update time —
      action validity is state-dependent (qi pool, position, curriculum).
    * Safe masked entropy via ``Categorical(logits=masked_fill(min))``.
    * Action-mask repair: if all actions are masked (impossible per game design
      but defensive), force REST as the only legal action.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from murimsim.actions import Action

REST_ACTION: int = Action.REST.value


# ---------------------------------------------------------------------------
# Action mask helpers
# ---------------------------------------------------------------------------

def repair_action_mask(mask: torch.Tensor) -> torch.Tensor:
    """If a row has no legal actions, allow REST. Defensive — should be rare.

    Args:
        mask: bool tensor of shape (..., n_actions). Modified in place AND
            returned for convenience.

    Returns:
        Same tensor with at least one True per row. If the mask is too narrow
        to contain REST (e.g. n_actions < REST_ACTION+1), action 0 is used
        as a last-resort fallback so the distribution remains valid.
    """
    no_legal = ~mask.any(dim=-1)
    if no_legal.any():
        n_actions = mask.shape[-1]
        fallback = REST_ACTION if REST_ACTION < n_actions else 0
        idx = no_legal.nonzero(as_tuple=False)
        for row in idx:
            mask[tuple(row.tolist()) + (fallback,)] = True
    return mask


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    """Build a Categorical with forbidden actions zeroed out.

    Uses ``masked_fill`` with the dtype's min so probability of forbidden
    actions is exactly zero after softmax and ``Categorical.entropy()`` stays
    finite (no 0 * -inf hazards).
    """
    repaired = repair_action_mask(mask.clone())
    masked_logits = logits.masked_fill(~repaired, torch.finfo(logits.dtype).min)
    return Categorical(logits=masked_logits)


# ---------------------------------------------------------------------------
# Shared actor-critic network
# ---------------------------------------------------------------------------

class SharedActorCritic(nn.Module):
    """MLP shared across all slots/agents. Outputs (logits, value)."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 128,
        n_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.actor_head = nn.Linear(hidden_dim, n_actions)
        self.critic_head = nn.Linear(hidden_dim, 1)
        # Orthogonal init — standard PPO recipe.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        return self.actor_head(h), self.critic_head(h).squeeze(-1)

    def act(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action under the mask. Returns (action, logprob, value)."""
        logits, value = self.forward(obs)
        dist = masked_categorical(logits, action_mask)
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, value

    def evaluate(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score given actions under current policy. Returns (logprob, entropy, value)."""
        logits, value = self.forward(obs)
        dist = masked_categorical(logits, action_mask)
        return dist.log_prob(actions), dist.entropy(), value


# ---------------------------------------------------------------------------
# Rollout buffer + GAE
# ---------------------------------------------------------------------------

@dataclass
class RolloutBatch:
    """Flat (active-only) tensors ready for PPO minibatching."""

    obs: torch.Tensor
    action_mask: torch.Tensor
    actions: torch.Tensor
    old_logprobs: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    def __len__(self) -> int:
        return self.obs.shape[0]


class RolloutBuffer:
    """Stores (T, n_envs, n_agents, *) tensors and computes GAE per slot.

    GAE recurrence respects per-slot life boundaries: the bootstrap from
    V(s_{t+1}) is gated by ``next_active & ~done[t]`` so a deceased life
    never receives credit from its successor and an inactive gap never
    leaks values into a prior alive transition.
    """

    def __init__(
        self,
        rollout_length: int,
        n_envs: int,
        n_agents: int,
        obs_dim: int,
        n_actions: int,
        device: torch.device | str = "cpu",
    ) -> None:
        self.T = rollout_length
        self.n_envs = n_envs
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = torch.device(device)

        shape3 = (rollout_length, n_envs, n_agents)
        self.obs = torch.zeros(shape3 + (obs_dim,), dtype=torch.float32, device=self.device)
        self.action_masks = torch.zeros(shape3 + (n_actions,), dtype=torch.bool, device=self.device)
        self.actions = torch.zeros(shape3, dtype=torch.long, device=self.device)
        self.logprobs = torch.zeros(shape3, dtype=torch.float32, device=self.device)
        self.values = torch.zeros(shape3, dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros(shape3, dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(shape3, dtype=torch.bool, device=self.device)
        self.active = torch.zeros(shape3, dtype=torch.bool, device=self.device)
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray | torch.Tensor,
        action_mask: np.ndarray | torch.Tensor,
        action: torch.Tensor,
        logprob: torch.Tensor,
        value: torch.Tensor,
        reward: np.ndarray | torch.Tensor,
        done: np.ndarray | torch.Tensor,
        active: np.ndarray | torch.Tensor,
    ) -> None:
        assert self.ptr < self.T, "RolloutBuffer overflow"
        self.obs[self.ptr] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.action_masks[self.ptr] = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
        self.actions[self.ptr] = action.detach().to(self.device)
        self.logprobs[self.ptr] = logprob.detach().to(self.device)
        self.values[self.ptr] = value.detach().to(self.device)
        self.rewards[self.ptr] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        self.dones[self.ptr] = torch.as_tensor(done, dtype=torch.bool, device=self.device)
        self.active[self.ptr] = torch.as_tensor(active, dtype=torch.bool, device=self.device)
        self.ptr += 1

    def compute_gae(
        self,
        last_value: torch.Tensor,
        last_active: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages + returns. Inactive slots get zero advantage.

        Args:
            last_value:  V(s_T) for each (env, agent), shape (n_envs, n_agents).
            last_active: active mask AT step T (the post-rollout state),
                         shape (n_envs, n_agents).
            gamma, lam:  discount and GAE smoothing.

        Returns:
            advantages, returns — both shape (T, n_envs, n_agents).
        """
        assert self.ptr == self.T, "fill the buffer before computing GAE"
        adv = torch.zeros_like(self.rewards)
        last_gae = torch.zeros(self.n_envs, self.n_agents, device=self.device)
        next_value = last_value.to(self.device)
        next_active = last_active.to(self.device)

        for t in reversed(range(self.T)):
            # Bootstrap is gated by: not terminated this step AND next slot is active.
            next_nonterminal = (~self.dones[t]) & next_active
            delta = self.rewards[t] + gamma * next_value * next_nonterminal.float() - self.values[t]
            last_gae = delta + gamma * lam * next_nonterminal.float() * last_gae
            adv[t] = last_gae * self.active[t].float()
            next_value = self.values[t]
            next_active = self.active[t]

        returns = adv + self.values
        return adv, returns

    def flatten_active(
        self, advantages: torch.Tensor, returns: torch.Tensor
    ) -> RolloutBatch:
        """Flatten to (N_active, *) by selecting only active transitions."""
        flat_active = self.active.reshape(-1)
        idx = flat_active.nonzero(as_tuple=False).squeeze(-1)

        def _sel(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(-1, *t.shape[3:])[idx]

        return RolloutBatch(
            obs=_sel(self.obs),
            action_mask=_sel(self.action_masks),
            actions=_sel(self.actions),
            old_logprobs=_sel(self.logprobs),
            values=_sel(self.values),
            advantages=_sel(advantages),
            returns=_sel(returns),
        )

    def reset(self) -> None:
        self.ptr = 0


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

@dataclass
class PPOStats:
    pg_loss: float = 0.0
    vf_loss: float = 0.0
    entropy: float = 0.0
    approx_kl: float = 0.0
    clip_frac: float = 0.0
    n_updates: int = 0
    n_skipped: int = 0


def ppo_update(
    policy: SharedActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    clip_coef: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    n_epochs: int = 4,
    n_minibatches: int = 4,
    max_grad_norm: float = 0.5,
    normalize_adv: bool = True,
    rng: torch.Generator | None = None,
) -> PPOStats:
    """Run ``n_epochs`` of PPO over ``n_minibatches`` shuffles.

    All masking has been applied upstream by ``flatten_active`` — every
    sample in ``batch`` is from an active slot. Empty batch → no-op.
    """
    stats = PPOStats()
    n = len(batch)
    if n == 0:
        return stats

    advantages = batch.advantages
    if normalize_adv and n >= 2:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    minibatch_size = max(1, n // n_minibatches)
    perm_kwargs = {"generator": rng} if rng is not None else {}

    for _ in range(n_epochs):
        perm = torch.randperm(n, device=batch.obs.device, **perm_kwargs)
        for start in range(0, n, minibatch_size):
            mb_idx = perm[start : start + minibatch_size]
            if mb_idx.numel() == 0:
                stats.n_skipped += 1
                continue

            mb_obs = batch.obs[mb_idx]
            mb_mask = batch.action_mask[mb_idx]
            mb_actions = batch.actions[mb_idx]
            mb_old_logprobs = batch.old_logprobs[mb_idx]
            mb_values = batch.values[mb_idx]
            mb_adv = advantages[mb_idx]
            mb_ret = batch.returns[mb_idx]

            new_logprobs, entropy, new_values = policy.evaluate(mb_obs, mb_mask, mb_actions)

            log_ratio = new_logprobs - mb_old_logprobs
            ratio = log_ratio.exp()

            unclipped = ratio * mb_adv
            clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * mb_adv
            pg_loss = -torch.min(unclipped, clipped).mean()

            v_clipped = mb_values + torch.clamp(new_values - mb_values, -clip_coef, clip_coef)
            vf_unclipped = (new_values - mb_ret) ** 2
            vf_clipped_loss = (v_clipped - mb_ret) ** 2
            vf_loss = 0.5 * torch.max(vf_unclipped, vf_clipped_loss).mean()

            ent_loss = entropy.mean()
            loss = pg_loss + vf_coef * vf_loss - ent_coef * ent_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                stats.pg_loss += float(pg_loss)
                stats.vf_loss += float(vf_loss)
                stats.entropy += float(ent_loss)
                stats.approx_kl += float((-log_ratio).mean())
                stats.clip_frac += float(((ratio - 1.0).abs() > clip_coef).float().mean())
                stats.n_updates += 1

    if stats.n_updates > 0:
        stats.pg_loss /= stats.n_updates
        stats.vf_loss /= stats.n_updates
        stats.entropy /= stats.n_updates
        stats.approx_kl /= stats.n_updates
        stats.clip_frac /= stats.n_updates
    return stats
