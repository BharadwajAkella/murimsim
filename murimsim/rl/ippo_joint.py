"""v24 joint-action IPPO: split body (16) + social (2) action heads.

Mirror of ``ippo.py`` but with two independent action heads sharing the
trunk. Independence assumption ⇒ ``log P(b, s) = log P(b) + log P(s)`` and
total entropy = ``H(body) + H(social)``. PPO ratio uses summed log-probs;
losses otherwise unchanged.

Why split:
    v23 diagnostics showed ``COLLABORATE`` was chosen 0.32 % of active
    steps because it had to compete with ``TRAIN``/``MOVE_*``/etc. inside
    a single 17-action softmax. Splitting decouples opportunity cost so
    the policy can ``train`` and ``collaborate`` on the same step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from murimsim.actions import N_BODY_ACTIONS, N_SOCIAL_ACTIONS, BodyAction
from murimsim.rl.ippo import masked_categorical, repair_action_mask

REST_BODY_ACTION: int = BodyAction.REST.value


# ---------------------------------------------------------------------------
# Joint actor-critic
# ---------------------------------------------------------------------------

class JointSharedActorCritic(nn.Module):
    """Shared MLP trunk → body head (16) + social head (2) + value head."""

    def __init__(
        self,
        obs_dim: int,
        n_body_actions: int = N_BODY_ACTIONS,
        n_social_actions: int = N_SOCIAL_ACTIONS,
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
        self.body_head = nn.Linear(hidden_dim, n_body_actions)
        self.social_head = nn.Linear(hidden_dim, n_social_actions)
        self.critic_head = nn.Linear(hidden_dim, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.body_head.weight, gain=0.01)
        nn.init.orthogonal_(self.social_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        return (
            self.body_head(h),
            self.social_head(h),
            self.critic_head(h).squeeze(-1),
        )

    def act(
        self,
        obs: torch.Tensor,
        body_mask: torch.Tensor,
        social_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Sample (body, social). Returns (body_a, social_a, body_lp, social_lp, value)."""
        body_logits, social_logits, value = self.forward(obs)
        body_dist = masked_categorical(body_logits, body_mask)
        social_dist = masked_categorical(social_logits, social_mask)
        if deterministic:
            body_a = body_dist.probs.argmax(dim=-1)
            social_a = social_dist.probs.argmax(dim=-1)
        else:
            body_a = body_dist.sample()
            social_a = social_dist.sample()
        return (
            body_a,
            social_a,
            body_dist.log_prob(body_a),
            social_dist.log_prob(social_a),
            value,
        )

    def evaluate(
        self,
        obs: torch.Tensor,
        body_mask: torch.Tensor,
        social_mask: torch.Tensor,
        body_actions: torch.Tensor,
        social_actions: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Re-score given (body, social). Returns per-head (lp, ent) + value."""
        body_logits, social_logits, value = self.forward(obs)
        body_dist = masked_categorical(body_logits, body_mask)
        social_dist = masked_categorical(social_logits, social_mask)
        return (
            body_dist.log_prob(body_actions),
            social_dist.log_prob(social_actions),
            body_dist.entropy(),
            social_dist.entropy(),
            value,
        )


# ---------------------------------------------------------------------------
# Joint rollout buffer
# ---------------------------------------------------------------------------

@dataclass
class JointRolloutBatch:
    obs: torch.Tensor
    body_mask: torch.Tensor
    social_mask: torch.Tensor
    body_actions: torch.Tensor
    social_actions: torch.Tensor
    old_body_logprobs: torch.Tensor
    old_social_logprobs: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    def __len__(self) -> int:
        return self.obs.shape[0]


class JointRolloutBuffer:
    """Two-headed twin of ``RolloutBuffer``. GAE logic identical."""

    def __init__(
        self,
        rollout_length: int,
        n_envs: int,
        n_agents: int,
        obs_dim: int,
        n_body_actions: int = N_BODY_ACTIONS,
        n_social_actions: int = N_SOCIAL_ACTIONS,
        device: torch.device | str = "cpu",
    ) -> None:
        self.T = rollout_length
        self.n_envs = n_envs
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.n_body_actions = n_body_actions
        self.n_social_actions = n_social_actions
        self.device = torch.device(device)

        shape3 = (rollout_length, n_envs, n_agents)
        self.obs = torch.zeros(shape3 + (obs_dim,), dtype=torch.float32, device=self.device)
        self.body_masks = torch.zeros(shape3 + (n_body_actions,), dtype=torch.bool, device=self.device)
        self.social_masks = torch.zeros(shape3 + (n_social_actions,), dtype=torch.bool, device=self.device)
        self.body_actions = torch.zeros(shape3, dtype=torch.long, device=self.device)
        self.social_actions = torch.zeros(shape3, dtype=torch.long, device=self.device)
        self.body_logprobs = torch.zeros(shape3, dtype=torch.float32, device=self.device)
        self.social_logprobs = torch.zeros(shape3, dtype=torch.float32, device=self.device)
        self.values = torch.zeros(shape3, dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros(shape3, dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(shape3, dtype=torch.bool, device=self.device)
        self.active = torch.zeros(shape3, dtype=torch.bool, device=self.device)
        self.ptr = 0

    def add(
        self,
        obs,
        body_mask,
        social_mask,
        body_action: torch.Tensor,
        social_action: torch.Tensor,
        body_logprob: torch.Tensor,
        social_logprob: torch.Tensor,
        value: torch.Tensor,
        reward,
        done,
        active,
    ) -> None:
        assert self.ptr < self.T, "JointRolloutBuffer overflow"
        t = self.ptr
        self.obs[t] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.body_masks[t] = torch.as_tensor(body_mask, dtype=torch.bool, device=self.device)
        self.social_masks[t] = torch.as_tensor(social_mask, dtype=torch.bool, device=self.device)
        self.body_actions[t] = body_action.detach().to(self.device)
        self.social_actions[t] = social_action.detach().to(self.device)
        self.body_logprobs[t] = body_logprob.detach().to(self.device)
        self.social_logprobs[t] = social_logprob.detach().to(self.device)
        self.values[t] = value.detach().to(self.device)
        self.rewards[t] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        self.dones[t] = torch.as_tensor(done, dtype=torch.bool, device=self.device)
        self.active[t] = torch.as_tensor(active, dtype=torch.bool, device=self.device)
        self.ptr += 1

    def compute_gae(
        self,
        last_value: torch.Tensor,
        last_active: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.ptr == self.T, "fill buffer before GAE"
        adv = torch.zeros_like(self.rewards)
        last_gae = torch.zeros(self.n_envs, self.n_agents, device=self.device)
        next_value = last_value.to(self.device)
        next_active = last_active.to(self.device)

        for t in reversed(range(self.T)):
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
    ) -> JointRolloutBatch:
        flat_active = self.active.reshape(-1)
        idx = flat_active.nonzero(as_tuple=False).squeeze(-1)

        def _sel(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(-1, *t.shape[3:])[idx]

        return JointRolloutBatch(
            obs=_sel(self.obs),
            body_mask=_sel(self.body_masks),
            social_mask=_sel(self.social_masks),
            body_actions=_sel(self.body_actions),
            social_actions=_sel(self.social_actions),
            old_body_logprobs=_sel(self.body_logprobs),
            old_social_logprobs=_sel(self.social_logprobs),
            values=_sel(self.values),
            advantages=_sel(advantages),
            returns=_sel(returns),
        )

    def reset(self) -> None:
        self.ptr = 0


# ---------------------------------------------------------------------------
# Joint PPO update
# ---------------------------------------------------------------------------

@dataclass
class JointPPOStats:
    pg_loss: float = 0.0
    vf_loss: float = 0.0
    body_entropy: float = 0.0
    social_entropy: float = 0.0
    approx_kl: float = 0.0
    clip_frac: float = 0.0
    n_updates: int = 0
    n_skipped: int = 0


def joint_ppo_update(
    policy: JointSharedActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: JointRolloutBatch,
    clip_coef: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    n_epochs: int = 4,
    n_minibatches: int = 4,
    max_grad_norm: float = 0.5,
    normalize_adv: bool = True,
    rng: torch.Generator | None = None,
) -> JointPPOStats:
    """PPO over joint (body, social) policy.

    Independence ⇒ summed log-probs and summed entropies. The PPO ratio
    therefore is ``exp((b_lp_new + s_lp_new) - (b_lp_old + s_lp_old))``.
    """
    stats = JointPPOStats()
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
            mb_body_mask = batch.body_mask[mb_idx]
            mb_social_mask = batch.social_mask[mb_idx]
            mb_body_a = batch.body_actions[mb_idx]
            mb_social_a = batch.social_actions[mb_idx]
            mb_old_body_lp = batch.old_body_logprobs[mb_idx]
            mb_old_social_lp = batch.old_social_logprobs[mb_idx]
            mb_values = batch.values[mb_idx]
            mb_adv = advantages[mb_idx]
            mb_ret = batch.returns[mb_idx]

            new_body_lp, new_social_lp, body_ent, social_ent, new_values = policy.evaluate(
                mb_obs, mb_body_mask, mb_social_mask, mb_body_a, mb_social_a
            )

            old_joint_lp = mb_old_body_lp + mb_old_social_lp
            new_joint_lp = new_body_lp + new_social_lp
            log_ratio = new_joint_lp - old_joint_lp
            ratio = log_ratio.exp()

            unclipped = ratio * mb_adv
            clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * mb_adv
            pg_loss = -torch.min(unclipped, clipped).mean()

            v_clipped = mb_values + torch.clamp(new_values - mb_values, -clip_coef, clip_coef)
            vf_unclipped = (new_values - mb_ret) ** 2
            vf_clipped_loss = (v_clipped - mb_ret) ** 2
            vf_loss = 0.5 * torch.max(vf_unclipped, vf_clipped_loss).mean()

            joint_ent = body_ent.mean() + social_ent.mean()
            loss = pg_loss + vf_coef * vf_loss - ent_coef * joint_ent

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                stats.pg_loss += float(pg_loss)
                stats.vf_loss += float(vf_loss)
                stats.body_entropy += float(body_ent.mean())
                stats.social_entropy += float(social_ent.mean())
                stats.approx_kl += float((-log_ratio).mean())
                stats.clip_frac += float(((ratio - 1.0).abs() > clip_coef).float().mean())
                stats.n_updates += 1

    if stats.n_updates > 0:
        stats.pg_loss /= stats.n_updates
        stats.vf_loss /= stats.n_updates
        stats.body_entropy /= stats.n_updates
        stats.social_entropy /= stats.n_updates
        stats.approx_kl /= stats.n_updates
        stats.clip_frac /= stats.n_updates
    return stats
