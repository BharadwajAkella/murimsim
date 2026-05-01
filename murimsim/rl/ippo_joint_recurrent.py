"""v24 joint-action recurrent IPPO — LSTM trunk + body + social heads.

Mirror of ``ippo_recurrent.py`` with the v24 head split. The LSTM
output feeds two heads (body Discrete(16), social Discrete(2)) plus a
shared value head. Sequence buffer + life-aware hidden reset semantics
are unchanged from the single-head variant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from murimsim.actions import N_BODY_ACTIONS, N_SOCIAL_ACTIONS
from murimsim.rl.ippo import masked_categorical


class JointRecurrentSharedActorCritic(nn.Module):
    """LSTM shared trunk → body head + social head + value head."""

    def __init__(
        self,
        obs_dim: int,
        n_body_actions: int = N_BODY_ACTIONS,
        n_social_actions: int = N_SOCIAL_ACTIONS,
        hidden_dim: int = 128,
        pre_lstm_dim: int = 128,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_body_actions = n_body_actions
        self.n_social_actions = n_social_actions
        self.hidden_dim = hidden_dim

        self.pre_lstm = nn.Sequential(nn.Linear(obs_dim, pre_lstm_dim), nn.Tanh())
        self.lstm = nn.LSTM(input_size=pre_lstm_dim, hidden_size=hidden_dim, num_layers=1)
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
        for name, p in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(p, gain=1.0)
            elif "bias" in name:
                nn.init.zeros_(p)

    def initial_hidden(
        self, batch_size: int, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = torch.device(device)
        h = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return h, c

    def act(
        self,
        obs: torch.Tensor,
        body_mask: torch.Tensor,
        social_mask: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
        deterministic: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
    ]:
        x = self.pre_lstm(obs).unsqueeze(0)
        out, new_hidden = self.lstm(x, hidden)
        feat = out.squeeze(0)
        body_logits = self.body_head(feat)
        social_logits = self.social_head(feat)
        value = self.critic_head(feat).squeeze(-1)
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
            new_hidden,
        )

    @torch.no_grad()
    def value_only(
        self,
        obs: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        x = self.pre_lstm(obs).unsqueeze(0)
        out, _ = self.lstm(x, hidden)
        feat = out.squeeze(0)
        return self.critic_head(feat).squeeze(-1)

    def evaluate_sequence(
        self,
        obs_seq: torch.Tensor,
        body_mask_seq: torch.Tensor,
        social_mask_seq: torch.Tensor,
        body_action_seq: torch.Tensor,
        social_action_seq: torch.Tensor,
        init_hidden: tuple[torch.Tensor, torch.Tensor],
        life_reset: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        T, B, _ = obs_seq.shape
        h, c = init_hidden
        x = self.pre_lstm(obs_seq)
        out_list: list[torch.Tensor] = []
        for t in range(T):
            if life_reset[t].any():
                reset_b = life_reset[t]
                mask = (~reset_b).view(1, B, 1).to(h.dtype)
                h = h * mask
                c = c * mask
            step_in = x[t : t + 1]
            step_out, (h, c) = self.lstm(step_in, (h, c))
            out_list.append(step_out.squeeze(0))
        feat = torch.stack(out_list, dim=0)
        body_logits = self.body_head(feat)
        social_logits = self.social_head(feat)
        values = self.critic_head(feat).squeeze(-1)

        flat_body_logits = body_logits.reshape(T * B, -1)
        flat_social_logits = social_logits.reshape(T * B, -1)
        flat_body_mask = body_mask_seq.reshape(T * B, -1)
        flat_social_mask = social_mask_seq.reshape(T * B, -1)
        flat_body_a = body_action_seq.reshape(T * B)
        flat_social_a = social_action_seq.reshape(T * B)

        body_dist = masked_categorical(flat_body_logits, flat_body_mask)
        social_dist = masked_categorical(flat_social_logits, flat_social_mask)
        body_lp = body_dist.log_prob(flat_body_a).reshape(T, B)
        social_lp = social_dist.log_prob(flat_social_a).reshape(T, B)
        body_ent = body_dist.entropy().reshape(T, B)
        social_ent = social_dist.entropy().reshape(T, B)
        return body_lp, social_lp, body_ent, social_ent, values


@dataclass
class JointSequenceBatch:
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
    active: torch.Tensor
    life_reset: torch.Tensor
    init_h: torch.Tensor
    init_c: torch.Tensor


class JointRecurrentRolloutBuffer:
    """Twin of ``RecurrentRolloutBuffer`` carrying body+social tensors."""

    def __init__(
        self,
        rollout_length: int,
        n_envs: int,
        n_agents: int,
        obs_dim: int,
        hidden_dim: int,
        n_body_actions: int = N_BODY_ACTIONS,
        n_social_actions: int = N_SOCIAL_ACTIONS,
        device: torch.device | str = "cpu",
    ) -> None:
        self.T = rollout_length
        self.n_envs = n_envs
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
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
        self.life_reset = torch.zeros(shape3, dtype=torch.bool, device=self.device)

        B = n_envs * n_agents
        self.init_h = torch.zeros(1, B, hidden_dim, device=self.device)
        self.init_c = torch.zeros(1, B, hidden_dim, device=self.device)
        self.ptr = 0

    def set_initial_hidden(self, h: torch.Tensor, c: torch.Tensor) -> None:
        assert h.shape == self.init_h.shape, (h.shape, self.init_h.shape)
        assert c.shape == self.init_c.shape, (c.shape, self.init_c.shape)
        self.init_h = h.detach().clone().to(self.device)
        self.init_c = c.detach().clone().to(self.device)

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
        life_reset,
    ) -> None:
        assert self.ptr < self.T, "JointRecurrentRolloutBuffer overflow"
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
        self.life_reset[t] = torch.as_tensor(life_reset, dtype=torch.bool, device=self.device)
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

    def to_sequence_batch(
        self, advantages: torch.Tensor, returns: torch.Tensor
    ) -> JointSequenceBatch:
        T = self.T
        B = self.n_envs * self.n_agents

        def _r(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(T, B, *t.shape[3:])

        return JointSequenceBatch(
            obs=_r(self.obs),
            body_mask=_r(self.body_masks),
            social_mask=_r(self.social_masks),
            body_actions=_r(self.body_actions),
            social_actions=_r(self.social_actions),
            old_body_logprobs=_r(self.body_logprobs),
            old_social_logprobs=_r(self.social_logprobs),
            values=_r(self.values),
            advantages=_r(advantages),
            returns=_r(returns),
            active=_r(self.active),
            life_reset=_r(self.life_reset),
            init_h=self.init_h,
            init_c=self.init_c,
        )

    def reset(self) -> None:
        self.ptr = 0


@dataclass
class JointRecurrentPPOStats:
    pg_loss: float = 0.0
    vf_loss: float = 0.0
    body_entropy: float = 0.0
    social_entropy: float = 0.0
    approx_kl: float = 0.0
    clip_frac: float = 0.0
    n_updates: int = 0
    n_skipped: int = 0


def joint_recurrent_ppo_update(
    policy: JointRecurrentSharedActorCritic,
    optimizer: torch.optim.Optimizer,
    seq_batch: JointSequenceBatch,
    clip_coef: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    n_epochs: int = 4,
    n_minibatches: int = 4,
    max_grad_norm: float = 0.5,
    normalize_adv: bool = True,
    rng: torch.Generator | None = None,
) -> JointRecurrentPPOStats:
    """Sequence-PPO with joint body+social action heads."""
    stats = JointRecurrentPPOStats()
    T, B, _ = seq_batch.obs.shape
    if B == 0 or T == 0:
        return stats

    advantages = seq_batch.advantages
    active = seq_batch.active.to(torch.float32)
    n_active_total = int(active.sum().item())
    if normalize_adv and n_active_total >= 2:
        mask_bool = seq_batch.active
        active_adv = advantages[mask_bool]
        mean = active_adv.mean()
        std = active_adv.std(unbiased=False) + 1e-8
        advantages = (advantages - mean) / std
        advantages = advantages * active

    minibatch_b = max(1, B // n_minibatches)
    perm_kwargs = {"generator": rng} if rng is not None else {}
    device = seq_batch.obs.device

    for _ in range(n_epochs):
        perm = torch.randperm(B, device=device, **perm_kwargs)
        for start in range(0, B, minibatch_b):
            mb_idx = perm[start : start + minibatch_b]
            if mb_idx.numel() == 0:
                stats.n_skipped += 1
                continue

            mb_obs = seq_batch.obs[:, mb_idx]
            mb_body_mask = seq_batch.body_mask[:, mb_idx]
            mb_social_mask = seq_batch.social_mask[:, mb_idx]
            mb_body_a = seq_batch.body_actions[:, mb_idx]
            mb_social_a = seq_batch.social_actions[:, mb_idx]
            mb_old_body_lp = seq_batch.old_body_logprobs[:, mb_idx]
            mb_old_social_lp = seq_batch.old_social_logprobs[:, mb_idx]
            mb_values = seq_batch.values[:, mb_idx]
            mb_adv = advantages[:, mb_idx]
            mb_ret = seq_batch.returns[:, mb_idx]
            mb_active = seq_batch.active[:, mb_idx].to(torch.float32)
            mb_life_reset = seq_batch.life_reset[:, mb_idx]
            mb_init_h = seq_batch.init_h[:, mb_idx].contiguous()
            mb_init_c = seq_batch.init_c[:, mb_idx].contiguous()

            mb_active_count = mb_active.sum()
            if float(mb_active_count.item()) == 0.0:
                stats.n_skipped += 1
                continue

            new_body_lp, new_social_lp, body_ent, social_ent, new_values = policy.evaluate_sequence(
                mb_obs,
                mb_body_mask,
                mb_social_mask,
                mb_body_a,
                mb_social_a,
                (mb_init_h, mb_init_c),
                mb_life_reset,
            )

            old_joint_lp = mb_old_body_lp + mb_old_social_lp
            new_joint_lp = new_body_lp + new_social_lp
            log_ratio = new_joint_lp - old_joint_lp
            ratio = log_ratio.exp()

            unclipped = ratio * mb_adv
            clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * mb_adv
            pg_per_step = -torch.min(unclipped, clipped) * mb_active
            pg_loss = pg_per_step.sum() / mb_active_count

            v_clipped = mb_values + torch.clamp(new_values - mb_values, -clip_coef, clip_coef)
            vf_unclipped = (new_values - mb_ret) ** 2
            vf_clipped_loss = (v_clipped - mb_ret) ** 2
            vf_per_step = 0.5 * torch.max(vf_unclipped, vf_clipped_loss) * mb_active
            vf_loss = vf_per_step.sum() / mb_active_count

            body_ent_loss = (body_ent * mb_active).sum() / mb_active_count
            social_ent_loss = (social_ent * mb_active).sum() / mb_active_count
            ent_loss = body_ent_loss + social_ent_loss
            loss = pg_loss + vf_coef * vf_loss - ent_coef * ent_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                stats.pg_loss += float(pg_loss)
                stats.vf_loss += float(vf_loss)
                stats.body_entropy += float(body_ent_loss)
                stats.social_entropy += float(social_ent_loss)
                kl = ((-log_ratio) * mb_active).sum() / mb_active_count
                stats.approx_kl += float(kl)
                clipped_mask = ((ratio - 1.0).abs() > clip_coef).float() * mb_active
                stats.clip_frac += float(clipped_mask.sum() / mb_active_count)
                stats.n_updates += 1

    if stats.n_updates > 0:
        stats.pg_loss /= stats.n_updates
        stats.vf_loss /= stats.n_updates
        stats.body_entropy /= stats.n_updates
        stats.social_entropy /= stats.n_updates
        stats.approx_kl /= stats.n_updates
        stats.clip_frac /= stats.n_updates
    return stats
