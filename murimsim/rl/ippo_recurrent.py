"""Recurrent IPPO — Phase 4 of the IPPO migration.

Adds an LSTM-based actor-critic on top of the same shared-parameters / per-slot
active-mask design as ``murimsim.rl.ippo``. Three pieces:

    * ``RecurrentSharedActorCritic`` — pre-LSTM linear → LSTM → actor + critic heads.
    * ``RecurrentRolloutBuffer``     — same (T, n_envs, n_agents, *) layout plus
                                        ``life_reset`` flags and per-slot initial
                                        hidden state captured at rollout start.
    * ``recurrent_ppo_update``       — minibatches over (env*agent) trajectories,
                                        not steps. PPO loss masked by ``active``.

Lifecycle / hidden-state semantics (CRITICAL — see P4 design notes)
-------------------------------------------------------------------
``life_reset[t, e, a]`` is True iff the LSTM hidden state ``(h, c)`` for slot
``(e, a)`` was reset to zero **before** consuming ``obs[t, e, a]``. This means
``life_reset[t]`` is the *pending* reset flag carried in **from** the previous
step's lifecycle outcome, **not** ``info['lifecycle'][i]['born']`` of step ``t``
itself (which describes what just happened to produce ``obs[t+1]``). The
trainer is responsible for converting ``born`` → ``pending_life_reset`` between
``step_all`` calls; this module's update path *trusts* the stored flag and
applies the same reset symmetry at evaluate time.

Truncated BPTT
--------------
The rollout length ``T`` is the BPTT window. Hidden state at step ``t`` carries
gradient through every prior step in the rollout. With T=128, n_envs=8,
n_agents=4, H=128 the activation tensors are well within memory budget.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from murimsim.rl.ippo import masked_categorical


# ---------------------------------------------------------------------------
# Recurrent shared actor-critic
# ---------------------------------------------------------------------------

class RecurrentSharedActorCritic(nn.Module):
    """LSTM actor-critic shared across all (env, slot) pairs.

    Parameters are shared; hidden state is per (env, slot) and tracked
    externally by the trainer / buffer. Single-layer LSTM keeps the parameter
    count modest and matches CleanRL convention.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 128,
        pre_lstm_dim: int = 128,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        self.pre_lstm = nn.Sequential(
            nn.Linear(obs_dim, pre_lstm_dim),
            nn.Tanh(),
        )
        self.lstm = nn.LSTM(
            input_size=pre_lstm_dim,
            hidden_size=hidden_dim,
            num_layers=1,
        )
        self.actor_head = nn.Linear(hidden_dim, n_actions)
        self.critic_head = nn.Linear(hidden_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        for name, p in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(p, gain=1.0)
            elif "bias" in name:
                nn.init.zeros_(p)

    def initial_hidden(
        self, batch_size: int, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero (h, c) of shape (1, batch_size, hidden_dim) — for new lives."""
        device = torch.device(device)
        h = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return h, c

    def act(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Single-step rollout call.

        Args:
            obs:         (B, obs_dim)
            action_mask: (B, n_actions) bool
            hidden:      ((1, B, H), (1, B, H))
            deterministic: argmax over the masked categorical when True.

        Returns:
            (action[B], logprob[B], value[B], (new_h, new_c))
        """
        x = self.pre_lstm(obs).unsqueeze(0)  # (T=1, B, pre_lstm_dim)
        out, new_hidden = self.lstm(x, hidden)
        feat = out.squeeze(0)
        logits = self.actor_head(feat)
        value = self.critic_head(feat).squeeze(-1)
        dist = masked_categorical(logits, action_mask)
        action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, value, new_hidden

    @torch.no_grad()
    def value_only(
        self,
        obs: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Critic-only forward for GAE bootstrap. Does NOT mutate hidden.

        We discard the new hidden state here — the trainer keeps the carried
        hidden from the actual ``act`` call that produced obs_T.
        """
        x = self.pre_lstm(obs).unsqueeze(0)
        out, _ = self.lstm(x, hidden)
        feat = out.squeeze(0)
        return self.critic_head(feat).squeeze(-1)

    def evaluate_sequence(
        self,
        obs_seq: torch.Tensor,
        mask_seq: torch.Tensor,
        action_seq: torch.Tensor,
        init_hidden: tuple[torch.Tensor, torch.Tensor],
        life_reset: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-score stored sequences for the PPO update.

        Walks ``T`` steps with a Python loop, zeroing per-batch hidden state
        slices where ``life_reset[t, b]`` is True BEFORE that step's LSTM call.
        Symmetry with rollout: at rollout time the trainer also resets BEFORE
        consuming ``obs[t]``.

        Args:
            obs_seq:     (T, B, obs_dim)
            mask_seq:    (T, B, n_actions) bool
            action_seq:  (T, B) long
            init_hidden: ((1, B, H), (1, B, H)) — the hidden state captured at
                          the start of the rollout for these B trajectories.
            life_reset:  (T, B) bool — True at steps where hidden was reset.

        Returns:
            (logprobs[T, B], entropy[T, B], values[T, B])
        """
        T, B, _ = obs_seq.shape
        h, c = init_hidden
        # Pre-LSTM is time-distributed; safe to apply once over the full T,B.
        x = self.pre_lstm(obs_seq)  # (T, B, pre_lstm_dim)

        out_list: list[torch.Tensor] = []
        for t in range(T):
            if life_reset[t].any():
                reset_b = life_reset[t]
                # In-place mutation of a leaf would break autograd; build a new
                # tensor zeroing the affected batch slices.
                mask = (~reset_b).view(1, B, 1).to(h.dtype)
                h = h * mask
                c = c * mask
            step_in = x[t : t + 1]  # (1, B, pre_lstm_dim)
            step_out, (h, c) = self.lstm(step_in, (h, c))
            out_list.append(step_out.squeeze(0))

        feat = torch.stack(out_list, dim=0)  # (T, B, H)
        logits = self.actor_head(feat)  # (T, B, n_actions)
        values = self.critic_head(feat).squeeze(-1)  # (T, B)

        flat_logits = logits.reshape(T * B, -1)
        flat_mask = mask_seq.reshape(T * B, -1)
        flat_actions = action_seq.reshape(T * B)
        dist = masked_categorical(flat_logits, flat_mask)
        logprobs = dist.log_prob(flat_actions).reshape(T, B)
        entropy = dist.entropy().reshape(T, B)
        return logprobs, entropy, values


# ---------------------------------------------------------------------------
# Recurrent rollout buffer
# ---------------------------------------------------------------------------

@dataclass
class SequenceBatch:
    """Per-trajectory (T, B, *) tensors ready for recurrent PPO minibatching."""

    obs: torch.Tensor          # (T, B, obs_dim)
    action_mask: torch.Tensor  # (T, B, n_actions)
    actions: torch.Tensor      # (T, B)
    old_logprobs: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    active: torch.Tensor       # (T, B) bool
    life_reset: torch.Tensor   # (T, B) bool
    init_h: torch.Tensor       # (1, B, H)
    init_c: torch.Tensor       # (1, B, H)


class RecurrentRolloutBuffer:
    """Stores rollouts AND the per-slot life_reset flag and initial hidden.

    Layout matches ``RolloutBuffer`` (T, n_envs, n_agents, *) for everything
    that wasn't recurrence-specific.
    """

    def __init__(
        self,
        rollout_length: int,
        n_envs: int,
        n_agents: int,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int,
        device: torch.device | str = "cpu",
    ) -> None:
        self.T = rollout_length
        self.n_envs = n_envs
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
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
        self.life_reset = torch.zeros(shape3, dtype=torch.bool, device=self.device)

        # (1, n_envs * n_agents, H) — captured ONCE at rollout start.
        B = n_envs * n_agents
        self.init_h = torch.zeros(1, B, hidden_dim, device=self.device)
        self.init_c = torch.zeros(1, B, hidden_dim, device=self.device)
        self.ptr = 0

    def set_initial_hidden(self, h: torch.Tensor, c: torch.Tensor) -> None:
        """Snapshot the (h, c) carried into this rollout. Called BEFORE filling.

        Args:
            h: (1, n_envs * n_agents, H)
            c: (1, n_envs * n_agents, H)
        """
        assert h.shape == self.init_h.shape, (h.shape, self.init_h.shape)
        assert c.shape == self.init_c.shape, (c.shape, self.init_c.shape)
        self.init_h = h.detach().clone().to(self.device)
        self.init_c = c.detach().clone().to(self.device)

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
        life_reset: np.ndarray | torch.Tensor,
    ) -> None:
        assert self.ptr < self.T, "RecurrentRolloutBuffer overflow"
        self.obs[self.ptr] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.action_masks[self.ptr] = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
        self.actions[self.ptr] = action.detach().to(self.device)
        self.logprobs[self.ptr] = logprob.detach().to(self.device)
        self.values[self.ptr] = value.detach().to(self.device)
        self.rewards[self.ptr] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        self.dones[self.ptr] = torch.as_tensor(done, dtype=torch.bool, device=self.device)
        self.active[self.ptr] = torch.as_tensor(active, dtype=torch.bool, device=self.device)
        self.life_reset[self.ptr] = torch.as_tensor(life_reset, dtype=torch.bool, device=self.device)
        self.ptr += 1

    def compute_gae(
        self,
        last_value: torch.Tensor,
        last_active: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-slot life-aware GAE — identical recurrence to the FF buffer."""
        assert self.ptr == self.T, "fill the buffer before computing GAE"
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
    ) -> SequenceBatch:
        """Reshape (T, n_envs, n_agents, *) → (T, B=n_envs*n_agents, *)."""
        T = self.T
        B = self.n_envs * self.n_agents

        def _reshape(t: torch.Tensor) -> torch.Tensor:
            # (T, n_envs, n_agents, *) → (T, B, *)
            return t.reshape(T, B, *t.shape[3:])

        return SequenceBatch(
            obs=_reshape(self.obs),
            action_mask=_reshape(self.action_masks),
            actions=_reshape(self.actions),
            old_logprobs=_reshape(self.logprobs),
            values=_reshape(self.values),
            advantages=_reshape(advantages),
            returns=_reshape(returns),
            active=_reshape(self.active),
            life_reset=_reshape(self.life_reset),
            init_h=self.init_h,
            init_c=self.init_c,
        )

    def reset(self) -> None:
        self.ptr = 0


# ---------------------------------------------------------------------------
# Recurrent PPO update
# ---------------------------------------------------------------------------

@dataclass
class RecurrentPPOStats:
    pg_loss: float = 0.0
    vf_loss: float = 0.0
    entropy: float = 0.0
    approx_kl: float = 0.0
    clip_frac: float = 0.0
    n_updates: int = 0
    n_skipped: int = 0


def recurrent_ppo_update(
    policy: RecurrentSharedActorCritic,
    optimizer: torch.optim.Optimizer,
    seq_batch: SequenceBatch,
    clip_coef: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    n_epochs: int = 4,
    n_minibatches: int = 4,
    max_grad_norm: float = 0.5,
    normalize_adv: bool = True,
    rng: torch.Generator | None = None,
) -> RecurrentPPOStats:
    """PPO clip update minibatched over the (env, slot) trajectory dimension.

    All elementwise PPO terms are masked by ``active`` and averaged over the
    active count. If the entire batch is inactive, the call is a no-op.
    """
    stats = RecurrentPPOStats()
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
        # Re-zero inactive positions; they should not contribute to PPO loss.
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
            mb_mask = seq_batch.action_mask[:, mb_idx]
            mb_actions = seq_batch.actions[:, mb_idx]
            mb_old_logprobs = seq_batch.old_logprobs[:, mb_idx]
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

            new_logprobs, entropy, new_values = policy.evaluate_sequence(
                mb_obs, mb_mask, mb_actions, (mb_init_h, mb_init_c), mb_life_reset
            )

            log_ratio = new_logprobs - mb_old_logprobs
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

            ent_loss = (entropy * mb_active).sum() / mb_active_count
            loss = pg_loss + vf_coef * vf_loss - ent_coef * ent_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                stats.pg_loss += float(pg_loss)
                stats.vf_loss += float(vf_loss)
                stats.entropy += float(ent_loss)
                kl = ((-log_ratio) * mb_active).sum() / mb_active_count
                stats.approx_kl += float(kl)
                clipped_mask = ((ratio - 1.0).abs() > clip_coef).float() * mb_active
                stats.clip_frac += float(clipped_mask.sum() / mb_active_count)
                stats.n_updates += 1

    if stats.n_updates > 0:
        stats.pg_loss /= stats.n_updates
        stats.vf_loss /= stats.n_updates
        stats.entropy /= stats.n_updates
        stats.approx_kl /= stats.n_updates
        stats.clip_frac /= stats.n_updates
    return stats
