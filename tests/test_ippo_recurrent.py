"""P4 — recurrent IPPO (LSTM) tests.

Coverage:
    A. RecurrentSharedActorCritic shapes and masked categorical handling.
    B. Hidden state reset via life_reset matches a manual unrolled forward.
    C. RecurrentRolloutBuffer stores life_reset and init_hidden correctly.
    D. SequenceBatch reshape preserves data ordering across (env, agent).
    E. recurrent_ppo_update is a no-op when nothing is active; produces finite
       losses and parameter gradients when there's signal.
    F. value_only does NOT mutate the carried hidden state (bootstrap safety).
    G. evaluate_sequence reset semantics: life_reset[t]=True zeroes hidden
       BEFORE consuming obs[t], not after.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from murimsim.rl.ippo_recurrent import (
    RecurrentRolloutBuffer,
    RecurrentSharedActorCritic,
    recurrent_ppo_update,
)


OBS_DIM = 8
N_ACTIONS = 5
H = 16


def _make_policy(seed: int = 0) -> RecurrentSharedActorCritic:
    torch.manual_seed(seed)
    return RecurrentSharedActorCritic(
        obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden_dim=H, pre_lstm_dim=H
    )


# ───────────────────────── A. shape / API ─────────────────────────

def test_initial_hidden_shape() -> None:
    p = _make_policy()
    h, c = p.initial_hidden(batch_size=4)
    assert h.shape == (1, 4, H)
    assert c.shape == (1, 4, H)
    assert torch.equal(h, torch.zeros_like(h))


def test_act_returns_correct_shapes_and_advances_hidden() -> None:
    p = _make_policy()
    obs = torch.randn(3, OBS_DIM)
    mask = torch.ones(3, N_ACTIONS, dtype=torch.bool)
    h, c = p.initial_hidden(3)
    a, lp, v, (nh, nc) = p.act(obs, mask, (h, c))
    assert a.shape == (3,)
    assert lp.shape == (3,)
    assert v.shape == (3,)
    assert nh.shape == (1, 3, H)
    assert nc.shape == (1, 3, H)
    # Hidden should have advanced from zeros.
    assert not torch.equal(nh, h)


def test_act_respects_action_mask() -> None:
    p = _make_policy()
    obs = torch.randn(8, OBS_DIM)
    mask = torch.zeros(8, N_ACTIONS, dtype=torch.bool)
    mask[:, 2] = True  # only action 2 is legal
    h, c = p.initial_hidden(8)
    a, _, _, _ = p.act(obs, mask, (h, c), deterministic=False)
    assert torch.all(a == 2)


# ─────── B. evaluate_sequence == manual unroll with reset ───────

def test_evaluate_sequence_matches_manual_unroll_no_reset() -> None:
    """Without any life_reset, evaluate_sequence and step-by-step act must
    produce the same values, modulo the actor sampling step (we feed actions)."""
    p = _make_policy(seed=42)
    T, B = 5, 2
    obs_seq = torch.randn(T, B, OBS_DIM)
    mask_seq = torch.ones(T, B, N_ACTIONS, dtype=torch.bool)
    actions = torch.randint(0, N_ACTIONS, (T, B))
    life_reset = torch.zeros(T, B, dtype=torch.bool)

    h0, c0 = p.initial_hidden(B)
    lp_seq, ent_seq, val_seq = p.evaluate_sequence(
        obs_seq, mask_seq, actions, (h0, c0), life_reset
    )

    # Manual: walk T steps re-using act() with deterministic=True so we can
    # compare values exactly. We feed the GIVEN actions into log_prob.
    h, c = p.initial_hidden(B)
    manual_values = []
    manual_lps = []
    for t in range(T):
        x = p.pre_lstm(obs_seq[t]).unsqueeze(0)
        out, (h, c) = p.lstm(x, (h, c))
        feat = out.squeeze(0)
        logits = p.actor_head(feat)
        value = p.critic_head(feat).squeeze(-1)
        from murimsim.rl.ippo import masked_categorical
        dist = masked_categorical(logits, mask_seq[t])
        manual_lps.append(dist.log_prob(actions[t]))
        manual_values.append(value)
    manual_values = torch.stack(manual_values, dim=0)
    manual_lps = torch.stack(manual_lps, dim=0)

    torch.testing.assert_close(val_seq, manual_values, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(lp_seq, manual_lps, atol=1e-6, rtol=1e-6)


def test_evaluate_sequence_reset_zeros_hidden_before_step() -> None:
    """life_reset[t]=True for batch slice b must produce identical output to
    starting a fresh sequence from t with zeroed hidden for that slice."""
    p = _make_policy(seed=7)
    T, B = 4, 2
    obs_seq = torch.randn(T, B, OBS_DIM)
    mask_seq = torch.ones(T, B, N_ACTIONS, dtype=torch.bool)
    actions = torch.randint(0, N_ACTIONS, (T, B))

    # Reset slot 0 at step 2.
    life_reset = torch.zeros(T, B, dtype=torch.bool)
    life_reset[2, 0] = True

    h0, c0 = p.initial_hidden(B)
    h0 = h0 + 0.5  # non-zero start to make the reset visible
    c0 = c0 + 0.5
    _, _, val_a = p.evaluate_sequence(obs_seq, mask_seq, actions, (h0, c0), life_reset)

    # Manual reference: run slot 0 alone with hidden zeroed at step 2.
    # Only compare slot 0's values from step 2 onward to a fresh-from-step-2 run.
    h_ref, c_ref = p.initial_hidden(1)  # zeros — matches the reset
    obs_slot0 = obs_seq[2:, 0:1, :]
    mask_slot0 = mask_seq[2:, 0:1, :]
    actions_slot0 = actions[2:, 0:1]
    no_reset = torch.zeros(T - 2, 1, dtype=torch.bool)
    _, _, val_ref = p.evaluate_sequence(
        obs_slot0, mask_slot0, actions_slot0, (h_ref, c_ref), no_reset
    )
    torch.testing.assert_close(val_a[2:, 0:1], val_ref, atol=1e-6, rtol=1e-6)


def test_value_only_does_not_mutate_hidden() -> None:
    """value_only is for GAE bootstrap — must NOT mutate the carried hidden."""
    p = _make_policy(seed=3)
    obs = torch.randn(4, OBS_DIM)
    h, c = p.initial_hidden(4)
    h = h + 0.3
    c = c + 0.3
    h_before = h.clone()
    c_before = c.clone()
    v = p.value_only(obs, (h, c))
    assert v.shape == (4,)
    assert torch.equal(h, h_before)
    assert torch.equal(c, c_before)


# ───────────────── C/D. RecurrentRolloutBuffer ─────────────────

def test_buffer_stores_life_reset_and_init_hidden() -> None:
    T, n_envs, n_agents = 4, 2, 3
    buf = RecurrentRolloutBuffer(T, n_envs, n_agents, OBS_DIM, N_ACTIONS, hidden_dim=H)

    h0 = torch.randn(1, n_envs * n_agents, H)
    c0 = torch.randn(1, n_envs * n_agents, H)
    buf.set_initial_hidden(h0, c0)
    torch.testing.assert_close(buf.init_h, h0)
    torch.testing.assert_close(buf.init_c, c0)

    for t in range(T):
        buf.add(
            obs=np.zeros((n_envs, n_agents, OBS_DIM), dtype=np.float32),
            action_mask=np.ones((n_envs, n_agents, N_ACTIONS), dtype=bool),
            action=torch.zeros(n_envs, n_agents, dtype=torch.long),
            logprob=torch.zeros(n_envs, n_agents),
            value=torch.zeros(n_envs, n_agents),
            reward=np.zeros((n_envs, n_agents), dtype=np.float32),
            done=np.zeros((n_envs, n_agents), dtype=bool),
            active=np.ones((n_envs, n_agents), dtype=bool),
            life_reset=np.array([[t == 1 and (e == 0 and a == 1)
                                   for a in range(n_agents)] for e in range(n_envs)]),
        )
    assert buf.life_reset[1, 0, 1].item() is True
    assert buf.life_reset[0, 0, 0].item() is False
    assert buf.life_reset[2, 0, 1].item() is False


def test_to_sequence_batch_preserves_ordering() -> None:
    """Reshape (T, n_envs, n_agents, *) → (T, n_envs*n_agents, *) must use
    row-major ordering so that buffer slot (e, a) maps to B index e*n_agents+a."""
    T, n_envs, n_agents = 3, 2, 2
    buf = RecurrentRolloutBuffer(T, n_envs, n_agents, OBS_DIM, N_ACTIONS, hidden_dim=H)
    buf.set_initial_hidden(
        torch.zeros(1, n_envs * n_agents, H),
        torch.zeros(1, n_envs * n_agents, H),
    )
    for t in range(T):
        # encode (e, a, t) into obs so we can verify ordering
        obs = np.zeros((n_envs, n_agents, OBS_DIM), dtype=np.float32)
        for e in range(n_envs):
            for a in range(n_agents):
                obs[e, a, 0] = float(e * 100 + a * 10 + t)
        buf.add(
            obs=obs,
            action_mask=np.ones((n_envs, n_agents, N_ACTIONS), dtype=bool),
            action=torch.zeros(n_envs, n_agents, dtype=torch.long),
            logprob=torch.zeros(n_envs, n_agents),
            value=torch.zeros(n_envs, n_agents),
            reward=np.zeros((n_envs, n_agents), dtype=np.float32),
            done=np.zeros((n_envs, n_agents), dtype=bool),
            active=np.ones((n_envs, n_agents), dtype=bool),
            life_reset=np.zeros((n_envs, n_agents), dtype=bool),
        )

    adv = torch.zeros(T, n_envs, n_agents)
    ret = torch.zeros(T, n_envs, n_agents)
    sb = buf.to_sequence_batch(adv, ret)
    assert sb.obs.shape == (T, n_envs * n_agents, OBS_DIM)
    # For (e=1, a=0, t=2), encoding = 1*100+0*10+2 = 102, mapping to B=1*2+0=2.
    assert float(sb.obs[2, 2, 0]) == 102.0
    assert float(sb.obs[0, 0, 0]) == 0.0  # (e=0, a=0, t=0)
    assert float(sb.obs[2, 3, 0]) == 112.0  # (e=1, a=1, t=2)


# ───────────────── E. recurrent_ppo_update ─────────────────

def _make_filled_buffer(
    T: int = 8, n_envs: int = 2, n_agents: int = 2, all_active: bool = True, seed: int = 0
) -> tuple[RecurrentRolloutBuffer, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    buf = RecurrentRolloutBuffer(T, n_envs, n_agents, OBS_DIM, N_ACTIONS, hidden_dim=H)
    buf.set_initial_hidden(
        torch.zeros(1, n_envs * n_agents, H),
        torch.zeros(1, n_envs * n_agents, H),
    )
    active = np.ones((n_envs, n_agents), dtype=bool)
    if not all_active:
        active[:] = False
    for t in range(T):
        buf.add(
            obs=np.random.randn(n_envs, n_agents, OBS_DIM).astype(np.float32),
            action_mask=np.ones((n_envs, n_agents, N_ACTIONS), dtype=bool),
            action=torch.randint(0, N_ACTIONS, (n_envs, n_agents)),
            logprob=torch.zeros(n_envs, n_agents),
            value=torch.zeros(n_envs, n_agents),
            reward=np.random.randn(n_envs, n_agents).astype(np.float32) * 0.1,
            done=np.zeros((n_envs, n_agents), dtype=bool),
            active=active,
            life_reset=np.zeros((n_envs, n_agents), dtype=bool),
        )
    last_value = torch.zeros(n_envs, n_agents)
    last_active = torch.as_tensor(active, dtype=torch.bool)
    return buf, last_value, last_active


def test_recurrent_ppo_update_no_op_on_inactive_batch() -> None:
    p = _make_policy()
    opt = torch.optim.Adam(p.parameters(), lr=1e-3)
    buf, last_value, last_active = _make_filled_buffer(all_active=False)
    adv, ret = buf.compute_gae(last_value, last_active)
    sb = buf.to_sequence_batch(adv, ret)
    stats = recurrent_ppo_update(p, opt, sb, n_epochs=1, n_minibatches=2)
    # All mb were skipped since mb_active_count == 0.
    assert stats.n_updates == 0
    assert stats.n_skipped >= 1


def test_recurrent_ppo_update_finite_losses() -> None:
    p = _make_policy(seed=1)
    opt = torch.optim.Adam(p.parameters(), lr=1e-3)
    buf, last_value, last_active = _make_filled_buffer(seed=1)
    adv, ret = buf.compute_gae(last_value, last_active)
    sb = buf.to_sequence_batch(adv, ret)
    stats = recurrent_ppo_update(p, opt, sb, n_epochs=2, n_minibatches=2)
    assert stats.n_updates > 0
    for v in (stats.pg_loss, stats.vf_loss, stats.entropy, stats.approx_kl, stats.clip_frac):
        assert np.isfinite(v), f"non-finite stat: {v}"


def test_recurrent_ppo_update_propagates_gradients_to_lstm() -> None:
    """LSTM weights must receive non-zero gradients from sequence loss."""
    p = _make_policy(seed=2)
    opt = torch.optim.Adam(p.parameters(), lr=1e-3)
    buf, last_value, last_active = _make_filled_buffer(seed=2)
    adv, ret = buf.compute_gae(last_value, last_active)
    sb = buf.to_sequence_batch(adv, ret)
    # Snapshot the LSTM weight before; check it changed after update.
    w_before = p.lstm.weight_ih_l0.detach().clone()
    recurrent_ppo_update(p, opt, sb, n_epochs=2, n_minibatches=2)
    w_after = p.lstm.weight_ih_l0.detach()
    assert not torch.equal(w_before, w_after), "LSTM weights did not change"
