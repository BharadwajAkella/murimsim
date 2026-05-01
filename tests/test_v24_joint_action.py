"""tests/test_v24_joint_action.py — v24 split body/social action heads.

Covers:
    * BodyAction / SocialAction enum integrity + lookup tables.
    * action_masks_body / action_masks_social derivations.
    * step_all_joint reproduces step_all body resolution when social=NOOP.
    * step_all_joint with social=COLLABORATE forms a group + grants reward.
    * Joint policy round-trip: act → buffer → evaluate produces consistent
      log-probs and finite gradients.
    * Joint recurrent policy: act → buffer → evaluate_sequence consistent.
"""
from __future__ import annotations

import numpy as np
import torch

from murimsim.actions import (
    BODY_TO_LEGACY,
    LEGACY_TO_BODY,
    Action,
    BodyAction,
    N_BODY_ACTIONS,
    N_SOCIAL_ACTIONS,
    SocialAction,
)
from pathlib import Path
import yaml

from murimsim.rl.ippo_env import IPPOEnv
from murimsim.rl.ippo_joint import (
    JointRolloutBuffer,
    JointSharedActorCritic,
    joint_ppo_update,
)
from murimsim.rl.ippo_joint_recurrent import (
    JointRecurrentRolloutBuffer,
    JointRecurrentSharedActorCritic,
    joint_recurrent_ppo_update,
)


CONFIG_PATH = Path("config/default.yaml")


def _load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Enum + table integrity
# ---------------------------------------------------------------------------

def test_body_social_enum_sizes():
    assert N_BODY_ACTIONS == 16
    assert N_SOCIAL_ACTIONS == 2
    assert int(SocialAction.NOOP) == 0
    assert int(SocialAction.COLLABORATE) == 1


def test_lookup_tables_round_trip():
    assert len(BODY_TO_LEGACY) == N_BODY_ACTIONS
    for body_idx, legacy_idx in BODY_TO_LEGACY.items():
        assert LEGACY_TO_BODY[legacy_idx] == body_idx
    # COLLABORATE must NOT exist in body lane.
    assert Action.COLLABORATE.value not in LEGACY_TO_BODY


# ---------------------------------------------------------------------------
# Mask derivations
# ---------------------------------------------------------------------------

def _make_env(seed=0):
    env = IPPOEnv(config=_load_cfg(), n_agents=4, seed=seed, curriculum_ramp_steps=0)
    env.reset_all(seed=seed)
    return env


def test_action_masks_body_drops_collaborate():
    env = _make_env()
    for i in range(env._n_agents):
        legacy = env.action_masks(i)
        body = env.action_masks_body(i)
        assert body.shape == (N_BODY_ACTIONS,)
        for body_idx, legacy_idx in BODY_TO_LEGACY.items():
            assert bool(body[body_idx]) == bool(legacy[legacy_idx])


def test_action_masks_social_shape_and_noop_always_true():
    env = _make_env()
    for i in range(env._n_agents):
        social = env.action_masks_social(i)
        assert social.shape == (N_SOCIAL_ACTIONS,)
        assert bool(social[SocialAction.NOOP]) is True


# ---------------------------------------------------------------------------
# step_all_joint vs step_all behavior
# ---------------------------------------------------------------------------

def test_step_all_joint_noop_matches_legacy_body():
    """With social=NOOP everywhere, joint step matches a legacy step using
    the same body actions translated to legacy ints."""
    env_a = _make_env(seed=42)
    env_b = _make_env(seed=42)

    # All REST (body idx == legacy REST idx).
    body_actions = np.full(env_a._n_agents, BodyAction.REST.value, dtype=np.int64)
    social_actions = np.zeros(env_a._n_agents, dtype=np.int64)
    legacy_actions = np.array([BODY_TO_LEGACY[int(b)] for b in body_actions])

    obs_a, rew_a, term_a, trunc_a, info_a = env_a.step_all_joint(body_actions, social_actions)
    obs_b, rew_b, term_b, trunc_b, info_b = env_b.step_all(legacy_actions)

    np.testing.assert_array_equal(obs_a, obs_b)
    np.testing.assert_array_equal(rew_a, rew_b)
    np.testing.assert_array_equal(term_a, term_b)
    np.testing.assert_array_equal(trunc_a, trunc_b)
    assert "action_masks_body_post" in info_a
    assert "action_masks_social_post" in info_a
    assert info_a["action_masks_body_post"].shape == (env_a._n_agents, N_BODY_ACTIONS)
    assert info_a["action_masks_social_post"].shape == (env_a._n_agents, N_SOCIAL_ACTIONS)


def test_step_all_joint_runs_many_steps():
    """Sanity: 64 random joint steps, no exception, rewards finite."""
    env = _make_env(seed=7)
    rng = np.random.default_rng(123)
    n = env._n_agents
    for _ in range(64):
        body_mask = np.stack([env.action_masks_body(i) for i in range(n)])
        social_mask = np.stack([env.action_masks_social(i) for i in range(n)])
        body = np.array(
            [int(rng.choice(np.where(body_mask[i])[0])) for i in range(n)],
            dtype=np.int64,
        )
        social = np.array(
            [int(rng.choice(np.where(social_mask[i])[0])) for i in range(n)],
            dtype=np.int64,
        )
        _obs, rew, _term, trunc, _info = env.step_all_joint(body, social)
        assert np.all(np.isfinite(rew))
        if bool(trunc.any()):
            env.reset_all(seed=7)


# ---------------------------------------------------------------------------
# Joint feedforward policy + buffer + update
# ---------------------------------------------------------------------------

def test_joint_policy_act_evaluate_consistency():
    torch.manual_seed(0)
    env = _make_env(seed=11)
    obs_dim = env.observation_space.shape[0]
    policy = JointSharedActorCritic(obs_dim=obs_dim)

    n = env._n_agents
    obs = torch.from_numpy(np.stack([env._build_obs(i) for i in range(n)])).float()
    body_mask = torch.from_numpy(np.stack([env.action_masks_body(i) for i in range(n)])).bool()
    social_mask = torch.from_numpy(np.stack([env.action_masks_social(i) for i in range(n)])).bool()

    body_a, social_a, body_lp, social_lp, value = policy.act(obs, body_mask, social_mask)
    nb_lp, ns_lp, b_ent, s_ent, value2 = policy.evaluate(
        obs, body_mask, social_mask, body_a, social_a
    )
    torch.testing.assert_close(body_lp, nb_lp)
    torch.testing.assert_close(social_lp, ns_lp)
    torch.testing.assert_close(value, value2)
    assert torch.isfinite(b_ent).all()
    assert torch.isfinite(s_ent).all()


def test_joint_buffer_and_update_finite_gradients():
    """Smoke: fill a small joint buffer, run joint_ppo_update, expect finite
    losses and non-zero updates."""
    torch.manual_seed(1)
    env = _make_env(seed=2)
    n = env._n_agents
    obs_dim = env.observation_space.shape[0]
    policy = JointSharedActorCritic(obs_dim=obs_dim)
    optim = torch.optim.Adam(policy.parameters(), lr=3e-4)
    T = 16
    buf = JointRolloutBuffer(rollout_length=T, n_envs=1, n_agents=n, obs_dim=obs_dim)

    obs_arr, info = env.reset_all(seed=2)
    body_mask = info["action_masks_body"]
    social_mask = info["action_masks_social"]
    active = info["active_mask"]

    for _ in range(T):
        obs_t = torch.from_numpy(obs_arr).float()
        bm_t = torch.from_numpy(body_mask).bool()
        sm_t = torch.from_numpy(social_mask).bool()
        with torch.no_grad():
            ba, sa, blp, slp, val = policy.act(obs_t, bm_t, sm_t)
        body_np = ba.numpy().astype(np.int64)
        social_np = sa.numpy().astype(np.int64)
        next_obs, rew, term, trunc, step_info = env.step_all_joint(body_np, social_np)
        done = term | trunc
        buf.add(
            obs=obs_arr.reshape(1, n, obs_dim),
            body_mask=body_mask.reshape(1, n, N_BODY_ACTIONS),
            social_mask=social_mask.reshape(1, n, N_SOCIAL_ACTIONS),
            body_action=ba.reshape(1, n),
            social_action=sa.reshape(1, n),
            body_logprob=blp.reshape(1, n),
            social_logprob=slp.reshape(1, n),
            value=val.reshape(1, n),
            reward=rew.reshape(1, n),
            done=done.reshape(1, n),
            active=active.reshape(1, n),
        )
        obs_arr = next_obs
        body_mask = step_info["action_masks_body_post"]
        social_mask = step_info["action_masks_social_post"]
        active = step_info["active_mask"]

    with torch.no_grad():
        last_obs = torch.from_numpy(obs_arr).float()
        last_bm = torch.from_numpy(body_mask).bool()
        last_sm = torch.from_numpy(social_mask).bool()
        _, _, _, _, last_val = policy.act(last_obs, last_bm, last_sm)
    last_active = torch.from_numpy(active).bool().reshape(1, n)
    adv, ret = buf.compute_gae(
        last_value=last_val.reshape(1, n),
        last_active=last_active,
    )
    batch = buf.flatten_active(adv, ret)
    stats = joint_ppo_update(policy, optim, batch, n_epochs=2, n_minibatches=2)
    assert stats.n_updates > 0
    assert np.isfinite(stats.pg_loss)
    assert np.isfinite(stats.vf_loss)
    assert np.isfinite(stats.body_entropy)
    assert np.isfinite(stats.social_entropy)


# ---------------------------------------------------------------------------
# Joint recurrent policy round-trip
# ---------------------------------------------------------------------------

def test_joint_recurrent_act_and_evaluate_consistency():
    torch.manual_seed(2)
    env = _make_env(seed=3)
    n = env._n_agents
    obs_dim = env.observation_space.shape[0]
    policy = JointRecurrentSharedActorCritic(obs_dim=obs_dim, hidden_dim=32, pre_lstm_dim=32)
    h, c = policy.initial_hidden(batch_size=n)

    obs_arr, info = env.reset_all(seed=3)
    obs_t = torch.from_numpy(obs_arr).float()
    bm_t = torch.from_numpy(info["action_masks_body"]).bool()
    sm_t = torch.from_numpy(info["action_masks_social"]).bool()

    ba, sa, blp, slp, _val, (h2, c2) = policy.act(obs_t, bm_t, sm_t, (h, c))

    # Replay through evaluate_sequence with T=1 — should reproduce log-probs.
    obs_seq = obs_t.unsqueeze(0)
    bm_seq = bm_t.unsqueeze(0)
    sm_seq = sm_t.unsqueeze(0)
    ba_seq = ba.unsqueeze(0)
    sa_seq = sa.unsqueeze(0)
    life_reset = torch.zeros(1, n, dtype=torch.bool)
    nb_lp, ns_lp, b_ent, s_ent, _vals = policy.evaluate_sequence(
        obs_seq, bm_seq, sm_seq, ba_seq, sa_seq, (h, c), life_reset
    )
    torch.testing.assert_close(nb_lp[0], blp)
    torch.testing.assert_close(ns_lp[0], slp)
    assert torch.isfinite(b_ent).all()
    assert torch.isfinite(s_ent).all()
    assert h2.shape == h.shape


def test_joint_recurrent_update_smoke():
    torch.manual_seed(3)
    env = _make_env(seed=5)
    n = env._n_agents
    obs_dim = env.observation_space.shape[0]
    H = 32
    policy = JointRecurrentSharedActorCritic(obs_dim=obs_dim, hidden_dim=H, pre_lstm_dim=32)
    optim = torch.optim.Adam(policy.parameters(), lr=3e-4)
    T = 8
    buf = JointRecurrentRolloutBuffer(
        rollout_length=T, n_envs=1, n_agents=n, obs_dim=obs_dim, hidden_dim=H
    )
    h, c = policy.initial_hidden(batch_size=n)
    buf.set_initial_hidden(h, c)

    obs_arr, info = env.reset_all(seed=5)
    body_mask = info["action_masks_body"]
    social_mask = info["action_masks_social"]
    active = info["active_mask"]
    pending_reset = np.zeros(n, dtype=bool)

    for _ in range(T):
        obs_t = torch.from_numpy(obs_arr).float()
        bm_t = torch.from_numpy(body_mask).bool()
        sm_t = torch.from_numpy(social_mask).bool()
        # Apply pending reset BEFORE act, mirroring trainer semantics.
        if pending_reset.any():
            mask_keep = (~torch.from_numpy(pending_reset).bool()).view(1, n, 1).float()
            h = h * mask_keep
            c = c * mask_keep
        with torch.no_grad():
            ba, sa, blp, slp, val, (h, c) = policy.act(obs_t, bm_t, sm_t, (h, c))
        body_np = ba.numpy().astype(np.int64)
        social_np = sa.numpy().astype(np.int64)
        next_obs, rew, term, trunc, step_info = env.step_all_joint(body_np, social_np)
        done = term | trunc
        buf.add(
            obs=obs_arr.reshape(1, n, obs_dim),
            body_mask=body_mask.reshape(1, n, N_BODY_ACTIONS),
            social_mask=social_mask.reshape(1, n, N_SOCIAL_ACTIONS),
            body_action=ba.reshape(1, n),
            social_action=sa.reshape(1, n),
            body_logprob=blp.reshape(1, n),
            social_logprob=slp.reshape(1, n),
            value=val.reshape(1, n),
            reward=rew.reshape(1, n),
            done=done.reshape(1, n),
            active=active.reshape(1, n),
            life_reset=pending_reset.reshape(1, n),
        )
        # Compute next-step pending_reset from lifecycle 'born' events.
        lifecycle = step_info.get("lifecycle", [])
        pending_reset = np.array(
            [bool(lc.get("born", False)) if isinstance(lc, dict) else False for lc in lifecycle],
            dtype=bool,
        )
        obs_arr = next_obs
        body_mask = step_info["action_masks_body_post"]
        social_mask = step_info["action_masks_social_post"]
        active = step_info["active_mask"]

    with torch.no_grad():
        last_obs = torch.from_numpy(obs_arr).float()
        last_val = policy.value_only(last_obs, (h, c))
    last_active = torch.from_numpy(active).bool().reshape(1, n)
    adv, ret = buf.compute_gae(last_val.reshape(1, n), last_active)
    seq_batch = buf.to_sequence_batch(adv, ret)
    stats = joint_recurrent_ppo_update(policy, optim, seq_batch, n_epochs=2, n_minibatches=2)
    assert stats.n_updates > 0
    assert np.isfinite(stats.pg_loss)
    assert np.isfinite(stats.vf_loss)
    assert np.isfinite(stats.body_entropy)
    assert np.isfinite(stats.social_entropy)


# ---------------------------------------------------------------------------
# Trainer + eval round-trip smoke (subprocess)
# ---------------------------------------------------------------------------

import os
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_train_ippo_joint_smoke(tmp_path):
    """v24 FF trainer end-to-end + eval auto-detects joint ckpt."""
    env = {**os.environ, "WANDB_DISABLED": "1", "PYTHONUNBUFFERED": "1"}
    out = subprocess.run(
        [
            sys.executable, "-m", "scripts.train_ippo_joint",
            "--total-steps", "256", "--rollout-length", "16",
            "--n-envs", "2", "--n-agents", "4", "--no-wandb",
            "--checkpoint-dir", str(tmp_path), "--checkpoint-interval", "1",
        ],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=180,
    )
    assert out.returncode == 0, f"trainer failed:\n{out.stderr}"
    ckpts = list(tmp_path.glob("*.pt"))
    assert ckpts, "no checkpoint produced"
    out2 = subprocess.run(
        [
            sys.executable, "-m", "scripts.eval_ippo",
            "--checkpoint", str(ckpts[-1]),
            "--steps", "100", "--n-envs", "2", "--n-agents", "4",
        ],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=180,
    )
    assert out2.returncode == 0, f"eval failed:\n{out2.stderr}"


def test_train_ippo_joint_recurrent_smoke(tmp_path):
    """v24 recurrent trainer end-to-end + eval auto-detects joint ckpt."""
    env = {**os.environ, "WANDB_DISABLED": "1", "PYTHONUNBUFFERED": "1"}
    out = subprocess.run(
        [
            sys.executable, "-m", "scripts.train_ippo_joint_recurrent",
            "--total-steps", "256", "--rollout-length", "16",
            "--n-envs", "2", "--n-agents", "4",
            "--hidden-dim", "32", "--pre-lstm-dim", "32",
            "--no-wandb",
            "--checkpoint-dir", str(tmp_path), "--checkpoint-interval", "1",
        ],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=240,
    )
    assert out.returncode == 0, f"trainer failed:\n{out.stderr}"
    ckpts = list(tmp_path.glob("*.pt"))
    assert ckpts, "no checkpoint produced"
    out2 = subprocess.run(
        [
            sys.executable, "-m", "scripts.eval_ippo",
            "--checkpoint", str(ckpts[-1]),
            "--steps", "100", "--n-envs", "2", "--n-agents", "4",
        ],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=240,
    )
    assert out2.returncode == 0, f"eval failed:\n{out2.stderr}"
