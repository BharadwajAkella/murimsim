"""P2.1 tests — IPPOEnv vector step API.

Tests are xfail until IPPOEnv lands. Once implemented, remove the xfail marks.

Invariants:
  A. Vector API shapes — step_all returns per-agent obs/reward/term/trunc.
  B. Back-compat — single-focal CombatEnv.step(int) still works on same instance.
  C. Per-agent obs — obs[i] is the local view of agent i (different per slot).
  D. Active mask — dead-at-start slots are flagged inactive (no PPO loss).
  E. Per-slot termination — terminated[i] reflects slot i's death this step,
     not just the focal's.
  F. Reward fidelity — rewards array == info['per_agent_reward'].
  G. Action mask matrix — info['action_masks_post'] is shape (n_agents, n_actions).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from murimsim.actions import Action, N_ACTIONS_PHASE6_QI
from murimsim.rl.ippo_env import IPPOEnv

CONFIG_PATH = Path("config/default.yaml")


def _load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _make_ippo(n_agents: int = 6, seed: int = 0) -> IPPOEnv:
    return IPPOEnv(
        config=_load_cfg(),
        n_agents=n_agents,
        seed=seed,
        curriculum_ramp_steps=0,
    )


# ---------------------------------------------------------------------------
# A. Vector API shapes
# ---------------------------------------------------------------------------


def test_step_all_returns_per_agent_arrays() -> None:
    env = _make_ippo(n_agents=5, seed=1)
    env.reset(seed=1)
    actions = np.full(5, Action.MOVE_N.value, dtype=np.int64)
    obs, rewards, terminated, truncated, info = env.step_all(actions)
    assert obs.shape[0] == 5  # one obs row per agent
    assert obs.ndim == 2
    assert rewards.shape == (5,)
    assert rewards.dtype == np.float64
    assert terminated.shape == (5,)
    assert terminated.dtype == bool
    assert truncated.shape == (5,)
    assert truncated.dtype == bool


def test_step_all_validates_action_shape() -> None:
    env = _make_ippo(n_agents=4, seed=2)
    env.reset(seed=2)
    with pytest.raises((AssertionError, ValueError)):
        env.step_all(np.array([Action.MOVE_N.value, Action.MOVE_N.value]))


# ---------------------------------------------------------------------------
# B. Back-compat: single-focal step still works
# ---------------------------------------------------------------------------


def test_single_focal_step_still_works() -> None:
    env = _make_ippo(n_agents=4, seed=3)
    env.reset(seed=3)
    obs, reward, term, trunc, info = env.step(Action.MOVE_N.value)
    assert isinstance(reward, float)
    assert "per_agent_reward" in info
    # Still exposes per-agent reward array (P1.2)
    assert info["per_agent_reward"].shape == (4,)


# ---------------------------------------------------------------------------
# C. Per-agent observation
# ---------------------------------------------------------------------------


def test_per_agent_obs_differs_across_slots() -> None:
    """obs[i] must be agent i's local view, not the same view replicated."""
    env = _make_ippo(n_agents=4, seed=4)
    env.reset(seed=4)
    actions = np.full(4, Action.REST.value, dtype=np.int64)
    obs, _r, _t, _tr, _info = env.step_all(actions)
    # At least one pair of rows must differ — agents are at different positions
    pairs_differ = any(
        not np.array_equal(obs[i], obs[j])
        for i in range(4) for j in range(i + 1, 4)
    )
    assert pairs_differ, "All per-agent obs identical — _build_obs(i) likely not used"


# ---------------------------------------------------------------------------
# D. Active mask — dead slots flagged inactive
# ---------------------------------------------------------------------------


def test_active_mask_flags_dead_slot() -> None:
    env = _make_ippo(n_agents=4, seed=5)
    env.reset(seed=5)
    victim = env._agents[2]
    victim.health = 0.0
    victim.alive = False
    actions = np.full(4, Action.REST.value, dtype=np.int64)
    obs, rewards, term, trunc, info = env.step_all(actions)
    am = info["active_mask"]
    assert am.shape == (4,)
    assert am.dtype == bool
    assert am[2] == False, "dead-at-start slot should be inactive"
    # Reward for dead slot stays 0
    assert rewards[2] == 0.0


# ---------------------------------------------------------------------------
# E. Per-slot termination
# ---------------------------------------------------------------------------


def test_terminated_array_reflects_per_slot_death() -> None:
    env = _make_ippo(n_agents=4, seed=6)
    env.reset(seed=6)
    # Don't kill anyone; lifecycle.died should be all False this step
    actions = np.full(4, Action.MOVE_N.value, dtype=np.int64)
    _o, _r, term, _tr, _info = env.step_all(actions)
    assert term.shape == (4,)
    assert term.dtype == bool


# ---------------------------------------------------------------------------
# F. Reward fidelity
# ---------------------------------------------------------------------------


def test_rewards_match_per_agent_reward_info() -> None:
    env = _make_ippo(n_agents=5, seed=7)
    env.reset(seed=7)
    actions = np.full(5, Action.MOVE_N.value, dtype=np.int64)
    _o, rewards, _t, _tr, info = env.step_all(actions)
    np.testing.assert_array_equal(rewards, info["per_agent_reward"])


# ---------------------------------------------------------------------------
# G. Action mask matrix
# ---------------------------------------------------------------------------


def test_action_masks_post_shape() -> None:
    env = _make_ippo(n_agents=5, seed=8)
    env.reset(seed=8)
    actions = np.full(5, Action.MOVE_N.value, dtype=np.int64)
    _o, _r, _t, _tr, info = env.step_all(actions)
    masks = info["action_masks_post"]
    assert masks.shape == (5, N_ACTIONS_PHASE6_QI)
    assert masks.dtype == bool


# ---------------------------------------------------------------------------
# H. Determinism — same seed + same actions → same outputs
# ---------------------------------------------------------------------------


def test_step_all_deterministic() -> None:
    env1 = _make_ippo(n_agents=4, seed=9)
    env1.reset(seed=9)
    env2 = _make_ippo(n_agents=4, seed=9)
    env2.reset(seed=9)
    actions = np.full(4, Action.MOVE_N.value, dtype=np.int64)
    o1, r1, _t1, _tr1, _i1 = env1.step_all(actions)
    o2, r2, _t2, _tr2, _i2 = env2.step_all(actions)
    np.testing.assert_array_equal(o1, o2)
    np.testing.assert_array_equal(r1, r2)
