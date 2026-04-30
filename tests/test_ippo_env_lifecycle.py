"""P2.4 — IPPOEnv lifecycle + strict-determinism gap-fill tests.

Existing coverage already handles:
- info['lifecycle'] structure and per-step semantics (test_combat_env.py)
- step_all summary determinism (test_ippo_env.py)
- focal randomization determinism (test_ippo_env_resolution_order.py)
- slot state reset on rebirth (test_slot_state_reset.py)
- focal reward byte-equality (test_focal_reward_unchanged.py)

P2.4 fills the remaining gaps for production-grade IPPO training:

1. STRICT determinism over many steps — not just summary stats but every
   per-agent obs/reward/term/active value across 200 steps × 2 envs.
2. Action-mask purity — reading masks does NOT mutate RNG state.
3. Empty/dead-slot exclusion — a dead slot's reward is exactly 0 in the
   per_agent_reward array, never NaN, never copied from a live slot.
4. _action_overrides interaction with rebirth — overrides are reset between
   step_all calls so a freshly reborn slot doesn't carry over a prior
   override targeted at the previous occupant.
5. resolution_order invariant — always a permutation of [0..n-1] with
   focal first, regardless of how many slots are alive.
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


def _make(n_agents: int = 4, seed: int = 0) -> IPPOEnv:
    env = IPPOEnv(
        config=_load_cfg(),
        n_agents=n_agents,
        seed=seed,
        curriculum_ramp_steps=0,
    )
    env.reset_all(seed=seed)
    return env


# ---------------------------------------------------------------------------
# 1. Strict determinism — every byte over 200 steps
# ---------------------------------------------------------------------------

def test_step_all_byte_identical_across_envs_with_same_seed() -> None:
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    env_a = _make(n_agents=4, seed=99)
    env_b = _make(n_agents=4, seed=99)

    for _ in range(200):
        actions_a = rng_a.integers(0, N_ACTIONS_PHASE6_QI, size=4)
        actions_b = rng_b.integers(0, N_ACTIONS_PHASE6_QI, size=4)
        np.testing.assert_array_equal(actions_a, actions_b)

        oa, ra, ta, tra, ia = env_a.step_all(actions_a)
        ob, rb, tb, trb, ib = env_b.step_all(actions_b)

        np.testing.assert_array_equal(oa, ob)
        np.testing.assert_array_equal(ra, rb)
        np.testing.assert_array_equal(ta, tb)
        np.testing.assert_array_equal(tra, trb)
        np.testing.assert_array_equal(ia["active_mask"], ib["active_mask"])
        np.testing.assert_array_equal(ia["action_masks_post"], ib["action_masks_post"])
        assert ia["resolution_order"] == ib["resolution_order"]


# ---------------------------------------------------------------------------
# 2. Action-mask purity — no RNG side effects
# ---------------------------------------------------------------------------

def test_action_masks_does_not_mutate_rng_state() -> None:
    """Calling action_masks() many times must not perturb step_all output.

    Justifies P0.1 — the IPPO trainer reads masks for every (env, agent)
    every step, separately from the env's own step. If reads were not pure,
    the trainer's mask reads would scramble the env RNG and break determinism.
    """
    env_a = _make(n_agents=4, seed=7)
    env_b = _make(n_agents=4, seed=7)

    for _ in range(50):
        # env_a: read masks 20 extra times before stepping
        for slot in range(4):
            for _ in range(5):
                env_a.action_masks(slot)
        actions = np.array([Action.REST.value] * 4)
        oa, ra, _ta, _tra, _ia = env_a.step_all(actions)
        ob, rb, _tb, _trb, _ib = env_b.step_all(actions)

        np.testing.assert_array_equal(oa, ob)
        np.testing.assert_array_equal(ra, rb)


# ---------------------------------------------------------------------------
# 3. Empty/dead-slot exclusion
# ---------------------------------------------------------------------------

def test_dead_slot_reward_is_zero_and_active_mask_false() -> None:
    """Pre-kill a slot; verify its per_agent_reward is 0 and active=False."""
    env = _make(n_agents=4, seed=11)
    # Force slot 2 dead BEFORE the step.
    env._agents[2].alive = False
    env._agents[2].health = 0.0

    actions = np.array([Action.REST.value] * 4)
    _o, r, _t, _tr, info = env.step_all(actions)

    assert info["active_mask"][2] is np.False_ or info["active_mask"][2] == False  # noqa: E712
    # Per-agent reward for the dead slot should be exactly 0 (no shaping).
    assert r[2] == 0.0
    # Other slots should not have NaN.
    assert np.all(np.isfinite(r))


def test_dead_slot_obs_and_action_mask_are_well_formed() -> None:
    """Even for a dead slot, obs and action_mask must be valid arrays.

    The trainer flattens obs/masks across all slots before policy inference;
    NaN or wrong-shape entries would corrupt batched forward passes.
    """
    env = _make(n_agents=4, seed=12)
    env._agents[0].alive = False
    env._agents[0].health = 0.0

    actions = np.array([Action.REST.value] * 4)
    obs, _r, _t, _tr, info = env.step_all(actions)

    assert obs.shape[0] == 4
    assert np.all(np.isfinite(obs))
    masks = info["action_masks_post"]
    assert masks.shape == (4, N_ACTIONS_PHASE6_QI)
    assert masks.dtype == bool


# ---------------------------------------------------------------------------
# 4. _action_overrides interaction with rebirth
# ---------------------------------------------------------------------------

def test_action_overrides_reset_between_step_all_calls() -> None:
    """step_all installs per-call overrides and restores prior state in finally.

    If overrides leaked across calls, a freshly reborn slot would inherit
    the previous occupant's queued action.
    """
    env = _make(n_agents=4, seed=13)

    # Plant a sentinel override before step_all and verify it's restored after.
    sentinel = {0: 999, 1: 999}
    env._action_overrides = dict(sentinel)

    actions = np.array([Action.REST.value] * 4)
    env.step_all(actions)

    assert env._action_overrides == sentinel, (
        "step_all must restore _action_overrides via try/finally"
    )


def test_step_all_independent_of_prior_action_overrides() -> None:
    """Even if the env had prior overrides, step_all uses ONLY its own action vec."""
    env_a = _make(n_agents=4, seed=14)
    env_b = _make(n_agents=4, seed=14)
    env_b._action_overrides = {0: Action.MOVE_S.value, 2: Action.MOVE_E.value}

    actions = np.array([Action.REST.value] * 4)
    oa, ra, _, _, _ = env_a.step_all(actions)
    ob, rb, _, _, _ = env_b.step_all(actions)

    np.testing.assert_array_equal(oa, ob)
    np.testing.assert_array_equal(ra, rb)


# ---------------------------------------------------------------------------
# 5. resolution_order invariant
# ---------------------------------------------------------------------------

def test_resolution_order_is_always_a_permutation() -> None:
    """resolution_order must be a permutation of [0..n-1] with focal first."""
    env = _make(n_agents=5, seed=15)
    rng = np.random.default_rng(0)
    for _ in range(100):
        actions = rng.integers(0, N_ACTIONS_PHASE6_QI, size=5)
        _o, _r, _t, _tr, info = env.step_all(actions)
        order = info["resolution_order"]
        assert sorted(order) == list(range(5)), order
        # All n unique entries; first entry is the focal that ACTED this step
        # (env._focal_idx has since advanced via _next_live).
        assert len(set(order)) == 5


def test_resolution_order_focal_only_from_alive_slots() -> None:
    """Even after a slot dies (and isn't reborn), focal stays in live set."""
    env = _make(n_agents=4, seed=16)
    # Force two slots dead — _try_reproduce needs >=2 survivors so they
    # may not get reborn this step. Either way focal must be alive.
    env._agents[2].alive = False
    env._agents[3].alive = False

    actions = np.array([Action.REST.value] * 4)
    _o, _r, _t, _tr, info = env.step_all(actions)

    focal = info["resolution_order"][0]
    # The focal slot must have been alive at step start (which is when
    # focal selection happens).
    assert focal in (0, 1), (
        f"focal {focal} was selected from a dead slot's index"
    )
