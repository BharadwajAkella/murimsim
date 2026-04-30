"""P2.3: slot state must be cleared on rebirth.

When an agent dies and is replaced via ``_try_reproduce``, the new agent
inherits only the deceased's *slot index* and inherited traits — nothing
else. All slot-keyed runtime state must be reset so the offspring starts
with a clean slate. Otherwise IPPO learns spurious correlations across
distinct lives in the same slot (and the rotating-focal symmetry is broken).

Slot-keyed state that must be cleared:
    * outgoing affinity row    : ``self._affinity_raw[idx]``
    * incoming affinity column : ``self._affinity_raw[other][idx]`` for all other
    * help-received bookkeeping: ``self._help_received[idx]`` and
      ``self._help_received[other][idx]``
    * reward EMA               : ``self._reward_ema[idx]``
    * damage-taken-last-step   : ``self._damage_taken_last_step[idx]``
    * group membership         : the new offspring must NOT inherit a group
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from murimsim.rl.multi_env import CombatEnv

CONFIG_PATH = Path("config/default.yaml")


def _load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _make_env(n_agents: int = 4, seed: int = 0) -> CombatEnv:
    env = CombatEnv(
        config=_load_cfg(),
        n_agents=n_agents,
        seed=seed,
        curriculum_ramp_steps=0,
    )
    env.reset(seed=seed)
    env._global_step_count = 1_000_000
    return env


def _force_kill(env: CombatEnv, idx: int) -> None:
    """Mark a slot as just-died so the next step's reproduce path replaces it."""
    agent = env._agents[idx]
    agent.health = 0.0
    agent.alive = False
    agent.death_cause = "starvation"


# ---------------------------------------------------------------------------
# Affinity row/column must reset
# ---------------------------------------------------------------------------

def test_outgoing_affinity_row_cleared_on_rebirth():
    env = _make_env(n_agents=4, seed=11)
    # Plant outgoing affinity from slot 2 → slot 0/1/3.
    env._record_affinity_event(2, 0, 0.5, 0.1)
    env._record_affinity_event(2, 1, -0.3, 0.05)
    env._record_affinity_event(2, 3, 0.2, 0.0)
    assert 2 in env._affinity_raw and len(env._affinity_raw[2]) == 3

    deceased = env._agents[2]
    _force_kill(env, 2)
    env._try_reproduce(deceased)

    # New agent in slot 2 must start with no outgoing affinity.
    assert env._affinity_raw.get(2, {}) == {}, (
        f"outgoing affinity row leaked across rebirth: {env._affinity_raw.get(2)}"
    )


def test_incoming_affinity_column_cleared_on_rebirth():
    env = _make_env(n_agents=4, seed=12)
    # Plant incoming affinity into slot 1 from slots 0/2/3.
    env._record_affinity_event(0, 1, 0.4, 0.4)
    env._record_affinity_event(2, 1, -0.2, -0.2)
    env._record_affinity_event(3, 1, 0.6, 0.6)
    for src in (0, 2, 3):
        assert 1 in env._affinity_raw[src]

    deceased = env._agents[1]
    _force_kill(env, 1)
    env._try_reproduce(deceased)

    # No surviving slot may still hold an affinity entry pointing at slot 1.
    for src in (0, 2, 3):
        row = env._affinity_raw.get(src, {})
        assert 1 not in row, (
            f"slot {src} still has incoming-column entry to reborn slot 1: {row}"
        )


# ---------------------------------------------------------------------------
# Help-received bookkeeping must reset
# ---------------------------------------------------------------------------

def test_help_received_cleared_on_rebirth():
    env = _make_env(n_agents=4, seed=13)
    env._help_received[0] = {2: 5}
    env._help_received[3] = {0: 7, 2: 9}
    env._help_received[2] = {0: 1, 1: 3}

    deceased = env._agents[0]
    _force_kill(env, 0)
    env._try_reproduce(deceased)

    assert env._help_received.get(0, {}) == {}
    # Other slots must drop entries keyed by reborn slot 0.
    assert 0 not in env._help_received.get(3, {})
    assert 0 not in env._help_received.get(2, {})
    # Unrelated entries are preserved.
    assert env._help_received[3].get(2) == 9
    assert env._help_received[2].get(1) == 3


# ---------------------------------------------------------------------------
# Scalar slot state must reset
# ---------------------------------------------------------------------------

def test_reward_ema_resets_on_rebirth():
    env = _make_env(n_agents=4, seed=14)
    env._reward_ema[1] = 0.42

    deceased = env._agents[1]
    _force_kill(env, 1)
    env._try_reproduce(deceased)

    assert env._reward_ema[1] == 0.0


def test_damage_taken_last_step_resets_on_rebirth():
    env = _make_env(n_agents=4, seed=15)
    env._damage_taken_last_step[3] = 0.7

    deceased = env._agents[3]
    _force_kill(env, 3)
    env._try_reproduce(deceased)

    assert env._damage_taken_last_step[3] == 0.0


# ---------------------------------------------------------------------------
# Group membership must not leak through rebirth
# ---------------------------------------------------------------------------

def test_group_membership_dropped_on_rebirth():
    env = _make_env(n_agents=4, seed=16)
    env._groups = [frozenset({0, 1, 2})]

    deceased = env._agents[1]
    _force_kill(env, 1)
    env._try_reproduce(deceased)

    # Slot 1's reborn agent must not be in any group.
    for g in env._groups:
        assert 1 not in g, f"reborn slot 1 still in group {g}"


def test_group_dissolved_when_drops_below_two_after_rebirth():
    env = _make_env(n_agents=4, seed=17)
    env._groups = [frozenset({0, 1})]  # 2-member group

    deceased = env._agents[0]
    _force_kill(env, 0)
    env._try_reproduce(deceased)

    # Reborn slot 0 leaves the group → only slot 1 remains → group dissolves.
    for g in env._groups:
        assert len(g) >= 2, f"group of size <2 still exists: {g}"


# ---------------------------------------------------------------------------
# End-to-end via env.step (more realistic — death + reproduce in same step)
# ---------------------------------------------------------------------------

def test_full_step_rebirth_clears_slot_state():
    """Drive rebirth through env.step (not direct _try_reproduce call)."""
    env = _make_env(n_agents=4, seed=18)

    # Plant state on slot 2.
    env._record_affinity_event(2, 0, 0.7, 0.1)
    env._record_affinity_event(0, 2, 0.5, 0.0)
    env._reward_ema[2] = 0.55
    env._damage_taken_last_step[2] = 0.3
    env._help_received[2] = {0: 4}
    env._help_received[0] = {2: 6}
    env._groups = [frozenset({1, 2, 3})]

    # Kill slot 2 — env.step's tick loop will detect death and call reproduce.
    env._agents[2].health = 0.0  # leave alive=True so tick() can flip it

    # Force tick to kill: set hunger over starvation threshold + zero resistance
    env._agents[2].hunger = 1.0
    env._agents[2].hunger_resistance = 0.0
    env._agents[2].health = 0.001  # one tick of starvation drain will finish it

    env.step(0)  # any action

    # Slot 2 may have been reborn (depends on starvation drain). If not reborn,
    # skip — this test is opportunistic on the integrated path.
    if env._agents[2].alive and env._agents[2].age == 0:
        # Reborn this step → all slot state must be clean.
        assert env._affinity_raw.get(2, {}) == {}
        assert 2 not in env._affinity_raw.get(0, {})
        assert env._reward_ema[2] == 0.0
        assert env._damage_taken_last_step[2] == 0.0
        assert env._help_received.get(2, {}) == {}
        assert 2 not in env._help_received.get(0, {})
        for g in env._groups:
            assert 2 not in g
    else:
        pytest.skip("slot 2 didn't rebirth this step; integration path not exercised")


# ---------------------------------------------------------------------------
# Determinism: rebirth path doesn't perturb RNG-dependent downstream state
# ---------------------------------------------------------------------------

def test_rebirth_reset_is_deterministic_with_seed():
    env_a = _make_env(n_agents=4, seed=99)
    env_b = _make_env(n_agents=4, seed=99)

    for env in (env_a, env_b):
        env._record_affinity_event(0, 1, 0.3, 0.2)
        env._reward_ema[1] = 0.4
        deceased = env._agents[1]
        _force_kill(env, 1)
        env._try_reproduce(deceased)

    assert env_a._reward_ema == env_b._reward_ema
    assert env_a._affinity_raw == env_b._affinity_raw
