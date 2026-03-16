"""Tests for Phase 3c: combat environment.

Validates the combat mechanics, curriculum schedule, and that all
Phase 1–2 tests still pass (gate enforcement).
"""
from __future__ import annotations

import numpy as np
import pytest
import yaml
from pathlib import Path

from murimsim.rl.multi_env import (
    CombatEnv,
    COMBAT_ATTACKER_SCALE,
    COMBAT_DEFENDER_SCALE,
    COMBAT_MAX_DAMAGE,
    CURRICULUM_START_PROB,
    CURRICULUM_RAMP_STEPS,
)
from murimsim.agent import Agent

CONFIG_PATH = Path("config/default.yaml")


def _load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _make_combat_env(n_agents: int = 5, seed: int = 0, ramp_steps: int = 300_000) -> CombatEnv:
    return CombatEnv(config=_load_cfg(), n_agents=n_agents, seed=seed, curriculum_ramp_steps=ramp_steps)


def _make_agent(strength: float, health: float = 1.0, pos: tuple = (0, 0)) -> Agent:
    """Build an Agent with controlled strength for testing."""
    import numpy as np
    rng = np.random.default_rng(0)
    a = Agent.spawn("test", pos, rng, _load_cfg())
    a.strength = strength
    a.health = health
    return a


# ---------------------------------------------------------------------------
# Test 1: combat_damage formula
# ---------------------------------------------------------------------------

def test_combat_damage() -> None:
    """damage = attacker.strength*0.3 - 0, clamped [0, 0.5]."""
    env = _make_combat_env()
    env.reset(seed=0)

    attacker = _make_agent(strength=0.8)
    defender = _make_agent(strength=0.4)
    damage = env._combat_damage(attacker, defender, is_defending=False)

    expected = attacker.strength * COMBAT_ATTACKER_SCALE
    assert abs(damage - expected) < 1e-6, f"Expected {expected:.4f}, got {damage:.4f}"
    assert 0.0 <= damage <= COMBAT_MAX_DAMAGE


# ---------------------------------------------------------------------------
# Test 2: defending reduces damage
# ---------------------------------------------------------------------------

def test_combat_damage_defending() -> None:
    """Defending agent takes less damage than a non-defending one."""
    env = _make_combat_env()
    env.reset(seed=0)

    attacker = _make_agent(strength=0.8)
    defender = _make_agent(strength=0.5)

    dmg_no_defend = env._combat_damage(attacker, defender, is_defending=False)
    dmg_defending = env._combat_damage(attacker, defender, is_defending=True)

    assert dmg_defending < dmg_no_defend, (
        f"Defending should reduce damage: {dmg_defending:.4f} vs {dmg_no_defend:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 3: attack requires adjacency
# ---------------------------------------------------------------------------

def test_attack_requires_adjacency() -> None:
    """ATTACK on a non-adjacent agent is a no-op (returns 0 damage)."""
    from murimsim.actions import Action

    env = _make_combat_env(n_agents=2, seed=5)
    env.reset(seed=5)

    focal = env._agents[env._focal_idx]
    other_idx = (env._focal_idx + 1) % 2
    other = env._agents[other_idx]

    # Force non-adjacent positions (distance > 1)
    focal.position = (0, 0)
    other.position = (5, 5)

    damage, killed = env._do_attack(focal)
    assert damage == 0.0, f"Expected 0 damage for non-adjacent attack, got {damage}"
    assert not killed


# ---------------------------------------------------------------------------
# Test 4: strength affects combat outcome
# ---------------------------------------------------------------------------

def test_strength_affects_combat() -> None:
    """Higher attacker strength → more damage; higher defender strength → less damage."""
    env = _make_combat_env()
    env.reset(seed=0)

    weak_attacker = _make_agent(strength=0.2)
    strong_attacker = _make_agent(strength=0.9)
    defender = _make_agent(strength=0.5)

    dmg_weak = env._combat_damage(weak_attacker, defender, is_defending=False)
    dmg_strong = env._combat_damage(strong_attacker, defender, is_defending=False)
    assert dmg_strong > dmg_weak, f"Strong attacker should deal more damage: {dmg_strong} vs {dmg_weak}"

    strong_defender = _make_agent(strength=0.9)
    weak_defender = _make_agent(strength=0.1)
    attacker = _make_agent(strength=0.6)

    dmg_vs_strong = env._combat_damage(attacker, strong_defender, is_defending=True)
    dmg_vs_weak = env._combat_damage(attacker, weak_defender, is_defending=True)
    assert dmg_vs_strong <= dmg_vs_weak, (
        f"Strong defender should take ≤ damage: {dmg_vs_strong} vs {dmg_vs_weak}"
    )


# ---------------------------------------------------------------------------
# Test 5: combat determinism
# ---------------------------------------------------------------------------

def test_combat_determinism() -> None:
    """Same positions, stats, and seed → same damage output every time."""
    env = _make_combat_env(n_agents=2, seed=7)
    env.reset(seed=7)

    focal = env._agents[env._focal_idx]
    other_idx = (env._focal_idx + 1) % 2
    other = env._agents[other_idx]

    # Place adjacent
    focal.position = (2, 2)
    other.position = (3, 2)
    focal.strength = 0.7
    other.strength = 0.4

    dmg1 = env._combat_damage(focal, other, is_defending=False)
    dmg2 = env._combat_damage(focal, other, is_defending=False)
    assert dmg1 == dmg2, f"Combat damage must be deterministic: {dmg1} vs {dmg2}"


# ---------------------------------------------------------------------------
# Test 6: curriculum schedule
# ---------------------------------------------------------------------------

def test_combat_curriculum_schedule() -> None:
    """combat_prob starts at CURRICULUM_START_PROB and ramps toward 1.0."""
    env = _make_combat_env(ramp_steps=1000)

    # At step 0
    assert abs(env.combat_prob - CURRICULUM_START_PROB) < 1e-6, (
        f"Initial combat_prob should be {CURRICULUM_START_PROB}, got {env.combat_prob}"
    )

    # After half the ramp
    env._global_step_count = 500
    mid_prob = env.combat_prob
    assert CURRICULUM_START_PROB < mid_prob < 1.0, (
        f"Mid-ramp combat_prob should be between {CURRICULUM_START_PROB} and 1.0, got {mid_prob}"
    )

    # After full ramp
    env._global_step_count = 1000
    assert abs(env.combat_prob - 1.0) < 1e-6, (
        f"Final combat_prob should be 1.0, got {env.combat_prob}"
    )

    # Beyond ramp — should not exceed 1.0
    env._global_step_count = 99_999
    assert env.combat_prob == 1.0, f"combat_prob should cap at 1.0, got {env.combat_prob}"
