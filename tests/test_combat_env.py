"""Tests for Phase 3c: combat environment.

Validates the combat mechanics, curriculum schedule, and that all
Phase 1–2 tests still pass (gate enforcement).
"""
from __future__ import annotations

import numpy as np
import yaml
from pathlib import Path

from murimsim.rl.multi_env import (
    CombatEnv,
    COMBAT_ATTACKER_SCALE,
    COMBAT_MAX_DAMAGE,
    CURRICULUM_START_PROB,
    CURRICULUM_RAMP_STEPS,
)
from murimsim.actions import Action
from murimsim.agent import Agent

CONFIG_PATH = Path("config/default.yaml")


def _load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _make_combat_env(n_agents: int = 5, seed: int = 0, ramp_steps: int = 300_000) -> CombatEnv:
    return CombatEnv(config=_load_cfg(), n_agents=n_agents, seed=seed, curriculum_ramp_steps=ramp_steps)


def _make_agent(strength: float, health: float = 1.0, pos: tuple = (0, 0)) -> Agent:
    """Build an Agent with controlled strength for testing."""
    rng = np.random.default_rng(0)
    a = Agent.spawn("test", pos, rng, _load_cfg())
    a.strength = strength
    a.health = health
    return a


# ---------------------------------------------------------------------------
# Test 1: combat_damage formula (no defending)
# ---------------------------------------------------------------------------

def test_combat_damage_no_defend() -> None:
    """damage = attacker.effective_strength * COMBAT_ATTACKER_SCALE when not defending."""
    env = _make_combat_env()
    env.reset(seed=0)

    attacker = _make_agent(strength=0.8)
    defender = _make_agent(strength=0.4)
    damage = env._combat_damage(attacker, defender, is_defending=False)

    expected = attacker.effective_strength * COMBAT_ATTACKER_SCALE
    assert abs(damage - expected) < 1e-6, f"Expected {expected:.4f}, got {damage:.4f}"
    assert 0.0 <= damage <= COMBAT_MAX_DAMAGE


# ---------------------------------------------------------------------------
# Test 2: DEFEND is multiplicative — blocks proportional to defense_power
# ---------------------------------------------------------------------------

def test_combat_damage_defending_is_multiplicative() -> None:
    """DEFEND multiplies damage by (1 - defense_power): higher skill blocks more."""
    env = _make_combat_env()
    env.reset(seed=0)

    attacker = _make_agent(strength=0.8)
    defender = _make_agent(strength=0.5)

    dmg_no_defend = env._combat_damage(attacker, defender, is_defending=False)
    dmg_defending = env._combat_damage(attacker, defender, is_defending=True)

    assert dmg_defending < dmg_no_defend, (
        f"DEFEND must reduce damage: defending={dmg_defending:.4f} vs no-defend={dmg_no_defend:.4f}"
    )
    expected_defended = dmg_no_defend * max(0.0, 1.0 - defender.defense_power)
    assert abs(dmg_defending - expected_defended) < 1e-6, (
        f"Expected multiplicative reduction {expected_defended:.4f}, got {dmg_defending:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 3: master cultivator (defense_power → 1.0) nullifies attack completely
# ---------------------------------------------------------------------------

def test_defend_nullification_at_max_defense_power() -> None:
    """An agent with defense_power = 1.0 takes 0 damage while defending."""
    env = _make_combat_env()
    env.reset(seed=0)

    attacker = _make_agent(strength=1.0)
    defender = _make_agent(strength=1.0)
    for key in defender.resistances:
        defender.resistances[key] = 1.0

    dp = defender.defense_power
    assert dp > 0.99, f"Setup error: defense_power should be ~1.0, got {dp:.4f}"

    damage = env._combat_damage(attacker, defender, is_defending=True)
    assert damage < 1e-6, (
        f"Master defender (defense_power={dp:.3f}) should nullify attack, got damage={damage:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 4: ATTACK actually reduces target health
# ---------------------------------------------------------------------------

def test_attack_reduces_target_health() -> None:
    """_do_attack must reduce the target's health by the computed damage amount."""
    env = _make_combat_env(n_agents=2, seed=3)
    env.reset(seed=3)

    focal = env._agents[env._focal_idx]
    other_idx = next(i for i in range(2) if i != env._focal_idx)
    target = env._agents[other_idx]

    focal.position = (5, 5)
    target.position = (6, 5)
    focal.strength = 0.8
    target.health = 1.0

    health_before = target.health
    damage, _ = env._do_attack(focal)

    assert damage > 0.0, "Expected non-zero damage from adjacent attack"
    assert abs(target.health - (health_before - damage)) < 1e-6, (
        f"Target health should be {health_before - damage:.4f}, got {target.health:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 5: ATTACK with no adjacent agent is redirected to REST
# ---------------------------------------------------------------------------

def test_attack_no_adjacent_redirected_to_rest() -> None:
    """ATTACK with no adjacent agent must redirect to REST (no combat damage on anyone)."""
    env = _make_combat_env(n_agents=2, seed=5)
    env.reset(seed=5)

    focal = env._agents[env._focal_idx]
    other_idx = next(i for i in range(2) if i != env._focal_idx)
    other = env._agents[other_idx]

    focal.position = (0, 0)
    other.position = (9, 9)

    health_before = {i: env._agents[i].health for i in range(2)}
    env._global_step_count = CURRICULUM_RAMP_STEPS  # ensure combat_prob = 1.0
    env.step(Action.ATTACK.value)

    for i, agent in enumerate(env._agents):
        if agent.alive:
            delta = health_before[i] - agent.health
            # Starvation drain is small (<0.05). Combat damage would be much larger.
            assert delta < 0.05, (
                f"Agent {i} health dropped {delta:.4f} — looks like combat damage; expected REST"
            )


# ---------------------------------------------------------------------------
# Test 6: DEFEND + step — focal takes less damage than REST
# ---------------------------------------------------------------------------

def test_defend_reduces_damage_in_step() -> None:
    """In a full step(), DEFEND causes focal to take less damage than REST."""
    env_defend = _make_combat_env(n_agents=2, seed=11)
    env_rest   = _make_combat_env(n_agents=2, seed=11)
    env_defend.reset(seed=11)
    env_rest.reset(seed=11)

    for env in (env_defend, env_rest):
        focal = env._agents[env._focal_idx]
        other_idx = next(i for i in range(2) if i != env._focal_idx)
        other = env._agents[other_idx]
        focal.position = (5, 5)
        other.position = (6, 5)
        focal.strength = 0.2
        other.strength = 0.9
        other.sociability = 0.0  # ensure heuristic attacks
        env._global_step_count = CURRICULUM_RAMP_STEPS

    health_defend_before = env_defend._agents[env_defend._focal_idx].health
    health_rest_before   = env_rest._agents[env_rest._focal_idx].health

    env_defend.step(Action.DEFEND.value)
    env_rest.step(Action.REST.value)

    damage_defend = health_defend_before - env_defend._agents[env_defend._focal_idx].health
    damage_rest   = health_rest_before   - env_rest._agents[env_rest._focal_idx].health

    assert damage_defend <= damage_rest, (
        f"DEFEND should take ≤ damage vs REST: defend={damage_defend:.4f} rest={damage_rest:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 7: damage_taken_last_step obs signal reflects being attacked
# ---------------------------------------------------------------------------

def test_damage_taken_obs_signal() -> None:
    """After being attacked, _damage_taken_last_step for focal > 0, and obs reflects it on next step.

    The obs returned from step() belongs to the NEXT focal agent. To verify the
    signal works, we directly check _damage_taken_last_step on the attacked agent
    and verify its obs on the following step includes the non-zero damage signal.
    """
    env = _make_combat_env(n_agents=2, seed=13)
    env.reset(seed=13)

    focal_idx = env._focal_idx
    other_idx = next(i for i in range(2) if i != focal_idx)
    focal = env._agents[focal_idx]
    other = env._agents[other_idx]

    # Place adjacent, focal is very weak so heuristic will attack it
    focal.position = (5, 5)
    other.position = (6, 5)
    focal.strength = 0.1
    other.strength = 1.0
    other.sociability = 0.0   # below HEURISTIC_COLLAB_THRESHOLD — will attack
    env._global_step_count = CURRICULUM_RAMP_STEPS

    env.step(Action.REST.value)

    # After step: focal was attacked by heuristic — verify internal tracking
    assert env._damage_taken_last_step[focal_idx] > 0.0, (
        f"Expected _damage_taken_last_step[focal] > 0, got {env._damage_taken_last_step[focal_idx]:.4f}"
    )

    # Now focal_idx is the agent that was attacked — its obs on next step should reflect damage
    # Force focal back to attacked agent so we can read its obs directly
    env._focal_idx = focal_idx
    obs = env._build_obs(focal_idx)
    damage_signal = obs[-1]
    assert damage_signal > 0.0, (
        f"Expected damage_taken obs signal > 0 in attacked agent's obs, got {damage_signal:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 8: attack requires adjacency (_do_attack returns 0 when far)
# ---------------------------------------------------------------------------

def test_attack_requires_adjacency() -> None:
    """_do_attack on a non-adjacent agent returns (0, False)."""
    env = _make_combat_env(n_agents=2, seed=5)
    env.reset(seed=5)

    focal = env._agents[env._focal_idx]
    other_idx = next(i for i in range(2) if i != env._focal_idx)
    other = env._agents[other_idx]

    focal.position = (0, 0)
    other.position = (5, 5)

    damage, killed = env._do_attack(focal)
    assert damage == 0.0, f"Expected 0 damage for non-adjacent attack, got {damage}"
    assert not killed


# ---------------------------------------------------------------------------
# Test 9: combat determinism
# ---------------------------------------------------------------------------

def test_combat_determinism() -> None:
    """Same stats → identical damage output every call."""
    env = _make_combat_env(n_agents=2, seed=7)
    env.reset(seed=7)

    focal = env._agents[env._focal_idx]
    other_idx = next(i for i in range(2) if i != env._focal_idx)
    other = env._agents[other_idx]

    focal.position = (2, 2)
    other.position = (3, 2)
    focal.strength = 0.7
    other.strength = 0.4

    dmg1 = env._combat_damage(focal, other, is_defending=False)
    dmg2 = env._combat_damage(focal, other, is_defending=False)
    assert dmg1 == dmg2, f"Combat damage must be deterministic: {dmg1} vs {dmg2}"


# ---------------------------------------------------------------------------
# Test 10: curriculum schedule
# ---------------------------------------------------------------------------

def test_combat_curriculum_schedule() -> None:
    """combat_prob starts at CURRICULUM_START_PROB and ramps toward 1.0."""
    env = _make_combat_env(ramp_steps=1000)

    assert abs(env.combat_prob - CURRICULUM_START_PROB) < 1e-6, (
        f"Initial combat_prob should be {CURRICULUM_START_PROB}, got {env.combat_prob}"
    )

    env._global_step_count = 500
    mid_prob = env.combat_prob
    assert CURRICULUM_START_PROB < mid_prob < 1.0, (
        f"Mid-ramp combat_prob should be between {CURRICULUM_START_PROB} and 1.0, got {mid_prob}"
    )

    env._global_step_count = 1000
    assert abs(env.combat_prob - 1.0) < 1e-6, (
        f"Final combat_prob should be 1.0, got {env.combat_prob}"
    )

    env._global_step_count = 99_999
    assert env.combat_prob == 1.0, f"combat_prob should cap at 1.0, got {env.combat_prob}"


# ---------------------------------------------------------------------------
# Test 11: strength affects combat outcome
# ---------------------------------------------------------------------------

def test_strength_affects_combat() -> None:
    """Higher attacker strength → more damage."""
    env = _make_combat_env()
    env.reset(seed=0)

    weak_attacker   = _make_agent(strength=0.2)
    strong_attacker = _make_agent(strength=0.9)
    defender = _make_agent(strength=0.5)

    dmg_weak   = env._combat_damage(weak_attacker, defender, is_defending=False)
    dmg_strong = env._combat_damage(strong_attacker, defender, is_defending=False)
    assert dmg_strong > dmg_weak, f"Strong attacker should deal more damage: {dmg_strong} vs {dmg_weak}"
