"""v21c — carry cost: inventory weight reduces combat strength and TRAIN gain."""
from __future__ import annotations

import yaml
import pytest

from murimsim.rl.ippo_env import IPPOEnv
from murimsim.rl.multi_env import (
    CARRY_MIN_STRENGTH_FRAC,
    CARRY_STRENGTH_PENALTY_PER_ITEM,
    STRIKE_BASIC,
)


@pytest.fixture
def cfg() -> dict:
    with open("config/default.yaml") as f:
        return yaml.safe_load(f)


def _make_env(cfg: dict, *, enable_carry_cost: bool) -> IPPOEnv:
    env = IPPOEnv(config=cfg, n_agents=4, seed=21, enable_carry_cost=enable_carry_cost)
    env.reset_all(seed=21)
    return env


def test_carry_factor_unencumbered_is_one(cfg: dict) -> None:
    env = _make_env(cfg, enable_carry_cost=True)
    a = env._agents[0]
    a.inventory.food = 0
    a.inventory.qi = 0
    a.inventory.materials = 0
    a.inventory.poison = 0
    assert env._carry_factor(a) == 1.0


def test_carry_factor_linear_then_floor(cfg: dict) -> None:
    env = _make_env(cfg, enable_carry_cost=True)
    a = env._agents[0]
    a.inventory.food = 5
    a.inventory.qi = 0
    a.inventory.materials = 0
    a.inventory.poison = 0
    expected = 1.0 - CARRY_STRENGTH_PENALTY_PER_ITEM * 5
    assert env._carry_factor(a) == pytest.approx(expected)
    # Push into the floor.
    a.inventory.food = 100
    assert env._carry_factor(a) == CARRY_MIN_STRENGTH_FRAC


def test_combat_damage_reduced_when_carrying(cfg: dict) -> None:
    env = _make_env(cfg, enable_carry_cost=True)
    attacker = env._agents[0]
    defender = env._agents[1]
    attacker.strength = 0.8
    attacker.hunger = 0.0
    # Empty inventory baseline.
    attacker.inventory.food = 0
    attacker.inventory.qi = 0
    attacker.inventory.materials = 0
    attacker.inventory.poison = 0
    base = env._combat_damage(attacker, defender, is_defending=False, tier=STRIKE_BASIC)
    # Heavy load.
    attacker.inventory.food = 10
    loaded = env._combat_damage(attacker, defender, is_defending=False, tier=STRIKE_BASIC)
    assert loaded < base
    assert loaded == pytest.approx(base * env._carry_factor(attacker))


def test_combat_damage_unchanged_when_disabled(cfg: dict) -> None:
    env = _make_env(cfg, enable_carry_cost=False)
    attacker = env._agents[0]
    defender = env._agents[1]
    attacker.strength = 0.8
    attacker.hunger = 0.0
    attacker.inventory.food = 0
    base = env._combat_damage(attacker, defender, is_defending=False, tier=STRIKE_BASIC)
    attacker.inventory.food = 10
    loaded = env._combat_damage(attacker, defender, is_defending=False, tier=STRIKE_BASIC)
    assert loaded == pytest.approx(base)


def test_train_gain_reduced_when_carrying(cfg: dict) -> None:
    env = _make_env(cfg, enable_carry_cost=True)
    a = env._agents[0]
    a.strength = 0.4

    # Baseline: no inventory.
    a.inventory.food = 0
    a.inventory.qi = 0
    a.inventory.materials = 0
    a.inventory.poison = 0
    pre = a.strength
    a.train(qi_field_value=0.0)
    base_gain = a.strength - pre

    # Heavy: same starting strength but loaded.
    a.strength = 0.4
    a.inventory.food = 5
    pre = a.strength
    a.train(qi_field_value=0.0)
    raw_gain = a.strength - pre
    # The raw agent.train doesn't know about carry cost; the ENV applies the
    # penalty in _execute_override_action / _apply_action wrappers. So calling
    # train() directly here should NOT scale — verifies our scoping is right.
    assert raw_gain == pytest.approx(base_gain)


def test_train_via_env_action_applies_carry_penalty(cfg: dict) -> None:
    """Drive TRAIN through _execute_override_action and verify scaling."""
    env = _make_env(cfg, enable_carry_cost=True)
    a = env._agents[0]
    a.strength = 0.4
    a.hunger = 0.0
    a.inventory.food = 0
    a.inventory.qi = 0
    a.inventory.materials = 0
    a.inventory.poison = 0

    from murimsim.actions import Action
    env._execute_override_action(a, 0, int(Action.TRAIN), env._agents[1], False)
    base_gain = a.strength - 0.4

    a.strength = 0.4
    a.inventory.food = 5
    pre = a.strength
    env._execute_override_action(a, 0, int(Action.TRAIN), env._agents[1], False)
    loaded_gain = a.strength - pre

    assert loaded_gain < base_gain
    assert loaded_gain == pytest.approx(base_gain * env._carry_factor(a), rel=1e-4)


def test_train_via_env_action_unchanged_when_disabled(cfg: dict) -> None:
    env = _make_env(cfg, enable_carry_cost=False)
    a = env._agents[0]
    a.strength = 0.4
    a.hunger = 0.0
    a.inventory.food = 0

    from murimsim.actions import Action
    env._execute_override_action(a, 0, int(Action.TRAIN), env._agents[1], False)
    base_gain = a.strength - 0.4

    a.strength = 0.4
    a.inventory.food = 10
    pre = a.strength
    env._execute_override_action(a, 0, int(Action.TRAIN), env._agents[1], False)
    loaded_gain = a.strength - pre
    assert loaded_gain == pytest.approx(base_gain, rel=1e-4)


def test_default_carry_cost_disabled(cfg: dict) -> None:
    env = IPPOEnv(config=cfg, n_agents=4, seed=21)
    env.reset_all(seed=21)
    assert env._enable_carry_cost is False
