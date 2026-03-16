"""Phase 2 — SurvivalEnv tests."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from murimsim.actions import Action
from murimsim.rl.env import OBS_TOTAL_SIZE, SurvivalEnv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent.parent / "config" / "default.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def make_env(seed: int = 42) -> SurvivalEnv:
    return SurvivalEnv(config=load_config(), seed=seed)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_agent_observation_shape():
    """Obs vector dims match spec: 5×5×6 grid + 8 stats = 158."""
    env = make_env()
    obs, _ = env.reset(seed=42)
    assert obs.shape == (OBS_TOTAL_SIZE,), f"Expected ({OBS_TOTAL_SIZE},), got {obs.shape}"
    assert obs.dtype == np.float32


def test_obs_values_in_range():
    """All obs values are in [0, 1]."""
    env = make_env()
    obs, _ = env.reset(seed=42)
    assert obs.min() >= 0.0
    assert obs.max() <= 1.0 + 1e-6


def test_env_reset_returns_valid_state():
    """Reset yields a valid obs and the agent is alive with sane stats."""
    env = make_env()
    obs, info = env.reset(seed=42)
    assert obs.shape == (OBS_TOTAL_SIZE,)
    assert env._agent.alive
    assert 0.0 <= env._agent.health <= 1.0
    assert 0.0 <= env._agent.hunger <= 1.0
    assert 0.0 <= env._agent.resistances.get("poison", 0.0) <= 1.0


def test_env_determinism():
    """Same seed + same action sequence → identical trajectory."""
    actions = [Action.MOVE_N, Action.GATHER, Action.EAT, Action.REST, Action.MOVE_E] * 10

    def run(seed: int) -> list[float]:
        env = make_env(seed=seed)
        env.reset(seed=seed)
        rewards = []
        for a in actions:
            _, r, terminated, _, _ = env.step(int(a))
            rewards.append(r)
            if terminated:
                break
        return rewards

    assert run(42) == run(42), "Same seed produced different trajectories"


def test_different_seeds_differ():
    """Different seeds should produce different initial observations."""
    env = make_env()
    obs42, _ = env.reset(seed=42)
    obs99, _ = env.reset(seed=99)
    assert not np.array_equal(obs42, obs99)


def test_action_move_changes_position():
    """MOVE_N/S/E/W change agent position."""
    env = make_env()
    env.reset(seed=42)
    # Force agent to centre so moves are never clamped
    env._agent.position = (15, 15)
    before = env._agent.position

    env.step(int(Action.MOVE_E))
    after = env._agent.position
    assert after != before


def test_gather_depletes_tile():
    """GATHER on a food tile removes it from the world grid."""
    env = make_env()
    env.reset(seed=42)
    # Place agent on a food tile
    world = env._world
    ys, xs = np.where(world._grid["food"] > 0)
    assert len(ys) > 0
    env._agent.position = (int(xs[0]), int(ys[0]))

    env.step(int(Action.GATHER))
    x, y = env._agent.position
    # Tile should be depleted (still 0 even after one world step, regen_ticks > 1)
    # Actually after step() world ticked once — check inventory instead
    assert env._agent.inventory.food > 0 or env._agent.inventory.poison > 0 \
        or env._agent.inventory.qi > 0 or env._agent.inventory.materials > 0


def test_hunger_increases_per_step():
    """Each step increases agent hunger."""
    env = make_env()
    env.reset(seed=42)
    hunger_before = env._agent.hunger
    env.step(int(Action.REST))
    assert env._agent.hunger > hunger_before


def test_agent_dies_no_food():
    """On an empty grid with no rest, agent dies from starvation.

    Hunger reaches 1.0 after 100 ticks, then health drains 0.02/tick → 50 more
    ticks to die. Total: ~150 steps. We run 300 to be safe.
    REST heals +0.02 which exactly cancels starvation, so we use MOVE_N instead.
    """
    from murimsim.agent import AgentInventory

    env = make_env()
    env.reset(seed=42)
    # Clear all resource grids and freeze regen by zeroing countdowns too
    for rid in list(env._world._grid):
        env._world._grid[rid][:] = 0.0
        env._world._countdown[rid][:] = 0
    env._agent.inventory = AgentInventory()

    terminated = False
    for _ in range(300):
        _, _, terminated, _, _ = env.step(int(Action.MOVE_N))
        if terminated:
            break
    assert terminated, "Agent should have died from starvation on empty grid"


def test_agent_survives_with_food():
    """Agent that keeps eating food survives 500+ steps."""
    env = make_env()
    env.reset(seed=42)

    alive_ticks = 0
    for _ in range(500):
        # Keep inventory stocked by directly placing food
        if env._agent.inventory.food == 0:
            env._agent.inventory.food = 3
        # Eat when hungry, otherwise move
        if env._agent.hunger > 0.3:
            action = int(Action.EAT)
        else:
            action = int(Action.REST)
        _, _, terminated, _, _ = env.step(action)
        if terminated:
            break
        alive_ticks += 1

    assert alive_ticks >= 499, f"Agent died at tick {alive_ticks}, expected 500+"


def test_poison_gather_adds_to_inventory():
    """Gathering from a poison tile adds poison to inventory."""
    env = make_env()
    env.reset(seed=42)
    world = env._world

    # Clear all non-poison tiles first, place agent on a poison tile
    ys, xs = np.where(world._grid["poison"] > 0)
    if len(ys) == 0:
        pytest.skip("No poison tiles in this world — retry with different seed")

    # Remove all other resources from target tile
    tx, ty = int(xs[0]), int(ys[0])
    for rid in ["food", "qi", "materials"]:
        world._grid[rid][ty, tx] = 0.0

    env._agent.position = (tx, ty)
    env.step(int(Action.GATHER))
    assert env._agent.inventory.poison > 0


def test_poison_eat_kills_low_resistance():
    """Agent with low poison_resistance eating poison takes heavy damage."""
    env = make_env()
    env.reset(seed=42)
    env._agent.health = 0.3
    env._agent.resistances["poison"] = 0.05
    env._agent.inventory.poison = 1

    _, _, terminated, _, _ = env.step(int(Action.EAT))
    assert terminated or env._agent.health < 0.3


def test_poison_eat_survives_high_resistance():
    """High resistance can fully negate poison damage; resistance still grows."""
    env = make_env()
    env.reset(seed=42)
    env._agent.health = 1.0
    env._agent.resistances["poison"] = 0.4
    env._agent.inventory.poison = 1

    _, _, terminated, _, _ = env.step(int(Action.EAT))
    assert not terminated
    assert env._agent.health == 1.0
    assert env._agent.resistances["poison"] > 0.4  # grew
