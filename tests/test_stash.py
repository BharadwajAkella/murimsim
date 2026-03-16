"""Tests for the personal stash system (Phase 3+).

Covers:
  1. deposit_transfers_inventory — inventory moves into stash, qi cost deducted
  2. deposit_requires_qi — agent with no qi cannot deposit
  3. withdraw_returns_resources — deposit then withdraw restores inventory
  4. steal_takes_enemy_stash — agent B steals agent A's stash
  5. stash_obs_channels — my_stash / enemy_stash channels populated correctly
  6. stash_resets_on_env_reset — StashRegistry clears on env.reset()
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from murimsim.agent import Agent, AgentInventory
from murimsim.stash import Stash, StashRegistry, STASH_QI_COST

CONFIG_PATH = Path("config/default.yaml")


def _load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _make_agent(agent_id: str = "agent_0", pos: tuple[int, int] = (5, 5)) -> Agent:
    """Create a minimal agent for testing without requiring an env."""
    rng = np.random.default_rng(0)
    cfg = _load_cfg()
    return Agent.spawn(agent_id, pos, rng, cfg)


# ---------------------------------------------------------------------------
# Test 1: deposit transfers inventory to stash
# ---------------------------------------------------------------------------

def test_deposit_transfers_inventory() -> None:
    """Depositing should empty the agent's inventory into the stash (minus qi cost)."""
    registry = StashRegistry()
    agent = _make_agent()
    agent.inventory = AgentInventory(food=5, qi=3, materials=2, poison=1)

    stash = registry.deposit(agent)

    assert stash is not None, "deposit() should return a Stash"
    assert stash.owner_id == agent.agent_id
    assert stash.position == agent.position

    # qi cost deducted, remaining qi moved to stash
    assert stash.food == 5
    assert stash.qi == 3 - STASH_QI_COST  # 2
    assert stash.materials == 2
    assert stash.poison == 1

    # Agent inventory zeroed
    assert agent.inventory.food == 0
    assert agent.inventory.qi == 0
    assert agent.inventory.materials == 0
    assert agent.inventory.poison == 0

    # Stash is registered
    assert stash in registry.all_stashes()
    assert stash in registry.get_stashes_at(*agent.position)


# ---------------------------------------------------------------------------
# Test 2: deposit requires at least 1 qi
# ---------------------------------------------------------------------------

def test_deposit_requires_qi() -> None:
    """An agent with no qi should not be able to deposit; returns None."""
    registry = StashRegistry()
    agent = _make_agent()
    agent.inventory = AgentInventory(food=10, qi=0, materials=0, poison=0)

    result = registry.deposit(agent)

    assert result is None, "deposit() should return None when agent has no qi"
    assert len(registry.all_stashes()) == 0, "No stash should be created"
    # Inventory unchanged
    assert agent.inventory.food == 10
    assert agent.inventory.qi == 0


# ---------------------------------------------------------------------------
# Test 3: withdraw returns resources to agent
# ---------------------------------------------------------------------------

def test_withdraw_returns_resources() -> None:
    """After deposit + withdraw the agent's inventory should be restored."""
    registry = StashRegistry()
    agent = _make_agent()
    agent.inventory = AgentInventory(food=4, qi=2, materials=1, poison=0)

    stash = registry.deposit(agent)
    assert stash is not None

    result = registry.withdraw(agent)

    assert result is True, "withdraw() should return True when resources found"
    # All stash contents returned to agent
    assert agent.inventory.food == stash.food
    assert agent.inventory.qi == stash.qi
    assert agent.inventory.materials == stash.materials
    assert agent.inventory.poison == stash.poison

    # Stash removed
    assert len(registry.all_stashes()) == 0


def test_withdraw_returns_false_when_empty() -> None:
    """withdraw() returns False when no own stash is at the agent's position."""
    registry = StashRegistry()
    agent = _make_agent()

    result = registry.withdraw(agent)
    assert result is False


# ---------------------------------------------------------------------------
# Test 4: steal takes an enemy stash
# ---------------------------------------------------------------------------

def test_steal_takes_enemy_stash() -> None:
    """Agent B stealing agent A's stash should give B the resources."""
    registry = StashRegistry()

    pos = (10, 10)
    agent_a = _make_agent("agent_a", pos)
    agent_b = _make_agent("agent_b", pos)

    agent_a.inventory = AgentInventory(food=6, qi=2, materials=0, poison=0)

    stash = registry.deposit(agent_a)
    assert stash is not None

    agent_b.inventory = AgentInventory(food=0, qi=0, materials=0, poison=0)
    stolen = registry.steal(agent_b)

    assert stolen is not None, "steal() should return the stolen stash"
    assert stolen.owner_id == agent_a.agent_id

    # Agent B now has the resources
    assert agent_b.inventory.food == stash.food
    assert agent_b.inventory.qi == stash.qi
    assert agent_b.inventory.materials == stash.materials

    # Stash is removed from registry
    assert len(registry.all_stashes()) == 0


def test_steal_returns_none_when_no_enemy_stash() -> None:
    """steal() returns None when there is no enemy stash at the agent's position."""
    registry = StashRegistry()
    agent = _make_agent()

    result = registry.steal(agent)
    assert result is None


# ---------------------------------------------------------------------------
# Test 5: stash obs channels
# ---------------------------------------------------------------------------

def test_stash_obs_channels() -> None:
    """After deposit, my_stash channel is 1.0 at agent position in obs.
    Before steal, enemy_stash channel is 1.0 at the stash position."""
    from murimsim.rl.env import (
        SurvivalEnv,
        OBS_CHANNEL_MY_STASH,
        OBS_CHANNEL_ENEMY_STASH,
        OBS_N_CHANNELS,
        OBS_VIEW_SIZE,
        OBS_GRID_SIZE,
    )

    cfg = _load_cfg()
    env = SurvivalEnv(config=cfg, seed=42)
    obs, _ = env.reset(seed=42)

    agent = env._agent
    # Give agent enough qi to deposit
    agent.inventory.qi = 3
    agent.inventory.food = 2

    # Deposit
    stash = env._stash_registry.deposit(agent)
    assert stash is not None

    obs_after = env._build_obs()
    grid = obs_after[:OBS_GRID_SIZE].reshape(OBS_VIEW_SIZE, OBS_VIEW_SIZE, OBS_N_CHANNELS)

    # Center of the 5×5 window is the agent's own position
    cx = OBS_VIEW_SIZE // 2
    cy = OBS_VIEW_SIZE // 2
    assert grid[cy, cx, OBS_CHANNEL_MY_STASH] == 1.0, (
        "my_stash channel should be 1.0 at agent's own position after deposit"
    )
    assert grid[cy, cx, OBS_CHANNEL_ENEMY_STASH] == 0.0, (
        "enemy_stash channel should be 0.0 for own stash"
    )

    # Now simulate another agent's stash at the same tile (enemy stash)
    from murimsim.stash import Stash
    enemy_stash = Stash(
        stash_id="agent_other_stash_0",
        owner_id="agent_other",
        position=agent.position,
        food=1,
    )
    env._stash_registry._stashes[enemy_stash.stash_id] = enemy_stash

    obs_enemy = env._build_obs()
    grid_enemy = obs_enemy[:OBS_GRID_SIZE].reshape(OBS_VIEW_SIZE, OBS_VIEW_SIZE, OBS_N_CHANNELS)
    assert grid_enemy[cy, cx, OBS_CHANNEL_ENEMY_STASH] == 1.0, (
        "enemy_stash channel should be 1.0 when enemy stash is at agent's position"
    )


# ---------------------------------------------------------------------------
# Test 6: stash registry resets on env.reset()
# ---------------------------------------------------------------------------

def test_stash_resets_on_env_reset() -> None:
    """After env.reset(), the StashRegistry should contain no stashes."""
    from murimsim.rl.env import SurvivalEnv

    cfg = _load_cfg()
    env = SurvivalEnv(config=cfg, seed=0)
    env.reset(seed=0)

    agent = env._agent
    agent.inventory.qi = 5
    agent.inventory.food = 3
    stash = env._stash_registry.deposit(agent)
    assert stash is not None
    assert len(env._stash_registry.all_stashes()) == 1

    # Reset the environment
    env.reset(seed=1)
    assert len(env._stash_registry.all_stashes()) == 0, (
        "StashRegistry must be cleared on env.reset()"
    )
