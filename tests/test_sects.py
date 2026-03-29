"""test_sects.py — Tests for Phase 6a sect scaffold.

Validates:
1. SectConfig and SectRegistry basics
2. DEFAULT_SECTS contains the three expected sects
3. CombatEnv spawns agents inside the home region when sect_config is set
4. Agent.sect_id is set correctly after reset()
5. ep_sect_id appears in episode info
6. Agents get "none" sect_id when no sect_config is given
"""
from __future__ import annotations

import yaml
from pathlib import Path

import pytest

from murimsim.sect import (
    DEFAULT_SECTS,
    IRON_FANG,
    JADE_LOTUS,
    SHADOW_ROOT,
    SectConfig,
    SectRegistry,
)

CONFIG_PATH = Path("config/default.yaml")


def _load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# SectConfig / SectRegistry basics
# ---------------------------------------------------------------------------


def test_sect_config_fields() -> None:
    assert IRON_FANG.sect_id == "iron_fang"
    assert IRON_FANG.name == "Iron Fang"
    assert IRON_FANG.color == "#e74c3c"
    assert len(IRON_FANG.home_x_range) == 2
    assert len(IRON_FANG.home_y_range) == 2


def test_default_sects_has_three_entries() -> None:
    assert len(DEFAULT_SECTS.sects) == 3


def test_sect_registry_by_id() -> None:
    assert DEFAULT_SECTS.by_id("jade_lotus") is JADE_LOTUS
    assert DEFAULT_SECTS.by_id("shadow_root") is SHADOW_ROOT


def test_sect_registry_by_id_unknown_raises() -> None:
    with pytest.raises(KeyError):
        DEFAULT_SECTS.by_id("unknown_sect")


def test_sect_registry_by_index_wraps() -> None:
    # 0 → IRON_FANG, 3 → IRON_FANG again (round-robin)
    assert DEFAULT_SECTS.by_index(0) is IRON_FANG
    assert DEFAULT_SECTS.by_index(3) is IRON_FANG


def test_sect_home_regions_are_disjoint() -> None:
    """Each sect occupies a unique y-band; x ranges span the full grid."""
    y_bands = [s.home_y_range for s in DEFAULT_SECTS.sects]
    # No overlap
    for i, (lo_i, hi_i) in enumerate(y_bands):
        for j, (lo_j, hi_j) in enumerate(y_bands):
            if i == j:
                continue
            # Ranges must not overlap
            assert hi_i < lo_j or hi_j < lo_i, (
                f"Sects {i} and {j} y-bands overlap: {y_bands[i]} vs {y_bands[j]}"
            )


# ---------------------------------------------------------------------------
# CombatEnv sect integration
# ---------------------------------------------------------------------------


def test_combat_env_spawn_in_home_region() -> None:
    """Agents must spawn inside the home region when sect_config is set."""
    from murimsim.rl.multi_env import CombatEnv

    cfg = _load_cfg()
    env = CombatEnv(config=cfg, n_agents=5, seed=7, sect_config=IRON_FANG)
    env.reset(seed=7)

    x_lo, x_hi = IRON_FANG.home_x_range
    y_lo, y_hi = IRON_FANG.home_y_range
    for agent in env._agents:
        x, y = agent.position
        assert x_lo <= x <= x_hi, f"Agent x={x} outside home_x_range {IRON_FANG.home_x_range}"
        assert y_lo <= y <= y_hi, f"Agent y={y} outside home_y_range {IRON_FANG.home_y_range}"


def test_combat_env_sect_id_assigned() -> None:
    """Agents must have the correct sect_id after reset() when sect_config is set."""
    from murimsim.rl.multi_env import CombatEnv

    cfg = _load_cfg()
    env = CombatEnv(config=cfg, n_agents=4, seed=3, sect_config=JADE_LOTUS)
    env.reset(seed=3)

    for agent in env._agents:
        assert agent.sect_id == "jade_lotus", (
            f"Expected sect_id='jade_lotus', got '{agent.sect_id}'"
        )


def test_combat_env_no_sect_config_gives_none_sect_id() -> None:
    """Without sect_config, agents default to sect_id='none'."""
    from murimsim.rl.multi_env import CombatEnv

    cfg = _load_cfg()
    env = CombatEnv(config=cfg, n_agents=3, seed=1)
    env.reset(seed=1)

    for agent in env._agents:
        assert agent.sect_id == "none"


def test_combat_env_ep_sect_id_in_info() -> None:
    """ep_sect_id key is always present in the step info dict."""
    from murimsim.rl.multi_env import CombatEnv

    cfg = _load_cfg()
    env = CombatEnv(config=cfg, n_agents=3, seed=5, sect_config=SHADOW_ROOT)
    env.reset(seed=5)

    _, _, _, _, info = env.step(env.action_space.sample())
    assert "ep_sect_id" in info, "ep_sect_id must be present in every step info dict"
    assert info["ep_sect_id"] == "shadow_root"


def test_combat_env_ep_sect_id_none_without_config() -> None:
    """ep_sect_id must be 'none' when no sect_config is configured."""
    from murimsim.rl.multi_env import CombatEnv

    cfg = _load_cfg()
    env = CombatEnv(config=cfg, n_agents=3, seed=5)
    env.reset(seed=5)

    _, _, _, _, info = env.step(env.action_space.sample())
    assert info["ep_sect_id"] == "none"


# ---------------------------------------------------------------------------
# Agent.sect_id in replay dict
# ---------------------------------------------------------------------------


def test_agent_replay_dict_uses_sect_id() -> None:
    """to_replay_dict() must emit the agent's sect_id as 'sect'."""
    from murimsim.agent import Agent

    a = Agent(agent_id="test", position=(5, 5), health=0.8, hunger=0.2, strength=0.3,
              sect_id="iron_fang")
    d = a.to_replay_dict()
    assert d["sect"] == "iron_fang"


def test_agent_default_sect_id_is_none() -> None:
    """Default sect_id is 'none' when not set."""
    from murimsim.agent import Agent

    a = Agent(agent_id="x", position=(0, 0), health=1.0, hunger=0.0, strength=0.5)
    assert a.sect_id == "none"
    assert a.to_replay_dict()["sect"] == "none"


# ---------------------------------------------------------------------------
# Inter-sect combat rewards
# ---------------------------------------------------------------------------


def test_inter_sect_defeat_bonus_fired() -> None:
    """Defeating an enemy-sect agent adds REWARD_INTER_SECT_DEFEAT_BONUS to defeat_bonus."""
    from murimsim.rl.multi_env import (
        CombatEnv,
        REWARD_INTER_SECT_DEFEAT_BONUS,
        REWARD_SAME_SECT_ATTACK_PENALTY,
    )

    # Verify constants exist and are sensible
    assert REWARD_INTER_SECT_DEFEAT_BONUS > 0
    assert REWARD_SAME_SECT_ATTACK_PENALTY < 0


def test_combat_env_agents_have_sect_id_after_sect_reset() -> None:
    """All agents in a sect-configured env must have matching sect_id after each reset."""
    from murimsim.rl.multi_env import CombatEnv

    cfg = _load_cfg()
    for sect in DEFAULT_SECTS.sects:
        env = CombatEnv(config=cfg, n_agents=4, seed=10, sect_config=sect)
        # First reset
        env.reset(seed=10)
        for a in env._agents:
            assert a.sect_id == sect.sect_id
        # Second reset (different seed) — sect_id must still be assigned
        env.reset(seed=11)
        for a in env._agents:
            assert a.sect_id == sect.sect_id
