"""Tests for v22 multi-arena training: minions + arena-mix build_envs."""
from __future__ import annotations

import yaml

from murimsim.monster import (
    MINION_BASE_HEALTH,
    MINION_BASE_STRENGTH,
    MINION_LOOT_FOOD,
    MinionMonster,
    MonsterRegistry,
)
from murimsim.rl.ippo_env import IPPOEnv
from scripts.train_ippo import _load_arena_config, _parse_arena_mix, build_envs


def _default_cfg() -> dict:
    with open("config/default.yaml") as f:
        return yaml.safe_load(f)


# ── MinionMonster unit tests ──────────────────────────────────────────────────


def test_spawn_minion_stats():
    reg = MonsterRegistry()
    m = reg.spawn_minion((5, 5))
    assert isinstance(m, MinionMonster)
    assert m.kind == "minion"
    assert m.health == MINION_BASE_HEALTH
    assert m.strength == MINION_BASE_STRENGTH
    assert m.alive


def test_minion_dies_after_enough_damage():
    reg = MonsterRegistry()
    m = reg.spawn_minion((3, 3))
    # Cumulative damage ≥ HP kills it (v23: HP=0.8)
    m.take_damage(0.3, "agent_a")
    m.take_damage(0.3, "agent_b")
    assert m.alive
    killed = m.take_damage(0.3, "agent_a")
    assert killed
    assert not m.alive
    assert m.attackers == {"agent_a", "agent_b"}


def test_minion_attack_damage_capped():
    reg = MonsterRegistry()
    m = reg.spawn_minion((0, 0))
    m.strength = 2.0
    assert m.attack_damage() <= 0.18


# ── IPPOEnv n_minions plumbing ────────────────────────────────────────────────


def test_ippo_env_spawns_n_minions():
    cfg = _default_cfg()
    env = IPPOEnv(config=cfg, n_agents=4, seed=22, n_minions=3)
    env.reset_all(seed=22)
    minions = [m for m in env._monsters.all_alive() if m.kind == "minion"]
    assert len(minions) == 3


def test_minion_respawn_keeps_count_constant():
    cfg = _default_cfg()
    env = IPPOEnv(config=cfg, n_agents=4, seed=22, n_minions=2)
    env.reset_all(seed=22)
    # Kill one directly
    alive = [m for m in env._monsters.all_alive() if m.kind == "minion"]
    alive[0].take_damage(10.0, "test")
    assert not alive[0].alive
    # Step env once — respawn should top-up to 2 again
    import numpy as np
    actions = np.zeros(4, dtype=np.int64)
    env.step_all(actions)
    n_alive = sum(1 for m in env._monsters.all_alive() if m.kind == "minion")
    assert n_alive == 2


def test_minion_loot_drop_on_kill():
    cfg = _default_cfg()
    env = IPPOEnv(config=cfg, n_agents=4, seed=22, n_minions=1)
    env.reset_all(seed=22)
    minion = next(m for m in env._monsters.all_alive() if m.kind == "minion")
    minion.take_damage(10.0, "agent_x")
    env._drop_boss_loot(minion)
    stashes = [s for s in env._stash_registry.all_stashes() if s.owner_id == minion.monster_id]
    assert len(stashes) == 1
    assert stashes[0].food == MINION_LOOT_FOOD


# ── arena-mix parser + build_envs ─────────────────────────────────────────────


def test_parse_arena_mix_counts():
    out = _parse_arena_mix("base:1,arena_minion:2,arena_boss:1")
    assert out == ["base", "arena_minion", "arena_minion", "arena_boss"]


def test_parse_arena_mix_default_count_one():
    assert _parse_arena_mix("base,arena_minion") == ["base", "arena_minion"]


def test_load_arena_config_base_returns_input():
    base = {"world": {"grid_size": 30}}
    cfg, flags = _load_arena_config("base", base)
    assert cfg is base
    assert flags == {}


def test_load_arena_config_minion_strips_arena_section():
    cfg, flags = _load_arena_config("arena_minion", {})
    assert cfg["world"]["grid_size"] == 12
    assert "arena" not in cfg
    assert flags["n_minions"] == 1
    assert flags["enable_boss"] is False


def test_build_envs_arena_mix_round_robin():
    cfg = _default_cfg()
    envs = build_envs(
        cfg, n_envs=4, n_agents=4, seed=22,
        arena_mix="base:1,arena_minion:2,arena_boss:1",
    )
    for e in envs:
        e.reset_all(seed=22)
    assert envs[0]._world.grid_size == 30
    assert envs[0]._n_minions == 0 and not envs[0]._enable_boss
    assert envs[1]._world.grid_size == 12
    assert envs[1]._n_minions == 1 and not envs[1]._enable_boss
    assert envs[2]._world.grid_size == 12
    assert envs[2]._n_minions == 1 and not envs[2]._enable_boss
    assert envs[3]._world.grid_size == 24
    assert envs[3]._n_minions == 0 and envs[3]._enable_boss


# ── v23: per-damage co-attack bond ────────────────────────────────────────────


def test_co_attack_monster_bond_fires_on_second_attacker():
    """When a second agent hits the same minion, both get a co-attack bond."""
    cfg = _default_cfg()
    env = IPPOEnv(config=cfg, n_agents=4, seed=22, n_minions=1)
    env.reset_all(seed=22)
    minion = next(m for m in env._monsters.all_alive() if m.kind == "minion")
    a0_id = env._agents[0].agent_id
    a1_id = env._agents[1].agent_id
    # First attacker hits — no bond yet (no other prior attacker)
    env._record_co_attack_bonds(env._agents[0], set())
    minion.take_damage(0.1, a0_id)
    assert 1 not in env._affinity_raw.get(0, {})
    # Second attacker hits — should bond with first.
    env._record_co_attack_bonds(env._agents[1], minion.attackers)
    minion.take_damage(0.1, a1_id)
    raw_0_to_1 = env._affinity_raw.get(0, {}).get(1, (0.0, 0))[0]
    raw_1_to_0 = env._affinity_raw.get(1, {}).get(0, (0.0, 0))[0]
    assert raw_0_to_1 > 0 and raw_1_to_0 > 0
    assert abs(raw_0_to_1 - raw_1_to_0) < 1e-6  # symmetric


def test_co_attack_skips_self_and_dead():
    cfg = _default_cfg()
    env = IPPOEnv(config=cfg, n_agents=4, seed=22, n_minions=1)
    env.reset_all(seed=22)
    a0_id = env._agents[0].agent_id
    a1_id = env._agents[1].agent_id
    env._agents[1].alive = False  # kill agent 1
    # Pass {a0, a1} as prior attackers — neither should bond (a0 is self, a1 dead)
    env._record_co_attack_bonds(env._agents[0], {a0_id, a1_id})
    assert env._affinity_raw.get(0, {}).get(1, (0.0, 0))[0] == 0
