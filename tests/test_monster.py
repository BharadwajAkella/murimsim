"""Tests for v17 boss monster and shared loot stash mechanics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from murimsim.actions import Action
from murimsim.monster import (
    BOSS_BASE_HEALTH,
    BOSS_LOOT_FOOD,
    BOSS_LOOT_QI,
    BossMonster,
    Monster,
    MonsterRegistry,
)
from murimsim.rl.multi_env import CombatEnv
from murimsim.stash import Stash, StashRegistry

CONFIG_PATH = Path("config/default.yaml")


def _cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ─── Monster + registry unit tests ───────────────────────────────────────────


def test_monster_take_damage_credits_attacker() -> None:
    m = BossMonster(
        monster_id="boss_0", kind="boss", position=(5, 5),
        health=2.0, max_health=2.0, strength=1.0,
    )
    killed = m.take_damage(0.5, "agent_0")
    assert not killed
    assert m.health == 1.5
    assert "agent_0" in m.attackers
    killed = m.take_damage(2.0, "agent_1")
    assert killed
    assert m.health == 0.0
    assert not m.alive
    assert m.attackers == {"agent_0", "agent_1"}


def test_monster_zero_damage_does_not_credit() -> None:
    m = BossMonster(
        monster_id="boss_0", kind="boss", position=(5, 5),
        health=2.0, max_health=2.0, strength=1.0,
    )
    m.take_damage(0.0, "agent_0")
    assert "agent_0" not in m.attackers


def test_monster_registry_spawn_and_query() -> None:
    reg = MonsterRegistry()
    boss = reg.spawn_boss((3, 4))
    assert boss.kind == "boss"
    assert boss.health == BOSS_BASE_HEALTH
    assert reg.get_at(3, 4) == [boss]
    assert reg.get_adjacent_to(2, 3) == [boss]   # diagonal adjacency
    assert reg.get_adjacent_to(0, 0) == []


# ─── Stash participants ──────────────────────────────────────────────────────


def test_stash_participants_grant_access() -> None:
    s = Stash(
        stash_id="x", owner_id="boss_0", position=(1, 1),
        food=10, participants=["agent_0", "agent_2"],
    )
    assert s.is_accessible_to("boss_0")
    assert s.is_accessible_to("agent_0")
    assert s.is_accessible_to("agent_2")
    assert not s.is_accessible_to("agent_1")


def test_stash_registry_register_pre_built() -> None:
    reg = StashRegistry()
    loot = Stash(
        stash_id="boss_0_loot", owner_id="boss_0", position=(2, 2),
        food=20, participants=["agent_a", "agent_b"],
    )
    reg.register(loot)
    # Both participants see it as own
    assert reg.get_own_stash_at("agent_a", 2, 2) == [loot]
    assert reg.get_own_stash_at("agent_b", 2, 2) == [loot]
    # Non-participant sees it as enemy (and can steal it)
    assert reg.get_enemy_stashes_at("agent_c", 2, 2) == [loot]
    assert reg.get_own_stash_at("agent_c", 2, 2) == []


# ─── CombatEnv integration ───────────────────────────────────────────────────


def test_combat_env_disabled_by_default() -> None:
    env = CombatEnv(config=_cfg(), n_agents=4, seed=0)
    env.reset(seed=0)
    assert env._monsters.all_alive() == []


def test_combat_env_spawns_boss_when_enabled() -> None:
    env = CombatEnv(config=_cfg(), n_agents=4, seed=0, enable_boss=True)
    env.reset(seed=0)
    live = env._monsters.all_alive()
    assert len(live) == 1
    assert live[0].kind == "boss"
    gs = env._world.grid_size
    bx, by = live[0].position
    # Boss spawns in a corner
    assert (bx in (0, gs - 1)) and (by in (0, gs - 1))


def test_boss_attack_credits_focal_attacker() -> None:
    env = CombatEnv(config=_cfg(), n_agents=4, seed=0, enable_boss=True)
    env.reset(seed=0)
    boss = env._monsters.all_alive()[0]
    focal = env._agents[env._focal_idx]
    # Teleport focal next to boss for adjacency
    bx, by = boss.position
    target_pos = (
        max(0, min(env._world.grid_size - 1, bx + 1)),
        by,
    )
    focal.position = target_pos
    focal.health = 1.0
    focal.strength = 0.5
    # Force ATTACK regardless of curriculum
    env.combat_prob  # noqa: B018 — ensure no error
    env._global_step_count = 10**9  # ensure combat_prob == 1.0
    env.step(Action.ATTACK.value)
    assert focal.agent_id in boss.attackers


def test_boss_death_drops_loot_with_participants() -> None:
    env = CombatEnv(config=_cfg(), n_agents=4, seed=0, enable_boss=True)
    env.reset(seed=0)
    boss = env._monsters.all_alive()[0]
    # Direct kill: simulate two attackers landing damage, then a kill
    boss.take_damage(1.0, "agent_a")
    boss.take_damage(1.0, "agent_b")
    killed = boss.take_damage(BOSS_BASE_HEALTH, "agent_c")
    assert killed
    env._drop_boss_loot(boss)
    loot_stashes = env._stash_registry.get_stashes_at(*boss.position)
    assert len(loot_stashes) == 1
    loot = loot_stashes[0]
    assert loot.food == BOSS_LOOT_FOOD
    assert loot.qi == BOSS_LOOT_QI
    assert set(loot.participants) == {"agent_a", "agent_b", "agent_c"}
    # Each attacker can withdraw; non-attacker cannot
    assert env._stash_registry.get_own_stash_at("agent_a", *boss.position) == [loot]
    assert env._stash_registry.get_own_stash_at("agent_z", *boss.position) == []


def test_boss_observation_overlay() -> None:
    env = CombatEnv(config=_cfg(), n_agents=4, seed=0, enable_boss=True)
    obs, _ = env.reset(seed=0)
    boss = env._monsters.all_alive()[0]
    focal = env._agents[env._focal_idx]
    # Place focal next to boss so boss is in the 5×5 view
    focal.position = boss.position
    obs = env._build_obs(env._focal_idx)
    # Agent grid sits at flat indices 100..200, shape (5,5,4) — center is wy=wx=2
    agent_grid = obs[100:200].reshape(5, 5, 4)
    # Boss at same tile → wy=wx=2 should have presence > 0 and strength > 1.0
    assert agent_grid[2, 2, 0] > 0.0
    assert agent_grid[2, 2, 2] >= boss.strength - 1e-6


def test_no_loot_when_boss_unkilled() -> None:
    env = CombatEnv(config=_cfg(), n_agents=4, seed=0, enable_boss=True)
    env.reset(seed=0)
    boss = env._monsters.all_alive()[0]
    assert env._stash_registry.get_stashes_at(*boss.position) == []
