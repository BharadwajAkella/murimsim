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


# ─── Behavioral tests (boss actually does things) ────────────────────────────


def test_boss_steps_toward_nearest_agent() -> None:
    """Boss should move one Chebyshev cell toward the nearest live agent each tick."""
    env = CombatEnv(config=_cfg(), n_agents=4, seed=0, enable_boss=True)
    env.reset(seed=0)
    boss = env._monsters.all_alive()[0]
    target = env._agents[0]
    target.position = (15, 15)
    target.health = 1.0
    boss.position = (5, 5)
    pre = boss.position
    env._monsters.tick_all(env._world, env._agents, env._rng)
    bx, by = boss.position
    px, py = pre
    cheb_pre = max(abs(15 - px), abs(15 - py))
    cheb_post = max(abs(15 - bx), abs(15 - by))
    assert cheb_post < cheb_pre, f"Boss did not approach: {pre} -> {(bx, by)}"


def test_boss_attacks_adjacent_agent_via_tick_all() -> None:
    """When boss is adjacent to a live agent, tick_all returns a damage event."""
    env = CombatEnv(config=_cfg(), n_agents=4, seed=0, enable_boss=True)
    env.reset(seed=0)
    boss = env._monsters.all_alive()[0]
    victim = env._agents[1]
    victim.position = boss.position
    victim.health = 1.0
    h0 = victim.health
    events = env._monsters.tick_all(env._world, env._agents, env._rng)
    assert len(events) >= 1
    mid, vid, dmg = events[0]
    assert mid == boss.monster_id
    assert vid == victim.agent_id
    assert dmg > 0
    assert victim.health < h0


def test_boss_damage_propagates_to_focal_obs_state() -> None:
    """Damage from boss to focal must show up in _damage_taken_last_step.

    Note: env._focal_idx rotates after step(), so capture it before.
    """
    env = CombatEnv(config=_cfg(), n_agents=4, seed=0, enable_boss=True)
    env.reset(seed=0)
    boss = env._monsters.all_alive()[0]
    focal_idx_before = env._focal_idx
    focal = env._agents[focal_idx_before]
    focal.position = boss.position
    focal.health = 1.0
    env._global_step_count = 10**9
    env.step(Action.TRAIN.value)
    assert env._damage_taken_last_step[focal_idx_before] > 0
    # Sanity: focal actually took damage
    assert focal.health < 1.0


def test_boss_lethal_hit_increments_kill_counter() -> None:
    """Boss damage that drops an agent to 0 HP bumps ep_agents_killed_by_boss."""
    env = CombatEnv(config=_cfg(), n_agents=4, seed=0, enable_boss=True)
    env.reset(seed=0)
    boss = env._monsters.all_alive()[0]
    victim = env._agents[1]
    victim.position = boss.position
    victim.health = 0.001
    pre = env._ep_agents_killed_by_boss
    env._global_step_count = 10**9
    env.step(Action.TRAIN.value)
    assert env._ep_agents_killed_by_boss > pre


def test_boss_permadeath_no_action_after_death() -> None:
    """A defeated boss must not act in subsequent ticks."""
    env = CombatEnv(config=_cfg(), n_agents=4, seed=0, enable_boss=True)
    env.reset(seed=0)
    boss = env._monsters.all_alive()[0]
    boss.take_damage(BOSS_BASE_HEALTH + 1.0, "agent_x")
    assert not boss.alive
    pre_pos = boss.position
    events = env._monsters.tick_all(env._world, env._agents, env._rng)
    assert events == []
    assert boss.position == pre_pos
    assert env._monsters.all_alive() == []


def test_boss_does_not_respawn_until_reset() -> None:
    """A defeated boss stays gone within the episode; reset spawns a fresh one."""
    env = CombatEnv(config=_cfg(), n_agents=4, seed=0, enable_boss=True)
    env.reset(seed=0)
    boss = env._monsters.all_alive()[0]
    boss.take_damage(BOSS_BASE_HEALTH + 1.0, "agent_x")
    env._global_step_count = 10**9
    env.step(Action.TRAIN.value)
    env.step(Action.TRAIN.value)
    assert env._monsters.all_alive() == []
    env.reset(seed=0)
    assert len(env._monsters.all_alive()) == 1


def test_heuristic_non_focal_attacks_adjacent_boss() -> None:
    """Non-focal heuristic agents must engage adjacent bosses (load-bearing for loot share)."""
    env = CombatEnv(config=_cfg(), n_agents=4, seed=0, enable_boss=True)
    env.reset(seed=0)
    boss = env._monsters.all_alive()[0]
    non_focal_idx = (env._focal_idx + 1) % env._n_agents
    non_focal = env._agents[non_focal_idx]
    bx, by = boss.position
    non_focal.position = (bx, by)
    non_focal.health = 1.0
    env._agents[env._focal_idx].position = (
        (bx + 10) % env._world.grid_size,
        (by + 10) % env._world.grid_size,
    )
    env._global_step_count = 10**9
    pre_attackers = set(boss.attackers)
    env.step(Action.TRAIN.value)
    new_attackers = boss.attackers - pre_attackers
    assert non_focal.agent_id in new_attackers, (
        f"Non-focal heuristic did not attack adjacent boss; attackers={boss.attackers}"
    )


def test_action_mask_allows_attack_when_only_monster_adjacent() -> None:
    """ATTACK must be unmasked when only a boss (not an agent) is adjacent."""
    env = CombatEnv(config=_cfg(), n_agents=4, seed=0, enable_boss=True)
    env.reset(seed=0)
    boss = env._monsters.all_alive()[0]
    focal = env._agents[env._focal_idx]
    focal.position = boss.position
    for i, a in enumerate(env._agents):
        if i != env._focal_idx:
            a.position = (
                (boss.position[0] + 10) % env._world.grid_size,
                (boss.position[1] + 10) % env._world.grid_size,
            )
    env._global_step_count = 10**9
    mask = env.action_masks()
    assert mask[Action.ATTACK]
    assert not mask[Action.COLLABORATE]
    assert not mask[Action.WALK_AWAY]


def test_end_to_end_loot_withdraw_after_boss_defeat() -> None:
    """Full pipeline: defeat boss -> loot stash drops -> participant WITHDRAWs food."""
    env = CombatEnv(config=_cfg(), n_agents=4, seed=0, enable_boss=True)
    env.reset(seed=0)
    boss = env._monsters.all_alive()[0]
    boss.take_damage(0.5, "agent_a")
    boss.take_damage(0.5, "agent_b")
    boss.take_damage(BOSS_BASE_HEALTH, "agent_c")
    assert not boss.alive
    env._drop_boss_loot(boss)
    agent = env._agents[0]
    agent.agent_id = "agent_a"
    agent.position = boss.position
    pre_food = agent.inventory.food
    ok = env._stash_registry.withdraw(agent)
    assert ok
    assert agent.inventory.food >= pre_food + BOSS_LOOT_FOOD


def test_boss_corner_spawn_seed_deterministic() -> None:
    """Same seed -> same corner."""
    env_a = CombatEnv(config=_cfg(), n_agents=4, seed=42, enable_boss=True)
    env_a.reset(seed=42)
    pos_a = env_a._monsters.all_alive()[0].position

    env_b = CombatEnv(config=_cfg(), n_agents=4, seed=42, enable_boss=True)
    env_b.reset(seed=42)
    pos_b = env_b._monsters.all_alive()[0].position

    assert pos_a == pos_b


def test_unique_attacker_count_no_double_count() -> None:
    """Unique attacker count ignores duplicate hits from the same agent."""
    env = CombatEnv(config=_cfg(), n_agents=4, seed=0, enable_boss=True)
    env.reset(seed=0)
    boss = env._monsters.all_alive()[0]
    boss.take_damage(0.1, "agent_a")
    boss.take_damage(0.1, "agent_b")
    boss.take_damage(0.1, "agent_a")
    unique = sum(len(m.attackers) for m in env._monsters.all() if m.kind == "boss")
    assert unique == 2
