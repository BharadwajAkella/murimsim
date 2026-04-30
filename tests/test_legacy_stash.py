"""v21b — death drop (legacy_stash) with affinity threshold + lockout."""
from __future__ import annotations

import yaml
import pytest

from murimsim.rl.ippo_env import IPPOEnv
from murimsim.rl.multi_env import (
    LEGACY_AFFINITY_THRESHOLD,
    LEGACY_UNLOCK_TICKS,
)


@pytest.fixture
def cfg() -> dict:
    with open("config/default.yaml") as f:
        return yaml.safe_load(f)


def _seed_inventory(agent, food=2, qi=1, materials=1, poison=0) -> None:
    agent.inventory.food = food
    agent.inventory.qi = qi
    agent.inventory.materials = materials
    agent.inventory.poison = poison


def _force_affinity_raw(env, src_idx: int, dst_idx: int, raw_value: float) -> None:
    """Bypass the affinity API to inject a controlled bond at current step."""
    env._affinity_raw.setdefault(src_idx, {})[dst_idx] = (raw_value, env._ep_step_count)


def test_death_drop_creates_legacy_stash_with_full_inventory(cfg: dict) -> None:
    env = IPPOEnv(config=cfg, n_agents=4, seed=21)
    env.reset_all(seed=21)
    deceased = env._agents[0]
    _seed_inventory(deceased, food=3, qi=2, materials=1, poison=1)
    death_pos = deceased.position

    env._drop_inventory(deceased, killer_idx=None)

    legacy_stashes = [
        s for s in env._stash_registry.all_stashes()
        if s.position == death_pos and s.claim_unlock_step >= 0
    ]
    assert len(legacy_stashes) == 1
    s = legacy_stashes[0]
    assert s.food == 3 and s.qi == 2 and s.materials == 1 and s.poison == 1
    # Owner is sentinel — not the deceased's own agent_id (so reincarnation cannot grab it).
    assert s.owner_id != deceased.agent_id
    assert s.owner_id.endswith("__deceased")
    # Inventory wiped on the corpse.
    assert deceased.inventory.total() == 0


def test_high_affinity_ally_becomes_participant(cfg: dict) -> None:
    env = IPPOEnv(config=cfg, n_agents=4, seed=21)
    env.reset_all(seed=21)
    deceased = env._agents[0]
    ally = env._agents[1]
    _seed_inventory(deceased, food=2)
    # Inject normalised affinity = 0.5 (above threshold 0.3) — raw = 0.5 * AFFINITY_NORM (5.0) = 2.5
    _force_affinity_raw(env, 0, 1, 2.5 * env._ep_step_count if False else 2.5)

    env._drop_inventory(deceased, killer_idx=None)
    legacy = [s for s in env._stash_registry.all_stashes() if s.claim_unlock_step >= 0][0]
    assert ally.agent_id in legacy.participants


def test_low_affinity_excluded(cfg: dict) -> None:
    env = IPPOEnv(config=cfg, n_agents=4, seed=21)
    env.reset_all(seed=21)
    deceased = env._agents[0]
    ally = env._agents[1]
    _seed_inventory(deceased, food=2)
    # Raw affinity 0.5 → normalised 0.1 < threshold 0.3 → excluded.
    _force_affinity_raw(env, 0, 1, 0.5)

    env._drop_inventory(deceased, killer_idx=None)
    legacy = [s for s in env._stash_registry.all_stashes() if s.claim_unlock_step >= 0][0]
    assert ally.agent_id not in legacy.participants


def test_killer_never_inherits(cfg: dict) -> None:
    env = IPPOEnv(config=cfg, n_agents=4, seed=21)
    env.reset_all(seed=21)
    deceased = env._agents[0]
    killer = env._agents[1]
    _seed_inventory(deceased, food=5)
    # Even if affinity is sky-high, killer is excluded.
    _force_affinity_raw(env, 0, 1, 100.0)

    env._drop_inventory(deceased, killer_idx=1)
    legacy = [s for s in env._stash_registry.all_stashes() if s.claim_unlock_step >= 0][0]
    assert killer.agent_id not in legacy.participants


def test_empty_inventory_creates_no_stash(cfg: dict) -> None:
    env = IPPOEnv(config=cfg, n_agents=4, seed=21)
    env.reset_all(seed=21)
    deceased = env._agents[0]
    deceased.inventory.food = 0
    deceased.inventory.qi = 0
    deceased.inventory.materials = 0
    deceased.inventory.poison = 0

    pre_count = len(env._stash_registry.all_stashes())
    env._drop_inventory(deceased)
    assert len(env._stash_registry.all_stashes()) == pre_count


def test_legacy_stash_locked_for_non_participants_initially(cfg: dict) -> None:
    env = IPPOEnv(config=cfg, n_agents=4, seed=21)
    env.reset_all(seed=21)
    deceased = env._agents[0]
    outsider = env._agents[2]
    _seed_inventory(deceased, food=4)
    # No affinity to anyone → no participants
    env._drop_inventory(deceased)
    legacy = [s for s in env._stash_registry.all_stashes() if s.claim_unlock_step >= 0][0]
    assert legacy.participants == []
    # Outsider cannot access yet.
    assert not legacy.is_accessible_to(outsider.agent_id, current_step=env._ep_step_count)


def test_legacy_stash_unlocks_after_lockout(cfg: dict) -> None:
    env = IPPOEnv(config=cfg, n_agents=4, seed=21)
    env.reset_all(seed=21)
    deceased = env._agents[0]
    outsider = env._agents[2]
    _seed_inventory(deceased, food=4)
    drop_step = env._ep_step_count
    env._drop_inventory(deceased)
    legacy = [s for s in env._stash_registry.all_stashes() if s.claim_unlock_step >= 0][0]

    # Just before unlock — still locked.
    assert not legacy.is_accessible_to(outsider.agent_id, current_step=drop_step + LEGACY_UNLOCK_TICKS - 1)
    # At unlock — accessible.
    assert legacy.is_accessible_to(outsider.agent_id, current_step=drop_step + LEGACY_UNLOCK_TICKS)
    # After unlock — accessible.
    assert legacy.is_accessible_to(outsider.agent_id, current_step=drop_step + LEGACY_UNLOCK_TICKS + 100)


def test_participant_can_access_immediately(cfg: dict) -> None:
    env = IPPOEnv(config=cfg, n_agents=4, seed=21)
    env.reset_all(seed=21)
    deceased = env._agents[0]
    ally = env._agents[1]
    _seed_inventory(deceased, food=2)
    _force_affinity_raw(env, 0, 1, 2.5)
    env._drop_inventory(deceased)
    legacy = [s for s in env._stash_registry.all_stashes() if s.claim_unlock_step >= 0][0]
    # Ally can access immediately, even without current_step (or with the actual one).
    assert legacy.is_accessible_to(ally.agent_id)
    assert legacy.is_accessible_to(ally.agent_id, current_step=env._ep_step_count)
