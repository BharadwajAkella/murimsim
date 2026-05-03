"""tests/test_gift.py — Phase 8c: GIFT social action.

Covers:
  * ResourceConfig.value loaded from YAML.
  * GIFT mask eligibility (no inventory, no neighbour, dead, etc.).
  * GIFT picks the highest-value owned resource.
  * GIFT picks the highest-affinity adjacent recipient.
  * Inventory transfer is exactly 1 unit, conserved.
  * Affinity bumps are asymmetric and non-zero on both sides.
  * Reward credit applied to both giver and receiver, value-scaled receiver.
  * Episode counters increment.
  * GIFT is independent of the courtship gate.
  * Determinism: byte-identical episode rollout under fixed seed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from murimsim.actions import BodyAction, N_BODY_ACTIONS, N_SOCIAL_ACTIONS, SocialAction
from murimsim.rl.multi_env import (
    GIFT_AFFINITY_BUMP_GIVER_TO_RECEIVER,
    GIFT_AFFINITY_BUMP_RECEIVER_TO_GIVER,
    GIFT_RANGE,
    GIFT_REWARD_GIVER,
    GIFT_REWARD_RECEIVER_PER_VALUE,
    CombatEnv,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def _load_cfg(enable_courtship: bool = False) -> dict:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("agent", {})["enable_courtship"] = enable_courtship
    return cfg


def _make_env(enable_courtship: bool = False, seed: int = 0, n_agents: int = 4) -> CombatEnv:
    env = CombatEnv(config=_load_cfg(enable_courtship), n_agents=n_agents, seed=seed)
    env.reset(seed=seed)
    return env


def _place_pair(env: CombatEnv, giver: int = 0, receiver: int = 1) -> None:
    env._agents[giver].position = (5, 5)
    env._agents[receiver].position = (5, 6)
    # Move all other agents far away so they cannot interfere as recipients.
    far = 0
    for k, a in enumerate(env._agents):
        if k in (giver, receiver):
            continue
        env._agents[k].position = (28 + (far % 2), 28)
        far += 1


# ---------------------------------------------------------------------------
# Resource value field
# ---------------------------------------------------------------------------

def test_resource_value_field_loaded():
    """Every resource in default.yaml must expose ``value`` >= 0."""
    env = _make_env(seed=1)
    assert env._world is not None
    for rid, rcfg in env._world.resources.items():
        assert hasattr(rcfg, "value"), f"{rid} missing value"
        assert rcfg.value > 0, f"{rid} value must be positive"
    # Spot-check the canonical ordering.
    assert env._world.resources["food"].value == 1.0
    assert env._world.resources["flame"].value == 3.0
    assert env._world.resources["qi"].value > env._world.resources["food"].value


def test_gift_lives_in_body_lane():
    """Phase 8c.2: GIFT moved from social to body lane (it's a physical action)."""
    assert N_SOCIAL_ACTIONS == 7  # Phase 8d.1: PROPOSE_TRADE/ACCEPT_TRADE/REJECT_TRADE
    assert N_BODY_ACTIONS == 17
    assert int(BodyAction.GIFT) == 16
    # Sanity: SocialAction must not have a GIFT slot any more.
    assert not hasattr(SocialAction, "GIFT")


# ---------------------------------------------------------------------------
# GIFT eligibility / masking — now on the BODY lane
# ---------------------------------------------------------------------------

def test_gift_mask_false_when_inventory_empty():
    env = _make_env(seed=2)
    _place_pair(env)
    env._agents[0].inventory.food = 0
    env._agents[0].inventory.qi = 0
    env._agents[0].inventory.materials = 0
    env._agents[0].inventory.poison = 0
    env._agents[0].inventory.flame = 0
    mask = env.action_masks_body(0)
    assert not bool(mask[BodyAction.GIFT])


def test_gift_mask_false_when_no_adjacent_agent():
    env = _make_env(seed=3)
    env._agents[0].position = (1, 1)
    env._agents[0].inventory.food = 5
    for k in range(1, env._n_agents):
        env._agents[k].position = (28, 28)
    mask = env.action_masks_body(0)
    assert not bool(mask[BodyAction.GIFT])


def test_gift_mask_true_when_eligible():
    env = _make_env(seed=4)
    _place_pair(env)
    env._agents[0].inventory.food = 1
    mask = env.action_masks_body(0)
    assert bool(mask[BodyAction.GIFT])


def test_gift_mask_false_when_dead():
    env = _make_env(seed=5)
    _place_pair(env)
    env._agents[0].inventory.food = 5
    env._agents[0].alive = False
    env._agents[0].health = 0.0
    mask = env.action_masks_body(0)
    assert not bool(mask[BodyAction.GIFT])


# ---------------------------------------------------------------------------
# Resource selection — highest value wins
# ---------------------------------------------------------------------------

def test_gift_picks_highest_value_resource():
    env = _make_env(seed=6)
    _place_pair(env)
    inv = env._agents[0].inventory
    inv.food = 3       # value 1.0
    inv.qi = 1         # value 2.0
    inv.flame = 2      # value 3.0 (should win)
    rewards = np.zeros(env._n_agents, dtype=np.float32)
    assert env._resolve_gift(0, rewards=rewards) is True
    assert env._agents[0].inventory.flame == 1
    assert env._agents[1].inventory.flame == 1
    # Other counters untouched.
    assert env._agents[0].inventory.food == 3
    assert env._agents[0].inventory.qi == 1


def test_gift_picks_only_owned_resource():
    """If only food is in inventory, food is gifted regardless of higher-value
    resources existing in the world."""
    env = _make_env(seed=7)
    _place_pair(env)
    inv = env._agents[0].inventory
    inv.food = 1
    inv.qi = 0
    inv.flame = 0
    rewards = np.zeros(env._n_agents, dtype=np.float32)
    assert env._resolve_gift(0, rewards=rewards) is True
    assert env._agents[0].inventory.food == 0
    assert env._agents[1].inventory.food == 1


# ---------------------------------------------------------------------------
# Recipient selection — highest affinity wins
# ---------------------------------------------------------------------------

def test_gift_picks_highest_affinity_recipient():
    env = _make_env(seed=8, n_agents=5)
    env._agents[0].position = (5, 5)
    env._agents[1].position = (5, 6)  # adjacent
    env._agents[2].position = (6, 5)  # adjacent
    env._agents[3].position = (28, 28)  # far
    env._agents[4].position = (29, 29)  # far
    env._agents[0].inventory.food = 1
    # Pre-seed affinity: agent 0 strongly prefers agent 2.
    env._record_affinity_event(0, 2, 1.0, 0.0)
    rewards = np.zeros(env._n_agents, dtype=np.float32)
    assert env._resolve_gift(0, rewards=rewards) is True
    assert env._agents[2].inventory.food == 1
    assert env._agents[1].inventory.food == 0


# ---------------------------------------------------------------------------
# Conservation, rewards, affinity bump
# ---------------------------------------------------------------------------

def test_gift_conserves_inventory():
    env = _make_env(seed=9)
    _place_pair(env)
    env._agents[0].inventory.qi = 3
    before = env._agents[0].inventory.total() + env._agents[1].inventory.total()
    rewards = np.zeros(env._n_agents, dtype=np.float32)
    assert env._resolve_gift(0, rewards=rewards) is True
    after = env._agents[0].inventory.total() + env._agents[1].inventory.total()
    assert before == after


def test_gift_no_scalar_reward_credit():
    """v27.1: GIFT no longer awards scalar reward to either side. The carrot
    is the affinity bump (with diminishing returns) and downstream resource
    utility / carry-penalty change."""
    env = _make_env(seed=10)
    _place_pair(env)
    env._agents[0].inventory.flame = 1   # value 3.0
    rewards = np.zeros(env._n_agents, dtype=np.float32)
    ok = env._resolve_gift(0, rewards=rewards)
    assert ok
    assert rewards[0] == 0.0
    assert rewards[1] == 0.0
    # Sanity-check the constants are zero (guard against accidental re-introduction).
    assert GIFT_REWARD_GIVER == 0.0
    assert GIFT_REWARD_RECEIVER_PER_VALUE == 0.0


def test_gift_affinity_bumps_exceed_collaborate():
    """GIFT must form stronger bonds than COLLABORATE (otherwise it would
    never be the dominant relationship-builder)."""
    from murimsim.rl.multi_env import AFFINITY_COLLAB_BOTH
    assert GIFT_AFFINITY_BUMP_GIVER_TO_RECEIVER > AFFINITY_COLLAB_BOTH
    assert GIFT_AFFINITY_BUMP_RECEIVER_TO_GIVER > AFFINITY_COLLAB_BOTH


def test_gift_affinity_bump_asymmetric():
    env = _make_env(seed=11)
    _place_pair(env)
    env._agents[0].inventory.food = 1
    aff_g_before = env._affinity(0, 1)
    aff_r_before = env._affinity(1, 0)
    rewards = np.zeros(env._n_agents, dtype=np.float32)
    env._resolve_gift(0, rewards=rewards)
    aff_g_after = env._affinity(0, 1)
    aff_r_after = env._affinity(1, 0)
    assert aff_g_after > aff_g_before
    assert aff_r_after > aff_r_before
    # Giver bump should be larger by design (commitment > gratitude per gift).
    assert (aff_g_after - aff_g_before) > (aff_r_after - aff_r_before)


def test_gift_diminishing_returns_on_receiver():
    """Receiver affinity bump shrinks as their stock of the resource grows."""
    env = _make_env(seed=110)
    _place_pair(env)
    # Stock the giver with plenty so we can gift repeatedly. Receiver starts empty.
    env._agents[0].inventory.food = 5
    rewards = np.zeros(env._n_agents, dtype=np.float32)

    # First gift: receiver had 0 → full bump.
    aff0 = env._affinity(1, 0)
    env._resolve_gift(0, rewards=rewards)
    aff1 = env._affinity(1, 0)
    bump1 = aff1 - aff0

    # Second gift: receiver had 1 → diminished bump.
    env._resolve_gift(0, rewards=rewards)
    aff2 = env._affinity(1, 0)
    bump2 = aff2 - aff1

    # Third gift: receiver had 2 → further diminished.
    env._resolve_gift(0, rewards=rewards)
    aff3 = env._affinity(1, 0)
    bump3 = aff3 - aff2

    assert bump1 > bump2 > bump3, f"expected strictly decreasing bumps: {bump1}, {bump2}, {bump3}"


def test_gift_sacrifice_amplifies_giver_bump():
    """Giver-side bump is larger when the giver has less stock left after gifting."""
    # Run A: giver gives last unit (qty after = 0 → factor 1.0)
    env_a = _make_env(seed=120)
    _place_pair(env_a)
    env_a._agents[0].inventory.food = 1
    aff_a_before = env_a._affinity(0, 1)
    rewards = np.zeros(env_a._n_agents, dtype=np.float32)
    env_a._resolve_gift(0, rewards=rewards)
    bump_last = env_a._affinity(0, 1) - aff_a_before

    # Run B: giver has plenty (qty after = 9 → factor 1/(1+0.5*9) ≈ 0.18)
    env_b = _make_env(seed=121)
    _place_pair(env_b)
    env_b._agents[0].inventory.food = 10
    aff_b_before = env_b._affinity(0, 1)
    rewards = np.zeros(env_b._n_agents, dtype=np.float32)
    env_b._resolve_gift(0, rewards=rewards)
    bump_plenty = env_b._affinity(0, 1) - aff_b_before

    assert bump_last > bump_plenty, (
        f"expected giving-last to bump more than giving-from-plenty, got {bump_last} vs {bump_plenty}"
    )


def test_gift_episode_counters_increment():
    env = _make_env(seed=12)
    _place_pair(env)
    env._agents[0].inventory.qi = 1   # value 2.0
    rewards = np.zeros(env._n_agents, dtype=np.float32)
    assert env._ep_gifts_made == 0
    env._resolve_gift(0, rewards=rewards)
    assert env._ep_gifts_made == 1
    assert env._ep_gift_value_transferred == env._world.resources["qi"].value


def test_gift_fails_silently_when_no_recipient():
    env = _make_env(seed=13)
    env._agents[0].position = (1, 1)
    env._agents[0].inventory.food = 5
    for k in range(1, env._n_agents):
        env._agents[k].position = (28, 28)
    rewards = np.zeros(env._n_agents, dtype=np.float32)
    assert env._resolve_gift(0, rewards=rewards) is False
    assert env._agents[0].inventory.food == 5
    assert env._ep_gifts_made == 0


def test_gift_fails_silently_when_inventory_empty():
    env = _make_env(seed=14)
    _place_pair(env)
    env._agents[0].inventory.food = 0
    env._agents[0].inventory.qi = 0
    env._agents[0].inventory.materials = 0
    env._agents[0].inventory.poison = 0
    env._agents[0].inventory.flame = 0
    rewards = np.zeros(env._n_agents, dtype=np.float32)
    assert env._resolve_gift(0, rewards=rewards) is False


# ---------------------------------------------------------------------------
# Independence from courtship gate
# ---------------------------------------------------------------------------

def test_gift_works_with_courtship_disabled():
    env = _make_env(enable_courtship=False, seed=15)
    _place_pair(env)
    env._agents[0].inventory.food = 1
    body_mask = env.action_masks_body(0)
    social_mask = env.action_masks_social(0)
    assert bool(body_mask[BodyAction.GIFT])
    # PROPOSE/ACCEPT must remain masked-out in courtship-disabled mode.
    assert not bool(social_mask[SocialAction.PROPOSE])
    assert not bool(social_mask[SocialAction.ACCEPT])
    rewards = np.zeros(env._n_agents, dtype=np.float32)
    assert env._resolve_gift(0, rewards=rewards) is True


def test_gift_works_with_courtship_enabled():
    env = _make_env(enable_courtship=True, seed=16)
    _place_pair(env)
    env._agents[0].inventory.food = 1
    body_mask = env.action_masks_body(0)
    assert bool(body_mask[BodyAction.GIFT])


# ---------------------------------------------------------------------------
# End-to-end via step_joint_all (via ippo_env path)
# ---------------------------------------------------------------------------

def test_gift_via_step_joint_all():
    """GIFT dispatched through the joint-action body lane transfers and
    no longer competes with social signals like COLLABORATE."""
    from murimsim.rl.ippo_env import IPPOEnv
    env = IPPOEnv(config=_load_cfg(False), n_agents=4, seed=17)
    env.reset(seed=17)
    _place_pair(env)
    env._agents[0].inventory.flame = 1
    body = np.full(env._n_agents, int(BodyAction.REST), dtype=np.int64)
    social = np.full(env._n_agents, int(SocialAction.NOOP), dtype=np.int64)
    body[0] = int(BodyAction.GIFT)
    obs, rewards, term, trunc, info = env.step_all_joint(body, social)
    assert env._agents[1].inventory.flame == 1
    assert env._agents[0].inventory.flame == 0
    assert env._ep_gifts_made == 1


def test_gift_and_collaborate_can_coexist_same_tick():
    """Phase 8c.2: with GIFT in body lane and COLLABORATE in social lane,
    an agent can do both physical (gift) and emotional (collab) in one tick."""
    from murimsim.rl.ippo_env import IPPOEnv
    env = IPPOEnv(config=_load_cfg(False), n_agents=4, seed=170)
    env.reset(seed=170)
    _place_pair(env)
    env._agents[0].inventory.food = 1
    # Phase 8c.3: bilateral COLLAB — both agents must pick COLLABORATE for
    # the group to form. The heuristic sociability gate no longer applies
    # in the joint path.
    body = np.full(env._n_agents, int(BodyAction.REST), dtype=np.int64)
    social = np.full(env._n_agents, int(SocialAction.NOOP), dtype=np.int64)
    body[0] = int(BodyAction.GIFT)
    social[0] = int(SocialAction.COLLABORATE)
    social[1] = int(SocialAction.COLLABORATE)
    env.step_all_joint(body, social)
    # Both effects must register: gift transferred AND group formed.
    assert env._agents[1].inventory.food >= 1
    assert env._ep_gifts_made == 1
    # Check a group was formed (either agent should be in one now).
    assert env._get_group(0) is not None or env._get_group(1) is not None


# ---------------------------------------------------------------------------
# Determinism — same seed → same gift outcomes
# ---------------------------------------------------------------------------

def test_gift_determinism_across_runs():
    """Two independent runs with the same seed must produce identical
    gift-related state after a fixed sequence."""
    def run(seed: int) -> tuple[int, float, list[tuple[int, int, int, int, int]]]:
        env = _make_env(enable_courtship=False, seed=seed)
        _place_pair(env)
        env._agents[0].inventory.flame = 2
        env._agents[0].inventory.qi = 2
        rewards = np.zeros(env._n_agents, dtype=np.float32)
        env._resolve_gift(0, rewards=rewards)
        env._resolve_gift(0, rewards=rewards)
        invs = [
            (
                a.inventory.food,
                a.inventory.qi,
                a.inventory.materials,
                a.inventory.poison,
                a.inventory.flame,
            )
            for a in env._agents
        ]
        return env._ep_gifts_made, float(env._ep_gift_value_transferred), invs

    a = run(42)
    b = run(42)
    assert a == b
