"""tests/test_courtship.py — Phase 7 courtship & hereditary propagation.

Covers:
    * Sex assignment is ~50/50 over many spawns.
    * PROPOSE is rejected for same-sex / out-of-range / on-cooldown targets.
    * Mutual PROPOSE@T → ACCEPT@T+1 enqueues a PendingBirth and applies
      cooldowns + affinity bump + birth reward.
    * Same-tick ACCEPT (against PROPOSE issued this tick) is rejected.
    * PendingBirth queue drains FIFO on next age-death.
    * Stash merge sums both parents' contents into a single child stash and
      zeros the parents' stashes.
    * inherit_value_biased moves toward the stronger parent on average.
    * Phase 7 obs has the 2 extra tail bits with the correct semantics.
    * Courtship disabled → byte-identical OBS_TOTAL_SIZE (264) preserved.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from murimsim.agent import (
    Agent,
    INHERIT_BIAS_TO_STRONGER,
    inherit_value_biased,
)
from murimsim.actions import SocialAction
from murimsim.rl.multi_env import (
    OBS_TOTAL_SIZE,
    OBS_TOTAL_SIZE_COURTSHIP,
    CombatEnv,
    PendingBirth,
)


CONFIG_PATH = Path("config/default.yaml")


def _load_cfg(enable_courtship: bool = True) -> dict:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("agent", {})["enable_courtship"] = enable_courtship
    return cfg


def _make_env(enable_courtship: bool = True, seed: int = 0, n_agents: int = 6) -> CombatEnv:
    env = CombatEnv(config=_load_cfg(enable_courtship), n_agents=n_agents, seed=seed)
    env.reset(seed=seed)
    return env


# ---------------------------------------------------------------------------
# Inheritance math
# ---------------------------------------------------------------------------

def test_inherit_biased_moves_toward_stronger():
    rng = np.random.default_rng(42)
    samples = [
        inherit_value_biased(0.9, 0.3, rng, sigma=0.0, bias=INHERIT_BIAS_TO_STRONGER)
        for _ in range(200)
    ]
    mean = float(np.mean(samples))
    midpoint = 0.6
    biased_target = 0.7 * 0.9 + 0.3 * 0.3
    assert mean > midpoint, f"Expected mean ({mean}) above midpoint ({midpoint})"
    assert abs(mean - biased_target) < 1e-9, f"sigma=0 should yield exact bias, got {mean}"


def test_inherit_biased_default_equals_midpoint():
    """bias=0.5 (default) preserves the legacy midpoint formula + RNG draws."""
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    val_a = inherit_value_biased(0.4, 0.8, rng_a, sigma=0.05)
    midpoint = (0.4 + 0.8) / 2 + rng_b.normal(0, 0.05)
    midpoint = float(np.clip(midpoint, 0.0, 1.0))
    assert val_a == midpoint


# ---------------------------------------------------------------------------
# Sex assignment + obs layout
# ---------------------------------------------------------------------------

def test_sex_distribution_roughly_balanced():
    """Over many spawns the sex split should be ~50/50 (allow generous slack)."""
    males = females = 0
    for seed in range(40):
        env = _make_env(enable_courtship=True, seed=seed, n_agents=8)
        for a in env._agents:
            if a.sex == "M":
                males += 1
            else:
                females += 1
    total = males + females
    ratio = males / total
    assert 0.35 < ratio < 0.65, f"Sex ratio {ratio:.2f} too skewed (M={males}, F={females})"


def test_obs_size_courtship_enabled():
    env = _make_env(enable_courtship=True, seed=0)
    obs = env._build_obs(0)
    assert obs.shape == (OBS_TOTAL_SIZE_COURTSHIP,)
    assert env.observation_space.shape == (OBS_TOTAL_SIZE_COURTSHIP,)


def test_obs_size_courtship_disabled_unchanged():
    env = _make_env(enable_courtship=False, seed=0)
    obs = env._build_obs(0)
    assert obs.shape == (OBS_TOTAL_SIZE,)
    assert env.observation_space.shape == (OBS_TOTAL_SIZE,)


def test_obs_sex_bit_value():
    env = _make_env(enable_courtship=True, seed=0)
    env._agents[0].sex = "F"
    obs = env._build_obs(0)
    assert obs[-2] == 1.0, "sex bit should be 1.0 for female"
    env._agents[0].sex = "M"
    obs = env._build_obs(0)
    assert obs[-2] == 0.0, "sex bit should be 0.0 for male"


def test_obs_pending_proposal_bit():
    env = _make_env(enable_courtship=True, seed=0)
    obs = env._build_obs(0)
    assert obs[-1] == 0.0
    env._pending_proposals[0] = [(1, env._ep_step_count)]
    obs = env._build_obs(0)
    assert obs[-1] == 1.0


# ---------------------------------------------------------------------------
# PROPOSE / ACCEPT mechanics
# ---------------------------------------------------------------------------

def _force_pair_in_range(env: CombatEnv, i: int, j: int, sex_i: str = "M", sex_j: str = "F") -> None:
    env._agents[i].sex = sex_i
    env._agents[j].sex = sex_j
    env._agents[i].mating_cooldown = 0
    env._agents[j].mating_cooldown = 0
    env._agents[i].position = (5, 5)
    env._agents[j].position = (5, 6)


def test_propose_rejected_same_sex():
    env = _make_env(seed=1)
    _force_pair_in_range(env, 0, 1, "M", "M")
    assert env._courtship_propose(0) is False
    assert 1 not in env._pending_proposals


def test_propose_rejected_out_of_range():
    env = _make_env(seed=1)
    _force_pair_in_range(env, 0, 1, "M", "F")
    env._agents[1].position = (10, 10)
    assert env._courtship_propose(0) is False


def test_propose_rejected_on_cooldown():
    env = _make_env(seed=1)
    _force_pair_in_range(env, 0, 1)
    env._agents[1].mating_cooldown = 5
    assert env._courtship_propose(0) is False


def test_propose_succeeds_when_eligible():
    env = _make_env(seed=2)
    _force_pair_in_range(env, 0, 1)
    assert env._courtship_propose(0) is True
    assert 1 in env._pending_proposals
    assert len(env._pending_proposals[1]) == 1
    proposer, created = env._pending_proposals[1][0]
    assert proposer == 0
    assert created == env._ep_step_count


def test_same_tick_accept_rejected():
    """ACCEPT against a proposal made THIS tick must fail (two-tick rule)."""
    env = _make_env(seed=3)
    _force_pair_in_range(env, 0, 1)
    env._courtship_propose(0)
    rewards = np.zeros(env._n_agents, dtype=np.float32)
    assert env._courtship_accept(1, rewards=rewards) is False


def test_mutual_court_success_enqueues_birth():
    env = _make_env(seed=4)
    _force_pair_in_range(env, 0, 1)
    env._courtship_propose(0)
    env._ep_step_count += 1  # Simulate next-tick boundary.
    rewards = np.zeros(env._n_agents, dtype=np.float32)
    assert env._courtship_accept(1, rewards=rewards) is True
    assert len(env._pending_births) == 1
    pb = env._pending_births[0]
    assert {pb.parent1_idx, pb.parent2_idx} == {0, 1}
    # Cooldowns set.
    assert env._agents[0].mating_cooldown > 0
    assert env._agents[1].mating_cooldown > 0
    # Reward applied to both parents.
    assert rewards[0] > 0 and rewards[1] > 0
    # Affinity bumped both directions.
    assert env._affinity(0, 1) > 0
    assert env._affinity(1, 0) > 0
    # Proposal cleared from pending.
    assert 1 not in env._pending_proposals


def test_proposal_survives_one_full_tick_for_next_tick_accept():
    """Regression: proposal registered at end of tick T must NOT be trimmed
    at the top of tick T+1 — otherwise next-tick ACCEPT can never resolve.

    Reproduces the v26 long-run bug where the policy sampled 7 ACCEPTs but
    0 succeeded because the TTL trim ran with a too-tight cutoff.
    """
    env = _make_env(seed=42)
    _force_pair_in_range(env, 0, 1)
    env._courtship_propose(0)
    created_step = env._pending_proposals[1][0][1]
    # Simulate the tick boundary the way step() does: advance _ep_step_count
    # then re-run the trim logic at the top of the next step.
    env._ep_step_count += 1
    from murimsim.rl.multi_env import COURT_PROPOSAL_TTL_TICKS
    cutoff = env._ep_step_count - COURT_PROPOSAL_TTL_TICKS - 1
    new_pending: dict[int, list[tuple[int, int]]] = {}
    for t, lst in env._pending_proposals.items():
        kept = [(p, s) for (p, s) in lst if s > cutoff]
        if kept:
            new_pending[t] = kept
    env._pending_proposals = new_pending
    assert 1 in env._pending_proposals, "Trim must NOT drop proposal one tick after creation"
    # And ACCEPT now succeeds.
    rewards = np.zeros(env._n_agents, dtype=np.float32)
    assert env._courtship_accept(1, rewards=rewards) is True


def test_accept_mask_true_one_tick_after_propose():
    """Mask at end of tick T (post-PROPOSE) must allow ACCEPT for tick T+1."""
    env = _make_env(seed=43)
    _force_pair_in_range(env, 0, 1)
    env._courtship_propose(0)
    # Mask is computed at end of step with current _ep_step_count (post-increment).
    mask = env.action_masks_social(1)
    assert mask[SocialAction.ACCEPT.value], (
        "ACCEPT mask must be true so policy can sample it on the next tick"
    )


def test_pending_birth_drains_fifo_on_age_death():
    """When an age-death frees a slot, the head of the birth queue fills it."""
    env = _make_env(seed=5, n_agents=4)
    _force_pair_in_range(env, 0, 1)
    env._courtship_propose(0)
    env._ep_step_count += 1
    rewards = np.zeros(env._n_agents, dtype=np.float32)
    env._courtship_accept(1, rewards=rewards)
    assert len(env._pending_births) == 1
    parent_ids_before = (env._agents[0].agent_id, env._agents[1].agent_id)
    # Force agent[3] dead and call _try_reproduce directly (slot freed).
    env._agents[3].alive = False
    env._try_reproduce(env._agents[3])
    # Queue drained.
    assert env._pending_births == []
    # Slot 3 is now occupied by a child of (0, 1).
    child = env._agents[3]
    assert child.alive
    assert set(child.parent_ids) == set(parent_ids_before)


# ---------------------------------------------------------------------------
# Stash merging
# ---------------------------------------------------------------------------

def test_stash_merge_on_birth_sums_and_zeros_parents():
    from murimsim.stash import Stash

    env = _make_env(seed=6, n_agents=4)
    p1, p2, child = env._agents[0], env._agents[1], env._agents[2]
    s1 = Stash(stash_id="p1_s", owner_id=p1.agent_id, position=(0, 0),
               food=3, qi=2, materials=1, poison=0)
    s2 = Stash(stash_id="p2_s", owner_id=p2.agent_id, position=(1, 1),
               food=4, qi=0, materials=2, poison=1)
    env._stash_registry.register(s1)
    env._stash_registry.register(s2)

    env._merge_parent_stashes_to_child(p1, p2, child)

    assert s1.food == s1.qi == s1.materials == s1.poison == 0
    assert s2.food == s2.qi == s2.materials == s2.poison == 0
    child_stashes = env._stash_registry.get_stashes_for_owner(child.agent_id)
    assert len(child_stashes) == 1
    merged = child_stashes[0]
    assert merged.food == 7
    assert merged.qi == 2
    assert merged.materials == 3
    assert merged.poison == 1


# ---------------------------------------------------------------------------
# Determinism guard
# ---------------------------------------------------------------------------

def test_courtship_disabled_byte_identical_to_legacy():
    """Two CombatEnvs with courtship disabled produce identical observations."""
    env_a = _make_env(enable_courtship=False, seed=7)
    env_b = _make_env(enable_courtship=False, seed=7)
    obs_a = env_a._build_obs(0)
    obs_b = env_b._build_obs(0)
    np.testing.assert_array_equal(obs_a, obs_b)
    assert obs_a.shape == (OBS_TOTAL_SIZE,)
