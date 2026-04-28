"""Tests for Phase 3c: combat environment.

Validates the combat mechanics, curriculum schedule, and that all
Phase 1–2 tests still pass (gate enforcement).
"""
from __future__ import annotations

import numpy as np
import yaml
from pathlib import Path

from murimsim.rl.multi_env import (
    CombatEnv,
    COMBAT_ATTACKER_SCALE,
    COMBAT_MAX_DAMAGE,
    CURRICULUM_START_PROB,
    CURRICULUM_RAMP_STEPS,
    STRIKE_BASIC,
    STRIKE_QI,
    STRIKE_BURST,
    ACTION_TO_STRIKE,
)
from murimsim.actions import Action
from murimsim.agent import Agent

CONFIG_PATH = Path("config/default.yaml")


def _load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _make_combat_env(n_agents: int = 5, seed: int = 0, ramp_steps: int = 300_000) -> CombatEnv:
    return CombatEnv(config=_load_cfg(), n_agents=n_agents, seed=seed, curriculum_ramp_steps=ramp_steps)


def _make_agent(strength: float, health: float = 1.0, pos: tuple = (0, 0)) -> Agent:
    """Build an Agent with controlled strength for testing."""
    rng = np.random.default_rng(0)
    a = Agent.spawn("test", pos, rng, _load_cfg())
    a.strength = strength
    a.health = health
    return a


# ---------------------------------------------------------------------------
# Test 1: combat_damage formula (no defending)
# ---------------------------------------------------------------------------

def test_combat_damage_no_defend() -> None:
    """damage = √(attacker.effective_strength) × COMBAT_ATTACKER_SCALE when not defending (v18)."""
    env = _make_combat_env()
    env.reset(seed=0)

    attacker = _make_agent(strength=0.8)
    defender = _make_agent(strength=0.4)
    damage = env._combat_damage(attacker, defender, is_defending=False)

    expected = float(np.sqrt(attacker.effective_strength)) * COMBAT_ATTACKER_SCALE
    assert abs(damage - expected) < 1e-6, f"Expected {expected:.4f}, got {damage:.4f}"
    assert 0.0 <= damage <= COMBAT_MAX_DAMAGE


# ---------------------------------------------------------------------------
# Test 2: DEFEND is multiplicative — blocks proportional to defense_power
# ---------------------------------------------------------------------------

def test_combat_damage_defending_is_multiplicative() -> None:
    """DEFEND multiplies damage by (1 - defense_power): higher skill blocks more."""
    env = _make_combat_env()
    env.reset(seed=0)

    attacker = _make_agent(strength=0.8)
    defender = _make_agent(strength=0.5)

    dmg_no_defend = env._combat_damage(attacker, defender, is_defending=False)
    dmg_defending = env._combat_damage(attacker, defender, is_defending=True)

    assert dmg_defending < dmg_no_defend, (
        f"DEFEND must reduce damage: defending={dmg_defending:.4f} vs no-defend={dmg_no_defend:.4f}"
    )
    expected_defended = dmg_no_defend * max(0.0, 1.0 - defender.defense_power)
    assert abs(dmg_defending - expected_defended) < 1e-6, (
        f"Expected multiplicative reduction {expected_defended:.4f}, got {dmg_defending:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 3: master cultivator (defense_power → 1.0) nullifies attack completely
# ---------------------------------------------------------------------------

def test_defend_nullification_at_max_defense_power() -> None:
    """An agent with defense_power = 1.0 takes 0 damage while defending."""
    env = _make_combat_env()
    env.reset(seed=0)

    attacker = _make_agent(strength=1.0)
    defender = _make_agent(strength=1.0)
    for key in defender.resistances:
        defender.resistances[key] = 1.0

    dp = defender.defense_power
    assert dp > 0.99, f"Setup error: defense_power should be ~1.0, got {dp:.4f}"

    damage = env._combat_damage(attacker, defender, is_defending=True)
    assert damage < 1e-6, (
        f"Master defender (defense_power={dp:.3f}) should nullify attack, got damage={damage:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 4: ATTACK actually reduces target health
# ---------------------------------------------------------------------------

def test_attack_reduces_target_health() -> None:
    """_do_attack must reduce the target's health by the computed damage amount."""
    env = _make_combat_env(n_agents=2, seed=3)
    env.reset(seed=3)

    focal = env._agents[env._focal_idx]
    other_idx = next(i for i in range(2) if i != env._focal_idx)
    target = env._agents[other_idx]

    focal.position = (5, 5)
    target.position = (6, 5)
    focal.strength = 0.8
    target.health = 1.0

    health_before = target.health
    damage, _ = env._do_attack(focal)

    assert damage > 0.0, "Expected non-zero damage from adjacent attack"
    assert abs(target.health - (health_before - damage)) < 1e-6, (
        f"Target health should be {health_before - damage:.4f}, got {target.health:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 5: ATTACK with no adjacent agent is redirected to REST
# ---------------------------------------------------------------------------

def test_attack_no_adjacent_redirected_to_rest() -> None:
    """ATTACK with no adjacent agent must redirect to REST (no combat damage on anyone)."""
    env = _make_combat_env(n_agents=2, seed=5)
    env.reset(seed=5)

    focal = env._agents[env._focal_idx]
    other_idx = next(i for i in range(2) if i != env._focal_idx)
    other = env._agents[other_idx]

    focal.position = (0, 0)
    other.position = (9, 9)

    health_before = {i: env._agents[i].health for i in range(2)}
    env._global_step_count = CURRICULUM_RAMP_STEPS  # ensure combat_prob = 1.0
    env.step(Action.ATTACK.value)

    for i, agent in enumerate(env._agents):
        if agent.alive:
            delta = health_before[i] - agent.health
            # Starvation drain is small (<0.05). Combat damage would be much larger.
            assert delta < 0.05, (
                f"Agent {i} health dropped {delta:.4f} — looks like combat damage; expected REST"
            )


# ---------------------------------------------------------------------------
# Test 6: DEFEND + step — focal takes less damage than REST
# ---------------------------------------------------------------------------

def test_defend_reduces_damage_in_step() -> None:
    """In a full step(), DEFEND causes focal to take less damage than REST."""
    env_defend = _make_combat_env(n_agents=2, seed=11)
    env_rest   = _make_combat_env(n_agents=2, seed=11)
    env_defend.reset(seed=11)
    env_rest.reset(seed=11)

    for env in (env_defend, env_rest):
        focal = env._agents[env._focal_idx]
        other_idx = next(i for i in range(2) if i != env._focal_idx)
        other = env._agents[other_idx]
        focal.position = (5, 5)
        other.position = (6, 5)
        focal.strength = 0.2
        other.strength = 0.9
        other.sociability = 0.0  # ensure heuristic attacks
        env._global_step_count = CURRICULUM_RAMP_STEPS

    health_defend_before = env_defend._agents[env_defend._focal_idx].health
    health_rest_before   = env_rest._agents[env_rest._focal_idx].health

    env_defend.step(Action.DEFEND.value)
    env_rest.step(Action.REST.value)

    damage_defend = health_defend_before - env_defend._agents[env_defend._focal_idx].health
    damage_rest   = health_rest_before   - env_rest._agents[env_rest._focal_idx].health

    assert damage_defend <= damage_rest, (
        f"DEFEND should take ≤ damage vs REST: defend={damage_defend:.4f} rest={damage_rest:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 7: damage_taken_last_step obs signal reflects being attacked
# ---------------------------------------------------------------------------

def test_damage_taken_obs_signal() -> None:
    """After being attacked, _damage_taken_last_step for focal > 0, and obs reflects it on next step.

    The obs returned from step() belongs to the NEXT focal agent. To verify the
    signal works, we directly check _damage_taken_last_step on the attacked agent
    and verify its obs on the following step includes the non-zero damage signal.
    """
    env = _make_combat_env(n_agents=2, seed=13)
    env.reset(seed=13)

    focal_idx = env._focal_idx
    other_idx = next(i for i in range(2) if i != focal_idx)
    focal = env._agents[focal_idx]
    other = env._agents[other_idx]

    # Place adjacent, focal is very weak so heuristic will attack it
    focal.position = (5, 5)
    other.position = (6, 5)
    focal.strength = 0.1
    other.strength = 1.0
    other.sociability = 0.0   # below HEURISTIC_COLLAB_THRESHOLD — will attack
    env._global_step_count = CURRICULUM_RAMP_STEPS

    env.step(Action.REST.value)

    # After step: focal was attacked by heuristic — verify internal tracking
    assert env._damage_taken_last_step[focal_idx] > 0.0, (
        f"Expected _damage_taken_last_step[focal] > 0, got {env._damage_taken_last_step[focal_idx]:.4f}"
    )

    # Now focal_idx is the agent that was attacked — its obs on next step should reflect damage
    # Force focal back to attacked agent so we can read its obs directly
    env._focal_idx = focal_idx
    obs = env._build_obs(focal_idx)
    damage_signal = obs[-1]
    assert damage_signal > 0.0, (
        f"Expected damage_taken obs signal > 0 in attacked agent's obs, got {damage_signal:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 8: attack requires adjacency (_do_attack returns 0 when far)
# ---------------------------------------------------------------------------

def test_attack_requires_adjacency() -> None:
    """_do_attack on a non-adjacent agent returns (0, False)."""
    env = _make_combat_env(n_agents=2, seed=5)
    env.reset(seed=5)

    focal = env._agents[env._focal_idx]
    other_idx = next(i for i in range(2) if i != env._focal_idx)
    other = env._agents[other_idx]

    focal.position = (0, 0)
    other.position = (5, 5)

    damage, killed = env._do_attack(focal)
    assert damage == 0.0, f"Expected 0 damage for non-adjacent attack, got {damage}"
    assert not killed


# ---------------------------------------------------------------------------
# Test 9: combat determinism
# ---------------------------------------------------------------------------

def test_combat_determinism() -> None:
    """Same stats → identical damage output every call."""
    env = _make_combat_env(n_agents=2, seed=7)
    env.reset(seed=7)

    focal = env._agents[env._focal_idx]
    other_idx = next(i for i in range(2) if i != env._focal_idx)
    other = env._agents[other_idx]

    focal.position = (2, 2)
    other.position = (3, 2)
    focal.strength = 0.7
    other.strength = 0.4

    dmg1 = env._combat_damage(focal, other, is_defending=False)
    dmg2 = env._combat_damage(focal, other, is_defending=False)
    assert dmg1 == dmg2, f"Combat damage must be deterministic: {dmg1} vs {dmg2}"


# ---------------------------------------------------------------------------
# Test 10: curriculum schedule
# ---------------------------------------------------------------------------

def test_combat_curriculum_schedule() -> None:
    """combat_prob starts at CURRICULUM_START_PROB and ramps toward 1.0."""
    env = _make_combat_env(ramp_steps=1000)

    assert abs(env.combat_prob - CURRICULUM_START_PROB) < 1e-6, (
        f"Initial combat_prob should be {CURRICULUM_START_PROB}, got {env.combat_prob}"
    )

    env._global_step_count = 500
    mid_prob = env.combat_prob
    assert CURRICULUM_START_PROB < mid_prob < 1.0, (
        f"Mid-ramp combat_prob should be between {CURRICULUM_START_PROB} and 1.0, got {mid_prob}"
    )

    env._global_step_count = 1000
    assert abs(env.combat_prob - 1.0) < 1e-6, (
        f"Final combat_prob should be 1.0, got {env.combat_prob}"
    )

    env._global_step_count = 99_999
    assert env.combat_prob == 1.0, f"combat_prob should cap at 1.0, got {env.combat_prob}"


# ---------------------------------------------------------------------------
# Test 11: strength affects combat outcome
# ---------------------------------------------------------------------------

def test_strength_affects_combat() -> None:
    """Higher attacker strength → more damage."""
    env = _make_combat_env()
    env.reset(seed=0)

    weak_attacker   = _make_agent(strength=0.2)
    strong_attacker = _make_agent(strength=0.9)
    defender = _make_agent(strength=0.5)

    dmg_weak   = env._combat_damage(weak_attacker, defender, is_defending=False)
    dmg_strong = env._combat_damage(strong_attacker, defender, is_defending=False)
    assert dmg_strong > dmg_weak, f"Strong attacker should deal more damage: {dmg_strong} vs {dmg_weak}"


# ---------------------------------------------------------------------------
# v18: Qi-infused strike tier tests (3 discrete tiers — basic / qi / burst)
# ---------------------------------------------------------------------------

def test_strike_tier_action_mapping() -> None:
    """Each ATTACK_* action maps to the correct StrikeTier."""
    assert ACTION_TO_STRIKE[Action.ATTACK]       is STRIKE_BASIC
    assert ACTION_TO_STRIKE[Action.ATTACK_QI]    is STRIKE_QI
    assert ACTION_TO_STRIKE[Action.ATTACK_BURST] is STRIKE_BURST


def test_strike_tier_qi_costs() -> None:
    """Qi costs follow the documented ladder: basic 0, qi 1, burst 3."""
    assert STRIKE_BASIC.qi_cost == 0
    assert STRIKE_QI.qi_cost    == 1
    assert STRIKE_BURST.qi_cost == 3


def test_strike_tier_damage_ordering() -> None:
    """At the same strength, damage is ordered: basic < qi < burst."""
    env = _make_combat_env()
    env.reset(seed=0)
    attacker = _make_agent(strength=0.5)
    defender = _make_agent(strength=0.5)

    dmg_basic = env._combat_damage(attacker, defender, is_defending=False, tier=STRIKE_BASIC)
    dmg_qi    = env._combat_damage(attacker, defender, is_defending=False, tier=STRIKE_QI)
    dmg_burst = env._combat_damage(attacker, defender, is_defending=False, tier=STRIKE_BURST)
    assert dmg_basic < dmg_qi < dmg_burst, f"Tier ordering broken: basic={dmg_basic}, qi={dmg_qi}, burst={dmg_burst}"


def test_strike_tier_rescues_weak_attacker() -> None:
    """At strength=0, basic deals 0 damage but qi/burst still hurt — gives weak agents agency."""
    env = _make_combat_env()
    env.reset(seed=0)
    attacker = _make_agent(strength=0.0)
    defender = _make_agent(strength=0.5)

    dmg_basic = env._combat_damage(attacker, defender, is_defending=False, tier=STRIKE_BASIC)
    dmg_qi    = env._combat_damage(attacker, defender, is_defending=False, tier=STRIKE_QI)
    dmg_burst = env._combat_damage(attacker, defender, is_defending=False, tier=STRIKE_BURST)

    assert dmg_basic < 1e-6, f"strength=0 basic should deal ~0 damage, got {dmg_basic}"
    assert dmg_qi    > 0.10, f"strength=0 qi-strike should still hurt, got {dmg_qi}"
    assert dmg_burst > 0.30, f"strength=0 burst-strike should hit hard, got {dmg_burst}"


def test_spend_strike_qi_consumes_inventory() -> None:
    """Calling _spend_strike_qi deducts the requested cost from inventory."""
    env = _make_combat_env()
    env.reset(seed=0)
    attacker = env._agents[0]
    attacker.inventory.qi = 5

    used = env._spend_strike_qi(attacker, STRIKE_QI)
    assert used is STRIKE_QI
    assert attacker.inventory.qi == 4
    assert env._ep_qi_strikes_used == 1
    assert env._ep_qi_spent_in_combat == 1

    used = env._spend_strike_qi(attacker, STRIKE_BURST)
    assert used is STRIKE_BURST
    assert attacker.inventory.qi == 1
    assert env._ep_burst_strikes_used == 1
    assert env._ep_qi_spent_in_combat == 4  # 1 (qi) + 3 (burst)


def test_spend_strike_qi_downgrades_when_insufficient() -> None:
    """BURST request with only 1 qi downgrades to QI, not BASIC."""
    env = _make_combat_env()
    env.reset(seed=0)
    attacker = env._agents[0]
    attacker.inventory.qi = 1

    used = env._spend_strike_qi(attacker, STRIKE_BURST)
    assert used is STRIKE_QI, "Should downgrade BURST→QI when only 1 qi available"
    assert attacker.inventory.qi == 0
    assert env._ep_qi_strikes_used == 1
    assert env._ep_burst_strikes_used == 0


def test_spend_strike_qi_falls_through_to_basic() -> None:
    """QI request with 0 qi falls back to BASIC (no consumption, no telemetry)."""
    env = _make_combat_env()
    env.reset(seed=0)
    attacker = env._agents[0]
    attacker.inventory.qi = 0

    used = env._spend_strike_qi(attacker, STRIKE_QI)
    assert used is STRIKE_BASIC
    assert attacker.inventory.qi == 0
    assert env._ep_qi_strikes_used == 0
    assert env._ep_qi_spent_in_combat == 0


def test_basic_strike_never_consumes_qi() -> None:
    """STRIKE_BASIC has qi_cost=0 — must never touch inventory or telemetry."""
    env = _make_combat_env()
    env.reset(seed=0)
    attacker = env._agents[0]
    attacker.inventory.qi = 5

    used = env._spend_strike_qi(attacker, STRIKE_BASIC)
    assert used is STRIKE_BASIC
    assert attacker.inventory.qi == 5
    assert env._ep_qi_strikes_used == 0
    assert env._ep_qi_spent_in_combat == 0


# ---------------------------------------------------------------------------
# v18: action space + masking
# ---------------------------------------------------------------------------

def test_action_space_size_is_17() -> None:
    """v18 expanded discrete action space from 15 (Phase 6) to 17 (Phase 6 + qi)."""
    from murimsim.actions import N_ACTIONS_PHASE6_QI
    env = _make_combat_env()
    env.reset(seed=0)
    assert env.action_space.n == 17 == N_ACTIONS_PHASE6_QI


def test_mask_gates_attack_qi_when_no_qi() -> None:
    """ATTACK_QI must be masked when attacker has 0 qi, even with adjacent target."""
    env = _make_combat_env(n_agents=2)
    env._global_step_count = 1_000_000  # past curriculum ramp
    env.reset(seed=0)
    focal = env._agents[env._focal_idx]
    other = env._agents[1 - env._focal_idx]
    focal.position = (5, 5)
    other.position = (5, 6)  # adjacent
    focal.inventory.qi = 0
    focal.inventory.food = 0  # avoid eat-mask interference

    mask = env.action_masks()
    assert not mask[Action.ATTACK_QI],    "ATTACK_QI should be masked when qi=0"
    assert not mask[Action.ATTACK_BURST], "ATTACK_BURST should be masked when qi<3"
    assert mask[Action.ATTACK],           "Basic ATTACK should still be available"


def test_mask_gates_attack_burst_when_qi_insufficient() -> None:
    """ATTACK_BURST requires 3 qi; ATTACK_QI requires 1. With 2 qi: only QI available."""
    env = _make_combat_env(n_agents=2)
    env._global_step_count = 1_000_000
    env.reset(seed=0)
    focal = env._agents[env._focal_idx]
    other = env._agents[1 - env._focal_idx]
    focal.position = (5, 5)
    other.position = (5, 6)
    focal.inventory.qi = 2
    focal.inventory.food = 0

    mask = env.action_masks()
    assert mask[Action.ATTACK]
    assert mask[Action.ATTACK_QI],        "ATTACK_QI should be available with qi=2"
    assert not mask[Action.ATTACK_BURST], "ATTACK_BURST should be masked with qi=2 (<3)"


def test_mask_gates_all_attacks_when_no_target() -> None:
    """All ATTACK_* tiers masked when no adjacent agent or monster, regardless of qi."""
    env = _make_combat_env(n_agents=2)
    env._global_step_count = 1_000_000
    env.reset(seed=0)
    focal = env._agents[env._focal_idx]
    other = env._agents[1 - env._focal_idx]
    focal.position = (0, 0)
    other.position = (10, 10)  # far away
    # Move boss away too
    if env._monsters._monsters:
        env._monsters._monsters[0].position = (15, 15)
    focal.inventory.qi = 10  # plenty of qi

    mask = env.action_masks()
    assert not mask[Action.ATTACK]
    assert not mask[Action.ATTACK_QI]
    assert not mask[Action.ATTACK_BURST]


def test_curriculum_gates_all_attack_tiers() -> None:
    """When combat_prob ramp is incomplete, ALL three attack tiers must be gated together."""
    # Use a long ramp and an early step count → combat_prob < 1.0 → some rolls fall through
    env = _make_combat_env(n_agents=2, ramp_steps=1_000_000)
    env._global_step_count = 0  # very early → combat_prob = CURRICULUM_START_PROB (~0.05)
    env.reset(seed=0)
    focal = env._agents[env._focal_idx]
    focal.inventory.qi = 10
    focal.inventory.food = 0

    # Sample many masks — at least one should have all attacks masked together
    saw_all_three_masked = False
    for _ in range(200):
        mask = env.action_masks()
        if not mask[Action.ATTACK] and not mask[Action.ATTACK_QI] and not mask[Action.ATTACK_BURST]:
            saw_all_three_masked = True
            break
    assert saw_all_three_masked, "Curriculum should occasionally gate all three attack tiers in lockstep"


# ---------------------------------------------------------------------------
# v18: end-to-end behavioral tests via env.step
# ---------------------------------------------------------------------------

def _setup_focal_vs_boss(qi: int, focal_strength: float = 0.5) -> tuple[CombatEnv, Agent]:
    """Place focal adjacent to the boss with controlled qi + strength. Returns (env, focal)."""
    env = _make_combat_env(n_agents=10)
    env._global_step_count = 1_000_000  # past curriculum ramp
    env.reset(seed=42)
    # Need an env with the boss enabled. _make_combat_env doesn't pass enable_boss.
    # Force-spawn a boss for this test.
    if not env._monsters._monsters:
        env._monsters.spawn_boss((0, 0))
    focal = env._agents[env._focal_idx]
    focal.position = (5, 5)
    focal.strength = focal_strength
    focal.inventory.qi = qi
    focal.inventory.food = 5  # avoid critical-eat redirect
    focal.health = 1.0
    env._monsters._monsters[0].position = (5, 6)  # adjacent to focal
    return env, focal


def test_env_step_attack_qi_consumes_one_qi_and_damages_boss() -> None:
    """End-to-end: env.step(ATTACK_QI) deducts 1 qi, damages boss, increments telemetry."""
    env, focal = _setup_focal_vs_boss(qi=5, focal_strength=0.5)
    boss = env._monsters._monsters[0]
    boss_hp_before = boss.health
    qi_before = focal.inventory.qi

    env.step(Action.ATTACK_QI.value)

    assert focal.inventory.qi == qi_before - 1, f"Expected qi {qi_before-1}, got {focal.inventory.qi}"
    assert boss.health < boss_hp_before, f"Boss HP did not drop: before={boss_hp_before}, after={boss.health}"
    assert env._ep_qi_strikes_used == 1
    assert env._ep_qi_spent_in_combat == 1
    assert env._ep_burst_strikes_used == 0


def test_env_step_attack_burst_consumes_three_qi() -> None:
    """End-to-end: env.step(ATTACK_BURST) deducts 3 qi when affordable."""
    env, focal = _setup_focal_vs_boss(qi=5, focal_strength=0.5)
    qi_before = focal.inventory.qi

    env.step(Action.ATTACK_BURST.value)

    assert focal.inventory.qi == qi_before - 3
    assert env._ep_burst_strikes_used == 1
    assert env._ep_qi_spent_in_combat == 3


def test_env_step_attack_burst_downgrades_to_qi_when_insufficient() -> None:
    """env.step(ATTACK_BURST) with 2 qi: downgrades to QI (spends 1, NOT 0)."""
    env, focal = _setup_focal_vs_boss(qi=2, focal_strength=0.5)

    env.step(Action.ATTACK_BURST.value)

    assert focal.inventory.qi == 1, f"Should downgrade BURST→QI, leaving 1 qi; got {focal.inventory.qi}"
    assert env._ep_qi_strikes_used == 1
    assert env._ep_burst_strikes_used == 0


def test_qi_not_consumed_on_whiff() -> None:
    """If ATTACK_QI hits no target (no adjacent monster/agent), qi must NOT be consumed.

    Critical: a misfired strike silently draining the resource pool would teach
    the policy to avoid ATTACK_QI even when it's the right move.
    """
    env = _make_combat_env(n_agents=2)
    env._global_step_count = 1_000_000
    env.reset(seed=0)
    focal = env._agents[env._focal_idx]
    other = env._agents[1 - env._focal_idx]
    focal.position = (0, 0)
    other.position = (10, 10)  # far
    if env._monsters._monsters:
        env._monsters._monsters[0].position = (15, 15)  # far
    focal.inventory.qi = 5
    focal.inventory.food = 5

    env.step(Action.ATTACK_QI.value)

    assert focal.inventory.qi == 5, f"Whiffed ATTACK_QI should not consume qi; got {focal.inventory.qi}"
    assert env._ep_qi_strikes_used == 0
    assert env._ep_qi_spent_in_combat == 0


def test_telemetry_surfaces_in_info_at_episode_end() -> None:
    """Per-episode qi-strike counters must appear in info dict on terminal step."""
    env, focal = _setup_focal_vs_boss(qi=4, focal_strength=0.5)
    env.step(Action.ATTACK_QI.value)
    env.step(Action.ATTACK_BURST.value)

    expected_qi    = env._ep_qi_strikes_used
    expected_burst = env._ep_burst_strikes_used
    expected_spent = env._ep_qi_spent_in_combat

    # focal_idx rotates after each step; kill the *current* focal so step's
    # `terminated = not focal.alive` check fires.
    current_focal = env._agents[env._focal_idx]
    current_focal.health = 0.0
    current_focal._check_death("test")
    obs, r, term, trunc, info = env.step(Action.REST.value)

    assert term, "Episode should have terminated after focal death"
    assert "ep_qi_strikes_used" in info, f"Missing ep_qi_strikes_used in info: keys={list(info.keys())}"
    assert "ep_burst_strikes_used" in info
    assert "ep_qi_spent_in_combat" in info
    assert info["ep_qi_strikes_used"]    == expected_qi
    assert info["ep_burst_strikes_used"] == expected_burst
    assert info["ep_qi_spent_in_combat"] == expected_spent


# ---------------------------------------------------------------------------
# v18: damage clamping + DEFEND interaction with high-tier strikes
# ---------------------------------------------------------------------------

def test_burst_damage_respects_max_damage_clamp() -> None:
    """Even at strength=1.0, BURST damage is clamped to COMBAT_MAX_DAMAGE."""
    env = _make_combat_env()
    env.reset(seed=0)
    attacker = _make_agent(strength=1.0)
    defender = _make_agent(strength=0.5)

    dmg_burst = env._combat_damage(attacker, defender, is_defending=False, tier=STRIKE_BURST)
    expected_uncapped = (1.0 + STRIKE_BURST.bonus) * COMBAT_ATTACKER_SCALE  # (1 + 0.7) * 0.5 = 0.85
    assert dmg_burst <= COMBAT_MAX_DAMAGE, f"Burst {dmg_burst} exceeds clamp {COMBAT_MAX_DAMAGE}"
    assert abs(dmg_burst - min(expected_uncapped, COMBAT_MAX_DAMAGE)) < 1e-6


def test_defend_reduces_burst_damage_multiplicatively() -> None:
    """DEFEND's multiplicative reduction must apply AFTER tier bonus, not before."""
    env = _make_combat_env()
    env.reset(seed=0)
    attacker = _make_agent(strength=0.8)
    defender = _make_agent(strength=0.5)

    burst_open    = env._combat_damage(attacker, defender, is_defending=False, tier=STRIKE_BURST)
    burst_blocked = env._combat_damage(attacker, defender, is_defending=True,  tier=STRIKE_BURST)
    assert burst_blocked < burst_open, f"DEFEND must reduce burst dmg: open={burst_open}, blocked={burst_blocked}"
    expected = burst_open * max(0.0, 1.0 - defender.defense_power)
    assert abs(burst_blocked - expected) < 1e-6


# ---------------------------------------------------------------------------
# v18: combat-focus fallback prefers highest affordable tier
# ---------------------------------------------------------------------------

def test_combat_focus_fallback_prefers_burst_when_qi_rich() -> None:
    """When in-combat fallback fires AND attacker has 3+ qi, return ATTACK_BURST."""
    env = _make_combat_env(n_agents=2)
    env.reset(seed=0)
    agent = env._agents[env._focal_idx]
    other = env._agents[1 - env._focal_idx]
    agent.position = (5, 5)
    other.position = (5, 6)  # adjacent target
    agent.inventory.food = 5  # avoid critical-eat path
    agent.inventory.qi = 5
    agent.health = 1.0

    fallback = env._combat_focus_fallback(agent)
    assert fallback == Action.ATTACK_BURST, f"Expected ATTACK_BURST with qi=5, got {fallback.name}"


def test_combat_focus_fallback_uses_qi_when_2_qi() -> None:
    """qi=2 → can't afford BURST(3), should pick ATTACK_QI(1)."""
    env = _make_combat_env(n_agents=2)
    env.reset(seed=0)
    agent = env._agents[env._focal_idx]
    other = env._agents[1 - env._focal_idx]
    agent.position = (5, 5)
    other.position = (5, 6)
    agent.inventory.food = 5
    agent.inventory.qi = 2
    agent.health = 1.0

    fallback = env._combat_focus_fallback(agent)
    assert fallback == Action.ATTACK_QI, f"Expected ATTACK_QI with qi=2, got {fallback.name}"


def test_combat_focus_fallback_falls_back_to_basic_when_no_qi() -> None:
    """qi=0 → must return basic ATTACK."""
    env = _make_combat_env(n_agents=2)
    env.reset(seed=0)
    agent = env._agents[env._focal_idx]
    other = env._agents[1 - env._focal_idx]
    agent.position = (5, 5)
    other.position = (5, 6)
    agent.inventory.food = 5
    agent.inventory.qi = 0
    agent.health = 1.0

    fallback = env._combat_focus_fallback(agent)
    assert fallback == Action.ATTACK, f"Expected basic ATTACK with qi=0, got {fallback.name}"


# ---------------------------------------------------------------------------
# v18: heuristic (non-focal) agents auto-spend qi against boss
# ---------------------------------------------------------------------------

def test_heuristic_agent_spends_qi_attacking_boss() -> None:
    """Non-focal agent adjacent to boss with qi should auto-spend 1 qi via STRIKE_QI."""
    env = _make_combat_env(n_agents=10)
    env._global_step_count = 1_000_000
    env.reset(seed=42)
    if not env._monsters._monsters:
        env._monsters.spawn_boss((0, 0))
    boss = env._monsters._monsters[0]
    boss.position = (5, 5)

    # Pick a non-focal agent and put them adjacent to boss with qi
    non_focal_idx = (env._focal_idx + 1) % env._n_agents
    helper = env._agents[non_focal_idx]
    helper.position = (5, 6)  # adjacent
    helper.inventory.qi = 3
    helper.inventory.food = 5
    helper.alive = True
    helper.health = 1.0
    helper.strength = 0.5

    qi_before = helper.inventory.qi
    boss_hp_before = boss.health
    # Trigger any focal action — the per-step heuristic loop processes non-focal agents
    env.step(Action.REST.value)

    # Helper should have spent 1 qi on STRIKE_QI against the boss
    assert helper.inventory.qi <= qi_before - 1, (
        f"Heuristic helper should auto-spend ≥1 qi attacking boss; before={qi_before}, after={helper.inventory.qi}"
    )
    assert boss.health < boss_hp_before, "Boss should have taken damage from heuristic helper"
    assert env._ep_qi_strikes_used >= 1
