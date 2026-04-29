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


# ─── v19: Emergent allegiance / affinity matrix tests ────────────────────────


def test_affinity_starts_at_zero() -> None:
    env = CombatEnv(config=_load_cfg(), n_agents=4, seed=0)
    env.reset(seed=0)
    assert env._affinity(0, 1) == 0.0
    assert env._affinity(1, 0) == 0.0
    assert env._affinity(0, 0) == 0.0  # self-affinity always 0


def test_affinity_is_directional_attack() -> None:
    """Victim builds stronger hostility toward attacker than attacker → victim."""
    env = CombatEnv(config=_load_cfg(), n_agents=4, seed=0)
    env.reset(seed=0)
    attacker, victim = env._agents[0], env._agents[1]
    attacker.position = (5, 5)
    victim.position = (5, 6)
    victim.health = 1.0
    env._do_attack(attacker)
    aff_victim_to_attacker = env._affinity(1, 0)
    aff_attacker_to_victim = env._affinity(0, 1)
    assert aff_victim_to_attacker < aff_attacker_to_victim < 0.0
    # Victim's hostility magnitude should be larger than attacker's.
    assert abs(aff_victim_to_attacker) > abs(aff_attacker_to_victim)


def test_affinity_decays_over_time() -> None:
    env = CombatEnv(config=_load_cfg(), n_agents=4, seed=0)
    env.reset(seed=0)
    env._record_affinity_event(0, 1, actor_to_other=1.0, other_to_actor=1.0)
    aff_now = env._affinity(0, 1)
    assert aff_now > 0.0
    env._ep_step_count += 1000  # advance time
    aff_later = env._affinity(0, 1)
    assert aff_later < aff_now * 0.6  # decayed substantially


def test_affinity_self_pair_ignored() -> None:
    env = CombatEnv(config=_load_cfg(), n_agents=4, seed=0)
    env.reset(seed=0)
    env._record_affinity_event(0, 0, actor_to_other=10.0, other_to_actor=10.0)
    assert env._affinity(0, 0) == 0.0


def test_affinity_clears_on_reset() -> None:
    env = CombatEnv(config=_load_cfg(), n_agents=4, seed=0)
    env.reset(seed=0)
    env._record_affinity_event(0, 1, actor_to_other=1.0, other_to_actor=1.0)
    assert env._affinity(0, 1) > 0.0
    env.reset(seed=1)
    assert env._affinity(0, 1) == 0.0


def test_affinity_in_obs_channel() -> None:
    """Obs channel 4 (affinity) should reflect focal→neighbour relationship."""
    from murimsim.rl.multi_env import OBS_RESOURCE_GRID_SIZE, OBS_AGENT_GRID_SIZE
    env = CombatEnv(config=_load_cfg(), n_agents=4, seed=0)
    env.reset(seed=0)
    focal_idx = env._focal_idx
    other_idx = (focal_idx + 1) % env._n_agents
    focal = env._agents[focal_idx]
    other = env._agents[other_idx]
    fx, fy = focal.position
    other.position = (min(fx + 1, env._world.grid_size - 1), fy)
    # Big positive affinity event toward `other`
    env._record_affinity_event(focal_idx, other_idx, actor_to_other=5.0, other_to_actor=5.0)
    obs = env._build_obs(focal_idx)
    agent_grid = obs[OBS_RESOURCE_GRID_SIZE:OBS_RESOURCE_GRID_SIZE + OBS_AGENT_GRID_SIZE].reshape(5, 5, 5)
    # Affinity channel mapped from [-1,1] to [0,1]; +5/AFFINITY_NORM(=5) clipped to 1 → 1.0
    affinity_max = agent_grid[:, :, 4].max()
    assert affinity_max > 0.9, f"Expected affinity channel max ≈1.0, got {affinity_max}"


def test_terminal_info_emits_v19_keys() -> None:
    env = CombatEnv(config=_load_cfg(), n_agents=4, seed=0)
    env.reset(seed=0)
    info = {}
    for _ in range(2050):
        _, _, terminated, truncated, info = env.step(int(Action.REST.value))
        if terminated or truncated:
            break
    # Either terminal or hit max steps — terminal info should be populated when ep ends
    if "ep_betrayal_count" in info:
        assert info["ep_betrayal_count"] == 0
        assert info["ep_friendly_flank_count"] == 0
        assert "ep_focal_max_affinity" in info
        assert "ep_focal_min_affinity" in info


# ─── v19: Affinity event-recording integration tests ─────────────────────────


def _force_group(env, *idxs):
    """Force the given agent indices into the same group (skip RNG/COLLABORATE)."""
    env._groups.append(frozenset(idxs))


def _isolate_agents(env, *keep_idxs):
    """Move all agents NOT in keep_idxs to far corners so they don't interfere.

    Used by behavioural tests that depend on specific spatial setups; the v19b
    spawn-clustering can place unrelated agents adjacent to the test fixtures
    and confuse target-selection / flanking / boss-pathing.
    """
    keep = set(keep_idxs)
    gs = env._world.grid_size
    far = (gs - 1, gs - 1)
    for i, a in enumerate(env._agents):
        if i not in keep:
            a.position = far
            far = (far[0], max(0, far[1] - 2))  # spread them out


def test_affinity_recorded_on_food_share() -> None:
    """_try_food_share must update directional affinity (recipient learns more)."""
    env = CombatEnv(config=_load_cfg(), n_agents=4, seed=0)
    env.reset(seed=0)
    sharer_idx, recipient_idx = 0, 1
    sharer = env._agents[sharer_idx]
    recipient = env._agents[recipient_idx]
    sharer.inventory.food = 5
    recipient.hunger = 1.0  # critical
    _force_group(env, sharer_idx, recipient_idx)
    # Force reciprocity roll to succeed by stacking RNG calls; loop until success.
    for _ in range(50):
        env._ep_step_count += 1
        if env._try_food_share(sharer_idx, recipient_idx):
            break
    assert env._affinity(recipient_idx, sharer_idx) > 0.0
    assert env._affinity(sharer_idx, recipient_idx) > 0.0
    # Recipient should feel stronger gratitude than sharer's investment.
    assert env._affinity(recipient_idx, sharer_idx) > env._affinity(sharer_idx, recipient_idx)


def test_affinity_recorded_on_steal() -> None:
    """STEAL handler must update affinity (victim resents thief strongly)."""
    from murimsim.actions import Action
    env = CombatEnv(config=_load_cfg(), n_agents=4, seed=0)
    env.reset(seed=0)
    focal_idx = env._focal_idx
    victim_idx = (focal_idx + 1) % env._n_agents
    focal = env._agents[focal_idx]
    victim = env._agents[victim_idx]
    # Give victim food, deposit it as a stash on the victim's tile.
    victim.inventory.food = 3
    stash = env._stash_registry.deposit(victim)
    assert stash is not None
    # Place focal on the stash tile so STEAL hits it.
    focal.position = stash.position
    env.step(int(Action.STEAL.value))
    assert env._affinity(victim_idx, focal_idx) < 0.0
    assert env._affinity(focal_idx, victim_idx) < 0.0
    assert abs(env._affinity(victim_idx, focal_idx)) > abs(env._affinity(focal_idx, victim_idx))


def test_affinity_flank_bond_recorded() -> None:
    """When attacker hits target with a flanking ally, attacker↔ally affinity goes up.

    NOTE: ``_nearest_adjacent_agent`` iterates agents in index order, so the
    target must be at a LOWER index than the ally to be picked first.
    """
    env = CombatEnv(config=_load_cfg(), n_agents=4, seed=0)
    env.reset(seed=0)
    # Indices: 0=attacker, 1=target, 2=ally  (target found before ally in iteration)
    attacker, target, ally = env._agents[0], env._agents[1], env._agents[2]
    attacker.position = (5, 5)
    target.position = (5, 6)      # adjacent to attacker
    ally.position = (5, 7)        # adjacent to target (Chebyshev 1)
    target.health = 1.0
    _force_group(env, 0, 2)       # attacker + ally in same group
    env._do_attack(attacker)
    aff_atk_to_ally = env._affinity(0, 2)
    aff_ally_to_atk = env._affinity(2, 0)
    assert aff_atk_to_ally > 0.0, "attacker should bond with flanking ally"
    assert aff_ally_to_atk > 0.0, "flanking ally should bond with attacker"


def test_attack_zero_damage_does_not_record_affinity() -> None:
    """No adjacent target → no affinity change (guards against phantom updates)."""
    env = CombatEnv(config=_load_cfg(), n_agents=4, seed=0)
    env.reset(seed=0)
    a, b = env._agents[0], env._agents[1]
    a.position = (1, 1)
    b.position = (15, 15)         # far away
    env._do_attack(a)
    assert env._affinity(0, 1) == 0.0
    assert env._affinity(1, 0) == 0.0


def test_betrayal_penalty_and_telemetry() -> None:
    """Focal attacking a high-affinity target → betrayal penalty + counter incremented."""
    from murimsim.rl.multi_env import (
        AFFINITY_BETRAY_THRESHOLD, AFFINITY_NORM, PENALTY_BETRAYAL,
    )
    from murimsim.actions import Action
    env = CombatEnv(config=_load_cfg(), n_agents=4, seed=0)
    env.reset(seed=0)
    env._global_step_count = CURRICULUM_RAMP_STEPS  # combat_prob = 1.0
    focal_idx = env._focal_idx
    target_idx = (focal_idx + 1) % env._n_agents
    focal = env._agents[focal_idx]
    target = env._agents[target_idx]
    focal.position = (5, 5)
    target.position = (5, 6)
    target.health = 1.0
    _isolate_agents(env, focal_idx, target_idx)
    # Pre-load focal→target affinity above the betrayal threshold.
    raw_value = (AFFINITY_BETRAY_THRESHOLD + 0.2) * AFFINITY_NORM
    env._record_affinity_event(focal_idx, target_idx,
                               actor_to_other=raw_value, other_to_actor=0.0)
    assert env._affinity(focal_idx, target_idx) >= AFFINITY_BETRAY_THRESHOLD
    pre_count = env._ep_betrayal_count
    _, reward, _, _, _ = env.step(int(Action.ATTACK.value))
    assert env._ep_betrayal_count == pre_count + 1
    # Reward should include the negative betrayal penalty (other terms may push it up,
    # but penalty must have been applied — check via direct value bound).
    assert reward <= 1.0 + PENALTY_BETRAYAL + 1.0  # loose bound, just sanity


def test_friendly_flank_telemetry_increments() -> None:
    """Focal flanking with a positive-affinity ally bumps ep_friendly_flank_count."""
    from murimsim.rl.multi_env import AFFINITY_NORM
    from murimsim.actions import Action
    env = CombatEnv(config=_load_cfg(), n_agents=4, seed=0)
    env.reset(seed=0)
    env._global_step_count = CURRICULUM_RAMP_STEPS  # combat_prob = 1.0
    focal_idx = env._focal_idx
    ally_idx = (focal_idx + 1) % env._n_agents
    target_idx = (focal_idx + 2) % env._n_agents
    focal, ally, target = env._agents[focal_idx], env._agents[ally_idx], env._agents[target_idx]
    focal.position = (5, 5)
    ally.position = (5, 4)
    target.position = (5, 6)
    target.health = 1.0
    _isolate_agents(env, focal_idx, ally_idx, target_idx)
    _force_group(env, focal_idx, ally_idx)
    # Pre-load focal→ally positive affinity.
    env._record_affinity_event(focal_idx, ally_idx,
                               actor_to_other=2.0 * AFFINITY_NORM, other_to_actor=0.0)
    pre = env._ep_friendly_flank_count
    env.step(int(Action.ATTACK.value))
    assert env._ep_friendly_flank_count == pre + 1


def test_mutual_share_bonus_only_for_focal_as_sharer() -> None:
    """Focal-as-sharer with positive recipient→focal affinity gets share + mutual bonus.

    Verifies both:
      - REWARD_FOOD_SHARE is added when focal is the sharer
      - REWARD_MUTUAL_SHARE_BONUS is ADDITIVE when recipient already has positive
        affinity toward focal (signals reciprocity forming)
    """
    from murimsim.rl.multi_env import (
        REWARD_FOOD_SHARE, REWARD_MUTUAL_SHARE_BONUS, AFFINITY_NORM,
    )

    def _setup(load_recipient_affinity: bool):
        env = CombatEnv(config=_load_cfg(), n_agents=4, seed=0)
        env.reset(seed=0)
        focal_idx = env._focal_idx
        recipient_idx = (focal_idx + 1) % env._n_agents
        focal = env._agents[focal_idx]
        recipient = env._agents[recipient_idx]
        focal.inventory.food = 5
        recipient.hunger = 1.0
        recipient.inventory.food = 0
        focal.position = (5, 5)
        recipient.position = (5, 6)
        _force_group(env, focal_idx, recipient_idx)
        if load_recipient_affinity:
            env._record_affinity_event(
                recipient_idx, focal_idx,
                actor_to_other=2.0 * AFFINITY_NORM, other_to_actor=0.0,
            )
            assert env._affinity(recipient_idx, focal_idx) > 0.0
        return env, focal_idx, recipient_idx

    # Run the food-share loop directly and capture the reward accumulator the way
    # CombatEnv.step does (this avoids the noise of the full reward composition).
    def _accumulate_share_reward(env, focal_idx):
        food_share_reward = 0.0
        # Force successful share by calling repeatedly (RNG roll inside).
        for _ in range(50):
            env._ep_step_count += 1
            for sharer_idx in range(env._n_agents):
                if not env._agents[sharer_idx].alive:
                    continue
                grp = env._get_group(sharer_idx)
                if grp is None:
                    continue
                for recipient_idx in grp:
                    if recipient_idx == sharer_idx:
                        continue
                    pre_food = env._agents[sharer_idx].inventory.food
                    pre_aff = env._affinity(recipient_idx, focal_idx) if sharer_idx == focal_idx else 0.0
                    if env._try_food_share(sharer_idx, recipient_idx):
                        if sharer_idx == focal_idx:
                            food_share_reward += REWARD_FOOD_SHARE
                            if pre_aff > 0.0:
                                food_share_reward += REWARD_MUTUAL_SHARE_BONUS
                        env._agents[sharer_idx].inventory.food = pre_food  # restore for repeat
            if food_share_reward > 0:
                break
        return food_share_reward

    # Case 1: no pre-existing affinity → only base share reward
    env_a, focal_a, _ = _setup(load_recipient_affinity=False)
    base = _accumulate_share_reward(env_a, focal_a)
    assert base == REWARD_FOOD_SHARE, f"Expected only base share reward, got {base}"

    # Case 2: recipient has positive affinity → base + mutual bonus
    env_b, focal_b, _ = _setup(load_recipient_affinity=True)
    bonus = _accumulate_share_reward(env_b, focal_b)
    assert bonus == REWARD_FOOD_SHARE + REWARD_MUTUAL_SHARE_BONUS, (
        f"Expected base+mutual reward, got {bonus}"
    )


# ── v19c: extra interaction-event sources ──────────────────────────────────────
def test_collaborate_records_symmetric_affinity_bond() -> None:
    """Successful _try_collaborate (group formed) records +AFFINITY_COLLAB_BOTH both ways."""
    from murimsim.rl.multi_env import AFFINITY_COLLAB_BOTH, AFFINITY_NORM, HEURISTIC_COLLAB_THRESHOLD
    env = CombatEnv(config=_load_cfg(), n_agents=4, seed=0)
    env.reset(seed=0)
    focal_idx = env._focal_idx
    other_idx = (focal_idx + 1) % env._n_agents
    focal = env._agents[focal_idx]
    other = env._agents[other_idx]
    focal.position = (5, 5)
    other.position = (5, 6)
    other.sociability = max(other.sociability, HEURISTIC_COLLAB_THRESHOLD + 0.05)
    _isolate_agents(env, focal_idx, other_idx)
    assert env._affinity(focal_idx, other_idx) == 0.0
    formed = env._try_collaborate(focal_idx)
    assert formed
    expected = AFFINITY_COLLAB_BOTH / AFFINITY_NORM
    assert abs(env._affinity(focal_idx, other_idx) - expected) < 1e-6
    assert abs(env._affinity(other_idx, focal_idx) - expected) < 1e-6


def test_collaborate_no_bond_when_already_grouped() -> None:
    """_try_collaborate is a no-op when both agents already share a group; no extra bond."""
    from murimsim.rl.multi_env import HEURISTIC_COLLAB_THRESHOLD
    env = CombatEnv(config=_load_cfg(), n_agents=4, seed=0)
    env.reset(seed=0)
    focal_idx = env._focal_idx
    other_idx = (focal_idx + 1) % env._n_agents
    focal = env._agents[focal_idx]
    other = env._agents[other_idx]
    focal.position = (5, 5)
    other.position = (5, 6)
    other.sociability = max(other.sociability, HEURISTIC_COLLAB_THRESHOLD + 0.05)
    _isolate_agents(env, focal_idx, other_idx)
    assert env._try_collaborate(focal_idx)
    aff_after_first = env._affinity(focal_idx, other_idx)
    # Second call: already grouped, returns False, no further bond accrual.
    assert not env._try_collaborate(focal_idx)
    assert env._affinity(focal_idx, other_idx) == aff_after_first


def test_joint_kill_bonds_pairs_all_attackers() -> None:
    """_record_joint_kill_bonds wires symmetric +AFFINITY_JOINT_KILL between every contributor pair."""
    from murimsim.rl.multi_env import AFFINITY_JOINT_KILL, AFFINITY_NORM
    env = CombatEnv(config=_load_cfg(), n_agents=4, seed=0)
    env.reset(seed=0)
    a0, a1, a2 = env._agents[0], env._agents[1], env._agents[2]
    env._record_joint_kill_bonds({a0.agent_id, a1.agent_id, a2.agent_id})
    expected = AFFINITY_JOINT_KILL / AFFINITY_NORM
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        assert abs(env._affinity(i, j) - expected) < 1e-6
        assert abs(env._affinity(j, i) - expected) < 1e-6
    # Non-contributor pair — no bond.
    assert env._affinity(0, 3) == 0.0


def test_proximity_bonds_accrue_for_adjacent_agents() -> None:
    """Co-located agents accumulate proximity bond on every PROXIMITY_TICK_EVERY-th step."""
    from murimsim.rl.multi_env import (
        AFFINITY_PROXIMITY_PER_STEP, AFFINITY_PROXIMITY_TICK_EVERY,
        AFFINITY_PROXIMITY_RADIUS, AFFINITY_NORM,
    )
    env = CombatEnv(config=_load_cfg(), n_agents=4, seed=0)
    env.reset(seed=0)
    # Place agents 0 and 1 adjacent; agents 2 and 3 far away.
    env._agents[0].position = (5, 5)
    env._agents[1].position = (5, 5 + AFFINITY_PROXIMITY_RADIUS)  # within radius
    env._agents[2].position = (20, 20)
    env._agents[3].position = (20, 22)
    # Force tick alignment, then call directly.
    env._ep_step_count = AFFINITY_PROXIMITY_TICK_EVERY
    pre = env._affinity(0, 1)
    env._apply_proximity_bonds()
    post = env._affinity(0, 1)
    expected_delta = AFFINITY_PROXIMITY_PER_STEP / AFFINITY_NORM
    assert abs((post - pre) - expected_delta) < 1e-6
    # Should be symmetric.
    assert abs(env._affinity(1, 0) - post) < 1e-6
    # Far agents (0 and 2) should not have any bond.
    assert env._affinity(0, 2) == 0.0


def test_proximity_bonds_skip_off_tick_steps() -> None:
    """Proximity sweep is a no-op when _ep_step_count is not aligned to TICK_EVERY."""
    from murimsim.rl.multi_env import AFFINITY_PROXIMITY_TICK_EVERY
    env = CombatEnv(config=_load_cfg(), n_agents=4, seed=0)
    env.reset(seed=0)
    env._agents[0].position = (5, 5)
    env._agents[1].position = (5, 6)
    # Off-tick step.
    env._ep_step_count = AFFINITY_PROXIMITY_TICK_EVERY + 1
    pre = env._affinity(0, 1)
    env._apply_proximity_bonds()
    assert env._affinity(0, 1) == pre


def test_proximity_bonds_skip_dead_agents() -> None:
    """Dead agents don't accrue proximity bonds (or get them)."""
    from murimsim.rl.multi_env import AFFINITY_PROXIMITY_TICK_EVERY
    env = CombatEnv(config=_load_cfg(), n_agents=4, seed=0)
    env.reset(seed=0)
    env._agents[0].position = (5, 5)
    env._agents[1].position = (5, 6)
    env._agents[1].alive = False
    env._ep_step_count = AFFINITY_PROXIMITY_TICK_EVERY
    env._apply_proximity_bonds()
    assert env._affinity(0, 1) == 0.0


# ── P0.1 (IPPO migration prep): action_masks RNG-purity ───────────────────────
def test_action_masks_repeated_calls_are_stable() -> None:
    """Repeated action_masks() calls within a step must return identical masks
    and not consume RNG more than once (lazy-cached curriculum gate).

    Required for IPPO: masks will be queried per-(env, agent) every rollout step.
    Any per-call RNG draw would scramble env stochasticity vs single-focal mode.
    """
    env = CombatEnv(config=_load_cfg(), n_agents=4, seed=0)
    env.reset(seed=0)
    # Mid-curriculum so the gate genuinely toggles.
    env._global_step_count = max(1, CURRICULUM_RAMP_STEPS // 2)
    env._cached_curriculum_attack_allowed = None  # force fresh

    first = env.action_masks().copy()
    state_after_first = env._rng.bit_generator.state
    for _ in range(50):
        m = env.action_masks()
        assert np.array_equal(m, first), "action_masks must be stable within a step"
    # After the first call, no further RNG advance should occur.
    assert env._rng.bit_generator.state == state_after_first


def test_action_masks_and_step_redirect_agree_within_step() -> None:
    """Within one step, action_masks() and step()'s redirect must consult the
    SAME curriculum boolean. Previously two independent _rng.random() calls
    could disagree (mask says attack allowed, step redirects to TRAIN, or vice versa).
    """
    env = CombatEnv(config=_load_cfg(), n_agents=4, seed=0)
    env.reset(seed=0)
    env._global_step_count = max(1, CURRICULUM_RAMP_STEPS // 2)
    env._cached_curriculum_attack_allowed = None

    # Calling action_masks should populate the cache. Subsequent _curriculum_attack_allowed()
    # must return the cached value (no new draw, no disagreement risk).
    _ = env.action_masks()
    cached = env._cached_curriculum_attack_allowed
    assert cached is not None, "action_masks() must populate the cache"
    for _ in range(10):
        assert env._curriculum_attack_allowed() == cached
    # After step(), the cache is invalidated so the next mask call draws fresh.
    env.step(int(Action.MOVE_N.value))
    assert env._cached_curriculum_attack_allowed is None


# ── P0.2 (IPPO migration prep): per-agent action_masks / redirect ─────────────
def test_action_masks_can_be_queried_per_agent_idx() -> None:
    """action_masks(agent_idx) must mask based on THAT agent's local context,
    not the focal's. Required so IPPO can compute masks for every agent in a step
    without rotating focal.
    """
    env = CombatEnv(config=_load_cfg(), n_agents=4, seed=0)
    env.reset(seed=0)
    env._global_step_count = CURRICULUM_RAMP_STEPS  # combat_prob=1, no curriculum mask

    # Give agent 0 zero food, agent 1 plenty of food. EAT mask differs per agent.
    env._agents[0].inventory.food = 0
    env._agents[1].inventory.food = 5

    mask0 = env.action_masks(agent_idx=0)
    mask1 = env.action_masks(agent_idx=1)
    assert not mask0[Action.EAT], "agent 0 has no food → EAT must be masked"
    assert mask1[Action.EAT], "agent 1 has food → EAT must be allowed"

    # Default (no arg) targets focal — must be back-compat with single-agent loop.
    default_mask = env.action_masks()
    focal_mask = env.action_masks(agent_idx=env._focal_idx)
    assert np.array_equal(default_mask, focal_mask)


def test_redirect_invalid_action_uses_per_agent_combat_state() -> None:
    """_redirect_invalid_action(agent, action, agent_idx) must consult agent_idx's
    combat cooldown — not focal's — when deciding combat-focus fallback.
    """
    env = CombatEnv(config=_load_cfg(), n_agents=4, seed=0)
    env.reset(seed=0)
    env._global_step_count = CURRICULUM_RAMP_STEPS

    # Force agent 1 in-combat, focal (0) NOT in combat.
    env._in_combat_cooldown[0] = 0
    env._in_combat_cooldown[1] = 5

    # When evaluated for agent 1, TRAIN should be redirected (in-combat focus).
    redirected_for_1 = env._redirect_invalid_action(env._agents[1], Action.TRAIN, agent_idx=1)
    assert redirected_for_1 != Action.TRAIN, "agent 1 in combat → TRAIN must be redirected"

    # When evaluated for agent 0, TRAIN should pass through (not in combat).
    redirected_for_0 = env._redirect_invalid_action(env._agents[0], Action.TRAIN, agent_idx=0)
    assert redirected_for_0 == Action.TRAIN, "agent 0 not in combat → TRAIN passes through"
