"""P1.3 — extra coverage for P1.2 reward extraction.

Catches drift the existing tests miss:
  1. `_compute_combat_reward` events= kwarg path is only tested through env.step.
     A direct unit test pins the contract that positional == events for matching
     inputs (this is what guarantees byte-identity).
  2. Per-agent reward array is not pinned — only focal is golden'd. A drift in
     non-focal reward semantics would go undetected.
  3. The acting_focal_idx capture (focal advances mid-step) — implicit only.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from murimsim.actions import Action
from murimsim.rl.agent_events import AgentStepEvents
from murimsim.rl.multi_env import CombatEnv

CONFIG_PATH = Path("config/default.yaml")
FIXTURE_DIR = Path(__file__).parent / "fixtures"
REGEN = os.environ.get("REGEN_REWARD_GOLDEN") == "1"


def _load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _make_env(n_agents: int = 6, seed: int = 0) -> CombatEnv:
    return CombatEnv(
        config=_load_cfg(),
        n_agents=n_agents,
        seed=seed,
        curriculum_ramp_steps=300_000,
    )


# ---------------------------------------------------------------------------
# Gap 1: events= override == positional for matching inputs.
# This is THE invariant that guarantees focal byte-identity post-refactor.
# ---------------------------------------------------------------------------


def test_compute_combat_reward_events_equals_positional() -> None:
    env = _make_env(n_agents=2, seed=11)
    env.reset(seed=11)
    agent = env._agents[0]
    ev = AgentStepEvents(
        slot=0,
        food_gathered=2,
        hazard_damage=0.05,
        damage_dealt=0.4,
        damage_taken=0.15,
        defeated=True,
    )
    r_pos = env._compute_combat_reward(
        hunger_prev=agent.hunger,
        health_prev=agent.health,
        food_gathered=ev.food_gathered,
        hazard_damage=ev.hazard_damage,
        agent=agent,
        exploration_reward=0.0,
        damage_dealt=ev.damage_dealt,
        damage_taken=ev.damage_taken,
        defeat_bonus=0.3,  # REWARD_DEFEAT_OPPONENT
        inv_food_prev=agent.inventory.food,
    )
    r_ev = env._compute_combat_reward(
        hunger_prev=agent.hunger,
        health_prev=agent.health,
        agent=agent,
        inv_food_prev=agent.inventory.food,
        events=ev,
    )
    # Bit-identical; defeat_bonus is derived from events.defeated == True
    assert r_pos == r_ev


def test_compute_reward_events_equals_positional() -> None:
    env = _make_env(n_agents=2, seed=12)
    env.reset(seed=12)
    agent = env._agents[0]
    ev = AgentStepEvents(slot=0, food_gathered=3, hazard_damage=0.1)
    r_pos = env._compute_reward(
        hunger_prev=agent.hunger,
        health_prev=agent.health,
        food_gathered=ev.food_gathered,
        hazard_damage=ev.hazard_damage,
        agent=agent,
        inv_food_prev=agent.inventory.food,
    )
    r_ev = env._compute_reward(
        hunger_prev=agent.hunger,
        health_prev=agent.health,
        agent=agent,
        inv_food_prev=agent.inventory.food,
        events=ev,
    )
    assert r_pos == r_ev


# ---------------------------------------------------------------------------
# Gap 2: per-agent reward golden across a multi-step rollout.
# Pins non-focal reward semantics — drift here would be missed by the
# focal-only goldens.
# ---------------------------------------------------------------------------


def _action_seq() -> list[int]:
    base = [
        Action.MOVE_N.value, Action.GATHER.value, Action.MOVE_E.value,
        Action.ATTACK.value, Action.DEFEND.value, Action.EAT.value,
        Action.REST.value, Action.COLLABORATE.value, Action.MOVE_S.value,
        Action.DEPOSIT.value,
    ]
    return [base[i % len(base)] for i in range(100)]


def test_per_agent_reward_array_golden() -> None:
    """Pin per-agent reward matrix [steps, n_agents] for a 100-step rollout."""
    env = _make_env(n_agents=4, seed=7)
    env.reset(seed=7)
    seq = _action_seq()
    rewards = np.zeros((len(seq), 4), dtype=np.float64)
    for t, act in enumerate(seq):
        _o, _r, term, trunc, info = env.step(int(act))
        rewards[t] = info["per_agent_reward"]
        if term or trunc:
            env.reset(seed=7 + t + 1)
    fixture = FIXTURE_DIR / "per_agent_reward_matrix.npy"
    if REGEN or not fixture.exists():
        np.save(fixture, rewards)
        if REGEN:
            print(f"REGENERATED {fixture}")
            return
    expected = np.load(fixture)
    np.testing.assert_array_equal(
        rewards, expected,
        err_msg=(
            "per_agent_reward matrix drifted. If intentional, regen with "
            "REGEN_REWARD_GOLDEN=1 pytest."
        ),
    )


# ---------------------------------------------------------------------------
# Gap 3: acting_focal_idx capture — focal advances mid-step.
# Regression guard for the bug we just hit.
# ---------------------------------------------------------------------------


def test_per_agent_reward_aligned_with_acting_focal_not_post_step_focal() -> None:
    """The slot whose action was applied this step receives the scalar reward,
    even though env._focal_idx has advanced by the time step() returns."""
    env = _make_env(n_agents=3, seed=8)
    env.reset(seed=8)
    acting = env._focal_idx
    _o, reward, _t, _tr, info = env.step(Action.MOVE_N.value)
    post_step_focal = env._focal_idx
    arr = info["per_agent_reward"]
    # acting slot got the scalar reward
    assert float(arr[acting]) == pytest.approx(float(reward))
    # If focal_idx actually advanced (n_agents > 1, all alive), the post-step
    # focal slot should NOT have the scalar reward (unless coincidentally equal).
    if post_step_focal != acting:
        # post_step_focal got its own reward computed fresh from events,
        # which for a standing-still slot is just REWARD_ALIVE-ish, not the
        # focal's MOVE_N reward.
        assert float(arr[post_step_focal]) != pytest.approx(float(reward)) or \
               float(arr[post_step_focal]) == pytest.approx(0.02)  # alive bonus


# ---------------------------------------------------------------------------
# Gap 4: agent_events length and slot indexing invariants.
# ---------------------------------------------------------------------------


def test_agent_events_length_matches_n_agents_and_slot_index_correct() -> None:
    env = _make_env(n_agents=5, seed=9)
    env.reset(seed=9)
    _o, _r, _t, _tr, info = env.step(Action.REST.value)
    events = info["agent_events"]
    assert len(events) == 5
    for i, ev in enumerate(events):
        assert ev.slot == i, f"events[{i}].slot should be {i}, got {ev.slot}"


def test_per_agent_reward_dtype_and_shape_invariant_across_steps() -> None:
    """Across many steps the array stays float64 and shape (n_agents,)."""
    env = _make_env(n_agents=4, seed=10)
    env.reset(seed=10)
    for _ in range(50):
        _o, _r, term, trunc, info = env.step(Action.MOVE_N.value)
        arr = info["per_agent_reward"]
        assert arr.dtype == np.float64
        assert arr.shape == (4,)
        if term or trunc:
            env.reset(seed=99)
