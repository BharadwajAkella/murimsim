"""P1.2 tests — per-agent reward extraction via AgentStepEvents.

These tests cover invariants A-E from the P1.2 strategy:
  A. Backwards compat — covered by tests/test_focal_reward_unchanged.py
  B. Symmetry — _compute_reward is a pure function of (agent, events, prev)
  C. Event attribution — events go to the causally responsible slot
  D. No double-counting — one world event → bounded number of slot events
  E. Dead/empty slots get zero reward (or terminal reward on death step)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from murimsim.actions import Action
from murimsim.rl.multi_env import CombatEnv

CONFIG_PATH = Path("config/default.yaml")


def _load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _make_env(n_agents: int = 6, seed: int = 0) -> CombatEnv:
    return CombatEnv(
        config=_load_cfg(),
        n_agents=n_agents,
        seed=seed,
        curriculum_ramp_steps=0,  # combat allowed immediately
    )


# ---------------------------------------------------------------------------
# Invariant C: event attribution
# ---------------------------------------------------------------------------


def test_events_present_in_step_info() -> None:
    """After step(), info['agent_events'] is a sequence of length n_agents."""
    env = _make_env(n_agents=4, seed=1)
    env.reset(seed=1)
    _obs, _r, _t, _tr, info = env.step(Action.MOVE_N.value)
    assert "agent_events" in info
    assert len(info["agent_events"]) == 4


def test_attack_attributes_dealt_and_taken_to_correct_slots() -> None:
    """ATTACK by focal → focal has damage_dealt > 0, target has damage_taken > 0,
    others have neither."""
    env = _make_env(n_agents=4, seed=2)
    env.reset(seed=2)
    # Force adjacency: place all agents next to focal
    fi = env._focal_idx  # capture BEFORE step (advances at step end)
    focal = env._agents[fi]
    target = env._agents[(fi + 1) % 4]
    fx, fy = focal.position
    target.position = (fx + 1, fy)
    target.health = 1.0
    ti = env._agents.index(target)
    # Force curriculum gate to allow combat (otherwise ATTACK is redirected to TRAIN)
    env._global_step_count = 1_000_000
    _obs, _r, _t, _tr, info = env.step(Action.ATTACK.value)
    events = info["agent_events"]
    assert events[fi].damage_dealt > 0, f"focal slot {fi} should have damage_dealt > 0"
    assert events[ti].damage_taken == pytest.approx(events[fi].damage_dealt)
    for i, ev in enumerate(events):
        if i not in (fi, ti):
            assert ev.damage_dealt == 0.0
            assert ev.damage_taken == 0.0


# ---------------------------------------------------------------------------
# Invariant D: no double-counting
# ---------------------------------------------------------------------------


def test_idle_step_zero_events_for_all() -> None:
    """A step where no agent gathers/attacks/etc → all event fields are zero."""
    env = _make_env(n_agents=3, seed=3)
    env.reset(seed=3)
    _obs, _r, _t, _tr, info = env.step(Action.REST.value)
    for ev in info["agent_events"]:
        assert ev.food_gathered == 0
        assert ev.hazard_damage == 0.0
        assert ev.damage_dealt == 0.0
        assert ev.damage_taken == 0.0
        assert ev.defeated is False


# ---------------------------------------------------------------------------
# Invariant B: pure-function reward
# ---------------------------------------------------------------------------


def test_reward_function_pure_given_events() -> None:
    """Same (agent, events, prev_state) → same reward across many calls."""
    from murimsim.rl.agent_events import AgentStepEvents

    env = _make_env(n_agents=2, seed=4)
    env.reset(seed=4)
    agent = env._agents[0]
    ev = AgentStepEvents(slot=0)
    ev.food_gathered = 1
    # _compute_reward must accept events as a single argument (new signature)
    r1 = env._compute_reward(
        hunger_prev=agent.hunger,
        health_prev=agent.health,
        agent=agent,
        inv_food_prev=agent.inventory.food,
        events=ev,
    )
    r2 = env._compute_reward(
        hunger_prev=agent.hunger,
        health_prev=agent.health,
        agent=agent,
        inv_food_prev=agent.inventory.food,
        events=ev,
    )
    assert r1 == r2


# ---------------------------------------------------------------------------
# Invariant E: dead/empty slots get zero reward
# ---------------------------------------------------------------------------


def test_per_agent_reward_array_shape_and_focal_match() -> None:
    """info['per_agent_reward'] is length n_agents and the acting focal entry
    equals the scalar reward returned by step()."""
    env = _make_env(n_agents=5, seed=5)
    env.reset(seed=5)
    acting_focal = env._focal_idx  # captured BEFORE step (advances at step end)
    _obs, reward, _t, _tr, info = env.step(Action.MOVE_N.value)
    arr = info["per_agent_reward"]
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (5,)
    assert float(arr[acting_focal]) == pytest.approx(float(reward))


def test_dead_slot_per_agent_reward_zero_after_death_step() -> None:
    """Once an agent is dead, subsequent steps give it 0.0 (until rebirth)."""
    env = _make_env(n_agents=4, seed=6)
    env.reset(seed=6)
    victim_idx = (env._focal_idx + 1) % 4
    env._agents[victim_idx].health = 0.0
    env._agents[victim_idx].alive = False
    _obs, _r, _t, _tr, info = env.step(Action.MOVE_N.value)
    # The victim was already dead at step start — its reward this step must be 0
    arr = info["per_agent_reward"]
    if info["lifecycle"][victim_idx]["alive"] is False:
        assert float(arr[victim_idx]) == 0.0
