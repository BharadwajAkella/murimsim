"""Tests for AgentStepEvents / StepEventBuffer (P1.1, IPPO migration prep)."""
from __future__ import annotations

import pytest

from murimsim.rl.agent_events import AgentStepEvents, StepEventBuffer


def test_agent_step_events_defaults_are_neutral() -> None:
    """All reward inputs default to no-op so an unset field never contributes."""
    e = AgentStepEvents(slot=3)
    assert e.slot == 3
    assert e.food_gathered == 0
    assert e.hazard_damage == 0.0
    assert e.stash_bonus == 0.0
    assert e.damage_dealt == 0.0
    assert e.damage_taken == 0.0
    assert e.defeated is False
    assert e.group_formed is False
    assert e.betrayal is False
    assert e.action_was_redirected is False


def test_agent_step_events_reset_preserves_slot() -> None:
    e = AgentStepEvents(slot=2, food_gathered=4, damage_dealt=1.5, betrayal=True)
    e.reset()
    assert e.slot == 2  # slot survives a reset
    assert e.food_gathered == 0
    assert e.damage_dealt == 0.0
    assert e.betrayal is False


def test_step_event_buffer_indexes_by_slot() -> None:
    buf = StepEventBuffer(n_agents=4)
    assert len(buf) == 4
    for i in range(4):
        assert buf[i].slot == i


def test_step_event_buffer_reset_clears_all_slots() -> None:
    buf = StepEventBuffer(n_agents=3)
    buf[0].food_gathered = 5
    buf[1].damage_dealt = 2.0
    buf[2].defeated = True
    buf.reset_for_step()
    for i in range(3):
        assert buf[i].food_gathered == 0
        assert buf[i].damage_dealt == 0.0
        assert buf[i].defeated is False
        assert buf[i].slot == i  # slot identity preserved


def test_step_event_buffer_independent_writes() -> None:
    """Updating one slot must not bleed into other slots."""
    buf = StepEventBuffer(n_agents=4)
    buf[1].food_gathered = 7
    buf[1].defeated = True
    for i in (0, 2, 3):
        assert buf[i].food_gathered == 0
        assert buf[i].defeated is False
    assert buf[1].food_gathered == 7
    assert buf[1].defeated is True
