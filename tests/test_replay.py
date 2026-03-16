"""Phase 1.5 — ReplayLogger tests.

All 4 tests must pass for the Phase 1.5 exit gate.
Phase 1 tests must continue to pass alongside these.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from murimsim.replay import ReplayLogger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_AGENT = {
    "id": "iron_fang_00",
    "sect": "iron_fang",
    "pos": [5, 10],
    "health": 0.9,
    "hunger": 0.2,
    "poison_resistance": 0.15,
    "action": "gather",
    "action_detail": "Gathered food",
    "alive": True,
}

_RESOURCES = {
    "food": [[5, 10, 1.0], [2, 3, 1.0]],
    "qi": [[15, 15, 1.0]],
    "materials": [],
    "poison": [[1, 1, 1.0]],
}

_EVENTS = [
    {"type": "combat", "attacker": "iron_fang_00", "defender": "jade_lotus_01", "damage": 0.12},
]


def _write_ticks(logger: ReplayLogger, n: int) -> None:
    for tick in range(n):
        logger.log_tick(
            tick=tick,
            generation=0,
            agents=[_AGENT],
            resources=_RESOURCES,
            events=_EVENTS if tick == 0 else [],
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_replay_logger_writes(tmp_path: Path) -> None:
    """10 ticks → exactly 10 lines in the JSONL file."""
    with ReplayLogger(seed=1, output_dir=tmp_path) as rl:
        _write_ticks(rl, 10)

    lines = rl.path.read_text().strip().splitlines()
    assert len(lines) == 10, f"Expected 10 lines, got {len(lines)}"


def test_replay_format_valid(tmp_path: Path) -> None:
    """Every line is valid JSON and contains the required top-level keys."""
    required_keys = {"tick", "generation", "agents", "resources", "events"}

    with ReplayLogger(seed=2, output_dir=tmp_path) as rl:
        _write_ticks(rl, 5)

    for i, line in enumerate(rl.path.read_text().strip().splitlines()):
        record = json.loads(line)  # raises if invalid JSON
        missing = required_keys - record.keys()
        assert not missing, f"Line {i}: missing keys {missing}"
        assert record["tick"] == i
        assert record["generation"] == 0


def test_replay_agent_fields(tmp_path: Path) -> None:
    """Agent dicts in the log include all required fields, incl. poison_resistance."""
    required_agent_keys = {
        "id", "sect", "pos", "health", "hunger",
        "poison_resistance", "action", "action_detail", "alive",
    }

    with ReplayLogger(seed=3, output_dir=tmp_path) as rl:
        _write_ticks(rl, 1)

    line = rl.path.read_text().strip()
    record = json.loads(line)
    assert len(record["agents"]) == 1
    agent = record["agents"][0]
    missing = required_agent_keys - agent.keys()
    assert not missing, f"Agent missing keys: {missing}"
    assert agent["poison_resistance"] == 0.15


def test_replay_toggle(tmp_path: Path) -> None:
    """When enabled=False, no file is created and log_tick is a no-op."""
    rl = ReplayLogger(seed=4, output_dir=tmp_path, enabled=False)
    _write_ticks(rl, 10)
    rl.close()

    assert rl.path is None, "path should be None when disabled"
    jsonl_files = list(tmp_path.glob("*.jsonl"))
    assert len(jsonl_files) == 0, f"No file should be written, found: {jsonl_files}"
