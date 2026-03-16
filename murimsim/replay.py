"""Replay logger: records simulation state to JSONL for the web viewer.

One JSON object is written per tick. The output file is:
    logs/replays/run_{seed}.jsonl

Format per line:
    {
        "tick": int,
        "generation": int,
        "agents": [ { agent fields } ],
        "resources": { rid: [[x, y, intensity], ...] },
        "events": [ { event fields } ]
    }

Design rules:
- When disabled (enabled=False), the logger is a no-op. No file is created.
- The logger owns the file handle; call close() or use it as a context manager.
- No murimsim imports here — callers pass plain dicts/lists so this module
  stays decoupled from agent/world internals.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default output directory
_DEFAULT_REPLAY_DIR = Path("/mnt/c/Users/bhara/Downloads/replays")


class ReplayLogger:
    """Writes per-tick simulation snapshots to a JSONL file.

    Args:
        seed:       World seed — used to name the output file.
        output_dir: Directory to write replay files. Created if absent.
        enabled:    When False the logger is a complete no-op (no file created).
        filename:   Override the output filename (default: ``run_{seed}.jsonl``).
    """

    def __init__(
        self,
        seed: int,
        output_dir: Path | str = _DEFAULT_REPLAY_DIR,
        enabled: bool = True,
        filename: str | None = None,
    ) -> None:
        self.seed = seed
        self.enabled = enabled
        self._file = None
        self._path: Path | None = None

        if not self.enabled:
            return

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self._path = out / (filename if filename is not None else f"run_{seed}.jsonl")
        self._file = open(self._path, "w", encoding="utf-8")  # noqa: WPS515
        logger.debug("ReplayLogger: writing to %s", self._path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path | None:
        """Absolute path to the JSONL file, or None when disabled."""
        return self._path

    def log_tick(
        self,
        tick: int,
        generation: int,
        agents: list[dict[str, Any]],
        resources: dict[str, list[list[float | int]]],
        events: list[dict[str, Any]],
    ) -> None:
        """Append one tick snapshot as a JSONL line.

        Args:
            tick:       Current simulation tick.
            generation: Current generation number.
            agents:     List of agent state dicts. Each must include at minimum:
                          id, sect, pos, health, hunger, poison_resistance,
                          action, action_detail, alive.
            resources:  Mapping of resource_id → list of [x, y, intensity] triples
                        for every present tile (intensity in [0,1]).
            events:     List of event dicts (combat, death, etc.) that occurred
                        this tick.
        """
        if not self.enabled:
            return

        record: dict[str, Any] = {
            "tick": tick,
            "generation": generation,
            "agents": agents,
            "resources": resources,
            "events": events,
        }
        self._file.write(json.dumps(record, separators=(",", ":")) + "\n")

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> ReplayLogger:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
