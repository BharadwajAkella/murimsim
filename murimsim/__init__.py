"""MurimSim: multi-agent martial arts simulation.

Public API for Phase 1 (World Mechanics):
    World          — tick-based 2D grid world
    ResourceConfig — immutable resource type specification
    WorldStats     — resource lifecycle bookkeeping

Public API for Phase 1.5 (Sim Replay Viewer):
    ReplayLogger   — per-tick JSONL writer for the web viewer
"""
from __future__ import annotations

from murimsim.replay import ReplayLogger
from murimsim.world import ResourceConfig, World, WorldStats

__all__ = [
    "World",
    "ResourceConfig",
    "WorldStats",
    "ReplayLogger",
]
