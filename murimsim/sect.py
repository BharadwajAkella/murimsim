"""sect.py — Sect (faction) configuration for MurimSim Phase 6a.

A *sect* is a named faction of agents that share a home territory on the map.
Each sect occupies a horizontal stripe of the 30×30 grid and is identified by
a unique color in the viewer.

Sects are data-only: they carry no RL logic. The RL environment uses
``SectConfig`` to constrain where agents spawn at episode reset.

Usage::

    from murimsim.sect import DEFAULT_SECTS, SectRegistry

    env = CombatEnv(config=cfg, sect_config=DEFAULT_SECTS.by_id("iron_fang"))
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class SectConfig:
    """Configuration for one sect (faction).

    All coordinates are in world-space: ``x`` is column, ``y`` is row
    (matching the ``(x, y)`` convention used throughout MurimSim).

    Args:
        sect_id:       Unique identifier, e.g. ``"iron_fang"``.
        name:          Human-readable display name.
        color:         Hex color string used by the viewer.
        home_x_range:  ``(x_min, x_max)`` inclusive column span for agent spawning.
        home_y_range:  ``(y_min, y_max)`` inclusive row span for agent spawning.
    """

    sect_id: str
    name: str
    color: str
    home_x_range: tuple[int, int]
    home_y_range: tuple[int, int]


@dataclasses.dataclass(frozen=True)
class SectRegistry:
    """Immutable registry of all active sects.

    Args:
        sects: Tuple of :class:`SectConfig` in definition order.
    """

    sects: tuple[SectConfig, ...]

    def by_id(self, sect_id: str) -> SectConfig:
        """Return the :class:`SectConfig` for *sect_id*.

        Raises:
            KeyError: If *sect_id* is not registered.
        """
        for s in self.sects:
            if s.sect_id == sect_id:
                return s
        raise KeyError(f"Unknown sect_id: {sect_id!r}")

    def by_index(self, i: int) -> SectConfig:
        """Return the i-th sect (wraps round-robin for ``i >= len(sects)``)."""
        return self.sects[i % len(self.sects)]


# ---------------------------------------------------------------------------
# Default three-sect configuration
# ---------------------------------------------------------------------------
# The 30×30 grid is divided into three equal horizontal stripes (y-bands).
# Iron Fang occupies the top (y 0–9), Jade Lotus the middle (y 10–19), and
# Shadow Root the bottom (y 20–29).  Full x-width for all three.

IRON_FANG = SectConfig(
    sect_id="iron_fang",
    name="Iron Fang",
    color="#e74c3c",          # red
    home_x_range=(0, 29),
    home_y_range=(0, 9),
)

JADE_LOTUS = SectConfig(
    sect_id="jade_lotus",
    name="Jade Lotus",
    color="#2ecc71",          # green
    home_x_range=(0, 29),
    home_y_range=(10, 19),
)

SHADOW_ROOT = SectConfig(
    sect_id="shadow_root",
    name="Shadow Root",
    color="#9b59b6",          # purple
    home_x_range=(0, 29),
    home_y_range=(20, 29),
)

DEFAULT_SECTS: SectRegistry = SectRegistry(sects=(IRON_FANG, JADE_LOTUS, SHADOW_ROOT))
