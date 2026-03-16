"""World: deterministic 2D grid world with a data-driven resource registry.

Design rules:
- All resources are declared in config YAML. Adding a new resource requires
  only a YAML change — no code changes here.
- Coordinates use (x, y) in the public API (x=column, y=row).
  Internal numpy arrays are indexed [row, col] = [y, x].
- Same seed always produces identical output across runs (determinism).
- Internal state is prefixed with `_`. Public API is clean and documented.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ResourceConfig:
    """Immutable specification for one resource type, loaded from config YAML."""

    id: str
    display_name: str
    regen_ticks: int        # ticks until a depleted tile respawns; 0 = permanent terrain
    spawn_density: float    # base probability [0,1] a tile has this resource at init
    effect: str             # 'positive' | 'negative' | 'neutral'
    effect_params: dict[str, Any]
    gatherable: bool = True         # if False, resource cannot be gathered (terrain)
    spawn_clusters: bool = False    # group spawn in spatial clusters (e.g. poison)
    cluster_count: int = 0
    cluster_radius: int = 0
    cluster_fill_prob: float = 0.5  # fill probability within cluster radius
    traversal_effects: tuple = dataclasses.field(default_factory=tuple)  # effect dicts for on_enter etc.

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ResourceConfig:
        """Construct from a raw YAML resource entry."""
        raw_effects = d.get("traversal_effects") or []
        return cls(
            id=str(d["id"]),
            display_name=str(d["display_name"]),
            regen_ticks=int(d["regen_ticks"]),
            spawn_density=float(d["spawn_density"]),
            effect=str(d["effect"]),
            effect_params=dict(d.get("effect_params") or {}),
            gatherable=bool(d.get("gatherable", True)),
            spawn_clusters=bool(d.get("spawn_clusters", False)),
            cluster_count=int(d.get("cluster_count", 0)),
            cluster_radius=int(d.get("cluster_radius", 0)),
            cluster_fill_prob=float(d.get("cluster_fill_prob", 0.5)),
            traversal_effects=tuple(dict(e) for e in raw_effects),
        )


@dataclasses.dataclass
class WorldStats:
    """Bookkeeping for resource lifecycle events.

    Invariant (holds at all times after any sequence of deplete/step calls):
        total_depleted[rid] == total_regenerated[rid] + currently_waiting_for_regen[rid]
    """

    initial_count: dict[str, int] = dataclasses.field(default_factory=dict)
    total_depleted: dict[str, int] = dataclasses.field(default_factory=dict)
    total_regenerated: dict[str, int] = dataclasses.field(default_factory=dict)


class World:
    """Tick-based 2D grid world.

    Each resource occupies its own layer:
        _grid[rid]:      float32 (H, W) — 1.0 = present, 0.0 = depleted/absent
        _countdown[rid]: int32   (H, W) — ticks remaining until regen; 0 = not counting

    All resources are declared in config YAML. No resource-specific logic lives here.

    Args:
        config: Parsed YAML dict (typically from config/default.yaml).
        rng:    Optional pre-seeded generator. If None, seeded from config world.seed.
    """

    def __init__(
        self,
        config: dict[str, Any],
        rng: np.random.Generator | None = None,
    ) -> None:
        world_cfg = config["world"]
        self.grid_size: int = int(world_cfg["grid_size"])
        self.seed: int = int(world_cfg["seed"])
        self.tick: int = 0

        self._rng: np.random.Generator = (
            rng if rng is not None else np.random.default_rng(self.seed)
        )
        self.resources: dict[str, ResourceConfig] = {
            r["id"]: ResourceConfig.from_dict(r) for r in config["resources"]
        }

        self._grid: dict[str, np.ndarray] = {}
        self._countdown: dict[str, np.ndarray] = {}
        self.stats = WorldStats()

        H = W = self.grid_size
        # Tracks which tiles already hold a resource — enforces one resource per tile.
        occupied = np.zeros((H, W), dtype=bool)
        for rid, rcfg in self.resources.items():
            self._grid[rid], self._countdown[rid] = self._spawn(rcfg, H, W, occupied)
            occupied |= self._grid[rid].astype(bool)
            self.stats.initial_count[rid] = int(self._grid[rid].sum())
            self.stats.total_depleted[rid] = 0
            self.stats.total_regenerated[rid] = 0

        logger.debug(
            "World initialised: grid=%dx%d seed=%d resources=%s",
            self.grid_size,
            self.grid_size,
            self.seed,
            list(self.resources),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance the simulation by one tick. Processes all resource regeneration."""
        self.tick += 1
        for rid in self.resources:
            self._process_regen(rid)

    def deplete(self, resource_id: str, x: int, y: int) -> None:
        """Remove the resource at grid position (x, y) and start its regen countdown.

        No-op if the tile is already depleted. Stats are only updated for an
        actual depletion event, keeping the bookkeeping invariant clean.

        Args:
            resource_id: Must be a registered resource ID.
            x: Column index in [0, grid_size).
            y: Row index in [0, grid_size).

        Raises:
            KeyError: Unknown resource_id.
            ValueError: (x, y) out of grid bounds.
        """
        if resource_id not in self.resources:
            raise KeyError(f"Unknown resource: {resource_id!r}")
        self._validate_position(x, y)

        if self._grid[resource_id][y, x] == 0.0:
            return  # already depleted — no-op keeps stats invariant clean

        self._grid[resource_id][y, x] = 0.0
        rcfg = self.resources[resource_id]
        if rcfg.regen_ticks > 0:
            # Regenerating resource: start countdown and track in stats
            self._countdown[resource_id][y, x] = rcfg.regen_ticks
            self.stats.total_depleted[resource_id] += 1
        # Permanent terrain (regen_ticks=0): mark absent but no countdown or stats

    def get_traversal_effects(self, x: int, y: int) -> list[dict]:
        """Return all traversal_effects for resources present at tile (x, y).

        Args:
            x: Column index.
            y: Row index.

        Returns:
            List of effect dicts from all resource configs present on this tile.
        """
        effects: list[dict] = []
        for rid, rcfg in self.resources.items():
            if self._grid[rid][y, x] > 0 and rcfg.traversal_effects:
                effects.extend(rcfg.traversal_effects)
        return effects

    def get_grid(self, resource_id: str) -> np.ndarray:
        """Return a snapshot copy of the resource grid (float32, shape [H, W]).

        Values: 1.0 = resource present, 0.0 = absent.
        A copy is returned so the caller cannot corrupt internal state.
        """
        return self._grid[resource_id].copy()

    def get_grid_view(self, resource_id: str) -> np.ndarray:
        """Return a direct view of the internal grid (read-only by convention).

        This is for performance-sensitive callers that only need to read.
        Do NOT mutate the returned array.
        """
        return self._grid[resource_id]

    def get_grids_view(self) -> dict[str, np.ndarray]:
        """Return views of all internal grids (read-only by convention)."""
        return self._grid

    def get_resource_ids(self) -> list[str]:
        """Ordered list of all registered resource IDs."""
        return list(self.resources.keys())

    def count(self, resource_id: str) -> int:
        """Current number of tiles that hold this resource."""
        return int(self._grid[resource_id].sum())

    def waiting_for_regen(self, resource_id: str) -> int:
        """Number of depleted tiles currently counting down to regeneration."""
        return int((self._countdown[resource_id] > 0).sum())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _spawn(
        self, rcfg: ResourceConfig, H: int, W: int, occupied: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Initialise the grid and countdown arrays for one resource type.

        Tiles already occupied by another resource are excluded, enforcing
        the invariant that each tile holds at most one resource type.
        """
        countdown = np.zeros((H, W), dtype=np.int32)
        if rcfg.spawn_clusters:
            grid = self._spawn_clustered(rcfg, H, W, occupied)
        else:
            mask = (self._rng.random((H, W)) < rcfg.spawn_density) & ~occupied
            grid = mask.astype(np.float32)
        return grid, countdown

    def _spawn_clustered(
        self, rcfg: ResourceConfig, H: int, W: int, occupied: np.ndarray
    ) -> np.ndarray:
        """Spawn a resource in spatial clusters (used for poison).

        Cluster centres are chosen randomly. Every tile within cluster_radius
        of a centre is filled with probability cluster_fill_prob. Coordinates
        wrap around the grid edges so clusters at boundaries are seamless.
        Already-occupied tiles are skipped to enforce one resource per tile.
        """
        grid = np.zeros((H, W), dtype=np.float32)
        n_clusters = max(rcfg.cluster_count, 1)
        radius = max(rcfg.cluster_radius, 1)
        fill_prob = float(np.clip(rcfg.cluster_fill_prob, 0.0, 1.0))

        centres_y = self._rng.integers(0, H, size=n_clusters)
        centres_x = self._rng.integers(0, W, size=n_clusters)

        for cy, cx in zip(centres_y, centres_x):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dy * dy + dx * dx <= radius * radius:
                        ny = int((cy + dy) % H)
                        nx = int((cx + dx) % W)
                        if not occupied[ny, nx] and self._rng.random() < fill_prob:
                            grid[ny, nx] = 1.0

        return grid

    def _process_regen(self, resource_id: str) -> None:
        """Decrement active regen countdowns; restore tiles whose countdown hits zero.

        Resources with regen_ticks == 0 are permanent terrain — their countdowns
        are never set, so nothing to process here.

        The active mask is computed BEFORE the decrement so that tiles moving
        from 1 → 0 are correctly identified as just-regenerated.
        """
        if self.resources[resource_id].regen_ticks == 0:
            return
        cd = self._countdown[resource_id]
        active = cd > 0           # boolean mask: tiles currently counting down
        cd[active] -= 1           # decrement in-place
        just_regenerated = active & (cd == 0)   # were counting, just reached zero

        n = int(just_regenerated.sum())
        if n > 0:
            self._grid[resource_id][just_regenerated] = 1.0
            self.stats.total_regenerated[resource_id] += n

    def _validate_position(self, x: int, y: int) -> None:
        """Raise ValueError if (x, y) is outside the grid boundaries."""
        gs = self.grid_size
        if not (0 <= x < gs and 0 <= y < gs):
            raise ValueError(
                f"Position ({x}, {y}) is out of bounds for grid_size={gs}"
            )
