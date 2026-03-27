"""sect.py — Three-sect scaffold for MurimSim Phase 3+.

Each sect is a distinct population of agents with shared trait preferences.
In later phases, each sect trains in its own isolated environment; agents
that outperform a threshold migrate to a shared combat arena.

Architecture invariants:
- All trait values remain in [0.0, 1.0].
- Adding a new sect = YAML change only (via SectRegistry.from_config).
- Sects are isolated by default; the shared arena is a Phase 6 concern.
"""
from __future__ import annotations

import dataclasses
import copy
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Default sect definitions (overridable via config YAML)
# ---------------------------------------------------------------------------

_DEFAULT_SECTS: list[dict[str, Any]] = [
    {
        "name": "iron_fang",
        "color": "#ef4444",
        "n_agents": 10,
        "trait_prefs": {
            "strength": 0.8,
            "adventure_spirit": 0.4,
            "poison_resistance": 0.2,
            "flame_resistance": 0.3,
        },
    },
    {
        "name": "jade_lotus",
        "color": "#22c55e",
        "n_agents": 10,
        "trait_prefs": {
            "strength": 0.3,
            "adventure_spirit": 0.7,
            "poison_resistance": 0.7,
            "flame_resistance": 0.3,
        },
    },
    {
        "name": "shadow_root",
        "color": "#a78bfa",
        "n_agents": 10,
        "trait_prefs": {
            "strength": 0.5,
            "adventure_spirit": 0.8,
            "poison_resistance": 0.3,
            "flame_resistance": 0.7,
        },
    },
]


# ---------------------------------------------------------------------------
# SectConfig
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SectConfig:
    """Immutable configuration for one sect.

    Args:
        name:        Unique sect identifier (snake_case).
        color:       Hex colour used in the viewer (e.g. "#ef4444").
        n_agents:    Number of agents in this sect's training population.
        trait_prefs: Preferred initial trait values in [0.0, 1.0].
                     Keys: "strength", "adventure_spirit", plus any resistance name.
                     Absent keys default to 0.5.
    """

    name: str
    color: str
    n_agents: int
    trait_prefs: dict[str, float] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_agents < 1:
            raise ValueError(f"SectConfig.n_agents must be >= 1, got {self.n_agents}")
        for key, val in self.trait_prefs.items():
            if not 0.0 <= val <= 1.0:
                raise ValueError(
                    f"SectConfig trait_prefs[{key!r}] = {val} is outside [0, 1]"
                )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SectConfig":
        """Construct from a plain dict (e.g. loaded from YAML)."""
        return cls(
            name=data["name"],
            color=data.get("color", "#7c3aed"),
            n_agents=int(data.get("n_agents", 10)),
            trait_prefs={k: float(v) for k, v in data.get("trait_prefs", {}).items()},
        )

    def get_trait(self, key: str, default: float = 0.5) -> float:
        """Return a trait preference value, falling back to ``default``."""
        return self.trait_prefs.get(key, default)

    def sample_traits(self, rng: np.random.Generator, noise: float = 0.1) -> dict[str, float]:
        """Sample concrete trait values centred on this sect's preferences.

        Each trait is drawn from a truncated normal: mean = pref, std = noise,
        clipped to [0, 1].  This gives inter-agent diversity while preserving
        the sect's character.

        Args:
            rng:   Seeded RNG (never modify global state).
            noise: Std-dev of per-agent trait noise (default 0.1).

        Returns:
            Dict of trait_name → value in [0, 1].
        """
        traits: dict[str, float] = {}
        for key, pref in self.trait_prefs.items():
            raw = rng.normal(loc=pref, scale=noise)
            traits[key] = float(np.clip(raw, 0.0, 1.0))
        return traits


# ---------------------------------------------------------------------------
# SectRegistry
# ---------------------------------------------------------------------------


class SectRegistry:
    """Holds the full set of sect configurations for a training run.

    Usage::

        registry = SectRegistry.default()
        iron_fang = registry["iron_fang"]
        env = registry.make_env("jade_lotus", base_config, seed=42)

    The registry is iterable — ``for sect in registry`` yields ``SectConfig``
    objects in insertion order.
    """

    def __init__(self, sects: list[SectConfig]) -> None:
        if len(sects) == 0:
            raise ValueError("SectRegistry requires at least one sect.")
        names = [s.name for s in sects]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate sect names: {names}")
        self._sects: dict[str, SectConfig] = {s.name: s for s in sects}
        self._order: list[str] = names

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def default(cls) -> "SectRegistry":
        """Return a registry with the three canonical sects."""
        return cls([SectConfig.from_dict(d) for d in _DEFAULT_SECTS])

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SectRegistry":
        """Build from the ``sects`` key of a YAML config dict.

        Falls back to :meth:`default` when the key is absent, so existing
        configs that predate the sect system continue to work.
        """
        sect_list = config.get("sects", _DEFAULT_SECTS)
        return cls([SectConfig.from_dict(d) for d in sect_list])

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def __getitem__(self, name: str) -> SectConfig:
        try:
            return self._sects[name]
        except KeyError:
            raise KeyError(f"Unknown sect {name!r}. Available: {self._order}") from None

    def __iter__(self):
        return (self._sects[n] for n in self._order)

    def __len__(self) -> int:
        return len(self._sects)

    def names(self) -> list[str]:
        """Return sect names in insertion order."""
        return list(self._order)

    # ------------------------------------------------------------------
    # Environment factory
    # ------------------------------------------------------------------

    def make_env(
        self,
        sect_name: str,
        base_config: dict[str, Any],
        seed: int = 0,
        n_agents: int | None = None,
    ) -> Any:
        """Create a :class:`~murimsim.rl.multi_env.CombatEnv` for one sect.

        The environment uses a deep-copy of ``base_config`` so each sect's
        env is fully isolated.  Agents are spawned with trait values sampled
        from the sect's preferences.

        Args:
            sect_name:   Name of the sect to create an env for.
            base_config: Base YAML config dict (will not be mutated).
            seed:        World seed.
            n_agents:    Override agent count; defaults to ``sect.n_agents``.

        Returns:
            A freshly reset :class:`CombatEnv` instance tagged with
            ``env.sect_name`` for identification.
        """
        # Import here to avoid circular imports at module level
        from murimsim.rl.multi_env import CombatEnv

        sect = self[sect_name]
        cfg = copy.deepcopy(base_config)
        cfg["sect_name"] = sect.name
        cfg["sect_color"] = sect.color

        n = n_agents if n_agents is not None else sect.n_agents
        env = CombatEnv(config=cfg, n_agents=n, seed=seed)
        env.sect_name = sect.name  # type: ignore[attr-defined]
        env.sect_color = sect.color  # type: ignore[attr-defined]
        return env

    def make_all_envs(
        self,
        base_config: dict[str, Any],
        seed: int = 0,
    ) -> dict[str, Any]:
        """Create one isolated :class:`CombatEnv` per sect.

        Seeds are offset by sect index so each world diverges: sect 0 uses
        ``seed``, sect 1 uses ``seed + 1``, etc.

        Returns:
            Dict of sect_name → CombatEnv, in insertion order.
        """
        return {
            sect.name: self.make_env(sect.name, base_config, seed=seed + i)
            for i, sect in enumerate(self)
        }
