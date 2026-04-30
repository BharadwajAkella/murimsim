"""Monsters — non-agent hostile entities that share the world with agents.

A `Monster` lives on the world grid alongside agents but is NOT an `Agent`:
it has no hunger, cultivation, sect, reproduction, or learning. It runs a
hand-coded `step()` heuristic and exists primarily as an environmental
pressure for agents to react to (e.g. forming groups around a common enemy).

This module is structured so future monster kinds (wolves, poison-spitters,
mini-bosses) subclass `Monster` and override `step()`. The `MonsterRegistry`
holds all live monsters in an env.

For v17 the only kind is `BossMonster` — a single, hardpowered, permadeath
NPC dropped into combat episodes. When killed by agents it leaves a high
capacity loot stash that can be withdrawn by every agent who landed an
attack.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from murimsim.agent import Agent
    from murimsim.world import World

logger = logging.getLogger(__name__)

# ── Boss tunables ────────────────────────────────────────────────────────────
BOSS_BASE_HEALTH: float = 5.0           # 5× a normal agent (which starts at 1.0)
BOSS_BASE_STRENGTH: float = 1.6         # ~2× a fresh agent (~0.8 average)
BOSS_LOOT_FOOD: int = 30
BOSS_LOOT_QI: int = 10
BOSS_LOOT_MATERIALS: int = 5
BOSS_ATTACK_DAMAGE_SCALE: float = 0.4   # multiplied by boss.strength for damage
BOSS_ATTACK_MAX_DAMAGE: float = 0.6

# ── Minion tunables ──────────────────────────────────────────────────────────
# v22: Easier-to-beat NPC for forcing combat in resource-rich arena maps.
# A solo strong agent can kill one in 3-4 hits; two weak agents can co-kill.
MINION_BASE_HEALTH: float = 1.5
MINION_BASE_STRENGTH: float = 0.9
MINION_LOOT_FOOD: int = 8
MINION_LOOT_QI: int = 2
MINION_LOOT_MATERIALS: int = 1
MINION_ATTACK_DAMAGE_SCALE: float = 0.25
MINION_ATTACK_MAX_DAMAGE: float = 0.30


@dataclasses.dataclass
class Monster:
    """Base class for all non-agent hostile entities.

    Args:
        monster_id: Unique identifier, e.g. ``"boss_0"``.
        kind: Short string tag for telemetry/replay (e.g. ``"boss"``).
        position: ``(x, y)`` grid coordinates.
        health: Current HP. Death triggered when <= 0.
        max_health: Initial HP — used for normalised health in obs/replay.
        strength: Damage scaler used by `attack_damage()`.
        attackers: Set of `agent_id`s who have landed at least one attack
            this monster's lifetime — credited as loot participants on death.
        alive: True until killed.
    """

    monster_id: str
    kind: str
    position: tuple[int, int]
    health: float
    max_health: float
    strength: float
    attackers: set[str] = dataclasses.field(default_factory=set)
    alive: bool = True

    def take_damage(self, damage: float, attacker_id: str) -> bool:
        """Apply ``damage`` to this monster and credit ``attacker_id``.

        Returns True if the monster died from this hit.
        """
        if not self.alive or damage <= 0:
            return False
        self.attackers.add(attacker_id)
        self.health = max(0.0, self.health - damage)
        if self.health <= 0:
            self.alive = False
            logger.debug(
                "Monster %s (%s) killed; attackers=%s",
                self.monster_id, self.kind, sorted(self.attackers),
            )
            return True
        return False

    def attack_damage(self) -> float:
        """Damage dealt when this monster lands a hit. Override for variety."""
        return float(np.clip(
            self.strength * BOSS_ATTACK_DAMAGE_SCALE,
            0.0, BOSS_ATTACK_MAX_DAMAGE,
        ))

    def step(
        self,
        world: World,
        agents: list[Agent],
        rng: np.random.Generator,
    ) -> tuple[str | None, float]:
        """Advance the monster one tick. Returns (victim_agent_id, damage_dealt).

        Default behaviour: stand still and do nothing. Subclasses override.
        """
        return None, 0.0

    def to_replay_dict(self) -> dict:
        """Serialise for replay JSON."""
        return {
            "monster_id": self.monster_id,
            "kind": self.kind,
            "position": list(self.position),
            "health": float(self.health),
            "max_health": float(self.max_health),
            "strength": float(self.strength),
            "alive": bool(self.alive),
            "attackers": sorted(self.attackers),
        }


@dataclasses.dataclass
class BossMonster(Monster):
    """Aggressive permadeath boss — hunts the nearest agent and steals stashes.

    Heuristic per tick:
      1. Find nearest live agent (Chebyshev distance).
      2. If adjacent → attack it.
      3. Else → step one cell toward it (Chebyshev movement, 8-direction).
      4. If standing on any stash, steal its full contents (purely for menace —
         contents discarded; the menace is the loss, not the boss inventory).
    """

    def step(
        self,
        world: World,
        agents: list[Agent],
        rng: np.random.Generator,
    ) -> tuple[str | None, float]:
        if not self.alive:
            return None, 0.0
        target = self._nearest_live_agent(agents)
        if target is None:
            return None, 0.0
        bx, by = self.position
        tx, ty = target.position
        cheb = max(abs(tx - bx), abs(ty - by))
        if cheb <= 1:
            damage = self.attack_damage()
            target.health = max(0.0, target.health - damage)
            target._check_death("combat")
            return target.agent_id, damage
        # Step one cell toward target (8-direction)
        dx = int(np.sign(tx - bx))
        dy = int(np.sign(ty - by))
        new_x = int(np.clip(bx + dx, 0, world.grid_size - 1))
        new_y = int(np.clip(by + dy, 0, world.grid_size - 1))
        self.position = (new_x, new_y)
        return None, 0.0

    @staticmethod
    def _nearest_live_agent(agents: list[Agent]) -> Agent | None:
        live = [a for a in agents if a.alive]
        if not live:
            return None
        return live[0]  # placeholder; registry passes a sorted list


@dataclasses.dataclass
class MinionMonster(Monster):
    """v22 minion — weaker hunter that's actually killable by 1-2 agents.

    Same hunt heuristic as :class:`BossMonster` but with reduced HP/strength
    and per-instance damage caps. Drops a small loot stash on death so combat
    has a real payoff. Used in the dense ``arena_minion`` map for forcing
    contact-combat that is winnable, generating the joint-kill bonds the
    affinity system rewards.
    """

    def step(
        self,
        world: World,
        agents: list[Agent],
        rng: np.random.Generator,
    ) -> tuple[str | None, float]:
        if not self.alive:
            return None, 0.0
        target = self._nearest_live_agent(agents)
        if target is None:
            return None, 0.0
        bx, by = self.position
        tx, ty = target.position
        cheb = max(abs(tx - bx), abs(ty - by))
        if cheb <= 1:
            damage = self.attack_damage()
            target.health = max(0.0, target.health - damage)
            target._check_death("combat")
            return target.agent_id, damage
        dx = int(np.sign(tx - bx))
        dy = int(np.sign(ty - by))
        new_x = int(np.clip(bx + dx, 0, world.grid_size - 1))
        new_y = int(np.clip(by + dy, 0, world.grid_size - 1))
        self.position = (new_x, new_y)
        return None, 0.0

    def attack_damage(self) -> float:
        return float(np.clip(
            self.strength * MINION_ATTACK_DAMAGE_SCALE,
            0.0, MINION_ATTACK_MAX_DAMAGE,
        ))

    @staticmethod
    def _nearest_live_agent(agents: list[Agent]) -> Agent | None:
        live = [a for a in agents if a.alive]
        if not live:
            return None
        return live[0]


class MonsterRegistry:
    """Holds all monsters in a single environment instance."""

    def __init__(self) -> None:
        self._monsters: list[Monster] = []
        self._next_idx: int = 0

    def reset(self) -> None:
        self._monsters.clear()
        self._next_idx = 0

    def spawn_boss(self, position: tuple[int, int]) -> BossMonster:
        """Create a boss at ``position`` and register it. Returns the new boss."""
        mid = f"boss_{self._next_idx}"
        self._next_idx += 1
        boss = BossMonster(
            monster_id=mid,
            kind="boss",
            position=position,
            health=BOSS_BASE_HEALTH,
            max_health=BOSS_BASE_HEALTH,
            strength=BOSS_BASE_STRENGTH,
        )
        self._monsters.append(boss)
        logger.debug("Spawned boss %s at %s", mid, position)
        return boss

    def spawn_minion(self, position: tuple[int, int]) -> MinionMonster:
        """v22 — create a minion at ``position`` and register it."""
        mid = f"minion_{self._next_idx}"
        self._next_idx += 1
        minion = MinionMonster(
            monster_id=mid,
            kind="minion",
            position=position,
            health=MINION_BASE_HEALTH,
            max_health=MINION_BASE_HEALTH,
            strength=MINION_BASE_STRENGTH,
        )
        self._monsters.append(minion)
        logger.debug("Spawned minion %s at %s", mid, position)
        return minion

    # ── Queries ──────────────────────────────────────────────────────────────

    def all(self) -> list[Monster]:
        """Every monster, alive or dead (for replay)."""
        return list(self._monsters)

    def all_alive(self) -> list[Monster]:
        return [m for m in self._monsters if m.alive]

    def get_at(self, x: int, y: int) -> list[Monster]:
        return [m for m in self._monsters if m.alive and m.position == (x, y)]

    def get_adjacent_to(self, x: int, y: int) -> list[Monster]:
        """Live monsters within Chebyshev distance 1 of ``(x, y)``."""
        out: list[Monster] = []
        for m in self._monsters:
            if not m.alive:
                continue
            mx, my = m.position
            if max(abs(mx - x), abs(my - y)) <= 1:
                out.append(m)
        return out

    def nearest_live_to(self, x: int, y: int) -> Monster | None:
        """Closest live monster by Chebyshev distance, ties broken by id."""
        live = self.all_alive()
        if not live:
            return None
        return min(
            live,
            key=lambda m: (
                max(abs(m.position[0] - x), abs(m.position[1] - y)),
                m.monster_id,
            ),
        )

    # ── Tick ─────────────────────────────────────────────────────────────────

    def tick_all(
        self,
        world: World,
        agents: list[Agent],
        rng: np.random.Generator,
    ) -> list[tuple[str, str, float]]:
        """Advance every live monster one tick.

        Returns a list of ``(monster_id, victim_agent_id, damage)`` events.
        """
        events: list[tuple[str, str, float]] = []
        for m in self._monsters:
            if not m.alive:
                continue
            # Boss heuristic uses its own nearest-target logic — pass live list
            # sorted by distance from monster to make _nearest_live_agent
            # deterministic and correct.
            mx, my = m.position
            sorted_agents = sorted(
                [a for a in agents if a.alive],
                key=lambda a: (
                    max(abs(a.position[0] - mx), abs(a.position[1] - my)),
                    a.agent_id,
                ),
            )
            victim_id, damage = m.step(world, sorted_agents, rng)
            if victim_id is not None and damage > 0:
                events.append((m.monster_id, victim_id, damage))
        return events
