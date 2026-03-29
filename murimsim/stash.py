"""Personal stash system for Phase 3+.

A stash is a persistent deposit of resources on the world grid owned by an agent.
Any agent can steal from any visible stash. Placing a stash costs 1 qi item.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from murimsim.agent import Agent

logger = logging.getLogger(__name__)

# Qi cost (in inventory units) to place a stash
STASH_QI_COST: int = 1


@dataclasses.dataclass
class Stash:
    """A resource deposit placed on the world grid by an agent.

    Args:
        stash_id:  Unique identifier, e.g. ``"agent_0_stash_3"``.
        owner_id:  The ``agent_id`` of the owning agent.
        position:  ``(x, y)`` grid coordinates of the stash.
        food:      Food items stored.
        qi:        Qi items stored.
        materials: Material items stored.
        poison:    Poison items stored.
    """

    stash_id: str
    owner_id: str
    position: tuple[int, int]
    food: int = 0
    qi: int = 0
    materials: int = 0
    poison: int = 0

    def total(self) -> int:
        """Return the total number of items in this stash."""
        return self.food + self.qi + self.materials + self.poison

    def as_dict(self) -> dict[str, int]:
        """Return resource counts as a plain dict."""
        return {
            "food": self.food,
            "qi": self.qi,
            "materials": self.materials,
            "poison": self.poison,
        }

    def to_replay_dict(self) -> dict:
        """Return a dict suitable for replay serialisation."""
        return {
            "stash_id": self.stash_id,
            "owner_id": self.owner_id,
            "position": list(self.position),
            "food": self.food,
            "qi": self.qi,
            "materials": self.materials,
            "poison": self.poison,
        }


class StashRegistry:
    """Manages all stashes in the world.

    Stashes are indexed by ``stash_id`` for O(1) lookup. Helper queries by
    position and owner are provided for environment logic.
    """

    def __init__(self) -> None:
        self._stashes: dict[str, Stash] = {}
        self._next_idx: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def deposit(self, agent: Agent, qi_cost: float = 0.2) -> Stash | None:
        """Create a stash at ``agent.position`` with the agent's full inventory.

        Costs ``STASH_QI_COST`` qi items from the agent's inventory. If the
        agent has no qi, the action is a no-op and ``None`` is returned.

        After a successful deposit:
        - ``STASH_QI_COST`` qi is deducted from the agent's inventory.
        - ALL remaining inventory (food, remaining qi, materials, poison) is
          moved into the new stash.
        - The agent's inventory is zeroed out.

        Args:
            agent:    The depositing agent.
            qi_cost:  Reserved parameter (ignored; cost is always STASH_QI_COST).

        Returns:
            The newly created :class:`Stash`, or ``None`` if the deposit failed.
        """
        if agent.inventory.qi < STASH_QI_COST:
            return None

        # Deduct placement cost
        agent.inventory.qi -= STASH_QI_COST

        # Build stash id
        owner_id = agent.agent_id
        idx = self._next_idx.get(owner_id, 0)
        stash_id = f"{owner_id}_stash_{idx}"
        self._next_idx[owner_id] = idx + 1

        stash = Stash(
            stash_id=stash_id,
            owner_id=owner_id,
            position=agent.position,
            food=agent.inventory.food,
            qi=agent.inventory.qi,
            materials=agent.inventory.materials,
            poison=agent.inventory.poison,
        )

        # Zero out agent inventory
        agent.inventory.food = 0
        agent.inventory.qi = 0
        agent.inventory.materials = 0
        agent.inventory.poison = 0

        self._stashes[stash_id] = stash
        logger.debug("Agent %s deposited stash %s at %s", owner_id, stash_id, stash.position)
        return stash

    def withdraw(self, agent: Agent) -> bool:
        """Move all own stash contents at ``agent.position`` into ``agent.inventory``.

        Merges resources from every stash owned by this agent at the current
        position. All such stashes are removed from the registry.

        Returns:
            ``True`` if at least one item was transferred, ``False`` otherwise.
        """
        own_stashes = self.get_own_stash_at(agent.agent_id, *agent.position)
        if not own_stashes:
            return False

        transferred = False
        for stash in own_stashes:
            if stash.total() > 0:
                transferred = True
            agent.inventory.food += stash.food
            agent.inventory.qi += stash.qi
            agent.inventory.materials += stash.materials
            agent.inventory.poison += stash.poison
            del self._stashes[stash.stash_id]
            logger.debug("Agent %s withdrew stash %s", agent.agent_id, stash.stash_id)

        return transferred

    def steal(self, agent: Agent) -> Stash | None:
        """Take the first enemy stash at ``agent.position``.

        The stolen stash is removed from the registry and its contents are
        transferred to ``agent.inventory``.

        Returns:
            The (now-removed) stolen :class:`Stash`, or ``None`` if no enemy
            stash was present.
        """
        enemy_stashes = self.get_enemy_stashes_at(agent.agent_id, *agent.position)
        if not enemy_stashes:
            return None

        stash = enemy_stashes[0]
        agent.inventory.food += stash.food
        agent.inventory.qi += stash.qi
        agent.inventory.materials += stash.materials
        agent.inventory.poison += stash.poison
        del self._stashes[stash.stash_id]
        logger.debug("Agent %s stole stash %s from %s", agent.agent_id, stash.stash_id, stash.owner_id)
        return stash

    def reset(self) -> None:
        """Remove all stashes and reset index counters."""
        self._stashes.clear()
        self._next_idx.clear()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_stashes_at(self, x: int, y: int) -> list[Stash]:
        """Return all stashes at grid position ``(x, y)``."""
        return [s for s in self._stashes.values() if s.position == (x, y)]

    def get_own_stash_at(self, agent_id: str, x: int, y: int) -> list[Stash]:
        """Return stashes owned by ``agent_id`` at ``(x, y)``."""
        return [
            s for s in self._stashes.values()
            if s.owner_id == agent_id and s.position == (x, y)
        ]

    def get_enemy_stashes_at(self, agent_id: str, x: int, y: int) -> list[Stash]:
        """Return stashes at ``(x, y)`` NOT owned by ``agent_id``."""
        return [
            s for s in self._stashes.values()
            if s.owner_id != agent_id and s.position == (x, y)
        ]

    def get_stashes_for_owner(self, agent_id: str) -> list[Stash]:
        """Return all stashes owned by ``agent_id`` regardless of position."""
        return [s for s in self._stashes.values() if s.owner_id == agent_id]

    def all_stashes(self) -> list[Stash]:
        """Return all registered stashes."""
        return list(self._stashes.values())
