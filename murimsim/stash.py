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
STASH_QI_COST: int = 0   # free to deposit — qi cost removed to unblock stash usage

# Hard cap on total items stored per (owner, position) stash location.
# Deposit is blocked — returns None — when existing + incoming items would exceed this.
STASH_MAX_ITEMS: int = 20


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
        participants: Extra agent_ids allowed to withdraw alongside the owner.
        claim_unlock_step: v21b — for legacy_stash bequests. -1 means never
            opens to non-participants. Otherwise, once the env step counter
            reaches this value, ANY agent at the stash position can withdraw.
    """

    stash_id: str
    owner_id: str
    position: tuple[int, int]
    food: int = 0
    qi: int = 0
    materials: int = 0
    poison: int = 0
    flame: int = 0
    # Extra agent_ids allowed to withdraw from this stash alongside the owner.
    # Used for shared loot drops (e.g. boss-kill rewards split across attackers).
    # Empty list (default) means owner-only access.
    participants: list[str] = dataclasses.field(default_factory=list)
    # v21b: legacy stash unlock — see class docstring.
    claim_unlock_step: int = -1

    def total(self) -> int:
        """Return the total number of items in this stash."""
        return self.food + self.qi + self.materials + self.poison + self.flame

    def is_accessible_to(self, agent_id: str, current_step: int | None = None) -> bool:
        """True if ``agent_id`` is the owner, a registered participant, or the
        legacy unlock step has been reached for the given ``current_step``.
        """
        if agent_id == self.owner_id or agent_id in self.participants:
            return True
        if (
            self.claim_unlock_step >= 0
            and current_step is not None
            and current_step >= self.claim_unlock_step
        ):
            return True
        return False

    def as_dict(self) -> dict[str, int]:
        """Return resource counts as a plain dict."""
        return {
            "food": self.food,
            "qi": self.qi,
            "materials": self.materials,
            "poison": self.poison,
            "flame": self.flame,
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
            "flame": self.flame,
            "participants": list(self.participants),
            "claim_unlock_step": self.claim_unlock_step,
        }


class StashRegistry:
    """Manages all stashes in the world.

    Stashes are indexed by ``stash_id`` for O(1) lookup. Helper queries by
    position and owner are provided for environment logic.
    """

    def __init__(self) -> None:
        self._stashes: dict[str, Stash] = {}
        self._next_idx: dict[str, int] = {}

    def get_stash_total_at(self, agent_id: str, x: int, y: int) -> int:
        """Return total items across all own stashes at ``(x, y)``."""
        return sum(s.total() for s in self.get_own_stash_at(agent_id, x, y))

    def is_stash_full(self, agent_id: str, x: int, y: int, incoming: int = 0) -> bool:
        """Return True if depositing ``incoming`` items at ``(x, y)`` would exceed the cap."""
        return self.get_stash_total_at(agent_id, x, y) + incoming > STASH_MAX_ITEMS

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def deposit(self, agent: Agent, qi_cost: float = 0.2) -> Stash | None:
        """Create a stash at ``agent.position`` with the agent's full inventory.

        No qi cost (STASH_QI_COST = 0). Requires at least 1 food item to deposit
        (prevents empty no-op deposits).

        After a successful deposit all inventory is moved into the new stash and
        the agent's inventory is zeroed out.

        Args:
            agent:    The depositing agent.
            qi_cost:  Unused — kept for API compatibility.

        Returns:
            The newly created :class:`Stash`, or ``None`` if deposit failed
            (empty inventory).
        """
        if agent.inventory.food < 1:
            return None

        owner_id = agent.agent_id
        x, y = agent.position
        if self.is_stash_full(owner_id, x, y, incoming=agent.inventory.total()):
            logger.debug(
                "Agent %s deposit blocked: stash at %s would exceed cap (%d)",
                owner_id, agent.position, STASH_MAX_ITEMS,
            )
            return None

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
            flame=agent.inventory.flame,
        )

        agent.inventory.food = 0
        agent.inventory.qi = 0
        agent.inventory.materials = 0
        agent.inventory.poison = 0
        agent.inventory.flame = 0

        self._stashes[stash_id] = stash
        logger.debug("Agent %s deposited stash %s at %s", owner_id, stash_id, stash.position)
        return stash

    def withdraw(self, agent: Agent, current_step: int | None = None) -> bool:
        """Move all own stash contents at ``agent.position`` into ``agent.inventory``.

        Merges resources from every stash owned by this agent at the current
        position. Legacy stashes whose unlock step has been reached are also
        included when ``current_step`` is provided. All merged stashes are
        removed from the registry.

        Returns:
            ``True`` if at least one item was transferred, ``False`` otherwise.
        """
        own_stashes = self.get_own_stash_at(agent.agent_id, *agent.position, current_step)
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
            agent.inventory.flame += stash.flame
            del self._stashes[stash.stash_id]
            logger.debug("Agent %s withdrew stash %s", agent.agent_id, stash.stash_id)

        return transferred

    def withdraw_group(
        self, agent: Agent, group_member_ids: list[str], current_step: int | None = None,
    ) -> int:
        """Withdraw food from any group member's stash at ``agent.position``.

        Merges food-only from stashes owned by any member of ``group_member_ids``
        (including agent itself) at the agent's current position. Legacy stashes
        whose unlock step has been reached are also included for the agent itself
        (via ``get_own_stash_at(agent.agent_id, ..., current_step)``).
        Non-food items stay in the stash (food is the shared resource).

        Args:
            agent:            The withdrawing agent.
            group_member_ids: Agent IDs of all group members including ``agent``.
            current_step:     Current env step (enables legacy-stash unlock).

        Returns:
            Total food items transferred (0 if nothing was available).
        """
        food_transferred = 0
        pos = agent.position
        for mid in group_member_ids:
            # Only the agent itself benefits from legacy unlock here; group-mates'
            # legacy stashes are handled when those agents withdraw themselves.
            cs = current_step if mid == agent.agent_id else None
            stashes = self.get_own_stash_at(mid, *pos, cs)
            for stash in stashes:
                if stash.food > 0:
                    food_transferred += stash.food
                    agent.inventory.food += stash.food
                    stash.food = 0
                    if stash.total() == 0:
                        del self._stashes[stash.stash_id]
                    logger.debug(
                        "Agent %s group-withdrew food from stash %s (owner %s)",
                        agent.agent_id, stash.stash_id, mid,
                    )
        return food_transferred

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
        agent.inventory.flame += stash.flame
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

    def get_own_stash_at(
        self, agent_id: str, x: int, y: int, current_step: int | None = None,
    ) -> list[Stash]:
        """Return stashes at ``(x, y)`` accessible to ``agent_id`` (owner, participant,
        or legacy stash whose unlock step has been reached for ``current_step``)."""
        return [
            s for s in self._stashes.values()
            if s.is_accessible_to(agent_id, current_step) and s.position == (x, y)
        ]

    def get_enemy_stashes_at(
        self, agent_id: str, x: int, y: int, current_step: int | None = None,
    ) -> list[Stash]:
        """Return stashes at ``(x, y)`` NOT accessible to ``agent_id``."""
        return [
            s for s in self._stashes.values()
            if not s.is_accessible_to(agent_id, current_step) and s.position == (x, y)
        ]

    def get_stashes_for_owner(self, agent_id: str) -> list[Stash]:
        """Return all stashes owned by ``agent_id`` regardless of position."""
        return [s for s in self._stashes.values() if s.owner_id == agent_id]

    def register(self, stash: Stash) -> Stash:
        """Insert a pre-built stash (e.g. monster loot drop) into the registry.

        Bumps the per-owner index counter so subsequent owner deposits don't
        collide. Returns the registered stash for chaining.
        """
        self._stashes[stash.stash_id] = stash
        owner = stash.owner_id
        # Best-effort index advance (parses suffix if present)
        try:
            suffix = int(stash.stash_id.rsplit("_", 1)[-1])
            self._next_idx[owner] = max(self._next_idx.get(owner, 0), suffix + 1)
        except ValueError:
            self._next_idx[owner] = self._next_idx.get(owner, 0) + 1
        return stash

    def all_stashes(self) -> list[Stash]:
        """Return all registered stashes."""
        return list(self._stashes.values())
