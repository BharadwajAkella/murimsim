"""Action definitions for Phase 2+.

Phase 2 action space: Discrete(7)
Phase 3 will extend this to Discrete(9) by appending ATTACK and DEFEND.
"""
from __future__ import annotations

from enum import IntEnum


class Action(IntEnum):
    """Discrete action indices. Values are stable across phases — never reorder."""

    MOVE_N = 0
    MOVE_S = 1
    MOVE_E = 2
    MOVE_W = 3
    GATHER = 4
    EAT    = 5
    REST   = 6
    # Phase 3+
    ATTACK = 7
    DEFEND = 8
    # Phase 3 stash
    DEPOSIT  = 9   # drop current inventory contents into a stash at current position
    WITHDRAW = 10  # pick up from own stash at current position
    STEAL    = 11  # pick up from an enemy stash at current position


# Convenient subsets
PHASE2_ACTIONS: tuple[Action, ...] = (
    Action.MOVE_N,
    Action.MOVE_S,
    Action.MOVE_E,
    Action.MOVE_W,
    Action.GATHER,
    Action.EAT,
    Action.REST,
)

N_ACTIONS_PHASE2: int = len(PHASE2_ACTIONS)  # 7
N_ACTIONS_PHASE3: int = N_ACTIONS_PHASE2 + 2  # 9
N_ACTIONS_STASH: int = N_ACTIONS_PHASE2 + 2 + 3  # 12 (7 base + 2 combat + 3 stash)

# Movement deltas: (dx, dy) for each move action
MOVE_DELTAS: dict[Action, tuple[int, int]] = {
    Action.MOVE_N: (0, -1),
    Action.MOVE_S: (0,  1),
    Action.MOVE_E: (1,  0),
    Action.MOVE_W: (-1, 0),
}
