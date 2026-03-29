"""Action definitions for Phase 2+.

Phase 2 action space: Discrete(7)
Phase 3 will extend this to Discrete(9) by appending ATTACK and DEFEND.
Phase 5 extends to Discrete(14) by appending COLLABORATE and WALK_AWAY.
Phase 6 extends to Discrete(15) by appending TRAIN.
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
    # Phase 5: group dynamics
    COLLABORATE = 12  # signal willingness to form a group with the nearest adjacent agent
    WALK_AWAY   = 13  # move one step away from nearest adjacent agent; no-op if alone
    # Phase 6: cultivation
    TRAIN = 14  # grow strength: 0.01/tick on qi tile, 0.002/tick elsewhere


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
N_ACTIONS_PHASE5: int = N_ACTIONS_STASH + 2  # 14 (12 + collaborate + walk_away)
N_ACTIONS_PHASE6: int = N_ACTIONS_PHASE5 + 1  # 15 (14 + train)

# Movement deltas: (dx, dy) for each move action
MOVE_DELTAS: dict[Action, tuple[int, int]] = {
    Action.MOVE_N: (0, -1),
    Action.MOVE_S: (0,  1),
    Action.MOVE_E: (1,  0),
    Action.MOVE_W: (-1, 0),
}
