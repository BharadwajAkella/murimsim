"""Action definitions for Phase 2+.

Phase 2 action space: Discrete(7)
Phase 3 will extend this to Discrete(9) by appending ATTACK and DEFEND.
Phase 5 extends to Discrete(14) by appending COLLABORATE and WALK_AWAY.
Phase 6 extends to Discrete(15) by appending TRAIN.
v18 (Phase 6 + qi-infused strikes) extends to Discrete(17) by appending
ATTACK_QI (1 qi) and ATTACK_BURST (3 qi). Action-space change is a hard break:
checkpoints from Phase 6 (Discrete(15)) cannot be warm-started into Phase 6 + qi.
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
    # v18: tiered qi-infused strikes — policy chooses how much qi to spend per attack
    ATTACK_QI    = 15  # ATTACK + spend 1 qi → +0.15 damage flat (after √strength scaling)
    ATTACK_BURST = 16  # ATTACK + spend 3 qi → +0.35 damage flat (high spend, finisher)


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
N_ACTIONS_PHASE6_QI: int = N_ACTIONS_PHASE6 + 2  # 17 (15 + attack_qi + attack_burst, v18)

# All ATTACK-family actions (basic + qi-infused tiers). Use this anywhere code
# needs to ask "is this a strike of any kind?" so adding a future tier only
# requires editing this tuple.
ATTACK_ACTIONS: tuple[Action, ...] = (
    Action.ATTACK,
    Action.ATTACK_QI,
    Action.ATTACK_BURST,
)

# Movement deltas: (dx, dy) for each move action
MOVE_DELTAS: dict[Action, tuple[int, int]] = {
    Action.MOVE_N: (0, -1),
    Action.MOVE_S: (0,  1),
    Action.MOVE_E: (1,  0),
    Action.MOVE_W: (-1, 0),
}
