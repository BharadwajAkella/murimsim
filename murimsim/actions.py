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


# ---------------------------------------------------------------------------
# v24: split action space into body + social heads.
#
# COLLABORATE (legacy id 12) is the only action without a body cost — it's
# purely a social signal. Splitting it into its own head means the policy can
# pick e.g. (TRAIN, COLLABORATE) in the same tick instead of having to choose
# between training strength and forming a group.
#
# IDs in BodyAction are renumbered 0..15 so the policy head outputs a clean
# Discrete(16) over the body lane; the legacy ``Action`` enum stays unchanged
# so all existing CombatEnv code (and tests) keep working.
# ---------------------------------------------------------------------------


class BodyAction(IntEnum):
    """16 actions that consume the agent's body for one tick.

    Indices are renumbered for the joint-action policy head; use
    ``BODY_TO_LEGACY`` / ``LEGACY_TO_BODY`` to translate to the legacy
    ``Action`` enum that ``CombatEnv.step()`` expects.
    """

    MOVE_N = 0
    MOVE_S = 1
    MOVE_E = 2
    MOVE_W = 3
    GATHER = 4
    EAT = 5
    REST = 6
    ATTACK = 7
    DEFEND = 8
    DEPOSIT = 9
    WITHDRAW = 10
    STEAL = 11
    WALK_AWAY = 12
    TRAIN = 13
    ATTACK_QI = 14
    ATTACK_BURST = 15
    GIFT = 16  # Phase 8c.2: physical resource transfer — moved from social to body lane


class SocialAction(IntEnum):
    """4 social signals that fire alongside the body action each tick.

    Phase 7 added ``PROPOSE`` and ``ACCEPT`` for the courtship system. The two
    new values are appended (NOT inserted) so existing checkpoints whose social
    head was Discrete(2) can be warm-started by zero-padding the policy/value
    head — see ``trainer-zero-init-expand`` ticket.

    Phase 8c.2: GIFT was briefly here (Phase 8c) but moved to the body lane
    because it's a physical action (transfers inventory) — competing with
    pure social signals like COLLABORATE was a category error.
    """

    NOOP = 0
    COLLABORATE = 1
    PROPOSE = 2  # Phase 7: signal courtship intent toward an adjacent opposite-sex agent
    ACCEPT = 3   # Phase 7: accept (one of) the pending courtship proposal(s) against this agent
    PROPOSE_TRADE = 4  # Phase 8d: offer a 1-for-1 inventory swap to nearest adjacent agent
    ACCEPT_TRADE = 5   # Phase 8d: accept the best pending trade offer addressed to this agent
    REJECT_TRADE = 6   # Phase 8d: explicitly clear all pending trade offers addressed to this agent


N_BODY_ACTIONS: int = len(BodyAction)        # 17 (Phase 8c.2: was 16 — added GIFT)
N_SOCIAL_ACTIONS: int = len(SocialAction)    # 7 (Phase 8d: NOOP/COLLAB/PROPOSE/ACCEPT/PROPOSE_TRADE/ACCEPT_TRADE/REJECT_TRADE)
N_BODY_ACTIONS_PRE_PHASE8C2: int = 16        # pre-GIFT body width — used by warm-start padding
N_SOCIAL_ACTIONS_PRE_PHASE7: int = 2         # legacy social head width — used by warm-start padding
N_SOCIAL_ACTIONS_PRE_PHASE8C: int = 4        # post-Phase 7 / pre-Phase 8c width — used by warm-start padding
N_SOCIAL_ACTIONS_PRE_PHASE8D: int = 4        # post-8c.2 / pre-8d width (GIFT moved out, TRADE not yet in)
# Phase 8c.2 social width happens to equal pre-8c width (both 4) — GIFT round-tripped.

# Body actions map 1-to-1 with the legacy Action enum minus COLLABORATE.
# GIFT (BodyAction.GIFT) has no legacy Action equivalent: the legacy single-
# action step path doesn't dispatch it. In joint mode, GIFT is intercepted
# from body_actions and routed to ``_resolve_gift`` after the body step;
# the slot's body translation falls back to REST so step_all is a safe no-op.
_BODY_TO_LEGACY_PAIRS: tuple[tuple[BodyAction, Action], ...] = (
    (BodyAction.MOVE_N, Action.MOVE_N),
    (BodyAction.MOVE_S, Action.MOVE_S),
    (BodyAction.MOVE_E, Action.MOVE_E),
    (BodyAction.MOVE_W, Action.MOVE_W),
    (BodyAction.GATHER, Action.GATHER),
    (BodyAction.EAT, Action.EAT),
    (BodyAction.REST, Action.REST),
    (BodyAction.ATTACK, Action.ATTACK),
    (BodyAction.DEFEND, Action.DEFEND),
    (BodyAction.DEPOSIT, Action.DEPOSIT),
    (BodyAction.WITHDRAW, Action.WITHDRAW),
    (BodyAction.STEAL, Action.STEAL),
    (BodyAction.WALK_AWAY, Action.WALK_AWAY),
    (BodyAction.TRAIN, Action.TRAIN),
    (BodyAction.ATTACK_QI, Action.ATTACK_QI),
    (BodyAction.ATTACK_BURST, Action.ATTACK_BURST),
    (BodyAction.GIFT, Action.REST),   # GIFT body slot is a no-op in legacy step; joint path dispatches
)

BODY_TO_LEGACY: dict[int, int] = {b.value: a.value for b, a in _BODY_TO_LEGACY_PAIRS}
# LEGACY_TO_BODY only round-trips the 16 legacy actions; GIFT has no inverse.
LEGACY_TO_BODY: dict[int, int] = {
    a.value: b.value for b, a in _BODY_TO_LEGACY_PAIRS if b is not BodyAction.GIFT
}

assert len(BODY_TO_LEGACY) == N_BODY_ACTIONS
assert Action.COLLABORATE.value not in LEGACY_TO_BODY, (
    "COLLABORATE must NOT be in the body lane — it lives in SocialAction."
)
