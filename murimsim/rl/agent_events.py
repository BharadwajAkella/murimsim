"""Per-agent per-step event accumulator (P1.1, IPPO migration prep).

Background
----------
The single-focal training loop tracks reward inputs as loose locals inside
``CombatEnv.step`` (``food_gathered``, ``hazard_damage``, ``damage_dealt``,
``defeat_bonus``, ``stash_bonus``, ``focal_betrayal``, ``group_formed``,
``damage_taken`` …) and feeds them straight to ``_compute_reward`` /
``_compute_combat_reward``. This works as long as exactly one agent (the
focal) has its reward computed per step.

For IPPO every slot needs its own reward each step, so we need a structured
container that:

* gives every reward input a single canonical name
* makes it trivial to attribute an event to the correct slot
* can be tested independently of the (still focal-coupled) reward function

This module only defines the dataclass and a small accumulator. Wiring it
into ``CombatEnv.step`` and ``_compute_reward`` is P1.2 — kept separate so
reward-semantics changes get reviewed on their own commit.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentStepEvents:
    """Reward-relevant events for a single agent during one ``env.step``.

    All fields default to a no-op value so an unset field never accidentally
    contributes to reward. Field names mirror the loose-locals naming used
    today in ``CombatEnv.step`` so the P1.2 refactor is mostly mechanical.

    Attributes:
        slot: Index of the agent slot in ``env._agents`` this event belongs to.
        food_gathered: Items picked up from food tiles (int >=0).
        hazard_damage: Damage taken from EAT-ing a poisoned/hazardous resource.
            Note: this is *taken* damage from the agent's own action, not
            damage from another agent's attack — that lives in ``damage_taken``.
        stash_bonus: Reward shaping from DEPOSIT / forage-outward / group withdraw.
        damage_dealt: Damage the agent inflicted via ATTACK* this step.
        damage_taken: Damage inflicted on this agent by other agents/monsters.
        defeated: True if the agent's ATTACK killed an opponent this step.
        group_formed: True if the agent's COLLABORATE created a new group.
        betrayal: True if the agent attacked someone it had high affinity with.
        action_was_redirected: True if the agent's chosen action was overridden
            by ``_redirect_invalid_action`` (used by future shaping; not a
            reward input today, but cheap to capture and useful for diagnostics).
    """

    slot: int
    food_gathered: int = 0
    hazard_damage: float = 0.0
    stash_bonus: float = 0.0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    defeated: bool = False
    group_formed: bool = False
    betrayal: bool = False
    action_was_redirected: bool = False

    def reset(self) -> None:
        """Zero all reward fields (slot is preserved)."""
        self.food_gathered = 0
        self.hazard_damage = 0.0
        self.stash_bonus = 0.0
        self.damage_dealt = 0.0
        self.damage_taken = 0.0
        self.defeated = False
        self.group_formed = False
        self.betrayal = False
        self.action_was_redirected = False


@dataclass
class StepEventBuffer:
    """Per-step events for every slot in an env.

    Indexed by slot id. ``reset_for_step()`` clears all entries before each
    step; field-by-field updates happen in ``CombatEnv.step``.
    """

    n_agents: int
    events: list[AgentStepEvents] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.events:
            self.events = [AgentStepEvents(slot=i) for i in range(self.n_agents)]

    def reset_for_step(self) -> None:
        for e in self.events:
            e.reset()

    def __getitem__(self, slot: int) -> AgentStepEvents:
        return self.events[slot]

    def __len__(self) -> int:
        return len(self.events)
