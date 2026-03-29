"""Agent: individual simulation entity for Phase 2+.

All mutable stats are floats in [0.0, 1.0] unless stated otherwise.
Inventory counts are non-negative integers (unbounded).
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Hunger increment per tick (constant drain)
HUNGER_PER_TICK: float = 0.01

# How much hunger eating one food item restores (loaded from resource config at runtime)
DEFAULT_HUNGER_RESTORE: float = 0.3

# Resistance gained per survival event (generalised: poison eat, terrain traversal, etc.)
RESISTANCE_GAIN: float = 0.05

# Health lost per tick when hunger is full (starving)
STARVATION_DAMAGE_PER_TICK: float = 0.02

# Minimum health damage from eating poison (deprecated — immunity allowed at resistance=1.0)
POISON_MIN_DAMAGE: float = 0.0

# How much _intakes increments per exposure event (eat or traversal)
INTAKE_PER_EXPOSURE: float = 0.30

# How quickly _intakes decays per tick (full recovery in ~15 ticks)
INTAKE_DECAY_PER_TICK: float = 0.02

# Inheritance noise: std-dev of Gaussian perturbation on midpoint trait value
INHERIT_SIGMA: float = 0.05


def inherit_value(
    parent_a: float,
    parent_b: float,
    rng: np.random.Generator,
    sigma: float = INHERIT_SIGMA,
) -> float:
    """Compute an inherited trait value as the parents' midpoint plus Gaussian noise.

    This models Mendelian blending with random mutation. The result is clamped
    to the valid trait range [0, 1].

    Args:
        parent_a: Trait value from the first parent (float in [0, 1]).
        parent_b: Trait value from the second parent (float in [0, 1]).
        rng:      Seeded RNG — must be the environment's RNG for determinism.
        sigma:    Standard deviation of the Gaussian noise term. Defaults to
                  ``INHERIT_SIGMA`` (0.05).

    Returns:
        Inherited trait value clamped to [0, 1].
    """
    midpoint = (parent_a + parent_b) / 2.0
    noise = float(rng.normal(0.0, sigma))
    return float(np.clip(midpoint + noise, 0.0, 1.0))


@dataclasses.dataclass
class AgentInventory:
    """Counts of each resource type the agent is carrying."""

    food: int = 0
    qi: int = 0
    materials: int = 0
    poison: int = 0

    def total(self) -> int:
        return self.food + self.qi + self.materials + self.poison

    def as_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Agent:
    """A single simulation agent.

    Args:
        agent_id:         Unique string identifier.
        position:         (x, y) grid coordinates.
        health:           Current health in [0, 1]. 0 = dead.
        hunger:           Hunger level in [0, 1]. 1 = starving.
        strength:         Combat strength in [0, 1].
        adventure_spirit: Curiosity / exploration drive in [0, 1]. Scales the
                          intrinsic reward for visiting new tiles.
        sociability:      Social personality in [0, 1]. 0 = fully independent,
                          1 = highly social. Influences collaborate/fight/walk-away
                          decisions when two agents are in range of each other.
        resistances:      Per-stat resistance values in [0, 1]. Keys: "poison",
                          "flame", "qi_drain", etc. Higher = more resistant.
        inventory:        Carried resources.
        alive:            False once health reaches 0.
    """

    agent_id: str
    position: tuple[int, int]
    health: float
    hunger: float
    strength: float
    adventure_spirit: float = 0.5   # default mid-range; overridden by spawn()
    sociability: float = 0.5        # [0, 1]: 0 = fully independent, 1 = highly social
    resistances: dict[str, float] = dataclasses.field(default_factory=dict)
    inventory: AgentInventory = dataclasses.field(default_factory=AgentInventory)
    alive: bool = True
    sect_id: str = "none"           # assigned by CombatEnv when sect_config is set
    age: int = 0                    # ticks lived; death occurs at max_age (from config)
    # Tracks cumulative recent exposure per resistance_stat; decays per tick
    _intakes: dict[str, float] = dataclasses.field(
        default_factory=dict, init=False, repr=False
    )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def spawn_from_parents(
        cls,
        agent_id: str,
        position: tuple[int, int],
        parent1: "Agent",
        parent2: "Agent",
        rng: np.random.Generator,
        sigma: float = INHERIT_SIGMA,
    ) -> "Agent":
        """Create an offspring agent with traits inherited from two parents.

        Each trait is computed as the parents' midpoint plus Gaussian noise
        (via :func:`inherit_value`). Resistances are Lamarckian — acquired
        resistance from parents is passed to the offspring along with the noise.

        The offspring starts with full health, zero hunger, zero age, and
        inherits the parents' sect_id if they agree (otherwise "none").

        Args:
            agent_id: Unique identifier for the new agent.
            position: Starting (x, y) grid cell.
            parent1:  First parent agent.
            parent2:  Second parent agent.
            rng:      Seeded RNG for determinism.
            sigma:    Noise level for trait inheritance.

        Returns:
            A new :class:`Agent` with inherited traits.
        """
        all_resistance_keys = set(parent1.resistances) | set(parent2.resistances)
        inherited_resistances = {
            key: inherit_value(
                parent1.resistances.get(key, 0.0),
                parent2.resistances.get(key, 0.0),
                rng,
                sigma,
            )
            for key in all_resistance_keys
        }
        sect = parent1.sect_id if parent1.sect_id == parent2.sect_id else "none"
        return cls(
            agent_id=agent_id,
            position=position,
            health=1.0,
            hunger=0.0,
            strength=         inherit_value(parent1.strength,         parent2.strength,         rng, sigma),
            adventure_spirit= inherit_value(parent1.adventure_spirit, parent2.adventure_spirit, rng, sigma),
            sociability=      inherit_value(parent1.sociability,      parent2.sociability,      rng, sigma),
            resistances=inherited_resistances,
            sect_id=sect,
        )

    @classmethod
    def spawn(
        cls,
        agent_id: str,
        position: tuple[int, int],
        rng: np.random.Generator,
        config: dict[str, Any],
    ) -> Agent:
        """Create an agent with randomised initial stats from config ranges.

        Args:
            agent_id: Unique identifier.
            position: Starting (x, y) grid cell.
            rng:      Seeded RNG — must be the world's RNG for determinism.
            config:   Parsed agent config section from default.yaml.
        """
        agent_cfg = config.get("agent", {})
        pr_min  = float(agent_cfg.get("poison_resistance_min",  0.05))
        pr_max  = float(agent_cfg.get("poison_resistance_max",  0.30))
        fr_min  = float(agent_cfg.get("flame_resistance_min",   0.0))
        fr_max  = float(agent_cfg.get("flame_resistance_max",   0.10))
        qr_min  = float(agent_cfg.get("qi_drain_resistance_min", 0.0))
        qr_max  = float(agent_cfg.get("qi_drain_resistance_max", 0.20))
        str_min = float(agent_cfg.get("strength_min",           0.2))
        str_max = float(agent_cfg.get("strength_max",           0.8))
        as_min  = float(agent_cfg.get("adventure_spirit_min",   0.1))
        as_max  = float(agent_cfg.get("adventure_spirit_max",   0.9))
        soc_min = float(agent_cfg.get("sociability_min",        0.1))
        soc_max = float(agent_cfg.get("sociability_max",        0.9))

        return cls(
            agent_id=agent_id,
            position=position,
            health=1.0,
            hunger=0.0,
            strength=         float(np.clip(rng.uniform(str_min, str_max), 0.0, 1.0)),
            adventure_spirit= float(np.clip(rng.uniform(as_min,  as_max),  0.0, 1.0)),
            sociability=      float(np.clip(rng.uniform(soc_min, soc_max), 0.0, 1.0)),
            resistances={
                "poison":   float(np.clip(rng.uniform(pr_min, pr_max), 0.0, 1.0)),
                "flame":    float(np.clip(rng.uniform(fr_min, fr_max), 0.0, 1.0)),
                "qi_drain": float(np.clip(rng.uniform(qr_min, qr_max), 0.0, 1.0)),
            },
        )

    # ------------------------------------------------------------------
    # Tick update
    # ------------------------------------------------------------------

    def tick(self, max_age: int = 0) -> bool:
        """Advance one tick: hunger, starvation, intake decay, and aging.

        Args:
            max_age: If > 0, the agent dies of old age when ``self.age >= max_age``.
                     Pass 0 (the default) to disable age-based death.

        Returns:
            True if the agent died of old age this tick (for death-cause tracking).
        """
        if not self.alive:
            return False
        self.age += 1
        if max_age > 0 and self.age >= max_age:
            self.health = 0.0
            self.alive = False
            return True
        self.hunger = min(1.0, self.hunger + HUNGER_PER_TICK)
        if self.hunger >= 1.0:
            self.health = max(0.0, self.health - STARVATION_DAMAGE_PER_TICK)
        for stat in list(self._intakes):
            self._intakes[stat] = max(0.0, self._intakes[stat] - INTAKE_DECAY_PER_TICK)
        self._check_death()
        return False

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def move(self, dx: int, dy: int, grid_size: int) -> None:
        """Move by (dx, dy), clamped to grid bounds."""
        if not self.alive:
            return
        x, y = self.position
        self.position = (
            int(np.clip(x + dx, 0, grid_size - 1)),
            int(np.clip(y + dy, 0, grid_size - 1)),
        )

    def gather(self, resource_id: str) -> None:
        """Pick up one unit of resource_id into inventory.

        The world tile depletion is handled by the environment, not here.
        """
        if not self.alive:
            return
        field = _resource_to_inventory_field(resource_id)
        if field is not None:
            current = getattr(self.inventory, field)
            setattr(self.inventory, field, current + 1)

    def apply_hazard(self, resistance_stat: str, raw_damage: float) -> float:
        """Core hazard application: apply raw_damage reduced by resistance and intake.

        Shared by traversal effects (flame, mountain) and consumption effects (poison eat).
        Formula: actual_damage = raw_damage * (1 - effective_resistance)
        where   effective_resistance = resistance * max(0, 1 - intake)

        Side-effects:
          - Reduces self.health by actual_damage.
          - Increments _intakes[resistance_stat] by INTAKE_PER_EXPOSURE (chips armor).
          - Grows resistances[resistance_stat] on survival (Lamarckian).
          - Calls _check_death().

        Args:
            resistance_stat: Key in self.resistances / self._intakes (e.g. "poison", "flame").
            raw_damage:      Damage before resistance reduction (≥ 0).

        Returns:
            Actual health damage dealt.
        """
        if not self.alive:
            return 0.0
        resistance = self.resistances.get(resistance_stat, 0.0)
        intake = self._intakes.get(resistance_stat, 0.0)
        effective_resistance = resistance * max(0.0, 1.0 - intake)
        actual_damage = max(0.0, raw_damage * (1.0 - effective_resistance))
        self.health = max(0.0, self.health - actual_damage)
        self._intakes[resistance_stat] = min(1.0, intake + INTAKE_PER_EXPOSURE)
        if self.health > 0:
            self.resistances[resistance_stat] = min(
                1.0, resistance + RESISTANCE_GAIN * (1.0 - resistance)
            )
        self._check_death()
        return actual_damage

    def apply_traversal_effects(self, effects: list[dict]) -> float:
        """Apply on_enter traversal effects for a tile the agent just stepped onto.

        Each effect dict: {trigger, attribute, delta, resistance_stat}.
        Only trigger="on_enter" effects are processed. Delegates per-effect damage
        to apply_hazard() which handles resistance/intake/growth uniformly.

        Args:
            effects: List of effect dicts from world.get_traversal_effects().

        Returns:
            Total health damage actually dealt this call.
        """
        if not self.alive:
            return 0.0
        total_damage = 0.0
        for effect in effects:
            if effect.get("trigger") != "on_enter":
                continue
            attribute = effect["attribute"]
            resistance_stat = effect["resistance_stat"]
            raw_damage = abs(float(effect["delta"]))
            if attribute == "health":
                total_damage += self.apply_hazard(resistance_stat, raw_damage)
            else:
                logger.warning(
                    "apply_traversal_effects: attribute %r not yet tracked — no-op", attribute
                )
        return total_damage

    def eat(self, resource_configs: dict[str, Any]) -> float:
        """Consume one food item from inventory. Returns damage taken (0 if food).

        Eat priority: food first. If no food, tries poison (agent is desperate).
        If inventory is empty, no-op.

        Poison damage: max(0, potency - effective_resistance)
        Surviving poison: resistance increases by RESISTANCE_GAIN * (1 - resistance)

        Args:
            resource_configs: Mapping of resource_id → ResourceConfig (or dict with effect_params).

        Returns:
            Damage dealt this eat action (0.0 for food, >0 for poison).
        """
        if not self.alive:
            return 0.0

        damage = 0.0

        if self.inventory.food > 0:
            self.inventory.food -= 1
            food_cfg = resource_configs.get("food")
            restore = (
                float(food_cfg.effect_params.get("hunger_restore", DEFAULT_HUNGER_RESTORE))
                if food_cfg is not None
                else DEFAULT_HUNGER_RESTORE
            )
            self.hunger = max(0.0, self.hunger - restore)

        elif self.inventory.poison > 0:
            self.inventory.poison -= 1
            poison_cfg = resource_configs.get("poison")
            potency = (
                float(poison_cfg.effect_params.get("potency", 0.4))
                if poison_cfg is not None
                else 0.4
            )
            damage = self.apply_hazard("poison", potency)

        return damage

    def rest(self) -> None:
        """Recover a small amount of health. No-op if dead."""
        if not self.alive:
            return
        self.health = min(1.0, self.health + 0.02)

    # Cultivation rates (per action tick)
    TRAIN_RATE_QI_TILE: float = 0.01    # strength growth on a qi tile
    TRAIN_RATE_ANYWHERE: float = 0.002  # strength growth anywhere else

    def train(self, qi_field_value: float = 0.0) -> float:
        """Cultivate: grow strength based on the local qi influence field.

        Training rate interpolates linearly between ``TRAIN_RATE_ANYWHERE``
        (qi_field_value=0) and ``TRAIN_RATE_QI_TILE`` (qi_field_value=1.0).
        When multiple qi sources overlap their contributions stack, creating
        "qi nexus" zones with maximum training effectiveness.

        Args:
            qi_field_value: Normalised qi influence at the agent's tile [0, 1].
                            0.0 = no nearby qi, 1.0 = maximum (saturated field).

        Returns:
            Actual strength delta (≥ 0).
        """
        if not self.alive:
            return 0.0
        rate = (
            self.TRAIN_RATE_ANYWHERE
            + (self.TRAIN_RATE_QI_TILE - self.TRAIN_RATE_ANYWHERE) * float(qi_field_value)
        )
        delta = rate * (1.0 - self.strength)
        self.strength = min(1.0, self.strength + delta)
        return delta

    # ------------------------------------------------------------------
    # Serialisation (for replay logger)
    # ------------------------------------------------------------------

    def to_replay_dict(self, action: str = "rest", action_detail: str = "") -> dict[str, Any]:
        """Return a dict matching the replay format spec."""
        return {
            "id": self.agent_id,
            "sect": self.sect_id,
            "pos": list(self.position),
            "health": round(self.health, 4),
            "hunger": round(self.hunger, 4),
            "age": self.age,
            "resistances": {k: round(v, 4) for k, v in self.resistances.items()},
            "intakes": {k: round(v, 4) for k, v in self._intakes.items()},
            "adventure_spirit": round(self.adventure_spirit, 4),
            "sociability": round(self.sociability, 4),
            "action": action,
            "action_detail": action_detail,
            "alive": self.alive,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_death(self) -> None:
        if self.health <= 0.0:
            self.health = 0.0
            self.alive = False


def _resource_to_inventory_field(resource_id: str) -> str | None:
    """Map a resource ID to the corresponding AgentInventory field name."""
    mapping = {
        "food": "food",
        "qi": "qi",
        "materials": "materials",
        "poison": "poison",
        "mountain": None,
        "flame": None,
    }
    return mapping.get(resource_id)
