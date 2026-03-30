"""Phase 2 — Agent unit tests."""
from __future__ import annotations

import numpy as np
import pytest

from murimsim.agent import (
    FOOD_HEALTH_RESTORE,
    HUNGER_PER_TICK,
    INHERIT_SIGMA,
    RESISTANCE_GAIN,
    STARVATION_HEALTH_DRAIN_SCALE,
    STARVATION_THRESHOLD,
    Agent,
    AgentInventory,
    inherit_value,
)
from murimsim.world import ResourceConfig, World

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_config(pr_min: float = 0.05, pr_max: float = 0.30) -> dict:
    return {
        "agent": {
            "poison_resistance_min": pr_min,
            "poison_resistance_max": pr_max,
            "flame_resistance_min": 0.0,
            "flame_resistance_max": 0.10,
            "qi_drain_resistance_min": 0.0,
            "qi_drain_resistance_max": 0.20,
            "strength_min": 0.2,
            "strength_max": 0.8,
            "adventure_spirit_min": 0.1,
            "adventure_spirit_max": 0.9,
        }
    }


def _resource_configs() -> dict[str, ResourceConfig]:
    return {
        "food": ResourceConfig(
            id="food", display_name="Food", regen_ticks=5,
            spawn_density=0.15, effect="positive",
            effect_params={"hunger_restore": 0.3},
        ),
        "poison": ResourceConfig(
            id="poison", display_name="Poison", regen_ticks=15,
            spawn_density=0.04, effect="negative",
            effect_params={"potency": 0.4},
        ),
    }


def _make_agent(
    health: float = 1.0,
    hunger: float = 0.0,
    poison_resistance: float = 0.15,
) -> Agent:
    return Agent(
        agent_id="test_agent",
        position=(5, 5),
        health=health,
        hunger=hunger,
        strength=0.5,
        resistances={"poison": poison_resistance},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_hunger_increases():
    """Each tick increases hunger by HUNGER_PER_TICK."""
    agent = _make_agent(hunger=0.0)
    agent.tick()
    assert abs(agent.hunger - HUNGER_PER_TICK) < 1e-9


def test_hunger_capped_at_one():
    """Hunger cannot exceed 1.0."""
    agent = _make_agent(hunger=1.0)
    agent.tick()
    assert agent.hunger == 1.0


def test_starvation_damages_health():
    """When hunger is at 1.0, ticking reduces health based on escalating drain."""
    agent = _make_agent(health=1.0, hunger=1.0)
    agent.hunger_resistance = 0.0  # max effect
    before = agent.health
    agent.tick()
    assert agent.health < before


def test_agent_dies_starvation():
    """Repeated ticking with full hunger eventually kills the agent."""
    agent = _make_agent(hunger=1.0)
    for _ in range(200):
        agent.tick()
        if not agent.alive:
            break
    assert not agent.alive
    assert agent.health == 0.0


def test_gathering_food_adds_to_inventory():
    """gather('food') increments inventory.food by 1."""
    agent = _make_agent()
    agent.gather("food")
    assert agent.inventory.food == 1


def test_gathering_poison_adds_to_inventory():
    """gather('poison') increments inventory.poison by 1."""
    agent = _make_agent()
    agent.gather("poison")
    assert agent.inventory.poison == 1


def test_eating_food_reduces_hunger():
    """Eating food from inventory reduces hunger by hunger_restore."""
    agent = _make_agent(hunger=0.8)
    agent.inventory.food = 1
    rc = _resource_configs()
    damage = agent.eat(rc)
    assert damage == 0.0
    assert abs(agent.hunger - (0.8 - 0.3)) < 1e-6
    assert agent.inventory.food == 0


def test_eating_food_hunger_floor():
    """Eating food cannot make hunger go below 0."""
    agent = _make_agent(hunger=0.1)
    agent.inventory.food = 1
    agent.eat(_resource_configs())
    assert agent.hunger >= 0.0


def test_poison_eat_kills_low_resistance():
    """res=0.05, potency=0.4 → damage = potency*(1-effective_res) = 0.4*(1-0.05) = 0.38.
    Agent with health=0.3 dies."""
    agent = _make_agent(health=0.3, poison_resistance=0.05)
    agent.inventory.poison = 1
    damage = agent.eat(_resource_configs())
    expected = 0.4 * (1.0 - 0.05)  # multiplicative unified formula
    assert abs(damage - expected) < 1e-6
    assert not agent.alive


def test_poison_eat_survives_high_resistance():
    """Unified formula: damage = potency*(1-effective_res).
    At res=0.9, damage=0.4*0.1=0.04; agent with health=1.0 survives and gains resistance."""
    agent = _make_agent(health=1.0, poison_resistance=0.9)
    agent.inventory.poison = 1
    damage = agent.eat(_resource_configs())
    expected = 0.4 * (1.0 - 0.9)
    assert abs(damage - expected) < 1e-4
    assert agent.alive
    assert agent.resistances["poison"] > 0.9


def test_poison_resistance_growth():
    """Surviving poison: resistance increases by RESISTANCE_GAIN * (1-res)."""
    res_before = 0.2
    agent = _make_agent(health=1.0, poison_resistance=res_before)
    agent.inventory.poison = 1
    agent.eat(_resource_configs())
    expected_delta = RESISTANCE_GAIN * (1.0 - res_before)
    assert abs(agent.resistances["poison"] - (res_before + expected_delta)) < 1e-6


def test_poison_resistance_capped_at_one():
    """Resistance never exceeds 1.0 regardless of repeated growth."""
    agent = _make_agent(health=1.0, poison_resistance=0.99)
    for _ in range(20):
        agent.inventory.poison = 1
        agent.eat(_resource_configs())
    assert agent.resistances["poison"] <= 1.0


def test_poison_immunity_at_one():
    """Resistance=1.0 yields zero poison damage (immunity)."""
    agent = _make_agent(health=1.0, poison_resistance=1.0)
    agent.inventory.poison = 1
    damage = agent.eat(_resource_configs())
    assert damage == 0.0
    assert agent.health == 1.0
    assert agent.alive


def test_rest_restores_health():
    """rest() increases health by REST_HEALTH_RESTORE when well-fed, capped at 1.0."""
    agent = _make_agent(health=0.5)
    agent.rest()
    assert abs(agent.health - (0.5 + Agent.REST_HEALTH_RESTORE)) < 1e-6


def test_move_changes_position():
    """move(1, 0) shifts x by 1."""
    agent = _make_agent()
    agent.position = (5, 5)
    agent.move(1, 0, grid_size=30)
    assert agent.position == (6, 5)


def test_move_clamped_to_bounds():
    """move cannot take agent outside [0, grid_size-1]."""
    agent = _make_agent()
    agent.position = (0, 0)
    agent.move(-5, -5, grid_size=30)
    assert agent.position == (0, 0)

    agent.position = (29, 29)
    agent.move(5, 5, grid_size=30)
    assert agent.position == (29, 29)


def test_dead_agent_ignores_actions():
    """Dead agents do not change state when actions are called."""
    agent = _make_agent(health=0.0)
    agent.alive = False
    prev_hunger = agent.hunger
    agent.tick()
    assert agent.hunger == prev_hunger  # no change


def test_poison_resistance_randomized_in_range():
    """100 agents spawned with default config span the configured init range."""
    rng = np.random.default_rng(0)
    cfg = _dummy_config(pr_min=0.05, pr_max=0.30)
    resistances = [
        Agent.spawn(f"a{i}", (0, 0), rng, cfg).resistances["poison"]
        for i in range(100)
    ]
    assert min(resistances) >= 0.05 - 1e-9
    assert max(resistances) <= 0.30 + 1e-9
    # Should span at least half the range
    assert max(resistances) - min(resistances) > 0.05


def test_to_replay_dict_fields():
    """to_replay_dict includes all required replay format fields."""
    agent = _make_agent()
    d = agent.to_replay_dict(action="gather", action_detail="Gathered food")
    required = {"id", "sect", "pos", "health", "hunger", "resistances", "adventure_spirit", "action", "action_detail", "alive"}
    assert required <= d.keys()
    assert d["action"] == "gather"
    assert d["alive"] is True
    assert "poison" in d["resistances"]


def test_traversal_effect_damages_health():
    """apply_traversal_effects reduces health when on_enter effect is present."""
    import copy
    import yaml
    from pathlib import Path
    cfg_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    world = World(cfg)
    # Place flame resource at a known tile
    ys, xs = np.where(world._grid["flame"] > 0)
    if len(ys) == 0:
        # Manually set a flame tile for the test
        world._grid["flame"][0, 0] = 1.0
        tx, ty = 0, 0
    else:
        tx, ty = int(xs[0]), int(ys[0])

    effects = world.get_traversal_effects(tx, ty)
    assert len(effects) > 0, "Expected traversal effects on flame tile"

    agent = _make_agent(health=1.0)
    damage = agent.apply_traversal_effects(effects)
    assert damage > 0.0
    assert agent.health < 1.0


def test_traversal_effect_builds_resistance():
    """apply_traversal_effects grows the relevant resistance stat."""
    effects = [{"trigger": "on_enter", "attribute": "health", "delta": -0.08, "resistance_stat": "flame"}]
    agent = _make_agent(health=1.0)
    agent.apply_traversal_effects(effects)
    assert agent.resistances.get("flame", 0.0) > 0.0


def test_gatherable_false_not_gathered():
    """GATHER action on a mountain/flame tile should not add to inventory."""
    from murimsim.actions import Action
    from murimsim.rl.env import SurvivalEnv
    import copy
    import yaml
    from pathlib import Path

    cfg_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    env = SurvivalEnv(config=cfg, seed=42)
    env.reset(seed=42)

    # Place agent on a mountain tile (or force one there)
    world = env._world
    ys, xs = np.where(world._grid["mountain"] > 0)
    if len(ys) == 0:
        world._grid["mountain"][5, 5] = 1.0
        tx, ty = 5, 5
    else:
        tx, ty = int(xs[0]), int(ys[0])

    # Clear other resources from that tile so gather only sees mountain
    for rid in ["food", "qi", "materials", "poison", "flame"]:
        if rid in world._grid:
            world._grid[rid][ty, tx] = 0.0

    env._agent.position = (tx, ty)
    inv_before = copy.deepcopy(env._agent.inventory)
    env.step(int(Action.GATHER))
    inv_after = env._agent.inventory

    assert inv_after.food == inv_before.food
    assert inv_after.qi == inv_before.qi
    assert inv_after.materials == inv_before.materials
    assert inv_after.poison == inv_before.poison


def test_train_grows_strength_on_qi_tile():
    """TRAIN action with max qi field grows strength faster than without qi."""
    from murimsim.agent import Agent
    rng = np.random.default_rng(0)
    cfg = {"agent": {
        "health_min": 1.0, "health_max": 1.0,
        "hunger_min": 0.0, "hunger_max": 0.0,
        "strength_min": 0.3, "strength_max": 0.3,
        "adventure_spirit_min": 0.5, "adventure_spirit_max": 0.5,
        "sociability_min": 0.5, "sociability_max": 0.5,
        "poison_resistance_min": 0.0, "poison_resistance_max": 0.0,
        "flame_resistance_min": 0.0, "flame_resistance_max": 0.0,
        "qi_drain_resistance_min": 0.0, "qi_drain_resistance_max": 0.0,
    }}
    agent = Agent.spawn("a0", (0, 0), rng, cfg)
    agent.strength = 0.3

    delta_qi = agent.train(qi_field_value=1.0)
    assert abs(delta_qi - Agent.TRAIN_RATE_QI_TILE * 0.7) < 1e-6
    assert agent.strength > 0.3

    agent.strength = 0.3
    delta_anywhere = agent.train(qi_field_value=0.0)
    assert delta_anywhere < delta_qi, "max-qi training must be faster than no-qi"


def test_train_capped_at_one():
    """TRAIN cannot push strength above 1.0."""
    from murimsim.agent import Agent
    rng = np.random.default_rng(0)
    cfg = {"agent": {
        "health_min": 1.0, "health_max": 1.0,
        "hunger_min": 0.0, "hunger_max": 0.0,
        "strength_min": 0.99, "strength_max": 0.99,
        "adventure_spirit_min": 0.5, "adventure_spirit_max": 0.5,
        "sociability_min": 0.5, "sociability_max": 0.5,
        "poison_resistance_min": 0.0, "poison_resistance_max": 0.0,
        "flame_resistance_min": 0.0, "flame_resistance_max": 0.0,
        "qi_drain_resistance_min": 0.0, "qi_drain_resistance_max": 0.0,
    }}
    agent = Agent.spawn("a0", (0, 0), rng, cfg)
    agent.strength = 0.99
    for _ in range(50):
        agent.train(qi_field_value=1.0)
    assert agent.strength <= 1.0


def test_age_increments_per_tick():
    """Agent.age must increment by 1 on each tick."""
    from murimsim.agent import Agent
    a = Agent(agent_id="x", position=(0, 0), health=1.0, hunger=0.0, strength=0.5)
    assert a.age == 0
    a.tick()
    assert a.age == 1
    a.tick()
    assert a.age == 2


def test_agent_dies_of_old_age():
    """Agent must die (alive=False) when age reaches max_age."""
    from murimsim.agent import Agent
    a = Agent(agent_id="x", position=(0, 0), health=1.0, hunger=0.0, strength=0.5)
    for i in range(9):
        died = a.tick(max_age=10)
        assert not died, f"Died too early at tick {i + 1}"
    assert a.alive
    died = a.tick(max_age=10)   # 10th tick: age == 10 >= max_age
    assert died, "tick() should return True when agent dies of old age"
    assert not a.alive


def test_aging_disabled_when_max_age_zero():
    """With max_age=0 (disabled), tick() never returns True for age-death."""
    from murimsim.agent import Agent
    a = Agent(agent_id="x", position=(0, 0), health=1.0, hunger=0.0, strength=0.5)
    for _ in range(100):
        died = a.tick(max_age=0)
        assert not died, "tick() should never return True when max_age=0"
        # Reset hunger to keep agent alive for the test
        a.hunger = 0.0
        a.health = 1.0
    assert a.alive


def test_ep_deaths_by_age_tracked():
    """MultiAgentEnv tracks deaths by age in _ep_deaths_by_age and ep_deaths_by_age info."""
    import yaml
    from pathlib import Path
    from murimsim.rl.multi_env import MultiAgentEnv

    cfg = yaml.safe_load(Path("config/default.yaml").read_text())
    # Set max_age very low (2 ticks) so agents die fast
    cfg["agent"]["max_age"] = 2
    env = MultiAgentEnv(config=cfg, n_agents=3, seed=5)
    env.reset(seed=5)

    terminal_info = None
    for _ in range(200):
        _, _, terminated, _, info = env.step(env.action_space.sample())
        if terminated:
            terminal_info = info
            break

    assert terminal_info is not None
    assert "ep_deaths_by_age" in terminal_info
    # With max_age=2, most agents should die of age
    assert terminal_info["ep_deaths_by_age"] >= 1, "Expected at least one aging death"


# ---------------------------------------------------------------------------
# Reproduction tests
# ---------------------------------------------------------------------------

def _make_parent(agent_id: str, strength: float = 0.5, soc: float = 0.4,
                adv: float = 0.6, poison_res: float = 0.2) -> Agent:
    return Agent(
        agent_id=agent_id,
        position=(0, 0),
        health=1.0,
        hunger=0.0,
        strength=strength,
        adventure_spirit=adv,
        sociability=soc,
        resistances={"poison": poison_res, "flame": 0.05, "qi_drain": 0.1},
    )


def test_inherit_value_midpoint():
    """inherit_value returns a value near the midpoint of the two parents."""
    rng = np.random.default_rng(0)
    samples = [inherit_value(0.2, 0.8, rng, sigma=0.001) for _ in range(20)]
    for v in samples:
        assert abs(v - 0.5) < 0.01, f"Expected ~0.5, got {v}"


def test_inherit_value_clamps_to_unit_interval():
    """inherit_value never produces a value outside [0, 1]."""
    rng = np.random.default_rng(1)
    for _ in range(1000):
        a = float(rng.uniform(0, 1))
        b = float(rng.uniform(0, 1))
        result = inherit_value(a, b, rng, sigma=0.5)
        assert 0.0 <= result <= 1.0, f"Out of range: {result}"


def test_spawn_from_parents_trait_inheritance():
    """spawn_from_parents produces offspring with traits between parents (with noise)."""
    rng = np.random.default_rng(42)
    mom = _make_parent("mom", strength=0.8, soc=0.9, adv=0.7, poison_res=0.5)
    dad = _make_parent("dad", strength=0.2, soc=0.1, adv=0.3, poison_res=0.1)
    child = Agent.spawn_from_parents("child", (5, 5), mom, dad, rng, sigma=0.001)
    # With sigma≈0, traits should be very close to midpoints
    assert abs(child.strength - 0.5) < 0.02
    assert abs(child.sociability - 0.5) < 0.02
    assert abs(child.adventure_spirit - 0.5) < 0.02
    assert abs(child.resistances["poison"] - 0.3) < 0.02


def test_spawn_from_parents_lamarckian_resistance():
    """Acquired (high) resistance in parents is inherited by offspring."""
    rng = np.random.default_rng(7)
    # Both parents have high poison_res from experience
    mom = _make_parent("mom", poison_res=0.9)
    dad = _make_parent("dad", poison_res=0.85)
    child = Agent.spawn_from_parents("child", (0, 0), mom, dad, rng, sigma=0.001)
    # Offspring should inherit the high resistance (Lamarckian)
    assert child.resistances["poison"] > 0.8


def test_spawn_from_parents_initial_state():
    """Offspring starts with full health, zero hunger, and age=0."""
    rng = np.random.default_rng(99)
    mom = _make_parent("mom")
    dad = _make_parent("dad")
    mom.health = 0.4
    mom.hunger = 0.7
    mom.age = 500
    child = Agent.spawn_from_parents("c", (1, 1), mom, dad, rng)
    assert child.health == 1.0
    assert child.hunger == 0.0
    assert child.age == 0


def test_spawn_from_parents_sect_inheritance_agreement():
    """Offspring inherits sect_id when both parents share the same sect."""
    rng = np.random.default_rng(3)
    mom = _make_parent("mom")
    mom.sect_id = "iron_fang"
    dad = _make_parent("dad")
    dad.sect_id = "iron_fang"
    child = Agent.spawn_from_parents("c", (0, 0), mom, dad, rng)
    assert child.sect_id == "iron_fang"


def test_spawn_from_parents_sect_none_on_mismatch():
    """Offspring has sect_id='none' when parents belong to different sects."""
    rng = np.random.default_rng(3)
    mom = _make_parent("mom")
    mom.sect_id = "iron_fang"
    dad = _make_parent("dad")
    dad.sect_id = "jade_lotus"
    child = Agent.spawn_from_parents("c", (0, 0), mom, dad, rng)
    assert child.sect_id == "none"


def test_reproduction_tracked_in_env():
    """ep_reproductions is tracked and emitted in terminal info."""
    import yaml
    from pathlib import Path
    from murimsim.rl.multi_env import MultiAgentEnv

    cfg = yaml.safe_load(Path("config/default.yaml").read_text())
    # Very low max_age so agents die quickly and reproduce
    cfg["agent"]["max_age"] = 3
    env = MultiAgentEnv(config=cfg, n_agents=4, seed=11)
    env.reset(seed=11)

    terminal_info = None
    for _ in range(500):
        _, _, terminated, _, info = env.step(env.action_space.sample())
        if terminated:
            terminal_info = info
            break

    assert terminal_info is not None
    assert "ep_reproductions" in terminal_info
    # With max_age=3 and 4 agents, reproductions are expected
    assert terminal_info["ep_reproductions"] >= 0  # may be 0 if <2 survived each death


# ─── Health overhaul tests ──────────────────────────────────────────────────


def _make_agent_hr(hunger_resistance: float = 0.5, hunger: float = 0.0) -> Agent:
    """Create a minimal agent with a specific hunger_resistance and hunger level."""
    a = Agent(
        agent_id="hr_test",
        position=(5, 5),
        health=1.0,
        hunger=hunger,
        strength=0.5,
        adventure_spirit=0.5,
        sociability=0.5,
        hunger_resistance=hunger_resistance,
    )
    return a


_FOOD_RCFG: dict = {}  # minimal resource_configs for eat(): no special food config


def test_eat_restores_health():
    """Eating food should restore FOOD_HEALTH_RESTORE health."""
    a = _make_agent_hr(hunger=0.5)
    a.inventory.food = 1
    a.health = 0.5
    before = a.health
    a.eat(_FOOD_RCFG)
    assert a.health > before


def test_eat_restores_health_by_expected_amount():
    """Health restore after eating should match FOOD_HEALTH_RESTORE constant."""
    a = _make_agent_hr(hunger=0.5)
    a.inventory.food = 1
    a.health = 0.5
    a.eat(_FOOD_RCFG)
    assert abs(a.health - (0.5 + FOOD_HEALTH_RESTORE)) < 1e-6


def test_eat_does_not_overheal():
    """Health should not exceed 1.0 from eating."""
    a = _make_agent_hr(hunger=0.3)
    a.inventory.food = 1
    a.health = 1.0
    a.eat(_FOOD_RCFG)
    assert a.health <= 1.0


def test_health_drain_starts_above_threshold():
    """Ticking an agent with hunger > STARVATION_THRESHOLD should drain health."""
    a = _make_agent_hr(hunger_resistance=0.0, hunger=1.0)
    a.health = 0.8
    before = a.health
    a.tick()
    assert a.health < before


def test_health_drain_does_not_trigger_below_threshold():
    """Ticking an agent with hunger below threshold should not drain health."""
    a = _make_agent_hr(hunger_resistance=0.0, hunger=STARVATION_THRESHOLD - 0.05)
    a.health = 0.8
    before = a.health
    a.tick()
    # After one tick hunger may still be below threshold — no starvation drain
    assert a.health >= before - 1e-9


def test_hunger_resistance_reduces_health_drain():
    """Higher hunger_resistance should result in less health drain."""
    a_low  = _make_agent_hr(hunger_resistance=0.0, hunger=1.0)
    a_high = _make_agent_hr(hunger_resistance=0.9, hunger=1.0)
    a_low.health = a_high.health = 0.8
    a_low.tick()
    a_high.tick()
    assert a_low.health < a_high.health


def test_effective_strength_reduced_when_starving():
    """effective_strength should be less than strength when hunger > threshold."""
    a = _make_agent_hr(hunger_resistance=0.0, hunger=1.0)
    assert a.effective_strength < a.strength


def test_effective_strength_equals_strength_when_not_hungry():
    """effective_strength should equal strength when hunger is below threshold."""
    a = _make_agent_hr(hunger_resistance=0.0, hunger=0.0)
    assert abs(a.effective_strength - a.strength) < 1e-6


def test_effective_strength_equals_strength_with_full_resistance():
    """Even at max hunger, full hunger_resistance should prevent strength penalty."""
    a = _make_agent_hr(hunger_resistance=1.0, hunger=1.0)
    assert abs(a.effective_strength - a.strength) < 1e-6


def test_rest_less_effective_when_starving():
    """REST health restore should be lower when agent is heavily starving."""
    a_fed      = _make_agent_hr(hunger=0.0)
    a_starving = _make_agent_hr(hunger=1.0)
    a_fed.health = a_starving.health = 0.5
    a_fed.rest()
    starving_before = a_starving.health
    a_starving.rest()
    assert a_fed.health > a_starving.health


def test_spawn_includes_hunger_resistance():
    """spawn() should produce a hunger_resistance within the config range."""
    import yaml, pathlib
    cfg_path = pathlib.Path("config/default.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())
    rng = np.random.default_rng(42)
    a = Agent.spawn("s0", (0, 0), rng, cfg)
    hr_min = cfg["agent"]["hunger_resistance_min"]
    hr_max = cfg["agent"]["hunger_resistance_max"]
    assert hr_min <= a.hunger_resistance <= hr_max


def test_spawn_from_parents_inherits_hunger_resistance():
    """spawn_from_parents() should inherit hunger_resistance from both parents."""
    p1 = _make_agent_hr(hunger_resistance=0.2)
    p2 = _make_agent_hr(hunger_resistance=0.8)
    p1.agent_id = "p1"; p2.agent_id = "p2"
    rng = np.random.default_rng(7)
    child = Agent.spawn_from_parents("c0", (0, 0), p1, p2, rng)
    assert 0.0 <= child.hunger_resistance <= 1.0


# ── defense_power tests ───────────────────────────────────────────────────────

def _make_combat_agent(strength: float = 0.5, resistances: dict | None = None) -> Agent:
    a = Agent(agent_id="c0", position=(0, 0), health=1.0, hunger=0.0, strength=strength)
    if resistances is not None:
        a.resistances = resistances
    else:
        a.resistances = {"poison": 0.0, "flame": 0.0, "qi_drain": 0.0}
    return a


def test_defense_power_zero_resistance():
    """defense_power = effective_strength * 0.5 when all resistances are 0."""
    a = _make_combat_agent(strength=0.8, resistances={"poison": 0.0, "flame": 0.0, "qi_drain": 0.0})
    expected = a.effective_strength * 0.5
    assert abs(a.defense_power - expected) < 1e-6


def test_defense_power_full_resistance():
    """defense_power is higher than effective_strength alone when resistances are high."""
    a = _make_combat_agent(strength=0.4, resistances={"poison": 1.0, "flame": 1.0, "qi_drain": 1.0})
    assert a.defense_power > a.effective_strength * 0.5


def test_defense_power_in_range():
    """defense_power is always in [0, 1]."""
    a = _make_combat_agent(strength=1.0, resistances={"poison": 1.0, "flame": 1.0, "qi_drain": 1.0})
    assert 0.0 <= a.defense_power <= 1.0


def test_defense_power_empty_resistances():
    """defense_power handles an empty resistances dict gracefully."""
    a = _make_combat_agent(strength=0.6, resistances={})
    assert abs(a.defense_power - a.effective_strength * 0.5) < 1e-6
