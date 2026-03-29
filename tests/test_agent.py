"""Phase 2 — Agent unit tests."""
from __future__ import annotations

import numpy as np
import pytest

from murimsim.agent import (
    HUNGER_PER_TICK,
    RESISTANCE_GAIN,
    STARVATION_DAMAGE_PER_TICK,
    Agent,
    AgentInventory,
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
    """When hunger is already 1.0, ticking reduces health."""
    agent = _make_agent(health=1.0, hunger=1.0)
    agent.tick()
    assert abs(agent.health - (1.0 - STARVATION_DAMAGE_PER_TICK)) < 1e-6


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
    """rest() increases health by 0.02, capped at 1.0."""
    agent = _make_agent(health=0.5)
    agent.rest()
    assert abs(agent.health - 0.52) < 1e-6


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
    """TRAIN action on qi tile grows strength faster than anywhere."""
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

    delta_qi = agent.train(on_qi_tile=True)
    assert abs(delta_qi - Agent.TRAIN_RATE_QI_TILE * 0.7) < 1e-6
    assert agent.strength > 0.3

    agent.strength = 0.3
    delta_anywhere = agent.train(on_qi_tile=False)
    assert delta_anywhere < delta_qi, "qi-tile training must be faster"


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
        agent.train(on_qi_tile=True)
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
