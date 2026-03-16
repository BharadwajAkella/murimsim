"""Phase 2 single-agent Gymnasium environment.

Observation space:
    5×5 local grid (centered on agent), 8 channels
    (food/qi/materials/poison/mountain/flame/my_stash/enemy_stash) flattened + own stats.

    Channels (in order): food=0, qi=1, materials=2, poison=3, mountain=4, flame=5,
                         my_stash=6, enemy_stash=7
    Own stats appended: health, hunger, inv_food, inv_poison,
                        poison_resistance, poison_intake, flame_resistance, qi_drain_resistance,
                        action_ticks  (normalised: action_ticks / 10.0)
    Total obs size: 5*5*8 + 9 = 209

Action space: Discrete(7) — see murimsim.actions.Action

Reward (per tick):
    +0.02  alive bonus
    +0.20  * (hunger_prev - hunger_now)   hunger relief
    +0.05  * food_gathered_this_step
    -0.30  * poison_damage_taken
    -0.30  * traversal_damage_taken
    -1.00  on death

Episode fitness (F-metric, reported in info on termination):
    F = 0.6 * survival_ratio + 0.3 * food_score + 0.1 * stat_gain
    where:
        survival_ratio = ticks_alive / max_episode_ticks
        food_score     = net_food_eaten / (max_episode_ticks * HUNGER_PER_TICK / hunger_restore)
        stat_gain      = mean resistance improvement over episode
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from murimsim.actions import Action, MOVE_DELTAS, N_ACTIONS_PHASE2
from murimsim.agent import Agent
from murimsim.stash import StashRegistry
from murimsim.world import World

# Observation layout constants
OBS_VIEW_SIZE: int = 5          # 5×5 local window
OBS_N_CHANNELS: int = 8        # food, qi, materials, poison, mountain, flame, my_stash, enemy_stash
OBS_CHANNEL_FOOD: int = 0
OBS_CHANNEL_QI: int = 1
OBS_CHANNEL_MATERIALS: int = 2
OBS_CHANNEL_POISON: int = 3
OBS_CHANNEL_MOUNTAIN: int = 4
OBS_CHANNEL_FLAME: int = 5
OBS_CHANNEL_MY_STASH: int = 6
OBS_CHANNEL_ENEMY_STASH: int = 7
# Resource channels sourced from world grids; stash channels are computed separately.
OBS_CHANNEL_ORDER: list[str] = ["food", "qi", "materials", "poison", "mountain", "flame"]

OBS_GRID_SIZE: int = OBS_VIEW_SIZE * OBS_VIEW_SIZE * OBS_N_CHANNELS  # 200
OBS_STATS_SIZE: int = 9   # health, hunger, inv_food, inv_poison, poison_resistance, poison_intake, flame_resistance, qi_drain_resistance, action_ticks
OBS_TOTAL_SIZE: int = OBS_GRID_SIZE + OBS_STATS_SIZE  # 209

# action_ticks normalisation: divide by this so 10 ticks → 1.0
OBS_ACTION_TICKS_MAX: float = 10.0

# F-metric weights (episode-level fitness score)
F_WEIGHT_SURVIVAL: float = 0.6
F_WEIGHT_FOOD: float = 0.3
F_WEIGHT_STAT_GAIN: float = 0.1

# Reward shaping coefficients — single source of truth
REWARD_ALIVE: float = 0.02
REWARD_HUNGER_RELIEF_SCALE: float = 0.20
REWARD_FOOD_GATHERED_SCALE: float = 0.05
REWARD_POISON_DAMAGE_SCALE: float = -0.30
REWARD_TRAVERSAL_DAMAGE_SCALE: float = -0.30
REWARD_DEATH: float = -1.00
REWARD_EXPLORE_BASE: float = 0.25   # scaled by agent.adventure_spirit per new tile

# Stash action rewards
REWARD_DEPOSIT: float = 0.01                 # small reward for depositing (encourages planning)
REWARD_WITHDRAW_FOOD_SCALE: float = 0.10     # per food item withdrawn
REWARD_STEAL_FOOD_SCALE: float = 0.15        # per food item stolen (slightly better than gathering)

# Food-density shaping (encourage "camping" near reliable food)
FOOD_DENSITY_RADIUS: int = 3
FOOD_DENSITY_HIGH: float = 0.08
REWARD_CAMP_REST: float = 0.05
PENALTY_MOVE_IN_FOOD: float = -0.03
PENALTY_MOVE_WHEN_HUNGRY: float = -0.05


class SurvivalEnv(gym.Env):
    """Single-agent survival environment (Phase 2).

    The agent must forage for food while avoiding poison on a 30×30 grid.

    Args:
        config:      Parsed YAML config dict.
        seed:        Optional seed override. If None, uses config world.seed.
        render_mode: Gymnasium render mode (not implemented — pass None).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: dict[str, Any],
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self._config = copy.deepcopy(config)
        self._seed = seed if seed is not None else int(config["world"]["seed"])
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_TOTAL_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS_PHASE2)

        # Initialised properly in reset()
        self._world: World | None = None
        self._agent: Agent | None = None
        self._rng: np.random.Generator | None = None
        self._resource_configs: dict[str, Any] = {}
        self._episode_rng_state: dict | None = None
        self._visited_tiles: set[tuple[int, int]] = set()
        self._action_ticks: int = int(config["world"].get("action_ticks", 1))
        self._stash_registry: StashRegistry = StashRegistry()

        # Episode-level stats for F-metric (reset each episode)
        self._ep_ticks_alive: int = 0
        self._ep_food_eaten: int = 0
        self._ep_resistance_start: dict[str, float] = {}
        self._ep_max_ticks: int = 500  # updated from config on reset

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        effective_seed = seed if seed is not None else self._seed

        # Deterministic RNG — one generator drives DR sampling, world and agent spawn
        self._rng = np.random.default_rng(effective_seed)

        cfg = copy.deepcopy(self._config)
        cfg["world"]["seed"] = effective_seed

        # --- Domain randomization: sample world params from ranges each episode ---
        dr = cfg.get("domain_randomization", {})
        if dr.get("enabled", False):
            for resource in cfg["resources"]:
                if resource["id"] == "food":
                    lo, hi = dr.get("food_regen_ticks", [50, 250])
                    resource["regen_ticks"] = int(self._rng.integers(lo, hi + 1))
                    lo, hi = dr.get("food_spawn_density", [0.03, 0.08])
                    resource["spawn_density"] = float(self._rng.uniform(lo, hi))
            lo, hi = dr.get("action_ticks", [3, 8])
            cfg["world"]["action_ticks"] = int(self._rng.integers(lo, hi + 1))

        self._action_ticks = int(cfg["world"].get("action_ticks", 1))

        self._world = World(cfg, rng=np.random.default_rng(effective_seed))
        self._resource_configs = self._world.resources

        gs = self._world.grid_size
        start_pos = (int(self._rng.integers(0, gs)), int(self._rng.integers(0, gs)))
        self._agent = Agent.spawn("agent_0", start_pos, self._rng, cfg)
        self._visited_tiles = {start_pos}   # starting tile is free

        self._stash_registry.reset()

        # Reset episode F-metric trackers
        self._ep_ticks_alive = 0
        self._ep_food_eaten = 0
        self._ep_resistance_start = dict(self._agent.resistances)
        self._ep_max_ticks = int(self._config.get("world", {}).get("max_ticks", 500))

        obs = self._build_obs()
        return obs, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._agent is not None and self._world is not None

        agent = self._agent
        hunger_prev = agent.hunger
        food_gathered = 0
        poison_damage = 0.0
        reward_deposit = 0.0
        reward_withdraw = 0.0
        reward_steal = 0.0

        action_enum = Action(action)

        # --- Execute action ---
        if action_enum in MOVE_DELTAS:
            dx, dy = MOVE_DELTAS[action_enum]
            agent.move(dx, dy, self._world.grid_size)
            effects = self._world.get_traversal_effects(*agent.position)
            if effects:
                traversal_damage = agent.apply_traversal_effects(effects)
                poison_damage += traversal_damage

        elif action_enum == Action.GATHER:
            x, y = agent.position
            # Gather the first present, gatherable resource on this tile
            # (priority: food > qi > materials > poison)
            for rid in OBS_CHANNEL_ORDER:
                rcfg = self._world.resources.get(rid)
                if rcfg is None or not rcfg.gatherable:
                    continue
                if self._world.get_grid_view(rid)[y, x] > 0:
                    self._world.deplete(rid, x, y)
                    agent.gather(rid)
                    if rid == "food":
                        food_gathered = 1
                    break

        elif action_enum == Action.EAT:
            poison_damage = agent.eat(self._resource_configs)

        elif action_enum == Action.REST:
            agent.rest()

        elif action_enum == Action.DEPOSIT:
            stash = self._stash_registry.deposit(agent)
            if stash is not None:
                reward_deposit = REWARD_DEPOSIT
            else:
                reward_deposit = 0.0
        elif action_enum == Action.WITHDRAW:
            food_before = agent.inventory.food
            self._stash_registry.withdraw(agent)
            food_withdrawn = agent.inventory.food - food_before
            reward_withdraw = REWARD_WITHDRAW_FOOD_SCALE * max(0, food_withdrawn)
        elif action_enum == Action.STEAL:
            food_before = agent.inventory.food
            self._stash_registry.steal(agent)
            food_stolen = agent.inventory.food - food_before
            reward_steal = REWARD_STEAL_FOOD_SCALE * max(0, food_stolen)

        # --- World + agent advance for action_ticks ticks (one action = n real ticks) ---
        for _ in range(self._action_ticks):
            self._world.step()
            agent.tick()

        # Track episode-level stats for F-metric
        if agent.alive:
            self._ep_ticks_alive += self._action_ticks

        # --- Exploration bonus: reward visiting tiles not seen this episode ---
        exploration_reward = 0.0
        if agent.alive and agent.position not in self._visited_tiles:
            self._visited_tiles.add(agent.position)
            exploration_reward = REWARD_EXPLORE_BASE * agent.adventure_spirit

        food_density = self._local_food_density(agent.position)

        # --- Reward ---
        reward = self._compute_reward(
            hunger_prev,
            food_gathered,
            poison_damage,
            agent,
            exploration_reward,
            action_enum,
            food_density,
        )
        reward += reward_deposit + reward_withdraw + reward_steal

        # Track food eaten for F-metric
        hunger_after = agent.hunger
        if hunger_after < hunger_prev:
            self._ep_food_eaten += 1

        terminated = not agent.alive
        obs = self._build_obs()
        info = {
            "hunger": agent.hunger,
            "health": agent.health,
            "resistances": agent.resistances,
        }
        if terminated:
            info["f_metric"] = self._compute_f_metric()
        return obs, reward, terminated, False, info

    def render(self) -> None:
        pass  # Rendering handled by the web viewer

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        """Build the flat observation vector for the current state.

        Layout:
            [0:200]  5×5 grid × 8 channels, row-major, channel-last
                     Channels: food, qi, materials, poison, mountain, flame,
                               my_stash, enemy_stash
                     Missing resources (not in world) are padded with zeros.
            [200:209] own stats: health, hunger, inv_food, inv_poison,
                      poison_resistance, poison_intake, flame_resistance,
                      qi_drain_resistance, action_ticks (normalised)
        """
        agent = self._agent
        world = self._world
        gs = world.grid_size
        half = OBS_VIEW_SIZE // 2  # 2
        ax, ay = agent.position

        # 6 resource channels (world grids)
        n_resource_ch = len(OBS_CHANNEL_ORDER)
        grid_obs = np.zeros((OBS_VIEW_SIZE, OBS_VIEW_SIZE, OBS_N_CHANNELS), dtype=np.float32)
        grids = world.get_grids_view()
        channel_grids = []
        for rid in OBS_CHANNEL_ORDER:
            if rid in grids:
                channel_grids.append(grids[rid])
            else:
                channel_grids.append(np.zeros((gs, gs), dtype=np.float32))
        grid_stack = np.stack(channel_grids, axis=-1)

        x0 = max(0, ax - half)
        x1 = min(gs, ax + half + 1)
        y0 = max(0, ay - half)
        y1 = min(gs, ay + half + 1)

        gx0 = x0 - (ax - half)
        gy0 = y0 - (ay - half)
        gx1 = gx0 + (x1 - x0)
        gy1 = gy0 + (y1 - y0)

        grid_obs[gy0:gy1, gx0:gx1, :n_resource_ch] = grid_stack[y0:y1, x0:x1, :]

        # Stash channels: scan the 5×5 window for own and enemy stashes
        for wy in range(OBS_VIEW_SIZE):
            for wx in range(OBS_VIEW_SIZE):
                wx_world = ax - half + wx
                wy_world = ay - half + wy
                if 0 <= wx_world < gs and 0 <= wy_world < gs:
                    own = self._stash_registry.get_own_stash_at(agent.agent_id, wx_world, wy_world)
                    enemy = self._stash_registry.get_enemy_stashes_at(agent.agent_id, wx_world, wy_world)
                    grid_obs[wy, wx, OBS_CHANNEL_MY_STASH] = 1.0 if own else 0.0
                    grid_obs[wy, wx, OBS_CHANNEL_ENEMY_STASH] = 1.0 if enemy else 0.0

        flat_grid = grid_obs.reshape(-1)  # 200

        inv = agent.inventory
        stats = np.array(
            [
                agent.health,
                agent.hunger,
                min(1.0, inv.food / 10.0),     # normalised: 10 items = 1.0
                min(1.0, inv.poison / 10.0),
                agent.resistances.get("poison", 0.0),
                agent._intakes.get("poison", 0.0),
                agent.resistances.get("flame", 0.0),
                agent.resistances.get("qi_drain", 0.0),
                min(1.0, self._action_ticks / OBS_ACTION_TICKS_MAX),
            ],
            dtype=np.float32,
        )

        return np.concatenate([flat_grid, stats])

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        hunger_prev: float,
        food_gathered: int,
        poison_damage: float,
        agent: Agent,
        exploration_reward: float = 0.0,
        action_enum: Action | None = None,
        food_density: float = 0.0,
    ) -> float:
        reward = REWARD_ALIVE
        hunger_relief = hunger_prev - agent.hunger
        if hunger_relief > 0:
            reward += REWARD_HUNGER_RELIEF_SCALE * hunger_relief
        reward += REWARD_FOOD_GATHERED_SCALE * food_gathered
        reward += REWARD_POISON_DAMAGE_SCALE * poison_damage
        reward += exploration_reward
        if action_enum is not None:
            hunger_low = agent.hunger < 0.3
            hunger_high = agent.hunger > 0.7
            in_food_zone = food_density >= FOOD_DENSITY_HIGH
            if in_food_zone and hunger_low:
                if action_enum == Action.REST:
                    reward += REWARD_CAMP_REST
                elif action_enum in MOVE_DELTAS:
                    reward += PENALTY_MOVE_IN_FOOD
            if in_food_zone and hunger_high and action_enum in MOVE_DELTAS:
                reward += PENALTY_MOVE_WHEN_HUNGRY
        if not agent.alive:
            reward += REWARD_DEATH
        return float(reward)

    def _compute_f_metric(self) -> float:
        """Compute episode-level fitness score on termination.

        F = 0.6 * survival_ratio + 0.3 * food_score + 0.1 * stat_gain

        Returns a float in [0.0, 1.0] suitable for comparing runs.
        """
        max_ticks = max(1, self._ep_max_ticks)
        survival_ratio = min(1.0, self._ep_ticks_alive / max_ticks)

        # food_score: fraction of hunger that was covered by eating
        # Max food events ≈ max_ticks * HUNGER_PER_TICK / hunger_restore_per_eat
        from murimsim.agent import HUNGER_PER_TICK
        hunger_restore = 0.3  # default; exact value not critical for normalisation
        max_food_events = max(1, (max_ticks * HUNGER_PER_TICK) / hunger_restore)
        food_score = min(1.0, self._ep_food_eaten / max_food_events)

        # stat_gain: mean improvement across all tracked resistances
        agent = self._agent
        if agent is not None and self._ep_resistance_start:
            gains = [
                max(0.0, agent.resistances.get(k, 0.0) - v)
                for k, v in self._ep_resistance_start.items()
            ]
            stat_gain = min(1.0, sum(gains) / max(1, len(gains)) / 0.5)
        else:
            stat_gain = 0.0

        return float(
            F_WEIGHT_SURVIVAL * survival_ratio
            + F_WEIGHT_FOOD * food_score
            + F_WEIGHT_STAT_GAIN * stat_gain
        )

    def _local_food_density(self, pos: tuple[int, int]) -> float:
        """Return local food density in a radius window around pos."""
        world = self._world
        ax, ay = pos
        gs = world.grid_size
        r = FOOD_DENSITY_RADIUS
        x0 = max(0, ax - r)
        x1 = min(gs, ax + r + 1)
        y0 = max(0, ay - r)
        y1 = min(gs, ay + r + 1)
        grid = world.get_grid_view("food")[y0:y1, x0:x1]
        area = (y1 - y0) * (x1 - x0)
        if area <= 0:
            return 0.0
        return float(grid.sum() / area)
