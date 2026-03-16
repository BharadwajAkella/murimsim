"""multi_env.py — Phase 3b/3c multi-agent environment.

N agents share one World. The policy is trained on all agents via a round-robin
focal-agent scheme: each call to step() acts on one focal agent while the
remaining agents follow a simple heuristic, then the focal index advances.

This design keeps the standard gym.Env interface so the env works directly
with SB3 PPO through DummyVecEnv / SubprocVecEnv.

Observation layout (Phase 3b onwards — 234 floats total):
    [0:100]   5×5 local grid × 4 resource channels  (food, qi, materials, poison)
    [100:175] 5×5 local grid × 3 agent channels     (agent_present, health, strength)
    [175:225] 5×5 local grid × 2 stash channels     (my_stash, enemy_stash)
    [225:234] Self stats × 9:
                health, hunger, inv_food, inv_poison,
                poison_resistance, poison_intake,
                combat_experience,      # Phase 3c: fights survived / 100, else 0
                terrain_familiarity,    # ticks near food / TERRAIN_FAM_SCALE, capped 1.0
                recent_reward_ema       # EMA of per-step rewards, normalised to [0,1]

Action space (Phase 3b): Discrete(7) — same as SurvivalEnv
Action space (Phase 3c): Discrete(9) — adds ATTACK, DEFEND
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from murimsim.actions import Action, MOVE_DELTAS, N_ACTIONS_PHASE2, N_ACTIONS_PHASE3
from murimsim.agent import Agent
from murimsim.stash import StashRegistry
from murimsim.world import World

# ── Observation layout constants ─────────────────────────────────────────────
OBS_VIEW_SIZE: int = 5

OBS_N_RESOURCE_CH: int = 4          # food, qi, materials, poison
OBS_N_AGENT_CH: int = 3             # agent_present, agent_health, agent_strength
OBS_N_STASH_CH: int = 2             # my_stash, enemy_stash
OBS_CHANNEL_ORDER: list[str] = ["food", "qi", "materials", "poison"]

OBS_RESOURCE_GRID_SIZE: int = OBS_VIEW_SIZE * OBS_VIEW_SIZE * OBS_N_RESOURCE_CH  # 100
OBS_AGENT_GRID_SIZE: int = OBS_VIEW_SIZE * OBS_VIEW_SIZE * OBS_N_AGENT_CH        # 75
OBS_STASH_GRID_SIZE: int = OBS_VIEW_SIZE * OBS_VIEW_SIZE * OBS_N_STASH_CH        # 50
OBS_STATS_SIZE: int = 9  # health, hunger, inv_food, inv_poison, pr, pi, combat_exp, terrain_fam, reward_ema
OBS_TOTAL_SIZE: int = OBS_RESOURCE_GRID_SIZE + OBS_AGENT_GRID_SIZE + OBS_STASH_GRID_SIZE + OBS_STATS_SIZE  # 234

# ── History signal constants ──────────────────────────────────────────────────
TERRAIN_FAM_SCALE: float = 200.0   # ticks_near_food / SCALE → [0, 1]
REWARD_EMA_ALPHA: float = 0.10
REWARD_EMA_SCALE: float = 0.5      # EMA normalised: 0 = −scale, 1 = +scale

# ── Heuristic constants (non-focal agents) ────────────────────────────────────
HEURISTIC_HUNGER_EAT: float = 0.5   # eat when hunger exceeds this
HEURISTIC_SCAN_RADIUS: int = 3      # Manhattan radius for food scan

# ── Reward shaping (mirrors SurvivalEnv) ─────────────────────────────────────
REWARD_ALIVE: float = 0.02
REWARD_HUNGER_RELIEF_SCALE: float = 0.20
REWARD_FOOD_GATHERED_SCALE: float = 0.05
REWARD_POISON_DAMAGE_SCALE: float = -0.30
REWARD_DEATH: float = -1.00
REWARD_EXPLORE_BASE: float = 0.25


class MultiAgentEnv(gym.Env):
    """Multi-agent survival environment (Phase 3b/3c).

    N agents share one World instance. The environment exposes a single-agent
    gym interface by rotating a *focal agent* every step. On each call to
    ``step(action)``:

    1. The action is applied to the current focal agent.
    2. All other agents execute a simple heuristic (eat → gather → navigate).
    3. The world and all agents advance one action-tick.
    4. The focal index advances to the next live agent.
    5. The observation and reward for the **new** focal agent are returned.

    An episode terminates when the focal agent dies. SB3 then calls ``reset()``,
    which respawns the world and all agents.

    Args:
        config:      Parsed YAML config dict.
        n_agents:    Number of agents sharing the world (default 10).
        seed:        Optional seed override.
        render_mode: Not implemented — pass None.
        n_actions:   7 (Phase 3b) or 9 (Phase 3c with ATTACK/DEFEND).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: dict[str, Any],
        n_agents: int = 10,
        seed: int | None = None,
        render_mode: str | None = None,
        n_actions: int = N_ACTIONS_PHASE2,
    ) -> None:
        super().__init__()
        self._config = copy.deepcopy(config)
        self._seed = seed if seed is not None else int(config["world"]["seed"])
        self._n_agents = n_agents
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_TOTAL_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(n_actions)

        self._world: World | None = None
        self._agents: list[Agent] = []
        self._rng: np.random.Generator | None = None
        self._resource_configs: dict[str, Any] = {}
        self._action_ticks: int = int(config["world"].get("action_ticks", 1))
        self._focal_idx: int = 0
        self._stash_registry: StashRegistry = StashRegistry()

        # Per-agent history state (reset each episode)
        self._visited_tiles: list[set[tuple[int, int]]] = []
        self._ticks_near_food: list[float] = []
        self._reward_ema: list[float] = []
        self._combat_experience: list[float] = []  # Phase 3c: updated on fights

    # ── Gymnasium API ────────────────────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        effective_seed = seed if seed is not None else self._seed

        self._rng = np.random.default_rng(effective_seed)

        cfg = copy.deepcopy(self._config)
        cfg["world"]["seed"] = effective_seed

        # Domain randomization (mirrors SurvivalEnv)
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
        self._agents = [
            Agent.spawn(
                f"agent_{i}",
                (int(self._rng.integers(0, gs)), int(self._rng.integers(0, gs))),
                self._rng,
                cfg,
            )
            for i in range(self._n_agents)
        ]

        self._focal_idx = 0
        self._visited_tiles = [{a.position} for a in self._agents]
        self._ticks_near_food = [0.0] * self._n_agents
        self._reward_ema = [0.0] * self._n_agents
        self._combat_experience = [0.0] * self._n_agents

        self._stash_registry.reset()

        return self._build_obs(self._focal_idx), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._world is not None

        focal = self._agents[self._focal_idx]
        hunger_prev = focal.hunger
        food_gathered = 0
        poison_damage = 0.0

        # 1. Apply action to focal agent
        action_enum = Action(action)
        food_gathered, poison_damage = self._apply_action(focal, action_enum)

        # 2. Heuristic step for all non-focal alive agents
        for i, agent in enumerate(self._agents):
            if i != self._focal_idx and agent.alive:
                self._heuristic_step(agent)

        # 3. Advance world + all agents
        for _ in range(self._action_ticks):
            self._world.step()
            for agent in self._agents:
                agent.tick()

        # 4. Update history for focal agent
        exploration_reward = 0.0
        if focal.alive and focal.position not in self._visited_tiles[self._focal_idx]:
            self._visited_tiles[self._focal_idx].add(focal.position)
            exploration_reward = REWARD_EXPLORE_BASE * focal.adventure_spirit

        if focal.alive:
            food_view = self._world.get_grid_view("food")
            ax, ay = focal.position
            half = OBS_VIEW_SIZE // 2
            gs = self._world.grid_size
            x0, x1 = max(0, ax - half), min(gs, ax + half + 1)
            y0, y1 = max(0, ay - half), min(gs, ay + half + 1)
            if food_view[y0:y1, x0:x1].sum() > 0:
                self._ticks_near_food[self._focal_idx] += 1.0

        # 5. Compute reward for focal agent's action
        reward = self._compute_reward(hunger_prev, food_gathered, poison_damage, focal, exploration_reward)
        ema = self._reward_ema[self._focal_idx]
        self._reward_ema[self._focal_idx] = (1.0 - REWARD_EMA_ALPHA) * ema + REWARD_EMA_ALPHA * reward

        terminated = not focal.alive

        # 6. Advance focal index to next live agent (skip dead agents)
        if not terminated:
            self._focal_idx = self._next_live(self._focal_idx)

        obs = self._build_obs(self._focal_idx)
        info = {
            "hunger": focal.hunger,
            "health": focal.health,
            "alive_count": sum(1 for a in self._agents if a.alive),
        }
        return obs, reward, terminated, False, info

    def render(self) -> None:
        pass  # Rendering handled by web viewer

    # ── Observation builder ───────────────────────────────────────────────────

    def _build_obs(self, agent_idx: int) -> np.ndarray:
        """Build the 184-float observation for agents[agent_idx].

        Layout:
            [0:100]   resource grid (5×5×4, row-major, channel-last)
            [100:175] agent grid    (5×5×3: present, health, strength)
            [175:184] self stats × 9
        """
        agent = self._agents[agent_idx]
        world = self._world
        gs = world.grid_size
        half = OBS_VIEW_SIZE // 2
        ax, ay = agent.position

        # Resource channels
        grids = world.get_grids_view()
        grid_stack = np.stack([grids[rid] for rid in OBS_CHANNEL_ORDER], axis=-1)

        x0 = max(0, ax - half);  x1 = min(gs, ax + half + 1)
        y0 = max(0, ay - half);  y1 = min(gs, ay + half + 1)
        gx0 = x0 - (ax - half);  gy0 = y0 - (ay - half)

        resource_window = np.zeros((OBS_VIEW_SIZE, OBS_VIEW_SIZE, OBS_N_RESOURCE_CH), dtype=np.float32)
        resource_window[gy0:gy0 + (y1 - y0), gx0:gx0 + (x1 - x0), :] = grid_stack[y0:y1, x0:x1, :]
        flat_resources = resource_window.reshape(-1)  # 100

        # Agent channels (agent_present, agent_health, agent_strength)
        agent_window = np.zeros((OBS_VIEW_SIZE, OBS_VIEW_SIZE, OBS_N_AGENT_CH), dtype=np.float32)
        for i, other in enumerate(self._agents):
            if i == agent_idx or not other.alive:
                continue
            ox, oy = other.position
            wx = (ox - ax) + half
            wy = (oy - ay) + half
            if 0 <= wx < OBS_VIEW_SIZE and 0 <= wy < OBS_VIEW_SIZE:
                agent_window[wy, wx, 0] = max(agent_window[wy, wx, 0], 1.0)          # present
                agent_window[wy, wx, 1] = max(agent_window[wy, wx, 1], other.health)  # health
                agent_window[wy, wx, 2] = max(agent_window[wy, wx, 2], other.strength)  # strength
        flat_agents = agent_window.reshape(-1)  # 75

        # Stash channels (my_stash, enemy_stash)
        stash_window = np.zeros((OBS_VIEW_SIZE, OBS_VIEW_SIZE, OBS_N_STASH_CH), dtype=np.float32)
        for wy in range(OBS_VIEW_SIZE):
            for wx in range(OBS_VIEW_SIZE):
                wx_world = ax - half + wx
                wy_world = ay - half + wy
                if 0 <= wx_world < gs and 0 <= wy_world < gs:
                    own = self._stash_registry.get_own_stash_at(agent.agent_id, wx_world, wy_world)
                    enemy = self._stash_registry.get_enemy_stashes_at(agent.agent_id, wx_world, wy_world)
                    stash_window[wy, wx, 0] = 1.0 if own else 0.0
                    stash_window[wy, wx, 1] = 1.0 if enemy else 0.0
        flat_stashes = stash_window.reshape(-1)  # 50

        # Self stats (9 values)
        inv = agent.inventory
        terrain_fam = min(1.0, self._ticks_near_food[agent_idx] / TERRAIN_FAM_SCALE)
        raw_ema = self._reward_ema[agent_idx]
        reward_ema_norm = float(np.clip(0.5 + raw_ema / (2.0 * REWARD_EMA_SCALE), 0.0, 1.0))
        combat_exp = min(1.0, self._combat_experience[agent_idx] / 100.0)

        stats = np.array([
            agent.health,
            agent.hunger,
            min(1.0, inv.food / 10.0),
            min(1.0, inv.poison / 10.0),
            agent.resistances.get("poison", 0.0),
            agent._intakes.get("poison", 0.0),
            combat_exp,
            terrain_fam,
            reward_ema_norm,
        ], dtype=np.float32)

        return np.concatenate([flat_resources, flat_agents, flat_stashes, stats])

    # ── Reward ───────────────────────────────────────────────────────────────

    def _compute_reward(
        self,
        hunger_prev: float,
        food_gathered: int,
        poison_damage: float,
        agent: Agent,
        exploration_reward: float = 0.0,
    ) -> float:
        reward = REWARD_ALIVE
        hunger_relief = hunger_prev - agent.hunger
        if hunger_relief > 0:
            reward += REWARD_HUNGER_RELIEF_SCALE * hunger_relief
        reward += REWARD_FOOD_GATHERED_SCALE * food_gathered
        reward += REWARD_POISON_DAMAGE_SCALE * poison_damage
        reward += exploration_reward
        if not agent.alive:
            reward += REWARD_DEATH
        return float(reward)

    # ── Action application ────────────────────────────────────────────────────

    def _apply_action(self, agent: Agent, action_enum: Action) -> tuple[int, float]:
        """Apply action_enum to agent. Returns (food_gathered, poison_damage)."""
        food_gathered = 0
        poison_damage = 0.0

        if action_enum in MOVE_DELTAS:
            dx, dy = MOVE_DELTAS[action_enum]
            agent.move(dx, dy, self._world.grid_size)

        elif action_enum == Action.GATHER:
            x, y = agent.position
            for rid in OBS_CHANNEL_ORDER:
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
            self._stash_registry.deposit(agent)

        elif action_enum == Action.WITHDRAW:
            self._stash_registry.withdraw(agent)

        elif action_enum == Action.STEAL:
            self._stash_registry.steal(agent)

        return food_gathered, poison_damage

    # ── Heuristic for non-focal agents ───────────────────────────────────────

    def _heuristic_step(self, agent: Agent) -> None:
        """Simple eat→gather→navigate heuristic for non-focal agents.

        Priority:
          1. Eat if hungry and carrying food.
          2. Gather if standing on a resource.
          3. Step one tile toward nearest food within HEURISTIC_SCAN_RADIUS.
          4. Random cardinal move.
        """
        # Eat
        if agent.hunger > HEURISTIC_HUNGER_EAT and agent.inventory.food > 0:
            agent.eat(self._resource_configs)
            return

        # Gather
        x, y = agent.position
        food_grid = self._world.get_grid_view("food")
        if food_grid[y, x] > 0:
            self._world.deplete("food", x, y)
            agent.gather("food")
            return

        # Navigate toward nearest food
        target = self._nearest_food(agent.position)
        if target is not None:
            tx, ty = target
            dx = int(np.sign(tx - x))
            dy = int(np.sign(ty - y))
            if dx != 0:
                agent.move(dx, 0, self._world.grid_size)
            elif dy != 0:
                agent.move(0, dy, self._world.grid_size)
            return

        # Random cardinal move
        moves = list(MOVE_DELTAS.values())
        dx, dy = moves[int(self._rng.integers(0, len(moves)))]
        agent.move(dx, dy, self._world.grid_size)

    def _nearest_food(self, pos: tuple[int, int]) -> tuple[int, int] | None:
        """Return (x, y) of nearest food within HEURISTIC_SCAN_RADIUS, or None."""
        food_grid = self._world.get_grid_view("food")
        gs = self._world.grid_size
        ax, ay = pos
        r = HEURISTIC_SCAN_RADIUS
        best: tuple[int, int] | None = None
        best_dist = float("inf")
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                nx, ny = ax + dx, ay + dy
                if 0 <= nx < gs and 0 <= ny < gs and food_grid[ny, nx] > 0:
                    d = abs(dx) + abs(dy)
                    if d < best_dist:
                        best_dist = d
                        best = (nx, ny)
        return best

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _next_live(self, current_idx: int) -> int:
        """Return the index of the next live agent after current_idx (wraps)."""
        n = self._n_agents
        for offset in range(1, n + 1):
            idx = (current_idx + offset) % n
            if self._agents[idx].alive:
                return idx
        return current_idx  # all dead — episode should have terminated


# ── Combat constants ──────────────────────────────────────────────────────────
COMBAT_ATTACKER_SCALE: float = 0.3
COMBAT_DEFENDER_SCALE: float = 0.1
COMBAT_MAX_DAMAGE: float = 0.5

REWARD_DEFEAT_OPPONENT: float = 0.3
REWARD_DAMAGE_TAKEN_SCALE: float = -0.2

# Curriculum: combat probability ramps from START → END over RAMP_STEPS global steps
CURRICULUM_START_PROB: float = 0.2
CURRICULUM_END_PROB: float = 1.0
CURRICULUM_RAMP_STEPS: int = 300_000


class CombatEnv(MultiAgentEnv):
    """Phase 3c: MultiAgentEnv with ATTACK / DEFEND and a combat curriculum.

    Extends MultiAgentEnv with:
    - Action space Discrete(9) — adds ATTACK and DEFEND.
    - Combat: ``damage = attacker.strength*0.3 − defender.strength*0.1*is_defending``
      clamped to [0, 0.5]. Requires adjacency (Manhattan ≤ 1).
    - Death drops inventory (food) onto the agent's tile.
    - Curriculum: ``combat_prob`` starts at 0.2, ramps to 1.0 over
      ``curriculum_ramp_steps`` global steps. ATTACK/DEFEND actions are silently
      replaced with REST while curriculum probability is not met. This keeps the
      action space Discrete(9) throughout, which is required for SB3 warm-starting.

    Args:
        curriculum_ramp_steps: Number of global steps over which combat_prob
            ramps from 0.2 to 1.0. Default 300_000.
    """

    def __init__(
        self,
        config: dict[str, Any],
        n_agents: int = 10,
        seed: int | None = None,
        render_mode: str | None = None,
        curriculum_ramp_steps: int = CURRICULUM_RAMP_STEPS,
    ) -> None:
        super().__init__(
            config=config,
            n_agents=n_agents,
            seed=seed,
            render_mode=render_mode,
            n_actions=N_ACTIONS_PHASE3,
        )
        self._curriculum_ramp_steps: int = curriculum_ramp_steps
        self._global_step_count: int = 0  # persists across episodes

    # ── Curriculum ────────────────────────────────────────────────────────────

    @property
    def combat_prob(self) -> float:
        """Current probability that a combat action is allowed (not masked to REST)."""
        frac = min(1.0, self._global_step_count / max(1, self._curriculum_ramp_steps))
        return CURRICULUM_START_PROB + (CURRICULUM_END_PROB - CURRICULUM_START_PROB) * frac

    # ── step override ─────────────────────────────────────────────────────────

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._world is not None
        self._global_step_count += 1

        # Curriculum: mask ATTACK/DEFEND to REST when not yet fully enabled
        action_enum = Action(action)
        if action_enum in (Action.ATTACK, Action.DEFEND):
            if self._rng.random() > self.combat_prob:
                action_enum = Action.REST

        focal = self._agents[self._focal_idx]
        hunger_prev = focal.hunger
        food_gathered = 0
        poison_damage = 0.0
        damage_dealt = 0.0
        defeat_bonus = 0.0
        focal_defending = (action_enum == Action.DEFEND)

        # 1. Apply focal agent's action
        if action_enum == Action.ATTACK:
            damage_dealt, defeated = self._do_attack(focal)
            defeat_bonus = REWARD_DEFEAT_OPPONENT if defeated else 0.0
        elif action_enum == Action.DEFEND:
            focal.rest()  # defending = hold ground + minor health recovery
        else:
            food_gathered, poison_damage = self._apply_action(focal, action_enum)

        # 2. Heuristic for non-focal agents (combat-aware in Phase 3c)
        damage_taken = 0.0
        for i, agent in enumerate(self._agents):
            if i != self._focal_idx and agent.alive:
                dmg = self._heuristic_combat_step(agent, focal, focal_defending)
                damage_taken += dmg

        # 3. Advance world + all agents
        for _ in range(self._action_ticks):
            self._world.step()
            for agent in self._agents:
                agent.tick()

        # 4. Update history for focal agent
        exploration_reward = 0.0
        if focal.alive and focal.position not in self._visited_tiles[self._focal_idx]:
            self._visited_tiles[self._focal_idx].add(focal.position)
            exploration_reward = REWARD_EXPLORE_BASE * focal.adventure_spirit

        if focal.alive:
            food_view = self._world.get_grid_view("food")
            ax, ay = focal.position
            half = OBS_VIEW_SIZE // 2
            gs = self._world.grid_size
            x0, x1 = max(0, ax - half), min(gs, ax + half + 1)
            y0, y1 = max(0, ay - half), min(gs, ay + half + 1)
            if food_view[y0:y1, x0:x1].sum() > 0:
                self._ticks_near_food[self._focal_idx] += 1.0

        # 5. Combat experience: increment if focal survived being attacked
        if damage_taken > 0 and focal.alive:
            self._combat_experience[self._focal_idx] += 1.0

        # 6. Compute reward
        reward = self._compute_combat_reward(
            hunger_prev, food_gathered, poison_damage, focal,
            exploration_reward, damage_dealt, damage_taken, defeat_bonus,
        )
        ema = self._reward_ema[self._focal_idx]
        self._reward_ema[self._focal_idx] = (1.0 - REWARD_EMA_ALPHA) * ema + REWARD_EMA_ALPHA * reward

        terminated = not focal.alive

        if not terminated:
            self._focal_idx = self._next_live(self._focal_idx)

        obs = self._build_obs(self._focal_idx)
        info = {
            "hunger": focal.hunger,
            "health": focal.health,
            "alive_count": sum(1 for a in self._agents if a.alive),
            "combat_prob": self.combat_prob,
        }
        return obs, reward, terminated, False, info

    # ── Combat helpers ────────────────────────────────────────────────────────

    def _do_attack(self, attacker: Agent) -> tuple[float, bool]:
        """Attack the nearest adjacent agent. Returns (damage_dealt, killed)."""
        target = self._nearest_adjacent_agent(attacker)
        if target is None:
            return 0.0, False
        damage = self._combat_damage(attacker, target, is_defending=False)
        target.health = max(0.0, target.health - damage)
        target._check_death()
        if not target.alive:
            self._drop_inventory(target)
            return damage, True
        return damage, False

    def _heuristic_combat_step(
        self, agent: Agent, focal: Agent, focal_defending: bool
    ) -> float:
        """Heuristic for one non-focal agent. Returns damage dealt to focal agent."""
        ax, ay = agent.position
        fx, fy = focal.position
        adjacent = abs(ax - fx) + abs(ay - fy) <= 1

        # Attack focal if adjacent and focal appears weaker
        if adjacent and agent.strength > focal.strength * 1.1 and focal.health > 0:
            damage = self._combat_damage(agent, focal, is_defending=focal_defending)
            focal.health = max(0.0, focal.health - damage)
            focal._check_death()
            if not focal.alive:
                self._drop_inventory(focal)
            return damage

        # Otherwise forage
        self._heuristic_step(agent)
        return 0.0

    def _combat_damage(
        self, attacker: Agent, defender: Agent, is_defending: bool
    ) -> float:
        """Compute combat damage.

        Formula: attacker.strength*0.3 − defender.strength*0.1*is_defending
        Clamped to [0, COMBAT_MAX_DAMAGE].
        """
        raw = (
            attacker.strength * COMBAT_ATTACKER_SCALE
            - defender.strength * COMBAT_DEFENDER_SCALE * (1.0 if is_defending else 0.0)
        )
        return float(np.clip(raw, 0.0, COMBAT_MAX_DAMAGE))

    def _nearest_adjacent_agent(self, agent: Agent) -> Agent | None:
        """Return nearest live agent within Manhattan distance 1, or None."""
        ax, ay = agent.position
        for other in self._agents:
            if other is agent or not other.alive:
                continue
            ox, oy = other.position
            if abs(ox - ax) + abs(oy - ay) <= 1:
                return other
        return None

    def _drop_inventory(self, agent: Agent) -> None:
        """Drop dead agent's food inventory onto its tile (if tile is empty)."""
        if agent.inventory.food <= 0:
            return
        x, y = agent.position
        if self._world.get_grid_view("food")[y, x] == 0.0:
            self._world._grid["food"][y, x] = 1.0
        agent.inventory.food = 0

    def _compute_combat_reward(
        self,
        hunger_prev: float,
        food_gathered: int,
        poison_damage: float,
        agent: Agent,
        exploration_reward: float,
        damage_dealt: float,
        damage_taken: float,
        defeat_bonus: float,
    ) -> float:
        """Phase 3c reward: Phase 3b reward + combat shaping."""
        reward = self._compute_reward(hunger_prev, food_gathered, poison_damage, agent, exploration_reward)
        reward += REWARD_DAMAGE_TAKEN_SCALE * damage_taken
        reward += defeat_bonus
        return reward
