"""multi_env.py — Multi-agent survival environment (active version).

N agents share one World. The policy is trained on all agents via a round-robin
focal-agent scheme: each call to step() acts on one focal agent while the
remaining agents follow a simple heuristic, then the focal index advances.

This design keeps the standard gym.Env interface so the env works directly
with SB3 PPO through DummyVecEnv / SubprocVecEnv.

Observation layout (263 floats total):
    [0:100]   5×5 local grid × 4 resource channels  (food, qi, materials, poison)
    [100:200] 5×5 local grid × 4 agent channels     (agent_present, health, strength, sociability)
    [200:250] 5×5 local grid × 2 stash channels     (my_stash, enemy_stash)
    [250:263] Self stats × 13:
                health, hunger, inv_food, inv_poison,
                poison_resistance, poison_intake,
                combat_experience,      # fights survived / 100
                terrain_familiarity,    # ticks near food / TERRAIN_FAM_SCALE, capped 1.0
                recent_reward_ema,      # EMA of per-step rewards, normalised to [0,1]
                sociability,            # own personality trait
                in_group,               # 1.0 if currently in a group, 0.0 otherwise
                strength,               # current base strength
                hunger_resistance       # trait: how well agent tolerates hunger

Action space: Discrete(15) — N_ACTIONS_PHASE6
    0–3:  MOVE (N/S/E/W)
    4:    EAT
    5:    GATHER
    6:    REST
    7:    DEPOSIT
    8:    WITHDRAW
    9:    STEAL
    10:   ATTACK
    11:   DEFEND
    12:   COLLABORATE
    13:   WALK_AWAY
    14:   TRAIN
"""
from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from murimsim.actions import Action, MOVE_DELTAS, N_ACTIONS_PHASE2, N_ACTIONS_PHASE3, N_ACTIONS_PHASE5, N_ACTIONS_PHASE6
from murimsim.agent import Agent, inherit_value  # noqa: F401 (inherit_value re-exported for tests)
from murimsim.sect import SectConfig
from murimsim.stash import StashRegistry
from murimsim.world import World

# ── Observation layout constants ─────────────────────────────────────────────
OBS_VIEW_SIZE: int = 5

OBS_N_RESOURCE_CH: int = 4          # food, qi, materials, poison
OBS_N_AGENT_CH: int = 4             # agent_present, agent_health, agent_strength, agent_sociability
OBS_N_STASH_CH: int = 2             # my_stash, enemy_stash
OBS_CHANNEL_ORDER: list[str] = ["food", "qi", "materials", "poison"]

OBS_RESOURCE_GRID_SIZE: int = OBS_VIEW_SIZE * OBS_VIEW_SIZE * OBS_N_RESOURCE_CH  # 100
OBS_AGENT_GRID_SIZE: int = OBS_VIEW_SIZE * OBS_VIEW_SIZE * OBS_N_AGENT_CH        # 100
OBS_STASH_GRID_SIZE: int = OBS_VIEW_SIZE * OBS_VIEW_SIZE * OBS_N_STASH_CH        # 50
OBS_STATS_SIZE: int = 13  # health, hunger, inv_food, inv_poison, pr, pi, combat_exp, terrain_fam, reward_ema, sociability, in_group, strength, hunger_resistance
OBS_TOTAL_SIZE: int = OBS_RESOURCE_GRID_SIZE + OBS_AGENT_GRID_SIZE + OBS_STASH_GRID_SIZE + OBS_STATS_SIZE  # 263

# ── History signal constants ──────────────────────────────────────────────────
TERRAIN_FAM_SCALE: float = 200.0   # ticks_near_food / SCALE → [0, 1]
REWARD_EMA_ALPHA: float = 0.10
REWARD_EMA_SCALE: float = 0.5      # EMA normalised: 0 = −scale, 1 = +scale

# ── Heuristic constants (non-focal agents) ────────────────────────────────────
HEURISTIC_HUNGER_EAT: float = 0.5   # eat when hunger exceeds this
HEURISTIC_SCAN_RADIUS: int = 3      # Manhattan radius for food scan

# ── Hazard tracking ───────────────────────────────────────────────────────────
# HAZARD_RESOURCE_IDS is derived from world.hazard_ids (effect=='negative') at env init.

# ── Reward shaping (Stage 5: potential-based) ────────────────────────────────
REWARD_ALIVE: float = 0.02
REWARD_HUNGER_RELIEF_SCALE: float = 0.20
REWARD_FOOD_GATHERED_SCALE: float = 0.15           # raised: gather must outpace eat reward
REWARD_HAZARD_DAMAGE_SCALE: float = -0.30          # unified: traversal + consumption damage
REWARD_DEATH: float = -1.00
REWARD_EXPLORE_BASE: float = 0.25                  # multiplied by (1-hunger) in step
# Potential-based inventory security shaping: reward Δ(food_in_hand / INV_CAP)
REWARD_INV_SECURITY_SCALE: float = 0.12
INV_SECURITY_CAP: float = 5.0                      # normalise over first 5 food items
# Starvation proximity penalty: discourages approaching the danger zone
PENALTY_STARVATION_APPROACH: float = -0.08
STARVATION_THRESHOLD: float = 0.80                 # synced with Agent.STARVATION_THRESHOLD
# Health recovery bonus: only fires when health is meaningfully low (< HEALTH_RECOVERY_GATE)
# This prevents eat-farming when already healthy
REWARD_HEALTH_RECOVERY_SCALE: float = 0.20
HEALTH_RECOVERY_GATE: float = 0.70                 # no recovery reward above this health level
# δ-reward for TRAIN action: incentivises training (strength delta × scale)
REWARD_TRAIN_STRENGTH_SCALE: float = 10.0
REWARD_RESISTANCE_GAIN_SCALE: float = 5.0   # reward per unit of total resistance grown

# Power score weights (used for ep_avg_power metric, logged per episode)
POWER_WEIGHT_STRENGTH: float = 0.4
POWER_WEIGHT_QI: float = 0.3      # maps to qi_drain_resistance (cultivation level)
POWER_WEIGHT_POISON: float = 0.2
POWER_WEIGHT_FLAME: float = 0.1


def compute_power_score(agent: "Agent") -> float:
    """Scalar cultivation power in [0, 1].

    power = 0.4 * strength
          + 0.3 * qi_drain_resistance
          + 0.2 * poison_resistance
          + 0.1 * flame_resistance
    """
    return (
        POWER_WEIGHT_STRENGTH * agent.strength
        + POWER_WEIGHT_QI * agent.resistances.get("qi_drain", 0.0)
        + POWER_WEIGHT_POISON * agent.resistances.get("poison", 0.0)
        + POWER_WEIGHT_FLAME * agent.resistances.get("flame", 0.0)
    )


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

        # Active groups: each entry is a frozenset of agent indices. An agent
        # belongs to at most one group. Groups dissolve when size drops to 1.
        self._groups: list[frozenset[int]] = []

        # Per-agent history state (reset each episode)
        self._visited_tiles: list[set[tuple[int, int]]] = []
        self._ticks_near_food: list[float] = []
        self._reward_ema: list[float] = []
        self._combat_experience: list[float] = []  # Phase 3c: updated on fights

    # ── Gymnasium API ────────────────────────────────────────────────────────

    def _initial_position(self, idx: int, grid_size: int) -> tuple[int, int]:
        """Return the starting ``(x, y)`` position for agent *idx* at episode reset.

        Subclasses may override this to constrain agents to a home region.
        By default, positions are uniformly random within the grid.
        """
        return (int(self._rng.integers(0, grid_size)), int(self._rng.integers(0, grid_size)))

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
                    # Clustered food maps: 40% of episodes spawn food in spatial clusters.
                    # Agents learn that food is patchily distributed — encourages foraging
                    # and returning to known sources rather than random wandering.
                    cluster_prob = float(dr.get("food_cluster_prob", 0.0))
                    if cluster_prob > 0 and self._rng.random() < cluster_prob:
                        resource["spawn_clusters"] = True
                        lo, hi = dr.get("food_cluster_count", [2, 4])
                        resource["cluster_count"] = int(self._rng.integers(lo, hi + 1))
                        resource["cluster_radius"] = int(dr.get("food_cluster_radius", 3))
                        resource["cluster_fill_prob"] = float(dr.get("food_cluster_fill_prob", 0.70))
                    else:
                        resource["spawn_clusters"] = False
            lo, hi = dr.get("action_ticks", [3, 8])
            cfg["world"]["action_ticks"] = int(self._rng.integers(lo, hi + 1))

        self._action_ticks = int(cfg["world"].get("action_ticks", 1))
        self._world = World(cfg, rng=np.random.default_rng(effective_seed))
        self._resource_configs = self._world.resources
        self._max_age: int = int(cfg.get("agent", {}).get("max_age", 0))

        gs = self._world.grid_size
        self._agents = [
            Agent.spawn(
                f"agent_{i}",
                self._initial_position(i, gs),
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
        self._groups = []

        # Reciprocity memory: _help_received[recipient][helper] = step_when_helped
        # Cleared each episode so past-life debts don't carry over.
        self._help_received: dict[int, dict[int, int]] = {}
        # Per-episode step counter (for reciprocity window comparison)
        self._ep_step_count: int = 0

        # Per-agent cumulative reward (individual credit assignment baseline)
        # Index = agent slot; reset each episode.
        self._ep_agent_rewards: list[float] = [0.0] * self._n_agents
        self._ep_agent_steps: list[int] = [0] * self._n_agents

        # Episode-level action counts for dashboard metrics (focal agent only)
        self._ep_action_counts: dict[str, int] = {}
        self._ep_steps: int = 0
        self._ep_focal_strength_sum: float = 0.0  # sum of focal agent's strength each step
        self._ep_focal_power_sum: float = 0.0     # sum of focal agent's power score each step
        # Hazard approach/flee counters — keyed by YAML hazard IDs (effect=='negative')
        hazard_ids = self._world.hazard_ids
        self._ep_hazard_approaches: dict[str, int] = {h: 0 for h in hazard_ids}
        self._ep_hazard_flees: dict[str, int] = {h: 0 for h in hazard_ids}

        # Settlement metrics (reset each episode)
        # visit_counts[i][(x,y)] = number of times agent i visited that tile
        self._visit_counts: list[dict[tuple[int, int], int]] = [{} for _ in range(self._n_agents)]
        self._ep_items_gathered: int = 0        # total items picked up from world
        self._ep_items_deposited: int = 0       # total items moved into stashes
        self._ep_items_withdrawn: int = 0       # total items retrieved from own stashes
        self._ep_dist_from_stash_sum: float = 0.0   # sum of per-(agent,step) min Chebyshev dist to own stash
        self._ep_dist_from_stash_count: int = 0     # denominator for the above
        self._ep_groups_formed: int = 0         # how many times _form_group was called
        self._ep_group_member_ticks: int = 0    # sum of group sizes across all steps
        self._ep_deaths_by_cause: dict[str, int] = {}  # cause -> count
        self._ep_reproductions: int = 0         # offspring spawned from parent pairs

        # Foraging-outward tracking: max Chebyshev dist from own stash since last deposit
        self._max_dist_since_deposit: list[float] = [0.0] * self._n_agents

        self._stash_registry.reset()

        return self._build_obs(self._focal_idx), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._world is not None

        focal = self._agents[self._focal_idx]
        hunger_prev = focal.hunger
        health_prev = focal.health
        inv_food_prev = focal.inventory.food
        strength_prev = focal.strength
        resistance_sum_prev = sum(focal.resistances.values())
        food_gathered = 0
        hazard_damage = 0.0

        # 1. Apply action to focal agent
        action_enum = Action(action)
        prev_pos = focal.position
        pre_hazard_dists = {h: self._nearest_hazard_dist(prev_pos, h) for h in self._ep_hazard_approaches}

        food_gathered, hazard_damage, stash_bonus = self._apply_action(focal, action_enum, self._focal_idx)

        # Update approach/flee counters for MOVE actions
        if action_enum in MOVE_DELTAS:
            for h in self._ep_hazard_approaches:
                post_dist = self._nearest_hazard_dist(focal.position, h)
                pre_dist = pre_hazard_dists[h]
                if pre_dist < float("inf"):
                    if post_dist < pre_dist:
                        self._ep_hazard_approaches[h] += 1
                    elif post_dist > pre_dist:
                        self._ep_hazard_flees[h] += 1

        # Track action counts for dashboard
        key = action_enum.name.lower()
        self._ep_action_counts[key] = self._ep_action_counts.get(key, 0) + 1
        self._ep_steps += 1
        self._ep_step_count += 1
        self._ep_focal_strength_sum += focal.strength
        self._ep_focal_power_sum += compute_power_score(focal)
        for i, agent in enumerate(self._agents):
            if i != self._focal_idx and agent.alive:
                self._heuristic_step(agent)

        # 3. Advance world + all agents
        for _ in range(self._action_ticks):
            self._world.step()
            for agent in self._agents:
                was_alive = agent.alive
                agent.tick(self._max_age)
                if was_alive and not agent.alive:
                    cause = agent.death_cause or "unknown"
                    self._ep_deaths_by_cause[cause] = self._ep_deaths_by_cause.get(cause, 0) + 1
                    self._drop_inventory(agent)
                    self._try_reproduce(agent)

        self._prune_dead_from_groups()

        # Settlement tracking: per-step updates for all agents
        self._ep_group_member_ticks += sum(len(g) for g in self._groups)
        for i, agent in enumerate(self._agents):
            if agent.alive:
                pos = agent.position
                self._visit_counts[i][pos] = self._visit_counts[i].get(pos, 0) + 1
                stashes = self._stash_registry.get_stashes_for_owner(agent.agent_id)
                if stashes:
                    min_dist = min(
                        max(abs(pos[0] - s.position[0]), abs(pos[1] - s.position[1]))
                        for s in stashes
                    )
                    self._ep_dist_from_stash_sum += min_dist
                    self._ep_dist_from_stash_count += 1

        # Food sharing: each live agent attempts to share with critically hungry group allies
        food_share_reward = 0.0
        focal_idx = self._focal_idx
        for sharer_idx in range(self._n_agents):
            if not self._agents[sharer_idx].alive:
                continue
            group = self._get_group(sharer_idx)
            if group is None:
                continue
            for recipient_idx in group:
                if recipient_idx == sharer_idx:
                    continue
                if self._try_food_share(sharer_idx, recipient_idx):
                    if sharer_idx == focal_idx or recipient_idx == focal_idx:
                        food_share_reward += REWARD_FOOD_SHARE

        # 4. Update history for focal agent
        # Exploration reward is survival-gated: full reward when well-fed, zero when starving
        exploration_reward = 0.0
        if focal.alive and focal.position not in self._visited_tiles[self._focal_idx]:
            self._visited_tiles[self._focal_idx].add(focal.position)
            survival_gate = max(0.0, 1.0 - focal.hunger)
            exploration_reward = REWARD_EXPLORE_BASE * focal.adventure_spirit * survival_gate

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
        reward = self._compute_reward(hunger_prev, health_prev, food_gathered, hazard_damage, focal, exploration_reward, inv_food_prev)
        if focal.alive:
            reward += food_share_reward
            reward += stash_bonus
            reward += self._stash_proximity_reward(self._focal_idx)
            # TRAIN action strength reward: delta(strength) * scale (potential-based)
            strength_delta = focal.strength - strength_prev
            if strength_delta > 0:
                reward += REWARD_TRAIN_STRENGTH_SCALE * strength_delta
            # Resistance growth reward: any resistance gained via hazard exposure
            resistance_delta = sum(focal.resistances.values()) - resistance_sum_prev
            if resistance_delta > 0:
                reward += REWARD_RESISTANCE_GAIN_SCALE * resistance_delta
        ema = self._reward_ema[self._focal_idx]
        self._reward_ema[self._focal_idx] = (1.0 - REWARD_EMA_ALPHA) * ema + REWARD_EMA_ALPHA * reward

        # Accumulate per-agent individual reward (credit assignment baseline)
        self._ep_agent_rewards[self._focal_idx] += reward
        self._ep_agent_steps[self._focal_idx] += 1

        terminated = not focal.alive

        # 6. Advance focal index to next live agent (skip dead agents)
        if not terminated:
            self._focal_idx = self._next_live(self._focal_idx)

        obs = self._build_obs(self._focal_idx)
        action_rates = {
            k: v / max(1, self._ep_steps)
            for k, v in self._ep_action_counts.items()
        }
        info = {
            "hunger": focal.hunger,
            "health": focal.health,
            "alive_count": sum(1 for a in self._agents if a.alive),
            "ep_steps": self._ep_steps,
            "ep_action_counts": dict(self._ep_action_counts),
            "ep_action_rates": action_rates,
            "ep_hazard_approaches": dict(self._ep_hazard_approaches),
            "ep_hazard_flees": dict(self._ep_hazard_flees),
        }
        if terminated:
            info["ep_lifespan"] = self._ep_steps
            info["ep_avg_strength"] = (
                self._ep_focal_strength_sum / self._ep_steps if self._ep_steps > 0 else 0.0
            )
            info["ep_avg_power"] = (
                self._ep_focal_power_sum / self._ep_steps if self._ep_steps > 0 else 0.0
            )
            info["ep_final_power"] = compute_power_score(focal)
            # Per-agent credit assignment data: cumulative and mean reward per slot
            info["ep_agent_rewards"] = list(self._ep_agent_rewards)
            info["ep_agent_steps"] = list(self._ep_agent_steps)
            info["ep_agent_mean_reward"] = [
                r / max(1, s)
                for r, s in zip(self._ep_agent_rewards, self._ep_agent_steps)
            ]
            # Settlement metrics
            info["ep_stash_fill_rate"] = self._ep_items_deposited / max(1, self._ep_items_gathered)
            info["ep_stash_withdraw_rate"] = self._ep_items_withdrawn / max(1, self._ep_items_deposited)
            info["ep_avg_dist_from_stash"] = (
                self._ep_dist_from_stash_sum / self._ep_dist_from_stash_count
                if self._ep_dist_from_stash_count > 0 else 0.0
            )
            info["ep_revisit_entropy"] = self._compute_revisit_entropy()
            info["ep_group_persistence"] = (
                self._ep_group_member_ticks / self._ep_groups_formed
                if self._ep_groups_formed > 0 else 0.0
            )
            info["ep_deaths_by_age"] = self._ep_deaths_by_cause.get("age", 0)
            info["ep_deaths_by_cause"] = dict(self._ep_deaths_by_cause)
            info["ep_reproductions"] = self._ep_reproductions
        return obs, reward, terminated, False, info

    def render(self) -> None:
        pass  # Rendering handled by web viewer

    def _compute_revisit_entropy(self) -> float:
        """Mean per-agent Shannon entropy of tile-visit frequency distribution.

        Low entropy means an agent visited few distinct tiles many times (settled).
        High entropy means visits are spread across many tiles (roaming).
        """
        entropies: list[float] = []
        for counts in self._visit_counts:
            total = sum(counts.values())
            if total == 0:
                continue
            h = -sum((c / total) * math.log(c / total) for c in counts.values())
            entropies.append(h)
        return float(np.mean(entropies)) if entropies else 0.0

    # ── Group helpers ─────────────────────────────────────────────────────────

    def _get_group(self, agent_idx: int) -> frozenset[int] | None:
        """Return the group the agent belongs to, or None if solo."""
        for g in self._groups:
            if agent_idx in g:
                return g
        return None

    def _form_group(self, idx_a: int, idx_b: int) -> None:
        """Merge two agents (or their groups) into one group."""
        group_a = self._get_group(idx_a)
        group_b = self._get_group(idx_b)
        members: set[int] = set()
        if group_a is not None:
            self._groups.remove(group_a)
            members.update(group_a)
        else:
            members.add(idx_a)
        if group_b is not None:
            self._groups.remove(group_b)
            members.update(group_b)
        else:
            members.add(idx_b)
        self._groups.append(frozenset(members))
        self._ep_groups_formed += 1

    def _leave_group(self, agent_idx: int) -> None:
        """Remove an agent from its group; dissolve group if only 1 member remains."""
        group = self._get_group(agent_idx)
        if group is None:
            return
        self._groups.remove(group)
        remaining = group - {agent_idx}
        if len(remaining) > 1:
            self._groups.append(frozenset(remaining))

    def _prune_dead_from_groups(self) -> None:
        """Remove dead agents from all groups; dissolve groups that become size ≤ 1."""
        alive_set = {i for i, a in enumerate(self._agents) if a.alive}
        new_groups: list[frozenset[int]] = []
        for g in self._groups:
            pruned = g & alive_set
            if len(pruned) >= 2:
                new_groups.append(frozenset(pruned))
        self._groups = new_groups

    def _drop_inventory(self, agent: Agent) -> None:
        """Drop a dead agent's food inventory onto its tile (if tile is empty)."""
        if agent.inventory.food <= 0:
            return
        x, y = agent.position
        if self._world.get_grid_view("food")[y, x] == 0.0:
            self._world._grid["food"][y, x] = 1.0
        agent.inventory.food = 0

    def _try_reproduce(self, deceased: Agent) -> None:
        """Replace a dead (age-death) agent with an offspring of two random survivors.

        Two living agents are chosen at random from the current population.  The
        deceased agent's slot is revived in-place with inherited traits via
        :meth:`Agent.spawn_from_parents`.  The offspring is placed at a random
        empty position (or the deceased's last position if no empty cell is
        found).  Nothing happens if fewer than 2 survivors are alive.
        """
        survivors = [a for a in self._agents if a.alive and a is not deceased]
        if len(survivors) < 2:
            return
        parent1, parent2 = self._rng.choice(survivors, size=2, replace=False)  # type: ignore[arg-type]
        # Find a random spawn position
        grid_size = self._world.grid_size
        pos = self._initial_position(len(self._agents), grid_size)
        offspring = Agent.spawn_from_parents(
            agent_id=deceased.agent_id,
            position=pos,
            parent1=parent1,
            parent2=parent2,
            rng=self._rng,
        )
        # Replace the deceased in the _agents list
        idx = self._agents.index(deceased)
        self._agents[idx] = offspring
        self._ep_reproductions += 1

    def _stash_proximity_reward(self, agent_idx: int) -> float:
        """Per-tick stash proximity bonus — currently disabled (REWARD_STASH_PROXIMITY=0.0).

        Was found to pull agents to individual stash locations, causing group
        dispersal and WALK_AWAY rate spike. Kept as a hook for future tuning.
        """
        if REWARD_STASH_PROXIMITY == 0.0:
            return 0.0
        agent = self._agents[agent_idx]
        if agent.hunger <= STASH_HUNGER_GATE:
            return 0.0
        stashes = self._stash_registry.get_stashes_for_owner(agent.agent_id)
        if not stashes:
            return 0.0
        ax, ay = agent.position
        for s in stashes:
            if max(abs(ax - s.position[0]), abs(ay - s.position[1])) <= STASH_PROXIMITY_RANGE:
                return REWARD_STASH_PROXIMITY
        return 0.0

    def _group_cohesion_reward(self, agent_idx: int) -> float:
        """Return a per-tick reward for each live group member within GROUP_COHESION_RANGE.

        Uses Chebyshev distance (consistent with 8-direction combat range).
        Incentivises staying physically close to group members rather than
        drifting apart after forming a group.
        """
        group = self._get_group(agent_idx)
        if group is None:
            return 0.0
        ax, ay = self._agents[agent_idx].position
        nearby_count = 0
        for ally_idx in group:
            if ally_idx == agent_idx:
                continue
            ally = self._agents[ally_idx]
            if not ally.alive:
                continue
            ox, oy = ally.position
            if max(abs(ox - ax), abs(oy - ay)) <= GROUP_COHESION_RANGE:
                nearby_count += 1
        return REWARD_GROUP_COHESION_PER_ALLY * nearby_count

    def _try_food_share(self, sharer_idx: int, recipient_idx: int) -> bool:
        """Attempt to share one food unit from sharer to recipient.

        Succeeds only if:
          - Recipient is alive, in the same group as sharer, and critically hungry.
          - Sharer has food in inventory.
          - Reciprocity roll passes (boosted if recipient helped sharer recently).

        Returns True if food was actually transferred.
        """
        sharer = self._agents[sharer_idx]
        recipient = self._agents[recipient_idx]

        if not (sharer.alive and recipient.alive):
            return False
        if self._get_group(sharer_idx) != self._get_group(recipient_idx):
            return False
        if self._get_group(sharer_idx) is None:
            return False
        if recipient.hunger < SHARE_HUNGER_THRESHOLD:
            return False
        if sharer.inventory.food == 0:
            return False

        # Reciprocity: did the recipient help the sharer recently?
        last_help = self._help_received.get(sharer_idx, {}).get(recipient_idx, -RECIPROCITY_WINDOW - 1)
        boosted = (self._ep_step_count - last_help) <= RECIPROCITY_WINDOW
        threshold = RECIPROCITY_BOOSTED if boosted else RECIPROCITY_BASE

        if self._rng.random() > threshold:
            return False

        # Transfer one food unit
        sharer.inventory.food -= 1
        recipient.inventory.food += 1

        # Record that sharer helped recipient
        if recipient_idx not in self._help_received:
            self._help_received[recipient_idx] = {}
        self._help_received[recipient_idx][sharer_idx] = self._ep_step_count

        return True

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

        # Agent channels (agent_present, agent_health, agent_strength, agent_sociability)
        agent_window = np.zeros((OBS_VIEW_SIZE, OBS_VIEW_SIZE, OBS_N_AGENT_CH), dtype=np.float32)
        for i, other in enumerate(self._agents):
            if i == agent_idx or not other.alive:
                continue
            ox, oy = other.position
            wx = (ox - ax) + half
            wy = (oy - ay) + half
            if 0 <= wx < OBS_VIEW_SIZE and 0 <= wy < OBS_VIEW_SIZE:
                agent_window[wy, wx, 0] = max(agent_window[wy, wx, 0], 1.0)                  # present
                agent_window[wy, wx, 1] = max(agent_window[wy, wx, 1], other.health)          # health
                agent_window[wy, wx, 2] = max(agent_window[wy, wx, 2], other.strength)        # strength
                agent_window[wy, wx, 3] = max(agent_window[wy, wx, 3], other.sociability)     # sociability
        flat_agents = agent_window.reshape(-1)  # 100

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

        # Self stats (11 values)
        inv = agent.inventory
        terrain_fam = min(1.0, self._ticks_near_food[agent_idx] / TERRAIN_FAM_SCALE)
        raw_ema = self._reward_ema[agent_idx]
        reward_ema_norm = float(np.clip(0.5 + raw_ema / (2.0 * REWARD_EMA_SCALE), 0.0, 1.0))
        combat_exp = min(1.0, self._combat_experience[agent_idx] / 100.0)
        in_group = 1.0 if self._get_group(agent_idx) is not None else 0.0

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
            agent.sociability,
            in_group,
            agent.strength,
            agent.hunger_resistance,
        ], dtype=np.float32)

        return np.concatenate([flat_resources, flat_agents, flat_stashes, stats])

    # ── Reward ───────────────────────────────────────────────────────────────

    def _compute_reward(
        self,
        hunger_prev: float,
        health_prev: float,
        food_gathered: int,
        hazard_damage: float,
        agent: Agent,
        exploration_reward: float = 0.0,
        inv_food_prev: int = 0,
    ) -> float:
        """Compute shaped per-step reward.

        Stage 5 additions:
        - Potential-based inventory security: reward Δ(food_in_hand) so gathering
          is worth more than the raw food-gathered bonus alone.
        - Starvation proximity penalty: discourages drifting into the danger zone
          rather than waiting until death.
        - Exploration is survival-gated (passed in pre-scaled by caller).

        Stage 6b additions:
        - Health recovery bonus: rewards regaining health after eating or resting.
        """
        reward = REWARD_ALIVE
        hunger_relief = hunger_prev - agent.hunger
        if hunger_relief > 0:
            reward += REWARD_HUNGER_RELIEF_SCALE * hunger_relief
        reward += REWARD_FOOD_GATHERED_SCALE * food_gathered
        reward += REWARD_HAZARD_DAMAGE_SCALE * hazard_damage
        reward += exploration_reward
        # Potential-based inventory shaping: φ(s) = food_in_hand / INV_SECURITY_CAP
        inv_delta = (agent.inventory.food - inv_food_prev) / INV_SECURITY_CAP
        reward += REWARD_INV_SECURITY_SCALE * inv_delta
        # Starvation proximity penalty
        if agent.hunger > STARVATION_THRESHOLD:
            reward += PENALTY_STARVATION_APPROACH * (agent.hunger - STARVATION_THRESHOLD)
        # Health recovery bonus: only reward recovering health when it's meaningfully low.
        # Gate prevents eat-farming: no bonus when already healthy (>= HEALTH_RECOVERY_GATE).
        health_delta = agent.health - health_prev
        if health_delta > 0 and health_prev < HEALTH_RECOVERY_GATE:
            reward += REWARD_HEALTH_RECOVERY_SCALE * health_delta
        if not agent.alive:
            reward += REWARD_DEATH
        return float(reward)

    # ── Action application ────────────────────────────────────────────────────

    def _apply_action(self, agent: Agent, action_enum: Action, agent_idx: int = -1) -> tuple[int, float, float]:
        """Apply action_enum to agent. Returns (food_gathered, hazard_damage, stash_bonus).

        stash_bonus is non-zero only when a DEPOSIT qualifies for the
        foraging-outward reward (agent was >= FORAGE_OUTWARD_MIN_DIST tiles from
        the stash between this and its previous deposit).
        """
        food_gathered = 0
        hazard_damage = 0.0
        stash_bonus = 0.0

        if action_enum in MOVE_DELTAS:
            dx, dy = MOVE_DELTAS[action_enum]
            agent.move(dx, dy, self._world.grid_size)
            # Update max distance from own stash since last deposit
            if agent_idx >= 0:
                stashes = self._stash_registry.get_stashes_for_owner(agent.agent_id)
                if stashes:
                    dist = min(
                        max(abs(agent.position[0] - s.position[0]), abs(agent.position[1] - s.position[1]))
                        for s in stashes
                    )
                    if dist > self._max_dist_since_deposit[agent_idx]:
                        self._max_dist_since_deposit[agent_idx] = dist

        elif action_enum == Action.GATHER:
            x, y = agent.position
            for rid in OBS_CHANNEL_ORDER:
                if self._world.get_grid_view(rid)[y, x] > 0:
                    self._world.deplete(rid, x, y)
                    agent.gather(rid)
                    if rid == "food":
                        food_gathered = 1
                    self._ep_items_gathered += 1
                    break

        elif action_enum == Action.EAT:
            hazard_damage = agent.eat(self._resource_configs)

        elif action_enum == Action.REST:
            agent.rest()

        elif action_enum == Action.DEPOSIT:
            stash = self._stash_registry.deposit(agent)
            if stash:
                self._ep_items_deposited += stash.total()
                if agent_idx >= 0 and self._max_dist_since_deposit[agent_idx] >= FORAGE_OUTWARD_MIN_DIST:
                    stash_bonus += REWARD_FORAGE_OUTWARD
                if agent_idx >= 0:
                    self._max_dist_since_deposit[agent_idx] = 0.0

        elif action_enum == Action.WITHDRAW:
            group = self._get_group(agent_idx) if agent_idx >= 0 else None
            if group:
                member_ids = [self._agents[i].agent_id for i in group]
                food_got = self._stash_registry.withdraw_group(agent, member_ids)
                self._ep_items_withdrawn += food_got
            else:
                at_pos = self._stash_registry.get_own_stash_at(agent.agent_id, *agent.position)
                self._ep_items_withdrawn += sum(s.total() for s in at_pos)
                self._stash_registry.withdraw(agent)

        elif action_enum == Action.STEAL:
            self._stash_registry.steal(agent)

        elif action_enum == Action.TRAIN:
            x, y = agent.position
            qi_val = self._world.get_qi_field_value(x, y) if "qi" in self._world.resources else 0.0
            agent.train(qi_field_value=qi_val)
            # Training reward is applied in _compute_reward via strength_delta tracking.

        return food_gathered, hazard_damage, stash_bonus

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

    def _nearest_hazard_dist(self, pos: tuple[int, int], hazard_id: str) -> float:
        """Return Manhattan distance to the nearest tile of hazard_id.

        Returns float('inf') if no such tile exists on the current map.
        """
        assert self._world is not None
        try:
            grid = self._world.get_grid_view(hazard_id)
        except (KeyError, Exception):
            return float("inf")
        ys, xs = np.nonzero(grid)
        if len(xs) == 0:
            return float("inf")
        return float(np.min(np.abs(xs - pos[0]) + np.abs(ys - pos[1])))


# ── Combat constants ──────────────────────────────────────────────────────────
COMBAT_ATTACKER_SCALE: float = 0.3
COMBAT_DEFENDER_SCALE: float = 0.1
COMBAT_MAX_DAMAGE: float = 0.5

REWARD_DEFEAT_OPPONENT: float = 0.3
REWARD_DAMAGE_TAKEN_SCALE: float = -0.2
# Sect-aware combat bonuses (Phase 6a): only active when both agents have named sects.
REWARD_INTER_SECT_DEFEAT_BONUS: float = 0.15   # extra reward for defeating an enemy-sect agent
REWARD_SAME_SECT_ATTACK_PENALTY: float = -0.10 # penalty for attacking a same-sect ally
REWARD_GROUP_FORMATION: float = 0.05   # small bonus when bilateral COLLABORATE succeeds
# Per-tick bonus for each live group member within proximity range.
# Rewards staying in a live, nearby group rather than just being "in" a group.
REWARD_GROUP_COHESION_PER_ALLY: float = 0.0
GROUP_COHESION_RANGE: int = 3  # Chebyshev distance within which an ally counts

# Bonus when focal attacks while a group ally is flanking (adjacent to) the target.
# Directly incentivises positioning next to allies before engaging.
REWARD_COORDINATED_ATTACK: float = 0.10

# Sociability threshold for heuristic agents to accept a collaboration signal
HEURISTIC_COLLAB_THRESHOLD: float = 0.5

# Food sharing with reciprocity
# A group member will attempt to share food when an ally's hunger exceeds this threshold.
SHARE_HUNGER_THRESHOLD: float = 0.85  # ally must be critically starving to trigger share
RECIPROCITY_BASE: float = 0.50        # base chance to share (50%)
RECIPROCITY_BOOSTED: float = 0.85     # boosted chance if ally helped me recently (85%)
RECIPROCITY_WINDOW: int = 100         # steps within which past help is remembered
REWARD_FOOD_SHARE: float = 0.04       # reward focal receives when it shares or is shared with

# Shared stash rewards
# Foraging-outward: deposit after having been >=N tiles away from stash since last deposit
FORAGE_OUTWARD_MIN_DIST: int = 5      # Chebyshev tiles away from stash to qualify
REWARD_FORAGE_OUTWARD: float = 0.03   # bonus for depositing after a foraging excursion
# Group withdrawal bonus: received by stash owner when a group member withdraws their food
REWARD_GROUP_WITHDRAW_BONUS: float = 0.02
# Stash proximity: disabled — per-tick pull toward individual stash was anti-cooperative,
# causing agents to disperse from groups and spike WALK_AWAY rate.
STASH_PROXIMITY_RANGE: int = 3        # kept for test compatibility
STASH_HUNGER_GATE: float = 0.50       # kept for test compatibility
REWARD_STASH_PROXIMITY: float = 0.0   # disabled

# Group combat mechanics
# Combat requires attacker and target to be within Chebyshev distance 1 (8 directions incl. diagonals)
# Attack bonus: +X% damage per group member flanking from any of the 8 surrounding cells
GROUP_ATTACK_BONUS_PER_ALLY: float = 0.20

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
        sect_config: SectConfig | None = None,
    ) -> None:
        super().__init__(
            config=config,
            n_agents=n_agents,
            seed=seed,
            render_mode=render_mode,
            n_actions=N_ACTIONS_PHASE6,
        )
        self._curriculum_ramp_steps: int = curriculum_ramp_steps
        self._global_step_count: int = 0  # persists across episodes
        self._sect_config: SectConfig | None = sect_config
        # Per-agent last action detail — target agent id, stash id, or "" — reset each step
        self._last_action_details: list[str] = [""] * n_agents
        # Per-agent last action name (tracks heuristic agents too, unlike record_combat.py)
        self._last_action_names: list[str] = ["rest"] * n_agents

    # ── Sect home-region spawning ─────────────────────────────────────────────

    def _initial_position(self, idx: int, grid_size: int) -> tuple[int, int]:
        """Spawn agents inside the sect's home region when a sect is configured."""
        if self._sect_config is None:
            return super()._initial_position(idx, grid_size)
        x_lo, x_hi = self._sect_config.home_x_range
        y_lo, y_hi = self._sect_config.home_y_range
        x = int(self._rng.integers(x_lo, x_hi + 1))
        y = int(self._rng.integers(y_lo, y_hi + 1))
        return (x, y)

    # ── Curriculum ────────────────────────────────────────────────────────────

    @property
    def combat_prob(self) -> float:
        """Current probability that a combat action is allowed (not masked to REST)."""
        frac = min(1.0, self._global_step_count / max(1, self._curriculum_ramp_steps))
        return CURRICULUM_START_PROB + (CURRICULUM_END_PROB - CURRICULUM_START_PROB) * frac

    # ── reset override ────────────────────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment and assign sect_id to all agents if configured."""
        obs, info = super().reset(seed=seed, options=options)
        if self._sect_config is not None:
            for agent in self._agents:
                agent.sect_id = self._sect_config.sect_id
        return obs, info

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
        health_prev = focal.health
        inv_food_prev = focal.inventory.food
        strength_prev = focal.strength
        resistance_sum_prev = sum(focal.resistances.values())
        food_gathered = 0
        hazard_damage = 0.0
        damage_dealt = 0.0
        defeat_bonus = 0.0
        focal_defending = (action_enum == Action.DEFEND)
        group_formed = False
        stash_bonus = 0.0
        sect_attack_penalty = 0.0

        # Snapshot hazard distances before the action for approach/flee tracking
        prev_pos = focal.position
        pre_hazard_dists = {h: self._nearest_hazard_dist(prev_pos, h) for h in self._ep_hazard_approaches}

        # 1. Apply focal agent's action
        flanking_bonus_earned = False
        self._last_action_details[self._focal_idx] = ""
        self._last_action_names[self._focal_idx] = action_enum.name.lower()
        if action_enum == Action.ATTACK:
            # Check for flanking allies before the attack (target may die during it)
            pre_target = self._nearest_adjacent_agent(focal)
            if pre_target is not None:
                flankers = self._adjacent_group_allies(self._focal_idx, pre_target)
                flanking_bonus_earned = len(flankers) > 0
                # Sect-aware reward shaping (only when both agents belong to named sects)
                if focal.sect_id != "none" and pre_target.sect_id != "none":
                    if focal.sect_id == pre_target.sect_id:
                        sect_attack_penalty = REWARD_SAME_SECT_ATTACK_PENALTY
                self._last_action_details[self._focal_idx] = pre_target.agent_id
            else:
                self._last_action_details[self._focal_idx] = "no_target"
            damage_dealt, defeated = self._do_attack(focal)
            defeat_bonus = REWARD_DEFEAT_OPPONENT if defeated else 0.0
            # Extra defeat bonus when eliminating an enemy-sect agent
            if defeated and pre_target is not None:
                if focal.sect_id != "none" and pre_target.sect_id != "none":
                    if focal.sect_id != pre_target.sect_id:
                        defeat_bonus += REWARD_INTER_SECT_DEFEAT_BONUS
        elif action_enum == Action.DEFEND:
            focal.rest()  # defending = hold ground + minor health recovery
        elif action_enum == Action.COLLABORATE:
            neighbour = self._nearest_adjacent_agent(focal)
            if neighbour is not None:
                self._last_action_details[self._focal_idx] = neighbour.agent_id
            group_formed = self._try_collaborate(self._focal_idx)
        elif action_enum == Action.WALK_AWAY:
            neighbour = self._nearest_adjacent_agent(focal)
            if neighbour is not None:
                self._last_action_details[self._focal_idx] = neighbour.agent_id
            self._walk_away(focal)
        else:
            food_gathered, hazard_damage, stash_bonus = self._apply_action(focal, action_enum, self._focal_idx)
            # Tag deposit/withdraw with the stash id
            if action_enum in (Action.DEPOSIT, Action.WITHDRAW):
                stashes = self._stash_registry.get_stashes_for_owner(focal.agent_id)
                if stashes:
                    nearest = min(stashes, key=lambda s: max(abs(s.position[0] - focal.position[0]), abs(s.position[1] - focal.position[1])))
                    self._last_action_details[self._focal_idx] = nearest.stash_id

        # Track action counts and hazard approach/flee for dashboard
        key = action_enum.name.lower()
        self._ep_action_counts[key] = self._ep_action_counts.get(key, 0) + 1
        self._ep_steps += 1
        self._ep_step_count += 1
        self._ep_focal_strength_sum += focal.strength
        self._ep_focal_power_sum += compute_power_score(focal)

        if action_enum in MOVE_DELTAS:
            for h in self._ep_hazard_approaches:
                post_dist = self._nearest_hazard_dist(focal.position, h)
                pre_dist = pre_hazard_dists[h]
                if pre_dist < float("inf"):
                    if post_dist < pre_dist:
                        self._ep_hazard_approaches[h] += 1
                    elif post_dist > pre_dist:
                        self._ep_hazard_flees[h] += 1

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
                was_alive = agent.alive
                agent.tick(self._max_age)
                if was_alive and not agent.alive:
                    cause = agent.death_cause or "unknown"
                    self._ep_deaths_by_cause[cause] = self._ep_deaths_by_cause.get(cause, 0) + 1
                    self._drop_inventory(agent)
                    self._try_reproduce(agent)

        # Remove dead agents from any groups
        self._prune_dead_from_groups()

        # Settlement tracking: per-step updates for all agents
        self._ep_group_member_ticks += sum(len(g) for g in self._groups)
        for i, agent in enumerate(self._agents):
            if agent.alive:
                pos = agent.position
                self._visit_counts[i][pos] = self._visit_counts[i].get(pos, 0) + 1
                stashes = self._stash_registry.get_stashes_for_owner(agent.agent_id)
                if stashes:
                    min_dist = min(
                        max(abs(pos[0] - s.position[0]), abs(pos[1] - s.position[1]))
                        for s in stashes
                    )
                    self._ep_dist_from_stash_sum += min_dist
                    self._ep_dist_from_stash_count += 1

        # Food sharing: each live agent attempts to share with critically hungry group allies.
        # Focal agent gets a reward signal; heuristic agents share silently.
        food_share_reward = 0.0
        focal_idx = self._focal_idx
        for sharer_idx in range(self._n_agents):
            if not self._agents[sharer_idx].alive:
                continue
            group = self._get_group(sharer_idx)
            if group is None:
                continue
            for recipient_idx in group:
                if recipient_idx == sharer_idx:
                    continue
                if self._try_food_share(sharer_idx, recipient_idx):
                    # Focal gets reward if it shared OR if it was the recipient
                    if sharer_idx == focal_idx or recipient_idx == focal_idx:
                        food_share_reward += REWARD_FOOD_SHARE

        # 4. Update history for focal agent
        # Exploration reward is survival-gated: full reward when well-fed, zero when starving
        exploration_reward = 0.0
        if focal.alive and focal.position not in self._visited_tiles[self._focal_idx]:
            self._visited_tiles[self._focal_idx].add(focal.position)
            survival_gate = max(0.0, 1.0 - focal.hunger)
            exploration_reward = REWARD_EXPLORE_BASE * focal.adventure_spirit * survival_gate

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

        # 6. Compute reward (group formation bonus + cohesion bonus added on top)
        reward = self._compute_combat_reward(
            hunger_prev, health_prev, food_gathered, hazard_damage, focal,
            exploration_reward, damage_dealt, damage_taken, defeat_bonus,
            inv_food_prev,
        )
        if group_formed:
            reward += REWARD_GROUP_FORMATION
        # Per-tick cohesion reward: alive group members within range
        if focal.alive:
            reward += self._group_cohesion_reward(self._focal_idx)
        # Coordinated attack: bonus when focal attacks with a flanking group ally
        if flanking_bonus_earned and focal.alive:
            reward += REWARD_COORDINATED_ATTACK
        # Sect-aware combat penalty: discourage attacking same-sect allies
        reward += sect_attack_penalty
        # Food sharing: reward focal for participating in mutual aid
        if focal.alive:
            reward += food_share_reward
            reward += stash_bonus
            reward += self._stash_proximity_reward(self._focal_idx)
            # TRAIN action strength reward: delta(strength) * scale (potential-based)
            strength_delta = focal.strength - strength_prev
            if strength_delta > 0:
                reward += REWARD_TRAIN_STRENGTH_SCALE * strength_delta
            # Resistance growth reward: any resistance gained via hazard exposure
            resistance_delta = sum(focal.resistances.values()) - resistance_sum_prev
            if resistance_delta > 0:
                reward += REWARD_RESISTANCE_GAIN_SCALE * resistance_delta
        ema = self._reward_ema[self._focal_idx]
        self._reward_ema[self._focal_idx] = (1.0 - REWARD_EMA_ALPHA) * ema + REWARD_EMA_ALPHA * reward

        # Accumulate per-agent individual reward (credit assignment baseline)
        self._ep_agent_rewards[self._focal_idx] += reward
        self._ep_agent_steps[self._focal_idx] += 1

        terminated = not focal.alive

        if not terminated:
            self._focal_idx = self._next_live(self._focal_idx)

        obs = self._build_obs(self._focal_idx)
        action_rates = {
            k: v / max(1, self._ep_steps)
            for k, v in self._ep_action_counts.items()
        }
        info = {
            "hunger": focal.hunger,
            "health": focal.health,
            "alive_count": sum(1 for a in self._agents if a.alive),
            "combat_prob": self.combat_prob,
            "ep_steps": self._ep_steps,
            "ep_action_counts": dict(self._ep_action_counts),
            "ep_action_rates": action_rates,
            "ep_hazard_approaches": dict(self._ep_hazard_approaches),
            "ep_hazard_flees": dict(self._ep_hazard_flees),
            "ep_sect_id": self._sect_config.sect_id if self._sect_config is not None else "none",
        }
        if terminated:
            info["ep_lifespan"] = self._ep_steps
            info["ep_avg_strength"] = (
                self._ep_focal_strength_sum / self._ep_steps if self._ep_steps > 0 else 0.0
            )
            info["ep_avg_power"] = (
                self._ep_focal_power_sum / self._ep_steps if self._ep_steps > 0 else 0.0
            )
            info["ep_final_power"] = compute_power_score(focal)
            info["ep_agent_rewards"] = list(self._ep_agent_rewards)
            info["ep_agent_steps"] = list(self._ep_agent_steps)
            info["ep_agent_mean_reward"] = [
                r / max(1, s)
                for r, s in zip(self._ep_agent_rewards, self._ep_agent_steps)
            ]
            # Settlement metrics
            info["ep_stash_fill_rate"] = self._ep_items_deposited / max(1, self._ep_items_gathered)
            info["ep_stash_withdraw_rate"] = self._ep_items_withdrawn / max(1, self._ep_items_deposited)
            info["ep_avg_dist_from_stash"] = (
                self._ep_dist_from_stash_sum / self._ep_dist_from_stash_count
                if self._ep_dist_from_stash_count > 0 else 0.0
            )
            info["ep_revisit_entropy"] = self._compute_revisit_entropy()
            info["ep_group_persistence"] = (
                self._ep_group_member_ticks / self._ep_groups_formed
                if self._ep_groups_formed > 0 else 0.0
            )
            info["ep_deaths_by_age"] = self._ep_deaths_by_cause.get("age", 0)
            info["ep_deaths_by_cause"] = dict(self._ep_deaths_by_cause)
            info["ep_reproductions"] = self._ep_reproductions
        return obs, reward, terminated, False, info

    # ── Combat helpers ────────────────────────────────────────────────────────

    def _do_attack(self, attacker: Agent) -> tuple[float, bool]:
        """Attack the nearest agent within Chebyshev distance 1 (8 directions). Returns (damage_dealt, killed).

        If the attacker has group members flanking (adjacent to the target from any direction),
        each ally grants a GROUP_ATTACK_BONUS_PER_ALLY multiplicative damage bonus.
        """
        target = self._nearest_adjacent_agent(attacker)
        if target is None:
            return 0.0, False
        attacker_idx = self._agents.index(attacker)
        flanking_allies = self._adjacent_group_allies(attacker_idx, target)
        damage = self._combat_damage(attacker, target, is_defending=False)
        if flanking_allies:
            bonus = GROUP_ATTACK_BONUS_PER_ALLY * len(flanking_allies)
            damage = float(np.clip(damage * (1.0 + bonus), 0.0, COMBAT_MAX_DAMAGE))
        target.health = max(0.0, target.health - damage)
        target._check_death("combat")
        if not target.alive:
            self._drop_inventory(target)
            return damage, True
        return damage, False

    def _heuristic_combat_step(
        self, agent: Agent, focal: Agent, focal_defending: bool
    ) -> float:
        """Heuristic for one non-focal agent. Returns damage dealt to focal agent.

        Requires Chebyshev distance ≤ 1 (8 directions) for combat to engage.
        """
        ax, ay = agent.position
        fx, fy = focal.position
        adjacent = max(abs(ax - fx), abs(ay - fy)) <= 1
        agent_idx = self._agents.index(agent)

        if adjacent:
            # Same group: never attack; just forage
            focal_idx = self._focal_idx
            if self._get_group(agent_idx) is not None and self._get_group(agent_idx) == self._get_group(focal_idx):
                self._heuristic_step(agent)
                self._last_action_names[agent_idx] = "rest"
                self._last_action_details[agent_idx] = ""
                return 0.0

            # Attack focal if adjacent, focal appears weaker, and agent is not very social
            if agent.strength > focal.strength * 1.1 and focal.health > 0 and agent.sociability < HEURISTIC_COLLAB_THRESHOLD:
                damage = self._combat_damage(agent, focal, is_defending=focal_defending)
                focal.health = max(0.0, focal.health - damage)
                focal._check_death("combat")
                if not focal.alive:
                    self._drop_inventory(focal)
                self._last_action_names[agent_idx] = "attack"
                self._last_action_details[agent_idx] = focal.agent_id
                return damage

        # Otherwise forage
        self._heuristic_step(agent)
        self._last_action_names[agent_idx] = "gather"
        self._last_action_details[agent_idx] = ""
        return 0.0

    def _combat_damage(
        self, attacker: Agent, defender: Agent, is_defending: bool
    ) -> float:
        """Compute combat damage.

        Formula: attacker.effective_strength*0.3 − defender.defense_power*0.1*is_defending
        Using effective_strength for attack (hunger weakens offense) and defense_power
        for defence (resistances from cultivation count here).
        Clamped to [0, COMBAT_MAX_DAMAGE].
        """
        raw = (
            attacker.effective_strength * COMBAT_ATTACKER_SCALE
            - defender.defense_power * COMBAT_DEFENDER_SCALE * (1.0 if is_defending else 0.0)
        )
        return float(np.clip(raw, 0.0, COMBAT_MAX_DAMAGE))

    def _nearest_adjacent_agent(self, agent: Agent) -> Agent | None:
        """Return nearest live agent within Chebyshev distance 1 (8 directions), or None."""
        ax, ay = agent.position
        for other in self._agents:
            if other is agent or not other.alive:
                continue
            ox, oy = other.position
            if max(abs(ox - ax), abs(oy - ay)) <= 1:
                return other
        return None

    def _adjacent_group_allies(self, agent_idx: int, ref: Agent) -> list[int]:
        """Return indices of live group members of agent_idx within Chebyshev distance 1 of ref.

        Used for flanking: allies adjacent to the target (from any of 8 directions) grant +damage.
        """
        group = self._get_group(agent_idx)
        if group is None:
            return []
        rx, ry = ref.position
        allies: list[int] = []
        for ally_idx in group:
            if ally_idx == agent_idx:
                continue
            ally = self._agents[ally_idx]
            if not ally.alive:
                continue
            ax, ay = ally.position
            if max(abs(ax - rx), abs(ay - ry)) <= 1:
                allies.append(ally_idx)
        return allies

    def _try_collaborate(self, focal_idx: int) -> bool:
        """Attempt to form a group with the nearest adjacent agent.

        Succeeds if the neighbour's sociability meets the collaboration threshold.
        Returns True if a new group was formed.
        """
        focal = self._agents[focal_idx]
        neighbour = self._nearest_adjacent_agent(focal)
        if neighbour is None:
            return False
        neighbour_idx = self._agents.index(neighbour)
        # Already in the same group — collaboration already established
        if (self._get_group(focal_idx) is not None
                and self._get_group(focal_idx) == self._get_group(neighbour_idx)):
            return False
        # Form a group only if the neighbour is also social enough
        if neighbour.sociability >= HEURISTIC_COLLAB_THRESHOLD:
            self._form_group(focal_idx, neighbour_idx)
            return True
        return False

    def _walk_away(self, agent: Agent) -> None:
        """Move one step away from the nearest adjacent agent.

        Direction is the opposite of the vector toward that agent.
        No-op if no adjacent agent exists.
        """
        neighbour = self._nearest_adjacent_agent(agent)
        if neighbour is None:
            return
        ax, ay = agent.position
        nx, ny = neighbour.position
        dx = ax - nx  # direction away from neighbour
        dy = ay - ny
        # Normalise to unit step (prioritise larger axis; ties: prefer x)
        if abs(dx) >= abs(dy):
            agent.move(1 if dx > 0 else -1, 0, self._world.grid_size)
        else:
            agent.move(0, 1 if dy > 0 else -1, self._world.grid_size)

    def _compute_combat_reward(
        self,
        hunger_prev: float,
        health_prev: float,
        food_gathered: int,
        hazard_damage: float,
        agent: Agent,
        exploration_reward: float,
        damage_dealt: float,
        damage_taken: float,
        defeat_bonus: float,
        inv_food_prev: int = 0,
    ) -> float:
        """Phase 3c reward: Phase 3b reward + combat shaping."""
        reward = self._compute_reward(hunger_prev, health_prev, food_gathered, hazard_damage, agent, exploration_reward, inv_food_prev)
        reward += REWARD_DAMAGE_TAKEN_SCALE * damage_taken
        reward += defeat_bonus
        return reward
