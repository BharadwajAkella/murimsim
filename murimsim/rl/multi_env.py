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

Action space: Discrete(17) — N_ACTIONS_PHASE6_QI (v18: + ATTACK_QI, ATTACK_BURST)
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
from dataclasses import dataclass
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from murimsim.actions import (
    Action,
    MOVE_DELTAS,
    N_ACTIONS_PHASE2,
    N_ACTIONS_PHASE3,
    N_ACTIONS_PHASE5,
    N_ACTIONS_PHASE6,
    N_ACTIONS_PHASE6_QI,
    ATTACK_ACTIONS,
)
from murimsim.agent import Agent, inherit_value  # noqa: F401 (inherit_value re-exported for tests)
from murimsim.monster import (
    BOSS_LOOT_FOOD,
    BOSS_LOOT_MATERIALS,
    BOSS_LOOT_QI,
    Monster,
    MonsterRegistry,
)
from murimsim.stash import Stash, StashRegistry
from murimsim.world import World
from murimsim.rl.agent_events import AgentStepEvents, StepEventBuffer

# ── Observation layout constants ─────────────────────────────────────────────
OBS_VIEW_SIZE: int = 5

OBS_N_RESOURCE_CH: int = 4          # food, qi, materials, poison
OBS_N_AGENT_CH: int = 5             # agent_present, agent_health, agent_strength, agent_sociability, affinity
OBS_N_STASH_CH: int = 2             # my_stash, enemy_stash
OBS_CHANNEL_ORDER: list[str] = ["food", "qi", "materials", "poison"]

OBS_RESOURCE_GRID_SIZE: int = OBS_VIEW_SIZE * OBS_VIEW_SIZE * OBS_N_RESOURCE_CH  # 100
OBS_AGENT_GRID_SIZE: int = OBS_VIEW_SIZE * OBS_VIEW_SIZE * OBS_N_AGENT_CH        # 100
OBS_STASH_GRID_SIZE: int = OBS_VIEW_SIZE * OBS_VIEW_SIZE * OBS_N_STASH_CH        # 50
OBS_STATS_SIZE: int = 14  # health, hunger, inv_food, inv_poison, pr, pi, combat_exp, terrain_fam, reward_ema, sociability, in_group, strength, hunger_resistance, damage_taken_last_step
OBS_TOTAL_SIZE: int = OBS_RESOURCE_GRID_SIZE + OBS_AGENT_GRID_SIZE + OBS_STASH_GRID_SIZE + OBS_STATS_SIZE  # 264

# ── History signal constants ──────────────────────────────────────────────────
TERRAIN_FAM_SCALE: float = 200.0   # ticks_near_food / SCALE → [0, 1]
REWARD_EMA_ALPHA: float = 0.10
REWARD_EMA_SCALE: float = 0.5      # EMA normalised: 0 = −scale, 1 = +scale

# ── Affinity / pairwise interaction memory (v19 emergent allegiance) ──────────
# Per-directed-pair scalar with exponential decay. Stored raw and clamped to
# [-AFFINITY_NORM, +AFFINITY_NORM] on access. Decay applied lazily on read/write
# so we don't sweep the whole matrix each step.
AFFINITY_DECAY_PER_STEP: float = 0.999   # half-life ~693 steps (≈35% of 2000-step ep)
AFFINITY_NORM: float = 5.0               # divisor → clamp to [-1, 1] for obs/reward use
# Asymmetric event magnitudes — see store_memory rationale.
# Each event records two updates: actor→other and other→actor with different magnitudes.
AFFINITY_SHARE_RECIPIENT: float = 1.0    # B (recipient) → A (sharer): strong gratitude
AFFINITY_SHARE_SHARER: float = 0.3       # A → B: mild investment ("I helped them")
AFFINITY_ATTACK_VICTIM: float = -1.0     # B (victim) → A (attacker): strong hostility
AFFINITY_ATTACK_ATTACKER: float = -0.3   # A → B: mild commitment to enmity
AFFINITY_STEAL_VICTIM: float = -0.7
AFFINITY_STEAL_THIEF: float = -0.2
AFFINITY_FLANK_BOTH: float = 0.5         # symmetric — both chose to engage same target
# v19c: additional event sources to raise per-pair event rate above the decay floor.
AFFINITY_COLLAB_BOTH: float = 0.4        # symmetric — voluntarily formed a group together
AFFINITY_JOINT_KILL: float = 0.6         # symmetric — both contributed to a monster kill
AFFINITY_PROXIMITY_PER_STEP: float = 0.015  # tiny per-tick bond for being within radius
AFFINITY_PROXIMITY_RADIUS: int = 2       # Chebyshev radius for proximity bond
AFFINITY_PROXIMITY_TICK_EVERY: int = 5   # apply proximity sweep every N env steps

# Reward shaping (v19)
REWARD_MUTUAL_SHARE_BONUS: float = 0.03  # extra to focal-sharer when both sides have positive affinity
REWARD_FRIENDLY_FLANK_MAX_MULT: float = 1.0  # flanking bonus scaled by (1 + min(1, max(0, mean_affinity) * MULT))
PENALTY_BETRAYAL: float = -0.20          # extra penalty for attacking high-affinity target
AFFINITY_BETRAY_THRESHOLD: float = 0.5   # focal's affinity-toward-target above this → betrayal

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
# Low-health danger penalty: continuous signal that fires every step health is critically low.
# Scale by distance below gate so the closer to death, the louder the signal.
PENALTY_LOW_HEALTH_SCALE: float = 0.50             # per step, × (gate - health)
LOW_HEALTH_PENALTY_GATE: float = 0.35              # penalty fires below this health level
# Survival redirect: when health is critical AND food is in inventory, force EAT regardless
# of what the model chose. Overrides even TRAIN. Acts like a hard survival instinct.
CRITICAL_HEALTH_EAT_THRESHOLD: float = 0.25        # health below this → redirect to EAT if possible
# δ-reward for TRAIN action: incentivises training (strength delta × scale)
REWARD_TRAIN_STRENGTH_SCALE: float = 10.0
REWARD_RESISTANCE_GAIN_SCALE: float = 0.0   # v17: dropped from 5.0 — let immunity emerge from survival benefit alone

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

        # v19: per-directed-pair affinity scalar with lazy exp decay.
        # _affinity_raw[i][j] = (value_at_last_update, step_at_last_update).
        # i, j are agent indices. Initialised lazily on first interaction.
        self._affinity_raw: dict[int, dict[int, tuple[float, int]]] = {}

        # v19b: anonymous spawn clustering — agents start in N small clusters so
        # the focal has nearby neighbours from tick 0 (no identity tag, just
        # spatial seeding). Replaces the sect-region spawn that was lost in
        # commit 6b1d1f2 without re-introducing sect identity.
        self._spawn_cluster_count: int = 3
        self._spawn_cluster_radius: int = 4
        self._spawn_cluster_centers: list[tuple[int, int]] | None = None

        # P0.3: per-slot lifecycle tracking for IPPO hidden-state resets.
        # ``_life_ids[s]`` is the unique id of the agent currently occupying
        # slot s; it changes on rebirth so the trainer knows to reset the
        # recurrent state for that slot. ``_lifecycle_died_step`` and
        # ``_lifecycle_born_step`` are per-step boolean flags emitted in
        # ``info["lifecycle"]`` and cleared at the start of every step.
        # A slot can have both flags True in the same step (death + same-step
        # rebirth via ``_try_reproduce``).
        self._life_ids: list[int] = list(range(n_agents))
        self._next_life_id: int = n_agents
        self._lifecycle_died_step: list[bool] = [False] * n_agents
        self._lifecycle_born_step: list[bool] = [False] * n_agents

    # ── Gymnasium API ────────────────────────────────────────────────────────

    def _initial_position(self, idx: int, grid_size: int) -> tuple[int, int]:
        """Return the starting ``(x, y)`` position for agent *idx* at episode reset.

        Default behaviour: spawn agents in a small number of anonymous clusters
        (no identity tags — coalition formation must still emerge from behaviour).
        Cluster centres are sampled once per reset and reused so all agents in a
        cluster start within ``SPAWN_CLUSTER_RADIUS`` Chebyshev cells of the centre.
        Set ``self._spawn_cluster_count`` to 0 to fall back to uniform random spawn.
        """
        if getattr(self, "_spawn_cluster_count", 0) <= 0:
            return (int(self._rng.integers(0, grid_size)),
                    int(self._rng.integers(0, grid_size)))
        # Build cluster centres lazily on first agent of the reset.
        if not getattr(self, "_spawn_cluster_centers", None):
            self._spawn_cluster_centers = [
                (int(self._rng.integers(0, grid_size)),
                 int(self._rng.integers(0, grid_size)))
                for _ in range(self._spawn_cluster_count)
            ]
        cx, cy = self._spawn_cluster_centers[idx % len(self._spawn_cluster_centers)]
        r = self._spawn_cluster_radius
        x = int(np.clip(cx + self._rng.integers(-r, r + 1), 0, grid_size - 1))
        y = int(np.clip(cy + self._rng.integers(-r, r + 1), 0, grid_size - 1))
        return (x, y)

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        effective_seed = seed if seed is not None else self._seed

        self._rng = np.random.default_rng(effective_seed)
        # P0.1: reset per-step curriculum gate cache.
        self._cached_curriculum_attack_allowed = None

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
        # v19b: clear cluster centres so they're freshly resampled this episode.
        self._spawn_cluster_centers = None
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
        self._damage_taken_last_step: list[float] = [0.0] * self._n_agents
        self._groups = []

        # Reciprocity memory: _help_received[recipient][helper] = step_when_helped
        # Cleared each episode so past-life debts don't carry over.
        self._help_received: dict[int, dict[int, int]] = {}
        # v19: pairwise affinity matrix — clear at episode reset so allegiances
        # don't leak across seeds.
        self._affinity_raw = {}
        # P0.3: re-init per-slot lifecycle tracking. Each slot gets a fresh
        # life id so the trainer treats episode start as a "born" boundary.
        self._life_ids = list(range(self._n_agents))
        self._next_life_id = self._n_agents
        self._lifecycle_died_step = [False] * self._n_agents
        self._lifecycle_born_step = [False] * self._n_agents
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
        # v17 instrumentation — perception of friendly vs enemy stash actions
        self._ep_pure_steals: int = 0           # steals from non-group-mate stash
        self._ep_friendly_steals: int = 0       # steals from group-mate stash
        self._ep_bank_withdrawals: int = 0      # WITHDRAW that retrieved from own stash
        self._ep_granary_withdrawals: int = 0   # WITHDRAW that retrieved from group-mate stash

        # v19 instrumentation — emergent allegiance signals
        self._ep_betrayal_count: int = 0          # focal attacked high-affinity target
        self._ep_friendly_flank_count: int = 0    # focal flanked alongside positive-affinity ally

        # Foraging-outward tracking: max Chebyshev dist from own stash since last deposit
        self._max_dist_since_deposit: list[float] = [0.0] * self._n_agents

        self._stash_registry.reset()

        return self._build_obs(self._focal_idx), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._world is not None

        # P0.3: clear per-step lifecycle flags. Populated by death detection
        # and _try_reproduce; emitted in info["lifecycle"] at end of step.
        self._lifecycle_died_step = [False] * self._n_agents
        self._lifecycle_born_step = [False] * self._n_agents

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
            for i, agent in enumerate(self._agents):
                was_alive = agent.alive
                agent.tick(self._max_age)
                if was_alive and not agent.alive:
                    cause = agent.death_cause or "unknown"
                    self._ep_deaths_by_cause[cause] = self._ep_deaths_by_cause.get(cause, 0) + 1
                    self._drop_inventory(agent)
                    # P0.3: flag the slot before reproduce — same-step rebirth
                    # will then also set the born flag (both True is valid).
                    self._lifecycle_died_step[i] = True
                    self._try_reproduce(agent)

        self._prune_dead_from_groups()

        # v19c: per-step proximity bond accrual.
        self._apply_proximity_bonds()

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
            # v17 perception/sociality metrics
            info["ep_pure_steals"] = self._ep_pure_steals
            info["ep_friendly_steals"] = self._ep_friendly_steals
            info["ep_bank_withdrawals"] = self._ep_bank_withdrawals
            info["ep_granary_withdrawals"] = self._ep_granary_withdrawals
        # P0.1: invalidate per-step curriculum cache so next step draws fresh.
        self._cached_curriculum_attack_allowed = None
        # P0.3: emit per-slot lifecycle metadata (IPPO hidden-state resets).
        info["lifecycle"] = self._build_lifecycle_info()
        return obs, reward, terminated, False, info

    def _build_lifecycle_info(self) -> list[dict[str, Any]]:
        """Per-slot lifecycle dicts for the just-completed step.

        Each entry: ``{"slot": idx, "life_id": int, "died": bool,
        "born": bool, "alive": bool}``. IPPO consumes this to (a) reset
        recurrent hidden state when ``born`` flips True (life_id changed)
        and (b) mask transitions from inactive slots in PPO loss.
        A slot can have both ``died`` and ``born`` True in the same step
        (rebirth via ``_try_reproduce``).
        """
        return [
            {
                "slot": i,
                "life_id": self._life_ids[i],
                "died": self._lifecycle_died_step[i],
                "born": self._lifecycle_born_step[i],
                "alive": self._agents[i].alive if i < len(self._agents) else False,
            }
            for i in range(self._n_agents)
        ]

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

    def _reset_slot_state(self, idx: int) -> None:
        """Wipe all slot-keyed runtime state for slot ``idx``.

        Called on rebirth (``_try_reproduce``) so the offspring inherits only
        the slot index and traits — never the deceased's social or shaping
        state. Without this, IPPO learns spurious cross-life correlations and
        the rotating-focal symmetry leaks bookkeeping between distinct lives.

        Cleared:
            * outgoing affinity row     (``_affinity_raw[idx]``)
            * incoming affinity column  (``_affinity_raw[other][idx]`` for all)
            * help-received (both directions in ``_help_received``)
            * reward EMA scalar
            * damage-taken-last-step scalar
            * group membership (slot is removed; group dissolves if size < 2)
        """
        # Drop the slot from any group BEFORE resurrection — _leave_group
        # handles the size<2 dissolution.
        self._leave_group(idx)

        # Outgoing affinity row.
        self._affinity_raw.pop(idx, None)
        # Incoming affinity column.
        for other_row in self._affinity_raw.values():
            other_row.pop(idx, None)

        # Help-received bookkeeping (both directions).
        self._help_received.pop(idx, None)
        for other_help in self._help_received.values():
            other_help.pop(idx, None)

        # Per-slot scalars.
        if 0 <= idx < len(self._reward_ema):
            self._reward_ema[idx] = 0.0
        if 0 <= idx < len(self._damage_taken_last_step):
            self._damage_taken_last_step[idx] = 0.0

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
        # P2.3: wipe all slot-keyed runtime state BEFORE installing the
        # offspring so the new life starts with a clean slate (no inherited
        # affinity, help-received, reward EMA, damage, or group membership).
        self._reset_slot_state(idx)
        self._agents[idx] = offspring
        self._ep_reproductions += 1
        # P0.3: record same-step rebirth for this slot. The slot transitions
        # alive→dead→born within one step; trainers must reset recurrent state.
        self._lifecycle_born_step[idx] = True
        self._life_ids[idx] = self._next_life_id
        self._next_life_id += 1

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

        # v19: directional affinity update (asymmetric — recipient learns more)
        self._record_affinity_event(
            actor_idx=sharer_idx, other_idx=recipient_idx,
            actor_to_other=AFFINITY_SHARE_SHARER,
            other_to_actor=AFFINITY_SHARE_RECIPIENT,
        )

        return True

    # ── Affinity (v19 emergent allegiance) ────────────────────────────────────

    def _record_affinity_event(
        self,
        actor_idx: int,
        other_idx: int,
        actor_to_other: float,
        other_to_actor: float,
    ) -> None:
        """Record a directed pairwise interaction event.

        ``actor_to_other`` is added to actor's affinity toward other; the
        symmetric ``other_to_actor`` is added to other's affinity toward
        actor. Magnitudes are typically asymmetric (e.g. attack victim
        builds more hostility than attacker).
        """
        if actor_idx == other_idx:
            return
        for i, j, delta in (
            (actor_idx, other_idx, actor_to_other),
            (other_idx, actor_idx, other_to_actor),
        ):
            row = self._affinity_raw.setdefault(i, {})
            cur_val, cur_step = row.get(j, (0.0, self._ep_step_count))
            # Decay existing value to current step before adding new event.
            decayed = cur_val * (AFFINITY_DECAY_PER_STEP ** max(0, self._ep_step_count - cur_step))
            new_val = decayed + delta
            row[j] = (new_val, self._ep_step_count)

    def _affinity(self, i: int, j: int) -> float:
        """Return i's current affinity toward j, normalised to [-1, 1].

        Uses lazy exponential decay: stored values are aged forward to the
        current episode step on read.
        """
        if i == j:
            return 0.0
        row = self._affinity_raw.get(i)
        if not row:
            return 0.0
        entry = row.get(j)
        if entry is None:
            return 0.0
        val, step = entry
        decayed = val * (AFFINITY_DECAY_PER_STEP ** max(0, self._ep_step_count - step))
        return float(np.clip(decayed / AFFINITY_NORM, -1.0, 1.0))

    def _record_joint_kill_bonds(self, attacker_ids: set[str] | list[str]) -> None:
        """Record symmetric joint-kill bonds between every pair of monster contributors.

        Called when a monster dies. Each unordered pair of attacker agent_ids that
        are still resolvable to live env agents gets a +AFFINITY_JOINT_KILL bond
        in both directions (shared-victory reciprocity).
        """
        if not attacker_ids:
            return
        # Resolve agent_ids → indices once.
        id_to_idx = {a.agent_id: i for i, a in enumerate(self._agents)}
        indices = [id_to_idx[aid] for aid in attacker_ids if aid in id_to_idx]
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                self._record_affinity_event(
                    actor_idx=indices[a], other_idx=indices[b],
                    actor_to_other=AFFINITY_JOINT_KILL,
                    other_to_actor=AFFINITY_JOINT_KILL,
                )

    def _apply_proximity_bonds(self) -> None:
        """Apply small symmetric +affinity to every pair of live agents within radius.

        Called every AFFINITY_PROXIMITY_TICK_EVERY env steps. O(N²) per tick;
        trivial for n_agents≤16. Steady-state target: with delta d per tick and
        decay λ per step, a continuously-adjacent pair settles at raw ≈
        d / (1 - λ^TICK_EVERY) — chosen so persistent neighbours can cross the
        betrayal/mutual-share thresholds without one-off events doing it alone.
        """
        if self._ep_step_count % AFFINITY_PROXIMITY_TICK_EVERY != 0:
            return
        r = AFFINITY_PROXIMITY_RADIUS
        n = self._n_agents
        for i in range(n):
            ai = self._agents[i]
            if not ai.alive:
                continue
            xi, yi = ai.position
            for j in range(i + 1, n):
                aj = self._agents[j]
                if not aj.alive:
                    continue
                xj, yj = aj.position
                if max(abs(xi - xj), abs(yi - yj)) <= r:
                    self._record_affinity_event(
                        actor_idx=i, other_idx=j,
                        actor_to_other=AFFINITY_PROXIMITY_PER_STEP,
                        other_to_actor=AFFINITY_PROXIMITY_PER_STEP,
                    )

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

        # Agent channels (agent_present, agent_health, agent_strength, agent_sociability, affinity)
        agent_window = np.zeros((OBS_VIEW_SIZE, OBS_VIEW_SIZE, OBS_N_AGENT_CH), dtype=np.float32)
        # Default affinity channel = 0.5 (neutral, mapped from affinity=0.0)
        agent_window[:, :, 4] = 0.5
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
                # v19: affinity to neighbour, mapped from [-1, 1] to [0, 1]; tile-MAX of strongest signal.
                aff = self._affinity(agent_idx, i)
                aff_mapped = (aff + 1.0) / 2.0
                # Take whichever value is further from neutral (0.5) — preserves dominant signal.
                cur = float(agent_window[wy, wx, 4])
                if abs(aff_mapped - 0.5) > abs(cur - 0.5):
                    agent_window[wy, wx, 4] = aff_mapped
        flat_agents = agent_window.reshape(-1)  # OBS_AGENT_GRID_SIZE

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
            min(1.0, self._damage_taken_last_step[agent_idx] / COMBAT_MAX_DAMAGE),
        ], dtype=np.float32)

        return np.concatenate([flat_resources, flat_agents, flat_stashes, stats])

    # ── Reward ───────────────────────────────────────────────────────────────

    def _compute_reward(
        self,
        hunger_prev: float,
        health_prev: float,
        food_gathered: int = 0,
        hazard_damage: float = 0.0,
        agent: Agent | None = None,
        exploration_reward: float = 0.0,
        inv_food_prev: int = 0,
        events: "AgentStepEvents | None" = None,
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

        P1.2 (IPPO migration prep):
        - Optional ``events`` kwarg: if supplied, ``food_gathered`` and
          ``hazard_damage`` are read from the events object instead of the
          positional args. All other positional callers behave identically
          (their args are passed through unchanged), guaranteeing byte-identity
          for the existing focal-only training path.
        """
        if events is not None:
            food_gathered = events.food_gathered
            hazard_damage = events.hazard_damage
        assert agent is not None, "_compute_reward requires an agent"
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
        # Low-health danger penalty: continuous signal every step health is critically low.
        # Scales with distance below gate — the closer to death, the louder the signal.
        if agent.health < LOW_HEALTH_PENALTY_GATE:
            reward += PENALTY_LOW_HEALTH_SCALE * (agent.health - LOW_HEALTH_PENALTY_GATE)
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
                items_deposited = stash.total()
                self._ep_items_deposited += items_deposited
                stash_bonus += REWARD_DEPOSIT_PER_ITEM * items_deposited
                if agent_idx >= 0 and self._max_dist_since_deposit[agent_idx] >= FORAGE_OUTWARD_MIN_DIST:
                    stash_bonus += REWARD_FORAGE_OUTWARD
                if agent_idx >= 0:
                    self._max_dist_since_deposit[agent_idx] = 0.0

        elif action_enum == Action.WITHDRAW:
            group = self._get_group(agent_idx) if agent_idx >= 0 else None
            if group:
                member_ids = [self._agents[i].agent_id for i in group]
                # Detect retrieval source: own (bank) or group-mate (granary)
                own_here = self._stash_registry.get_own_stash_at(agent.agent_id, *agent.position)
                own_food_here = sum(s.food for s in own_here)
                food_got = self._stash_registry.withdraw_group(agent, member_ids)
                self._ep_items_withdrawn += food_got
                if food_got > 0:
                    stash_bonus += REWARD_GROUP_WITHDRAW_BONUS
                    if food_got > own_food_here:
                        # At least some came from a group-mate's stash
                        self._ep_granary_withdrawals += 1
                    else:
                        self._ep_bank_withdrawals += 1
            else:
                at_pos = self._stash_registry.get_own_stash_at(agent.agent_id, *agent.position)
                items_here = sum(s.total() for s in at_pos)
                self._ep_items_withdrawn += items_here
                transferred = self._stash_registry.withdraw(agent)
                if transferred:
                    self._ep_bank_withdrawals += 1

        elif action_enum == Action.STEAL:
            stolen = self._stash_registry.steal(agent)
            if stolen is not None:
                is_friendly = False
                victim_idx = -1
                if agent_idx >= 0:
                    group = self._get_group(agent_idx)
                    if group:
                        is_friendly = any(
                            self._agents[i].agent_id == stolen.owner_id for i in group
                        )
                    # Find victim's agent index for affinity update
                    for i, a in enumerate(self._agents):
                        if a.agent_id == stolen.owner_id:
                            victim_idx = i
                            break
                if is_friendly:
                    self._ep_friendly_steals += 1
                else:
                    self._ep_pure_steals += 1
                # v19: directional affinity — victim resents thief strongly
                if agent_idx >= 0 and victim_idx >= 0:
                    self._record_affinity_event(
                        actor_idx=agent_idx, other_idx=victim_idx,
                        actor_to_other=AFFINITY_STEAL_THIEF,
                        other_to_actor=AFFINITY_STEAL_VICTIM,
                    )

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
          1. Eat if health is critically low and carrying food (survival override).
          2. Eat if hungry and carrying food.
          3. Gather if standing on a resource.
          4. Step one tile toward nearest food within HEURISTIC_SCAN_RADIUS.
          5. Random cardinal move.
        """
        # Survival override: critical health trumps everything else
        if agent.health < CRITICAL_HEALTH_EAT_THRESHOLD and agent.inventory.food > 0:
            agent.eat(self._resource_configs)
            return

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
COMBAT_ATTACKER_SCALE: float = 0.5   # base_damage = (√strength + tier.bonus) × scale  (v18)
# DEFEND is multiplicative: damage *= (1 − defender.defense_power)
# defense_power = effective_strength×0.5 + avg_resistance×0.5
# A master cultivator (defense_power→1.0) fully nullifies any attack
COMBAT_MAX_DAMAGE: float = 0.9        # v18: raised from 0.5 to leave headroom for qi-burst strikes ((1+0.7)·0.5 = 0.85)


@dataclass(frozen=True)
class StrikeTier:
    """v18 qi-infused strike tier. Picked by the policy via Action.ATTACK_*."""

    name: str           # human-readable label for logging / replay
    qi_cost: int        # qi consumed from attacker.inventory on a successful strike
    bonus: float        # additive term added to √strength BEFORE COMBAT_ATTACKER_SCALE


# Three discrete strike tiers. Bonuses are in √strength units (added before scale).
# Damage at strength=1.0:  basic 0.50, qi 0.65, burst 0.85.
# Damage at strength=0.0:  basic 0.00, qi 0.15, burst 0.35  ← qi/burst rescue weak agents.
STRIKE_BASIC: StrikeTier = StrikeTier(name="basic", qi_cost=0, bonus=0.0)
STRIKE_QI:    StrikeTier = StrikeTier(name="qi",    qi_cost=1, bonus=0.3)
STRIKE_BURST: StrikeTier = StrikeTier(name="burst", qi_cost=3, bonus=0.7)

# Map from Action enum to the strike tier the policy is requesting.
ACTION_TO_STRIKE: dict[Action, StrikeTier] = {
    Action.ATTACK:       STRIKE_BASIC,
    Action.ATTACK_QI:    STRIKE_QI,
    Action.ATTACK_BURST: STRIKE_BURST,
}

REWARD_DEFEAT_OPPONENT: float = 0.3
REWARD_DAMAGE_TAKEN_SCALE: float = -0.2
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

# v21a: boss respawn — after the boss dies, wait this many ticks then spawn a new
# boss at a random corner. Without respawn, the boss is a one-shot event per
# episode and provides no sustained learning signal across long rollouts.
BOSS_RESPAWN_DELAY: int = 60

# Shared stash rewards
# Foraging-outward: deposit after having been >=N tiles away from stash since last deposit
FORAGE_OUTWARD_MIN_DIST: int = 5      # Chebyshev tiles away from stash to qualify
REWARD_FORAGE_OUTWARD: float = 0.03   # bonus for depositing after a foraging excursion
REWARD_DEPOSIT_PER_ITEM: float = 0.05 # v18: reverted from v17's 0.02 (deposits collapsed to 0). Stash needs to be a real choice, not a vestigial behavior.
# Group withdrawal bonus: reward when agent withdraws from a group stash while hungry
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
    - Combat (v18): ``damage = √attacker.effective_strength × 0.5`` then × (1 − defender.defense_power) if defending
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
        enable_boss: bool = False,
    ) -> None:
        super().__init__(
            config=config,
            n_agents=n_agents,
            seed=seed,
            render_mode=render_mode,
            n_actions=N_ACTIONS_PHASE6_QI,
        )
        self._curriculum_ramp_steps: int = curriculum_ramp_steps
        self._global_step_count: int = 0  # persists across episodes
        # Per-agent last action detail — target agent id, stash id, or "" — reset each step
        self._last_action_details: list[str] = [""] * n_agents
        # Per-agent last action name (tracks heuristic agents too, unlike record_combat.py)
        self._last_action_names: list[str] = ["rest"] * n_agents
        # Optional per-agent action overrides for full-team inference (replay only).
        # Keys are agent indices (non-focal); values are Action int values.
        # When set, _heuristic_combat_step is bypassed in favour of the policy action.
        self._action_overrides: dict[int, int] | None = None
        # v17: boss monster — common-enemy emergence pressure. Disabled by default
        # so existing tests/training runs don't change behaviour silently.
        self._enable_boss: bool = enable_boss
        self._monsters: MonsterRegistry = MonsterRegistry()
        # v17: combat-focus lock — when an agent is in combat (hostile adjacent
        # OR took damage last tick), mask out non-combat actions like TRAIN,
        # REST, GATHER, DEPOSIT, COLLABORATE. Prevents the immersion-breaking
        # "agent meditates while being hit by the boss" replay frame. Cooldown
        # of 1 tick after taking damage so a fleeing boss can't be insta-trained.
        self._enable_combat_focus: bool = True
        self._in_combat_cooldown: list[int] = [0] * n_agents
        # P0.1 (IPPO migration prep): per-step cached curriculum gate for ATTACK/DEFEND.
        # None = no draw yet for the current step. Set by _curriculum_attack_allowed()
        # on first access (action_masks or step), invalidated at end of step. Same draw
        # used by both mask construction and action redirect, so they always agree.
        # Removes the previous RNG side-effect from action_masks(), which would otherwise
        # be called per-(env, agent) every rollout step in IPPO and scramble env RNG.
        self._cached_curriculum_attack_allowed: bool | None = None

    # ── Curriculum ────────────────────────────────────────────────────────────

    @property
    def combat_prob(self) -> float:
        """Current probability that a combat action is allowed (not masked to REST)."""
        frac = min(1.0, self._global_step_count / max(1, self._curriculum_ramp_steps))
        return CURRICULUM_START_PROB + (CURRICULUM_END_PROB - CURRICULUM_START_PROB) * frac

    def _curriculum_attack_allowed(self) -> bool:
        """Pure per-step curriculum gate. Draws once per step on first access, caches.

        Repeated calls within the same step return the same value without consuming
        RNG. The cache is invalidated at the end of every ``step()`` so the next step
        draws fresh against the updated ``combat_prob``. Tests that manipulate
        ``_global_step_count`` directly will see the gate re-drawn on the next call
        because the cache is always invalidated post-step.
        """
        if self._cached_curriculum_attack_allowed is None:
            self._cached_curriculum_attack_allowed = (
                self._rng.random() <= self.combat_prob
            )
        return self._cached_curriculum_attack_allowed

    # ── reset override ────────────────────────────────────────────────────────
    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment."""
        obs, info = super().reset(seed=seed, options=options)
        # v17 instrumentation — flee context (CombatEnv-only since WALK_AWAY is a combat action)
        self._ep_walk_away_count: int = 0
        self._ep_flee_strength_diff_sum: float = 0.0
        self._ep_flee_health_sum: float = 0.0
        # v17 boss monster — common-enemy survival pressure
        self._monsters.reset()
        self._ep_boss_killed: int = 0
        self._ep_boss_damage_dealt: float = 0.0
        self._ep_boss_attacks_landed: int = 0
        self._ep_damage_from_boss: float = 0.0
        self._ep_agents_killed_by_boss: int = 0
        # v21a: boss respawn countdown — -1 means no respawn pending (boss alive
        # or never enabled). Set to BOSS_RESPAWN_DELAY when boss dies; decremented
        # each tick; spawn fires when it reaches 0.
        self._boss_respawn_countdown: int = -1
        # v18: qi-infused strike telemetry
        self._ep_qi_strikes_used: int = 0      # successful ATTACK_QI strikes (1 qi spent)
        self._ep_burst_strikes_used: int = 0   # successful ATTACK_BURST strikes (3 qi spent)
        self._ep_qi_spent_in_combat: int = 0   # total qi consumed by all strike actions
        # v17 combat-focus: reset shaken-cooldown for all agents
        self._in_combat_cooldown = [0] * self._n_agents
        if self._enable_boss:
            corner = self._random_corner()
            self._monsters.spawn_boss(corner)
        # Rebuild obs so first frame reflects the boss spawn
        obs = self._build_obs(self._focal_idx)
        return obs, info

    def _build_obs(self, agent_idx: int) -> np.ndarray:
        """Build observation; overlay live monsters into the agent channel.

        Monsters are encoded into the same 5×5 agent window using channels:
        0=present, 1=health (normalised), 2=strength, 3=sociability.
        Monsters get sociability=0 and strength as raw value (typically ~1.6
        for the boss vs ~0.5–1.0 for agents — discriminable signal).
        """
        obs = super()._build_obs(agent_idx)
        if not self._monsters.all_alive():
            return obs
        agent = self._agents[agent_idx]
        ax, ay = agent.position
        half = OBS_VIEW_SIZE // 2
        # Agent grid sits at flat index 100..200, shape (5, 5, 4)
        agent_window = obs[OBS_RESOURCE_GRID_SIZE:
                           OBS_RESOURCE_GRID_SIZE + OBS_AGENT_GRID_SIZE].reshape(
            OBS_VIEW_SIZE, OBS_VIEW_SIZE, OBS_N_AGENT_CH
        )
        for monster in self._monsters.all_alive():
            mx, my = monster.position
            wx = (mx - ax) + half
            wy = (my - ay) + half
            if 0 <= wx < OBS_VIEW_SIZE and 0 <= wy < OBS_VIEW_SIZE:
                health_norm = monster.health / max(1e-6, monster.max_health)
                agent_window[wy, wx, 0] = max(agent_window[wy, wx, 0], 1.0)
                agent_window[wy, wx, 1] = max(agent_window[wy, wx, 1], health_norm)
                agent_window[wy, wx, 2] = max(agent_window[wy, wx, 2], monster.strength)
                # sociability stays 0 — monsters can't collaborate
                # affinity stays 0.5 (neutral) — no relationship with monsters
        # Writing back is unnecessary because reshape is a view, but be explicit
        obs[OBS_RESOURCE_GRID_SIZE:
            OBS_RESOURCE_GRID_SIZE + OBS_AGENT_GRID_SIZE] = agent_window.reshape(-1)
        return obs

    def _random_corner(self) -> tuple[int, int]:
        """Pick one of the four grid corners deterministically from the env RNG."""
        gs = self._world.grid_size  # type: ignore[union-attr]
        corners = [(0, 0), (gs - 1, 0), (0, gs - 1), (gs - 1, gs - 1)]
        idx = int(self._rng.integers(0, len(corners)))
        return corners[idx]

    # ── step override ─────────────────────────────────────────────────────────

    def _withdraw_target_available(self, agent: Agent) -> bool:
        """Return True if WITHDRAW would actually retrieve at least one item.

        Checks for a non-empty stash at ``agent.position`` owned by the agent
        itself or, if the agent is in a group, by any group-mate. Just being
        in a group is not enough — the group might hold no stash at this tile.
        """
        own = self._stash_registry.get_own_stash_at(agent.agent_id, *agent.position)
        if any(s.total() > 0 for s in own):
            return True
        # Agent index lookup safe for focal agent path; falls back to no group.
        try:
            agent_idx = self._agents.index(agent)
        except ValueError:
            return False
        group = self._get_group(agent_idx)
        if not group:
            return False
        for member_idx in group:
            mid = self._agents[member_idx].agent_id
            if mid == agent.agent_id:
                continue
            mate_stashes = self._stash_registry.get_own_stash_at(mid, *agent.position)
            if any(s.total() > 0 for s in mate_stashes):
                return True
        return False

    def _smart_fallback(self, agent: Agent) -> Action:
        """Return the best productive action given the agent's current state.

        Priority:
          1. EAT — if hungry and carrying food
          2. GATHER — if standing on food tile
          3. TRAIN — default cultivation (always positive expected reward)
          4. MOVE toward nearest food — if starving and food is visible
        """
        if agent.hunger > HEURISTIC_HUNGER_EAT and agent.inventory.food > 0:
            return Action.EAT
        x, y = agent.position
        food_grid = self._world.get_grid_view("food")
        if food_grid[y, x] > 0:
            return Action.GATHER
        if agent.hunger > STARVATION_THRESHOLD:
            target = self._nearest_food(agent.position)
            if target is not None:
                tx, ty = target
                dx = int(np.sign(tx - x))
                dy = int(np.sign(ty - y))
                if dx > 0: return Action.MOVE_E
                if dx < 0: return Action.MOVE_W
                if dy > 0: return Action.MOVE_S
                return Action.MOVE_N
        return Action.TRAIN

    def _combat_focus_fallback(self, agent: Agent) -> Action:
        """Redirect non-combat action while in combat → sensible combat default.

        v17. Priority:
          1. EAT — if low health and carrying food (defensive heal)
          2. ATTACK — if any enemy adjacent (boss prioritised in _do_attack)
          3. DEFEND — fall back to bracing
        """
        if agent.health < CRITICAL_HEALTH_EAT_THRESHOLD and agent.inventory.food > 0:
            return Action.EAT
        ax, ay = agent.position
        # v18: when in combat and qi is available, prefer the highest-affordable
        # qi-infused tier — burst > qi > basic. Encourages spending hoarded qi
        # offensively rather than letting it sit in inventory.
        if self._monsters.get_adjacent_to(ax, ay):
            if agent.inventory.qi >= STRIKE_BURST.qi_cost:
                return Action.ATTACK_BURST
            if agent.inventory.qi >= STRIKE_QI.qi_cost:
                return Action.ATTACK_QI
            return Action.ATTACK
        if self._nearest_adjacent_agent(agent) is not None:
            if agent.inventory.qi >= STRIKE_BURST.qi_cost:
                return Action.ATTACK_BURST
            if agent.inventory.qi >= STRIKE_QI.qi_cost:
                return Action.ATTACK_QI
            return Action.ATTACK
        return Action.DEFEND

    def _redirect_invalid_action(
        self, agent: Agent, action: Action, agent_idx: int | None = None
    ) -> Action:
        """Redirect context-invalid actions to a productive fallback.

        Two redirect types:
        1. Survival override: health critically low + food in inventory → force EAT.
           Fires before any other check — survival beats cultivation.
        2. Invalid-action redirect: if the chosen action has no valid target or
           precondition, redirect to the best productive action via _smart_fallback()
           instead of wasting the turn on REST.

        Note: the REAL fix is action masking (action_masks() method below) — that
        prevents the model from picking infeasible actions in the first place.
        These redirects are a safety net for inference and early training.
        """
        # 1. Survival override — hardwired instinct: eat when near death
        if (
            agent.health < CRITICAL_HEALTH_EAT_THRESHOLD
            and agent.inventory.food > 0
            and action != Action.EAT
        ):
            return Action.EAT

        # 1b. v17 combat-focus override: in combat, redirect non-combat actions
        # to a sensible combat fallback. Survival EAT (above) still wins.
        # Disabled if _enable_combat_focus is False or no hostile near.
        if action in (Action.TRAIN, Action.REST, Action.GATHER, Action.DEPOSIT, Action.COLLABORATE):
            # P0.2: prefer caller-supplied agent_idx; fall back to focal for back-compat.
            idx = agent_idx if agent_idx is not None else self._focal_idx
            if self._in_combat(idx):
                return self._combat_focus_fallback(agent)

        # 2. Invalid-action redirects → smart fallback (not REST)
        if action in ATTACK_ACTIONS or action in (Action.COLLABORATE, Action.WALK_AWAY):
            no_adjacent_agent = self._nearest_adjacent_agent(agent) is None
            # ATTACK* is also valid against an adjacent monster (boss).
            # COLLABORATE/WALK_AWAY remain agent-only.
            if action in ATTACK_ACTIONS:
                ax, ay = agent.position
                if no_adjacent_agent and not self._monsters.get_adjacent_to(ax, ay):
                    return self._smart_fallback(agent)
            elif no_adjacent_agent:
                return self._smart_fallback(agent)
        elif action == Action.EAT:
            if agent.inventory.food == 0:
                return self._smart_fallback(agent)
        elif action == Action.DEPOSIT:
            if agent.inventory.food == 0:
                return self._smart_fallback(agent)
            if self._stash_registry.is_stash_full(
                agent.agent_id, *agent.position, incoming=agent.inventory.total()
            ):
                return self._smart_fallback(agent)
        elif action == Action.WITHDRAW:
            if not self._withdraw_target_available(agent):
                return self._smart_fallback(agent)
        elif action == Action.STEAL:
            enemy = self._stash_registry.get_enemy_stashes_at(agent.agent_id, *agent.position)
            if not enemy:
                return self._smart_fallback(agent)
        return action

    def action_masks(self, agent_idx: int | None = None) -> np.ndarray:
        """Return a boolean mask of currently valid actions for an agent.

        Defaults to the focal agent for backward compatibility with the
        single-agent SB3 training loop. P0.2 (IPPO prep): pass agent_idx
        to query masks for any agent in the same env without mutating focal.

        True = action is feasible right now.  Used by MaskableRecurrentPPO (v17+)
        to zero out infeasible logits before sampling, so the model never needs to
        learn 'don't pick ATTACK when nobody is adjacent' — it simply can't.

        This is the correct long-term fix. _redirect_invalid_action() is a safety
        net for RecurrentPPO which doesn't support masking natively.
        """
        idx = agent_idx if agent_idx is not None else self._focal_idx
        agent = self._agents[idx]
        mask = np.ones(self.action_space.n, dtype=bool)

        # Social actions require an adjacent agent — except ATTACK, which is
        # also valid against an adjacent boss-monster (v17).
        no_adjacent_agent = self._nearest_adjacent_agent(agent) is None
        ax, ay = agent.position
        no_adjacent_monster = not self._monsters.get_adjacent_to(ax, ay)
        if no_adjacent_agent and no_adjacent_monster:
            mask[Action.ATTACK] = False
            mask[Action.ATTACK_QI] = False
            mask[Action.ATTACK_BURST] = False
        # v18: qi-spend tiers also require sufficient qi in inventory
        if agent.inventory.qi < STRIKE_QI.qi_cost:
            mask[Action.ATTACK_QI] = False
        if agent.inventory.qi < STRIKE_BURST.qi_cost:
            mask[Action.ATTACK_BURST] = False
        if no_adjacent_agent:
            mask[Action.COLLABORATE] = False
            mask[Action.WALK_AWAY] = False

        # EAT / DEPOSIT require food in hand
        if agent.inventory.food == 0:
            mask[Action.EAT] = False
            mask[Action.DEPOSIT] = False

        # DEPOSIT also blocked when own stash at current position is full
        if agent.inventory.food > 0 and self._stash_registry.is_stash_full(
            agent.agent_id, *agent.position, incoming=agent.inventory.total()
        ):
            mask[Action.DEPOSIT] = False

        # WITHDRAW requires an actual stash with food/items at this position —
        # either own stash here, or any group-mate's stash here. Just being in
        # a group is not enough: the group might have no stashes here.
        if not self._withdraw_target_available(agent):
            mask[Action.WITHDRAW] = False

        # STEAL requires an enemy stash at current position
        if not self._stash_registry.get_enemy_stashes_at(agent.agent_id, *agent.position):
            mask[Action.STEAL] = False

        # Curriculum: mask combat actions when not yet enabled.
        # Uses cached per-step gate (P0.1) — pure with respect to RNG.
        if not self._curriculum_attack_allowed():
            mask[Action.ATTACK] = False
            mask[Action.ATTACK_QI] = False
            mask[Action.ATTACK_BURST] = False
            mask[Action.DEFEND] = False

        # v17 combat-focus: when in combat, mask out non-combat actions.
        # Allowed in combat: MOVE_*, EAT (survival), ATTACK, DEFEND, WALK_AWAY,
        # WITHDRAW, STEAL. Blocked: TRAIN, REST, GATHER, DEPOSIT, COLLABORATE.
        # MOVE actions are never masked, so the mask remains non-empty without
        # any unconditional "force TRAIN=True" guarantee.
        if self._in_combat(idx):
            mask[Action.TRAIN] = False
            mask[Action.REST] = False
            mask[Action.GATHER] = False
            mask[Action.DEPOSIT] = False
            mask[Action.COLLABORATE] = False
        return mask

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._world is not None
        self._global_step_count += 1

        # P0.3: clear per-step lifecycle flags before death detection runs.
        self._lifecycle_died_step = [False] * self._n_agents
        self._lifecycle_born_step = [False] * self._n_agents

        # P1.2: per-step event buffer + per-agent pre-state snapshot.
        # Snapshot BEFORE any action is applied so end-of-step deltas are correct.
        # Capture the acting focal index — it advances at end-of-step, so we
        # need to remember which slot actually performed the focal action.
        acting_focal_idx = self._focal_idx
        step_events = StepEventBuffer(self._n_agents)
        agent_pre_state = [
            {
                "hunger": ag.hunger,
                "health": ag.health,
                "inv_food": ag.inventory.food,
                "alive": ag.alive,
            }
            for ag in self._agents
        ]
        # Track per-non-focal damage dealt this step (heuristic / override path).
        heuristic_damage_dealt = [0.0] * self._n_agents

        # Curriculum: redirect combat actions to TRAIN when not yet fully enabled.
        # TRAIN is always productive — better signal than REST. Uses the same cached
        # gate as action_masks() so the two never disagree within a single step.
        action_enum = Action(action)
        if action_enum in (Action.ATTACK, Action.ATTACK_QI, Action.ATTACK_BURST, Action.DEFEND):
            if not self._curriculum_attack_allowed():
                action_enum = Action.TRAIN

        focal = self._agents[self._focal_idx]

        # Redirect actions that have no valid target/precondition → REST
        action_enum = self._redirect_invalid_action(focal, action_enum, agent_idx=self._focal_idx)
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

        # Snapshot hazard distances before the action for approach/flee tracking
        prev_pos = focal.position
        pre_hazard_dists = {h: self._nearest_hazard_dist(prev_pos, h) for h in self._ep_hazard_approaches}

        # Reset per-step damage tracking — populated by ATTACK and heuristic actions below
        self._damage_taken_last_step = [0.0] * self._n_agents

        # 1. Apply focal agent's action
        flanking_bonus_earned = False
        # v19 emergent allegiance trackers (per step)
        focal_betrayal: bool = False           # focal attacked a high-affinity target
        focal_flank_affinity: float = 0.0      # mean focal→flanker affinity, ∈ [0, 1]
        self._last_action_details[self._focal_idx] = ""
        self._last_action_names[self._focal_idx] = action_enum.name.lower()
        if action_enum in ATTACK_ACTIONS:
            requested_tier = ACTION_TO_STRIKE[action_enum]
            # Check for flanking allies before the attack (target may die during it)
            pre_target = self._nearest_adjacent_agent(focal)
            if pre_target is not None:
                flankers = self._adjacent_group_allies(self._focal_idx, pre_target)
                flanking_bonus_earned = len(flankers) > 0
                self._last_action_details[self._focal_idx] = pre_target.agent_id
                # v19: scale flank bonus by focal's affinity to flankers (clip to non-negative).
                if flankers:
                    target_idx_for_aff = self._agents.index(pre_target)
                    affs = [max(0.0, self._affinity(self._focal_idx, fi)) for fi in flankers]
                    focal_flank_affinity = float(sum(affs) / len(affs))
                    # Betrayal: focal attacking a target it has positive history with.
                    if self._affinity(self._focal_idx, target_idx_for_aff) >= AFFINITY_BETRAY_THRESHOLD:
                        focal_betrayal = True
                else:
                    target_idx_for_aff = self._agents.index(pre_target)
                    if self._affinity(self._focal_idx, target_idx_for_aff) >= AFFINITY_BETRAY_THRESHOLD:
                        focal_betrayal = True
            else:
                self._last_action_details[self._focal_idx] = "no_target"
            damage_dealt, defeated = self._do_attack(focal, requested_tier=requested_tier)
            # Track damage the target just took so its next obs reflects being hit
            if damage_dealt > 0 and pre_target is not None:
                target_idx = self._agents.index(pre_target)
                self._damage_taken_last_step[target_idx] = damage_dealt
            defeat_bonus = REWARD_DEFEAT_OPPONENT if defeated else 0.0
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
                # v17: log flee context — strength differential and health
                self._ep_walk_away_count += 1
                self._ep_flee_strength_diff_sum += float(focal.strength - neighbour.strength)
                self._ep_flee_health_sum += float(focal.health)
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

        # 2. Heuristic (or policy override) for non-focal agents
        damage_taken = 0.0
        for i, agent in enumerate(self._agents):
            if i != self._focal_idx and agent.alive:
                if self._action_overrides is not None and i in self._action_overrides:
                    dmg = self._execute_override_action(agent, i, Action(self._action_overrides[i]), focal, focal_defending)
                else:
                    dmg = self._heuristic_combat_step(agent, focal, focal_defending)
                damage_taken += dmg
                # P1.2: attribute damage_dealt to the non-focal slot.
                if dmg > 0:
                    heuristic_damage_dealt[i] += dmg

        # 2b. v17: monster tick (after agent actions). Boss may attack any
        # adjacent live agent; we bookkeep damage to focal so its obs reflects
        # the hit, and per-episode totals for telemetry.
        monster_events = self._monsters.tick_all(self._world, self._agents, self._rng)
        for _mid, victim_id, dmg in monster_events:
            self._ep_damage_from_boss += dmg
            for vi, va in enumerate(self._agents):
                if va.agent_id == victim_id:
                    self._damage_taken_last_step[vi] += dmg
                    if vi == self._focal_idx:
                        damage_taken += dmg
                    if not va.alive:
                        self._ep_agents_killed_by_boss += 1
                    break

        # v21a: boss respawn — if enabled and no live boss, run countdown and
        # spawn a fresh boss at a random corner when it reaches 0. Provides
        # sustained boss pressure across long IPPO rollouts.
        if self._enable_boss:
            no_live_boss = not any(
                m.alive for m in self._monsters.all() if m.kind == "boss"
            )
            if no_live_boss:
                if self._boss_respawn_countdown < 0:
                    self._boss_respawn_countdown = BOSS_RESPAWN_DELAY
                elif self._boss_respawn_countdown == 0:
                    corner = self._random_corner()
                    self._monsters.spawn_boss(corner)
                    self._boss_respawn_countdown = -1
                else:
                    self._boss_respawn_countdown -= 1
            else:
                self._boss_respawn_countdown = -1

        # Record damage focal took this step (from all heuristic agents combined)
        self._damage_taken_last_step[self._focal_idx] = damage_taken

        # v17 combat-focus cooldown: decrement existing cooldowns; arm 1-tick
        # shaken cooldown for any agent that took damage this step. This keeps
        # the combat-lock active for one tick after disengagement, preventing
        # the immersion-breaking "boss steps away → instant TRAIN" frame.
        for i in range(self._n_agents):
            if self._in_combat_cooldown[i] > 0:
                self._in_combat_cooldown[i] -= 1
            if self._damage_taken_last_step[i] > 0:
                self._in_combat_cooldown[i] = 1

        # 3. Advance world + all agents
        for _ in range(self._action_ticks):
            self._world.step()
            for i, agent in enumerate(self._agents):
                was_alive = agent.alive
                agent.tick(self._max_age)
                if was_alive and not agent.alive:
                    cause = agent.death_cause or "unknown"
                    self._ep_deaths_by_cause[cause] = self._ep_deaths_by_cause.get(cause, 0) + 1
                    self._drop_inventory(agent)
                    # P0.3: flag slot died — _try_reproduce may also flag born (rebirth same step).
                    self._lifecycle_died_step[i] = True
                    self._try_reproduce(agent)

        # Remove dead agents from any groups
        self._prune_dead_from_groups()

        # v19c: per-step proximity bond accrual.
        self._apply_proximity_bonds()

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
        # v19: focal-as-recipient bonus removed (rubber-duck critique #2 — taught
        # "be near scripted sharers"). Only reward focal-as-sharer, with extra for
        # mutual positive affinity.
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
                # v19: capture pre-share affinity so the bonus reflects EXISTING
                # reciprocity rather than the affinity bump from this very share.
                pre_share_affinity = self._affinity(recipient_idx, focal_idx) if sharer_idx == focal_idx else 0.0
                if self._try_food_share(sharer_idx, recipient_idx):
                    if sharer_idx == focal_idx:
                        food_share_reward += REWARD_FOOD_SHARE
                        if pre_share_affinity > 0.0:
                            food_share_reward += REWARD_MUTUAL_SHARE_BONUS

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
        # Coordinated attack: bonus when focal attacks with a flanking group ally.
        # v19: scaled by mean focal→flanker affinity — coordination with familiar
        # allies pays more than coordination with strangers.
        if flanking_bonus_earned and focal.alive:
            scale = 1.0 + REWARD_FRIENDLY_FLANK_MAX_MULT * focal_flank_affinity
            reward += REWARD_COORDINATED_ATTACK * scale
            if focal_flank_affinity > 0.0:
                self._ep_friendly_flank_count += 1
        # v19: betrayal penalty — focal attacked someone it had positive history with.
        if focal_betrayal:
            reward += PENALTY_BETRAYAL
            self._ep_betrayal_count += 1
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
            # v17 perception/sociality metrics
            info["ep_pure_steals"] = self._ep_pure_steals
            info["ep_friendly_steals"] = self._ep_friendly_steals
            info["ep_bank_withdrawals"] = self._ep_bank_withdrawals
            info["ep_granary_withdrawals"] = self._ep_granary_withdrawals
            info["ep_walk_away_count"] = self._ep_walk_away_count
            info["ep_avg_flee_strength_diff"] = (
                self._ep_flee_strength_diff_sum / self._ep_walk_away_count
                if self._ep_walk_away_count > 0 else 0.0
            )
            info["ep_avg_flee_health"] = (
                self._ep_flee_health_sum / self._ep_walk_away_count
                if self._ep_walk_away_count > 0 else 0.0
            )
            info["ep_focal_collaborate_count"] = self._ep_action_counts.get("collaborate", 0)
            # v17 boss monster metrics
            info["ep_boss_killed"] = self._ep_boss_killed
            info["ep_boss_attacks_landed"] = self._ep_boss_attacks_landed
            info["ep_boss_damage_dealt"] = self._ep_boss_damage_dealt
            info["ep_qi_strikes_used"] = self._ep_qi_strikes_used
            info["ep_burst_strikes_used"] = self._ep_burst_strikes_used
            info["ep_qi_spent_in_combat"] = self._ep_qi_spent_in_combat
            info["ep_damage_from_boss"] = self._ep_damage_from_boss
            info["ep_agents_killed_by_boss"] = self._ep_agents_killed_by_boss
            info["ep_boss_unique_attackers"] = sum(
                len(m.attackers) for m in self._monsters.all() if m.kind == "boss"
            )
            # v19 emergent allegiance signals
            info["ep_betrayal_count"] = self._ep_betrayal_count
            info["ep_friendly_flank_count"] = self._ep_friendly_flank_count
            # Sample of focal's strongest current relationships (max abs affinity to any other agent)
            focal_row = self._affinity_raw.get(self._focal_idx, {})
            if focal_row:
                affs = [self._affinity(self._focal_idx, j) for j in focal_row.keys()]
                info["ep_focal_max_affinity"] = max(affs) if affs else 0.0
                info["ep_focal_min_affinity"] = min(affs) if affs else 0.0
            else:
                info["ep_focal_max_affinity"] = 0.0
                info["ep_focal_min_affinity"] = 0.0
        # P0.1: invalidate per-step curriculum cache so next step draws fresh.
        self._cached_curriculum_attack_allowed = None
        # P0.3: emit per-slot lifecycle metadata (IPPO hidden-state resets).
        info["lifecycle"] = self._build_lifecycle_info()

        # P1.2: populate per-agent events + per-agent reward.
        # Use acting_focal_idx (captured before _focal_idx advanced) so events
        # are attributed to the slot that actually performed the focal action.
        focal_ev = step_events[acting_focal_idx]
        focal_ev.food_gathered = int(food_gathered)
        focal_ev.hazard_damage = float(hazard_damage)
        focal_ev.damage_dealt = float(damage_dealt)
        focal_ev.damage_taken = float(damage_taken)
        focal_ev.defeated = bool(defeat_bonus > 0.0)
        focal_ev.stash_bonus = float(stash_bonus)
        focal_ev.group_formed = bool(group_formed)
        focal_ev.betrayal = bool(focal_betrayal)
        # Non-focal slots: damage_dealt from heuristic/override return, and
        # damage_taken from _damage_taken_last_step (already populated by
        # focal ATTACK + monster ticks above).
        for i in range(self._n_agents):
            if i == acting_focal_idx:
                continue
            ev = step_events[i]
            ev.damage_dealt = float(heuristic_damage_dealt[i])
            ev.damage_taken = float(self._damage_taken_last_step[i])

        # Compute per-agent reward array. Acting-focal slot reuses the existing
        # scalar to preserve byte-identity. Other slots use _compute_combat_reward
        # with the per-slot pre-state and the events buffer; dead-at-start slots
        # get 0.0.
        per_agent_reward = np.zeros(self._n_agents, dtype=np.float64)
        per_agent_reward[acting_focal_idx] = float(reward)
        for i in range(self._n_agents):
            if i == acting_focal_idx:
                continue
            if not agent_pre_state[i]["alive"]:
                continue
            ag_i = self._agents[i]
            per_agent_reward[i] = self._compute_combat_reward(
                hunger_prev=agent_pre_state[i]["hunger"],
                health_prev=agent_pre_state[i]["health"],
                agent=ag_i,
                inv_food_prev=agent_pre_state[i]["inv_food"],
                events=step_events[i],
            )

        info["agent_events"] = step_events.events
        info["per_agent_reward"] = per_agent_reward

        return obs, reward, terminated, False, info

    # ── Combat helpers ────────────────────────────────────────────────────────

    def _do_attack(self, attacker: Agent, requested_tier: StrikeTier = STRIKE_BASIC) -> tuple[float, bool]:
        """Attack the nearest target within Chebyshev distance 1 (8 directions). Returns (damage_dealt, killed).

        Targeting priority: live boss-monster (if adjacent) > nearest agent.
        Boss attacks credit the attacker via Monster.take_damage; on kill the
        env drops a shared loot stash for all attackers.

        If the attacker has group members flanking (adjacent to the target from any direction),
        each ally grants a GROUP_ATTACK_BONUS_PER_ALLY multiplicative damage bonus
        (agent targets only — no flanking bonus for boss yet).

        v18: ``requested_tier`` is the qi-spend tier the policy chose
        (STRIKE_BASIC / STRIKE_QI / STRIKE_BURST). Qi is consumed only when a
        target is found (no qi wasted on whiffs); insufficient-qi cases
        downgrade automatically.
        """
        # Boss takes priority when adjacent
        bx, by = attacker.position
        adjacent_monsters = self._monsters.get_adjacent_to(bx, by)
        if adjacent_monsters:
            monster = adjacent_monsters[0]
            tier = self._spend_strike_qi(attacker, requested_tier)
            damage = self._combat_damage(attacker, attacker, is_defending=False, tier=tier)
            # NB: `_combat_damage` ignores the defender unless is_defending=True,
            # so passing attacker as a stand-in is safe and keeps damage scaling
            # tied to the attacker's strength only.
            killed = monster.take_damage(damage, attacker.agent_id)
            self._ep_boss_attacks_landed += 1
            self._ep_boss_damage_dealt += damage
            if killed:
                self._ep_boss_killed += 1
                # v19c: shared victory bond for every pair of contributors.
                self._record_joint_kill_bonds(monster.attackers)
                self._drop_boss_loot(monster)
            return damage, killed

        target = self._nearest_adjacent_agent(attacker)
        if target is None:
            return 0.0, False
        attacker_idx = self._agents.index(attacker)
        target_idx = self._agents.index(target)
        flanking_allies = self._adjacent_group_allies(attacker_idx, target)
        tier = self._spend_strike_qi(attacker, requested_tier)
        damage = self._combat_damage(attacker, target, is_defending=False, tier=tier)
        if flanking_allies:
            bonus = GROUP_ATTACK_BONUS_PER_ALLY * len(flanking_allies)
            damage = float(np.clip(damage * (1.0 + bonus), 0.0, COMBAT_MAX_DAMAGE))
        target.health = max(0.0, target.health - damage)
        target._check_death("combat")
        # v19: directional affinity — record only when the attack actually landed.
        # Attacker → target: mild commitment to enmity. Target → attacker: strong hostility.
        if damage > 0:
            self._record_affinity_event(
                actor_idx=attacker_idx, other_idx=target_idx,
                actor_to_other=AFFINITY_ATTACK_ATTACKER,
                other_to_actor=AFFINITY_ATTACK_VICTIM,
            )
            # Flank-assist bond: each flanking ally and the attacker reinforce each
            # other (both chose to engage the same target — coordinated combat).
            for fidx in flanking_allies:
                self._record_affinity_event(
                    actor_idx=attacker_idx, other_idx=fidx,
                    actor_to_other=AFFINITY_FLANK_BOTH,
                    other_to_actor=AFFINITY_FLANK_BOTH,
                )
        if not target.alive:
            self._drop_inventory(target)
            return damage, True
        return damage, False

    def _drop_boss_loot(self, monster: Monster) -> None:
        """On boss death, register a shared loot stash for every attacker.

        The stash's owner_id is the monster_id (sentinel), with all attacker
        agent_ids as participants — so every contributor can WITHDRAW from it
        via the standard stash interface.
        """
        if not monster.attackers:
            return
        loot = Stash(
            stash_id=f"{monster.monster_id}_loot",
            owner_id=monster.monster_id,
            position=monster.position,
            food=BOSS_LOOT_FOOD,
            qi=BOSS_LOOT_QI,
            materials=BOSS_LOOT_MATERIALS,
            poison=0,
            participants=sorted(monster.attackers),
        )
        self._stash_registry.register(loot)

    def _execute_override_action(
        self, agent: Agent, agent_idx: int, action: Action, focal: Agent, focal_defending: bool
    ) -> float:
        """Execute a policy-chosen action for a non-focal agent during full-team inference.

        Returns damage dealt to the focal agent (0 unless agent attacks focal).
        Used by record_combat.py when _action_overrides is set — replaces the heuristic
        so all agents run the actual trained policy during replay.
        """
        # Apply the same survival redirect + invalid-action guards that focal agents get.
        action = self._redirect_invalid_action(agent, action, agent_idx=agent_idx)

        ax, ay = agent.position
        fx, fy = focal.position
        adjacent = max(abs(ax - fx), abs(ay - fy)) <= 1
        gs = self._world.grid_size
        damage_to_focal = 0.0

        if action in ATTACK_ACTIONS:
            target = self._nearest_adjacent_agent(agent)
            if target is not None and target.alive:
                is_focal_target = (target is focal)
                defending = focal_defending if is_focal_target else False
                tier = self._spend_strike_qi(agent, ACTION_TO_STRIKE[action])
                damage = self._combat_damage(agent, target, is_defending=defending, tier=tier)
                target.health = max(0.0, target.health - damage)
                target._check_death("combat")
                if not target.alive:
                    self._drop_inventory(target)
                if is_focal_target:
                    damage_to_focal = damage
                self._last_action_names[agent_idx] = action.name.lower()
                self._last_action_details[agent_idx] = target.agent_id
            else:
                self._heuristic_step(agent)
                self._last_action_names[agent_idx] = "gather"
                self._last_action_details[agent_idx] = ""

        elif action == Action.DEFEND:
            # DEFEND for non-focal: mark them as defending so if they get attacked this tick
            # (by focal or another agent) damage is reduced. Nothing else to do here.
            self._last_action_names[agent_idx] = "defend"
            self._last_action_details[agent_idx] = ""

        elif action == Action.EAT:
            if agent.inventory.food > 0:
                agent.eat(self._resource_configs)
                self._last_action_names[agent_idx] = "eat"
            else:
                self._last_action_names[agent_idx] = "rest"
            self._last_action_details[agent_idx] = ""

        elif action == Action.GATHER:
            x, y = agent.position
            food_grid = self._world.get_grid_view("food")
            if food_grid[y, x] > 0:
                self._world.deplete("food", x, y)
                agent.gather("food")
            self._last_action_names[agent_idx] = "gather"
            self._last_action_details[agent_idx] = ""

        elif action == Action.TRAIN:
            x, y = agent.position
            qi_val = self._world.get_qi_field_value(x, y) if "qi" in self._world.resources else 0.0
            agent.train(qi_field_value=qi_val)
            self._last_action_names[agent_idx] = "train"
            self._last_action_details[agent_idx] = ""

        elif action == Action.REST:
            self._last_action_names[agent_idx] = "rest"
            self._last_action_details[agent_idx] = ""

        elif action in MOVE_DELTAS:
            dx, dy = MOVE_DELTAS[action]
            agent.move(dx, dy, gs)
            self._last_action_names[agent_idx] = action.name.lower()
            self._last_action_details[agent_idx] = ""

        else:
            # Unsupported override action — fall back to heuristic
            damage_to_focal = self._heuristic_combat_step(agent, focal, focal_defending)

        return damage_to_focal

    def _heuristic_combat_step(
        self, agent: Agent, focal: Agent, focal_defending: bool
    ) -> float:
        """Heuristic for one non-focal agent. Returns damage dealt to focal agent.

        Requires Chebyshev distance ≤ 1 (8 directions) for combat to engage.

        v17: any non-focal agent adjacent to a live monster will attack it
        (regardless of strength differential). This lets all participants —
        not just the focal — earn boss-loot share, which is the whole point
        of the common-enemy emergence test.
        """
        ax, ay = agent.position
        agent_idx = self._agents.index(agent)

        # Boss takes priority over agent target (matches focal _do_attack policy)
        adjacent_monsters = self._monsters.get_adjacent_to(ax, ay)
        if adjacent_monsters:
            monster = adjacent_monsters[0]
            # v18: heuristic agents use STRIKE_QI when affordable (1 qi for +0.15 damage),
            # else fall back to basic. Burst is reserved for the policy to discover.
            requested = STRIKE_QI if agent.inventory.qi >= STRIKE_QI.qi_cost else STRIKE_BASIC
            tier = self._spend_strike_qi(agent, requested)
            damage = self._combat_damage(agent, agent, is_defending=False, tier=tier)
            killed = monster.take_damage(damage, agent.agent_id)
            self._ep_boss_attacks_landed += 1
            self._ep_boss_damage_dealt += damage
            if killed:
                self._ep_boss_killed += 1
                self._drop_boss_loot(monster)
            self._last_action_names[agent_idx] = "attack"
            self._last_action_details[agent_idx] = monster.monster_id
            return 0.0

        fx, fy = focal.position
        adjacent = max(abs(ax - fx), abs(ay - fy)) <= 1

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
                requested = STRIKE_QI if agent.inventory.qi >= STRIKE_QI.qi_cost else STRIKE_BASIC
                tier = self._spend_strike_qi(agent, requested)
                damage = self._combat_damage(agent, focal, is_defending=focal_defending, tier=tier)
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
        self, attacker: Agent, defender: Agent, is_defending: bool, tier: StrikeTier = STRIKE_BASIC,
    ) -> float:
        """Compute combat damage dealt to the defender.

        Base damage: (√(attacker.effective_strength) + tier.bonus) × COMBAT_ATTACKER_SCALE

        v18 curve: damage uses √strength so early training gains are 3-5× more
        impactful than they would be under linear scaling, while still saturating
        cleanly at strength=1.0. Combined with the saturating strength growth in
        Agent.train(), this gives the realistic "fast early gains, slow late
        gains" arc the cultivation genre expects, AND makes TRAIN feel rewarding
        in policy gradient updates because the early derivative is steep.

        Qi-infused strikes (v18, thematic): the policy chooses one of three
        Action.ATTACK_* variants (basic, qi, burst). Each maps to a StrikeTier
        with a flat bonus added to √strength *before* scaling. The qi-cost is
        consumed at the call site via ``_spend_strike_qi`` (which downgrades
        the tier if the attacker has insufficient qi). This keeps the function
        pure and lets the policy learn to trade qi reserves for spike damage —
        weak agents get a meaningful contribution against monsters, and rich
        cultivators get a finisher tool.

        When the defender chose DEFEND, damage is multiplied by
        (1 − defender.defense_power).  defense_power ∈ [0, 1] is a blend of
        effective_strength (0.5 weight) and avg cultivated resistance (0.5 weight),
        so both raw strength and hazard cultivation reduce damage taken.

        Scale:
          defense_power = 0.0  → DEFEND blocks  0 % of damage (untrained, starving)
          defense_power = 0.5  → DEFEND blocks 50 % of damage (moderate cultivator)
          defense_power = 1.0  → DEFEND blocks 100% of damage (master — full nullification)

        Damage is clamped to [0, COMBAT_MAX_DAMAGE].
        """
        strength_term = float(np.sqrt(attacker.effective_strength)) + tier.bonus
        base = strength_term * COMBAT_ATTACKER_SCALE
        if is_defending:
            base *= max(0.0, 1.0 - defender.defense_power)
        return float(np.clip(base, 0.0, COMBAT_MAX_DAMAGE))

    def _spend_strike_qi(self, attacker: Agent, requested: StrikeTier) -> StrikeTier:
        """Consume qi for the requested strike tier; downgrade if insufficient.

        Returns the actual tier used (≤ requested). Mutates attacker.inventory.qi
        and bumps per-episode telemetry counters. Free tiers (qi_cost=0) never
        downgrade and never touch inventory.

        Downgrade ladder:
          BURST (3 qi) → QI (1 qi) → BASIC (0 qi)
        """
        if requested.qi_cost == 0:
            return requested
        if attacker.inventory.qi >= requested.qi_cost:
            attacker.inventory.qi -= requested.qi_cost
            self._ep_qi_spent_in_combat += requested.qi_cost
            if requested is STRIKE_BURST:
                self._ep_burst_strikes_used += 1
            elif requested is STRIKE_QI:
                self._ep_qi_strikes_used += 1
            return requested
        # Insufficient qi — try to downgrade BURST → QI before falling to BASIC
        if requested is STRIKE_BURST and attacker.inventory.qi >= STRIKE_QI.qi_cost:
            return self._spend_strike_qi(attacker, STRIKE_QI)
        return STRIKE_BASIC

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

    def _has_adjacent_hostile(self, agent_idx: int) -> bool:
        """Return True if a monster is within Chebyshev 1 of the agent.

        Agent-on-agent hostility is detected via the damage-cooldown path
        instead of adjacency, because two strangers standing next to each
        other are not necessarily in combat (they may be COLLABORATE-ing).
        Once anyone takes damage, the 1-tick shaken cooldown takes over.
        """
        agent = self._agents[agent_idx]
        ax, ay = agent.position
        return bool(self._monsters.get_adjacent_to(ax, ay))

    def _in_combat(self, agent_idx: int) -> bool:
        """v17: agent is 'in combat' if hostile is adjacent OR shaken cooldown active.

        Hostile = any monster, or any non-group-mate live agent.
        Cooldown = took damage in the previous tick (1-tick shaken window).
        Returns False when the combat-focus feature is disabled.
        """
        if not self._enable_combat_focus:
            return False
        if self._in_combat_cooldown[agent_idx] > 0:
            return True
        return self._has_adjacent_hostile(agent_idx)

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
            # v19c: voluntary group formation creates a symmetric bond.
            self._record_affinity_event(
                actor_idx=focal_idx, other_idx=neighbour_idx,
                actor_to_other=AFFINITY_COLLAB_BOTH,
                other_to_actor=AFFINITY_COLLAB_BOTH,
            )
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
        food_gathered: int = 0,
        hazard_damage: float = 0.0,
        agent: Agent | None = None,
        exploration_reward: float = 0.0,
        damage_dealt: float = 0.0,
        damage_taken: float = 0.0,
        defeat_bonus: float = 0.0,
        inv_food_prev: int = 0,
        events: "AgentStepEvents | None" = None,
    ) -> float:
        """Phase 3c reward: Phase 3b reward + combat shaping.

        P1.2: optional ``events`` kwarg overrides ``food_gathered``,
        ``hazard_damage``, ``damage_dealt``, ``damage_taken``, ``defeat_bonus``.
        Existing positional callers are unchanged.
        """
        if events is not None:
            food_gathered = events.food_gathered
            hazard_damage = events.hazard_damage
            damage_dealt = events.damage_dealt
            damage_taken = events.damage_taken
            defeat_bonus = REWARD_DEFEAT_OPPONENT if events.defeated else 0.0
        reward = self._compute_reward(hunger_prev, health_prev, food_gathered, hazard_damage, agent, exploration_reward, inv_food_prev)
        reward += REWARD_DAMAGE_TAKEN_SCALE * damage_taken
        reward += defeat_bonus
        return reward
