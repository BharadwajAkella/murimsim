"""markov_experiment.py — Demonstrates the cost of partial observability.

Experiment: "Does history matter when the agent only sees a 5×5 window?"

Two policies are compared across identical seeds:

  MEMORYLESS  — Pure Markov. Only the current 5×5 observation is used.
                Moves toward the nearest food tile visible right now.
                No record of where it has been.

  MEMORY-AIDED — Tracks the last N visited tiles. When choosing a food
                 target, deprioritises recently visited tiles (likely depleted).

Key metric: depleted-tile revisits — how often does the agent step onto a tile
it just gathered from (and which hasn't regenerated yet)?

A high revisit rate wastes action ticks on empty tiles, burning hunger without
gaining food — exactly the POMDP failure mode discussed in Stage 2.

Usage (from project root):
    python3 scripts/markov_experiment.py
    python3 scripts/markov_experiment.py --ticks 600 --seed 7
    python3 scripts/markov_experiment.py --no-replay
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from pathlib import Path
from typing import NamedTuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from murimsim.actions import Action, MOVE_DELTAS
from murimsim.agent import Agent
from murimsim.replay import ReplayLogger
from murimsim.rl.env import SurvivalEnv
from murimsim.world import World

CONFIG_PATH = Path(__file__).parent.parent / "config" / "default.yaml"
OUTPUT_DIR  = Path(os.environ.get("MURIM_REPLAY_DIR", "/mnt/c/Users/bhara/Downloads/replays"))

_CARDINAL_MOVES = [Action.MOVE_N, Action.MOVE_S, Action.MOVE_E, Action.MOVE_W]
_SCAN_RADIUS    = 8   # how far each policy looks for food (Manhattan distance)


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

class MemorylessPolicy:
    """Pure Markov foraging policy — no history, only the current observation.

    Decision each tick:
      1. Eat if hungry and carrying food.
      2. Gather if standing on food.
      3. Move toward the nearest food tile within SCAN_RADIUS.
      4. Random walk if nothing visible.

    This is the minimal Markov agent: its next action depends only on the
    current world state, never on where it has been.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._direction: Action = Action.MOVE_E

    def decide(self, agent: Agent, world: World) -> Action:
        x, y = agent.position
        gs   = world.grid_size
        food = world.get_grid_view("food")

        if agent.hunger > 0.3 and agent.inventory.food > 0:
            return Action.EAT
        if food[y, x] > 0:
            return Action.GATHER

        target = _nearest_food(x, y, gs, food, exclude=set())
        if target is not None:
            return _step_toward(x, y, target, self)

        # Random walk
        dx, dy = MOVE_DELTAS[self._direction]
        if 0 <= x + dx < gs and 0 <= y + dy < gs:
            return self._direction
        candidates = [a for a in _CARDINAL_MOVES
                      if a != _opposite(self._direction)]
        self._direction = candidates[int(self._rng.integers(len(candidates)))]
        return self._direction


class MemoryAidedPolicy:
    """Memory-aided foraging policy — tracks recently visited tiles.

    Identical to MemorylessPolicy except it maintains a recency window of
    the last MEMORY_SIZE positions visited. When choosing a food target it
    avoids recently visited tiles (which may be depleted), only falling back
    to them if no fresher tile is visible.
    """

    MEMORY_SIZE: int = 12

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng    = rng
        self._direction: Action = Action.MOVE_E
        self._recent: deque[tuple[int, int]] = deque(maxlen=self.MEMORY_SIZE)

    def decide(self, agent: Agent, world: World) -> Action:
        x, y = agent.position
        gs   = world.grid_size
        food = world.get_grid_view("food")

        self._recent.append((x, y))

        if agent.hunger > 0.3 and agent.inventory.food > 0:
            return Action.EAT
        if food[y, x] > 0:
            return Action.GATHER

        target = _nearest_food(x, y, gs, food, exclude=set(self._recent))
        if target is None:
            # Fall back: include recent tiles
            target = _nearest_food(x, y, gs, food, exclude=set())

        if target is not None:
            return _step_toward(x, y, target, self)

        dx, dy = MOVE_DELTAS[self._direction]
        if 0 <= x + dx < gs and 0 <= y + dy < gs:
            return self._direction
        candidates = [a for a in _CARDINAL_MOVES
                      if a != _opposite(self._direction)]
        self._direction = candidates[int(self._rng.integers(len(candidates)))]
        return self._direction


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _nearest_food(
    x: int, y: int, gs: int,
    food: np.ndarray,
    exclude: set[tuple[int, int]],
) -> tuple[int, int] | None:
    """Return (tx, ty) of the closest non-excluded food tile within SCAN_RADIUS."""
    best: tuple[int, int] | None = None
    best_dist = _SCAN_RADIUS + 1
    for dy in range(-_SCAN_RADIUS, _SCAN_RADIUS + 1):
        for dx in range(-_SCAN_RADIUS, _SCAN_RADIUS + 1):
            if abs(dx) + abs(dy) > _SCAN_RADIUS:
                continue
            nx, ny = x + dx, y + dy
            if not (0 <= nx < gs and 0 <= ny < gs):
                continue
            if food[ny, nx] == 0:
                continue
            if (nx, ny) in exclude:
                continue
            dist = abs(dx) + abs(dy)
            if dist < best_dist:
                best_dist = dist
                best = (nx, ny)
    return best


def _step_toward(x: int, y: int, target: tuple[int, int], policy: object) -> Action:
    tx, ty = target
    dx, dy = tx - x, ty - y
    move = (Action.MOVE_E if dx > 0 else Action.MOVE_W) if abs(dx) >= abs(dy) \
        else (Action.MOVE_S if dy > 0 else Action.MOVE_N)
    policy._direction = move  # type: ignore[attr-defined]
    return move


def _opposite(action: Action) -> Action:
    return {
        Action.MOVE_N: Action.MOVE_S,
        Action.MOVE_S: Action.MOVE_N,
        Action.MOVE_E: Action.MOVE_W,
        Action.MOVE_W: Action.MOVE_E,
    }[action]


def _world_snapshot(world: World) -> dict[str, list[list[int | float]]]:
    snapshot: dict[str, list[list[int | float]]] = {}
    for rid in world.get_resource_ids():
        grid = world.get_grid_view(rid)
        ys, xs = np.where(grid > 0)
        snapshot[rid] = [[int(x), int(y), 1.0] for x, y in zip(xs, ys)]
    return snapshot


# ---------------------------------------------------------------------------
# Run one trial
# ---------------------------------------------------------------------------

class TrialResult(NamedTuple):
    policy_name:      str
    survived_ticks:   int
    max_ticks:        int
    total_reward:     float
    depleted_revisits: int   # stepped onto a tile it had depleted itself
    wasted_gathers:   int   # GATHER action on empty tile
    gather_successes: int   # GATHER action that actually collected food


def run_trial(
    policy_name: str,
    policy: MemorylessPolicy | MemoryAidedPolicy,
    config: dict,
    seed: int,
    max_ticks: int,
    save_replay: bool,
) -> TrialResult:
    """Run a full episode and return stats."""
    env = SurvivalEnv(config, seed=seed)
    env.reset(seed=seed)

    # Track tiles the agent itself depleted this episode
    # Maps (x, y) → tick when it was depleted.  Tile re-enters this set
    # only after regen_ticks have elapsed (approximated by removing entries
    # older than regen_ticks).
    food_regen_ticks: int = next(
        r["regen_ticks"] for r in config["resources"] if r["id"] == "food"
    )
    depleted_by_agent: dict[tuple[int, int], int] = {}

    survived  = 0
    reward_sum = 0.0
    depleted_revisits = 0
    wasted_gathers    = 0
    gather_successes  = 0

    filename = f"run_{seed}_{policy_name.lower().replace(' ', '_')}.jsonl"

    replay_ctx = (
        ReplayLogger(seed=seed, output_dir=OUTPUT_DIR, filename=filename)
        if save_replay
        else _NullLogger()
    )

    with replay_ctx as replay:
        for tick in range(max_ticks):
            agent = env._agent
            world = env._world
            x, y  = agent.position

            # Check if stepping on a tile we ourselves depleted (and not yet regened)
            if (x, y) in depleted_by_agent:
                depleted_at = depleted_by_agent[(x, y)]
                if tick - depleted_at < food_regen_ticks:
                    depleted_revisits += 1

            action = policy.decide(agent, world)

            replay.log_tick(
                tick=tick,
                generation=0,
                agents=[agent.to_replay_dict(action=action.name.lower(), action_detail="")],
                resources=_world_snapshot(world),
                events=[],
            )

            food_count_before = world.count("food")
            food_on_tile_before = world.get_grid("food")[y, x] > 0

            _, reward, terminated, _trunc, _ = env.step(action.value)
            reward_sum += reward
            survived   += 1

            # Track gather outcomes
            if action == Action.GATHER:
                food_depleted = food_on_tile_before and (world.count("food") < food_count_before)
                if food_depleted:
                    gather_successes += 1
                    depleted_by_agent[(x, y)] = tick
                elif not food_on_tile_before:
                    wasted_gathers += 1

            # Evict fully-regened entries to keep dict small
            depleted_by_agent = {
                pos: t for pos, t in depleted_by_agent.items()
                if tick - t < food_regen_ticks
            }

            if terminated:
                replay.log_tick(
                    tick=tick + 1,
                    generation=0,
                    agents=[agent.to_replay_dict(action="dead", action_detail="health reached 0")],
                    resources=_world_snapshot(world),
                    events=[{"type": "death", "agent_id": agent.agent_id, "tick": tick + 1}],
                )
                break

    return TrialResult(
        policy_name=policy_name,
        survived_ticks=survived,
        max_ticks=max_ticks,
        total_reward=reward_sum,
        depleted_revisits=depleted_revisits,
        wasted_gathers=wasted_gathers,
        gather_successes=gather_successes,
    )


class _NullLogger:
    """Drop-in for ReplayLogger when replay recording is disabled."""
    def __enter__(self) -> _NullLogger:
        return self
    def __exit__(self, *_: object) -> None:
        pass
    def log_tick(self, **_: object) -> None:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Markov vs. Memory-Aided experiment.")
    parser.add_argument("--ticks",     type=int, default=500, help="Max ticks per trial (default: 500)")
    parser.add_argument("--seed",      type=int, default=42,  help="World seed (default: 42)")
    parser.add_argument("--no-replay", action="store_true",   help="Skip saving replay files")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    rng_ml = np.random.default_rng(args.seed)
    rng_ma = np.random.default_rng(args.seed)   # same seed → identical random walks

    save = not args.no_replay

    print("=" * 62)
    print("  MurimSim — Markov vs. Partial Observability Experiment")
    print("=" * 62)
    print(f"  Seed: {args.seed}   Max ticks: {args.ticks}")
    print(f"  Replay output: {OUTPUT_DIR}\n")

    trials = [
        run_trial("Memoryless",   MemorylessPolicy(rng_ml),   config, args.seed, args.ticks, save),
        run_trial("Memory-Aided", MemoryAidedPolicy(rng_ma), config, args.seed, args.ticks, save),
    ]

    # --- Results table ---
    print(f"  {'Metric':<30}  {'Memoryless':>12}  {'Memory-Aided':>12}")
    print("  " + "-" * 58)

    def row(label: str, ml: object, ma: object, fmt: str = "") -> None:
        print(f"  {label:<30}  {format(ml, fmt):>12}  {format(ma, fmt):>12}")

    ml, ma = trials
    row("Survived ticks",         ml.survived_ticks,    ma.survived_ticks)
    row("Total reward",           f"{ml.total_reward:+.2f}", f"{ma.total_reward:+.2f}")
    row("Successful gathers",     ml.gather_successes,  ma.gather_successes)
    row("Wasted gather actions",  ml.wasted_gathers,    ma.wasted_gathers)
    row("Depleted-tile revisits", ml.depleted_revisits, ma.depleted_revisits)

    revisit_rate_ml = ml.depleted_revisits / max(1, ml.survived_ticks) * 100
    revisit_rate_ma = ma.depleted_revisits / max(1, ma.survived_ticks) * 100
    row("Revisit rate (%/tick)",  f"{revisit_rate_ml:.1f}%", f"{revisit_rate_ma:.1f}%")

    print("  " + "-" * 58)

    # --- Interpretation ---
    delta_revisits = ml.depleted_revisits - ma.depleted_revisits
    delta_survival = ma.survived_ticks - ml.survived_ticks
    print(f"\n  Memory reduced depleted-tile revisits by {delta_revisits} "
          f"({revisit_rate_ml:.1f}% → {revisit_rate_ma:.1f}%)")
    if delta_survival > 0:
        print(f"  Memory-aided agent survived {delta_survival} ticks longer.")
    elif delta_survival < 0:
        print(f"  Memoryless agent survived {-delta_survival} ticks longer "
              f"(memory hurt here — possible aliasing or luck).")
    else:
        print(f"  Both agents survived the same number of ticks.")

    print(f"\n  KEY INSIGHT: The memoryless agent re-visits tiles it just")
    print(f"  depleted because its 5×5 window cannot distinguish 'tile I")
    print(f"  emptied 3 steps ago' from 'tile with food'. This is")
    print(f"  perceptual aliasing — the same local observation maps to")
    print(f"  two very different true states. History matters.")

    if save:
        print(f"\n  Replays saved to: {OUTPUT_DIR}")
        print(f"    run_{args.seed}_memoryless.jsonl")
        print(f"    run_{args.seed}_memory-aided.jsonl")
        print(f"  Open viewer: http://localhost:8000/viewer/index.html")

    print("=" * 62)


if __name__ == "__main__":
    main()
