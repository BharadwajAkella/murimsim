"""record_agent.py — Record an agent playing in the sim for the web viewer.

Two modes:
  Heuristic (default): rule-based foraging policy, no model required.
    Output: logs/replays/run_{seed}_heuristic.jsonl

  Model: loads a trained PPO checkpoint and runs deterministic inference.
    Output: logs/replays/run_{seed}_model.jsonl
    Usage:  --model checkpoints/limbic_v1/limbic_v1_final.zip

Usage (from project root):
    python scripts/record_agent.py
    python scripts/record_agent.py --ticks 500 --seed 42
    python scripts/record_agent.py --model checkpoints/limbic_v1/limbic_v1_final.zip
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from murimsim.actions import Action, MOVE_DELTAS
from murimsim.agent import Agent
from murimsim.replay import ReplayLogger
from murimsim.rl.env import SurvivalEnv
from murimsim.world import World


CONFIG_PATH = Path(__file__).parent.parent / "config" / "default.yaml"
import os
from pathlib import Path
OUTPUT_DIR = Path(os.environ.get("MURIM_REPLAY_DIR", "logs/replays"))

# Cardinal move actions in priority order for food scanning
_CARDINAL_MOVES = [Action.MOVE_N, Action.MOVE_S, Action.MOVE_E, Action.MOVE_W]


def world_to_resource_snapshot(world: World) -> dict[str, list[list[int | float]]]:
    """Convert all present resource tiles to [[x, y, 1.0], ...] lists for the viewer."""
    snapshot: dict[str, list[list[int | float]]] = {}
    for rid in world.get_resource_ids():
        grid = world.get_grid_view(rid)
        ys, xs = np.where(grid > 0)
        snapshot[rid] = [[int(x), int(y), 1.0] for x, y in zip(xs, ys)]
    return snapshot


def stash_snapshot(env: SurvivalEnv, focal_agent_id: str) -> list[dict]:
    """Serialise all known stashes to viewer format: {pos, owner, is_own}."""
    return [
        {
            "pos": list(s.position),
            "owner": s.owner_id,
            "is_own": s.owner_id == focal_agent_id,
        }
        for s in env._stash_registry.all_stashes()
    ]


class HeuristicPolicy:
    """Rule-based foraging policy.

    Decision priority each tick:
      1. Eat — if hungry (hunger > 0.3) and carrying food.
      2. Gather — if standing on a food tile.
      3. Navigate toward nearest food within SCAN_RADIUS tiles (Manhattan distance).
         Step one tile along the shortest path; skip recently visited tiles when
         multiple equidistant candidates exist.
      4. Explore — keep walking in _direction; bounce to a random new direction on walls.
    """

    _HUNGER_EAT_THRESHOLD: float = 0.3
    _SCAN_RADIUS: int = 6          # look up to 6 tiles away (Manhattan)
    _RECENCY_WINDOW: int = 6       # ignore recently visited tiles when choosing a target

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._direction: Action = Action.MOVE_E
        self._recent: list[tuple[int, int]] = []

    def decide(self, agent: Agent, world: World) -> Action:
        """Return the next action given the agent's current state and world."""
        x, y = agent.position
        gs = world.grid_size
        food_grid = world.get_grid_view("food")

        # Track recency
        self._recent.append((x, y))
        if len(self._recent) > self._RECENCY_WINDOW:
            self._recent.pop(0)

        # 1. Eat if hungry and carrying food
        if agent.hunger > self._HUNGER_EAT_THRESHOLD and agent.inventory.food > 0:
            return Action.EAT

        # 2. Gather if standing on food
        if food_grid[y, x] > 0:
            return Action.GATHER

        # 3. Find nearest food within SCAN_RADIUS (Manhattan) and step toward it
        target = self._nearest_food(x, y, gs, food_grid)
        if target is not None:
            return self._step_toward(x, y, target)

        # 4. Explore: continue in current direction; bounce on walls
        dx, dy = MOVE_DELTAS[self._direction]
        nx, ny = x + dx, y + dy
        if 0 <= nx < gs and 0 <= ny < gs:
            return self._direction

        # Hit a wall — pick a new random cardinal direction (not back the way we came)
        opposite = {
            Action.MOVE_N: Action.MOVE_S,
            Action.MOVE_S: Action.MOVE_N,
            Action.MOVE_E: Action.MOVE_W,
            Action.MOVE_W: Action.MOVE_E,
        }
        candidates = [a for a in _CARDINAL_MOVES if a != opposite[self._direction]]
        self._direction = candidates[int(self._rng.integers(len(candidates)))]
        return self._direction

    def _nearest_food(
        self, x: int, y: int, gs: int, food_grid: np.ndarray
    ) -> tuple[int, int] | None:
        """Return the (tx, ty) of the closest food tile within SCAN_RADIUS.

        Prefers tiles not in _recent. Falls back to any food tile if all are recent.
        """
        best: tuple[int, int] | None = None
        best_recent: tuple[int, int] | None = None
        best_dist = self._SCAN_RADIUS + 1
        best_recent_dist = self._SCAN_RADIUS + 1

        r = self._SCAN_RADIUS
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if abs(dx) + abs(dy) > r:
                    continue
                nx, ny = x + dx, y + dy
                if not (0 <= nx < gs and 0 <= ny < gs):
                    continue
                if food_grid[ny, nx] == 0:
                    continue
                dist = abs(dx) + abs(dy)
                if (nx, ny) not in self._recent:
                    if dist < best_dist:
                        best_dist = dist
                        best = (nx, ny)
                else:
                    if dist < best_recent_dist:
                        best_recent_dist = dist
                        best_recent = (nx, ny)

        return best if best is not None else best_recent

    def _step_toward(self, x: int, y: int, target: tuple[int, int]) -> Action:
        """Return one cardinal step toward target; updates _direction."""
        tx, ty = target
        dx, dy = tx - x, ty - y
        # Prefer the axis with the larger gap; ties broken by current direction
        if abs(dx) >= abs(dy):
            move = Action.MOVE_E if dx > 0 else Action.MOVE_W
        else:
            move = Action.MOVE_S if dy > 0 else Action.MOVE_N
        self._direction = move
        return move



def main() -> None:
    parser = argparse.ArgumentParser(description="Record a MurimSim agent replay.")
    parser.add_argument("--ticks", type=int, default=500, help="Max ticks to record (default: 500)")
    parser.add_argument("--seed",  type=int, default=42,  help="World seed (default: 42)")
    parser.add_argument("--debug", action="store_true", help="Print per-tick trace logs")
    parser.add_argument("--model", type=Path, default=None,
                        help="Path to PPO checkpoint zip (omit to use heuristic policy)")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    # --- Choose policy ---
    if args.model is not None:
        if not args.model.exists():
            print(f"[ERROR] Checkpoint not found: {args.model}", file=sys.stderr)
            sys.exit(1)
        from stable_baselines3 import PPO
        _ppo = PPO.load(args.model)
        def decide_action(agent: Agent, world: World, obs: np.ndarray) -> tuple[Action, np.ndarray]:
            action_int, _ = _ppo.predict(obs, deterministic=True)
            return Action(int(action_int)), obs   # obs updated by env.step
        policy_name = f"PPO  ({args.model.name})"
        out_filename = f"run_{args.seed}_model.jsonl"
    else:
        _heuristic = HeuristicPolicy(rng=np.random.default_rng(args.seed))
        def decide_action(agent: Agent, world: World, obs: np.ndarray) -> tuple[Action, np.ndarray]:
            return _heuristic.decide(agent, world), obs
        policy_name = "heuristic"
        out_filename = f"run_{args.seed}_heuristic.jsonl"

    # --- Create environment ---
    env = SurvivalEnv(config, seed=args.seed)
    obs, _ = env.reset(seed=args.seed)

    print(f"Policy             : {policy_name}")
    print(f"Regen              : enabled (regen_ticks from config)")
    print(f"Environment        : {env._world.grid_size}x{env._world.grid_size}  seed={args.seed}")
    print(f"Max ticks          : {args.ticks}")
    print(f"Output             : {OUTPUT_DIR / out_filename}\n")

    tick = 0
    survived_ticks = 0
    total_reward = 0.0
    action_counts: dict[str, int] = {}

    with ReplayLogger(seed=args.seed, output_dir=OUTPUT_DIR, filename=out_filename) as replay:
        while tick < args.ticks:
            agent = env._agent
            world = env._world
            x, y = agent.position
            food_here = world.get_grid("food")[y, x] > 0

            # Decide action
            action, _ = decide_action(agent, world, obs)
            action_int = action.value
            act_name = action.name.lower()
            action_counts[act_name] = action_counts.get(act_name, 0) + 1

            # --- Verbose per-tick trace (--debug only) ---
            if args.debug:
                print(f"[tick {tick:4d}] entered ({x}, {y})  "
                      f"food_here={'YES' if food_here else 'no '}  "
                      f"inv_food={agent.inventory.food}  "
                      f"hunger={agent.hunger:.2f}  "
                      f"→ {act_name}")
                if act_name == "gather" and food_here:
                    print(f"           gathering food at ({x}, {y})")
                if act_name == "eat" and agent.inventory.food > 0:
                    print(f"           eating food (inv before: {agent.inventory.food})")

            # Record state BEFORE the step executes
            replay.log_tick(
                tick=tick,
                generation=0,
                agents=[agent.to_replay_dict(action=act_name, action_detail="")],
                resources=world_to_resource_snapshot(world),
                stashes=stash_snapshot(env, agent.agent_id),
                events=[],
            )

            food_before = world.count("food")

            # Step environment (obs updated here for model policy)
            obs, reward, terminated, _truncated, info = env.step(action_int)
            total_reward += reward
            survived_ticks += 1

            food_after = world.count("food")
            if args.debug:
                if food_before != food_after:
                    print(f"           food on map: {food_before} → {food_after}  "
                          f"(depleted tile ({x}, {y}))")
                elif act_name == "gather":
                    print(f"           WARNING: gather action but food count unchanged "
                          f"(food_here was {'YES' if food_here else 'NO'})")

            if terminated:
                agent = env._agent
                replay.log_tick(
                    tick=tick + 1,
                    generation=0,
                    agents=[agent.to_replay_dict(action="dead", action_detail="health reached 0")],
                    resources=world_to_resource_snapshot(world),
                    stashes=stash_snapshot(env, agent.agent_id),
                    events=[{"type": "death", "agent_id": agent.agent_id, "tick": tick + 1}],
                )
                print(f"\n  [DIED] Agent died at tick {tick + 1}.")
                break

            tick += 1

    # --- Summary ---
    print(f"\n{'─'*55}")
    print(f"  Survived : {survived_ticks} / {args.ticks} ticks")
    print(f"  Total reward : {total_reward:+.3f}")
    print(f"  Action breakdown:")
    for act, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / max(1, survived_ticks)
        print(f"    {act:<14} {count:4d}  ({pct:.1f}%)")
    print(f"{'─'*55}")
    print(f"\nReplay saved → logs/replays/{out_filename}")
    print(f"Open viewer : http://localhost:8000/viewer/index.html  (then load the file)")


if __name__ == "__main__":
    main()
