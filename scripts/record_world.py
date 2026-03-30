"""record_world.py — Phase 1.5 demo: record a world-only replay (no agents).

Runs the world for N ticks, depletes a handful of resource tiles each tick
to demonstrate regeneration, and writes a JSONL file loadable by the viewer.

Usage (from project root):
    python scripts/record_world.py
    python scripts/record_world.py --ticks 200 --seed 7
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from murimsim.replay import ReplayLogger
from murimsim.world import World



CONFIG_PATH = Path(__file__).parent.parent / "config" / "default.yaml"
OUTPUT_DIR  = Path(os.environ.get("MURIM_REPLAY_DIR", "logs/replays"))


def world_to_resource_snapshot(world: World) -> dict[str, list[list[int | float]]]:
    """Convert all present resource tiles to [[x, y, 1.0], ...] lists for the viewer."""
    snapshot: dict[str, list[list[int | float]]] = {}
    for rid in world.get_resource_ids():
        grid = world.get_grid(rid)
        ys, xs = np.where(grid > 0)
        snapshot[rid] = [[int(x), int(y), 1.0] for x, y in zip(xs, ys)]
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a world-only MurimSim replay.")
    parser.add_argument("--ticks", type=int, default=100, help="Number of ticks to record (default: 100)")
    parser.add_argument("--seed",  type=int, default=42,  help="World seed (default: 42)")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    config["world"]["seed"] = args.seed

    world = World(config)

    print(f"World   : {world.grid_size}x{world.grid_size}  seed={args.seed}")
    print(f"Ticks   : {args.ticks}")
    print(f"Output  : {OUTPUT_DIR / f'run_{args.seed}.jsonl'}\n")

    with ReplayLogger(seed=args.seed, output_dir=OUTPUT_DIR) as logger:
        for tick in range(args.ticks):
            logger.log_tick(
                tick=tick,
                generation=0,
                agents=[],
                resources=world_to_resource_snapshot(world),
                events=[],
            )

            world.step()

            if tick % 10 == 0:
                counts = {rid: world.count(rid) for rid in world.get_resource_ids()}
                print(f"  tick {tick:4d}: " + " | ".join(f"{rid}={c}" for rid, c in counts.items()))

    print(f"\nDone. Load  logs/replays/run_{args.seed}.jsonl  in the viewer.")


if __name__ == "__main__":
    main()
