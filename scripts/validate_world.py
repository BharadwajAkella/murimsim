"""validate_world.py — Phase 1 validation script.

Runs the world for 100 ticks and produces:
  - Console output: resource counts every 10 ticks
  - logs/validation/resource_density_heatmap.png  — final-state density per resource
  - logs/validation/resource_count_timeseries.png — count over time per resource

Usage (from project root):
    python scripts/validate_world.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

# Allow running from project root or from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

from murimsim.world import World

CONFIG_PATH = Path(__file__).parent.parent / "config" / "default.yaml"
OUTPUT_DIR = Path(__file__).parent.parent / "logs" / "validation"

_RESOURCE_CMAPS = {
    "food": "Greens",
    "qi": "Blues",
    "materials": "Oranges",
    "poison": "Reds",
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    world = World(config)
    resource_ids = world.get_resource_ids()
    n_ticks = 100

    print(f"Grid size : {world.grid_size}x{world.grid_size}")
    print(f"Resources : {resource_ids}")
    print(f"Running   : {n_ticks} ticks\n")

    # ---- Run simulation, record counts per tick ----
    counts: dict[str, list[int]] = {rid: [] for rid in resource_ids}

    for tick in range(n_ticks):
        for rid in resource_ids:
            counts[rid].append(world.count(rid))

        if tick % 10 == 0:
            row = " | ".join(
                f"{world.resources[rid].display_name}: {counts[rid][-1]:4d}"
                for rid in resource_ids
            )
            print(f"Tick {tick:3d}: {row}")

        world.step()

    # Final tick counts
    for rid in resource_ids:
        counts[rid].append(world.count(rid))
    row = " | ".join(
        f"{world.resources[rid].display_name}: {counts[rid][-1]:4d}"
        for rid in resource_ids
    )
    print(f"Tick {n_ticks:3d}: {row}")

    # ---- Heatmap: final resource density ----
    n = len(resource_ids)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, rid in zip(axes, resource_ids):
        cmap = _RESOURCE_CMAPS.get(rid, "viridis")
        im = ax.imshow(world.get_grid(rid), cmap=cmap, vmin=0, vmax=1, origin="upper")
        ax.set_title(world.resources[rid].display_name, fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Resource Density — Final State (tick {n_ticks})", fontsize=13)
    heatmap_path = OUTPUT_DIR / "resource_density_heatmap.png"
    fig.savefig(heatmap_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nHeatmap    saved → {heatmap_path}")

    # ---- Timeseries: resource count over time ----
    fig, ax = plt.subplots(figsize=(10, 4))
    for rid in resource_ids:
        ax.plot(counts[rid], label=world.resources[rid].display_name, linewidth=1.5)
    ax.set_xlabel("Tick")
    ax.set_ylabel("Tile count")
    ax.set_title(f"Resource Count Timeseries ({n_ticks} ticks)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ts_path = OUTPUT_DIR / "resource_count_timeseries.png"
    fig.savefig(ts_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Timeseries saved → {ts_path}")


if __name__ == "__main__":
    main()
