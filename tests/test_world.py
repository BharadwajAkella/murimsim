"""Phase 1 — World Mechanics tests.

All 9 tests must pass for the Phase 1 exit gate.
These tests must continue to pass in all subsequent phases.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from murimsim.world import World

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent.parent / "config" / "default.yaml"


def load_config() -> dict:
    """Load the canonical default config."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def make_world(seed: int = 42, config: dict | None = None) -> World:
    """Create a World with an optional seed override."""
    cfg = copy.deepcopy(config or load_config())
    cfg["world"]["seed"] = seed
    return World(cfg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_world_determinism():
    """Seed 42 run 200 ticks == seed 42 run 200 ticks (byte-identical)."""

    def run_200() -> dict[str, np.ndarray]:
        w = make_world(seed=42)
        for _ in range(200):
            w.step()
        return {rid: w.get_grid(rid) for rid in w.get_resource_ids()}

    state_a = run_200()
    state_b = run_200()

    for rid in state_a:
        np.testing.assert_array_equal(
            state_a[rid], state_b[rid], err_msg=f"Grid mismatch for resource '{rid}'"
        )


def test_resource_conservation():
    """Bookkeeping invariant: total_depleted == total_regenerated + waiting_for_regen."""
    w = make_world()

    # Deplete up to 5 present tiles per resource
    for rid in w.get_resource_ids():
        ys, xs = np.where(w._grid[rid] > 0)
        for y, x in zip(ys[:5], xs[:5]):
            w.deplete(rid, int(x), int(y))

    # Run long enough that some tiles have already regenerated
    for _ in range(50):
        w.step()

    for rid in w.get_resource_ids():
        depleted = w.stats.total_depleted[rid]
        regenerated = w.stats.total_regenerated[rid]
        waiting = w.waiting_for_regen(rid)
        assert depleted == regenerated + waiting, (
            f"Conservation violated for '{rid}': "
            f"depleted={depleted}, regenerated={regenerated}, waiting={waiting}"
        )


def test_resource_regeneration():
    """Deplete a node → still absent after regen_ticks-1 steps, present after regen_ticks."""
    w = make_world()
    cfg = load_config()
    food_cfg = next(r for r in cfg["resources"] if r["id"] == "food")
    regen_ticks: int = food_cfg["regen_ticks"]

    ys, xs = np.where(w._grid["food"] > 0)
    assert len(ys) > 0, "No food tiles found — cannot test regeneration"
    y, x = int(ys[0]), int(xs[0])

    w.deplete("food", x, y)
    assert w.get_grid("food")[y, x] == 0.0, "Tile should be absent immediately after depletion"

    for _ in range(regen_ticks - 1):
        w.step()
    assert w.get_grid("food")[y, x] == 0.0, "Tile regenerated too early"

    w.step()  # exactly regen_ticks steps since depletion
    assert w.get_grid("food")[y, x] == 1.0, "Tile did not regenerate after regen_ticks"


def test_world_bounds():
    """Nothing exists outside [0, grid_size); out-of-bounds deplete raises."""
    w = make_world()
    gs = w.grid_size

    for rid in w.get_resource_ids():
        g = w.get_grid(rid)
        assert g.shape == (gs, gs), f"Grid shape mismatch for '{rid}'"

    # All of these must raise — exact exception type may vary
    invalid_positions = [(gs, 0), (0, gs), (-1, 0), (0, -1)]
    for bx, by in invalid_positions:
        with pytest.raises((ValueError, IndexError)):
            w.deplete("food", bx, by)


def test_config_loading():
    """YAML changes apply correctly — grid_size change is reflected in world."""
    cfg = load_config()
    cfg["world"]["grid_size"] = 20
    w = World(cfg)

    assert w.grid_size == 20
    for rid in w.get_resource_ids():
        assert w.get_grid(rid).shape == (20, 20), f"Grid shape wrong for '{rid}'"


def test_different_seeds():
    """Seed 42 and seed 99 produce different worlds."""
    w42 = make_world(seed=42)
    w99 = make_world(seed=99)

    any_diff = any(
        not np.array_equal(w42.get_grid(rid), w99.get_grid(rid))
        for rid in w42.get_resource_ids()
    )
    assert any_diff, "Different seeds produced identical worlds"


def test_resource_registry_extensible():
    """A new resource added via YAML only spawns correctly and regenerates."""
    cfg = load_config()
    cfg["resources"].append(
        {
            "id": "jade",
            "display_name": "Jade",
            "regen_ticks": 8,
            "spawn_density": 0.08,
            "effect": "positive",
            "effect_params": {},
        }
    )
    w = World(cfg)

    assert "jade" in w.get_resource_ids(), "New resource not registered"
    assert w.count("jade") > 0, "New resource has no tiles"

    # Verify it regenerates correctly
    ys, xs = np.where(w._grid["jade"] > 0)
    y, x = int(ys[0]), int(xs[0])
    w.deplete("jade", x, y)
    for _ in range(8):
        w.step()
    assert w.get_grid("jade")[y, x] == 1.0, "New resource did not regenerate"


def test_poison_spawns():
    """Poison appears at non-zero density; it is tracked as a separate layer from food."""
    w = make_world()

    poison_count = w.count("poison")
    food_count = w.count("food")

    assert poison_count > 0, "No poison tiles spawned"
    assert poison_count < food_count, (
        f"Poison ({poison_count}) should be sparser than food ({food_count})"
    )
    # Poison and food are distinct, independently tracked resource layers
    assert "poison" in w.get_resource_ids()
    assert "food" in w.get_resource_ids()
    assert not np.array_equal(w.get_grid("poison"), w.get_grid("food")), (
        "Poison and food grids are identical — they must be separate layers"
    )


def test_terrain_permanent():
    """Mountain terrain with regen_ticks=0 never regenerates after depletion."""
    cfg = load_config()
    # Ensure mountain exists in config (it should after the update)
    mountain_cfg = next((r for r in cfg["resources"] if r["id"] == "mountain"), None)
    if mountain_cfg is None:
        pytest.skip("No mountain resource in config")

    w = World(cfg)
    # Place a mountain tile manually if none were spawned
    ys, xs = np.where(w._grid["mountain"] > 0)
    if len(ys) == 0:
        w._grid["mountain"][5, 5] = 1.0
        y, x = 5, 5
    else:
        y, x = int(ys[0]), int(xs[0])

    # Deplete the tile
    w.deplete("mountain", x, y)
    assert w.get_grid("mountain")[y, x] == 0.0, "Mountain should be depleted"

    # Step 10 ticks — should remain depleted (permanent terrain)
    for _ in range(10):
        w.step()

    assert w.get_grid("mountain")[y, x] == 0.0, (
        "Mountain tile should remain depleted (regen_ticks=0 = permanent)"
    )
    """Poison respawns after its configured regen_ticks, same as other resources."""
    w = make_world()
    cfg = load_config()
    poison_cfg = next(r for r in cfg["resources"] if r["id"] == "poison")
    regen_ticks: int = poison_cfg["regen_ticks"]

    ys, xs = np.where(w._grid["poison"] > 0)
    assert len(ys) > 0, "No poison tiles found — cannot test poison regen"
    y, x = int(ys[0]), int(xs[0])

    w.deplete("poison", x, y)
    assert w.get_grid("poison")[y, x] == 0.0

    for _ in range(regen_ticks - 1):
        w.step()
    assert w.get_grid("poison")[y, x] == 0.0, "Poison regenerated too early"

    w.step()
    assert w.get_grid("poison")[y, x] == 1.0, "Poison did not regenerate after regen_ticks"


def test_corner_patch_denser_than_sparse() -> None:
    """dense_patch.yaml corner patches must be significantly denser than the sparse background."""
    import yaml
    from pathlib import Path
    from murimsim.rl.train_multienv import _deep_merge_env_override

    with open("config/default.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg = _deep_merge_env_override(cfg, Path("config/envs/dense_patch.yaml"))

    world = World(cfg, rng=np.random.default_rng(42))
    food = world.get_grid_view("food")
    H, W = food.shape

    tl_density = food[:H // 2, :W // 2].mean()
    sparse_density = food[:H // 2, W // 2:].mean()  # top-right has no patch

    assert tl_density > sparse_density * 5, (
        f"Top-left patch ({tl_density:.3f}) should be >5x denser than sparse ({sparse_density:.3f})"
    )
