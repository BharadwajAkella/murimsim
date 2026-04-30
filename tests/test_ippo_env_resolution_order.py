"""P2.2 tests — randomized action resolution order in IPPOEnv.

Today the focal slot's action runs first (winning any kill-steal contention),
then non-focal slots iterate in ascending index order. For IPPO, slots are
symmetric — privilege should rotate across steps so no single slot dominates.

Strategy: IPPOEnv.step_all randomly picks which slot is the focal each step,
using the env's RNG so it's deterministic with seed.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from murimsim.actions import Action
from murimsim.rl.ippo_env import IPPOEnv

CONFIG_PATH = Path("config/default.yaml")


def _load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _make_ippo(n_agents: int = 6, seed: int = 0) -> IPPOEnv:
    return IPPOEnv(
        config=_load_cfg(),
        n_agents=n_agents,
        seed=seed,
        curriculum_ramp_steps=0,
    )


def test_resolution_order_logged_in_info() -> None:
    env = _make_ippo(n_agents=4, seed=20)
    env.reset(seed=20)
    actions = np.full(4, Action.MOVE_N.value, dtype=np.int64)
    _o, _r, _t, _tr, info = env.step_all(actions)
    assert "resolution_order" in info
    order = info["resolution_order"]
    assert isinstance(order, list)
    assert len(order) == 4
    assert sorted(order) == list(range(4)), "must be a permutation of [0..n_agents-1]"


def test_focal_choice_rotates_across_steps() -> None:
    """Over many steps, every slot should be chosen focal at least once."""
    env = _make_ippo(n_agents=4, seed=21)
    env.reset(seed=21)
    actions = np.full(4, Action.MOVE_N.value, dtype=np.int64)
    focal_counts: Counter = Counter()
    for _ in range(200):
        _o, _r, _t, _tr, info = env.step_all(actions)
        focal_counts[info["resolution_order"][0]] += 1
        if any(_t):
            env.reset(seed=21)
    # All 4 slots should have been focal at least once
    assert len(focal_counts) == 4, f"some slots never focal: {focal_counts}"
    # Roughly uniform: no slot should be > 50% of all picks
    total = sum(focal_counts.values())
    for slot, c in focal_counts.items():
        assert c / total < 0.5, f"slot {slot} dominated focal picks: {c}/{total}"


def test_step_all_deterministic_with_randomized_focal() -> None:
    """Same seed → same resolution order across steps."""
    env1 = _make_ippo(n_agents=5, seed=22)
    env1.reset(seed=22)
    env2 = _make_ippo(n_agents=5, seed=22)
    env2.reset(seed=22)
    actions = np.full(5, Action.MOVE_N.value, dtype=np.int64)
    orders1 = []
    orders2 = []
    for _ in range(20):
        _, _, _, _, i1 = env1.step_all(actions)
        _, _, _, _, i2 = env2.step_all(actions)
        orders1.append(tuple(i1["resolution_order"]))
        orders2.append(tuple(i2["resolution_order"]))
    assert orders1 == orders2


def test_focal_picked_only_from_live_slots() -> None:
    """Dead slots should never be chosen as focal."""
    env = _make_ippo(n_agents=4, seed=23)
    env.reset(seed=23)
    # Kill slots 0 and 2
    env._agents[0].health = 0.0
    env._agents[0].alive = False
    env._agents[2].health = 0.0
    env._agents[2].alive = False
    actions = np.full(4, Action.REST.value, dtype=np.int64)
    for _ in range(30):
        _o, _r, _t, _tr, info = env.step_all(actions)
        chosen = info["resolution_order"][0]
        assert chosen in (1, 3), f"focal {chosen} but only slots 1,3 alive"
        if any(_t):
            break


def test_single_live_agent_always_focal() -> None:
    """When only one slot is alive, it must always be the focal."""
    env = _make_ippo(n_agents=4, seed=24)
    env.reset(seed=24)
    for i in (0, 1, 3):
        env._agents[i].health = 0.0
        env._agents[i].alive = False
    actions = np.full(4, Action.REST.value, dtype=np.int64)
    _o, _r, _t, _tr, info = env.step_all(actions)
    assert info["resolution_order"][0] == 2
