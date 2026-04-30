"""Golden tests pinning focal-agent reward to current behavior.

Captured BEFORE the P1.2 per-agent reward refactor. They MUST keep passing
byte-identical after the refactor — they are the single most important
invariant for proving the refactor changed nothing for the focal agent.

Regeneration:
    REGEN_REWARD_GOLDEN=1 pytest tests/test_focal_reward_unchanged.py

This rewrites the .npy fixtures from current behavior. Only do this if you
explicitly intend to change reward semantics.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import yaml

from murimsim.actions import Action
from murimsim.rl.multi_env import CombatEnv

CONFIG_PATH = Path("config/default.yaml")
FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE_DIR.mkdir(exist_ok=True)

GOLDEN_STEPS = 200
N_AGENTS = 6
SEED = 42
REGEN = os.environ.get("REGEN_REWARD_GOLDEN") == "1"


def _load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _make_env(seed: int = SEED) -> CombatEnv:
    return CombatEnv(
        config=_load_cfg(),
        n_agents=N_AGENTS,
        seed=seed,
        curriculum_ramp_steps=300_000,
    )


def _action_sequence_gather() -> list[int]:
    """Deterministic action loop that exercises gather/move/eat/rest paths."""
    base = [
        Action.MOVE_N.value,
        Action.GATHER.value,
        Action.MOVE_E.value,
        Action.GATHER.value,
        Action.EAT.value,
        Action.MOVE_S.value,
        Action.REST.value,
        Action.MOVE_W.value,
        Action.DEPOSIT.value,
        Action.MOVE_N.value,
    ]
    return [base[i % len(base)] for i in range(GOLDEN_STEPS)]


def _action_sequence_combat() -> list[int]:
    """Deterministic action loop that exercises combat paths."""
    base = [
        Action.MOVE_N.value,
        Action.ATTACK.value,
        Action.DEFEND.value,
        Action.MOVE_E.value,
        Action.ATTACK.value,
        Action.COLLABORATE.value,
        Action.MOVE_S.value,
        Action.ATTACK_QI.value,
        Action.GATHER.value,
        Action.EAT.value,
        Action.REST.value,
        Action.MOVE_W.value,
        Action.ATTACK_BURST.value,
    ]
    return [base[i % len(base)] for i in range(GOLDEN_STEPS)]


def _run_and_collect(action_seq: list[int], seed: int) -> np.ndarray:
    """Run env for GOLDEN_STEPS, return per-step focal reward as float64 array."""
    env = _make_env(seed=seed)
    env.reset(seed=seed)
    rewards = np.zeros(GOLDEN_STEPS, dtype=np.float64)
    for t, act in enumerate(action_seq):
        _obs, reward, terminated, truncated, _info = env.step(int(act))
        rewards[t] = float(reward)
        if terminated or truncated:
            env.reset(seed=seed + t + 1)
    return rewards


def _check_or_regen(name: str, rewards: np.ndarray) -> None:
    fixture_path = FIXTURE_DIR / f"focal_reward_{name}.npy"
    if REGEN or not fixture_path.exists():
        np.save(fixture_path, rewards)
        if REGEN:
            print(f"REGENERATED {fixture_path}")
            return
    expected = np.load(fixture_path)
    # Byte-identical equality. If reward semantics drift even by 1 ULP,
    # this fails. assert_array_equal compares bit-for-bit on float arrays.
    np.testing.assert_array_equal(
        rewards,
        expected,
        err_msg=(
            f"Focal reward drifted from golden fixture {fixture_path}. "
            "If this drift is intentional, regenerate with "
            "REGEN_REWARD_GOLDEN=1 pytest. Otherwise, the refactor changed "
            "reward semantics — fix the refactor."
        ),
    )


def test_focal_reward_byte_identical_gather_scenario() -> None:
    """200-step gather/eat/move scenario — focal reward must be bit-identical."""
    rewards = _run_and_collect(_action_sequence_gather(), seed=SEED)
    _check_or_regen("gather", rewards)


def test_focal_reward_byte_identical_combat_scenario() -> None:
    """200-step combat-flavored scenario — focal reward must be bit-identical."""
    rewards = _run_and_collect(_action_sequence_combat(), seed=SEED + 1)
    _check_or_regen("combat", rewards)


def test_golden_determinism_self_check() -> None:
    """Sanity: running twice with same seed yields identical reward arrays."""
    r1 = _run_and_collect(_action_sequence_gather(), seed=SEED)
    r2 = _run_and_collect(_action_sequence_gather(), seed=SEED)
    np.testing.assert_array_equal(r1, r2)
