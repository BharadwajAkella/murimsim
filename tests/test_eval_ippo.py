"""P5 — eval harness tests.

Coverage:
    A. Metric primitives (affinity_summary, group_summary, help_event_count)
       compute correctly on synthetic env state.
    B. eval_checkpoint loads both FF and recurrent checkpoints and produces
       finite, well-shaped EvalMetrics.
    C. Determinism: same seed + checkpoint → identical metrics.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

os.environ["WANDB_DISABLED"] = "1"

from murimsim.actions import N_ACTIONS_PHASE6_QI
from murimsim.rl.ippo import SharedActorCritic
from murimsim.rl.ippo_env import IPPOEnv
from murimsim.rl.ippo_recurrent import RecurrentSharedActorCritic
from murimsim.rl.multi_env import OBS_TOTAL_SIZE
from scripts.eval_ippo import (
    EvalMetrics,
    affinity_summary,
    eval_checkpoint,
    group_summary,
    help_event_count,
    load_policy,
)


CONFIG = "config/default.yaml"


def _load_cfg() -> dict:
    with open(CONFIG) as f:
        return yaml.safe_load(f)


def _make_env(seed: int = 0, n_agents: int = 4) -> IPPOEnv:
    e = IPPOEnv(config=_load_cfg(), n_agents=n_agents, seed=seed, curriculum_ramp_steps=0)
    e.reset_all(seed=seed)
    return e


# ─────────────────────── A. Metric primitives ───────────────────────

def test_affinity_summary_empty_env() -> None:
    e = _make_env(seed=1)
    max_abs, mean_abs, recip = affinity_summary([e])
    assert max_abs == 0.0
    assert mean_abs == 0.0
    assert recip == 0.0


def test_affinity_summary_synthetic_dyads() -> None:
    """Inject affinity values directly; verify summary stats are correct."""
    e = _make_env(seed=2, n_agents=3)
    # symmetric positive bond between 0 and 1; weak negative on 0→2.
    e._affinity_raw[0] = {1: (0.8, 0), 2: (-0.3, 0)}
    e._affinity_raw[1] = {0: (0.7, 0)}
    e._affinity_raw[2] = {0: (-0.4, 0)}

    max_abs, mean_abs, recip = affinity_summary([e])
    assert max_abs == pytest.approx(0.8)
    # 4 entries: 0.8, 0.3, 0.7, 0.4 → mean = 0.55
    assert mean_abs == pytest.approx(0.55)
    # paired: (0,1)=(0.8,0.7), (0,2)=(-0.3,-0.4) → strong + correlation
    assert recip > 0.99


def test_affinity_summary_anti_reciprocal() -> None:
    """If A loves B but B hates A in every dyad, reciprocity is strongly negative."""
    e = _make_env(seed=3, n_agents=3)
    e._affinity_raw[0] = {1: (0.9, 0), 2: (0.7, 0)}
    e._affinity_raw[1] = {0: (-0.8, 0)}
    e._affinity_raw[2] = {0: (-0.6, 0)}
    _max, _mean, recip = affinity_summary([e])
    assert recip < -0.95


def test_group_summary_counts_only_multi_member_groups() -> None:
    e = _make_env(seed=4, n_agents=4)
    e._groups = [frozenset({0, 1}), frozenset({2, 3, 0}), frozenset({1})]
    n_groups, mean_size = group_summary([e])
    # Singleton (size 1) excluded; two multi-member groups of sizes 2 and 3.
    assert n_groups == pytest.approx(2.0)
    assert mean_size == pytest.approx(2.5)


def test_help_event_count_aggregates_across_envs() -> None:
    e1 = _make_env(seed=5)
    e2 = _make_env(seed=6)
    e1._help_received = {0: {1: 10, 2: 12}, 3: {0: 14}}  # 3 events
    e2._help_received = {1: {0: 5}}                      # 1 event
    assert help_event_count([e1, e2]) == 4


# ─────────────────────── B. End-to-end on real checkpoints ───────────────────────

def _save_ff_checkpoint(path: Path) -> None:
    torch.manual_seed(0)
    policy = SharedActorCritic(obs_dim=OBS_TOTAL_SIZE, n_actions=N_ACTIONS_PHASE6_QI)
    torch.save({"policy": policy.state_dict(), "args": {}}, path)


def _save_recurrent_checkpoint(path: Path, hidden_dim: int = 16) -> None:
    torch.manual_seed(0)
    policy = RecurrentSharedActorCritic(
        obs_dim=OBS_TOTAL_SIZE, n_actions=N_ACTIONS_PHASE6_QI,
        hidden_dim=hidden_dim, pre_lstm_dim=hidden_dim,
    )
    torch.save(
        {"policy": policy.state_dict(), "args": {"hidden_dim": hidden_dim, "pre_lstm_dim": hidden_dim},
         "hidden_dim": hidden_dim},
        path,
    )


def test_load_policy_detects_ff_vs_recurrent(tmp_path: Path) -> None:
    ff_path = tmp_path / "ff.pt"
    rc_path = tmp_path / "rc.pt"
    _save_ff_checkpoint(ff_path)
    _save_recurrent_checkpoint(rc_path)

    _, _, is_recur_ff = load_policy(ff_path)
    _, _, is_recur_rc = load_policy(rc_path)
    assert is_recur_ff is False
    assert is_recur_rc is True


def test_eval_checkpoint_ff_returns_finite_metrics(tmp_path: Path) -> None:
    path = tmp_path / "ff.pt"
    _save_ff_checkpoint(path)
    m = eval_checkpoint(
        checkpoint_path=path, steps=200, n_envs=2, n_agents=4, seed=42,
    )
    assert isinstance(m, EvalMetrics)
    assert m.is_recurrent is False
    assert m.steps == 200
    for v in (m.max_abs_affinity, m.mean_abs_affinity, m.dyadic_reciprocity,
              m.n_active_groups, m.mean_group_size, m.mean_life_reward,
              m.mean_lifespan):
        assert np.isfinite(v), f"non-finite metric {v}"
    assert m.help_events >= 0
    assert m.completed_lives >= 0


def test_eval_checkpoint_recurrent_returns_finite_metrics(tmp_path: Path) -> None:
    path = tmp_path / "rc.pt"
    _save_recurrent_checkpoint(path)
    m = eval_checkpoint(
        checkpoint_path=path, steps=200, n_envs=2, n_agents=4, seed=99,
    )
    assert m.is_recurrent is True
    for v in (m.max_abs_affinity, m.mean_abs_affinity, m.dyadic_reciprocity):
        assert np.isfinite(v)


# ─────────────────────── C. Determinism ───────────────────────

def test_eval_checkpoint_deterministic_with_seed(tmp_path: Path) -> None:
    path = tmp_path / "ff.pt"
    _save_ff_checkpoint(path)
    m1 = eval_checkpoint(checkpoint_path=path, steps=150, n_envs=2, n_agents=4, seed=7)
    m2 = eval_checkpoint(checkpoint_path=path, steps=150, n_envs=2, n_agents=4, seed=7)
    assert m1 == m2, f"non-deterministic eval:\n  m1={m1}\n  m2={m2}"
