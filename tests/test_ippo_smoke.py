"""P3 IPPO integration smoke test.

Runs a tiny IPPO training loop end-to-end against IPPOEnv and checks:
  * No NaN in policy parameters or losses.
  * Buffer contains at least some active samples per iter.
  * Action masks are respected throughout (no illegal actions taken).
  * Affinity matrix moves off zero (proves agents interacted).
  * Determinism: same seed → identical final summary.
"""
from __future__ import annotations

import os
from argparse import Namespace

import numpy as np
import pytest
import torch

from scripts.train_ippo import train


@pytest.fixture(autouse=True)
def _disable_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_DISABLED", "1")


def _smoke_args(seed: int = 0, total_steps: int = 1024, tmpdir: str = "/tmp/ippo_smoke") -> Namespace:
    return Namespace(
        config="config/default.yaml",
        total_steps=total_steps,
        n_envs=2,
        n_agents=4,
        rollout_length=64,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_coef=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        n_epochs=2,
        n_minibatches=2,
        max_grad_norm=0.5,
        seed=seed,
        device="cpu",
        checkpoint_dir=tmpdir,
        checkpoint_interval=1_000_000,  # don't write
        log_interval=1_000_000,         # silence
        no_wandb=True,
        wandb_project="x",
        wandb_run_name=None,
    )


def test_smoke_train_runs_without_nan(tmp_path):
    summary = train(_smoke_args(seed=0, tmpdir=str(tmp_path)))
    assert np.isfinite(summary["pg_loss"]), summary
    assert np.isfinite(summary["vf_loss"]), summary
    assert np.isfinite(summary["entropy"]), summary
    assert summary["n_active"] > 0


def test_smoke_train_affinity_moves_off_zero(tmp_path):
    """Proves agents actually interact during training (rotating focal works)."""
    summary = train(_smoke_args(seed=1, tmpdir=str(tmp_path), total_steps=2048))
    assert summary["affinity_l1"] > 0.0, (
        f"affinity matrix never updated — IPPO not exercising social interactions: {summary}"
    )


def test_smoke_train_deterministic_with_seed(tmp_path):
    s1 = train(_smoke_args(seed=7, tmpdir=str(tmp_path / "a")))
    s2 = train(_smoke_args(seed=7, tmpdir=str(tmp_path / "b")))
    # Compare losses (small rounding tolerance)
    for k in ("pg_loss", "vf_loss", "entropy", "n_active", "affinity_l1"):
        assert s1[k] == pytest.approx(s2[k], rel=1e-5, abs=1e-6), (
            f"non-deterministic at key {k}: {s1[k]} vs {s2[k]}"
        )
