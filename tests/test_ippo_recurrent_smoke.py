"""End-to-end smoke tests for the recurrent IPPO trainer."""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ["WANDB_DISABLED"] = "1"

from scripts.train_ippo_recurrent import train as train_recurrent  # noqa: E402


def _args(tmp_path, **kw):
    import argparse
    a = argparse.Namespace(
        config="config/default.yaml",
        total_steps=512,
        n_envs=2,
        n_agents=4,
        rollout_length=32,
        hidden_dim=16,
        pre_lstm_dim=16,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_coef=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        n_epochs=1,
        n_minibatches=2,
        max_grad_norm=0.5,
        seed=0,
        device="cpu",
        checkpoint_dir=str(tmp_path / "ckpt"),
        checkpoint_interval=1,
        log_interval=1,
        no_wandb=True,
        wandb_project="murimsim-ippo",
        wandb_run_name=None,
    )
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_smoke_recurrent_train_completes(tmp_path) -> None:
    """Recurrent IPPO trains 512 transitions without NaN; checkpoint is written."""
    summary = train_recurrent(_args(tmp_path))
    assert summary["iter"] >= 1
    for k in ("pg_loss", "vf_loss", "entropy", "approx_kl"):
        assert np.isfinite(summary[k]), f"non-finite {k}: {summary[k]}"
    ckpt_files = list((tmp_path / "ckpt").glob("*.pt"))
    assert len(ckpt_files) >= 1, f"no checkpoint written: {list((tmp_path / 'ckpt').iterdir())}"


def test_smoke_recurrent_train_deterministic_with_seed(tmp_path) -> None:
    """Two runs with same seed produce same per-iter summary stats."""
    s1 = train_recurrent(_args(tmp_path / "a", seed=7))
    s2 = train_recurrent(_args(tmp_path / "b", seed=7))
    for k in ("pg_loss", "vf_loss", "entropy", "approx_kl", "n_active"):
        assert s1[k] == s2[k], f"non-deterministic at {k}: {s1[k]} vs {s2[k]}"
