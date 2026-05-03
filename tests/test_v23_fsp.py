"""tests/test_v23_fsp.py — frozen-policy self-play (FSP) smoke tests.

Verifies that ``train_ippo`` and ``train_ippo_recurrent`` can be launched
with ``--n-policy-agents < --n-agents`` and a ``--frozen-ckpt``, and that
only the first n_policy slots feed into the buffer.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _has_ckpt(path: str) -> bool:
    return (REPO_ROOT / path).exists()


def _ckpt_obs_dim_matches(path: str) -> bool:
    """True if the checkpoint's pre_lstm/pre_lstm.0 input dim matches current env."""
    full = REPO_ROOT / path
    if not full.exists():
        return False
    try:
        from murimsim.rl.multi_env import OBS_TOTAL_SIZE
        ckpt = torch.load(full, map_location="cpu", weights_only=False)
        sd = ckpt.get("policy", ckpt)
        for key in ("pre_lstm.0.weight", "pre_lstm.weight", "fc1.weight", "trunk.0.weight"):
            if key in sd:
                return sd[key].shape[1] == OBS_TOTAL_SIZE
        return True  # unknown layout — let the test try
    except Exception:
        return False


@pytest.mark.skipif(
    not _has_ckpt("checkpoints/ippo_v22a_ff/ippo_iter_000732.pt")
    or not _ckpt_obs_dim_matches("checkpoints/ippo_v22a_ff/ippo_iter_000732.pt"),
    reason="frozen FF checkpoint absent or obs-dim incompatible with current env",
)
def test_fsp_ff_smoke(tmp_path):
    """FF trainer with FSP runs end-to-end, saves a checkpoint."""
    env = {**os.environ, "WANDB_DISABLED": "1", "PYTHONUNBUFFERED": "1"}
    out = subprocess.run(
        [
            sys.executable, "-m", "scripts.train_ippo",
            "--total-steps", "256",
            "--rollout-length", "16",
            "--n-envs", "2",
            "--n-agents", "4",
            "--n-policy-agents", "2",
            "--frozen-ckpt", "checkpoints/ippo_v22a_ff/ippo_iter_000732.pt",
            "--checkpoint-dir", str(tmp_path),
            "--checkpoint-interval", "1",
            "--no-wandb",
            "--seed", "0",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, f"stderr:\n{out.stderr}\nstdout:\n{out.stdout}"
    # Confirm at least one checkpoint was emitted, and active_n reflects only
    # the policy slots (rollout 16 × n_envs 2 × n_policy 2 = 64).
    assert any(p.suffix == ".pt" for p in tmp_path.iterdir())
    assert "active_n=64" in out.stderr or "active_n=64" in out.stdout, (
        f"expected active_n=64 in logs:\n{out.stderr}"
    )


@pytest.mark.skipif(
    not _has_ckpt("checkpoints/ippo_v22b_rec/ippo_recurrent_iter_000732.pt")
    or not _ckpt_obs_dim_matches("checkpoints/ippo_v22b_rec/ippo_recurrent_iter_000732.pt"),
    reason="frozen recurrent checkpoint absent or obs-dim incompatible with current env",
)
def test_fsp_recurrent_smoke(tmp_path):
    """Recurrent trainer with recurrent frozen baseline runs end-to-end."""
    env = {**os.environ, "WANDB_DISABLED": "1", "PYTHONUNBUFFERED": "1"}
    out = subprocess.run(
        [
            sys.executable, "-m", "scripts.train_ippo_recurrent",
            "--total-steps", "256",
            "--rollout-length", "16",
            "--n-envs", "2",
            "--n-agents", "4",
            "--n-policy-agents", "2",
            "--frozen-ckpt",
            "checkpoints/ippo_v22b_rec/ippo_recurrent_iter_000732.pt",
            "--checkpoint-dir", str(tmp_path),
            "--checkpoint-interval", "1",
            "--no-wandb",
            "--seed", "0",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert out.returncode == 0, f"stderr:\n{out.stderr}\nstdout:\n{out.stdout}"
    assert any(p.suffix == ".pt" for p in tmp_path.iterdir())


def test_fsp_requires_frozen_ckpt():
    """Using --n-policy-agents < --n-agents without --frozen-ckpt errors."""
    env = {**os.environ, "WANDB_DISABLED": "1"}
    out = subprocess.run(
        [
            sys.executable, "-m", "scripts.train_ippo",
            "--total-steps", "32",
            "--rollout-length", "8",
            "--n-envs", "1",
            "--n-agents", "4",
            "--n-policy-agents", "2",
            "--no-wandb",
            "--checkpoint-dir", "/tmp/fsp_no_ckpt_should_fail",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert out.returncode != 0
    assert "frozen-ckpt required" in (out.stderr + out.stdout)


def test_fsp_disabled_when_n_policy_equals_n_agents(tmp_path):
    """No FSP path when n_policy_agents == n_agents (or default)."""
    env = {**os.environ, "WANDB_DISABLED": "1"}
    out = subprocess.run(
        [
            sys.executable, "-m", "scripts.train_ippo",
            "--total-steps", "128",
            "--rollout-length", "16",
            "--n-envs", "2",
            "--n-agents", "2",
            "--no-wandb",
            "--checkpoint-dir", str(tmp_path),
            "--seed", "0",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode == 0, f"stderr:\n{out.stderr}"
