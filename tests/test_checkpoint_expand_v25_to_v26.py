"""tests/test_checkpoint_expand_v25_to_v26.py — Phase 7 warm-start expansion.

Validates that a synthetic v25 checkpoint expands cleanly into a v26 one and
that the resulting state_dict loads into a v26-shaped JointRecurrentSharedActorCritic
without a shape mismatch.
"""
from __future__ import annotations

from pathlib import Path

import torch

from murimsim.actions import N_BODY_ACTIONS
from murimsim.rl.ippo_joint_recurrent import JointRecurrentSharedActorCritic
from murimsim.rl.multi_env import OBS_TOTAL_SIZE, OBS_TOTAL_SIZE_COURTSHIP

from scripts.expand_checkpoint_v25_to_v26 import (
    V25_OBS_DIM,
    V25_SOCIAL_OUT,
    V26_OBS_DIM,
    V26_SOCIAL_OUT,
    expand_checkpoint,
)


def test_obs_and_social_dim_constants_match_env():
    # Historical migration constants from the v25→v26 (pre-Phase 8) era.
    # The env has since evolved (Phase 8 added flame as a 5th resource
    # channel), so these no longer equal the current OBS_TOTAL_SIZE — but
    # the migration script remains valid for upgrading v25-shape ckpts.
    assert V25_OBS_DIM == 289
    assert V26_OBS_DIM == 291


def test_expand_round_trip(tmp_path: Path):
    v25_policy = JointRecurrentSharedActorCritic(
        obs_dim=V25_OBS_DIM,
        n_body_actions=N_BODY_ACTIONS,
        n_social_actions=V25_SOCIAL_OUT,
    )
    src = tmp_path / "v25.pt"
    torch.save(
        {"iter": 100, "global_step": 1234, "policy": v25_policy.state_dict(),
         "optimizer": {"placeholder": True}},
        src,
    )

    dst = tmp_path / "v26.pt"
    expand_checkpoint(src, dst)

    new_ckpt = torch.load(dst, map_location="cpu", weights_only=False)
    assert new_ckpt["v26_courtship"] is True
    assert new_ckpt["iter"] == 0
    assert new_ckpt["global_step"] == 0
    assert "optimizer" not in new_ckpt

    v26_policy = JointRecurrentSharedActorCritic(
        obs_dim=V26_OBS_DIM,
        n_body_actions=N_BODY_ACTIONS,
        n_social_actions=V26_SOCIAL_OUT,
    )
    missing, unexpected = v26_policy.load_state_dict(new_ckpt["policy"], strict=True)
    assert not missing
    assert not unexpected

    pre_lstm_w = v26_policy.pre_lstm[0].weight.detach()
    assert torch.all(pre_lstm_w[:, V25_OBS_DIM:] == 0)

    social_w = v26_policy.social_head.weight.detach()
    social_b = v26_policy.social_head.bias.detach()
    assert torch.all(social_w[V25_SOCIAL_OUT:, :] == 0)
    assert torch.all(social_b[V25_SOCIAL_OUT:] == 0)
