"""Expand a v25 IPPO joint-recurrent checkpoint into a v26-compatible one.

v25 → v26 architectural deltas:
  * obs_dim: 264 → 266 (sex bit + pending-proposal bit appended at tail)
  * social head out_features: 2 → 4 (added PROPOSE, ACCEPT)

Both deltas are handled by zero-padding the corresponding weight matrices so
the new dimensions contribute nothing at warm-start. The model recovers prior
behaviour exactly on tick 0 and learns to use the new signals over training.

Affected tensors in `JointRecurrentSharedActorCritic.state_dict()`:
  * pre_lstm.0.weight           : [pre_lstm_dim, 264] → [pre_lstm_dim, 266]
                                  Append zero columns on the right.
  * social_head.weight          : [2, hidden_dim] → [4, hidden_dim]
                                  Append zero rows on the bottom.
  * social_head.bias            : [2] → [4]
                                  Append zero entries on the bottom.

All other tensors (LSTM, body_head, critic_head) are copied unchanged.

Usage:
    python3 scripts/expand_checkpoint_v25_to_v26.py \
        --source checkpoints/ippo_v25b_rec/ippo_joint_recurrent_iter_XXXXXX.pt \
        --dest   checkpoints/ippo_v26_courtship_init/ippo_joint_recurrent_iter_000000.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

V25_OBS_DIM: int = 289
V26_OBS_DIM: int = 291
V25_SOCIAL_OUT: int = 2
V26_SOCIAL_OUT: int = 4

PRE_LSTM_WEIGHT_KEY: str = "pre_lstm.0.weight"
SOCIAL_WEIGHT_KEY: str = "social_head.weight"
SOCIAL_BIAS_KEY: str = "social_head.bias"


def _pad_obs_input(weight: torch.Tensor) -> torch.Tensor:
    """Right-pad a [out, V25_OBS_DIM] linear weight with zero columns."""
    if weight.shape[1] != V25_OBS_DIM:
        raise ValueError(
            f"{PRE_LSTM_WEIGHT_KEY} has obs dim {weight.shape[1]}, expected {V25_OBS_DIM}"
        )
    pad = torch.zeros(weight.shape[0], V26_OBS_DIM - V25_OBS_DIM, dtype=weight.dtype)
    return torch.cat([weight, pad], dim=1)


def _pad_social_out(tensor: torch.Tensor, key: str) -> torch.Tensor:
    """Bottom-pad a tensor whose first dim is V25_SOCIAL_OUT to V26_SOCIAL_OUT."""
    if tensor.shape[0] != V25_SOCIAL_OUT:
        raise ValueError(
            f"{key} has out_features {tensor.shape[0]}, expected {V25_SOCIAL_OUT}"
        )
    extra_rows = V26_SOCIAL_OUT - V25_SOCIAL_OUT
    if tensor.dim() == 2:
        pad = torch.zeros(extra_rows, tensor.shape[1], dtype=tensor.dtype)
    else:
        pad = torch.zeros(extra_rows, dtype=tensor.dtype)
    return torch.cat([tensor, pad], dim=0)


def expand_checkpoint(source: Path, dest: Path) -> None:
    print(f"Loading v25 checkpoint: {source}")
    ckpt = torch.load(source, map_location="cpu", weights_only=False)
    sd = ckpt["policy"]

    expanded = {}
    for key, val in sd.items():
        if key == PRE_LSTM_WEIGHT_KEY:
            expanded[key] = _pad_obs_input(val)
            print(f"  {key}: {tuple(val.shape)} → {tuple(expanded[key].shape)} (zero-pad obs cols)")
        elif key == SOCIAL_WEIGHT_KEY:
            expanded[key] = _pad_social_out(val, key)
            print(f"  {key}: {tuple(val.shape)} → {tuple(expanded[key].shape)} (zero-pad social rows)")
        elif key == SOCIAL_BIAS_KEY:
            expanded[key] = _pad_social_out(val, key)
            print(f"  {key}: {tuple(val.shape)} → {tuple(expanded[key].shape)} (zero-pad social bias)")
        else:
            expanded[key] = val

    new_ckpt = dict(ckpt)
    new_ckpt["policy"] = expanded
    new_ckpt.pop("optimizer", None)  # invalid for the new param shapes; reinit on load
    new_ckpt["iter"] = 0
    new_ckpt["global_step"] = 0
    new_ckpt["expanded_from"] = str(source)
    new_ckpt["v26_courtship"] = True

    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_ckpt, dest)
    print(f"Saved v26 init checkpoint: {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--dest", type=Path, required=True)
    args = ap.parse_args()
    expand_checkpoint(args.source, args.dest)


if __name__ == "__main__":
    main()
