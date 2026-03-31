"""Transfer weights from a v15 (obs=263) checkpoint to a v16 (obs=264) compatible checkpoint.

The only architectural change is obs_size 263→264 (new damage_taken_last_step stat).
This affects only the LSTM input weight matrices (weight_ih_l0) for actor and critic.
We extend those from [256, 263] → [256, 264] by appending a zero-initialised column.

The new obs dimension starts contributing nothing (zero weight) and the model learns
to use it over time — all prior knowledge (survival, combat, train) is preserved.

Usage:
    python3 scripts/transfer_weights.py \
        --source checkpoints/limbic_lstm_v15/limbic_lstm_v15_final.zip \
        --dest   checkpoints/limbic_lstm_v16_init/limbic_lstm_v16_init.zip
"""
from __future__ import annotations

import argparse
import copy
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
import yaml
from sb3_contrib import RecurrentPPO

from murimsim.rl.multi_env import CombatEnv

OLD_OBS_SIZE = 263
NEW_OBS_SIZE = 264
LSTM_INPUT_KEYS = ("lstm_actor.weight_ih_l0", "lstm_critic.weight_ih_l0")


def transfer(source: Path, dest: Path, config_path: Path) -> None:
    print(f"Loading source checkpoint: {source}")
    old_model = RecurrentPPO.load(str(source))

    old_sd = old_model.policy.state_dict()

    # Verify source obs size
    for key in LSTM_INPUT_KEYS:
        shape = old_sd[key].shape
        assert shape[1] == OLD_OBS_SIZE, (
            f"{key} has input dim {shape[1]}, expected {OLD_OBS_SIZE}"
        )

    # Build new env with obs_size=264
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    env = CombatEnv(config=cfg, n_agents=10, seed=0)
    env.reset(seed=0)

    # Infer LSTM hidden size from the source model's weight shape: [4*hidden, obs]
    lstm_hidden_size = old_sd["lstm_actor.weight_hh_l0"].shape[1]
    print(f"Detected LSTM hidden size: {lstm_hidden_size}")

    print("Creating new model with updated obs space …")
    new_model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        verbose=0,
        policy_kwargs={"lstm_hidden_size": lstm_hidden_size},
        n_steps=old_model.n_steps,
        batch_size=old_model.batch_size,
        n_epochs=old_model.n_epochs,
        learning_rate=old_model.learning_rate,
        gamma=old_model.gamma,
        gae_lambda=old_model.gae_lambda,
        clip_range=old_model.clip_range,
        ent_coef=old_model.ent_coef,
        vf_coef=old_model.vf_coef,
        max_grad_norm=old_model.max_grad_norm,
    )

    new_sd = new_model.policy.state_dict()

    # Transfer: copy all matching keys, pad the two LSTM input weight matrices
    patched_sd = copy.deepcopy(new_sd)
    patched_keys, padded_keys, skipped_keys = [], [], []

    for key, old_tensor in old_sd.items():
        if key not in new_sd:
            skipped_keys.append(key)
            continue
        new_tensor = new_sd[key]
        if old_tensor.shape == new_tensor.shape:
            patched_sd[key] = old_tensor.cpu().clone()
            patched_keys.append(key)
        elif key in LSTM_INPUT_KEYS and old_tensor.shape[1] == OLD_OBS_SIZE:
            # Pad with one zero column on the right: [256, 263] → [256, 264]
            old_cpu = old_tensor.cpu()
            zero_col = torch.zeros(old_cpu.shape[0], 1, dtype=old_cpu.dtype)
            patched_sd[key] = torch.cat([old_cpu, zero_col], dim=1)
            padded_keys.append(key)
        else:
            skipped_keys.append(f"{key} (shape mismatch: {old_tensor.shape} vs {new_tensor.shape})")

    new_model.policy.load_state_dict(patched_sd)

    dest.parent.mkdir(parents=True, exist_ok=True)
    new_model.save(str(dest))

    print(f"\n{'='*60}")
    print(f"Copied unchanged  ({len(patched_keys)} tensors)")
    print(f"Zero-padded       ({len(padded_keys)} tensors): {padded_keys}")
    if skipped_keys:
        print(f"Skipped           ({len(skipped_keys)}): {skipped_keys}")
    print(f"\nSaved patched checkpoint → {dest}")
    print("The new obs dimension (damage_taken_last_step) starts with zero weight.")
    print("All prior survival/combat/train behaviour is preserved.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Path to v15 .zip checkpoint")
    parser.add_argument("--dest",   required=True, help="Destination .zip path for patched checkpoint")
    parser.add_argument("--config", default="config/default.yaml", help="YAML config path")
    args = parser.parse_args()
    transfer(Path(args.source), Path(args.dest), Path(args.config))


if __name__ == "__main__":
    main()
