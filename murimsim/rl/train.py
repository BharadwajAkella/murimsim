"""train.py — Phase 2 PPO training script.

Trains a single agent on the SurvivalEnv using Stable-Baselines3 PPO.
Saves checkpoints to checkpoints/limbic_v1/.

Usage (from project root):
    python murimsim/rl/train.py
    python murimsim/rl/train.py --timesteps 1000000 --seed 42
    python murimsim/rl/train.py --resume checkpoints/limbic_v3/limbic_v3_final.zip --timesteps 1000000
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_N_LOG_LINES = 5   # total progress lines printed during the whole training run


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase 2 PPO survival agent.")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total_timesteps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--train-config", type=Path, default=Path("config/training.yaml"))
    parser.add_argument("--resume", type=Path, default=None, help="Resume from existing checkpoint (.zip)")
    args = parser.parse_args()

    # Lazy imports so the module is importable without torch/sb3 installed
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    except ImportError:
        print("ERROR: stable-baselines3 is required for training. Install with:")
        print("  pip install 'murimsim[rl]'")
        sys.exit(1)

    from murimsim.rl.env import SurvivalEnv

    with open(args.config) as f:
        world_config = yaml.safe_load(f)
    with open(args.train_config) as f:
        train_config = yaml.safe_load(f)

    world_config["world"]["seed"] = args.seed
    if "domain_randomization" in train_config:
        world_config["domain_randomization"] = train_config["domain_randomization"]
    total_timesteps = args.timesteps or int(train_config["training"]["total_timesteps"])
    checkpoint_dir = Path(train_config["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ppo_cfg = train_config["ppo"]

    logging.basicConfig(level=logging.WARNING)   # silence SB3 internal loggers

    env = SurvivalEnv(config=world_config, seed=args.seed)

    # ------------------------------------------------------------------
    # Progress callback — prints exactly _N_LOG_LINES times
    # ------------------------------------------------------------------
    class ProgressCallback(BaseCallback):
        def __init__(self, total: int, n_lines: int) -> None:
            super().__init__(verbose=0)
            self._total = total
            self._interval = max(1, total // n_lines)
            self._next_log = self._interval

        def _on_step(self) -> bool:
            if self.num_timesteps >= self._next_log:
                pct = 100.0 * self.num_timesteps / self._total
                print(f"  [{pct:5.1f}%]  {self.num_timesteps:>8,} / {self._total:,} steps")
                self._next_log += self._interval
            return True

    checkpoint_cb = CheckpointCallback(
        save_freq=int(train_config["training"]["checkpoint_freq"]),
        save_path=str(checkpoint_dir),
        name_prefix=checkpoint_dir.name,
        verbose=0,
    )

    if args.resume is not None:
        print(f"Resuming from {args.resume}")
        model = PPO.load(str(args.resume), env=env, device=str(ppo_cfg.get("device", "cpu")))
    else:
        model = PPO(
            policy=ppo_cfg["policy"],
            env=env,
            n_steps=int(ppo_cfg["n_steps"]),
            batch_size=int(ppo_cfg["batch_size"]),
            n_epochs=int(ppo_cfg["n_epochs"]),
            gamma=float(ppo_cfg["gamma"]),
            gae_lambda=float(ppo_cfg["gae_lambda"]),
            learning_rate=float(ppo_cfg["learning_rate"]),
            clip_range=float(ppo_cfg["clip_range"]),
            ent_coef=float(ppo_cfg["ent_coef"]),
            policy_kwargs={"net_arch": list(ppo_cfg["policy_kwargs"]["net_arch"])},
            device=str(ppo_cfg.get("device", "cpu")),
            seed=args.seed,
            verbose=0,
        )

    print(f"Training PPO  —  {total_timesteps:,} steps  seed={args.seed}")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, ProgressCallback(total_timesteps, _N_LOG_LINES)],
        reset_num_timesteps=args.resume is None,
    )

    final_path = checkpoint_dir / f"{checkpoint_dir.name}_final.zip"
    model.save(str(final_path))
    print(f"Done. Saved → {final_path}")


if __name__ == "__main__":
    main()
