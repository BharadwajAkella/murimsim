"""train_multienv.py — Phase 3a: multi-environment PPO training.

Trains a single shared policy across 3 resource environments simultaneously:
  - poison_heavy   (3 copies): large poison clusters, learn to read resistance vs danger
  - resource_scarce (3 copies): sparse food, learn to range further when obs shows low density
  - resource_dense  (2 copies): abundant food, learn camping when obs shows high density

The policy receives NO environment label — specialization emerges purely from
the different obs values each env produces. After training the single policy
handles all three environments with appropriate behaviour.

Usage (from project root):
    python murimsim/rl/train_multienv.py
    python murimsim/rl/train_multienv.py --timesteps 300000 --seed 42
    python murimsim/rl/train_multienv.py --no-warmstart  # train from scratch
"""
from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_N_LOG_LINES = 5


def _deep_merge_env_override(base_cfg: dict[str, Any], override_path: Path) -> dict[str, Any]:
    """Apply a partial env override YAML on top of base_cfg.

    Override format::

        resource_overrides:
          - id: food
            spawn_density: 0.02
            regen_ticks: 400

    Resources are matched by ``id``; only listed keys are updated.
    A ``world:`` key in the override directly updates the world section.
    """
    with open(override_path) as f:
        override = yaml.safe_load(f)

    cfg = copy.deepcopy(base_cfg)

    for res_override in override.get("resource_overrides", []):
        rid = str(res_override["id"])
        for res in cfg["resources"]:
            if res["id"] == rid:
                res.update({k: v for k, v in res_override.items() if k != "id"})
                break

    if "world" in override:
        cfg["world"].update(override["world"])

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3a multi-environment PPO training.")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--train-config", type=Path, default=Path("config/training.yaml"))
    parser.add_argument(
        "--warmstart",
        type=Path,
        default=Path("checkpoints/limbic_v1/limbic_v1_final.zip"),
        help="Checkpoint to warm-start from (must have matching obs/action space).",
    )
    parser.add_argument(
        "--no-warmstart",
        action="store_true",
        help="Train from scratch instead of warm-starting.",
    )
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except ImportError:
        print("ERROR: stable-baselines3 is required. Install with: pip install 'murimsim[rl]'")
        sys.exit(1)

    from murimsim.rl.env import SurvivalEnv

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)
    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f)

    # Merge domain randomization from training config
    if "domain_randomization" in train_cfg:
        base_cfg["domain_randomization"] = train_cfg["domain_randomization"]

    envs_dir = Path("config/envs")
    env_variants: list[tuple[Path, int]] = [
        (envs_dir / "poison_heavy.yaml",    3),
        (envs_dir / "resource_scarce.yaml", 3),
        (envs_dir / "resource_dense.yaml",  2),
    ]

    # Build per-variant configs
    variant_cfgs: list[dict[str, Any]] = []
    for override_path, n_copies in env_variants:
        cfg = _deep_merge_env_override(base_cfg, override_path)
        variant_cfgs.extend([cfg] * n_copies)

    p3a = train_cfg.get("phase3a", {})
    total_timesteps = args.timesteps or int(p3a.get("total_timesteps", 300_000))
    checkpoint_dir = Path(p3a.get("checkpoint_dir", "checkpoints/limbic_v1b"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.WARNING)

    def make_env(cfg: dict[str, Any], seed_offset: int):
        """Return a no-arg factory for SubprocVecEnv."""
        def _init() -> SurvivalEnv:
            return SurvivalEnv(config=cfg, seed=args.seed + seed_offset)
        return _init

    env_fns = [make_env(cfg, i) for i, cfg in enumerate(variant_cfgs)]
    vec_env = SubprocVecEnv(env_fns)

    warmstart_path = args.warmstart if not args.no_warmstart else None
    if warmstart_path and warmstart_path.exists():
        print(f"Warm-starting from {warmstart_path}")
        model = PPO.load(str(warmstart_path), env=vec_env, device="cpu")
    else:
        if warmstart_path and not warmstart_path.exists():
            print(f"Warning: warm-start checkpoint not found at {warmstart_path}, training from scratch.")
        ppo_cfg = train_cfg["ppo"]
        model = PPO(
            policy=ppo_cfg["policy"],
            env=vec_env,
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
        save_freq=int(p3a.get("checkpoint_freq", 100_000)),
        save_path=str(checkpoint_dir),
        name_prefix="limbic_v1b",
        verbose=0,
    )

    n_envs = len(env_fns)
    print(f"Training PPO (Phase 3a)  —  {total_timesteps:,} steps  {n_envs} envs  seed={args.seed}")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, ProgressCallback(total_timesteps, _N_LOG_LINES)],
    )

    final_path = checkpoint_dir / "limbic_v1b_final.zip"
    model.save(str(final_path))
    print(f"Done. Saved → {final_path}")
    vec_env.close()


if __name__ == "__main__":
    main()
