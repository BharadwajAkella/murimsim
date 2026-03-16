"""train_combat.py — Phase 3c: multi-agent combat training with curriculum.

Agents learn fight/flight in a multi-agent world. Combat starts disabled (20%)
and ramps to fully enabled over 300k steps so foraging skills are not forgotten.

Warm-starts from Phase 3b checkpoint (limbic_v1c_final.zip).

Usage (from project root):
    python murimsim/rl/train_combat.py
    python murimsim/rl/train_combat.py --timesteps 500000 --seed 42
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_N_LOG_LINES = 5

_WARMSTART_CANDIDATES = [
    Path("checkpoints/limbic_v1c/limbic_v1c_final.zip"),
    Path("checkpoints/limbic_v1b/limbic_v1b_final.zip"),
    Path("checkpoints/limbic_v1/limbic_v1_final.zip"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3c combat PPO training.")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--train-config", type=Path, default=Path("config/training.yaml"))
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--no-warmstart", action="store_true")
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except ImportError:
        print("ERROR: stable-baselines3 is required. Install with: pip install 'murimsim[rl]'")
        sys.exit(1)

    from murimsim.rl.multi_env import CombatEnv, CURRICULUM_RAMP_STEPS
    from murimsim.rl.train_multienv import _deep_merge_env_override

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)
    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f)

    if "domain_randomization" in train_cfg:
        base_cfg["domain_randomization"] = train_cfg["domain_randomization"]

    p3c = train_cfg.get("phase3c", {})
    total_timesteps = args.timesteps or int(p3c.get("total_timesteps", 500_000))
    checkpoint_dir = Path(p3c.get("checkpoint_dir", "checkpoints/limbic_v2"))
    n_envs = int(p3c.get("n_envs", 4))
    curriculum_ramp_steps = int(p3c.get("curriculum_ramp_steps", CURRICULUM_RAMP_STEPS))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.WARNING)

    envs_dir = Path("config/envs")
    env_variants = [
        envs_dir / "poison_heavy.yaml",
        envs_dir / "resource_scarce.yaml",
        envs_dir / "resource_dense.yaml",
        envs_dir / "resource_dense.yaml",
    ]
    variant_cfgs = []
    for override_path in env_variants[:n_envs]:
        variant_cfgs.append(_deep_merge_env_override(base_cfg, override_path))
    while len(variant_cfgs) < n_envs:
        variant_cfgs.append(base_cfg)

    def make_env(cfg, seed_offset: int, n_agents: int, ramp_steps: int):
        def _init():
            return CombatEnv(
                config=cfg,
                n_agents=n_agents,
                seed=args.seed + seed_offset,
                curriculum_ramp_steps=ramp_steps,
            )
        return _init

    env_fns = [
        make_env(cfg, i, args.n_agents, curriculum_ramp_steps)
        for i, cfg in enumerate(variant_cfgs)
    ]
    vec_env = SubprocVecEnv(env_fns)

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

    # Phase 3c obs space (184) matches Phase 3b — warm-start is clean
    warmstart_path: Path | None = None
    if not args.no_warmstart:
        for candidate in _WARMSTART_CANDIDATES:
            if candidate.exists():
                warmstart_path = candidate
                break

    if warmstart_path:
        print(f"Warm-starting feature weights from {warmstart_path}")
        try:
            import torch
            warm_model = PPO.load(str(warmstart_path), device="cpu")
            # Copy only layers with matching shapes (action head changes 7→9 due to ATTACK/DEFEND)
            src = warm_model.policy.state_dict()
            dst = model.policy.state_dict()
            compatible = {k: v for k, v in src.items() if k in dst and dst[k].shape == v.shape}
            dst.update(compatible)
            model.policy.load_state_dict(dst)
            n_total, n_copied = len(dst), len(compatible)
            print(f"  Transferred {n_copied}/{n_total} layers (skipped {n_total - n_copied} with shape mismatch).")
        except Exception as e:
            print(f"  Could not transfer weights ({e}). Training from scratch.")

    class ProgressCallback(BaseCallback):
        def __init__(self, total: int, n_lines: int) -> None:
            super().__init__(verbose=0)
            self._total = total
            self._interval = max(1, total // n_lines)
            self._next_log = self._interval

        def _on_step(self) -> bool:
            if self.num_timesteps >= self._next_log:
                pct = 100.0 * self.num_timesteps / self._total
                # Show curriculum progress too
                print(f"  [{pct:5.1f}%]  {self.num_timesteps:>8,} / {self._total:,} steps")
                self._next_log += self._interval
            return True

    checkpoint_cb = CheckpointCallback(
        save_freq=int(p3c.get("checkpoint_freq", 100_000)),
        save_path=str(checkpoint_dir),
        name_prefix="limbic_v2",
        verbose=0,
    )

    print(f"Training PPO (Phase 3c)  —  {total_timesteps:,} steps  {n_envs} envs  "
          f"{args.n_agents} agents/env  seed={args.seed}")
    print(f"  Combat curriculum: prob 0.2→1.0 over first {curriculum_ramp_steps:,} steps")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, ProgressCallback(total_timesteps, _N_LOG_LINES)],
    )

    final_path = checkpoint_dir / "limbic_v2_final.zip"
    model.save(str(final_path))
    print(f"Done. Saved → {final_path}")
    vec_env.close()


if __name__ == "__main__":
    main()
