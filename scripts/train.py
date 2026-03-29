"""train.py — Unified RecurrentPPO training entry point.

Replaces all train_lstm_vN.py scripts. Pass --run-name to start a new named
run; warm-start is auto-detected from the latest checkpoint unless overridden.

Usage (from project root):
    python scripts/train.py --run-name lstm_v4
    python scripts/train.py --run-name lstm_v4 --timesteps 2000000 --seed 42
    python scripts/train.py --run-name lstm_v5 --warmstart checkpoints/limbic_lstm_v4/limbic_lstm_v4_final.zip
    python scripts/train.py --run-name scratch_test --no-warmstart
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

_N_LOG_LINES = 15


def _latest_checkpoint() -> Path | None:
    """Return the most recently modified final.zip across all checkpoint dirs."""
    candidates = sorted(
        Path("checkpoints").glob("*/*_final.zip"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified RecurrentPPO training.")
    parser.add_argument("--run-name", required=True,
                        help="Name for this run, e.g. 'lstm_v4'. "
                             "Checkpoint dir: checkpoints/limbic_{run_name}/")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override total timesteps (default: from training.yaml)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--train-config", type=Path, default=Path("config/training.yaml"))
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--warmstart", type=Path, default=None,
                        help="Explicit warmstart checkpoint path. "
                             "Omit to auto-detect latest; use --no-warmstart to train from scratch.")
    parser.add_argument("--no-warmstart", action="store_true",
                        help="Train from scratch instead of warm-starting.")
    args = parser.parse_args()

    try:
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except ImportError:
        print("ERROR: sb3-contrib required. pip install sb3-contrib")
        sys.exit(1)

    from murimsim.rl.multi_env import CombatEnv, CURRICULUM_RAMP_STEPS
    from murimsim.rl.train_multienv import _deep_merge_env_override
    from murimsim.rl.metrics_callback import MetricsDashboardCallback

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)
    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f)

    if "domain_randomization" in train_cfg:
        base_cfg["domain_randomization"] = train_cfg["domain_randomization"]

    p3c = train_cfg.get("phase3c", {})
    lstm_cfg = train_cfg.get("lstm", {})

    total_timesteps = args.timesteps or int(lstm_cfg.get("total_timesteps", 2_000_000))
    n_envs = int(lstm_cfg.get("n_envs", p3c.get("n_envs", 4)))
    lstm_hidden_size = int(lstm_cfg.get("lstm_hidden_size", 64))
    curriculum_ramp_steps = int(lstm_cfg.get(
        "curriculum_ramp_steps",
        p3c.get("curriculum_ramp_steps", CURRICULUM_RAMP_STEPS),
    ))

    checkpoint_dir = Path(f"checkpoints/limbic_{args.run_name}")
    run_prefix = f"limbic_{args.run_name}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.WARNING)

    # Build per-env configs from variant overrides
    envs_dir = Path("config/envs")
    env_variants = [
        envs_dir / "poison_heavy.yaml",
        envs_dir / "resource_scarce.yaml",
        envs_dir / "resource_dense.yaml",
        envs_dir / "dense_patch.yaml",
    ]
    variant_cfgs: list[dict] = []
    for override_path in env_variants[:n_envs]:
        variant_cfgs.append(_deep_merge_env_override(base_cfg, override_path))
    while len(variant_cfgs) < n_envs:
        variant_cfgs.append(base_cfg)

    def make_env(cfg: dict, seed_offset: int, n_agents: int, ramp_steps: int):
        def _init() -> CombatEnv:
            return CombatEnv(
                config=cfg,
                n_agents=n_agents,
                seed=args.seed + seed_offset,
                curriculum_ramp_steps=ramp_steps,
            )
        return _init

    vec_env = SubprocVecEnv([
        make_env(cfg, i, args.n_agents, curriculum_ramp_steps)
        for i, cfg in enumerate(variant_cfgs)
    ])

    ppo_cfg = train_cfg["ppo"]
    n_steps = int(lstm_cfg.get("n_steps", ppo_cfg["n_steps"]))
    batch_size = int(lstm_cfg.get("batch_size", ppo_cfg["batch_size"]))

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=vec_env,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=int(ppo_cfg["n_epochs"]),
        gamma=float(ppo_cfg["gamma"]),
        gae_lambda=float(ppo_cfg["gae_lambda"]),
        learning_rate=float(ppo_cfg["learning_rate"]),
        clip_range=float(ppo_cfg["clip_range"]),
        ent_coef=float(ppo_cfg["ent_coef"]),
        policy_kwargs={
            "net_arch": list(ppo_cfg["policy_kwargs"]["net_arch"]),
            "lstm_hidden_size": lstm_hidden_size,
            "n_lstm_layers": int(lstm_cfg.get("n_lstm_layers", 1)),
            "shared_lstm": bool(lstm_cfg.get("shared_lstm", False)),
        },
        device=str(ppo_cfg.get("device", "cpu")),
        seed=args.seed,
        verbose=0,
    )

    # Warm-start: explicit path > auto-detect latest > scratch
    warmstart_path: Path | None = None
    if not args.no_warmstart:
        if args.warmstart:
            warmstart_path = args.warmstart
        else:
            detected = _latest_checkpoint()
            # Don't warm-start from ourselves (if re-running same run name)
            if detected and run_prefix not in detected.parts[-2]:
                warmstart_path = detected

    if warmstart_path and warmstart_path.exists():
        print(f"Warm-starting from {warmstart_path}", flush=True)
        try:
            warm_model = RecurrentPPO.load(str(warmstart_path), device="cpu")
            src = warm_model.policy.state_dict()
            dst = model.policy.state_dict()
            compatible = {k: v for k, v in src.items()
                          if k in dst and dst[k].shape == v.shape}
            dst.update(compatible)
            model.policy.load_state_dict(dst)
            print(f"  Transferred {len(compatible)}/{len(dst)} layers.", flush=True)
        except Exception as e:
            print(f"  Weight transfer failed ({e}). Training from scratch.", flush=True)
    elif args.no_warmstart:
        print("Training from scratch (--no-warmstart).", flush=True)
    else:
        print("No prior checkpoint found — training from scratch.", flush=True)

    class ProgressCallback(BaseCallback):
        def __init__(self, total: int, n_lines: int) -> None:
            super().__init__(verbose=0)
            self._total = total
            self._interval = max(1, total // n_lines)
            self._next_log = self._interval

        def _on_step(self) -> bool:
            if self.num_timesteps >= self._next_log:
                pct = 100.0 * self.num_timesteps / self._total
                print(f"  [{pct:5.1f}%]  {self.num_timesteps:>8,} / {self._total:,} steps",
                      flush=True)
                self._next_log += self._interval
            return True

    checkpoint_cb = CheckpointCallback(
        save_freq=int(lstm_cfg.get("checkpoint_freq", 100_000)),
        save_path=str(checkpoint_dir),
        name_prefix=run_prefix,
        verbose=0,
    )
    dashboard_cb = MetricsDashboardCallback(
        run_name=f"{args.run_name} seed={args.seed}",
        total_timesteps=total_timesteps,
    )

    print(f"\nTraining RecurrentPPO  run={args.run_name}  {total_timesteps:,} steps  "
          f"{n_envs} envs  {args.n_agents} agents/env  "
          f"lstm_hidden={lstm_hidden_size}  seed={args.seed}", flush=True)
    print(f"  Curriculum: combat prob 0.2 → 1.0 over first {curriculum_ramp_steps:,} steps", flush=True)
    print(f"  Checkpoint dir: {checkpoint_dir}", flush=True)
    print(f"  Dashboard: logs/dashboard_data.js\n", flush=True)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, ProgressCallback(total_timesteps, _N_LOG_LINES), dashboard_cb],
    )

    final_path = checkpoint_dir / f"{run_prefix}_final.zip"
    model.save(str(final_path))
    print(f"\nDone. Saved → {final_path}", flush=True)
    vec_env.close()


if __name__ == "__main__":
    main()
