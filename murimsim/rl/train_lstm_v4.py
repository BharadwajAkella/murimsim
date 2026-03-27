"""train_lstm_v4.py — LSTM v4: strength-as-priority reward.

Key changes vs v3:
  - REWARD_FOOD_GATHERED_SCALE 0.10 → 0.18  (gather = hunger relief)
  - PENALTY_EMPTY_INV_FOOD_NEARBY −0.06 → −0.08
  - REWARD_STRENGTH_SCALE = 0.008  (continuous per-step bonus × sum(resistances))
  - F-metric: stat_gain weight 0.1 → 0.3, food 0.3 → 0.1

Priority order: survival > strength > food
Target: gather >20%, agents building toward immunity.

Warm-starts from LSTM v3.

Usage (from project root):
    python murimsim/rl/train_lstm_v4.py
    python murimsim/rl/train_lstm_v4.py --timesteps 2000000 --seed 42
    python murimsim/rl/train_lstm_v4.py --no-warmstart
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_N_LOG_LINES = 15

_WARMSTART_CANDIDATES = [
    Path("checkpoints/limbic_lstm_v3/limbic_lstm_v3_final.zip"),
    Path("checkpoints/limbic_lstm_v2/limbic_lstm_v2_final.zip"),
]

_CHECKPOINT_DIR = Path("checkpoints/limbic_lstm_v4")
_RUN_PREFIX = "limbic_lstm_v4"


def main() -> None:
    parser = argparse.ArgumentParser(description="LSTM v4 RecurrentPPO — strength priority.")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--train-config", type=Path, default=Path("config/training.yaml"))
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--no-warmstart", action="store_true")
    args = parser.parse_args()

    try:
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except ImportError:
        print("ERROR: sb3-contrib is required. Install with: pip install sb3-contrib")
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
    curriculum_ramp_steps = int(lstm_cfg.get("curriculum_ramp_steps",
                                              p3c.get("curriculum_ramp_steps", CURRICULUM_RAMP_STEPS)))
    _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

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

    def make_env(cfg: dict, seed_offset: int, n_agents: int, ramp_steps: int):
        def _init() -> CombatEnv:
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

    warmstart_path: Path | None = None
    if not args.no_warmstart:
        for candidate in _WARMSTART_CANDIDATES:
            if candidate.exists():
                warmstart_path = candidate
                break

    if warmstart_path:
        print(f"Warm-starting from {warmstart_path}")
        try:
            import torch  # noqa: F401
            warm_model = RecurrentPPO.load(str(warmstart_path), device="cpu")
            src = warm_model.policy.state_dict()
            dst = model.policy.state_dict()
            compatible = {k: v for k, v in src.items()
                          if k in dst and dst[k].shape == v.shape}
            dst.update(compatible)
            model.policy.load_state_dict(dst)
            print(f"  Transferred {len(compatible)}/{len(dst)} layers.")
        except Exception as e:
            print(f"  Could not transfer weights ({e}). Training from scratch.")
    else:
        print("Training from scratch (--no-warmstart).")

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
        save_path=str(_CHECKPOINT_DIR),
        name_prefix=_RUN_PREFIX,
        verbose=0,
    )
    dashboard_cb = MetricsDashboardCallback(
        run_name=f"lstm_v4 seed={args.seed}",
        total_timesteps=total_timesteps,
    )

    print(f"Training RecurrentPPO LSTM v4  —  {total_timesteps:,} steps  "
          f"{n_envs} envs  {args.n_agents} agents/env  "
          f"lstm_hidden={lstm_hidden_size}  seed={args.seed}")
    print(f"  New: continuous strength bonus (REWARD_STRENGTH_SCALE=0.008)")
    print(f"  New: gather reward 0.10 → 0.18, empty-inv penalty −0.06 → −0.08")
    print(f"  Priority: survival > strength > food")
    print(f"  Combat curriculum: prob 0.2→1.0 over first {curriculum_ramp_steps:,} steps")
    print(f"  Checkpoint dir: {_CHECKPOINT_DIR}")
    print(f"  Dashboard: logs/dashboard_data.js\n")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, ProgressCallback(total_timesteps, _N_LOG_LINES), dashboard_cb],
    )

    final_path = _CHECKPOINT_DIR / f"{_RUN_PREFIX}_final.zip"
    model.save(str(final_path))
    print(f"\nDone. Saved → {final_path}")
    vec_env.close()


if __name__ == "__main__":
    main()
