"""train_multiagent.py — Phase 3b: multi-agent coexistence training.

10 agents share one World. No combat — agents compete passively for resources.
The policy learns to navigate a crowded grid, observe other agents, and
factor personal history signals (terrain_familiarity, reward_ema) into decisions.

Warm-starts from the Phase 3a checkpoint (limbic_v1b_final.zip).
Agents are individually the focal agent in round-robin, so the policy is
updated from all agents' experiences.

Usage (from project root):
    python murimsim/rl/train_multiagent.py
    python murimsim/rl/train_multiagent.py --timesteps 200000 --seed 42
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
    Path("checkpoints/limbic_v1b/limbic_v1b_final.zip"),
    Path("checkpoints/limbic_v1/limbic_v1_final.zip"),  # fallback to Phase 2
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3b multi-agent coexistence training.")
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

    from murimsim.rl.multi_env import MultiAgentEnv
    from murimsim.rl.train_multienv import _deep_merge_env_override

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)
    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f)

    if "domain_randomization" in train_cfg:
        base_cfg["domain_randomization"] = train_cfg["domain_randomization"]

    p3b = train_cfg.get("phase3b", {})
    total_timesteps = args.timesteps or int(p3b.get("total_timesteps", 200_000))
    checkpoint_dir = Path(p3b.get("checkpoint_dir", "checkpoints/limbic_v1c"))
    n_envs = int(p3b.get("n_envs", 4))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.WARNING)

    # Mix env types for diversity (same env configs as Phase 3a)
    envs_dir = Path("config/envs")
    env_variants = [
        envs_dir / "poison_heavy.yaml",
        envs_dir / "resource_scarce.yaml",
        envs_dir / "resource_dense.yaml",
        envs_dir / "resource_dense.yaml",  # extra dense copy — more diverse
    ]
    variant_cfgs = []
    for i, override_path in enumerate(env_variants[:n_envs]):
        cfg = _deep_merge_env_override(base_cfg, override_path)
        variant_cfgs.append(cfg)
    # Pad with base config if n_envs > len(env_variants)
    while len(variant_cfgs) < n_envs:
        variant_cfgs.append(base_cfg)

    def make_env(cfg, seed_offset: int, n_agents: int):
        def _init():
            return MultiAgentEnv(config=cfg, n_agents=n_agents, seed=args.seed + seed_offset)
        return _init

    env_fns = [make_env(cfg, i, args.n_agents) for i, cfg in enumerate(variant_cfgs)]
    vec_env = SubprocVecEnv(env_fns)

    # Warm-start: try Phase 3a first, fall back to Phase 2
    warmstart_path: Path | None = None
    if not args.no_warmstart:
        for candidate in _WARMSTART_CANDIDATES:
            if candidate.exists():
                warmstart_path = candidate
                break

    ppo_cfg = train_cfg["ppo"]

    if warmstart_path:
        print(f"Warm-starting from {warmstart_path}")
        # Phase 3b obs space (184) differs from Phase 2 (106) — must reinit policy
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
        print("  Note: obs space changed (106→184). Initializing fresh policy with Phase 3b obs.")
    else:
        if not args.no_warmstart:
            print("Warning: no warm-start checkpoint found — training from scratch.")
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
        save_freq=int(p3b.get("checkpoint_freq", 50_000)),
        save_path=str(checkpoint_dir),
        name_prefix="limbic_v1c",
        verbose=0,
    )

    print(f"Training PPO (Phase 3b)  —  {total_timesteps:,} steps  {n_envs} envs  "
          f"{args.n_agents} agents/env  seed={args.seed}")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, ProgressCallback(total_timesteps, _N_LOG_LINES)],
    )

    final_path = checkpoint_dir / "limbic_v1c_final.zip"
    model.save(str(final_path))
    print(f"Done. Saved → {final_path}")
    vec_env.close()


if __name__ == "__main__":
    main()
