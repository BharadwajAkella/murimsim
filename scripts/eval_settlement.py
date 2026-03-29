"""eval_settlement.py — Baseline settlement metrics comparison.

Runs LSTM v8 (frozen) on the default map and the dense-patch map,
collecting settlement metrics over N episodes per map.
Prints a side-by-side table so we have a before/after baseline
before training v9.

Usage (from project root):
    python scripts/eval_settlement.py
    python scripts/eval_settlement.py --episodes 20 --ticks 600
    python scripts/eval_settlement.py --model checkpoints/limbic_lstm_v8/limbic_lstm_v8_final.zip
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from murimsim.rl.multi_env import CombatEnv
from murimsim.rl.train_multienv import _deep_merge_env_override

CONFIG_PATH = Path("config/default.yaml")
DENSE_PATCH_PATH = Path("config/envs/dense_patch.yaml")
DEFAULT_MODEL = Path("checkpoints/limbic_lstm_v8/limbic_lstm_v8_final.zip")

SETTLEMENT_KEYS = [
    "ep_stash_fill_rate",
    "ep_stash_withdraw_rate",
    "ep_avg_dist_from_stash",
    "ep_revisit_entropy",
    "ep_group_persistence",
    "ep_lifespan",
]

LABELS = {
    "ep_stash_fill_rate":     "Stash fill rate       (deposited/gathered)",
    "ep_stash_withdraw_rate": "Stash withdraw rate   (withdrawn/deposited)",
    "ep_avg_dist_from_stash": "Avg dist from stash   (Chebyshev tiles)",
    "ep_revisit_entropy":     "Revisit entropy       (lower = more settled)",
    "ep_group_persistence":   "Group persistence     (member-ticks/formation)",
    "ep_lifespan":            "Lifespan              (steps)",
}


def _run_episodes(
    config: dict,
    model,
    n_episodes: int,
    max_ticks: int,
    n_agents: int,
    base_seed: int,
    is_recurrent: bool,
) -> dict[str, list[float]]:
    """Run model for n_episodes, collecting settlement metrics from each terminal info."""
    results: dict[str, list[float]] = {k: [] for k in SETTLEMENT_KEYS}

    for ep in range(n_episodes):
        seed = base_seed + ep
        env = CombatEnv(config=config, n_agents=n_agents, seed=seed)
        obs, _ = env.reset(seed=seed)
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        for _ in range(max_ticks):
            if model is not None:
                if is_recurrent:
                    action, lstm_states = model.predict(
                        obs[np.newaxis],
                        state=lstm_states,
                        episode_start=episode_starts,
                        deterministic=True,
                    )
                    action = int(action[0])
                    episode_starts = np.zeros((1,), dtype=bool)
                else:
                    action, _ = model.predict(obs, deterministic=True)
                    action = int(action)
            else:
                action = env.action_space.sample()

            obs, _, terminated, _, info = env.step(action)
            if terminated:
                for k in SETTLEMENT_KEYS:
                    if k in info:
                        results[k].append(float(info[k]))
                episode_starts = np.ones((1,), dtype=bool)
                lstm_states = None
                break

        env.close()
        sys.stdout.write(f"\r  episode {ep + 1}/{n_episodes}")
        sys.stdout.flush()

    print()
    return results


def _mean(vals: list[float]) -> str:
    if not vals:
        return "  n/a  "
    return f"{np.mean(vals):7.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Settlement metrics baseline eval.")
    parser.add_argument("--episodes", type=int, default=15,
                        help="Episodes per map (default: 15)")
    parser.add_argument("--ticks", type=int, default=800,
                        help="Max steps per episode (default: 800)")
    parser.add_argument("--agents", type=int, default=6,
                        help="Number of agents (default: 6)")
    parser.add_argument("--seed", type=int, default=100,
                        help="Base random seed (default: 100)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="Checkpoint to evaluate")
    args = parser.parse_args()

    # Load model
    model = None
    is_recurrent = False
    if args.model.exists():
        try:
            from sb3_contrib import RecurrentPPO
            model = RecurrentPPO.load(str(args.model), device="cpu")
            is_recurrent = True
            print(f"Loaded RecurrentPPO: {args.model.name}")
        except Exception:
            try:
                from stable_baselines3 import PPO
                model = PPO.load(str(args.model), device="cpu")
                print(f"Loaded PPO: {args.model.name}")
            except Exception as e:
                print(f"[WARN] Could not load model ({e}). Using random actions.", file=sys.stderr)
    else:
        print(f"[WARN] Model not found: {args.model}. Using random actions.", file=sys.stderr)

    with open(CONFIG_PATH) as f:
        base_cfg = yaml.safe_load(f)

    dense_cfg = _deep_merge_env_override(base_cfg, DENSE_PATCH_PATH)

    print(f"\nRunning {args.episodes} episodes × 2 maps  ({args.agents} agents, max {args.ticks} steps)\n")

    print("Default map:")
    default_results = _run_episodes(
        base_cfg, model, args.episodes, args.ticks, args.agents, args.seed, is_recurrent
    )

    print("\nDense-patch map:")
    dense_results = _run_episodes(
        dense_cfg, model, args.episodes, args.ticks, args.agents, args.seed, is_recurrent
    )

    # Print comparison table
    col_w = 44
    print(f"\n{'Metric':<{col_w}}  {'Default':>9}  {'Dense-patch':>11}  {'Δ':>8}")
    print("-" * (col_w + 34))
    for k in SETTLEMENT_KEYS:
        label = LABELS[k]
        dv = np.mean(default_results[k]) if default_results[k] else float("nan")
        pv = np.mean(dense_results[k]) if dense_results[k] else float("nan")
        delta = pv - dv
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        print(f"  {label:<{col_w - 2}}  {dv:>9.3f}  {pv:>11.3f}  {arrow}{abs(delta):>6.3f}")

    print()
    # Settlement verdict
    entropy_improved = (
        dense_results["ep_revisit_entropy"] and default_results["ep_revisit_entropy"] and
        np.mean(dense_results["ep_revisit_entropy"]) < np.mean(default_results["ep_revisit_entropy"])
    )
    fill_improved = (
        dense_results["ep_stash_fill_rate"] and default_results["ep_stash_fill_rate"] and
        np.mean(dense_results["ep_stash_fill_rate"]) > np.mean(default_results["ep_stash_fill_rate"])
    )
    if entropy_improved and fill_improved:
        print("✅ Settlement signal detected: lower entropy + higher stash fill on dense-patch map.")
        print("   Shared-stash improvements may amplify this further.")
    elif entropy_improved or fill_improved:
        print("⚠️  Partial settlement signal. One metric improved; shared-stash needed to reinforce.")
    else:
        print("❌ No settlement signal yet. Dense-patch map alone is not sufficient.")
        print("   Proceed with shared-stash reward improvements before v9 training.")


if __name__ == "__main__":
    main()
