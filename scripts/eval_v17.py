"""eval_v17.py — Comparative evaluation for v17 emergence pressure test.

Runs a frozen checkpoint over N episodes with enable_boss=True and
prints aggregate metrics, focusing on the v17 emergence signals:

  Boss combat:
    - ep_boss_killed, ep_boss_unique_attackers, ep_boss_damage_dealt
    - ep_damage_from_boss, ep_agents_killed_by_boss
  Survival / strength:
    - ep_lifespan, ep_avg_strength, ep_avg_power
  Stash semantics:
    - ep_stash_fill_rate, ep_stash_withdraw_rate
    - ep_pure_steals, ep_friendly_steals
    - ep_bank_withdrawals, ep_granary_withdrawals
  Combat / social:
    - ep_walk_away_count, ep_focal_collaborate_count
  Action distribution:
    - top action counts (e.g. train, attack, defend, gather)

Usage:
    python scripts/eval_v17.py --model checkpoints/limbic_lstm_v17/limbic_lstm_v17_final.zip
    python scripts/eval_v17.py --model checkpoints/limbic_lstm_v16/limbic_lstm_v16_final.zip --enable-boss
        (use the v16 model in the v17 boss-enabled environment for an A/B baseline)
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from murimsim.rl.multi_env import CombatEnv

CONFIG_PATH = Path("config/default.yaml")


def _load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _load_model(checkpoint: Path):
    from sb3_contrib import RecurrentPPO
    return RecurrentPPO.load(str(checkpoint), device="cpu")


def run_episodes(
    model_path: Path,
    n_episodes: int,
    n_agents: int,
    seed: int,
    enable_boss: bool,
    max_steps: int,
) -> list[dict]:
    """Run n_episodes deterministic episodes and return list of terminal info dicts."""
    cfg = _load_cfg()
    model = _load_model(model_path)
    results: list[dict] = []
    for ep in range(n_episodes):
        env = CombatEnv(
            config=cfg,
            n_agents=n_agents,
            seed=seed + ep,
            curriculum_ramp_steps=1,
            enable_boss=enable_boss,
        )
        obs, _ = env.reset(seed=seed + ep)
        lstm_state = None
        episode_starts = np.ones((1,), dtype=bool)
        steps = 0
        last_info: dict = {}
        while steps < max_steps:
            action, lstm_state = model.predict(
                obs, state=lstm_state, episode_start=episode_starts, deterministic=True
            )
            episode_starts = np.zeros((1,), dtype=bool)
            obs, reward, terminated, truncated, info = env.step(int(action))
            steps += 1
            last_info = info
            if terminated or truncated:
                break
        last_info.setdefault("ep_action_counts", dict(env._ep_action_counts))
        results.append(last_info)
    return results


def _summarise(values: list[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{mean(values):.3f} ± {stdev(values):.3f}"


def aggregate(results: list[dict]) -> dict[str, str]:
    """Aggregate scalar metrics across episodes."""
    keys = [
        "ep_lifespan", "ep_avg_strength", "ep_avg_power", "ep_final_power",
        "ep_stash_fill_rate", "ep_stash_withdraw_rate", "ep_avg_dist_from_stash",
        "ep_revisit_entropy", "ep_group_persistence",
        "ep_pure_steals", "ep_friendly_steals",
        "ep_bank_withdrawals", "ep_granary_withdrawals",
        "ep_walk_away_count", "ep_avg_flee_strength_diff", "ep_avg_flee_health",
        "ep_focal_collaborate_count", "ep_reproductions", "ep_deaths_by_age",
        "ep_boss_killed", "ep_boss_attacks_landed", "ep_boss_damage_dealt",
        "ep_damage_from_boss", "ep_agents_killed_by_boss", "ep_boss_unique_attackers",
        "ep_betrayal_count", "ep_friendly_flank_count",
        "ep_focal_max_affinity", "ep_focal_min_affinity",
    ]
    out: dict[str, str] = {}
    for k in keys:
        vals = [float(r.get(k, 0)) for r in results if k in r]
        out[k] = _summarise(vals)
    # Action counts: aggregate the top 6 by total
    action_totals: dict[str, int] = defaultdict(int)
    for r in results:
        for k, v in r.get("ep_action_counts", {}).items():
            action_totals[k] += int(v)
    top_actions = sorted(action_totals.items(), key=lambda kv: -kv[1])[:6]
    for name, total in top_actions:
        out[f"action_{name}"] = f"{total / max(1, len(results)):.1f}/ep"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--agents", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--enable-boss", action="store_true",
                        help="Enable boss monster (default: enabled). "
                             "Use --no-boss to disable.")
    parser.add_argument("--no-boss", dest="enable_boss", action="store_false")
    parser.set_defaults(enable_boss=True)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    print(f"Eval {args.model.name}", flush=True)
    print(f"  episodes={args.episodes}  agents={args.agents}  seed={args.seed}  "
          f"max_steps={args.max_steps}  boss={'on' if args.enable_boss else 'off'}",
          flush=True)
    results = run_episodes(
        model_path=args.model,
        n_episodes=args.episodes,
        n_agents=args.agents,
        seed=args.seed,
        enable_boss=args.enable_boss,
        max_steps=args.max_steps,
    )
    summary = aggregate(results)

    print("\n── Summary (mean ± std across episodes) ─────────────")
    for k, v in summary.items():
        print(f"  {k:36s}  {v}")
    print()


if __name__ == "__main__":
    main()
