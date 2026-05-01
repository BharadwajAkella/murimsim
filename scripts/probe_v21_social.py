"""Compare social signals under boss-on vs boss-off for v21 checkpoints.

Tracks per-step:
  * mean pairwise Chebyshev distance between alive agents
  * % of steps with ≥1 agent-pair within affinity-proximity radius (=2)
  * cumulative action counts: gather, share, collaborate, attack_*
  * mean inventory.total per agent (used as proxy for legacy_stash bequest potential)

Usage:
    python -m scripts.probe_v21_social --checkpoint <ckpt> [--enable-boss] [--enable-carry-cost] [--steps N]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations

import numpy as np
import torch
import yaml

from murimsim.rl.ippo_env import IPPOEnv
from murimsim.rl.multi_env import AFFINITY_PROXIMITY_RADIUS
from scripts.eval_ippo import load_policy


def chebyshev(a, b) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def probe(checkpoint, enable_boss, enable_carry_cost, steps=5000, n_envs=4, n_agents=4, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    with open("config/default.yaml") as f:
        cfg = yaml.safe_load(f)
    policy, _, is_recurrent = load_policy(checkpoint, device="cpu")

    envs = [
        IPPOEnv(
            config=cfg, n_agents=n_agents, seed=seed + i, curriculum_ramp_steps=0,
            enable_boss=enable_boss, enable_carry_cost=enable_carry_cost,
        )
        for i in range(n_envs)
    ]
    obs_list, mask_list = [], []
    for i, env in enumerate(envs):
        o, info = env.reset_all(seed=seed + i)
        obs_list.append(o); mask_list.append(info["action_masks"])
    obs = np.stack(obs_list, axis=0)
    masks = np.stack(mask_list, axis=0)

    hidden = None
    if is_recurrent:
        hidden = (
            torch.zeros(1, n_envs * n_agents, policy.hidden_dim),
            torch.zeros(1, n_envs * n_agents, policy.hidden_dim),
        )

    pairwise_dist_sum = 0.0
    pairwise_dist_n = 0
    steps_with_close_pair = 0
    total_steps = 0
    inventory_sum = 0.0
    inventory_n = 0
    boss_adjacent_steps = 0   # any agent within radius 2 of a boss
    action_counts = Counter()

    for _ in range(steps):
        flat_obs = torch.tensor(obs.reshape(n_envs * n_agents, -1), dtype=torch.float32)
        flat_masks = torch.tensor(masks.reshape(n_envs * n_agents, -1), dtype=torch.bool)
        with torch.no_grad():
            if is_recurrent:
                actions_t, _, _, hidden = policy.act(flat_obs, flat_masks, hidden=hidden, deterministic=True)
            else:
                actions_t, _, _ = policy.act(flat_obs, flat_masks, deterministic=True)
            actions = actions_t.cpu().numpy().reshape(n_envs, n_agents)

        for ei, env in enumerate(envs):
            obs_e, _, _, _, info = env.step_all(actions[ei])
            obs[ei] = obs_e
            masks[ei] = info["action_masks_post"]

            alive = [a for a in env._agents if a.alive]
            if len(alive) >= 2:
                close = False
                for a, b in combinations(alive, 2):
                    d = chebyshev(a.position, b.position)
                    pairwise_dist_sum += d
                    pairwise_dist_n += 1
                    if d <= AFFINITY_PROXIMITY_RADIUS:
                        close = True
                if close:
                    steps_with_close_pair += 1
            for a in alive:
                inventory_sum += a.inventory.total()
                inventory_n += 1

            # boss proximity
            for boss in [m for m in env._monsters.all() if m.kind == "boss" and m.alive]:
                for a in alive:
                    if chebyshev(a.position, boss.position) <= 2:
                        boss_adjacent_steps += 1
                        break

            total_steps += 1

        # accumulate action counts (sample one env's _ep_action_counts at end is enough)
    for env in envs:
        for k, v in env._ep_action_counts.items():
            action_counts[k] += v

    return {
        "mean_pairwise_distance": round(pairwise_dist_sum / max(1, pairwise_dist_n), 2),
        "pct_steps_with_close_pair": round(100 * steps_with_close_pair / max(1, total_steps), 1),
        "mean_inventory_per_agent": round(inventory_sum / max(1, inventory_n), 2),
        "pct_envs_x_steps_agent_near_boss": round(100 * boss_adjacent_steps / max(1, total_steps), 2),
        "action_counts_top": dict(action_counts.most_common(12)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--enable-boss", action="store_true")
    p.add_argument("--enable-carry-cost", action="store_true")
    p.add_argument("--steps", type=int, default=5000)
    args = p.parse_args()
    print(json.dumps(probe(
        args.checkpoint, args.enable_boss, args.enable_carry_cost, args.steps
    ), indent=2))


if __name__ == "__main__":
    main()
