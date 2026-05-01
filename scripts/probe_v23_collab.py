"""scripts/probe_v23_collab.py — diagnose COLLABORATE action usage in v23.

Counts:
  - action distribution across all agent steps
  - COLLABORATE attempts vs successes (group formations)
  - share-food attempts vs successes
  - mask: how often is COLLABORATE even *available*?
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

from murimsim.actions import Action, N_ACTIONS_PHASE6_QI
from murimsim.rl.multi_env import OBS_TOTAL_SIZE
from scripts.eval_ippo import load_policy
from scripts.train_ippo import build_envs, collect_initial_state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--n-agents", type=int, default=10)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--arena-mix", type=str, default="base:1,arena_minion:2,arena_boss:1")
    ap.add_argument("--enable-carry-cost", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    obs_dim = OBS_TOTAL_SIZE
    n_actions = N_ACTIONS_PHASE6_QI

    policy, _, is_recurrent = load_policy(args.checkpoint, device=device)
    envs = build_envs(
        cfg, args.n_envs, args.n_agents, args.seed,
        enable_boss=False, enable_carry_cost=args.enable_carry_cost,
        arena_mix=args.arena_mix,
    )
    obs, action_masks, active_mask = collect_initial_state(envs, args.seed)

    if is_recurrent:
        B = args.n_envs * args.n_agents
        carried_h, carried_c = policy.initial_hidden(B, device=device)

    action_counts: Counter[int] = Counter()
    collab_available_steps = 0
    collab_chosen_steps = 0
    total_active_steps = 0

    # snapshot group counts and help events from each env tick
    max_groups_seen = 0
    total_groups_formed = 0
    total_groups_disbanded = 0  # via _ep_groups_formed delta at reset
    help_events_seen = 0

    for _t in range(args.steps):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).reshape(-1, obs_dim)
            mask_t = torch.as_tensor(action_masks, dtype=torch.bool, device=device).reshape(-1, n_actions)
            if is_recurrent:
                action_flat, _, _, (carried_h, carried_c) = policy.act(
                    obs_t, mask_t, (carried_h, carried_c), deterministic=True
                )
            else:
                action_flat, _, _ = policy.act(obs_t, mask_t, deterministic=True)
            actions_np = action_flat.cpu().numpy().reshape(args.n_envs, args.n_agents)

        # mask analysis BEFORE step
        flat_active = active_mask.reshape(-1)
        flat_collab_avail = action_masks.reshape(-1, n_actions)[:, Action.COLLABORATE]
        flat_actions = actions_np.reshape(-1)
        for slot in range(flat_active.size):
            if not flat_active[slot]:
                continue
            total_active_steps += 1
            action_counts[int(flat_actions[slot])] += 1
            if flat_collab_avail[slot]:
                collab_available_steps += 1
                if flat_actions[slot] == int(Action.COLLABORATE):
                    collab_chosen_steps += 1

        for e_i, env in enumerate(envs):
            o, r, term, trunc, info = env.step_all(actions_np[e_i])
            obs[e_i] = o
            action_masks[e_i] = info["action_masks_post"]
            env_next_active = np.array([a.alive for a in env._agents], dtype=bool)
            active_mask[e_i] = env_next_active
            cur_groups = [g for g in env._groups if len(g) >= 2]
            max_groups_seen = max(max_groups_seen, len(cur_groups))
            help_events_seen = max(
                help_events_seen,
                sum(len(d) for d in env._help_received.values())
            )
            total_groups_formed = max(total_groups_formed, env._ep_groups_formed)
            if not env_next_active.any():
                reset_seed = args.seed + e_i + 1_000_000 * (_t + 1)
                o, info = env.reset_all(seed=reset_seed)
                obs[e_i] = o
                action_masks[e_i] = info["action_masks"]
                active_mask[e_i] = info["active_mask"]
                if is_recurrent:
                    # zero hidden for this env's slots
                    start = e_i * args.n_agents
                    end = start + args.n_agents
                    carried_h[:, start:end, :] = 0.0
                    carried_c[:, start:end, :] = 0.0

    print(f"\n=== Probe: {args.checkpoint} ({'recurrent' if is_recurrent else 'ff'}) ===")
    print(f"total_active_steps         = {total_active_steps}")
    print(f"collab_available_steps     = {collab_available_steps}  ({100*collab_available_steps/max(1,total_active_steps):.1f}%)")
    print(f"collab_chosen_steps        = {collab_chosen_steps}  ({100*collab_chosen_steps/max(1,collab_available_steps):.2f}% of available)")
    print(f"max_groups_active_simul    = {max_groups_seen}")
    print(f"_ep_groups_formed (max)    = {total_groups_formed}")
    print(f"help_events_seen (max)     = {help_events_seen}")
    print(f"\nAction distribution (top 15) — name, count, pct:")
    name_map = {a.value: a.name for a in Action}
    for a, c in action_counts.most_common(15):
        print(f"  {name_map.get(a, a):20s} {c:7d}  {100*c/max(1,total_active_steps):.2f}%")


if __name__ == "__main__":
    main()
