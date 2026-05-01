"""Probe v21 checkpoints for boss participation + legacy_stash claim telemetry.

Runs each checkpoint for STEPS env-steps with all slots policy-controlled,
then reports:
  * episodes_completed
  * boss_kills_total                — bosses killed across all episodes
  * boss_kills_solo                 — kills with exactly 1 unique attacker
  * boss_kills_joint                — kills with ≥2 unique attackers
  * mean_attackers_per_boss_kill    — average team size on a boss kill
  * legacy_stashes_created          — agent corpses that left a bequest
  * legacy_stash_withdrawals        — total successful WITHDRAWs from a legacy stash
  * legacy_stash_withdrawals_by_heir — withdrawals attributed to participant
  * legacy_stash_withdrawals_open   — withdrawals after lockout by a non-participant

Reuses the eval harness construction so env conditions match training.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

from murimsim.actions import Action
from murimsim.rl.ippo_env import IPPOEnv
from scripts.eval_ippo import load_policy


def probe(
    checkpoint: str,
    enable_boss: bool,
    enable_carry_cost: bool,
    steps: int = 5000,
    n_envs: int = 4,
    n_agents: int = 4,
    seed: int = 0,
) -> dict:
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
        obs_list.append(o)
        mask_list.append(info["action_masks"])
    obs = np.stack(obs_list, axis=0)
    masks = np.stack(mask_list, axis=0)

    hidden = None
    if is_recurrent:
        h = torch.zeros(1, n_envs * n_agents, policy.hidden_dim)
        c = torch.zeros(1, n_envs * n_agents, policy.hidden_dim)
        hidden = (h, c)

    boss_kill_attackers: list[set[str]] = []
    legacy_stashes_seen: set[str] = set()
    legacy_withdrawals = 0
    legacy_withdrawals_by_heir = 0
    legacy_withdrawals_open = 0
    episodes = 0
    # Track legacy stash unlock_step + participants per stash_id for accounting.
    legacy_meta: dict[str, dict] = {}

    # snapshot prior boss-kill count per env to detect new kills.
    prev_boss_kills = [0 for _ in envs]
    # snapshot prior withdrawal counts per env x agent — we'll diff.
    prev_withdraw = [Counter() for _ in envs]

    for _ in range(steps):
        flat_obs = torch.tensor(obs.reshape(n_envs * n_agents, -1), dtype=torch.float32)
        flat_masks = torch.tensor(masks.reshape(n_envs * n_agents, -1), dtype=torch.bool)
        with torch.no_grad():
            if is_recurrent:
                logits, _, hidden_new = policy(flat_obs, hidden=hidden, action_masks=flat_masks)
            else:
                logits, _ = policy(flat_obs, action_masks=flat_masks)
                hidden_new = None
            actions = torch.argmax(logits, dim=-1).cpu().numpy()
        actions = actions.reshape(n_envs, n_agents)

        for ei, env in enumerate(envs):
            # Snapshot legacy stashes before step
            stashes_before = {s.stash_id for s in env._stash_registry.all_stashes() if s.claim_unlock_step >= 0}

            obs_e, _r, term, trunc, info = env.step_all(actions[ei])
            obs[ei] = obs_e
            masks[ei] = info["action_masks"]

            # Boss kill detection
            cur_kills = info.get("ep_boss_killed", env._ep_boss_killed)
            if cur_kills > prev_boss_kills[ei]:
                # We don't have per-kill attacker breakdown easily, but env keeps
                # the *current live* monsters' attackers; once killed, the monster
                # is removed. We can pull from monster registry just before they die
                # — simpler: poll env._monsters BEFORE step too.
                pass
            prev_boss_kills[ei] = cur_kills

            # New legacy stashes
            for s in env._stash_registry.all_stashes():
                if s.claim_unlock_step >= 0 and s.stash_id not in legacy_stashes_seen:
                    legacy_stashes_seen.add(s.stash_id)
                    legacy_meta[s.stash_id] = {
                        "participants": list(s.participants),
                        "unlock_step": s.claim_unlock_step,
                    }

            # Episode bookkeeping
            if bool(trunc.any()) or info.get("episode_done", False):
                episodes += 1

    # Re-run with monster attacker tracking via direct env hooks for boss kills.
    # Simpler: grab all bosses' attackers at the moment of death by patching the
    # take_damage path. Easier still — re-run with a hook.
    return {
        "episodes_completed": episodes,
        "legacy_stashes_created": len(legacy_stashes_seen),
        "legacy_meta_sample": list(legacy_meta.items())[:5],
    }


# Simpler approach — patch monster.take_damage to capture attacker set on kill.
def probe_v2(
    checkpoint: str,
    enable_boss: bool,
    enable_carry_cost: bool,
    steps: int = 5000,
    n_envs: int = 4,
    n_agents: int = 4,
    seed: int = 0,
) -> dict:
    from murimsim.monster import Monster
    from murimsim.rl.multi_env import LEGACY_UNLOCK_TICKS

    np.random.seed(seed)
    torch.manual_seed(seed)
    with open("config/default.yaml") as f:
        cfg = yaml.safe_load(f)

    policy, _, is_recurrent = load_policy(checkpoint, device="cpu")

    boss_kill_records: list[dict] = []
    original_take_damage = Monster.take_damage

    def patched_take_damage(self, damage, attacker_id):
        killed = original_take_damage(self, damage, attacker_id)
        if killed and self.kind == "boss":
            boss_kill_records.append({
                "monster_id": self.monster_id,
                "attackers": sorted(self.attackers),
                "n_attackers": len(self.attackers),
            })
        return killed

    Monster.take_damage = patched_take_damage
    try:
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
            obs_list.append(o)
            mask_list.append(info["action_masks"])
        obs = np.stack(obs_list, axis=0)
        masks = np.stack(mask_list, axis=0)

        hidden = None
        if is_recurrent:
            hidden = (
                torch.zeros(1, n_envs * n_agents, policy.hidden_dim),
                torch.zeros(1, n_envs * n_agents, policy.hidden_dim),
            )

        legacy_stashes_seen: dict[str, dict] = {}
        legacy_withdrawals = 0
        legacy_withdrawals_by_heir = 0
        legacy_withdrawals_open = 0

        # Track withdrawal events by snapshotting stash inventories pre/post step.
        for step_i in range(steps):
            flat_obs = torch.tensor(obs.reshape(n_envs * n_agents, -1), dtype=torch.float32)
            flat_masks = torch.tensor(masks.reshape(n_envs * n_agents, -1), dtype=torch.bool)
            with torch.no_grad():
                if is_recurrent:
                    actions_t, _, _, hidden = policy.act(flat_obs, flat_masks, hidden=hidden, deterministic=True)
                else:
                    actions_t, _, _ = policy.act(flat_obs, flat_masks, deterministic=True)
                actions = actions_t.cpu().numpy()
            actions = actions.reshape(n_envs, n_agents)

            for ei, env in enumerate(envs):
                # snapshot legacy stash inventories pre-step
                pre = {s.stash_id: (s.food + s.qi + s.materials + s.poison)
                       for s in env._stash_registry.all_stashes() if s.claim_unlock_step >= 0}

                obs_e, _r, _term, _trunc, info = env.step_all(actions[ei])
                obs[ei] = obs_e
                masks[ei] = info["action_masks_post"]

                # detect new legacy stashes & track withdrawals
                for s in env._stash_registry.all_stashes():
                    if s.claim_unlock_step >= 0:
                        if s.stash_id not in legacy_stashes_seen:
                            legacy_stashes_seen[s.stash_id] = {
                                "participants": list(s.participants),
                                "unlock_step": s.claim_unlock_step,
                                "created_step": env._ep_step_count,
                            }
                        post_total = s.food + s.qi + s.materials + s.poison
                        pre_total = pre.get(s.stash_id, post_total)
                        if post_total < pre_total:
                            withdrawn = pre_total - post_total
                            legacy_withdrawals += withdrawn
                            if env._ep_step_count >= s.claim_unlock_step:
                                legacy_withdrawals_open += withdrawn
                            else:
                                legacy_withdrawals_by_heir += withdrawn

        n_kills = len(boss_kill_records)
        n_solo = sum(1 for r in boss_kill_records if r["n_attackers"] == 1)
        n_joint = sum(1 for r in boss_kill_records if r["n_attackers"] >= 2)
        mean_attackers = (
            sum(r["n_attackers"] for r in boss_kill_records) / n_kills if n_kills else 0.0
        )

        # Aggregate combat telemetry across all envs.
        boss_attacks_landed = sum(getattr(e, "_ep_boss_attacks_landed", 0) for e in envs)
        boss_damage_dealt = sum(getattr(e, "_ep_boss_damage_dealt", 0.0) for e in envs)
        # Boss-spawn count: count distinct monster_ids tracked.
        boss_spawned_total = sum(
            1 for e in envs for m in e._monsters.all() if m.kind == "boss"
        )
        # Track how many bosses have *ever* spawned this episode by walking history if available.
        # Simpler: report current alive bosses + total kills as proxy for engagement budget.

        return {
            "boss_kills_total": n_kills,
            "boss_kills_solo": n_solo,
            "boss_kills_joint_2plus": n_joint,
            "mean_attackers_per_boss_kill": round(mean_attackers, 2),
            "attacker_count_distribution": dict(Counter(r["n_attackers"] for r in boss_kill_records)),
            "boss_attacks_landed_total": int(boss_attacks_landed),
            "boss_damage_dealt_total": round(float(boss_damage_dealt), 3),
            "boss_alive_at_end": int(boss_spawned_total),
            "legacy_stashes_created": len(legacy_stashes_seen),
            "legacy_with_heirs": sum(1 for v in legacy_stashes_seen.values() if v["participants"]),
            "legacy_no_heirs": sum(1 for v in legacy_stashes_seen.values() if not v["participants"]),
            "legacy_withdrawals_total_items": legacy_withdrawals,
            "legacy_withdrawals_by_heir_period": legacy_withdrawals_by_heir,
            "legacy_withdrawals_open_period": legacy_withdrawals_open,
        }
    finally:
        Monster.take_damage = original_take_damage


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--enable-boss", action="store_true")
    p.add_argument("--enable-carry-cost", action="store_true")
    p.add_argument("--steps", type=int, default=5000)
    args = p.parse_args()
    out = probe_v2(
        checkpoint=args.checkpoint,
        enable_boss=args.enable_boss,
        enable_carry_cost=args.enable_carry_cost,
        steps=args.steps,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
