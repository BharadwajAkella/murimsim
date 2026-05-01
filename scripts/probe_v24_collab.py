"""Probe: count cumulative group formations + help events across v24 eval rollout."""
from __future__ import annotations
import sys, numpy as np, torch, yaml
from pathlib import Path
sys.path.insert(0, str(Path("/home/ravakella/murimsim")))
from scripts.eval_ippo import is_joint_checkpoint, load_joint_policy
from scripts.train_ippo import _load_arena_config, _parse_arena_mix
from murimsim.rl.ippo_env import IPPOEnv

CKPT = sys.argv[1]
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
N_AGENTS = 10
N_ENVS = 4
SEED = 12345

policy, args, is_recurrent = load_joint_policy(CKPT, "cpu")
with open("config/default.yaml") as f:
    cfg = yaml.safe_load(f)

arena_names = _parse_arena_mix("base:1,arena_minion:2,arena_boss:1")
envs = []
for i in range(N_ENVS):
    arena = arena_names[i % len(arena_names)]
    env_cfg, arena_flags = _load_arena_config(arena, cfg)
    envs.append(IPPOEnv(
        config=env_cfg, n_agents=N_AGENTS, seed=SEED + i, curriculum_ramp_steps=0,
        enable_boss=bool(arena_flags.get("enable_boss", False)),
        enable_carry_cost=bool(arena_flags.get("enable_carry_cost", True)),
        n_minions=int(arena_flags.get("n_minions", 0)),
    ))

obs_list, body_mask_list, social_mask_list = [], [], []
for i, e in enumerate(envs):
    o, info = e.reset_all(seed=SEED + i)
    obs_list.append(o); body_mask_list.append(info["action_masks_body"])
    social_mask_list.append(info["action_masks_social"])
obs = np.stack(obs_list); body_mask = np.stack(body_mask_list); social_mask = np.stack(social_mask_list)

if is_recurrent:
    hd = args["hidden_dim"]
    h = torch.zeros(N_ENVS, N_AGENTS, hd)
    c = torch.zeros(N_ENVS, N_AGENTS, hd)

collab_picks, groups_formed, help_count = 0, 0, 0
groups_seen = []

for t in range(STEPS):
    obs_t = torch.from_numpy(obs).float().reshape(-1, obs.shape[-1])
    bm_t = torch.from_numpy(body_mask).reshape(-1, body_mask.shape[-1])
    sm_t = torch.from_numpy(social_mask).reshape(-1, social_mask.shape[-1])
    with torch.no_grad():
        if is_recurrent:
            h_in = h.reshape(1, -1, hd)
            c_in = c.reshape(1, -1, hd)
            ba, sa, _, _, _, (h_new, c_new) = policy.act(obs_t, bm_t, sm_t, (h_in, c_in))
            h = h_new.reshape(N_ENVS, N_AGENTS, hd)
            c = c_new.reshape(N_ENVS, N_AGENTS, hd)
        else:
            ba, sa, _, _, _ = policy.act(obs_t, bm_t, sm_t)
    ba = ba.numpy().reshape(N_ENVS, N_AGENTS)
    sa = sa.numpy().reshape(N_ENVS, N_AGENTS)

    for ei, e in enumerate(envs):
        groups_before = len(e._groups)
        helps_before = sum(len(d) for d in e._help_received.values())
        o, r, term, trunc, info = e.step_all_joint(ba[ei], sa[ei])
        collab_picks += int((sa[ei] == 1).sum())
        groups_after = len(e._groups)
        helps_after = sum(len(d) for d in e._help_received.values())
        if groups_after > groups_before:
            groups_formed += groups_after - groups_before
        if helps_after > helps_before:
            help_count += helps_after - helps_before
        obs[ei] = o
        body_mask[ei] = info["action_masks_body_post"]
        social_mask[ei] = info["action_masks_social_post"]
        if is_recurrent:
            for ai in range(N_AGENTS):
                if ai < len(info.get("lifecycle", [])) and info["lifecycle"][ai].get("born", False):
                    h[ei, ai] = 0; c[ei, ai] = 0
        if not any(a.alive for a in e._agents):
            o, info = e.reset_all(seed=SEED + (t+1)*1000 + ei)
            obs[ei] = o
            body_mask[ei] = info["action_masks_body"]
            social_mask[ei] = info["action_masks_social"]
            if is_recurrent:
                h[ei] = 0; c[ei] = 0
        groups_seen.append(len(e._groups))

print(f"\n=== {CKPT} ({STEPS} steps × {N_ENVS} envs × {N_AGENTS} agents) ===")
print(f"  collab_picks (social=COLLAB)  = {collab_picks}")
print(f"  groups_formed (cumulative)    = {groups_formed}")
print(f"  help_events (cumulative)      = {help_count}")
print(f"  mean_active_groups_per_step   = {np.mean(groups_seen):.3f}")
print(f"  max_active_groups_seen        = {max(groups_seen)}")
print(f"  collab_success_rate           = {groups_formed/max(collab_picks,1)*100:.2f}%")
