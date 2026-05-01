"""scripts/train_ippo_joint_recurrent.py — v24 joint-action recurrent IPPO trainer.

Mirror of ``train_ippo_recurrent.py`` swapped to the joint policy + buffer
+ env API. Same hidden-state lifecycle (per-slot pending_life_reset on
born, full reset on env wipe). FSP omitted — see train_ippo_joint.py.

Usage:
    python -m scripts.train_ippo_joint_recurrent --total-steps 1_500_000 \
        --checkpoint-dir checkpoints/ippo_v24b_rec
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from murimsim.actions import N_BODY_ACTIONS, N_SOCIAL_ACTIONS
from murimsim.rl.ippo_env import IPPOEnv
from murimsim.rl.ippo_joint_recurrent import (
    JointRecurrentRolloutBuffer,
    JointRecurrentSharedActorCritic,
    joint_recurrent_ppo_update,
)
from murimsim.rl.multi_env import OBS_TOTAL_SIZE
from scripts.train_ippo import affinity_l1, build_envs

logger = logging.getLogger("train_ippo_joint_recurrent")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--total-steps", type=int, default=500_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--n-agents", type=int, default=4)
    p.add_argument("--rollout-length", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--pre-lstm-dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--n-minibatches", type=int, default=4)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--checkpoint-dir", default="checkpoints/ippo_v24_recurrent")
    p.add_argument("--checkpoint-interval", type=int, default=20)
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", default="murimsim-ippo")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--enable-boss", action="store_true")
    p.add_argument("--enable-carry-cost", action="store_true")
    p.add_argument(
        "--disable-formation-bonus",
        action="store_true",
        help="v25: drop the +0.05 REWARD_GROUP_FORMATION shaping bonus to test "
             "whether cooperation survives without artificial reinforcement.",
    )
    p.add_argument("--arena-mix", type=str, default=None)
    return p.parse_args()


def _collect_initial_state(envs: list[IPPOEnv], seed: int):
    obs_l, body_l, social_l, active_l = [], [], [], []
    for i, env in enumerate(envs):
        obs, info = env.reset_all(seed=seed + i)
        obs_l.append(obs)
        body_l.append(info["action_masks_body"])
        social_l.append(info["action_masks_social"])
        active_l.append(info["active_mask"])
    return (
        np.stack(obs_l, axis=0),
        np.stack(body_l, axis=0),
        np.stack(social_l, axis=0),
        np.stack(active_l, axis=0),
    )


def train(args: argparse.Namespace) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    obs_dim = OBS_TOTAL_SIZE
    B = args.n_envs * args.n_agents

    envs = build_envs(
        cfg, args.n_envs, args.n_agents, args.seed,
        enable_boss=getattr(args, "enable_boss", False),
        enable_carry_cost=getattr(args, "enable_carry_cost", False),
        arena_mix=getattr(args, "arena_mix", None),
        enable_formation_bonus=not getattr(args, "disable_formation_bonus", False),
    )
    obs, body_mask, social_mask, active_mask = _collect_initial_state(envs, args.seed)

    policy = JointRecurrentSharedActorCritic(
        obs_dim=obs_dim,
        hidden_dim=args.hidden_dim,
        pre_lstm_dim=args.pre_lstm_dim,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)

    buffer = JointRecurrentRolloutBuffer(
        rollout_length=args.rollout_length,
        n_envs=args.n_envs,
        n_agents=args.n_agents,
        obs_dim=obs_dim,
        hidden_dim=args.hidden_dim,
        device=device,
    )

    use_wandb = (not args.no_wandb) and os.environ.get("WANDB_DISABLED", "") != "1"
    wandb = None
    if use_wandb:
        try:
            import wandb as _wandb
            wandb = _wandb
            wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
        except Exception as e:
            logger.warning("wandb init failed: %s — continuing without wandb", e)
            wandb = None

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    transitions_per_iter = args.rollout_length * B
    total_iters = max(1, args.total_steps // transitions_per_iter)
    logger.info(
        "v24 recurrent joint IPPO: %d iters × %d transitions = %d (target %d)",
        total_iters, transitions_per_iter, total_iters * transitions_per_iter,
        args.total_steps,
    )

    carried_h, carried_c = policy.initial_hidden(B, device=device)
    pending_life_reset = np.ones((args.n_envs, args.n_agents), dtype=bool)

    ep_reward_sum = np.zeros((args.n_envs, args.n_agents), dtype=np.float64)
    ep_step_count = np.zeros((args.n_envs, args.n_agents), dtype=np.int64)
    completed_lives_reward: list[float] = []
    completed_lives_steps: list[int] = []
    collab_count = 0

    t_start = time.time()
    summary: dict = {}

    for it in range(total_iters):
        buffer.reset()
        buffer.set_initial_hidden(carried_h, carried_c)

        for _t in range(args.rollout_length):
            if pending_life_reset.any():
                reset_flat = pending_life_reset.reshape(-1)
                reset_t = torch.as_tensor(reset_flat, dtype=torch.bool, device=device)
                keep = (~reset_t).view(1, B, 1).to(carried_h.dtype)
                carried_h = carried_h * keep
                carried_c = carried_c * keep

            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                bm_t = torch.as_tensor(body_mask, dtype=torch.bool, device=device)
                sm_t = torch.as_tensor(social_mask, dtype=torch.bool, device=device)
                obs_flat = obs_t.reshape(-1, obs_dim)
                bm_flat = bm_t.reshape(-1, N_BODY_ACTIONS)
                sm_flat = sm_t.reshape(-1, N_SOCIAL_ACTIONS)
                ba_flat, sa_flat, blp_flat, slp_flat, val_flat, (carried_h, carried_c) = policy.act(
                    obs_flat, bm_flat, sm_flat, (carried_h, carried_c)
                )
                ba = ba_flat.reshape(args.n_envs, args.n_agents)
                sa = sa_flat.reshape(args.n_envs, args.n_agents)
                blp = blp_flat.reshape(args.n_envs, args.n_agents)
                slp = slp_flat.reshape(args.n_envs, args.n_agents)
                val = val_flat.reshape(args.n_envs, args.n_agents)

            ba_np = ba.cpu().numpy().astype(np.int64)
            sa_np = sa.cpu().numpy().astype(np.int64)
            collab_count += int((sa_np == 1).sum())

            rewards = np.zeros((args.n_envs, args.n_agents), dtype=np.float32)
            dones = np.zeros((args.n_envs, args.n_agents), dtype=bool)
            next_obs = np.zeros_like(obs)
            next_body = np.zeros_like(body_mask)
            next_social = np.zeros_like(social_mask)
            next_active = np.zeros_like(active_mask)
            next_pending = np.zeros((args.n_envs, args.n_agents), dtype=bool)

            stored_life_reset = pending_life_reset.copy()

            for e_i, env in enumerate(envs):
                o, r, term, trunc, info = env.step_all_joint(ba_np[e_i], sa_np[e_i])
                rewards[e_i] = r
                dones[e_i] = term | trunc
                next_obs[e_i] = o
                next_body[e_i] = info["action_masks_body_post"]
                next_social[e_i] = info["action_masks_social_post"]
                env_next_active = np.array([a.alive for a in env._agents], dtype=bool)

                lc = info.get("lifecycle", [])
                for a_i in range(args.n_agents):
                    if a_i < len(lc) and lc[a_i].get("born", False):
                        next_pending[e_i, a_i] = True

                if not env_next_active.any():
                    reset_seed = args.seed + e_i + 1_000_000 * (it + 1)
                    o, info = env.reset_all(seed=reset_seed)
                    next_obs[e_i] = o
                    next_body[e_i] = info["action_masks_body"]
                    next_social[e_i] = info["action_masks_social"]
                    env_next_active = info["active_mask"]
                    next_pending[e_i, :] = True
                next_active[e_i] = env_next_active

            buffer.add(
                obs=obs,
                body_mask=body_mask,
                social_mask=social_mask,
                body_action=ba,
                social_action=sa,
                body_logprob=blp,
                social_logprob=slp,
                value=val,
                reward=rewards,
                done=dones,
                active=active_mask,
                life_reset=stored_life_reset,
            )

            ep_reward_sum += rewards
            ep_step_count += active_mask.astype(np.int64)
            died = dones
            if died.any():
                for e_i in range(args.n_envs):
                    for a_i in range(args.n_agents):
                        if died[e_i, a_i]:
                            completed_lives_reward.append(float(ep_reward_sum[e_i, a_i]))
                            completed_lives_steps.append(int(ep_step_count[e_i, a_i]))
                            ep_reward_sum[e_i, a_i] = 0.0
                            ep_step_count[e_i, a_i] = 0

            obs = next_obs
            body_mask = next_body
            social_mask = next_social
            active_mask = next_active
            pending_life_reset = next_pending

        # Bootstrap value uses carried hidden — does NOT mutate it.
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            obs_flat = obs_t.reshape(-1, obs_dim)
            scratch_h = carried_h.clone()
            scratch_c = carried_c.clone()
            if pending_life_reset.any():
                reset_flat = pending_life_reset.reshape(-1)
                reset_t = torch.as_tensor(reset_flat, dtype=torch.bool, device=device)
                keep = (~reset_t).view(1, B, 1).to(scratch_h.dtype)
                scratch_h = scratch_h * keep
                scratch_c = scratch_c * keep
            last_value_flat = policy.value_only(obs_flat, (scratch_h, scratch_c))
            last_value = last_value_flat.reshape(args.n_envs, args.n_agents)
            last_active = torch.as_tensor(active_mask, dtype=torch.bool, device=device)

        adv, ret = buffer.compute_gae(
            last_value=last_value, last_active=last_active,
            gamma=args.gamma, lam=args.lam,
        )
        sb = buffer.to_sequence_batch(adv, ret)
        stats = joint_recurrent_ppo_update(
            policy,
            optimizer,
            sb,
            clip_coef=args.clip_coef,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            n_epochs=args.n_epochs,
            n_minibatches=args.n_minibatches,
            max_grad_norm=args.max_grad_norm,
        )

        carried_h = carried_h.detach()
        carried_c = carried_c.detach()

        global_step = (it + 1) * transitions_per_iter
        elapsed = max(time.time() - t_start, 1e-6)
        sps = global_step / elapsed
        recent = completed_lives_reward[-100:] if completed_lives_reward else [0.0]
        affinity_avg = float(np.mean([affinity_l1(e) for e in envs]))
        n_active = int(buffer.active.sum().item())

        if it % args.log_interval == 0:
            logger.info(
                "iter=%d step=%d sps=%.1f pg=%.4f vf=%.4f bent=%.4f sent=%.4f "
                "kl=%.4f clip=%.3f active_n=%d life_reward_mean=%.3f "
                "affinity_l1=%.3f collab=%d",
                it, global_step, sps,
                stats.pg_loss, stats.vf_loss, stats.body_entropy, stats.social_entropy,
                stats.approx_kl, stats.clip_frac, n_active,
                float(np.mean(recent)), affinity_avg, collab_count,
            )
        if wandb is not None:
            wandb.log({
                "train/pg_loss": stats.pg_loss,
                "train/vf_loss": stats.vf_loss,
                "train/body_entropy": stats.body_entropy,
                "train/social_entropy": stats.social_entropy,
                "train/approx_kl": stats.approx_kl,
                "train/clip_frac": stats.clip_frac,
                "train/n_active": n_active,
                "rollout/sps": sps,
                "rollout/life_reward_mean": float(np.mean(recent)),
                "rollout/affinity_l1": affinity_avg,
                "rollout/collab_picks": collab_count,
                "global_step": global_step,
            })

        if (it + 1) % args.checkpoint_interval == 0 or it == total_iters - 1:
            ckpt = ckpt_dir / f"ippo_joint_recurrent_iter_{it + 1:06d}.pt"
            torch.save(
                {
                    "iter": it + 1,
                    "global_step": global_step,
                    "policy": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                    "hidden_dim": args.hidden_dim,
                    "joint_action": True,
                    "recurrent": True,
                },
                ckpt,
            )
            logger.info("checkpoint saved: %s", ckpt)

        summary = {
            "iter": it + 1,
            "global_step": global_step,
            "pg_loss": stats.pg_loss,
            "vf_loss": stats.vf_loss,
            "body_entropy": stats.body_entropy,
            "social_entropy": stats.social_entropy,
            "n_active": n_active,
            "affinity_l1": affinity_avg,
            "life_reward_mean": float(np.mean(recent)),
            "completed_lives": len(completed_lives_reward),
            "collab_picks": collab_count,
        }

    if wandb is not None:
        wandb.finish()
    return summary


if __name__ == "__main__":
    train(parse_args())
