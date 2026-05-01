"""scripts/train_ippo_joint.py — v24 joint-action FF IPPO trainer.

Same training loop as ``train_ippo.py`` but uses the joint body+social
policy (``JointSharedActorCritic``) and ``IPPOEnv.step_all_joint``.

FSP is intentionally OMITTED here — v23 single-head checkpoints are
incompatible with the v24 action space. After the first v24 baseline
trains, future runs can re-add FSP using a v24 frozen checkpoint
(symmetric action shape).

Usage:
    python -m scripts.train_ippo_joint --total-steps 1_500_000 \
        --checkpoint-dir checkpoints/ippo_v24a_ff
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
from murimsim.rl.ippo_joint import (
    JointRolloutBuffer,
    JointSharedActorCritic,
    joint_ppo_update,
)
from murimsim.rl.multi_env import OBS_TOTAL_SIZE
from scripts.train_ippo import (
    _load_arena_config,
    _parse_arena_mix,
    affinity_l1,
    build_envs,
)

logger = logging.getLogger("train_ippo_joint")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--total-steps", type=int, default=500_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--n-agents", type=int, default=4)
    p.add_argument("--rollout-length", type=int, default=128)
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
    p.add_argument("--checkpoint-dir", default="checkpoints/ippo_v24_ff")
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


def _collect_initial_state_joint(
    envs: list[IPPOEnv], seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    envs = build_envs(
        cfg, args.n_envs, args.n_agents, args.seed,
        enable_boss=getattr(args, "enable_boss", False),
        enable_carry_cost=getattr(args, "enable_carry_cost", False),
        arena_mix=getattr(args, "arena_mix", None),
        enable_formation_bonus=not getattr(args, "disable_formation_bonus", False),
    )
    obs, body_mask, social_mask, active_mask = _collect_initial_state_joint(
        envs, args.seed
    )

    policy = JointSharedActorCritic(obs_dim=obs_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)

    buffer = JointRolloutBuffer(
        rollout_length=args.rollout_length,
        n_envs=args.n_envs,
        n_agents=args.n_agents,
        obs_dim=obs_dim,
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

    transitions_per_iter = args.rollout_length * args.n_envs * args.n_agents
    total_iters = max(1, args.total_steps // transitions_per_iter)
    logger.info(
        "v24 joint IPPO: %d iters × %d transitions = %d (target %d)",
        total_iters, transitions_per_iter, total_iters * transitions_per_iter,
        args.total_steps,
    )

    ep_reward_sum = np.zeros((args.n_envs, args.n_agents), dtype=np.float64)
    ep_step_count = np.zeros((args.n_envs, args.n_agents), dtype=np.int64)
    completed_lives_reward: list[float] = []
    completed_lives_steps: list[int] = []
    collab_count = 0

    t_start = time.time()
    summary: dict = {}

    for it in range(total_iters):
        buffer.reset()
        for _t in range(args.rollout_length):
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                bm_t = torch.as_tensor(body_mask, dtype=torch.bool, device=device)
                sm_t = torch.as_tensor(social_mask, dtype=torch.bool, device=device)
                obs_flat = obs_t.reshape(-1, obs_dim)
                bm_flat = bm_t.reshape(-1, N_BODY_ACTIONS)
                sm_flat = sm_t.reshape(-1, N_SOCIAL_ACTIONS)
                ba_flat, sa_flat, blp_flat, slp_flat, val_flat = policy.act(
                    obs_flat, bm_flat, sm_flat
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
            for e_i, env in enumerate(envs):
                o, r, term, trunc, info = env.step_all_joint(ba_np[e_i], sa_np[e_i])
                rewards[e_i] = r
                dones[e_i] = term | trunc
                next_obs[e_i] = o
                next_body[e_i] = info["action_masks_body_post"]
                next_social[e_i] = info["action_masks_social_post"]
                env_next_active = np.array([a.alive for a in env._agents], dtype=bool)
                if not env_next_active.any():
                    reset_seed = args.seed + e_i + 1_000_000 * (it + 1)
                    o, info = env.reset_all(seed=reset_seed)
                    next_obs[e_i] = o
                    next_body[e_i] = info["action_masks_body"]
                    next_social[e_i] = info["action_masks_social"]
                    env_next_active = info["active_mask"]
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

        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            obs_flat = obs_t.reshape(-1, obs_dim)
            _bl, _sl, last_value_flat = policy.forward(obs_flat)
            last_value = last_value_flat.reshape(args.n_envs, args.n_agents)
            last_active = torch.as_tensor(active_mask, dtype=torch.bool, device=device)

        adv, ret = buffer.compute_gae(
            last_value=last_value,
            last_active=last_active,
            gamma=args.gamma,
            lam=args.lam,
        )
        batch = buffer.flatten_active(adv, ret)
        stats = joint_ppo_update(
            policy,
            optimizer,
            batch,
            clip_coef=args.clip_coef,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            n_epochs=args.n_epochs,
            n_minibatches=args.n_minibatches,
            max_grad_norm=args.max_grad_norm,
        )

        global_step = (it + 1) * transitions_per_iter
        elapsed = max(time.time() - t_start, 1e-6)
        sps = global_step / elapsed
        recent = completed_lives_reward[-100:] if completed_lives_reward else [0.0]
        affinity_avg = float(np.mean([affinity_l1(e) for e in envs]))

        if it % args.log_interval == 0:
            logger.info(
                "iter=%d step=%d sps=%.1f pg=%.4f vf=%.4f bent=%.4f sent=%.4f "
                "kl=%.4f clip=%.3f active_n=%d life_reward_mean=%.3f "
                "affinity_l1=%.3f collab=%d",
                it, global_step, sps,
                stats.pg_loss, stats.vf_loss, stats.body_entropy, stats.social_entropy,
                stats.approx_kl, stats.clip_frac, len(batch),
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
                "train/n_active": len(batch),
                "rollout/sps": sps,
                "rollout/life_reward_mean": float(np.mean(recent)),
                "rollout/affinity_l1": affinity_avg,
                "rollout/collab_picks": collab_count,
                "global_step": global_step,
            })

        if (it + 1) % args.checkpoint_interval == 0 or it == total_iters - 1:
            ckpt = ckpt_dir / f"ippo_joint_iter_{it + 1:06d}.pt"
            torch.save(
                {
                    "iter": it + 1,
                    "global_step": global_step,
                    "policy": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                    "joint_action": True,
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
            "n_active": len(batch),
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
