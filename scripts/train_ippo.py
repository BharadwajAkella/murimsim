"""scripts/train_ippo.py — feedforward IPPO trainer (Phase 3 of IPPO migration).

Usage:
    python -m scripts.train_ippo --total-steps 500_000 --n-envs 4 --n-agents 4

For smoke testing:
    python -m scripts.train_ippo --total-steps 4096 --n-envs 2 --n-agents 4 --no-wandb

Design:
    * n_envs IPPOEnv instances run sequentially (no SubprocVecEnv yet).
    * Single shared SharedActorCritic — one forward pass per (env, agent) per step.
    * Per-slot active mask drives loss weighting; dead slots contribute zero.
    * Pre-action masks stored in the rollout buffer; reused at update time.
    * Checkpoints written to ``checkpoints/ippo_v1/`` every checkpoint_interval iters.
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

from murimsim.actions import N_ACTIONS_PHASE6_QI
from murimsim.rl.ippo import (
    RolloutBuffer,
    SharedActorCritic,
    ppo_update,
)
from murimsim.rl.ippo_env import IPPOEnv
from murimsim.rl.multi_env import OBS_TOTAL_SIZE

logger = logging.getLogger("train_ippo")


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
    p.add_argument("--checkpoint-dir", default="checkpoints/ippo_v1")
    p.add_argument("--checkpoint-interval", type=int, default=20)
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", default="murimsim-ippo")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument(
        "--enable-boss",
        action="store_true",
        help="Spawn boss monster per episode (v17 common-enemy emergence pressure).",
    )
    return p.parse_args()


def build_envs(
    config: dict,
    n_envs: int,
    n_agents: int,
    seed: int,
    enable_boss: bool = False,
) -> list[IPPOEnv]:
    envs: list[IPPOEnv] = []
    for i in range(n_envs):
        env = IPPOEnv(
            config=config,
            n_agents=n_agents,
            seed=seed + i,
            curriculum_ramp_steps=0,
            enable_boss=enable_boss,
        )
        envs.append(env)
    return envs


def collect_initial_state(
    envs: list[IPPOEnv], seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reset all envs, return stacked (obs, action_masks, active_mask)."""
    obs_list, mask_list, active_list = [], [], []
    for i, env in enumerate(envs):
        obs, info = env.reset_all(seed=seed + i)
        obs_list.append(obs)
        mask_list.append(info["action_masks"])
        active_list.append(info["active_mask"])
    return (
        np.stack(obs_list, axis=0),
        np.stack(mask_list, axis=0),
        np.stack(active_list, axis=0),
    )


def affinity_l1(env: IPPOEnv) -> float:
    """L1 norm of all stored affinity values — proxy for social interaction."""
    total = 0.0
    for row in env._affinity_raw.values():
        for val, _step in row.values():
            total += abs(val)
    return total


def train(args: argparse.Namespace) -> dict:
    """Run the IPPO training loop. Returns a summary dict for callers/tests."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    obs_dim = OBS_TOTAL_SIZE
    n_actions = N_ACTIONS_PHASE6_QI

    envs = build_envs(
        cfg, args.n_envs, args.n_agents, args.seed,
        enable_boss=getattr(args, "enable_boss", False),
    )
    obs, action_masks, active_mask = collect_initial_state(envs, args.seed)

    policy = SharedActorCritic(obs_dim=obs_dim, n_actions=n_actions).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)

    buffer = RolloutBuffer(
        rollout_length=args.rollout_length,
        n_envs=args.n_envs,
        n_agents=args.n_agents,
        obs_dim=obs_dim,
        n_actions=n_actions,
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
        "starting IPPO: %d iters × %d transitions = %d (target: %d)",
        total_iters, transitions_per_iter, total_iters * transitions_per_iter, args.total_steps,
    )

    # Episode-level accounting (population continuum — no real episodes)
    ep_reward_sum = np.zeros((args.n_envs, args.n_agents), dtype=np.float64)
    ep_step_count = np.zeros((args.n_envs, args.n_agents), dtype=np.int64)
    completed_lives_reward: list[float] = []
    completed_lives_steps: list[int] = []

    t_start = time.time()
    summary: dict = {}

    for it in range(total_iters):
        buffer.reset()
        for _t in range(args.rollout_length):
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                mask_t = torch.as_tensor(action_masks, dtype=torch.bool, device=device)
                # Flatten (n_envs, n_agents, ...) → (n_envs*n_agents, ...)
                flat_obs = obs_t.reshape(-1, obs_dim)
                flat_mask = mask_t.reshape(-1, n_actions)
                action_flat, logp_flat, value_flat = policy.act(flat_obs, flat_mask)
                action = action_flat.reshape(args.n_envs, args.n_agents)
                logp = logp_flat.reshape(args.n_envs, args.n_agents)
                value = value_flat.reshape(args.n_envs, args.n_agents)

            actions_np = action.cpu().numpy()
            rewards = np.zeros((args.n_envs, args.n_agents), dtype=np.float32)
            dones = np.zeros((args.n_envs, args.n_agents), dtype=bool)
            next_obs = np.zeros_like(obs)
            next_masks = np.zeros_like(action_masks)
            next_active = np.zeros_like(active_mask)
            for e_i, env in enumerate(envs):
                o, r, term, trunc, info = env.step_all(actions_np[e_i])
                rewards[e_i] = r
                dones[e_i] = term | trunc
                next_obs[e_i] = o
                next_masks[e_i] = info["action_masks_post"]
                # active for the NEXT step = currently alive slots
                env_next_active = np.array([a.alive for a in env._agents], dtype=bool)
                # Auto-reset any env where the entire population has collapsed.
                # _try_reproduce requires >=2 survivors, so once population drops
                # below 2 the env permanently dies out. We reset rather than
                # leaving a dead env collecting empty rollouts.
                if not env_next_active.any():
                    reset_seed = args.seed + e_i + 1_000_000 * (it + 1)
                    o, info = env.reset_all(seed=reset_seed)
                    next_obs[e_i] = o
                    next_masks[e_i] = info["action_masks"]
                    env_next_active = info["active_mask"]
                next_active[e_i] = env_next_active

            buffer.add(
                obs=obs,
                action_mask=action_masks,
                action=action,
                logprob=logp,
                value=value,
                reward=rewards,
                done=dones,
                active=active_mask,
            )

            # Episode-level accounting
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
            action_masks = next_masks
            active_mask = next_active

        # Bootstrap value for GAE
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            mask_t = torch.as_tensor(action_masks, dtype=torch.bool, device=device)
            _logits, last_value = policy(obs_t.reshape(-1, obs_dim))
            last_value = last_value.reshape(args.n_envs, args.n_agents)
            last_active = torch.as_tensor(active_mask, dtype=torch.bool, device=device)

        adv, ret = buffer.compute_gae(
            last_value=last_value,
            last_active=last_active,
            gamma=args.gamma,
            lam=args.lam,
        )
        batch = buffer.flatten_active(adv, ret)
        stats = ppo_update(
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
                "iter=%d step=%d sps=%.1f pg=%.4f vf=%.4f ent=%.4f kl=%.4f "
                "clip=%.3f active_n=%d life_reward_mean=%.3f affinity_l1=%.3f",
                it, global_step, sps,
                stats.pg_loss, stats.vf_loss, stats.entropy, stats.approx_kl,
                stats.clip_frac, len(batch),
                float(np.mean(recent)), affinity_avg,
            )
        if wandb is not None:
            wandb.log({
                "train/pg_loss": stats.pg_loss,
                "train/vf_loss": stats.vf_loss,
                "train/entropy": stats.entropy,
                "train/approx_kl": stats.approx_kl,
                "train/clip_frac": stats.clip_frac,
                "train/n_active": len(batch),
                "rollout/sps": sps,
                "rollout/life_reward_mean": float(np.mean(recent)),
                "rollout/affinity_l1": affinity_avg,
                "global_step": global_step,
            })

        if (it + 1) % args.checkpoint_interval == 0 or it == total_iters - 1:
            ckpt = ckpt_dir / f"ippo_iter_{it + 1:06d}.pt"
            torch.save(
                {
                    "iter": it + 1,
                    "global_step": global_step,
                    "policy": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                },
                ckpt,
            )
            logger.info("checkpoint saved: %s", ckpt)

        summary = {
            "iter": it + 1,
            "global_step": global_step,
            "pg_loss": stats.pg_loss,
            "vf_loss": stats.vf_loss,
            "entropy": stats.entropy,
            "approx_kl": stats.approx_kl,
            "n_active": len(batch),
            "affinity_l1": affinity_avg,
            "life_reward_mean": float(np.mean(recent)),
            "completed_lives": len(completed_lives_reward),
        }

    if wandb is not None:
        wandb.finish()
    return summary


if __name__ == "__main__":
    train(parse_args())
