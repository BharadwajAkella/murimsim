"""scripts/train_ippo_recurrent.py — Phase 4 recurrent IPPO trainer.

Mirrors ``scripts/train_ippo.py`` but uses an LSTM actor-critic with proper
hidden-state lifecycle:

    * Each (env, slot) carries its own (h, c).
    * Hidden is reset to zeros BEFORE the first step of each new life — driven
      by ``info['lifecycle'][i]['born']`` from the *previous* env step
      (pending_life_reset semantics).
    * When an env auto-resets after population collapse, ALL of its slots'
      hidden states are zeroed and ``pending_life_reset`` is forced True so the
      buffer records that as life_reset[t] for the next step.
    * GAE bootstrap uses the carried hidden state (not zero) but does NOT
      mutate it — see ``RecurrentSharedActorCritic.value_only``.
    * Hidden is detached across iter boundaries to cap BPTT at one rollout.

Usage:
    python -m scripts.train_ippo_recurrent --total-steps 500_000

Smoke:
    python -m scripts.train_ippo_recurrent --total-steps 4096 --n-envs 2 --no-wandb
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
from murimsim.rl.ippo_env import IPPOEnv
from murimsim.rl.ippo_recurrent import (
    RecurrentRolloutBuffer,
    RecurrentSharedActorCritic,
    recurrent_ppo_update,
)
from murimsim.rl.multi_env import OBS_TOTAL_SIZE

logger = logging.getLogger("train_ippo_recurrent")


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
    p.add_argument("--checkpoint-dir", default="checkpoints/ippo_recurrent_v1")
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
    return [
        IPPOEnv(
            config=config,
            n_agents=n_agents,
            seed=seed + i,
            curriculum_ramp_steps=0,
            enable_boss=enable_boss,
        )
        for i in range(n_envs)
    ]


def collect_initial_state(envs: list[IPPOEnv], seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    total = 0.0
    for row in env._affinity_raw.values():
        for val, _step in row.values():
            total += abs(val)
    return total


def train(args: argparse.Namespace) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    obs_dim = OBS_TOTAL_SIZE
    n_actions = N_ACTIONS_PHASE6_QI
    B = args.n_envs * args.n_agents

    envs = build_envs(
        cfg, args.n_envs, args.n_agents, args.seed,
        enable_boss=getattr(args, "enable_boss", False),
    )
    obs, action_masks, active_mask = collect_initial_state(envs, args.seed)

    policy = RecurrentSharedActorCritic(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_dim=args.hidden_dim,
        pre_lstm_dim=args.pre_lstm_dim,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)

    buffer = RecurrentRolloutBuffer(
        rollout_length=args.rollout_length,
        n_envs=args.n_envs,
        n_agents=args.n_agents,
        obs_dim=obs_dim,
        n_actions=n_actions,
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
        "starting recurrent IPPO: %d iters × %d transitions = %d (target: %d)",
        total_iters, transitions_per_iter, total_iters * transitions_per_iter, args.total_steps,
    )

    # Carried hidden across iters — detached at iter boundary.
    carried_h, carried_c = policy.initial_hidden(B, device=device)
    # All slots get a fresh life on iter 0 → pending_life_reset starts True.
    pending_life_reset = np.ones((args.n_envs, args.n_agents), dtype=bool)

    ep_reward_sum = np.zeros((args.n_envs, args.n_agents), dtype=np.float64)
    ep_step_count = np.zeros((args.n_envs, args.n_agents), dtype=np.int64)
    completed_lives_reward: list[float] = []
    completed_lives_steps: list[int] = []

    t_start = time.time()
    summary: dict = {}

    for it in range(total_iters):
        buffer.reset()
        # Snapshot the hidden state we're about to consume — this is what
        # the update will replay against.
        buffer.set_initial_hidden(carried_h, carried_c)

        for _t in range(args.rollout_length):
            # Apply pending hidden reset BEFORE acting on obs.
            if pending_life_reset.any():
                reset_flat = pending_life_reset.reshape(-1)
                reset_t = torch.as_tensor(reset_flat, dtype=torch.bool, device=device)
                keep = (~reset_t).view(1, B, 1).to(carried_h.dtype)
                carried_h = carried_h * keep
                carried_c = carried_c * keep

            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).reshape(-1, obs_dim)
                mask_t = torch.as_tensor(action_masks, dtype=torch.bool, device=device).reshape(-1, n_actions)
                action_flat, logp_flat, value_flat, (carried_h, carried_c) = policy.act(
                    obs_t, mask_t, (carried_h, carried_c)
                )
                action = action_flat.reshape(args.n_envs, args.n_agents)
                logp = logp_flat.reshape(args.n_envs, args.n_agents)
                value = value_flat.reshape(args.n_envs, args.n_agents)

            actions_np = action.cpu().numpy()
            rewards = np.zeros((args.n_envs, args.n_agents), dtype=np.float32)
            dones = np.zeros((args.n_envs, args.n_agents), dtype=bool)
            next_obs = np.zeros_like(obs)
            next_masks = np.zeros_like(action_masks)
            next_active = np.zeros_like(active_mask)
            next_pending = np.zeros((args.n_envs, args.n_agents), dtype=bool)

            stored_life_reset = pending_life_reset.copy()

            for e_i, env in enumerate(envs):
                o, r, term, trunc, info = env.step_all(actions_np[e_i])
                rewards[e_i] = r
                dones[e_i] = term | trunc
                next_obs[e_i] = o
                next_masks[e_i] = info["action_masks_post"]
                env_next_active = np.array([a.alive for a in env._agents], dtype=bool)

                # born-this-step → pending_life_reset for the NEXT step.
                lc = info.get("lifecycle", [])
                for a_i in range(args.n_agents):
                    if a_i < len(lc) and lc[a_i].get("born", False):
                        next_pending[e_i, a_i] = True

                if not env_next_active.any():
                    reset_seed = args.seed + e_i + 1_000_000 * (it + 1)
                    o, info = env.reset_all(seed=reset_seed)
                    next_obs[e_i] = o
                    next_masks[e_i] = info["action_masks"]
                    env_next_active = info["active_mask"]
                    # Force ALL of this env's slots to reset hidden next step.
                    next_pending[e_i, :] = True
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
            action_masks = next_masks
            active_mask = next_active
            pending_life_reset = next_pending

        # Bootstrap value uses carried hidden — does NOT mutate it.
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).reshape(-1, obs_dim)
            # Apply pending reset to a SCRATCH copy so we don't disturb the
            # actual carry forward — the trainer applies pending_life_reset
            # again at the start of next iter's rollout loop.
            scratch_h = carried_h.clone()
            scratch_c = carried_c.clone()
            if pending_life_reset.any():
                reset_flat = pending_life_reset.reshape(-1)
                reset_t = torch.as_tensor(reset_flat, dtype=torch.bool, device=device)
                keep = (~reset_t).view(1, B, 1).to(scratch_h.dtype)
                scratch_h = scratch_h * keep
                scratch_c = scratch_c * keep
            last_value_flat = policy.value_only(obs_t, (scratch_h, scratch_c))
            last_value = last_value_flat.reshape(args.n_envs, args.n_agents)
            last_active = torch.as_tensor(active_mask, dtype=torch.bool, device=device)

        adv, ret = buffer.compute_gae(
            last_value=last_value, last_active=last_active, gamma=args.gamma, lam=args.lam
        )
        sb = buffer.to_sequence_batch(adv, ret)
        stats = recurrent_ppo_update(
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

        # Detach hidden across iter boundary — caps BPTT at one rollout.
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
                "iter=%d step=%d sps=%.1f pg=%.4f vf=%.4f ent=%.4f kl=%.4f "
                "clip=%.3f active_n=%d life_reward_mean=%.3f affinity_l1=%.3f",
                it, global_step, sps,
                stats.pg_loss, stats.vf_loss, stats.entropy, stats.approx_kl,
                stats.clip_frac, n_active,
                float(np.mean(recent)), affinity_avg,
            )
        if wandb is not None:
            wandb.log({
                "train/pg_loss": stats.pg_loss,
                "train/vf_loss": stats.vf_loss,
                "train/entropy": stats.entropy,
                "train/approx_kl": stats.approx_kl,
                "train/clip_frac": stats.clip_frac,
                "train/n_active": n_active,
                "rollout/sps": sps,
                "rollout/life_reward_mean": float(np.mean(recent)),
                "rollout/affinity_l1": affinity_avg,
                "global_step": global_step,
            })

        if (it + 1) % args.checkpoint_interval == 0 or it == total_iters - 1:
            ckpt = ckpt_dir / f"ippo_recurrent_iter_{it + 1:06d}.pt"
            torch.save(
                {
                    "iter": it + 1,
                    "global_step": global_step,
                    "policy": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                    "hidden_dim": args.hidden_dim,
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
            "n_active": n_active,
            "affinity_l1": affinity_avg,
            "life_reward_mean": float(np.mean(recent)),
            "completed_lives": len(completed_lives_reward),
        }

    if wandb is not None:
        wandb.finish()
    return summary


if __name__ == "__main__":
    train(parse_args())
