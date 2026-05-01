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
    p.add_argument(
        "--enable-carry-cost",
        action="store_true",
        help="v21c: inventory weight reduces combat strength and TRAIN gain.",
    )
    p.add_argument(
        "--arena-mix",
        type=str,
        default=None,
        help=(
            "v22: round-robin mix of arena configs across n_envs. Format: "
            "'base:1,arena_minion:2,arena_boss:1' — each token resolves to "
            "config/envs/<name>.yaml ('base' uses --config). Per-arena flags "
            "(enable_boss, n_minions) are read from the YAML's `arena` section. "
            "If omitted, all envs use --config with global flags."
        ),
    )
    p.add_argument(
        "--n-policy-agents",
        type=int,
        default=None,
        help=(
            "v23 FSP: number of slots [0..N) controlled by the TRAINING policy. "
            "Remaining slots [N..n_agents) are controlled by --frozen-ckpt. "
            "Default = n_agents (no frozen baseline; vanilla IPPO)."
        ),
    )
    p.add_argument(
        "--frozen-ckpt",
        type=str,
        default=None,
        help=(
            "v23 FSP: checkpoint to load as the frozen baseline policy for "
            "non-training slots. Required if n_policy_agents < n_agents."
        ),
    )
    return p.parse_args()


# v22: arena-mix support — load alternate env configs and per-arena flags.
ARENA_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config" / "envs"


def _parse_arena_mix(spec: str) -> list[str]:
    """Parse 'base:1,arena_minion:2,arena_boss:1' into a flat list of arena names."""
    out: list[str] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            name, count = token.split(":", 1)
            out.extend([name.strip()] * int(count))
        else:
            out.append(token)
    return out


def _load_arena_config(arena_name: str, base_cfg: dict) -> tuple[dict, dict]:
    """Return (env_config, arena_flags) for an arena name.

    ``arena_flags`` keys: enable_boss (bool), enable_carry_cost (bool), n_minions (int).
    'base' returns (base_cfg, {}) — caller layers global CLI flags on top.
    """
    if arena_name == "base":
        return base_cfg, {}
    path = ARENA_CONFIG_DIR / f"{arena_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Arena config not found: {path}")
    with open(path) as f:
        full = yaml.safe_load(f)
    arena_flags = full.pop("arena", {}) or {}
    return full, arena_flags


def build_envs(
    config: dict,
    n_envs: int,
    n_agents: int,
    seed: int,
    enable_boss: bool = False,
    enable_carry_cost: bool = False,
    arena_mix: str | None = None,
    enable_formation_bonus: bool = True,
) -> list[IPPOEnv]:
    envs: list[IPPOEnv] = []
    if arena_mix:
        arena_names = _parse_arena_mix(arena_mix)
        if not arena_names:
            raise ValueError(f"Empty arena-mix: {arena_mix!r}")
    else:
        arena_names = None
    for i in range(n_envs):
        if arena_names is not None:
            arena = arena_names[i % len(arena_names)]
            env_cfg, arena_flags = _load_arena_config(arena, config)
            env = IPPOEnv(
                config=env_cfg,
                n_agents=n_agents,
                seed=seed + i,
                curriculum_ramp_steps=0,
                enable_boss=bool(arena_flags.get("enable_boss", enable_boss)),
                enable_carry_cost=bool(
                    arena_flags.get("enable_carry_cost", enable_carry_cost)
                ),
                n_minions=int(arena_flags.get("n_minions", 0)),
                enable_formation_bonus=enable_formation_bonus,
            )
        else:
            env = IPPOEnv(
                config=config,
                n_agents=n_agents,
                seed=seed + i,
                curriculum_ramp_steps=0,
                enable_boss=enable_boss,
                enable_carry_cost=enable_carry_cost,
                enable_formation_bonus=enable_formation_bonus,
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
        enable_carry_cost=getattr(args, "enable_carry_cost", False),
        arena_mix=getattr(args, "arena_mix", None),
    )
    obs, action_masks, active_mask = collect_initial_state(envs, args.seed)

    # v23 FSP: split agent slots into [0..n_policy) trained + [n_policy..n_agents) frozen.
    n_policy = getattr(args, "n_policy_agents", None) or args.n_agents
    n_frozen = args.n_agents - n_policy
    if n_frozen < 0:
        raise ValueError(f"n_policy_agents ({n_policy}) > n_agents ({args.n_agents})")
    if n_frozen > 0 and not getattr(args, "frozen_ckpt", None):
        raise ValueError("--frozen-ckpt required when n_policy_agents < n_agents")

    policy = SharedActorCritic(obs_dim=obs_dim, n_actions=n_actions).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)

    frozen_policy = None
    frozen_is_recurrent = False
    frozen_hidden = None
    if n_frozen > 0:
        from scripts.eval_ippo import load_policy
        frozen_policy, _frozen_args, frozen_is_recurrent = load_policy(
            args.frozen_ckpt, device=device,
        )
        for p in frozen_policy.parameters():
            p.requires_grad_(False)
        frozen_policy.eval()
        if frozen_is_recurrent:
            # Per-env per-frozen-slot hidden state.
            frozen_hidden = [
                frozen_policy.initial_hidden(n_frozen, device)
                for _ in range(args.n_envs)
            ]
        logger.info(
            "FSP enabled: n_policy=%d n_frozen=%d frozen_ckpt=%s recurrent=%s",
            n_policy, n_frozen, args.frozen_ckpt, frozen_is_recurrent,
        )

    buffer = RolloutBuffer(
        rollout_length=args.rollout_length,
        n_envs=args.n_envs,
        n_agents=n_policy,
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

    transitions_per_iter = args.rollout_length * args.n_envs * n_policy
    total_iters = max(1, args.total_steps // transitions_per_iter)
    logger.info(
        "starting IPPO: %d iters × %d transitions = %d (target: %d) "
        "[n_policy=%d, n_frozen=%d]",
        total_iters, transitions_per_iter, total_iters * transitions_per_iter,
        args.total_steps, n_policy, n_frozen,
    )

    # Episode-level accounting (population continuum — no real episodes).
    # Sized to n_policy: we only track training-policy slot lives for log metrics.
    ep_reward_sum = np.zeros((args.n_envs, n_policy), dtype=np.float64)
    ep_step_count = np.zeros((args.n_envs, n_policy), dtype=np.int64)
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
                # POLICY forward — only the first n_policy slots per env.
                pol_obs = obs_t[:, :n_policy].reshape(-1, obs_dim)
                pol_mask = mask_t[:, :n_policy].reshape(-1, n_actions)
                action_flat, logp_flat, value_flat = policy.act(pol_obs, pol_mask)
                pol_action = action_flat.reshape(args.n_envs, n_policy)
                pol_logp = logp_flat.reshape(args.n_envs, n_policy)
                pol_value = value_flat.reshape(args.n_envs, n_policy)

                # FROZEN forward — slots [n_policy..n_agents).
                if n_frozen > 0:
                    frz_obs_full = obs_t[:, n_policy:]  # (n_envs, n_frozen, obs_dim)
                    frz_mask_full = mask_t[:, n_policy:]
                    if frozen_is_recurrent:
                        # Step per-env so each env has its own hidden state.
                        frz_actions_per_env = []
                        for e_i in range(args.n_envs):
                            f_act, _, _, f_h = frozen_policy.act(
                                frz_obs_full[e_i], frz_mask_full[e_i],
                                frozen_hidden[e_i],
                            )
                            frozen_hidden[e_i] = f_h
                            frz_actions_per_env.append(f_act)
                        frz_action = torch.stack(frz_actions_per_env, dim=0)
                    else:
                        frz_obs_flat = frz_obs_full.reshape(-1, obs_dim)
                        frz_mask_flat = frz_mask_full.reshape(-1, n_actions)
                        f_act, _, _ = frozen_policy.act(frz_obs_flat, frz_mask_flat)
                        frz_action = f_act.reshape(args.n_envs, n_frozen)
                    full_action = torch.cat([pol_action, frz_action], dim=1)
                else:
                    full_action = pol_action

            actions_np = full_action.cpu().numpy()
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
                env_next_active = np.array([a.alive for a in env._agents], dtype=bool)
                if not env_next_active.any():
                    reset_seed = args.seed + e_i + 1_000_000 * (it + 1)
                    o, info = env.reset_all(seed=reset_seed)
                    next_obs[e_i] = o
                    next_masks[e_i] = info["action_masks"]
                    env_next_active = info["active_mask"]
                    # Reset frozen hidden state for any frozen slots in this env.
                    if frozen_is_recurrent and n_frozen > 0:
                        frozen_hidden[e_i] = frozen_policy.initial_hidden(n_frozen, device)
                next_active[e_i] = env_next_active

            # Buffer stores ONLY policy slots — losses computed on policy slots only.
            buffer.add(
                obs=obs[:, :n_policy],
                action_mask=action_masks[:, :n_policy],
                action=pol_action,
                logprob=pol_logp,
                value=pol_value,
                reward=rewards[:, :n_policy],
                done=dones[:, :n_policy],
                active=active_mask[:, :n_policy],
            )

            # Episode-level accounting (policy slots only)
            ep_reward_sum += rewards[:, :n_policy]
            ep_step_count += active_mask[:, :n_policy].astype(np.int64)
            died = dones[:, :n_policy]
            if died.any():
                for e_i in range(args.n_envs):
                    for a_i in range(n_policy):
                        if died[e_i, a_i]:
                            completed_lives_reward.append(float(ep_reward_sum[e_i, a_i]))
                            completed_lives_steps.append(int(ep_step_count[e_i, a_i]))
                            ep_reward_sum[e_i, a_i] = 0.0
                            ep_step_count[e_i, a_i] = 0

            obs = next_obs
            action_masks = next_masks
            active_mask = next_active

        # Bootstrap value for GAE — policy slots only
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            mask_t = torch.as_tensor(action_masks, dtype=torch.bool, device=device)
            _logits, last_value = policy(obs_t[:, :n_policy].reshape(-1, obs_dim))
            last_value = last_value.reshape(args.n_envs, n_policy)
            last_active = torch.as_tensor(
                active_mask[:, :n_policy], dtype=torch.bool, device=device,
            )

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
