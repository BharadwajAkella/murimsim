"""scripts/eval_ippo.py — Phase 5 validation harness for IPPO checkpoints.

Loads a feedforward (`scripts.train_ippo`) OR recurrent (`scripts.train_ippo_recurrent`)
checkpoint and rolls it out across N steps with EVERY slot policy-controlled
(no heuristic non-focal). Reports the metrics that matter for "did the
affinity/coalition feedback loop actually close":

    * max_abs_affinity         — strongest dyadic bond seen across the run.
    * mean_abs_affinity        — average non-zero affinity magnitude.
    * dyadic_reciprocity       — Pearson r between aff(i→j) and aff(j→i)
                                 across all observed dyads. Strong + r ⇒ mutual.
    * n_active_groups          — number of multi-agent groups at end of rollout.
    * mean_group_size          — average size of a group at end of rollout.
    * mean_life_reward         — mean cumulative reward across completed lives.
    * mean_lifespan            — mean number of steps a life lasted.
    * help_events              — count of distinct helper→recipient pairs.

The numbers are returned as a plain dict so callers / tests can introspect.

Usage:

    python -m scripts.eval_ippo --checkpoint checkpoints/ippo_v1/ippo_iter_000048.pt
    python -m scripts.eval_ippo --checkpoint checkpoints/ippo_recurrent_v1/ippo_recurrent_iter_000016.pt --steps 5000

Auto-detects FF vs recurrent from the checkpoint payload's ``hidden_dim`` key.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from murimsim.actions import N_ACTIONS_PHASE6_QI, N_BODY_ACTIONS, N_SOCIAL_ACTIONS
from murimsim.rl.ippo import SharedActorCritic
from murimsim.rl.ippo_env import IPPOEnv
from murimsim.rl.ippo_joint import JointSharedActorCritic
from murimsim.rl.ippo_joint_recurrent import JointRecurrentSharedActorCritic
from murimsim.rl.ippo_recurrent import RecurrentSharedActorCritic
from murimsim.rl.multi_env import OBS_TOTAL_SIZE

logger = logging.getLogger("eval_ippo")


@dataclass
class EvalMetrics:
    """All affinity-loop closure metrics computed by ``eval_checkpoint``."""

    steps: int
    n_envs: int
    n_agents: int
    is_recurrent: bool
    max_abs_affinity: float
    mean_abs_affinity: float
    dyadic_reciprocity: float
    n_active_groups: float
    mean_group_size: float
    mean_life_reward: float
    mean_lifespan: float
    help_events: int
    completed_lives: int


# ---------------------------------------------------------------------------
# Metric primitives — pure functions of env state, easy to unit-test.
# ---------------------------------------------------------------------------

def affinity_summary(envs: list[IPPOEnv]) -> tuple[float, float, float]:
    """Return (max_abs, mean_abs, reciprocity) over all dyads across envs.

    ``mean_abs`` averages over recorded non-zero entries only; an env that
    never updates affinity gets mean=0 (no dyads stored).

    Reciprocity = Pearson correlation between paired (a→b, b→a) affinity
    values. Returns 0.0 if fewer than 2 paired dyads exist.
    """
    paired: list[tuple[float, float]] = []
    all_abs: list[float] = []
    for env in envs:
        raw = env._affinity_raw
        for i, row in raw.items():
            for j, (val, _step) in row.items():
                if i == j:
                    continue
                all_abs.append(abs(float(val)))
                # Only count each dyad once per env (i < j) for reciprocity.
                if i < j and j in raw and i in raw[j]:
                    paired.append((float(val), float(raw[j][i][0])))

    max_abs = max(all_abs) if all_abs else 0.0
    mean_abs = float(np.mean(all_abs)) if all_abs else 0.0

    if len(paired) < 2:
        recip = 0.0
    else:
        a = np.array([p[0] for p in paired])
        b = np.array([p[1] for p in paired])
        if a.std() < 1e-12 or b.std() < 1e-12:
            recip = 0.0
        else:
            recip = float(np.corrcoef(a, b)[0, 1])
    return max_abs, mean_abs, recip


def group_summary(envs: list[IPPOEnv]) -> tuple[float, float]:
    """Mean number of groups per env and mean group size at this snapshot."""
    n_groups_per_env: list[int] = []
    sizes: list[int] = []
    for env in envs:
        groups = [g for g in env._groups if len(g) >= 2]
        n_groups_per_env.append(len(groups))
        for g in groups:
            sizes.append(len(g))
    n_groups = float(np.mean(n_groups_per_env)) if n_groups_per_env else 0.0
    mean_size = float(np.mean(sizes)) if sizes else 0.0
    return n_groups, mean_size


def help_event_count(envs: list[IPPOEnv]) -> int:
    """Total number of (recipient, helper) pairs recorded across all envs."""
    total = 0
    for env in envs:
        for recipient_dict in env._help_received.values():
            total += len(recipient_dict)
    return total


# ---------------------------------------------------------------------------
# Checkpoint loader (auto-detects FF vs recurrent)
# ---------------------------------------------------------------------------

def load_policy(
    checkpoint_path: str | Path, device: torch.device | str = "cpu"
) -> tuple[torch.nn.Module, dict, bool]:
    """Load a checkpoint and rebuild the matching policy.

    Returns (policy, args_dict, is_recurrent). ``is_recurrent`` is inferred
    from presence of ``hidden_dim`` in the saved payload (recurrent trainer
    writes this field; FF trainer does not).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    is_recurrent = "hidden_dim" in ckpt or "hidden_dim" in args

    obs_dim = OBS_TOTAL_SIZE
    n_actions = N_ACTIONS_PHASE6_QI
    if is_recurrent:
        hidden_dim = ckpt.get("hidden_dim", args.get("hidden_dim", 128))
        pre_lstm_dim = args.get("pre_lstm_dim", hidden_dim)
        policy = RecurrentSharedActorCritic(
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden_dim=hidden_dim,
            pre_lstm_dim=pre_lstm_dim,
        )
    else:
        policy = SharedActorCritic(obs_dim=obs_dim, n_actions=n_actions)
    policy.load_state_dict(ckpt["policy"])
    policy.to(device)
    policy.eval()
    return policy, args, is_recurrent


def is_joint_checkpoint(checkpoint_path: str | Path) -> bool:
    """Return True if the checkpoint was written by a v24 joint-action trainer."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return bool(ckpt.get("joint_action", False))


def load_joint_policy(
    checkpoint_path: str | Path, device: torch.device | str = "cpu"
) -> tuple[torch.nn.Module, dict, bool]:
    """Load a v24 joint-action checkpoint. Returns (policy, args, is_recurrent)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    is_recurrent = bool(ckpt.get("recurrent", False)) or "hidden_dim" in ckpt
    obs_dim = OBS_TOTAL_SIZE
    if is_recurrent:
        hidden_dim = ckpt.get("hidden_dim", args.get("hidden_dim", 128))
        pre_lstm_dim = args.get("pre_lstm_dim", hidden_dim)
        policy = JointRecurrentSharedActorCritic(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            pre_lstm_dim=pre_lstm_dim,
        )
    else:
        policy = JointSharedActorCritic(obs_dim=obs_dim)
    policy.load_state_dict(ckpt["policy"])
    policy.to(device)
    policy.eval()
    return policy, args, is_recurrent


# ---------------------------------------------------------------------------
# Main eval rollout
# ---------------------------------------------------------------------------

def eval_checkpoint(
    checkpoint_path: str | Path,
    config_path: str = "config/default.yaml",
    steps: int = 5000,
    n_envs: int = 4,
    n_agents: int = 4,
    seed: int = 0,
    device: torch.device | str = "cpu",
    deterministic: bool = True,
    enable_boss: bool = False,
    enable_carry_cost: bool = False,
    arena_mix: str | None = None,
) -> EvalMetrics:
    """Run a deterministic rollout with all slots policy-controlled.

    Args:
        checkpoint_path: Path to .pt file from train_ippo or train_ippo_recurrent.
        config_path:     YAML config used to construct the env (must match training).
        steps:           Number of env steps to roll. 5000 is enough for stable
                         affinity statistics in 4-agent envs.
        n_envs, n_agents: Vector / population sizes.
        seed:            RNG seed for deterministic eval reruns.
        deterministic:   If True, use argmax over masked categorical (no exploration
                         noise). False → policy.sample() for diversity.

    Returns:
        ``EvalMetrics`` dataclass — also serializable via ``asdict``.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    is_joint = is_joint_checkpoint(checkpoint_path)
    if is_joint:
        policy, _train_args, is_recurrent = load_joint_policy(checkpoint_path, device=device)
    else:
        policy, _train_args, is_recurrent = load_policy(checkpoint_path, device=device)

    if arena_mix:
        from scripts.train_ippo import _load_arena_config, _parse_arena_mix
        arena_names = _parse_arena_mix(arena_mix)
        envs = []
        for i in range(n_envs):
            arena = arena_names[i % len(arena_names)]
            env_cfg, arena_flags = _load_arena_config(arena, cfg)
            envs.append(IPPOEnv(
                config=env_cfg, n_agents=n_agents, seed=seed + i,
                curriculum_ramp_steps=0,
                enable_boss=bool(arena_flags.get("enable_boss", enable_boss)),
                enable_carry_cost=bool(
                    arena_flags.get("enable_carry_cost", enable_carry_cost)
                ),
                n_minions=int(arena_flags.get("n_minions", 0)),
            ))
    else:
        envs = [
            IPPOEnv(
                config=cfg, n_agents=n_agents, seed=seed + i, curriculum_ramp_steps=0,
                enable_boss=enable_boss, enable_carry_cost=enable_carry_cost,
            )
            for i in range(n_envs)
        ]
    obs_list, mask_list, body_mask_list, social_mask_list = [], [], [], []
    for i, env in enumerate(envs):
        o, info = env.reset_all(seed=seed + i)
        obs_list.append(o)
        mask_list.append(info["action_masks"])
        body_mask_list.append(info["action_masks_body"])
        social_mask_list.append(info["action_masks_social"])
    obs = np.stack(obs_list, axis=0)
    action_masks = np.stack(mask_list, axis=0)
    body_mask = np.stack(body_mask_list, axis=0)
    social_mask = np.stack(social_mask_list, axis=0)

    obs_dim = OBS_TOTAL_SIZE
    n_actions = N_ACTIONS_PHASE6_QI
    B = n_envs * n_agents

    if is_recurrent:
        carried_h, carried_c = policy.initial_hidden(B, device=device)
        pending_life_reset = np.ones((n_envs, n_agents), dtype=bool)
    else:
        carried_h = carried_c = None
        pending_life_reset = None

    completed_lives_reward: list[float] = []
    completed_lives_steps: list[int] = []
    ep_reward_sum = np.zeros((n_envs, n_agents), dtype=np.float64)
    ep_step_count = np.zeros((n_envs, n_agents), dtype=np.int64)

    for _t in range(steps):
        active_mask = np.array(
            [[a.alive for a in env._agents] for env in envs], dtype=bool
        )

        if is_recurrent and pending_life_reset.any():
            reset_t = torch.as_tensor(pending_life_reset.reshape(-1), dtype=torch.bool, device=device)
            keep = (~reset_t).view(1, B, 1).to(carried_h.dtype)
            carried_h = carried_h * keep
            carried_c = carried_c * keep

        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).reshape(-1, obs_dim)
            if is_joint:
                bm_t = torch.as_tensor(body_mask, dtype=torch.bool, device=device).reshape(-1, N_BODY_ACTIONS)
                sm_t = torch.as_tensor(social_mask, dtype=torch.bool, device=device).reshape(-1, N_SOCIAL_ACTIONS)
                if is_recurrent:
                    ba, sa, _, _, _, (carried_h, carried_c) = policy.act(
                        obs_t, bm_t, sm_t, (carried_h, carried_c), deterministic=deterministic
                    )
                else:
                    ba, sa, _, _, _ = policy.act(obs_t, bm_t, sm_t, deterministic=deterministic)
                ba_np = ba.cpu().numpy().reshape(n_envs, n_agents)
                sa_np = sa.cpu().numpy().reshape(n_envs, n_agents)
            else:
                mask_t = torch.as_tensor(action_masks, dtype=torch.bool, device=device).reshape(-1, n_actions)
                if is_recurrent:
                    action_flat, _, _, (carried_h, carried_c) = policy.act(
                        obs_t, mask_t, (carried_h, carried_c), deterministic=deterministic
                    )
                else:
                    action_flat, _, _ = policy.act(obs_t, mask_t, deterministic=deterministic)
                actions_np = action_flat.cpu().numpy().reshape(n_envs, n_agents)

        next_pending = np.zeros((n_envs, n_agents), dtype=bool) if is_recurrent else None
        for e_i, env in enumerate(envs):
            if is_joint:
                o, r, term, trunc, info = env.step_all_joint(ba_np[e_i], sa_np[e_i])
            else:
                o, r, term, trunc, info = env.step_all(actions_np[e_i])
            ep_reward_sum[e_i] += r
            ep_step_count[e_i] += active_mask[e_i].astype(np.int64)
            died = term | trunc
            for a_i in range(n_agents):
                if died[a_i]:
                    completed_lives_reward.append(float(ep_reward_sum[e_i, a_i]))
                    completed_lives_steps.append(int(ep_step_count[e_i, a_i]))
                    ep_reward_sum[e_i, a_i] = 0.0
                    ep_step_count[e_i, a_i] = 0
            if is_recurrent:
                lc = info.get("lifecycle", [])
                for a_i in range(n_agents):
                    if a_i < len(lc) and lc[a_i].get("born", False):
                        next_pending[e_i, a_i] = True

            obs[e_i] = o
            if is_joint:
                body_mask[e_i] = info["action_masks_body_post"]
                social_mask[e_i] = info["action_masks_social_post"]
            else:
                action_masks[e_i] = info["action_masks_post"]
            env_next_active = np.array([a.alive for a in env._agents], dtype=bool)
            if not env_next_active.any():
                reset_seed = seed + e_i + 1_000_000 * (_t + 1)
                o, info = env.reset_all(seed=reset_seed)
                obs[e_i] = o
                if is_joint:
                    body_mask[e_i] = info["action_masks_body"]
                    social_mask[e_i] = info["action_masks_social"]
                else:
                    action_masks[e_i] = info["action_masks"]
                if is_recurrent:
                    next_pending[e_i, :] = True

        if is_recurrent:
            pending_life_reset = next_pending

    max_abs, mean_abs, recip = affinity_summary(envs)
    n_groups, mean_size = group_summary(envs)
    helps = help_event_count(envs)

    return EvalMetrics(
        steps=steps,
        n_envs=n_envs,
        n_agents=n_agents,
        is_recurrent=is_recurrent,
        max_abs_affinity=max_abs,
        mean_abs_affinity=mean_abs,
        dyadic_reciprocity=recip,
        n_active_groups=n_groups,
        mean_group_size=mean_size,
        mean_life_reward=float(np.mean(completed_lives_reward)) if completed_lives_reward else 0.0,
        mean_lifespan=float(np.mean(completed_lives_steps)) if completed_lives_steps else 0.0,
        help_events=helps,
        completed_lives=len(completed_lives_reward),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--n-agents", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--stochastic", action="store_true",
                   help="Sample actions instead of argmax (default deterministic).")
    p.add_argument("--json", action="store_true", help="Emit metrics as JSON only.")
    p.add_argument("--enable-boss", action="store_true",
                   help="Enable boss monster in eval env (match training).")
    p.add_argument("--enable-carry-cost", action="store_true",
                   help="Enable v21c carry cost in eval env (match training).")
    p.add_argument("--arena-mix", type=str, default=None,
                   help="v22: round-robin arena mix matching training.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args()
    metrics = eval_checkpoint(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        steps=args.steps,
        n_envs=args.n_envs,
        n_agents=args.n_agents,
        seed=args.seed,
        device=args.device,
        deterministic=not args.stochastic,
        enable_boss=args.enable_boss,
        enable_carry_cost=args.enable_carry_cost,
        arena_mix=args.arena_mix,
    )
    if args.json:
        print(json.dumps(asdict(metrics), indent=2))
    else:
        logger.info("eval results for %s:", args.checkpoint)
        for k, v in asdict(metrics).items():
            logger.info("  %-22s = %s", k, v)


if __name__ == "__main__":
    main()
