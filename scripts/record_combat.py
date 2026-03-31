"""record_combat.py — Record a multi-agent combat simulation for the web viewer.

Runs CombatEnv with the trained limbic_v2 model (or heuristic fallback).
All agents are recorded every step so the viewer shows a full 10-agent world.

Usage (from project root):
    python scripts/record_combat.py
    python scripts/record_combat.py --ticks 1000 --seed 7 --agents 10
    python scripts/record_combat.py --model checkpoints/limbic_v2/limbic_v2_final.zip
    python scripts/record_combat.py --output logs/replays/my_replay.jsonl
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from murimsim.actions import Action
from murimsim.replay import ReplayLogger
from murimsim.rl.multi_env import CombatEnv
from murimsim.sect import DEFAULT_SECTS

CONFIG_PATH = Path(__file__).parent.parent / "config" / "default.yaml"
TRAIN_CONFIG_PATH = Path(__file__).parent.parent / "config" / "training.yaml"
OUTPUT_DIR = Path(os.environ.get("MURIM_REPLAY_DIR", "/mnt/c/Users/bhara/Downloads/replays"))
DEFAULT_MODEL = Path(__file__).parent.parent / "checkpoints" / "limbic_lstm_v2" / "limbic_lstm_v2_final.zip"


def _world_resources(env: CombatEnv) -> dict[str, list[list[int | float]]]:
    """Snapshot of all resource tiles: {resource_id: [[x, y, 1.0], ...]}."""
    world = env._world
    snapshot: dict[str, list[list[int | float]]] = {}
    for rid in world.get_resource_ids():
        grid = world.get_grid_view(rid)
        ys, xs = np.where(grid > 0)
        snapshot[rid] = [[int(x), int(y), 1.0] for x, y in zip(xs, ys)]
    return snapshot


def _agent_snapshots(env: CombatEnv) -> list[dict]:
    """Snapshot of all agents' current state for the viewer.

    Uses the env's own per-agent action/detail tracking so heuristic agents
    also report their real last action (not a stale value from a prior focal turn).
    Attaches group membership so the viewer can draw collaborator maps.
    """
    # Build group map: agent_id -> list of group-member agent_ids
    group_map: dict[str, list[str]] = {}
    for group in env._groups:
        members = [env._agents[i].agent_id for i in group if env._agents[i].alive]
        for agent_id in members:
            group_map[agent_id] = [m for m in members if m != agent_id]

    snapshots = []
    for i, agent in enumerate(env._agents):
        d = agent.to_replay_dict(
            action=env._last_action_names[i],
            action_detail=env._last_action_details[i],
        )
        d["collaborators"] = group_map.get(agent.agent_id, [])
        snapshots.append(d)
    return snapshots


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a MurimSim combat replay.")
    parser.add_argument("--ticks", type=int, default=800,
                        help="Number of env steps to record (default: 800)")
    parser.add_argument("--seed", type=int, default=42, help="World seed (default: 42)")
    parser.add_argument("--agents", type=int, default=10, help="Number of agents (default: 10)")
    parser.add_argument("--model", type=Path, default=None,
                        help="PPO checkpoint (default: limbic_v2_final.zip)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output .jsonl path (default: logs/replays/combat_{seed}.jsonl)")
    parser.add_argument("--no-combat", action="store_true",
                        help="Disable combat curriculum (foraging-only replay)")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    with open(TRAIN_CONFIG_PATH) as f:
        train_cfg = yaml.safe_load(f)
    if "domain_randomization" in train_cfg:
        config["domain_randomization"] = train_cfg["domain_randomization"]

    model_path = args.model or DEFAULT_MODEL
    model = None
    model_obs_size: int | None = None
    is_recurrent = False
    if model_path.exists():
        try:
            # Try RecurrentPPO first (LSTM checkpoints)
            from sb3_contrib import RecurrentPPO
            model = RecurrentPPO.load(str(model_path), device="cpu")
            model_obs_size = model.observation_space.shape[0]
            policy_name = f"RecurrentPPO ({model_path.name})"
            is_recurrent = True
        except Exception:
            try:
                from stable_baselines3 import PPO
                model = PPO.load(str(model_path), device="cpu")
                model_obs_size = model.observation_space.shape[0]
                policy_name = f"PPO ({model_path.name})"
            except Exception as e:
                print(f"[WARN] Could not load model ({e}). Using heuristic fallback.", file=sys.stderr)
    if model is None:
        policy_name = "heuristic"

    env = CombatEnv(config=config, n_agents=args.agents, seed=args.seed)
    env_obs_size: int = env.observation_space.shape[0]

    # Assign round-robin sect IDs for viewer coloring (no home-region spawning in recording mode)
    def _assign_sect_ids() -> None:
        for i, agent in enumerate(env._agents):
            agent.sect_id = DEFAULT_SECTS.by_index(i).sect_id

    # Compat shim: if model was trained on a smaller obs space (e.g. M3=209 vs M4=234),
    # strip the extra features by dropping the enemy-stash channel from the stash section.
    # Layout: resources(100) + agents(75) + stash(N) + stats(9)
    # M3 stash was 1 channel (25 values); M4 stash is 2 channels (50 values, interleaved).
    def _compat_obs(obs: np.ndarray) -> np.ndarray:
        if model_obs_size is None or model_obs_size == env_obs_size:
            return obs
        # Determine stash sizes
        static_prefix = 100 + 75   # resource + agent grids
        stats_size = 9
        env_stash_size = env_obs_size - static_prefix - stats_size
        model_stash_size = model_obs_size - static_prefix - stats_size
        if env_stash_size > model_stash_size and env_stash_size % model_stash_size == 0:
            # Interleaved channels: take channel-0 (my_stash) values
            n_stash_cells = model_stash_size
            n_channels = env_stash_size // model_stash_size
            stash_flat = obs[static_prefix : static_prefix + env_stash_size]
            stash_ch0 = stash_flat.reshape(n_stash_cells, n_channels)[:, 0]
            return np.concatenate([obs[:static_prefix], stash_ch0, obs[-stats_size:]])
        # Fallback: truncate
        return obs[:model_obs_size]

    obs, _ = env.reset(seed=args.seed)
    _assign_sect_ids()

    # Enable full combat immediately (skip curriculum ramp for interesting replays)
    if not args.no_combat:
        env._global_step_count = env._curriculum_ramp_steps

    # Default filename: combat_<model_stem>.jsonl (e.g. combat_v16.jsonl from limbic_lstm_v16_final.zip)
    # Strip common prefix/suffix to get a clean version tag, fall back to seed if no model.
    if args.output:
        out_path = args.output
    elif model_path.exists():
        stem = model_path.stem  # e.g. "limbic_lstm_v16_final"
        # Extract version tag: last segment that starts with 'v' and a digit, else full stem
        parts = stem.split("_")
        tag = next((p for p in reversed(parts) if p.startswith("v") and p[1:].isdigit()), stem)
        out_path = OUTPUT_DIR / f"combat_{tag}.jsonl"
    else:
        out_path = OUTPUT_DIR / f"combat_{args.seed}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_filename = out_path.name

    compat_note = (
        f" [compat: env={env_obs_size} → model={model_obs_size}]"
        if model_obs_size and model_obs_size != env_obs_size
        else ""
    )
    print(f"Policy   : {policy_name}{compat_note}")
    print(f"Agents   : {args.agents}  |  Combat: {'enabled' if not args.no_combat else 'disabled'}")
    print(f"World    : {env._world.grid_size}×{env._world.grid_size}  seed={args.seed}")
    print(f"Steps    : {args.ticks}")
    print(f"Output   : {out_path}\n")

    action_counts: dict[str, int] = {}
    step = 0
    generation = 0
    n = args.agents

    # Per-agent LSTM states and episode-start flags for full-team inference.
    # Each agent runs the model independently with its own hidden state.
    agent_lstm_states: list = [None] * n
    agent_ep_starts: list[bool] = [True] * n

    def _reset_agent_lstm(idx: int) -> None:
        agent_lstm_states[idx] = None
        agent_ep_starts[idx] = True

    def _get_all_actions() -> dict[int, int]:
        """Run the model for every alive agent. Returns {agent_idx: action_int}."""
        actions: dict[int, int] = {}
        for i, agent in enumerate(env._agents):
            if not agent.alive:
                continue
            obs_i = env._build_obs(i)
            ep_start_i = np.array([agent_ep_starts[i]], dtype=bool)
            if is_recurrent:
                act, new_state = model.predict(
                    _compat_obs(obs_i),
                    state=agent_lstm_states[i],
                    episode_start=ep_start_i,
                    deterministic=True,
                )
                agent_lstm_states[i] = new_state
                agent_ep_starts[i] = False
            else:
                act, _ = model.predict(_compat_obs(obs_i), deterministic=True)
            actions[i] = int(act)
        return actions

    with ReplayLogger(seed=args.seed, output_dir=out_path.parent, filename=out_filename) as replay:
        while step < args.ticks:
            focal_idx = env._focal_idx

            if model is not None:
                # Full-team inference: every alive agent runs the policy this tick.
                all_actions = _get_all_actions()
                action_int = all_actions.get(focal_idx, 0)
                # Inject non-focal actions so _execute_override_action replaces heuristics.
                env._action_overrides = {i: a for i, a in all_actions.items() if i != focal_idx}
            else:
                action_int = env.action_space.sample()
                env._action_overrides = None

            action_name = Action(action_int).name.lower()
            action_counts[action_name] = action_counts.get(action_name, 0) + 1

            # Log tick BEFORE step so the snapshot matches what the agent decided this tick
            replay.log_tick(
                tick=step,
                generation=generation,
                agents=_agent_snapshots(env),
                resources=_world_resources(env),
                events=[],
            )

            obs, _reward, terminated, _truncated, _info = env.step(action_int)
            step += 1

            if terminated:
                alive_agents = [i for i, a in enumerate(env._agents) if a.alive]
                if not alive_agents:
                    # All dead — start a new generation (fresh world) and continue
                    generation += 1
                    obs, _ = env.reset(seed=args.seed + generation)
                    if not args.no_combat:
                        env._global_step_count = env._curriculum_ramp_steps
                    agent_lstm_states = [None] * n
                    agent_ep_starts = [True] * n
                else:
                    # Focal died — rotate to next alive agent; reset only focal's LSTM state.
                    _reset_agent_lstm(focal_idx)
                    env._focal_idx = env._next_live(focal_idx)
                    agent_ep_starts[env._focal_idx] = True

    print(f"{'─'*55}")
    print(f"  Steps recorded : {step}")
    print(f"  Generations    : {generation + 1}")
    print(f"  Action breakdown:")
    for act, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / max(1, step)
        print(f"    {act:<14} {count:4d}  ({pct:.1f}%)")
    print(f"{'─'*55}")
    print(f"\nReplay saved → {out_path}")
    print(f"Open viewer: python -m http.server 8000  then  http://localhost:8000/viewer/")


if __name__ == "__main__":
    main()
