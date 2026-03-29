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
OUTPUT_DIR = Path("/mnt/c/Users/bhara/Downloads/replays")
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


def _agent_snapshots(
    env: CombatEnv,
    last_actions: list[str],
    last_details: list[str],
) -> list[dict]:
    """Snapshot of all agents' current state for the viewer."""
    return [
        agent.to_replay_dict(action=last_actions[i], action_detail=last_details[i])
        for i, agent in enumerate(env._agents)
    ]


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

    out_path = args.output or (OUTPUT_DIR / f"combat_{args.seed}.jsonl")
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

    # Track last action per agent for recording
    last_actions: list[str] = ["rest"] * args.agents
    last_details: list[str] = [""] * args.agents

    action_counts: dict[str, int] = {}
    dead_agents: set[int] = set()
    step = 0
    generation = 0
    # LSTM hidden states — reset on each new generation
    lstm_states = None
    ep_start = np.ones((1,), dtype=bool)

    with ReplayLogger(seed=args.seed, output_dir=out_path.parent, filename=out_filename) as replay:
        while step < args.ticks:
            focal_idx = env._focal_idx

            if model is not None:
                obs = env._build_obs(focal_idx)
                if is_recurrent:
                    action_int, lstm_states = model.predict(
                        _compat_obs(obs),
                        state=lstm_states,
                        episode_start=ep_start,
                        deterministic=True,
                    )
                    ep_start = np.zeros((1,), dtype=bool)
                else:
                    action_int, _ = model.predict(_compat_obs(obs), deterministic=True)
                action_int = int(action_int)
            else:
                action_int = env.action_space.sample()

            action_name = Action(action_int).name.lower()
            last_actions[focal_idx] = action_name
            last_details[focal_idx] = ""
            action_counts[action_name] = action_counts.get(action_name, 0) + 1

            alive_before = {i for i, a in enumerate(env._agents) if a.alive}

            replay.log_tick(
                tick=step,
                generation=generation,
                agents=_agent_snapshots(env, last_actions, last_details),
                resources=_world_resources(env),
                events=[],
            )

            obs, _reward, terminated, _truncated, _info = env.step(action_int)
            step += 1

            # Mark newly dead agents
            alive_after = {i for i, a in enumerate(env._agents) if a.alive}
            for i in alive_before - alive_after - dead_agents:
                dead_agents.add(i)
                last_actions[i] = "dead"
                last_details[i] = "health reached 0"

            if terminated:
                alive_agents = [i for i, a in enumerate(env._agents) if a.alive]
                if not alive_agents:
                    # All dead — start a new generation (fresh world) and continue
                    generation += 1
                    obs, _ = env.reset(seed=args.seed + generation)
                    if not args.no_combat:
                        env._global_step_count = env._curriculum_ramp_steps
                    last_actions = ["rest"] * args.agents
                    last_details = [""] * args.agents
                    dead_agents = set()
                    lstm_states = None
                    ep_start = np.ones((1,), dtype=bool)
                else:
                    # Only focal died — continue in same world with next alive agent
                    env._focal_idx = env._next_live(env._focal_idx)
                    obs = env._build_obs(env._focal_idx)
                    ep_start = np.ones((1,), dtype=bool)

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
