"""IPPOEnv — multi-agent step API on top of CombatEnv (P2.1, IPPO migration).

The single-focal ``CombatEnv.step(action: int)`` is preserved unchanged for
back-compat with existing tests and SB3 RecurrentPPO training. IPPOEnv adds
``step_all(actions: np.ndarray)`` which advances all slots in one call:

    obs, rewards, terminated, truncated, info = env.step_all(actions)
        obs        : np.ndarray, shape (n_agents, obs_dim)
        rewards    : np.ndarray, shape (n_agents,)         — float64
        terminated : np.ndarray, shape (n_agents,)         — bool, per-slot death
        truncated  : np.ndarray, shape (n_agents,)         — bool, env-level
        info       : dict — augmented with:
            'active_mask'        : bool[n_agents], True for slots alive at step start
            'action_masks_post'  : bool[n_agents, n_actions], usable for next step
            'per_agent_reward'   : same as rewards (already from P1.2)
            'agent_events'       : per-slot AgentStepEvents (P1.2)
            'lifecycle'          : per-slot died/born/life_id/alive (P0.3)

Implementation notes
--------------------
This is a thin wrapper. It delegates to ``CombatEnv.step`` via the existing
``_action_overrides`` mechanism — the focal slot's action runs through the
focal path (full reward shaping); non-focal slots' actions run through
``_execute_override_action`` (subset of shaping, sufficient for IPPO P3).
Per-agent reward and events are populated by P1.2 inside the underlying
``step`` and surfaced unchanged.

Limitations (intentional, deferred to later phases)
---------------------------------------------------
- P2.3: Slot state on rebirth is not yet centrally reset. Tracked separately.
- Non-focal slots only receive base + combat-shaping reward (no exploration,
  flank, betrayal, food share, group cohesion, train, resistance, stash
  proximity). Sufficient for proof-of-life IPPO; extend in P3 as needed.

P2.2: Action resolution order
-----------------------------
The focal slot is randomly chosen from currently-alive slots each step,
using the env's RNG (deterministic with seed). Logged in
``info['resolution_order']`` (length n_agents, focal first then ascending
non-focal indices). This rotates kill-steal privilege fairly across slots
over many steps.
"""
from __future__ import annotations

import numpy as np

from murimsim.actions import Action
from murimsim.rl.multi_env import CombatEnv


class IPPOEnv(CombatEnv):
    """CombatEnv subclass adding a vector ``step_all`` API for IPPO training."""

    def step_all(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """Apply one action per slot, advance the world, return per-agent tensors.

        Args:
            actions: int array of shape (n_agents,). Each entry is an Action int.

        Returns:
            (obs, rewards, terminated, truncated, info) — see module docstring.
        """
        actions = np.asarray(actions)
        assert actions.shape == (self._n_agents,), (
            f"actions must have shape ({self._n_agents},), got {actions.shape}"
        )

        # Snapshot pre-step alive mask for active_mask (P3 will use this to
        # exclude dead-at-start slots from PPO loss).
        active_mask = np.array([a.alive for a in self._agents], dtype=bool)

        # P2.2: randomize which slot is treated as the focal this step. The
        # focal slot's action runs first in CombatEnv.step (winning kill-steal
        # contention), then non-focal slots iterate in ascending index order.
        # To prevent permanent low-index privilege, pick the focal uniformly
        # from currently-alive slots using the env's RNG (deterministic with
        # seed). Over many steps, every slot has equal chance of being focal,
        # so kill-steal advantage averages out across slots.
        live_indices = np.where(active_mask)[0]
        if len(live_indices) > 0:
            focal_choice = int(self._rng.choice(live_indices))
            self._focal_idx = focal_choice
        focal_idx = self._focal_idx

        # Build resolution order = [focal, then non-focal in ascending order].
        # Logged in info for determinism debugging and downstream analysis.
        resolution_order = [focal_idx] + [
            i for i in range(self._n_agents) if i != focal_idx
        ]

        focal_action = int(actions[focal_idx])
        overrides = {
            i: int(actions[i])
            for i in range(self._n_agents)
            if i != focal_idx
        }
        prev_overrides = self._action_overrides
        self._action_overrides = overrides
        try:
            _focal_obs, _scalar_reward, _focal_term, _truncated, info = self.step(
                focal_action
            )
        finally:
            self._action_overrides = prev_overrides

        # Per-agent obs (each slot sees its own local view).
        obs_arr = np.stack(
            [self._build_obs(i) for i in range(self._n_agents)],
            axis=0,
        )

        # Per-agent reward: surfaced by P1.2.
        rewards = info["per_agent_reward"].astype(np.float64, copy=False)

        # Per-slot termination from P0.3 lifecycle metadata.
        lifecycle = info.get("lifecycle", [])
        terminated = np.array(
            [
                bool(lifecycle[i]["died"]) if i < len(lifecycle) else False
                for i in range(self._n_agents)
            ],
            dtype=bool,
        )
        # Truncated is env-level (same across all slots).
        truncated = np.full(self._n_agents, bool(_truncated), dtype=bool)

        # Per-agent action masks for next-step policy sampling.
        action_masks_post = np.stack(
            [self.action_masks(i) for i in range(self._n_agents)],
            axis=0,
        ).astype(bool, copy=False)

        info["active_mask"] = active_mask
        info["action_masks_post"] = action_masks_post
        info["resolution_order"] = resolution_order

        return obs_arr, rewards, terminated, truncated, info
