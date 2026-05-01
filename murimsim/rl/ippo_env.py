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

from murimsim.actions import (
    Action,
    BODY_TO_LEGACY,
    BodyAction,
    N_BODY_ACTIONS,
    N_SOCIAL_ACTIONS,
    SocialAction,
)
from murimsim.rl.multi_env import CombatEnv, REWARD_GROUP_FORMATION


class IPPOEnv(CombatEnv):
    """CombatEnv subclass adding a vector ``step_all`` API for IPPO training."""

    def reset_all(
        self, *, seed: int | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset env and return per-agent observations + initial action masks.

        Returns:
            obs        : np.ndarray, shape (n_agents, obs_dim)
            info       : dict with
                'action_masks' : bool[n_agents, n_actions], pre-action masks
                                  for the first step of the new episode
                'active_mask'  : bool[n_agents], True for slots alive at reset
        """
        _focal_obs, base_info = self.reset(seed=seed)
        obs_arr = np.stack(
            [self._build_obs(i) for i in range(self._n_agents)],
            axis=0,
        )
        action_masks = np.stack(
            [self.action_masks(i) for i in range(self._n_agents)],
            axis=0,
        ).astype(bool, copy=False)
        active_mask = np.array([a.alive for a in self._agents], dtype=bool)
        info = dict(base_info) if base_info else {}
        info["action_masks"] = action_masks
        info["active_mask"] = active_mask
        # v24 joint-action: also surface body+social masks for joint trainers.
        info["action_masks_body"] = np.stack(
            [self.action_masks_body(i) for i in range(self._n_agents)],
            axis=0,
        ).astype(bool, copy=False)
        info["action_masks_social"] = np.stack(
            [self.action_masks_social(i) for i in range(self._n_agents)],
            axis=0,
        ).astype(bool, copy=False)
        return obs_arr, info

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

    def step_all_joint(
        self,
        body_actions: np.ndarray,
        social_actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """v24 joint-action step: body + social heads run side by side.

        Body actions resolve through the legacy ``CombatEnv.step`` path
        (movement, combat, gather, train, etc.). Social actions resolve
        AFTER the body step, as free signals that don't preempt the body.

        Args:
            body_actions: int array (n_agents,) of ``BodyAction`` values.
            social_actions: int array (n_agents,) of ``SocialAction`` values.

        Returns:
            (obs, rewards, terminated, truncated, info). info also contains
            ``action_masks_body_post`` (n_agents, 16) and
            ``action_masks_social_post`` (n_agents, 2).
        """
        body_actions = np.asarray(body_actions)
        social_actions = np.asarray(social_actions)
        assert body_actions.shape == (self._n_agents,), (
            f"body_actions shape {body_actions.shape} != ({self._n_agents},)"
        )
        assert social_actions.shape == (self._n_agents,), (
            f"social_actions shape {social_actions.shape} != ({self._n_agents},)"
        )

        # Translate body indices to the legacy Action ints CombatEnv.step expects.
        legacy_actions = np.array(
            [BODY_TO_LEGACY[int(b)] for b in body_actions],
            dtype=np.int64,
        )

        # Run the existing body-only path.
        obs_arr, rewards, terminated, truncated, info = self.step_all(legacy_actions)

        # Resolve social actions AFTER the body step. Iterate in the same
        # focal-first / ascending-non-focal order so any ordering effects on
        # group formation match how body actions resolve.
        resolution_order = info["resolution_order"]
        for slot in resolution_order:
            if int(social_actions[slot]) != int(SocialAction.COLLABORATE):
                continue
            # Skip if the slot was dead at step start (matches body semantics).
            if not info["active_mask"][slot]:
                continue
            agent = self._agents[slot]
            if not agent.alive:
                continue
            # Re-check mask post-body — a body action may have moved the agent
            # away from neighbours, invalidating COLLABORATE eligibility.
            if not bool(self.action_masks_social(slot)[1]):
                continue
            formed = self._try_collaborate(slot)
            if formed and self._enable_formation_bonus:
                rewards[slot] += REWARD_GROUP_FORMATION

        # Refresh per-agent info that depends on social outcomes.
        info["per_agent_reward"] = rewards.astype(np.float32, copy=False)

        # Surface joint-action masks for next step.
        info["action_masks_body_post"] = np.stack(
            [self.action_masks_body(i) for i in range(self._n_agents)],
            axis=0,
        ).astype(bool, copy=False)
        info["action_masks_social_post"] = np.stack(
            [self.action_masks_social(i) for i in range(self._n_agents)],
            axis=0,
        ).astype(bool, copy=False)

        return obs_arr, rewards, terminated, truncated, info
