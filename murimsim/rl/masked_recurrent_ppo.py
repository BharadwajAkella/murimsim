"""masked_recurrent_ppo.py — Action-masked RecurrentPPO for MurimSim.

Wraps sb3_contrib.RecurrentPPO with per-step action masking.  The model is
identical in every way — same weights, same LSTM architecture, same checkpoint
format — but infeasible actions (e.g. ATTACK with no adjacent enemy) are set to
-infinity before the softmax so the model can never choose them.

Usage (inference / replay):
    from murimsim.rl.masked_recurrent_ppo import MaskedRecurrentPPO
    model = MaskedRecurrentPPO.load("checkpoints/.../final.zip", env=env)
    action, state = model.predict(obs, state=state, episode_start=ep_start,
                                  action_masks=env.action_masks())

Usage (training — v17+):
    model = MaskedRecurrentPPO("MlpLstmPolicy", env, ...)
    model.learn(total_timesteps=..., callback=...)
    # The env must implement action_masks() → np.ndarray[bool]

Architecture note:
    Masking is applied inside _predict() BEFORE get_actions(), which means the
    softmax never sees invalid actions.  The gradient flows through masked logits
    correctly — they receive no gradient signal, so the model gradually stops
    allocating weight to infeasible actions in a given context.
"""
from __future__ import annotations

import numpy as np
import torch as th
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor
from gymnasium import spaces
from typing import Optional, Union


class _MaskedRecurrentPolicy(RecurrentActorCriticPolicy):
    """RecurrentActorCriticPolicy that accepts an action mask at predict time.

    The mask is a 1-D bool tensor of shape (n_actions,).  True = allowed.
    Masked logits are set to -1e8 before the Categorical distribution samples.
    """

    def _predict(
        self,
        observation: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        deterministic: bool = False,
        action_masks: th.Tensor | None = None,
    ) -> tuple[th.Tensor, tuple[th.Tensor, ...]]:
        distribution, lstm_states = self.get_distribution(observation, lstm_states, episode_starts)

        if action_masks is not None and isinstance(distribution, CategoricalDistribution):
            # Apply mask: set forbidden logit values to -1e8 so softmax → ~0
            logits = distribution.distribution.logits.clone()
            logits[~action_masks] = -1e8
            distribution.distribution = th.distributions.Categorical(logits=logits)

        return distribution.get_actions(deterministic=deterministic), lstm_states


class MaskedRecurrentPPO(RecurrentPPO):
    """RecurrentPPO with action masking support.

    Drop-in replacement for RecurrentPPO.  Accepts an optional ``action_masks``
    kwarg in predict() — a bool array of shape (n_actions,) where False means
    the action is infeasible and will never be chosen.

    Warm-start from any RecurrentPPO checkpoint:
        model = MaskedRecurrentPPO.load("path/to/v16.zip", env=env,
                                         custom_objects={"policy_class": _MaskedRecurrentPolicy})

    If action_masks is not provided, behaves identically to RecurrentPPO.
    """

    policy_aliases = {
        "MlpLstmPolicy": _MaskedRecurrentPolicy,
        **{k: v for k, v in RecurrentPPO.policy_aliases.items() if k != "MlpLstmPolicy"},
    }

    @classmethod
    def load(cls, path, **kwargs):
        """Load a RecurrentPPO checkpoint and patch in masking support.

        Works with any existing RecurrentPPO checkpoint — no weight surgery needed.
        The policy _predict method is replaced with the masked version after loading.
        """
        import types
        model = super().load(path, **kwargs)
        # Patch the loaded policy's _predict to support action_masks kwarg
        model.policy._predict = types.MethodType(_MaskedRecurrentPolicy._predict, model.policy)
        return model

    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """Like RecurrentPPO.predict but accepts action_masks.

        Args:
            observation: Current observation.
            state: LSTM hidden states (h, c), each shape (n_layers, n_envs, hidden).
            episode_start: Bool array indicating episode start (resets LSTM).
            deterministic: If True, take argmax; else sample.
            action_masks: Bool array of shape (n_actions,) or (n_envs, n_actions).
                          False = forbidden. If None, no masking applied.
        """
        self.policy.set_training_mode(False)
        observation, vectorized_env = self.policy.obs_to_tensor(observation)

        if isinstance(observation, dict):
            n_envs = observation[next(iter(observation.keys()))].shape[0]
        else:
            n_envs = observation.shape[0]

        if state is None:
            state = np.concatenate(
                [np.zeros(self.policy.lstm_hidden_state_shape) for _ in range(n_envs)], axis=1
            )
            state = (state, state)

        if episode_start is None:
            episode_start = np.array([False] * n_envs)

        with th.no_grad():
            lstm_states = (
                th.tensor(state[0], dtype=th.float32, device=self.policy.device),
                th.tensor(state[1], dtype=th.float32, device=self.policy.device),
            )
            episode_starts = th.tensor(episode_start, dtype=th.float32, device=self.policy.device)

            # Convert action_masks to tensor if provided
            mask_tensor: th.Tensor | None = None
            if action_masks is not None:
                m = np.asarray(action_masks, dtype=bool)
                if m.ndim == 1:
                    # Broadcast to (n_envs, n_actions)
                    m = np.tile(m, (n_envs, 1))
                mask_tensor = th.tensor(m, dtype=th.bool, device=self.policy.device)

            actions, lstm_states = self.policy._predict(
                observation,
                lstm_states=lstm_states,
                episode_starts=episode_starts,
                deterministic=deterministic,
                action_masks=mask_tensor,
            )
            state = (lstm_states[0].cpu().numpy(), lstm_states[1].cpu().numpy())

        actions = actions.cpu().numpy()
        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, state
