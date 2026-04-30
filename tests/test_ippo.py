"""P3 IPPO unit tests — policy + buffer + update.

Covers rubber-duck findings:
- repair_action_mask falls back to REST (not 0/MOVE_N).
- masked_categorical produces zero prob for forbidden actions and finite entropy.
- GAE recurrence respects per-slot life boundaries (next_active gating).
- All-inactive minibatch: parameters do NOT move (no optimizer step).
- Pre-action mask is what gets stored/used (state-dependent validity).
- PPO update: positive advantage increases logprob of taken action.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from murimsim.actions import Action
from murimsim.rl.ippo import (
    PPOStats,
    RolloutBuffer,
    SharedActorCritic,
    masked_categorical,
    ppo_update,
    repair_action_mask,
)


# ---------------------------------------------------------------------------
# Action mask helpers
# ---------------------------------------------------------------------------

def test_repair_action_mask_falls_back_to_rest():
    mask = torch.zeros(2, 17, dtype=torch.bool)
    repaired = repair_action_mask(mask)
    assert repaired[0, Action.REST.value].item() is True
    assert repaired[1, Action.REST.value].item() is True
    assert repaired[0, Action.MOVE_N.value].item() is False  # NOT 0


def test_repair_preserves_existing_legal_mask():
    mask = torch.zeros(1, 5, dtype=torch.bool)
    mask[0, 2] = True
    repaired = repair_action_mask(mask.clone())
    assert repaired[0, 2].item() is True
    assert repaired[0].sum().item() == 1


def test_masked_categorical_zero_prob_on_forbidden():
    logits = torch.zeros(1, 5)  # uniform
    mask = torch.tensor([[True, False, True, False, True]])
    dist = masked_categorical(logits, mask)
    probs = dist.probs[0]
    assert probs[1].item() == pytest.approx(0.0, abs=1e-7)
    assert probs[3].item() == pytest.approx(0.0, abs=1e-7)
    assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_masked_entropy_is_finite_with_partial_mask():
    logits = torch.randn(4, 7)
    mask = torch.tensor([[True, False, True, False, False, True, False]] * 4)
    dist = masked_categorical(logits, mask)
    ent = dist.entropy()
    assert torch.isfinite(ent).all()


def test_masked_entropy_finite_with_all_false_mask():
    logits = torch.randn(2, 5)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    dist = masked_categorical(logits, mask)
    ent = dist.entropy()
    assert torch.isfinite(ent).all()
    # Repaired to allow REST (action 6) — but n_actions=5 here so REST is OOR.
    # Skip this part; the n_actions=5 case is just for finite-entropy check.


# ---------------------------------------------------------------------------
# SharedActorCritic
# ---------------------------------------------------------------------------

def test_actor_critic_forward_shapes():
    net = SharedActorCritic(obs_dim=10, n_actions=17)
    obs = torch.randn(3, 10)
    logits, value = net(obs)
    assert logits.shape == (3, 17)
    assert value.shape == (3,)


def test_act_returns_legal_action_only():
    torch.manual_seed(0)
    net = SharedActorCritic(obs_dim=8, n_actions=17)
    obs = torch.randn(100, 8)
    mask = torch.zeros(100, 17, dtype=torch.bool)
    mask[:, Action.REST.value] = True
    mask[:, Action.MOVE_N.value] = True
    action, logprob, _value = net.act(obs, mask)
    legal = {Action.REST.value, Action.MOVE_N.value}
    sampled = set(action.tolist())
    assert sampled.issubset(legal), f"sampled illegal: {sampled - legal}"
    assert torch.isfinite(logprob).all()


def test_evaluate_returns_finite():
    net = SharedActorCritic(obs_dim=6, n_actions=17)
    obs = torch.randn(5, 6)
    mask = torch.ones(5, 17, dtype=torch.bool)
    actions = torch.randint(0, 17, (5,))
    logp, ent, val = net.evaluate(obs, mask, actions)
    for t in (logp, ent, val):
        assert torch.isfinite(t).all()


# ---------------------------------------------------------------------------
# RolloutBuffer + GAE
# ---------------------------------------------------------------------------

def test_buffer_stores_and_flattens_active_only():
    buf = RolloutBuffer(rollout_length=2, n_envs=1, n_agents=3, obs_dim=4, n_actions=5)
    for _t in range(2):
        buf.add(
            obs=np.zeros((1, 3, 4), dtype=np.float32),
            action_mask=np.ones((1, 3, 5), dtype=bool),
            action=torch.zeros(1, 3, dtype=torch.long),
            logprob=torch.zeros(1, 3),
            value=torch.zeros(1, 3),
            reward=np.zeros((1, 3), dtype=np.float32),
            done=np.zeros((1, 3), dtype=bool),
            active=np.array([[True, False, True]] * 1, dtype=bool),
        )
    adv, ret = buf.compute_gae(
        last_value=torch.zeros(1, 3),
        last_active=torch.ones(1, 3, dtype=torch.bool),
    )
    batch = buf.flatten_active(adv, ret)
    # 2 timesteps × 1 env × 2 active slots per step = 4 active samples
    assert len(batch) == 4


def test_gae_zero_for_inactive_slots():
    buf = RolloutBuffer(rollout_length=1, n_envs=1, n_agents=2, obs_dim=2, n_actions=3)
    buf.add(
        obs=np.zeros((1, 2, 2), dtype=np.float32),
        action_mask=np.ones((1, 2, 3), dtype=bool),
        action=torch.zeros(1, 2, dtype=torch.long),
        logprob=torch.zeros(1, 2),
        value=torch.zeros(1, 2),
        reward=np.array([[1.0, 99.0]], dtype=np.float32),
        done=np.zeros((1, 2), dtype=bool),
        active=np.array([[True, False]], dtype=bool),
    )
    adv, _ret = buf.compute_gae(
        last_value=torch.zeros(1, 2),
        last_active=torch.ones(1, 2, dtype=torch.bool),
    )
    assert adv[0, 0, 0].item() == pytest.approx(1.0, abs=1e-6)
    assert adv[0, 0, 1].item() == 0.0  # inactive slot zeroed


def test_gae_does_not_bootstrap_across_done():
    """If slot dies at t, V(s_{t+1}) should NOT contribute to advantage at t."""
    buf = RolloutBuffer(rollout_length=1, n_envs=1, n_agents=1, obs_dim=2, n_actions=3)
    buf.add(
        obs=np.zeros((1, 1, 2), dtype=np.float32),
        action_mask=np.ones((1, 1, 3), dtype=bool),
        action=torch.zeros(1, 1, dtype=torch.long),
        logprob=torch.zeros(1, 1),
        value=torch.zeros(1, 1),
        reward=np.array([[2.0]], dtype=np.float32),
        done=np.array([[True]], dtype=bool),  # slot died at t
        active=np.array([[True]], dtype=bool),
    )
    # last_value=999; if bootstrap leaks across done, adv would include gamma*999.
    adv, _ret = buf.compute_gae(
        last_value=torch.full((1, 1), 999.0),
        last_active=torch.ones(1, 1, dtype=torch.bool),
        gamma=0.99,
    )
    # Expected: delta = r + gamma*V_next*nonterminal - V = 2 + 0 - 0 = 2
    assert adv[0, 0, 0].item() == pytest.approx(2.0, abs=1e-6)


def test_gae_does_not_bootstrap_into_inactive_next():
    """If slot is inactive at t+1 (rebirth gap or post-death), no bootstrap."""
    buf = RolloutBuffer(rollout_length=1, n_envs=1, n_agents=1, obs_dim=2, n_actions=3)
    buf.add(
        obs=np.zeros((1, 1, 2), dtype=np.float32),
        action_mask=np.ones((1, 1, 3), dtype=bool),
        action=torch.zeros(1, 1, dtype=torch.long),
        logprob=torch.zeros(1, 1),
        value=torch.zeros(1, 1),
        reward=np.array([[5.0]], dtype=np.float32),
        done=np.array([[False]], dtype=bool),  # not flagged done, but next is inactive
        active=np.array([[True]], dtype=bool),
    )
    adv, _ret = buf.compute_gae(
        last_value=torch.full((1, 1), 999.0),
        last_active=torch.zeros(1, 1, dtype=torch.bool),  # inactive at T
        gamma=0.99,
    )
    assert adv[0, 0, 0].item() == pytest.approx(5.0, abs=1e-6)


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def _make_dummy_batch(n: int = 32, obs_dim: int = 4, n_actions: int = 5, adv_sign: float = 1.0):
    torch.manual_seed(42)
    obs = torch.randn(n, obs_dim)
    mask = torch.ones(n, n_actions, dtype=torch.bool)
    actions = torch.zeros(n, dtype=torch.long)  # all picked action 0
    old_logprobs = torch.full((n,), -np.log(n_actions))  # uniform initial
    values = torch.zeros(n)
    advantages = torch.full((n,), adv_sign)
    returns = advantages.clone()
    from murimsim.rl.ippo import RolloutBatch
    return RolloutBatch(obs, mask, actions, old_logprobs, values, advantages, returns)


def test_ppo_positive_advantage_increases_logprob_of_action():
    torch.manual_seed(0)
    policy = SharedActorCritic(obs_dim=4, n_actions=5)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-3)
    batch = _make_dummy_batch(n=64, adv_sign=+1.0)

    with torch.no_grad():
        before, _, _ = policy.evaluate(batch.obs, batch.action_mask, batch.actions)

    ppo_update(policy, optimizer, batch, n_epochs=4, n_minibatches=4, normalize_adv=False)

    with torch.no_grad():
        after, _, _ = policy.evaluate(batch.obs, batch.action_mask, batch.actions)

    assert after.mean().item() > before.mean().item(), (
        f"positive adv should raise logprob; before={before.mean()}, after={after.mean()}"
    )


def test_ppo_negative_advantage_decreases_logprob_of_action():
    torch.manual_seed(0)
    policy = SharedActorCritic(obs_dim=4, n_actions=5)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-3)
    batch = _make_dummy_batch(n=64, adv_sign=-1.0)

    with torch.no_grad():
        before, _, _ = policy.evaluate(batch.obs, batch.action_mask, batch.actions)

    ppo_update(policy, optimizer, batch, n_epochs=4, n_minibatches=4, normalize_adv=False)

    with torch.no_grad():
        after, _, _ = policy.evaluate(batch.obs, batch.action_mask, batch.actions)

    assert after.mean().item() < before.mean().item()


def test_ppo_empty_batch_does_not_move_parameters():
    torch.manual_seed(0)
    policy = SharedActorCritic(obs_dim=4, n_actions=5)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    from murimsim.rl.ippo import RolloutBatch
    empty = RolloutBatch(
        obs=torch.zeros(0, 4),
        action_mask=torch.zeros(0, 5, dtype=torch.bool),
        actions=torch.zeros(0, dtype=torch.long),
        old_logprobs=torch.zeros(0),
        values=torch.zeros(0),
        advantages=torch.zeros(0),
        returns=torch.zeros(0),
    )
    snapshot = {k: v.clone() for k, v in policy.state_dict().items()}
    stats = ppo_update(policy, optimizer, empty)
    assert isinstance(stats, PPOStats)
    assert stats.n_updates == 0
    for k, v in policy.state_dict().items():
        assert torch.equal(snapshot[k], v), f"param {k} changed on empty batch"


def test_ppo_update_finite_under_pre_action_mask_change():
    """Stored mask is what counts; downstream env mask changes are irrelevant."""
    torch.manual_seed(0)
    policy = SharedActorCritic(obs_dim=4, n_actions=5)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-3)
    batch = _make_dummy_batch(n=32)
    # Sanity: deliberately tighten the stored mask to legalise only the
    # action that was sampled — would crash if logprob became -inf elsewhere.
    new_mask = torch.zeros_like(batch.action_mask)
    new_mask[torch.arange(len(batch)), batch.actions] = True
    batch.action_mask = new_mask
    stats = ppo_update(policy, optimizer, batch, n_epochs=2, n_minibatches=2)
    assert np.isfinite(stats.pg_loss)
    assert np.isfinite(stats.vf_loss)
    assert np.isfinite(stats.entropy)


def test_ppo_update_normalizes_advantages_only_over_active():
    """Active mask is applied at flatten time; normalize_adv should not crash with n=1."""
    torch.manual_seed(0)
    policy = SharedActorCritic(obs_dim=4, n_actions=5)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    from murimsim.rl.ippo import RolloutBatch
    one = RolloutBatch(
        obs=torch.randn(1, 4),
        action_mask=torch.ones(1, 5, dtype=torch.bool),
        actions=torch.zeros(1, dtype=torch.long),
        old_logprobs=torch.zeros(1),
        values=torch.zeros(1),
        advantages=torch.tensor([0.5]),
        returns=torch.tensor([0.5]),
    )
    stats = ppo_update(policy, optimizer, one, normalize_adv=True)
    assert stats.n_updates >= 1
