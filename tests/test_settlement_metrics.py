"""Tests for settlement-metrics instrumentation in MetricsDashboardCallback.

Verifies that per-episode settlement metrics emitted in episode info dicts
are consumed and surfaced in the dashboard flush payload.  No SB3 environment
is required — we call the internal helpers directly.
"""
from __future__ import annotations

from collections import deque
from unittest.mock import MagicMock, patch

import pytest

from murimsim.rl.metrics_callback import MetricsDashboardCallback


def _make_callback() -> MetricsDashboardCallback:
    cb = MetricsDashboardCallback(run_name="test", total_timesteps=1000)
    # Attach a mock model so num_timesteps is accessible (SB3 pattern)
    cb.model = MagicMock()
    cb.model.num_timesteps = 0
    cb.num_timesteps = 0
    return cb


def _episode_info(
    stash_fill_rate: float = 0.5,
    stash_withdraw_rate: float = 0.4,
    avg_dist_from_stash: float = 3.0,
    revisit_entropy: float = 2.1,
    group_persistence: float = 8.0,
) -> dict:
    """Build a minimal terminal episode info dict with settlement metrics."""
    return {
        "ep_lifespan": 100,
        "ep_stash_fill_rate": stash_fill_rate,
        "ep_stash_withdraw_rate": stash_withdraw_rate,
        "ep_avg_dist_from_stash": avg_dist_from_stash,
        "ep_revisit_entropy": revisit_entropy,
        "ep_group_persistence": group_persistence,
    }


# ---------------------------------------------------------------------------
# 1. Buffers are populated by _consume_episode_info
# ---------------------------------------------------------------------------

def test_consume_episode_info_populates_buffers() -> None:
    """All five settlement metric deques should fill on a terminal info dict."""
    cb = _make_callback()
    info = _episode_info()

    cb._consume_episode_info(info)

    assert len(cb._stash_fill_rates) == 1
    assert len(cb._stash_withdraw_rates) == 1
    assert len(cb._avg_dist_from_stash) == 1
    assert len(cb._revisit_entropies) == 1
    assert len(cb._group_persistences) == 1

    assert cb._stash_fill_rates[0] == pytest.approx(0.5)
    assert cb._stash_withdraw_rates[0] == pytest.approx(0.4)
    assert cb._avg_dist_from_stash[0] == pytest.approx(3.0)
    assert cb._revisit_entropies[0] == pytest.approx(2.1)
    assert cb._group_persistences[0] == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# 2. Buffers are not populated when keys are absent (non-terminal steps)
# ---------------------------------------------------------------------------

def test_consume_episode_info_skips_missing_keys() -> None:
    """A step info dict without settlement keys must not touch the buffers."""
    cb = _make_callback()
    cb._consume_episode_info({"ep_lifespan": 50})

    assert len(cb._stash_fill_rates) == 0
    assert len(cb._stash_withdraw_rates) == 0
    assert len(cb._avg_dist_from_stash) == 0
    assert len(cb._revisit_entropies) == 0
    assert len(cb._group_persistences) == 0


# ---------------------------------------------------------------------------
# 3. Rolling mean is correct after multiple episodes
# ---------------------------------------------------------------------------

def test_rolling_mean_across_multiple_episodes() -> None:
    """Rolling means should average values across all consumed episodes."""
    cb = _make_callback()
    for rate in [0.2, 0.4, 0.6]:
        cb._consume_episode_info(_episode_info(stash_fill_rate=rate))

    mean = cb._rolling_mean(cb._stash_fill_rates)
    assert mean == pytest.approx(0.4, rel=1e-5)


# ---------------------------------------------------------------------------
# 4. Flush payload contains all settlement metric keys
# ---------------------------------------------------------------------------

def test_flush_payload_contains_settlement_keys(tmp_path) -> None:
    """_flush() should include all five settlement metric averages."""
    cb = MetricsDashboardCallback(
        run_name="test",
        total_timesteps=1000,
        dashboard_path=tmp_path / "dashboard_data.js",
    )
    cb.model = MagicMock()
    cb.model.num_timesteps = 100
    cb.num_timesteps = 100

    cb._consume_episode_info(_episode_info())
    cb._flush()

    import json
    json_path = (tmp_path / "dashboard_data.js").with_suffix(".json")
    payload = json.loads(json_path.read_text())

    assert "avg_stash_fill_rate" in payload
    assert "avg_stash_withdraw_rate" in payload
    assert "avg_dist_from_stash" in payload
    assert "avg_revisit_entropy" in payload
    assert "avg_group_persistence" in payload

    assert payload["avg_stash_fill_rate"] == pytest.approx(0.5)
    assert payload["avg_stash_withdraw_rate"] == pytest.approx(0.4)
    assert payload["avg_dist_from_stash"] == pytest.approx(3.0)
    assert payload["avg_revisit_entropy"] == pytest.approx(2.1)
    assert payload["avg_group_persistence"] == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# 5. None is returned when no episodes have been consumed yet
# ---------------------------------------------------------------------------

def test_flush_settlement_keys_are_none_when_no_episodes(tmp_path) -> None:
    """Settlement metric keys are None in the payload when no episodes ran."""
    cb = MetricsDashboardCallback(
        run_name="test",
        total_timesteps=1000,
        dashboard_path=tmp_path / "dashboard_data.js",
    )
    cb.model = MagicMock()
    cb.model.num_timesteps = 0
    cb.num_timesteps = 0

    cb._flush()

    import json
    json_path = (tmp_path / "dashboard_data.js").with_suffix(".json")
    payload = json.loads(json_path.read_text())

    assert payload["avg_stash_fill_rate"] is None
    assert payload["avg_stash_withdraw_rate"] is None
    assert payload["avg_dist_from_stash"] is None
    assert payload["avg_revisit_entropy"] is None
    assert payload["avg_group_persistence"] is None
