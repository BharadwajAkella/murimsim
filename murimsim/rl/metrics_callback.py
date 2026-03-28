"""metrics_callback.py — SB3 callback that writes dashboard_data.js after every checkpoint.

Collects episode-terminal info dicts from the vectorised environment and
aggregates them into rolling statistics. Writes ``logs/dashboard_data.js``
(overwriting each time) so the HTML dashboard always reflects the latest run.
The JS file sets ``window.DASHBOARD_DATA`` so it loads cleanly from file:// URLs
without any CORS / fetch restrictions.

Usage — add to any training script::

    from murimsim.rl.metrics_callback import MetricsDashboardCallback
    cb = MetricsDashboardCallback(run_name="lstm_v3", total_timesteps=1_500_000)
    model.learn(..., callback=[checkpoint_cb, cb])
"""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

# Rolling window: keep stats for the last N completed episodes
_ROLLING_WINDOW: int = 200

DASHBOARD_PATH = Path("logs/dashboard_data.js")


class MetricsDashboardCallback(BaseCallback):
    """Aggregates per-episode metrics and writes ``logs/dashboard.json``.

    Args:
        run_name:          Label shown in the dashboard (e.g. "lstm_v3").
        total_timesteps:   Total training steps (for progress bar).
        write_freq:        How often (in steps) to flush to disk.
                           Also writes on every checkpoint and at training end.
        dashboard_path:    Output JSON path. Defaults to ``logs/dashboard.json``.
    """

    def __init__(
        self,
        run_name: str,
        total_timesteps: int,
        write_freq: int = 20_000,
        dashboard_path: Path = DASHBOARD_PATH,
    ) -> None:
        super().__init__(verbose=0)
        self._run_name = run_name
        self._total_timesteps = total_timesteps
        self._write_freq = write_freq
        self._dashboard_path = Path(dashboard_path)
        # Also write a .json sibling for tools that read raw JSON
        self._json_path = self._dashboard_path.with_suffix(".json")
        self._next_write = write_freq
        self._start_time = time.time()

        # Rolling buffers — each entry is one completed episode
        self._lifespans: deque[float] = deque(maxlen=_ROLLING_WINDOW)
        self._f_metrics: deque[float] = deque(maxlen=_ROLLING_WINDOW)
        self._strengths: deque[float] = deque(maxlen=_ROLLING_WINDOW)
        # action_rates: deque of dicts
        self._action_rates: deque[dict[str, float]] = deque(maxlen=_ROLLING_WINDOW)
        # hazard behaviour
        self._hazard_approaches: deque[dict[str, int]] = deque(maxlen=_ROLLING_WINDOW)
        self._hazard_flees: deque[dict[str, int]] = deque(maxlen=_ROLLING_WINDOW)
        self._resistance_gained: deque[dict[str, float]] = deque(maxlen=_ROLLING_WINDOW)
        # per-agent credit assignment: list of per-agent mean rewards per episode
        self._agent_mean_rewards: deque[list[float]] = deque(maxlen=_ROLLING_WINDOW)
        # timestep history for sparkline
        self._history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # SB3 hooks
    # ------------------------------------------------------------------

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for done, info in zip(dones, infos):
            if not done:
                continue
            self._consume_episode_info(info)

        if self.num_timesteps >= self._next_write:
            self._flush()
            self._next_write += self._write_freq
        return True

    def _on_training_end(self) -> None:
        self._flush()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _consume_episode_info(self, info: dict[str, Any]) -> None:
        """Extract metrics from a terminal episode info dict."""
        if "ep_lifespan" in info:
            self._lifespans.append(float(info["ep_lifespan"]))
        if "ep_avg_strength" in info:
            self._strengths.append(float(info["ep_avg_strength"]))
        if "f_metric" in info:
            self._f_metrics.append(float(info["f_metric"]))
        if "ep_action_rates" in info:
            self._action_rates.append(dict(info["ep_action_rates"]))
        if "ep_hazard_approaches" in info:
            self._hazard_approaches.append(dict(info["ep_hazard_approaches"]))
        if "ep_hazard_flees" in info:
            self._hazard_flees.append(dict(info["ep_hazard_flees"]))
        if "ep_resistance_gained" in info:
            self._resistance_gained.append(dict(info["ep_resistance_gained"]))
        if "ep_agent_mean_reward" in info:
            self._agent_mean_rewards.append(list(info["ep_agent_mean_reward"]))

    def _rolling_mean(self, buf: deque) -> float | None:
        if not buf:
            return None
        return float(np.mean(list(buf)))

    def _mean_dict(self, buf: deque[dict]) -> dict[str, float]:
        """Average each key across all dicts in the deque."""
        if not buf:
            return {}
        keys: set[str] = set()
        for d in buf:
            keys.update(d.keys())
        result = {}
        for k in keys:
            vals = [d[k] for d in buf if k in d]
            result[k] = float(np.mean(vals)) if vals else 0.0
        return result

    def _flush(self) -> None:
        """Build the dashboard JSON payload and write to disk."""
        elapsed = time.time() - self._start_time
        n_episodes = len(self._lifespans)

        avg_action_rates = self._mean_dict(self._action_rates)
        avg_hazard_approaches = self._mean_dict(self._hazard_approaches)
        avg_hazard_flees = self._mean_dict(self._hazard_flees)
        avg_resistance_gained = self._mean_dict(self._resistance_gained)

        # Per-agent credit: average mean reward per slot across all episodes
        avg_agent_credit: list[float] = []
        if self._agent_mean_rewards:
            arr = np.array(list(self._agent_mean_rewards))  # shape (episodes, n_agents)
            avg_agent_credit = [round(float(v), 4) for v in arr.mean(axis=0)]

        # Build approach-ratio: approaches / (approaches + flees) per hazard
        approach_ratio: dict[str, float] = {}
        for h in set(avg_hazard_approaches) | set(avg_hazard_flees):
            a = avg_hazard_approaches.get(h, 0.0)
            f = avg_hazard_flees.get(h, 0.0)
            total = a + f
            approach_ratio[h] = round(a / total, 3) if total > 0 else 0.0

        snapshot = {
            "timestep": self.num_timesteps,
            "f_metric": self._rolling_mean(self._f_metrics),
            "avg_lifespan": self._rolling_mean(self._lifespans),
        }
        self._history.append(snapshot)

        payload: dict[str, Any] = {
            "run_name": self._run_name,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": round(elapsed),
            "timestep": self.num_timesteps,
            "total_timesteps": self._total_timesteps,
            "progress_pct": round(100.0 * self.num_timesteps / max(1, self._total_timesteps), 1),
            "n_episodes_sampled": n_episodes,
            "rolling_window": _ROLLING_WINDOW,
            "avg_lifespan": self._rolling_mean(self._lifespans),
            "avg_f_metric": self._rolling_mean(self._f_metrics),
            "avg_strength": self._rolling_mean(self._strengths),
            "avg_action_rates": {k: round(v, 4) for k, v in avg_action_rates.items()},
            "avg_hazard_approaches": {k: round(v, 2) for k, v in avg_hazard_approaches.items()},
            "avg_hazard_flees": {k: round(v, 2) for k, v in avg_hazard_flees.items()},
            "hazard_approach_ratio": approach_ratio,
            "avg_resistance_gained": {k: round(v, 4) for k, v in avg_resistance_gained.items()},
            "avg_agent_credit": avg_agent_credit,   # per-agent mean reward (credit baseline)
            "history": self._history[-500:],  # cap sparkline history
        }

        self._dashboard_path.parent.mkdir(parents=True, exist_ok=True)

        json_str = json.dumps(payload, indent=2)

        # Write raw JSON (for scripts/tooling)
        tmp_json = self._json_path.with_suffix(".tmp")
        tmp_json.write_text(json_str)
        tmp_json.replace(self._json_path)

        # Write JS data file — works from file:// without CORS issues
        js_content = f"// Auto-generated by MetricsDashboardCallback — do not edit\nwindow.DASHBOARD_DATA = {json_str};\n"
        tmp_js = self._dashboard_path.with_suffix(".tmp")
        tmp_js.write_text(js_content)
        tmp_js.replace(self._dashboard_path)
