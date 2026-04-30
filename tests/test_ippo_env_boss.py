"""IPPOEnv + boss enable + respawn (v21a)."""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from murimsim.rl.ippo_env import IPPOEnv
from murimsim.rl.multi_env import BOSS_RESPAWN_DELAY


@pytest.fixture
def cfg() -> dict:
    with open("config/default.yaml") as f:
        return yaml.safe_load(f)


def test_ippo_env_accepts_enable_boss(cfg: dict) -> None:
    env = IPPOEnv(config=cfg, n_agents=4, seed=21, enable_boss=True)
    env.reset_all(seed=21)
    bosses = [m for m in env._monsters.all() if m.kind == "boss"]
    assert len(bosses) == 1, "boss should spawn at reset when enable_boss=True"
    assert bosses[0].alive


def test_ippo_env_boss_disabled_by_default(cfg: dict) -> None:
    env = IPPOEnv(config=cfg, n_agents=4, seed=21)
    env.reset_all(seed=21)
    bosses = [m for m in env._monsters.all() if m.kind == "boss"]
    assert len(bosses) == 0, "boss should NOT spawn when enable_boss is default False"


def test_boss_respawn_after_kill(cfg: dict) -> None:
    """Killing the boss schedules a respawn after BOSS_RESPAWN_DELAY ticks."""
    env = IPPOEnv(config=cfg, n_agents=4, seed=21, enable_boss=True)
    env.reset_all(seed=21)

    boss = env._monsters.all_alive()[0]
    boss.health = 0.0
    boss.alive = False

    # Step forward; first idle step should arm the countdown
    actions = np.zeros(4, dtype=np.int64)  # all WAIT/REST-equivalent
    env.step_all(actions)
    assert env._boss_respawn_countdown == BOSS_RESPAWN_DELAY - 0 or env._boss_respawn_countdown >= 0

    # Step BOSS_RESPAWN_DELAY more ticks; should respawn within that window
    initial_boss_count = len(env._monsters.all())
    for _ in range(BOSS_RESPAWN_DELAY + 5):
        env.step_all(actions)
        live_bosses = [m for m in env._monsters.all() if m.kind == "boss" and m.alive]
        if live_bosses:
            break
    live_bosses = [m for m in env._monsters.all() if m.kind == "boss" and m.alive]
    assert len(live_bosses) == 1, "boss should have respawned within BOSS_RESPAWN_DELAY ticks"
    assert len(env._monsters.all()) == initial_boss_count + 1, "respawn adds a new monster entry"


def test_boss_respawn_countdown_resets_when_alive(cfg: dict) -> None:
    """Countdown should be -1 (no pending respawn) while a boss is alive."""
    env = IPPOEnv(config=cfg, n_agents=4, seed=21, enable_boss=True)
    env.reset_all(seed=21)
    actions = np.zeros(4, dtype=np.int64)
    env.step_all(actions)
    assert env._boss_respawn_countdown == -1, "countdown should be -1 while boss alive"


def test_boss_disabled_no_respawn(cfg: dict) -> None:
    """With enable_boss=False, countdown remains -1 forever."""
    env = IPPOEnv(config=cfg, n_agents=4, seed=21, enable_boss=False)
    env.reset_all(seed=21)
    actions = np.zeros(4, dtype=np.int64)
    for _ in range(10):
        env.step_all(actions)
    assert env._boss_respawn_countdown == -1
    assert not any(m.kind == "boss" for m in env._monsters.all())
