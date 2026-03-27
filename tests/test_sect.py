"""Tests for the sect scaffold (murimsim/sect.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from murimsim.sect import SectConfig, SectRegistry

CONFIG_PATH = Path("config/default.yaml")


# ---------------------------------------------------------------------------
# SectConfig tests
# ---------------------------------------------------------------------------


def test_sect_config_from_dict_roundtrip():
    """SectConfig.from_dict constructs correctly from a plain dict."""
    data = {
        "name": "iron_fang",
        "color": "#ef4444",
        "n_agents": 5,
        "trait_prefs": {"strength": 0.8, "adventure_spirit": 0.4},
    }
    cfg = SectConfig.from_dict(data)
    assert cfg.name == "iron_fang"
    assert cfg.color == "#ef4444"
    assert cfg.n_agents == 5
    assert cfg.get_trait("strength") == 0.8
    assert cfg.get_trait("adventure_spirit") == 0.4


def test_sect_config_get_trait_default():
    """get_trait returns 0.5 for absent keys."""
    cfg = SectConfig(name="test", color="#fff", n_agents=1, trait_prefs={})
    assert cfg.get_trait("poison_resistance") == 0.5
    assert cfg.get_trait("anything", default=0.1) == 0.1


def test_sect_config_rejects_out_of_range_traits():
    """SectConfig raises ValueError when a trait is outside [0, 1]."""
    with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
        SectConfig(
            name="bad",
            color="#000",
            n_agents=1,
            trait_prefs={"strength": 1.5},
        )


def test_sect_config_rejects_zero_agents():
    """SectConfig raises ValueError for n_agents < 1."""
    with pytest.raises(ValueError, match="n_agents"):
        SectConfig(name="bad", color="#000", n_agents=0)


def test_sample_traits_respects_bounds():
    """sample_traits always returns values in [0, 1]."""
    rng = np.random.default_rng(0)
    cfg = SectConfig(
        name="iron_fang",
        color="#ef4444",
        n_agents=10,
        trait_prefs={"strength": 0.95, "poison_resistance": 0.05},
    )
    for _ in range(200):
        traits = cfg.sample_traits(rng, noise=0.3)
        for k, v in traits.items():
            assert 0.0 <= v <= 1.0, f"Trait {k}={v} out of [0, 1]"


def test_sample_traits_centred_on_prefs():
    """sample_traits mean is close to pref over many samples (law of large numbers)."""
    rng = np.random.default_rng(42)
    pref = 0.7
    cfg = SectConfig(
        name="jade_lotus",
        color="#22c55e",
        n_agents=1,
        trait_prefs={"adventure_spirit": pref},
    )
    values = [cfg.sample_traits(rng)["adventure_spirit"] for _ in range(500)]
    assert abs(np.mean(values) - pref) < 0.05, "Sample mean too far from pref"


# ---------------------------------------------------------------------------
# SectRegistry tests
# ---------------------------------------------------------------------------


def test_sect_registry_default_has_three_sects():
    """Default registry contains exactly the three canonical sects."""
    reg = SectRegistry.default()
    assert len(reg) == 3
    assert set(reg.names()) == {"iron_fang", "jade_lotus", "shadow_root"}


def test_sect_registry_lookup_by_name():
    """Registry supports dict-style name lookup."""
    reg = SectRegistry.default()
    iron = reg["iron_fang"]
    assert iron.name == "iron_fang"
    assert iron.get_trait("strength") > 0.5  # iron_fang is strong


def test_sect_registry_unknown_name_raises():
    """KeyError with helpful message for unknown sect name."""
    reg = SectRegistry.default()
    with pytest.raises(KeyError, match="unknown_sect"):
        _ = reg["unknown_sect"]


def test_sect_registry_iterable():
    """Registry is iterable and yields SectConfig objects."""
    reg = SectRegistry.default()
    sects = list(reg)
    assert len(sects) == 3
    assert all(isinstance(s, SectConfig) for s in sects)


def test_sect_registry_rejects_duplicate_names():
    """SectRegistry raises ValueError when two sects share a name."""
    dup = SectConfig(name="clone", color="#000", n_agents=1)
    with pytest.raises(ValueError, match="Duplicate"):
        SectRegistry([dup, dup])


def test_sect_registry_from_config_fallback():
    """from_config falls back to defaults when 'sects' key is absent."""
    reg = SectRegistry.from_config({})
    assert len(reg) == 3


def test_sect_registry_from_config_custom():
    """from_config reads custom sect definitions from config dict."""
    cfg = {
        "sects": [
            {"name": "alpha", "color": "#f00", "n_agents": 3, "trait_prefs": {"strength": 0.9}},
            {"name": "beta", "color": "#0f0", "n_agents": 3, "trait_prefs": {"strength": 0.1}},
        ]
    }
    reg = SectRegistry.from_config(cfg)
    assert len(reg) == 2
    assert reg["alpha"].get_trait("strength") == 0.9
    assert reg["beta"].get_trait("strength") == 0.1


def test_sect_registry_sects_are_isolated():
    """Trait preferences differ across the default sects (no copy-paste bug)."""
    reg = SectRegistry.default()
    iron = reg["iron_fang"]
    jade = reg["jade_lotus"]
    shadow = reg["shadow_root"]
    # Iron fang is strongest
    assert iron.get_trait("strength") > jade.get_trait("strength")
    # Jade lotus is most poison resistant
    assert jade.get_trait("poison_resistance") > iron.get_trait("poison_resistance")
    # Shadow root is most adventurous
    assert shadow.get_trait("adventure_spirit") >= jade.get_trait("adventure_spirit")


# ---------------------------------------------------------------------------
# Environment factory tests
# ---------------------------------------------------------------------------


@pytest.fixture
def base_config():
    return yaml.safe_load(CONFIG_PATH.read_text())


def test_make_env_returns_combat_env(base_config):
    """make_env returns a CombatEnv tagged with sect metadata."""
    from murimsim.rl.multi_env import CombatEnv

    reg = SectRegistry.default()
    env = reg.make_env("iron_fang", base_config, seed=0)
    assert isinstance(env, CombatEnv)
    assert env.sect_name == "iron_fang"
    assert env.sect_color == "#ef4444"


def test_make_env_respects_n_agents_override(base_config):
    """make_env respects the n_agents override."""
    reg = SectRegistry.default()
    env = reg.make_env("jade_lotus", base_config, seed=1, n_agents=4)
    env.reset(seed=1)
    assert len(env._agents) == 4


def test_make_all_envs_returns_three_envs(base_config):
    """make_all_envs returns one env per sect, each with a distinct seed."""
    reg = SectRegistry.default()
    envs = reg.make_all_envs(base_config, seed=100)
    assert set(envs.keys()) == {"iron_fang", "jade_lotus", "shadow_root"}
    # Each env should be its own object
    assert envs["iron_fang"] is not envs["jade_lotus"]


def test_make_all_envs_isolated_worlds(base_config):
    """Envs from different sects diverge (different seeds produce different worlds)."""
    reg = SectRegistry.default()
    envs = reg.make_all_envs(base_config, seed=7)
    obs_iron, _ = envs["iron_fang"].reset(seed=7)
    obs_jade, _ = envs["jade_lotus"].reset(seed=8)
    # Different seeds → different observations
    assert not np.array_equal(obs_iron, obs_jade)
