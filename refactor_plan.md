# MurimSim — Refactor & Production Quality Plan

> Status: research-grade prototype → production-quality package
> Scope: `murimsim/` Python package only (viewer addressed separately)

---

## 1. Critical Bugs / Hard-Coded State

### 1.1 Hardcoded Windows path in `replay.py`

```python
# murimsim/replay.py:31
_DEFAULT_REPLAY_DIR = Path("/mnt/c/Users/bhara/Downloads/replays")
```

This will silently fail on any machine that is not WSL on `bhara`'s laptop.

**Fix:** Replace with a relative path or env-var override:

```python
_DEFAULT_REPLAY_DIR = Path(os.environ.get("MURIM_REPLAY_DIR", "logs/replays"))
```

Add `MURIM_REPLAY_DIR` to the project `.env.example`.

### 1.2 Phase comment drift in `multi_env.py`

The docstring at the top of `multi_env.py` says "Action space (Phase 3b): Discrete(7)" but the active code uses Phase 5/6 constants.  Phase numbering in constants (`N_ACTIONS_PHASE2`, `N_ACTIONS_PHASE3`, etc.) is never cleaned up between versions, making it hard to know which phase is actually live.

**Fix:** Remove superseded phase constants; keep only the one that maps to the current action space.

---

## 2. Architecture Improvements

### 2.1 `multi_env.py` is too large (~600 lines)

The file mixes three distinct concerns:
- Observation construction (`_build_obs`)
- Reward computation (`_compute_reward`)
- Environment lifecycle (reset/step/heuristic)

**Proposed split:**

```
murimsim/rl/
  multi_env.py        ← step/reset/lifecycle only (~200 lines)
  obs_builder.py      ← _build_obs, channel slicing, normalization
  reward.py           ← all reward functions + their constants
  heuristic.py        ← non-focal agent policy
```

Each file becomes independently testable. Reward constants (`REWARD_ALIVE`, `PENALTY_STARVATION_APPROACH`, etc.) belong in `reward.py` next to the functions that use them.

### 2.2 `agent.py` method count creep (562 lines)

`Agent` currently holds both **state** (dataclass fields) and a large number of **mechanics** methods (`eat`, `rest`, `train`, `apply_hazard`, `spawn_from_parents`, etc.). This makes it hard to test mechanics in isolation.

**Option A (minimal):** Extract factory functions (`spawn`, `spawn_from_parents`) into `murimsim/agent_factory.py`. The dataclass itself stays lean.

**Option B (full):** Introduce an `AgentMechanics` class that wraps an `Agent` instance and holds all mutation methods. The dataclass becomes a pure data container.

Option A is recommended: low-risk, meaningful improvement.

### 2.3 Observation constants scattered across files

`OBS_TOTAL_SIZE = 263`, `OBS_STATS_SIZE = 13`, etc. are defined in `multi_env.py` but also implicitly depended on by `env.py` (single-agent baseline) and by tests. Any observation shape change requires hunting across files.

**Fix:** Create `murimsim/rl/obs_spec.py` as the single source of truth for all observation layout constants. Both envs import from there.

---

## 3. Code Quality

### 3.1 String-based action dispatch in `step()`

The `step()` method likely uses a long `if action == Action.MOVE_N: ...` chain (or similar). This pattern breaks silently when new actions are added.

**Fix:** Use a dispatch dict:
```python
ACTION_HANDLERS: dict[Action, Callable[[MultiEnv, Agent], StepResult]] = {
    Action.MOVE_N: _handle_move_north,
    ...
}
```
Missing actions raise `KeyError` at registration time, not silently at runtime.

### 3.2 `replay.py` never flushes on exception

If the simulation crashes mid-episode, the JSONL file may be missing the last N ticks (OS buffer not flushed). The context manager calls `close()` on `__exit__` but only if used as `with ReplayLogger(...) as rl:` — there is no guarantee callers do this.

**Fix:** Add periodic flush (e.g., every 100 ticks) in `log_tick`, or call `self._file.flush()` unconditionally after each write (small perf cost, worth it for correctness).

### 3.3 `world.py` `_compute_qi_field` is O(N²) and runs at init

The qi influence field is computed by iterating all qi tiles × all grid cells. For a 30×30 grid this is fine, but the approach doesn't scale and isn't cached.

**Fix:** Use `scipy.ndimage.maximum_filter` or a simple BFS flood from qi tiles. Or document explicitly that this is an init-only cost and add an `assert` that it's never called more than once.

### 3.4 Mutable defaults in `Agent.__init__` / factory

Python's mutable default arguments are a latent bug source. Check that no `dict` or `list` default appears in any `__init__` signature — frozen dataclasses with `field(default_factory=...)` are the right pattern here.

---

## 4. Typing & Contracts

### 4.1 `Any` overuse in `replay.py`

```python
agents: list[dict[str, Any]]
events: list[dict[str, Any]]
```

These should be `TypedDict`s so callers get IDE completion and typos are caught:

```python
class AgentSnapshot(TypedDict):
    id: str
    pos: tuple[int, int]
    health: float
    hunger: float
    action: str
    alive: bool
    resistances: dict[str, float]

class CombatEvent(TypedDict):
    type: Literal["combat"]
    attacker: str
    defender: str
    damage: float
```

### 4.2 No `py.typed` marker

`murimsim` has no `py.typed` file, so mypy/pyright treats it as untyped when used as a library. Add an empty `murimsim/py.typed` file and add `"packages": ["murimsim"]` to `pyproject.toml`.

### 4.3 No strict mypy in CI

Add to `pyproject.toml`:
```toml
[tool.mypy]
strict = true
ignore_missing_imports = false
```
And run `mypy murimsim/` in the test suite. The existing type hints are good — make them enforceable.

---

## 5. Configuration

### 5.1 `default.yaml` resource registry is hand-maintained

When a new resource is added to YAML, there is no validation that required fields (`regen_ticks`, `spawn_density`, etc.) are present. A schema validation step at startup (using `pydantic` or `cerberus`) would catch config typos immediately rather than surfacing as AttributeError mid-training.

```python
# At World.__init__ time:
for rid, spec in config["resources"].items():
    ResourceConfig(**spec)  # raises ValidationError early
```

If `ResourceConfig` is a `pydantic.BaseModel` rather than a frozen dataclass, validation is free.

### 5.2 Training hyperparameters mixed into env constants

`multi_env.py` defines reward weights (effectively hyperparameters) as module-level constants rather than in `training.yaml`. This means you cannot sweep reward weights with the existing config system — you must edit code.

**Fix:** Move all `REWARD_*` and `PENALTY_*` constants into `config/training.yaml` under a `reward:` key, and pass them into the env constructor via `EnvConfig`.

---

## 6. Performance

### 6.1 Per-tick resource scan in viewer sync path

`multi_env.py` builds the full resource dict for `ReplayLogger.log_tick` every step, even during training when the logger is disabled. The enabled check is in `log_tick` but the dict construction is not guarded.

**Fix:** Guard the construction:
```python
if self._replay_logger.enabled:
    resources = self._build_resource_snapshot()
    self._replay_logger.log_tick(...)
```

### 6.2 `copy.deepcopy` in `step()`

If `step()` deep-copies agent state for rollback safety, this is the single largest per-step allocation. Profile with `cProfile` before the next version bump — it may be unnecessary if the environment is always reset between episodes.

### 6.3 Stash lookup is O(n_stashes) per agent per step

`StashRegistry` uses list iteration for `get_enemy_stashes_at`. For 10 agents × 10 stashes this is trivial, but it's worth switching to a `dict[tuple[int,int], list[Stash]]` spatial index now while the code is small.

---

## 7. Testing Gaps

| Area | Current | Recommended |
|---|---|---|
| Reward function unit tests | None | One test per reward signal |
| Observation shape contract | Implicit | Assert `obs.shape == (OBS_TOTAL_SIZE,)` in env test |
| Inheritance math | Partial | Property-based test with Hypothesis |
| Replay file round-trip | None | Write ticks, reload, assert field equality |
| Config validation | None | Test that bad YAML raises at init, not mid-run |

---

## 8. Dev Tooling

- Add `ruff` for linting (replaces flake8/isort/pyupgrade in one tool)
- Add `pre-commit` config: `ruff`, `mypy`, `pytest -x`
- Add `Makefile` targets: `make train`, `make test`, `make replay`, `make lint`
- Pin `stable-baselines3` and `sb3-contrib` to exact versions in `requirements.txt` — these libraries break APIs between minor versions

---

## Priority Order

| Priority | Item | Effort | Risk |
|---|---|---|---|
| P0 | Fix hardcoded replay path (1.1) | 5 min | None |
| P0 | Guard resource snapshot under `logger.enabled` (6.1) | 10 min | None |
| P1 | `TypedDict` for replay snapshots (4.1) | 1 hr | Low |
| P1 | Move `REWARD_*` to config (5.2) | 2 hr | Medium |
| P1 | Split `multi_env.py` → obs/reward/heuristic (2.1) | 3 hr | Medium |
| P2 | Extract agent factory (2.2 Option A) | 1 hr | Low |
| P2 | Add mypy strict + `py.typed` (4.2, 4.3) | 2 hr | Low |
| P2 | Reward unit tests (7) | 2 hr | None |
| P3 | Pydantic config validation (5.1) | 3 hr | Low |
| P3 | Action dispatch dict (3.1) | 1 hr | Low |
| P3 | Stash spatial index (6.3) | 1 hr | Low |
