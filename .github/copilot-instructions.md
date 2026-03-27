# GitHub Copilot — Workspace Instructions

## Code Quality Standards

- Write **clean, production-quality, easily extensible, and readable** code.
- Prefer clarity over cleverness. Code is read more than it is written.
- Use type hints everywhere (Python 3.11+ style: `X | Y` instead of `Optional[X]`).
- Write docstrings for all public classes and methods.
- Keep functions small and focused (single responsibility).
- Use `@dataclass` for structured data.
- Avoid magic numbers — define named constants or read from config.
- **If you write code that invalidates or replaces old code, remove the old code.** No dead code, no commented-out blocks.

## Architecture Invariants (DO NOT VIOLATE)

- **All traits/params are floats in [0.0, 1.0].** No 0–100 scale anywhere in code — display-only scaling only (in the viewer).
- **Determinism is sacred.** Every simulation run with the same seed must produce byte-identical output.
- **Phase gates:** Tests from all earlier phases must keep passing when implementing a new phase.
- **Data-driven resources:** Adding a resource type = YAML change only, zero code changes.
- **RL is frozen in Phase 4+.** Never update RL weights outside of dedicated training phases.

## Patterns to Follow

- Config via `config/default.yaml` (PyYAML). No hardcoded paths or magic values.
- Seeded RNG: always `np.random.default_rng(seed)`. Never `np.random.seed()` (modifies global state).
- Tests live in `tests/`. Run with `pytest`. Every feature has a corresponding test.
- Internal state prefixed with `_` (e.g., `_grid`, `_countdown`). Public API is clean and documented.
- Stats/bookkeeping goes in dedicated `@dataclass` types, not loose module-level variables.
- Use `from __future__ import annotations` at the top of every module.

## Import Order

```
# 1. stdlib
# 2. third-party  (numpy, yaml, torch, …)
# 3. local        (murimsim.*)
```

Separated by blank lines. No unused imports.

## Logging

- No `print()` in library code (`murimsim/`). Use `logging` or return values.
- Validation scripts and notebooks may use `print()`.
- wandb is **disabled in all tests** (mock or skip).

## Project Structure

```
murimsim/          # library code only
config/            # YAML configs only — no Python here
tests/             # pytest tests, one file per module
scripts/           # validation + experiment runners (may print)
logs/              # generated output — gitignored
checkpoints/       # model weights — gitignored
viewer/            # web replay viewer (Phase 1.5)
```

## Tech Stack

| Layer       | Library                     |
|-------------|------------------------------|
| Core sim    | Python 3.11+, numpy, PyYAML |
| Testing     | pytest                       |
| Validation  | matplotlib                  |
| RL (Ph 2+)  | CleanRL / SB3, PettingZoo   |
| LLM (Ph 5+) | litellm                     |
| Logging     | wandb (disabled in tests)   |

## Two-Lane Development Approach

### Fast Lane (this repo — default mode)
**Every session is a coding or training session.** Default behavior is to ship.

- **`build:` / `train:`** — implement feature or run training script, commit when green
- **`fix:`** — debug, patch, test
- **`plan:`** — update plan.md only, no code
- **`coach:`** — brief explanation only; redirect deep theory to ChatGPT

When a task is given without a prefix, treat it as `build:`.

### Slow Lane (NOT this repo)
Theory, coaching, curriculum design → ChatGPT Plus or Gemini Pro.
Do not spend repo session tokens on RL concept explanations.

### Current Fast-Lane Ticket Queue (in priority order)
1. `fix-gather-eat` — decouple starvation penalty from inventory shaping in `murimsim/rl/env.py`
2. `lstm-v3-training` — retrain LSTM after reward fix (depends on #1)
3. `sect-scaffold` — SectConfig + SectRegistry + 3 isolated envs
4. `multiagent-credit` — per-agent reward tracking in multi_env.py (depends on #3)
5. `viewer-replay` — verify web viewer renders LSTM v2 replay

**Always run `python3 -m pytest tests/ -q` before starting and after finishing a ticket.**

### Current Project State (as of 2026-03-27)
- Phases 1–3 complete: world, viewer, RL survival + combat trained
- Latest checkpoint: `checkpoints/limbic_lstm_v2/limbic_lstm_v2_final.zip`
- 70 tests passing, 3 skipped
- Open bug: gather rate dropped (12%) due to reward interaction — fix is ticket #1
