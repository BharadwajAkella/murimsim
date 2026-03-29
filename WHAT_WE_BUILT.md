# MurimSim — What We Built

A multi-agent survival simulation where emergent group behavior (alliances, cooperation, combat tactics) arises naturally from reinforcement learning — not hand-coded rules.

**Current state:** 88 tests passing, LSTM v8 trained to 2M steps, lifespan of 105.7 (peak 115.0), group food sharing and coordinated combat active, replay viewer running.

---

## Table of Contents

1. [Project Vision](#1-project-vision)
2. [Architecture Overview](#2-architecture-overview)
3. [The World](#3-the-world)
4. [The Agent](#4-the-agent)
5. [Reinforcement Learning Environment](#5-reinforcement-learning-environment)
6. [Group Mechanics (Phase 5)](#6-group-mechanics-phase-5)
7. [Training Pipeline](#7-training-pipeline)
8. [Replay Viewer](#8-replay-viewer)
9. [Training History — LSTM v1 → v8](#9-training-history--lstm-v1--v8)
10. [Reward Function Reference](#10-reward-function-reference)
11. [Key Design Decisions](#11-key-design-decisions)
12. [What Comes Next](#12-what-comes-next)

---

## 1. Project Vision

Build a simulation of 3 martial arts sects in a resource-scarce 2D world. Agents have:
- **Instincts** — driven by a trained RL model (frozen after training)
- **Personality** — deterministic trait parameters assigned at birth (not yet active)
- **Culture** — LLM-shaped selection pressures that evolve across generations (not yet active)

**The rule:** Named abstractions like "alliance" or "territory" are only added after the behavior they represent is observed emerging from the simulation on its own.

---

## 2. Architecture Overview

```
┌────────────────────────────────────────────────────┐
│  LLM Culture Layer      (per generation — Phase 6) │
│  Observes sect stats → biases parent selection      │
├────────────────────────────────────────────────────┤
│  Deterministic Individual Model  (per tick — Ph 4) │
│  AgentParams modulate RL logits at decision time    │
├────────────────────────────────────────────────────┤
│  RL Limbic Model        (trained — frozen in sim)  │
│  RecurrentPPO, LSTM hidden state, obs size 261      │
└────────────────────────────────────────────────────┘
         ↕
    MultiAgentEnv  (PettingZoo-compatible, Phase 5)
         ↕
    World (30×30 grid, 6 resource types)
```

Currently only the RL layer is active. The individual and culture layers are designed and planned but not yet implemented in code.

---

## 3. The World

**File:** `murimsim/world.py`

A 30×30 grid where each cell can hold resources. The world is **data-driven** — adding a new resource type requires only a YAML change, no code changes.

### Resources (defined in `config/default.yaml`)

| ID | Effect | Notes |
|----|--------|-------|
| `food` | Positive — restores 0.30 hunger | Gatherable, respawns every 200 ticks, 5% spawn density |
| `qi` | Positive — strength gain | Respawns every 400 ticks |
| `materials` | Neutral | Gatherable, respawns every 300 ticks |
| `poison` | Negative — health damage | Spawns in clusters, resistance reduces damage |
| `flame` | Negative — hazard tile | Traversal damage |
| `mountain` | Negative — impassable terrain | Blocks movement |

### World Mechanics
- `world.step()` advances time: handles resource regeneration, decay
- `deplete(resource_id, x, y)` removes a unit from a tile
- Domain randomisation optional: food regen, spawn density, action ticks all randomisable per episode
- **Seeded RNG throughout** — `np.random.default_rng(seed)` only; no global state mutation

---

## 4. The Agent

**File:** `murimsim/agent.py`

All mutable stats are floats in **[0.0, 1.0]**. This is a hard invariant throughout the codebase.

### Agent Stats

| Stat | Range | Description |
|------|-------|-------------|
| `health` | [0, 1] | 0 = dead. Damaged by combat, starvation, poison |
| `hunger` | [0, 1] | 1 = starving. Increases by 0.01/tick |
| `strength` | [0, 1] | Combat effectiveness; also tracks fitness |
| `adventure_spirit` | [0, 1] | Scales exploration reward — curious agents range farther |
| `sociability` | [0, 1] | Personality trait — influences collaborate/fight/flee decisions |
| `resistances` | dict[str, float] | Per-hazard resistance, acquired through survival |
| `inventory` | `AgentInventory` | Counts of food, qi, materials, poison carried |

### Inventory

```python
@dataclass
class AgentInventory:
    food: int = 0
    qi: int = 0
    materials: int = 0
    poison: int = 0
```

Gathering puts items into inventory. Eating consumes from inventory. An agent carrying poison and blindly eating will take damage.

### Key Behaviours

- `tick()` — hunger +0.01/tick; starvation damage at hunger ≥ 1.0
- `eat(resource_configs)` — eats from inventory (food first; poison if no food)
- `gather(resource_id)` — adds one unit to inventory
- `move(dx, dy, grid_size)` — moves with boundary clamping
- `rest()` — passive: health +0.01/tick (recovery)
- Poison resistance is **Lamarckian**: end-of-life resistance value is inherited by children (planned for Phase 5)

---

## 5. Reinforcement Learning Environment

### Environment Stack

```
CombatEnv (Phase 5)
    └─ MultiAgentEnv (Phase 3–4)
           └─ Env (Phase 2 — base survival env)
```

Each layer adds mechanics on top of the previous and inherits the same test suite.

### Observation Space — 261 floats

```
[0:100]   Resource grid   — 5×5 local view × 4 channels (food, qi, materials, poison)
[100:200] Agent grid      — 5×5 local view × 4 channels (present, health, strength, sociability)
[200:250] Stash grid      — 5×5 local view × 2 channels (stash presence, food quantity)
[250:261] Self stats      — health, hunger, inv_food, inv_poison, poison_resist, poison_intake,
                            combat_experience, terrain_familiarity, reward_ema, sociability, in_group
```

All values normalised to [0, 1]. The LSTM hidden state (64 units) persists across steps, giving memory of past events.

### Action Space — 14 discrete actions (Phase 5)

| Action | Description |
|--------|-------------|
| `move_n/s/e/w` | Cardinal movement |
| `gather` | Pick up resource from current tile |
| `eat` | Consume food from inventory |
| `rest` | Recover health, no movement |
| `deposit` | Put food into a shared stash |
| `withdraw` | Take food from a shared stash |
| `steal` | Take from another agent's stash |
| `attack` | Attack nearest agent within Chebyshev distance 1 |
| `defend` | Block stance — reduces incoming damage |
| `collaborate` | Attempt to form a group with adjacent agent |
| `walk_away` | Create distance from a nearby threat |

### Multi-Agent Setup

- **10 agents per environment, 4 parallel envs** during training
- One agent is the **focal agent** whose action the RL model controls each step
- All other agents run a **heuristic policy**: eat if hungry → gather if food nearby → navigate toward food
- Focal index rotates among live agents
- Combat-aware heuristic: heuristic agents attack the focal if they are stronger and unsocial

### Curriculum

Combat starts masked off. Over the first 300K training steps, the probability of allowing combat actions ramps from 20% → 100%. This prevents the agent being overwhelmed by combat before it can survive on its own.

---

## 6. Group Mechanics (Phase 5)

All group behavior was added in Phase 5 across versions v5–v8. The key principle: **never hand-code strategy** — only add reward signals that make the desired behavior more valuable, then let the agent discover it.

### Group Formation

- **`COLLABORATE` action** — focal signals group intent to the nearest adjacent agent
- If that agent's `sociability ≥ 0.5`, a group is formed (bilateral consent required)
- Groups are stored as `frozenset[int]` (agent indices) in `self._groups`
- An agent can belong to at most one group
- Dead agents are pruned from groups each tick

### Combat Mechanics

Combat requires **Chebyshev distance ≤ 1** (8 directions including diagonals). This means diagonal positioning is valid for both attacking and flanking.

**Damage formula:**
```
damage = attacker.strength × 0.3 − defender.strength × 0.1 × is_defending
damage = clip(damage, 0, 0.5)
```

**Flanking attack bonus:**
When the focal agent attacks and has a group ally adjacent to the target (from any of the 8 surrounding cells), each flanking ally grants +20% damage:
```
damage *= (1 + 0.20 × N_flankers)
```

This is the only group combat advantage. There is no damage sharing/shielding — the benefit is purely offensive, meaning groups only help when agents coordinate positioning before attacking.

### Group Cohesion Reward

Every step, the focal agent earns +0.02 per live group member within Chebyshev distance 3. This creates a continuous pull toward clustering with allies rather than dispersing after formation.

### Coordinated Attack Reward

When the focal agent attacks while a group ally is flanking the target, it earns an additional +0.10 reward. This directly incentivises the agent to position itself next to allies **before** engaging in combat.

### Food Sharing with Reciprocity

When any group member's hunger exceeds 0.85 (critical starvation), nearby group members automatically attempt to share food. This runs as a background mechanic — it is not an agent action.

**Reciprocity system:**
- Base share probability: **50%**
- Boosted probability: **85%** if the recipient helped the sharer in the last 100 steps
- Memory resets each episode (no cross-life debt)
- `_help_received[recipient_idx][helper_idx] = step_when_helped` tracks who helped whom

**Effect in v8:** Eat rate surged from 22% → 32%, confirming agents are eating food shared by allies, not just their own gathered food.

### Reward for Sharing

Focal agent earns +0.04 per food share event it participates in (as sharer or recipient), creating a bidirectional incentive to stay in a group near allies.

---

## 7. Training Pipeline

**Script:** `scripts/train.py`

```bash
python scripts/train.py --run-name lstm_v8 --warmstart checkpoints/limbic_lstm_v7/limbic_lstm_v7_final.zip
```

### Algorithm: RecurrentPPO (sb3-contrib)

| Parameter | Value |
|-----------|-------|
| Policy | `MlpLstmPolicy` |
| LSTM hidden size | 64 |
| LSTM layers | 1 |
| n_envs | 4 |
| n_steps | 512 |
| batch_size | 64 |
| total_timesteps | 2,000,000 |

### Warm-Starting

The training script auto-detects the latest checkpoint and transfers all matching layers. When obs size is unchanged (as in v5→v6→v7→v8), all 20 layers transfer cleanly. When obs size changes, only matching layers transfer and new layers initialise randomly.

### Dashboard

A live `logs/dashboard_data.js` file is written every 20K steps during training. Opening `viewer/dashboard.html` shows:
- Live lifespan curve
- Action rate breakdown
- Avg strength KPI
- Stage comparison table (all versions v1–current)

---

## 8. Replay Viewer

**Files:** `viewer/index.html`, `viewer/viewer.js`

A pure-JS web viewer that loads `.jsonl` replay files.

### Features
- Canvas-based 2D grid render (30×30)
- Resource tiles coloured by type (green=food, blue=qi, yellow=materials, purple=poison, red=flame, brown=mountain)
- Agents rendered as circles:
  - **Soft purple** `#a78bfa` when alive and neutral
  - **Flashing red** `#ef4444` when `action === "attack"` or `"defend"` (alternates each tick + red glow ring)
  - **Grey** `#555` when dead
- Selected tile highlighted in yellow
- Selected agent highlighted with gold ring
- Inspector panel: full agent stats, current action, resistances, inventory
- Event log per tick
- Play/pause, step, speed slider (1–60 FPS), scrub bar
- Generation boundary markers on scrub bar

### Generating Replays

```bash
python scripts/record_combat.py --model checkpoints/limbic_lstm_v8/limbic_lstm_v8_final.zip --ticks 1200 --seed 42
# Output: /mnt/c/Users/bhara/Downloads/replays/combat_42.jsonl
```

Replay files are `.jsonl` — one JSON object per tick, containing all agent states, positions, actions, and world resource grids.

---

## 9. Training History — LSTM v1 → v8

| Version | Steps | Lifespan | Key Changes |
|---------|-------|----------|-------------|
| MLP v1 | 500K | ~40 | Baseline survival, MLP policy |
| MLP v1b | +200K | ~55 | Reward tuning |
| MLP v1c | +300K | ~60 | Combat curriculum added |
| LSTM v1 | 500K | ~75 | MlpLstmPolicy, memory of past |
| LSTM v2 | 2M | ~85 | Full Phase 3 combat, potential-based shaping |
| LSTM v3 | 300K | ~89 | Decoupled starvation penalty |
| LSTM v4 | 2M | ~90 | Strength tracking, reward EMA in obs |
| LSTM v5 | 2M | **94.1** | Phase 5: sociability trait, COLLABORATE/WALK_AWAY |
| LSTM v6 | 2M | 93.7 | Flanking bonus, cohesion reward, damage split |
| LSTM v7 | 600K* | 85.5 | 8-dir combat, coordinated attack reward (*config bug cut short) |
| **LSTM v8** | **2M** | **105.7** | Food sharing + reciprocity. Peak 115.0 at 1.74M steps |

\* v7 was cut off by a config bug (`training.yaml` had 600K instead of 2M). Despite the short run, collaborate jumped to 7.1% (best ever) before recovery was interrupted.

### Key Milestones
- **v5 → v6:** Agents discovered collaborative actions (collaborate 0% → 3.2% → 4.6%)
- **v7:** Collaborate hit 7.1% — best ever — demonstrating 8-dir flanking mechanics working
- **v8:** Lifespan 85.5 → **105.7** (+20 points). Attack rate 1.7% → **0.3%** (near-zero). Food sharing confirmed active via eat rate 22% → 32%.

---

## 10. Reward Function Reference

### Survival Rewards (`MultiAgentEnv._compute_reward`)

| Signal | Value | Condition |
|--------|-------|-----------|
| Alive per step | +0.02 | Always |
| Hunger relief | +0.20 × Δhunger | When hunger decreases |
| Food gathered | +0.10 per item | On GATHER action |
| Inventory security | +0.12 × Δ(food/5) | Potential-based shaping |
| Exploration | +0.25 × adventure_spirit × (1−hunger) | New tile visited |
| Starvation approach | −0.08 × (hunger − 0.70) | When hunger > 0.70 |
| Poison damage | −0.30 × damage | When poisoned |
| Death | −1.00 | On death |

### Combat Rewards (`CombatEnv._compute_combat_reward`)

| Signal | Value | Condition |
|--------|-------|-----------|
| Defeat opponent | +0.30 | On killing an enemy |
| Damage taken | −0.20 × damage | Per damage unit received |

### Group Rewards

| Signal | Value | Condition |
|--------|-------|-----------|
| Group formation | +0.05 | When COLLABORATE succeeds (both agents) |
| Cohesion per ally | +0.02/step | Per live group member within Chebyshev 3 |
| Coordinated attack | +0.10 | On ATTACK with flanking ally present |
| Food share | +0.04 | When focal shares or receives food from group |

---

## 11. Key Design Decisions

### All Traits Are [0, 1]
Every agent stat, reward signal, and personality trait is a float in [0, 1]. Display-only scaling to 0–100 is done in the viewer. This makes the math consistent and prevents scale mismatches across subsystems.

### Seeded Determinism
`np.random.default_rng(seed)` is used everywhere. Global state (`np.random.seed()`) is never touched. Same seed → byte-identical simulation.

### Data-Driven Resources
Adding a new resource type requires only a YAML entry in `config/default.yaml`. No code changes. The world, agent, and env all read resource configs at runtime.

### RL Is Frozen After Training
Once trained, the RL weights are never updated during simulation. Personality traits (planned Phase 4) modulate the logits at decision time without changing the underlying model. This separation is fundamental to the three-layer architecture.

### Emergent-First Design
No group mechanics were hand-coded as policies. Every feature is a **reward signal** that makes some behavior more valuable. The agent discovers when to use it. The test for success: observe behavior that wasn't programmed.

### Phase Gates
All tests from earlier phases must keep passing when new phases are added. This is enforced by running the full test suite before and after every commit.

---

## 12. What Comes Next

The sim has stable group survival mechanics. The next layer is **sect differentiation** — three populations that evolve different strategies.

### Tier 1 — Immediate
1. **`biome-maps`** — 3 YAML env configs with distinct terrain (one already pending in todos)
2. **`sect-scaffold`** — SectConfig + SectRegistry + 3 parallel isolated envs
3. **`viewer: territory`** — restore per-sect colors, show territory boundaries

### Tier 2 — After Sects
4. **`inter-sect combat`** — sect-aware combat bonuses vs. enemies, penalties vs. allies
5. **`territory anchor reward`** — reward agents for staying in their biome region
6. **`multi-env credit assignment`** — per-sect scoreboard for generation stats

### Tier 3 — Phase 6
7. **`generation stats collector`** — emit sect-level summary (pop, food, kills, territory) at end of generation
8. **`LLM culture layer`** — litellm call per generation; returns `selection_bias` and `mutation_rate_modifier` to drive next-gen trait inheritance

### Trait Inheritance System (Phase 4/5)
Agents are born with personality traits inherited from parent survivors:
```python
def inherit_value(mom, dad, rng, sigma):
    return clip(0.5 * mom + 0.5 * dad + rng.normal(0, sigma), 0, 1)
```
- Traits: `aggression`, `selfcontrol`, `greed`, `courage`, `loyalty`, `poison_resistance`
- Default mutation sigma per trait: 0.05–0.10
- LLM can scale sigma up to 2× (more variation) or down to 0.5× (conservative)
- Poison resistance is Lamarckian: end-of-life value passes to children

---

*Last updated after LSTM v8 training — lifespan 105.7, peak 115.0, 88 tests passing.*
