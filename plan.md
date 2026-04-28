# MurimSim — Emergent Behavior Master Plan (Canonical 0–1 Trait Scale)

## ⚓ MVP Anchor — Read Before Every Session

> This project has a hard demo deadline of **June 30, 2026** for outreach to Dr. Katja Hofmann (MSR) and Joel Leibo (DeepMind).
> Strategic context lives in `docs/positioning.md`. MVP scope cuts live in `docs/MVP_SCOPE.md`. Read both before making design decisions.

### The three claims this demo must demonstrate

1. **Generational cultural drift in a multi-agent population via LLM transmission medium** — combination not present in Park et al., Voyager, or Cook et al.
2. **Reproducible strategic emergence under survival pressure** — same seed → same inter-sect dynamics
3. **Identity-aware agents with sect-level cultural state** — sect identity in observations, sect-level inherited culture

### Architecture commitments (do not revisit without explicit unlock)

- **LLM is called only at generation boundaries**, not within generations. Deliberate cost + reproducibility delta vs Park et al. (LLM-every-step) and a capability delta vs Cook et al. (no LLM).
- **Sect identity in agent observations.** Cook et al. agents are interchangeable; ours are not.
- **Sect-level outcomes feed individual rewards.** Without this, sects are cosmetic.
- **Cultural transmission Mechanism 1 first** (programmed parent→offspring + sect-elder transmission). Mechanism 2 is **stretch**, decision deferred to mid-June.
- **Relationships in v1 are pairwise parameter maps** (trust, interaction frequency).

### MVP path — the only Fast Lane that matters until June 30

1. ✅ `settlement-metrics` — pure instrumentation, no game changes
2. ✅ `shared-stash` — group withdraw, foraging-outward, stash capacity cap
3. ✅ `sect-scaffold` — three sects, home regions, inter-sect combat rewards
4. ✅ `aging` + `reproduction` — age-death, trait inheritance, `spawn_from_parents`
5. ✅ Combat audit + power growth (v15→v16) — multiplicative DEFEND, damage obs, invalid-action redirect
6. 🔲 `v17-fixes` — investigate WITHDRAW non-retrieval and strength regression (small scope; before sects-with-multiagent)
7. 🔲 `viewer-territory` — per-sect colors for demo legibility
8. 🔲 Phase 8 LLM culture (Mechanism 1) — selection bias + mutation modifier at generation boundaries
9. 🔲 Reproducibility sweeps — multi-seed runs, same-seed → same emergence
10. 🔲 Demo recording + repo polish + outreach email (June 30)

**Architectural note (Entry 21 in journal.md):** Current training uses focal-agent (1 RL student + 9 heuristic NPCs per env). True multi-agent learning requires IPPO migration. **For the MVP demo, focal-agent is acceptable** — sect-level cultural drift across generations is the claim, not multi-agent emergent coordination per-step. IPPO is post-Katja unless time allows.

### What's cut for MVP

- Phase 7 (movable resources, haul action) — post-Katja
- `settlement-training-v9` full retraining run — map and metrics exist; skip retraining
- Replay viewer polish — screen recording with overlay text is sufficient
- Mechanism 2 — stretch only; default to deferral

### The "greed at end of project" rule

New ideas during paternity leave → `docs/positioning.md` "Deferred extensions / future work". Not into code.

---


> A multi-agent simulation where agents compete over resources in a procedural world.
> Named abstractions (groups, alliances, culture) are built **only after** the behavior they
> represent is observed emerging from the simulation.
> Emergent group strategies — cooperation, warfare, cultural drift — with minimal manual intervention.
>
> **Canonical rule:** All traits/params are floats in **[0.0, 1.0]** everywhere in code + logs.
> If you want “0–100” for UI, do **display-only scaling** in the viewer: `display = round(100 * value)`.

---

## Table of Contents

1. [Vision & Scope](#vision--scope)
2. [Architecture — Three Models, Three Timescales](#architecture--three-models-three-timescales)
3. [Training Order](#training-order--why-rl-first)
4. [Phase 1: World Mechanics](#phase-1-world-mechanics)
5. [Phase 1.5: Sim Replay Viewer (Web)](#phase-15-sim-replay-viewer-web)
6. [Phase 2: RL Limbic — Survival](#phase-2-rl-limbic-model--survival-only)
7. [Phase 3: RL Limbic — Combat](#phase-3-rl-limbic-model--add-combat)
8. [Phase 4: Deterministic Individual Model](#phase-4-deterministic-individual-model)
9. [Phase 5: Sects + LLM Culture](#phase-5-sects--llm-culture-layer)
10. [Phase 6: Full Integration + Emergence](#phase-6-full-integration--emergence-analysis)
11. [Project Structure](#project-structure)
12. [Tech Stack](#tech-stack)
13. [Working with AI Coding Agents](#working-with-ai-coding-agents)
14. [Study Materials](#study-materials)
15. [Study vs. Doing](#study-vs-doing-honest-assessment)
16. [Progress Tracker](#progress-tracker)

---

## Vision & Scope

**What we're building:** A vertical-slice simulation with 3 martial-arts sects in a resource-scarce 2D world. Agents have instincts (RL), personalities (deterministic params), and culture (LLM-shaped). We want emergent strategies — alliances, wars, resource denial, cultural assimilation — that we didn’t hand-code.

**What we're NOT building (yet):**

* Physics-based martial arts animation (Project 1 — connect later)
* Massive scale (100s of agents per sect) — start with ~10 per sect
* 3D rendering — 2D grid with simple visualization
* Real-time gameplay — this is a research simulation

**Success looks like:** Run the sim for 50+ generations, observe at least one non-trivial emergent behavior (e.g., two sects gang up on the third, a sect deliberately starves another, cultural convergence under pressure) that was NOT programmed.

---

## Architecture — Three Models, Three Timescales

Three distinct layers. Validate each layer independently before adding the next.

```
┌───────────────────────────────────────────────────────────┐
│                                                           │
│   LLM CULTURE LAYER            (slowest: per generation)  │
│                                                           │
│   Observes sect-level outcomes (survival rate, wars won,  │
│   resources, population trends).                          │
│   Outputs: cultural pressures that affect parent          │
│   selection and mutation this generation.                 │
│                                                           │
│   Does NOT touch RL weights or the deterministic fn.      │
│                                                           │
├───────────────────────────────────────────────────────────┤
│                                                           │
│   DETERMINISTIC INDIVIDUAL MODEL         (per tick)       │
│                                                           │
│   Takes: RL logits + individual params + observation      │
│   Returns: a single action (deterministic, no randomness) │
│                                                           │
│   Params: aggression, selfcontrol, greed, courage,        │
│           loyalty (floats 0.0–1.0)                        │
│   Same inputs → same output, always.                      │
│   No learning. Params assigned at birth.                  │
│                                                           │
├───────────────────────────────────────────────────────────┤
│                                                           │
│   RL LIMBIC MODEL               (slow: across generations)│
│                                                           │
│   Small MLP trained with PPO.                             │
│   Input: local observations (5x5 grid + own stats)        │
│   Output: action logits (base behavioral preferences)     │
│   Trained on: survival + resource reward                  │
│   Frozen during each generation's lifetime.               │
│                                                           │
│   THIS GETS TRAINED FIRST.                                │
│                                                           │
├───────────────────────────────────────────────────────────┤
│                                                           │
│   WORLD                          (tick-based 2D grid)     │
│                                                           │
│   30x30 grid (expand to 50x50 once stable)                │
│   Resources: food, qi, materials, poison — procedural     │
│   Fog-of-war: agents see 5x5 local neighborhood          │
│   Deterministic given seed. Tick-based.                   │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

### How Data Flows (Per Tick)

```
observation ──→ RL Limbic Model ──→ base logits ──→ Individual Model ──→ action
                  (frozen)            │                (deterministic)
                                      │
                              individual params
                              (born from genetics + selection,
                               influenced by culture pressures)
```

### How Culture Updates (Per Generation)

```
generation ends ──→ collect sect stats ──→ LLM prompt ──→ selection/mutation pressures
                    (survival, wars,        │               │
                     resources, pop, poison)│               ▼
                                            │         next generation:
                                            │         parents selected + children bred
                                            │
                                        knowledge base
                                        (append summary)
```

---

## Training Order — Why RL First

**RL limbic MUST be trained first.** You can’t layer personality/culture on broken instincts.

```
Phase 1:    World only (no agents)
Phase 1.5:  Sim Replay Viewer (build after Phase 1, before RL training)
Phase 2:    RL trains in simple world (forage + poison)
Phase 3:    RL retrains with combat added (curriculum)
Phase 4:    Freeze RL, add deterministic personality modulation
Phase 5:    Add sects + inheritance + LLM culture pressures
Phase 6:    Full integration + experiments + emergence analysis
```

Hard gate: **all tests from previous phases must still pass.**

---

## Phase 1: World Mechanics

> Goal: Deterministic 2D grid world with resources. No agents yet.

### What to Build

| Component         | Details                                                                                                                                                                                                                             |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Project structure | Python package, `pyproject.toml`, dependencies, `config.yaml`                                                                                                                                                                       |
| Config system     | YAML-based. Grid size, resource types, spawn densities, regen ticks, seed                                                                                                                                                           |
| RNG               | Seeded numpy RNG. Same seed = same world, always                                                                                                                                                                                    |
| Grid              | 2D world state (30×30). Internal representation supports multiple resource layers                                                                                                                                                   |
| Resource registry | **Data-driven resource system.** Each resource defined in config with: `id`, `display_name`, `regen_ticks`, `spawn_density`, `effect` (positive/negative/neutral), `effect_params`. Adding a resource = YAML only, no code changes. |
| Resource types    | `food` (fast regen, abundant, positive), `qi` (slow regen, scarce, positive), `materials` (medium regen, neutral), `poison` (slow regen, sparse, negative)                                                                          |
| Poison            | Spawns in clusters. Visually distinct internally. **Untrained agents can’t distinguish it from food** unless you implement a learned-knowledge channel later. (Phase 1: just world mechanics + correct internal IDs.)               |
| Regeneration      | Depleted node → respawns after `regen_ticks`                                                                                                                                                                                        |
| World tick        | `world.step()` advances time, handles regeneration, updates state                                                                                                                                                                   |
| Logging           | wandb init optional (disabled for tests)                                                                                                                                                                                            |

### Automated Tests (pytest)

| Test                                | What it checks                                                    |
| ----------------------------------- | ----------------------------------------------------------------- |
| `test_world_determinism`            | Seed 42 run 200 ticks == seed 42 run 200 ticks                    |
| `test_resource_conservation`        | Bookkeeping: created/depleted/regenerated adds up                 |
| `test_resource_regeneration`        | Deplete node → respawns after configured ticks                    |
| `test_world_bounds`                 | Nothing exists outside `[0, grid_size)`                           |
| `test_config_loading`               | YAML changes apply correctly                                      |
| `test_different_seeds`              | Seed 42 != seed 99                                                |
| `test_resource_registry_extensible` | New resource via config only works (spawn + regen)                |
| `test_poison_spawns`                | Poison appears at expected density; distinct from food internally |
| `test_poison_regen`                 | Poison respawns after configured ticks                            |

### Validation Script

`scripts/validate_world.py`:

* Runs 100 ticks
* Prints resource counts per type per tick
* Saves resource density heatmap PNG
* Saves resource count timeseries plot

### Exit Gate

✅ All 9 tests pass
✅ Heatmap reasonable; poison clusters visible and distinct
✅ Timeseries stable (not → 0 or → ∞)
✅ Same seed identical output
✅ Adding new resource via YAML works with no code changes

---

## Phase 2: RL Limbic Model — Survival Only

> Goal: Single-agent PPO learns foraging + survival in world with poison. No combat, no sects.

### What to Build

| Component              | Details                                                                                                                                     |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Agent class            | `health` (0–1), `hunger` (0–1), inventory (counts per resource), position, `poison_resistance` (0–1)                                        |
| Poison resistance init | Randomized at birth from config range, default `[0.05, 0.30]`                                                                               |
| Observation            | 5×5 local grid **4 channels**: food, qi, materials, poison + own stats (health, hunger, inventory summary, poison_resistance) → flat vector |
| Actions                | Discrete(7): move_N/S/E/W, gather, eat, rest                                                                                                |
| Gather                 | Picks up resource from current tile (food or poison possible)                                                                               |
| Eat                    | Consumes from inventory. If poison: `damage = max(0, poison_potency - effective_resistance)`; **immunity at res>=1.0 → damage=0**          |
| Poison adaptation      | If survives poison: `poison_resistance += 0.05 * (1 - poison_resistance)`                                                                   |
| Hunger                 | `hunger += 0.01/tick`; if `hunger > 1.0` health decreases                                                                                   |
| Gym env                | Wrap World + single Agent. Seed-deterministic                                                                                               |
| PPO training           | CleanRL or SB3. MLP 2×64                                                                                                                    |
| Reward (shaped)        | Base: `+0.02` alive + `+0.2*(hunger_prev - hunger_now)` + `+0.05*food_gained` − `0.3*poison_damage` − `1.0` death                            |
| Reward (behavioral)    | Encourage “camping” near reliable food: bonus for resting when local food density is high + hunger low; penalty for moving in that state    |
| Reward (explore gate)  | Penalize exploration when hunger high unless moving increases local food density                                                            |
| Logging                | wandb: survival time, reward, episode length                                                                                                |
| Checkpoints            | Save every 100k steps → `checkpoints/limbic_v1/`                                                                                            |

### Behavioral Targets (Phase 2)

| Target                                  | Definition                                                                                               |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Fear poison, seek food                  | Avoid poison tiles when food is available; prioritize food gathering and eating                          |
| Explore when safe                       | Explore for new food when hunger is low or food density is poor                                          |
| “Safe haven” camping                    | If local food density is high, prefer REST / low movement until hunger rises                             |
| Poison-resistant archetype              | Learn to build resistance to 1.0; once immune, poison zones become viable safe territory                 |

### Automated Tests

| Test                                   | What it checks                                      |
| -------------------------------------- | --------------------------------------------------- |
| `test_agent_observation_shape`         | Obs vector dims match spec                          |
| `test_action_execution`                | All 7 actions change state correctly                |
| `test_reward_calculation`              | Hand-crafted transitions → exact reward             |
| `test_env_reset`                       | Reset yields valid initial state                    |
| `test_env_determinism`                 | Same seed + action seq → identical trajectory       |
| `test_agent_dies_no_food`              | Empty grid → death within expected ticks            |
| `test_agent_survives_with_food`        | On food tile + optimal actions → survive 500+ ticks |
| `test_hunger_increases`                | Hunger increments as expected                       |
| `test_gathering_works`                 | Gather adds inventory, depletes tile                |
| `test_poison_gather`                   | Poison tile gather → poison in inventory            |
| `test_poison_eat_kills_low_resist`     | `res=0.05`, potency=0.4 → damage=0.35               |
| `test_poison_eat_survives_high_resist` | `res>=0.4` → damage small, resistance grows         |
| `test_poison_resistance_growth`        | Survive poison → `+= 0.05*(1-res)`                  |
| `test_poison_resistance_randomized`    | 100 agents span configured init range               |
| `test_poison_immunity_at_one`          | `res=1.0` → damage=0.0                              |

### Behavioral Probes

| Probe                        |                                          Criterion | How to test                |
| ---------------------------- | -------------------------------------------------: | -------------------------- |
| `probe_survival_improvement` |                       Trained survival > 3× random | 100 eps each               |
| `probe_food_seeking`         |                             Moves toward food >70% | Controlled placement       |
| `probe_eating_when_hungry`   |                       Hunger>0.7 + food → eat >90% | Controlled                 |
| `probe_not_wasting_food`     |                              Hunger<0.2 → eat <20% | Controlled                 |
| `probe_poison_avoidance`     |        Avoids poison >60% when food also available | Controlled                 |
| `probe_poison_vs_starvation` | With only poison + hunger>0.9: eats poison or dies | Either valid, log strategy |
| `probe_camping_behavior`     |             When local food density high → REST >50% | Controlled                 |
| `probe_hunger_gated_explore` | When hunger>0.7 and food density high → MOVE <20% | Controlled                 |

### Exit Gate

✅ Phase 1 tests still pass
✅ Phase 2 tests pass
✅ Probes pass (the 4 core probes + poison probe acceptable)
✅ Learning curve improves
✅ `limbic_v1.ckpt` saved

---

## Phase 3: RL Limbic Model — Add Combat

> Goal: Multi-agent RL learns fight/flight without collapsing foraging.

### What to Build

| Component         | Details                                                                                                   |
| ----------------- | --------------------------------------------------------------------------------------------------------- |
| Multi-agent world | 10 agents on 30×30; each has `strength` in 0–1                                                            |
| New actions       | `attack`, `defend`                                                                                        |
| Action space      | Discrete(9): Phase2 actions + attack + defend                                                             |
| Combat            | `damage = attacker.strength*0.3 - defender.strength*0.1*is_defending`, clamped [0, 0.5]                   |
| Death             | health ≤ 0 → dies, inventory dropped                                                                      |
| Multi-agent env   | PettingZoo parallel API; simultaneous actions                                                             |
| Observation       | Add nearby agents + approximate strength if adjacent                                                      |
| Reward            | Phase2 reward + `−0.2*damage_taken` + `+0.3*defeat_opponent` − `1.0` death                                |
| Curriculum        | Combat disabled **80%** at start. Linearly ramp combat probability from 20% → 100% over first 300k steps. |
| Training          | Warm-start from `limbic_v1.ckpt`; train 500k more steps                                                   |

### Automated Tests

| Test                                     | What it checks                            |
| ---------------------------------------- | ----------------------------------------- |
| `test_combat_determinism`                | Same stats/positions → same damage        |
| `test_combat_damage`                     | Known attacker/defender → expected damage |
| `test_combat_damage_defending`           | Defend reduces damage                     |
| `test_death_drops_inventory`             | Drops appear on tile                      |
| `test_multiple_agents_coexist`           | 10 agents, 1000 ticks no crash            |
| `test_attack_requires_adjacency`         | Non-adjacent attack → no-op               |
| `test_agent_observation_includes_others` | Obs includes nearby agents                |
| `test_strength_affects_combat`           | High strength wins in controlled setup    |
| `test_combat_curriculum_schedule`        | Prob schedule matches spec                |

### Behavioral Probes

| Probe                        |                                 Criterion | How to test   |
| ---------------------------- | ----------------------------------------: | ------------- |
| `probe_flight_from_stronger` | Flees >60% when opponent stronger by >0.3 | 1v1 scenarios |
| `probe_fight_weaker`         |               Attacks >50% vs much weaker | 1v1 scenarios |
| `probe_survival_with_combat` |                  Mean survival >200 ticks | Full sim      |
| `probe_not_suicidal`         |             health<0.3 avoids combat >80% | Controlled    |

### Exit Gate

✅ Phase 1–2 tests still pass
✅ Phase 3 tests pass
✅ Probes pass
✅ Differentiated behavior (fight/flight depends on strength)
✅ `limbic_v2.ckpt` saved

---

## Phase 4: Deterministic Individual Model

> Goal: Deterministic function modulates RL logits using personality params. No learning.

### What to Build

| Component             | Details                                                                                                                              |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `AgentParams`         | `aggression`, `selfcontrol`, `greed`, `courage`, `loyalty`, `poison_resistance` all in [0,1]. **Only poison_resistance is mutable.** |
| `individual_action()` | `(rl_logits, params, observation) → action` (deterministic)                                                                          |
| Frozen RL             | Load `limbic_v2.ckpt` inference-only                                                                                                 |

### Modulation Function (reference)

```python
@dataclass
class AgentParams:
    aggression: float
    selfcontrol: float
    greed: float
    courage: float
    loyalty: float
    poison_resistance: float  # mutable

def individual_action(rl_logits: np.ndarray, params: AgentParams, obs: dict) -> int:
    logits = rl_logits.copy()

    logits[ACTION_ATTACK] *= (1.0 + params.aggression * 1.5)
    logits[ACTION_REST]   *= (1.0 - params.aggression * 0.5)

    if obs['hunger'] < 0.3:
        logits[ACTION_EAT] *= (1.0 - params.selfcontrol * 0.8)
    if obs['health'] > 0.7:
        logits[ACTION_ATTACK] *= (1.0 - params.selfcontrol * 0.4)

    logits[ACTION_GATHER] *= (1.0 + params.greed * 1.0)

    logits[ACTION_FLEE_N:ACTION_FLEE_W+1] *= (1.0 - params.courage * 0.6)

    return int(np.argmax(logits))
```

### Automated Tests

| Test                                  | What it checks                                         |
| ------------------------------------- | ------------------------------------------------------ |
| `test_determinism`                    | Same inputs → same output always                       |
| `test_aggression_increases_attack`    | aggression 0→1 increases attack logit monotonically    |
| `test_selfcontrol_suppresses_impulse` | High selfcontrol attacks less in same scenario         |
| `test_greed_increases_gathering`      | High greed increases gather preference                 |
| `test_courage_reduces_fleeing`        | High courage reduces flee preference                   |
| `test_extreme_params`                 | No NaN/crash at 0.0 or 1.0                             |
| `test_default_params`                 | params=0.5 stays close to raw RL behavior              |
| `test_params_clamped`                 | Out-of-range → clamped or error (no silent corruption) |

### Behavioral Probes

| Probe                         |                                          Criterion | How to test                     |
| ----------------------------- | -------------------------------------------------: | ------------------------------- |
| `probe_personality_diversity` |                     30 random agents → ≥3 clusters | K-means on action distributions |
| `probe_param_sensitivity`     | Change one param by 0.3 shifts behavior measurably | KL divergence > threshold       |
| `probe_rl_backbone_preserved` |                   Default-param survival ~ pure RL | Compare survival distributions  |

### Exit Gate

✅ Phase 1–3 tests still pass
✅ Phase 4 tests + probes pass
✅ Deterministic, inspectable, testable

---

## Phase 5: Sects + LLM Culture Layer

> Goal: 3 sects. Traits evolve via inheritance + selection; LLM applies cultural pressures (selection bias + mutation scaling).

### What to Build

| Component        | Details                                                                                                                           |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `Sect`           | members, **parent pool** (survivors), knowledge base, treasury, inheritance config                                                |
| Sects            | Iron Fang (warriors), Jade Lotus (diplomats), Shadow Root (hoarders)                                                              |
| Territory        | Start in corners; contested middle                                                                                                |
| Generation cycle | Run 1000 ticks → collect stats → survivors become parent pool → breed children → LLM returns pressures → apply pressures next gen |
| Stats            | survival rate, resources, wars, pop, territory size, avg poison_resistance, poison deaths                                         |
| LLM output       | `selection_bias` in [-1,1] per trait; `mutation_rate_modifier` in [0.5,2.0]                                                       |

### Trait Inheritance (Canonical 0–1)

**All traits are [0,1].** `sigma` is also in 0–1 units.

```python
def inherit_value(mom: float, dad: float, rng: np.random.Generator, sigma: float) -> float:
    # midpoint blending + genetic noise
    return float(np.clip(0.5 * mom + 0.5 * dad + rng.normal(0.0, sigma), 0.0, 1.0))

def inherit_params(mom: AgentParams, dad: AgentParams, rng: np.random.Generator,
                   mutation_scale: float = 1.0) -> AgentParams:
    # Per-trait sigma in 0–1 units
    trait_sigma = {
        "aggression":         0.10,  # moderate
        "selfcontrol":        0.10,
        "greed":              0.10,
        "courage":            0.10,
        "loyalty":            0.10,
        "poison_resistance":  0.05,  # more heritable
    }
    child_kwargs = {}
    for trait, base_sigma in trait_sigma.items():
        sigma = float(np.clip(base_sigma * mutation_scale, 0.0, 0.50))
        mom_v = getattr(mom, trait)
        dad_v = getattr(dad, trait)
        child_kwargs[trait] = inherit_value(mom_v, dad_v, rng, sigma)
    return AgentParams(**child_kwargs)
```

### Parent Selection

* Parent pool = survivors.
* Pair within sect (seeded RNG).
* Each pair produces 1–3 children based on treasury food.
* If <2 survivors: extinct OR recruit (config).

### Poison Resistance Evolution

Poison resistance is both **inherited** and **acquired**:

* Birth: via inheritance with `sigma=0.05`
* Lifetime: on surviving poison
  `poison_resistance += 0.05 * (1 - poison_resistance)`
* **Lamarckian twist (intentional):** the *end-of-life* poison_resistance is used as the parent value.

### How LLM Culture Interacts

LLM doesn’t set trait distributions directly. It outputs pressures:

```json
{
  "selection_bias": {
    "aggression": 0.3,
    "selfcontrol": -0.1,
    "greed": 0.0,
    "courage": -0.2,
    "loyalty": 0.1,
    "poison_resistance": 0.2
  },
  "mutation_rate_modifier": 1.2,
  "reasoning": "After poison losses, favor resistant and cautious parents and increase variation."
}
```

**Interpretation rule (simple):**

* For each trait t, compute a parent score modifier:

  * If bias `b > 0`: higher trait values selected more.
  * If bias `b < 0`: lower trait values selected more.
* Use soft selection:

  * `score = exp(k * b * (trait - 0.5))` with k ~ 2–6
* Normalize scores into sampling probabilities.
* `mutation_rate_modifier` scales inheritance sigma (`mutation_scale`).

Clamp:

* bias ∈ [-1, 1]
* mutation_rate_modifier ∈ [0.5, 2.0]

### LLM Prompt Template

```
You are the cultural memory of the {sect_name} sect.

Current generation stats:
- Survival rate: {survival_rate}%
- Resources gathered: food={food}, qi={qi}, materials={materials}
- Wars initiated: {wars_initiated}, won: {wars_won}, lost: {wars_lost}
- Population: {population} (started with {start_pop})
- Territory: {territory_tiles} tiles
- Poison deaths: {poison_deaths}
- Avg poison resistance (survivors): {avg_poison_resist:.2f}

Recent history (last 3 generations):
{knowledge_base_excerpt}

Surviving parent pool trait averages:
- aggression: {agg_avg:.2f}
- selfcontrol: {sc_avg:.2f}
- greed: {greed_avg:.2f}
- courage: {cour_avg:.2f}
- loyalty: {loy_avg:.2f}
- poison_resistance: {pr_avg:.2f}

You influence who breeds and how much variation the next generation has.
Output ONLY valid JSON:
{
  "selection_bias": {
    "aggression": <float, -1.0 to 1.0>,
    "selfcontrol": <float, -1.0 to 1.0>,
    "greed": <float, -1.0 to 1.0>,
    "courage": <float, -1.0 to 1.0>,
    "loyalty": <float, -1.0 to 1.0>,
    "poison_resistance": <float, -1.0 to 1.0>
  },
  "mutation_rate_modifier": <float, 0.5 to 2.0>,
  "reasoning": "<1 sentence>"
}
```

### Automated Tests

| Test                                 | What it checks                                   |
| ------------------------------------ | ------------------------------------------------ |
| `test_sect_initialization`           | 3 sects created; correct initial param ranges    |
| `test_generation_cycle`              | Run gen; stats; parent select; breed; spawn      |
| `test_trait_inheritance_continuous`  | Child mean ≈ parent midpoint over many births    |
| `test_trait_inheritance_determinism` | Same parents + seed → same child                 |
| `test_trait_sigma_per_trait`         | poison_resistance variance < aggression variance |
| `test_parent_selection_bias`         | bias=+1 favors high-trait parents (chi-square)   |
| `test_mutation_rate_modifier`        | modifier=2 → child variance increases ~2×        |
| `test_poison_resistance_inherited`   | High-res parents → higher-res children           |
| `test_lamarckian_pass_through`       | Uses end-of-life poison_resistance               |
| `test_llm_output_parsing`            | Parse mock response correctly                    |
| `test_selection_bias_clamping`       | clamp bias/mutation outputs                      |
| `test_sect_stats_accurate`           | survival + poison deaths correct                 |
| `test_qi_accumulation`               | gather qi → treasury increases                   |
| `test_knowledge_base_grows`          | KB appends per generation                        |
| `test_mock_llm_determinism`          | Same stats + seed → same pressures/outcomes      |
| `test_sect_extinction`               | <2 survivors handled per config                  |

### Behavioral Probes

| Probe                           |                              Criterion | How to test             |
| ------------------------------- | -------------------------------------: | ----------------------- |
| `probe_sect_differentiation`    |              Sects differ after 5 gens | KS test on action hists |
| `probe_culture_drift`           |           Trait means change over gens | assert not static       |
| `probe_llm_responds_to_loss`    |             3 war losses → bias shifts | direction change        |
| `probe_no_trait_collapse`       |    trait variance > 0.03 after 10 gens | per-trait std           |
| `probe_sects_play_differently`  | Iron Fang attacks more than Jade Lotus | compare rates           |
| `probe_poison_resistance_rises` |     Poison-heavy → resistance trend up | Spearman r > 0.3        |
| `probe_inheritance_matters`     |     High-agg parents → higher-agg kids | distribution comparison |

### Exit Gate

✅ Phase 1–4 tests still pass
✅ Phase 5 tests + probes pass
✅ Cheap LLM integration works; mock LLM for tests
✅ Sects differentiate; traits drift sensibly

---

## Phase 6: Full Integration + Emergence Analysis

> Goal: Everything active; run experiments; detect emergence.

### What to Build

| Component         | Details                                                       |
| ----------------- | ------------------------------------------------------------- |
| Full loop         | World + RL + individual + sects + genetics + culture + qi     |
| Experiment runner | `run_experiment.py --config exp1.yaml --seed 42` reproducible |
| Batch runner      | N seeds; aggregate                                            |
| Visualization     | (Optional) pygame live view; primary viewer is replay         |
| Dashboard         | per-generation charts                                         |
| Analysis          | scripts below                                                 |

### Automated Tests

| Test                            | What it checks                            |
| ------------------------------- | ----------------------------------------- |
| `test_full_sim_100_generations` | no crashes/NaN/loops                      |
| `test_population_nonzero`       | at least one sect survives                |
| `test_resource_economy_stable`  | bounded resources                         |
| `test_reproducibility`          | same seed + mock LLM identical trajectory |
| `test_all_layers_active`        | RL loaded, individual applied, LLM called |
| `test_metrics_logged`           | expected keys exist                       |

### Emergence Analysis Scripts

| Analysis                        | Question               | Method                       |
| ------------------------------- | ---------------------- | ---------------------------- |
| `analysis_strategy_diversity`   | Different strategies?  | K-means on action dists      |
| `analysis_war_rationality`      | Wars rational?         | correlate wars with scarcity |
| `analysis_alliance_detection`   | 2v1 alliance?          | pairwise attack rates        |
| `analysis_cultural_convergence` | converge/diverge?      | KL divergence of trait dists |
| `analysis_equilibrium`          | reach equilibrium?     | variance trends              |
| `analysis_qi_advantage`         | qi → combat advantage? | correlate qi with win rate   |

### Exit Gate

✅ Stable 100+ generations
✅ At least one emergent behavior not programmed
✅ Analysis reports meaningful
✅ Patterns reproducible across seeds

---

## Phase 1.5: Sim Replay Viewer (Web)

> Goal: browser-based replay of recorded runs. No backend. Built after Phase 1, before RL training.

### Record → Replay

```
Simulation run                           Browser viewer
─────────────                            ──────────────
sim.py ──→ logs/replays/run_42.jsonl ──→ viewer/index.html (loads file)
```

### Tick Log Format (`run_{seed}.jsonl`)

One JSON object per line:

```json
{
  "tick": 147,
  "generation": 3,
  "agents": [
    {
      "id": "iron_fang_04",
      "sect": "iron_fang",
      "pos": [12, 7],
      "health": 0.82,
      "hunger": 0.45,
      "poison_resistance": 0.21,
      "action": "gather",
      "action_detail": "Gathered food (inv_food: 3)",
      "alive": true
    }
  ],
  "resources": {
    "food": [[12,7,1.0], [2,5,0.4]],
    "qi": [[10,10,0.7]]
  },
  "events": [
    {"type": "combat", "attacker": "iron_fang_04", "defender": "shadow_root_02", "damage": 0.12},
    {"type": "death", "agent": "jade_lotus_07", "cause": "starvation"}
  ]
}
```

### Viewer Features

* Canvas grid; resources colored by type/intensity
* Agent dots by sect
* Click agent → sidebar stats (health/hunger/resistance) + action + inventory
* Playback controls + speed + scrub
* Generation markers
* Overlay stats

### What to Build

| Component   | File                 | Details                                      |
| ----------- | -------------------- | -------------------------------------------- |
| Tick logger | `murimsim/replay.py` | `ReplayLogger` called each tick writes JSONL |
| Viewer      | `viewer/index.html`  | Single file + canvas                         |
| JS          | `viewer/viewer.js`   | parsing, playback, rendering, click          |

### Tests (Logger)

| Test                        | What it checks                               |
| --------------------------- | -------------------------------------------- |
| `test_replay_logger_writes` | 10 ticks → 10 lines                          |
| `test_replay_format_valid`  | each line parses; required keys              |
| `test_replay_agent_fields`  | agent fields present incl. poison_resistance |
| `test_replay_toggle`        | replay disabled → no file                    |

---

## Project Structure

```
murimsim/
├── pyproject.toml
├── config/
│   ├── default.yaml
│   └── experiments/
├── murimsim/
│   ├── world.py
│   ├── agent.py
│   ├── actions.py
│   ├── individual.py
│   ├── sect.py
│   ├── culture.py
│   ├── genetics.py
│   ├── replay.py
│   ├── rl/
│   │   ├── env.py
│   │   ├── train.py
│   │   └── policy.py
│   ├── sim.py
│   └── viz.py
├── viewer/
│   ├── index.html
│   └── viewer.js
├── scripts/
│   ├── validate_world.py
│   ├── run_experiment.py
│   ├── batch_run.py
│   └── analyze.py
├── tests/
│   ├── test_world.py
│   ├── test_agent.py
│   ├── test_env.py
│   ├── test_combat.py
│   ├── test_individual.py
│   ├── test_sect.py
│   ├── test_culture.py
│   ├── test_genetics.py
│   ├── test_replay.py
│   ├── test_integration.py
│   └── probes/
├── checkpoints/
│   ├── limbic_v1.ckpt
│   └── limbic_v2.ckpt
├── logs/
│   ├── llm_calls/
│   ├── replays/
│   └── experiments/
└── README.md
```

---

## Tech Stack

| Component   | Choice              | Why             |
| ----------- | ------------------- | --------------- |
| Language    | Python 3.11+        | RL ecosystem    |
| RL          | CleanRL or SB3      | readable PPO    |
| Multi-agent | PettingZoo          | standard API    |
| World sim   | numpy               | speed + control |
| Viz         | pygame / matplotlib | live + analysis |
| Logging     | wandb               | training + runs |
| Config      | PyYAML              | simple          |
| LLM         | litellm / direct    | swap models     |
| Testing     | pytest              | fast + standard |

`pyproject.toml` deps unchanged from your draft.

---

## Working with AI Coding Agents (Prompt Principles)

1. Every task has a test.
2. Determinism is sacred.
3. Phase gates: earlier tests must keep passing.
4. One layer per session.
5. Paste the architecture diagram in every AI prompt.

(Your prompt templates remain valid; the only change is: **traits are always 0–1** and `sigma` values are in 0–1 units.)

---

## Study Materials / Study vs Doing / Progress Tracker

Keep as-is; no unit changes needed.

---

## Progress Tracker

| Phase                         | Status        | Key Milestone                                        | Exit Gate                                              |
| ----------------------------- | ------------- | ---------------------------------------------------- | ------------------------------------------------------ |
| Phase 1: World                | ✅ Done        | Deterministic grid + resource registry + poison      | 9 tests pass + heatmaps                                |
| Phase 1.5: Viewer             | ✅ Done        | Browser replay viewer + tick logger                  | Logger tests pass + viewer renders recorded ticks      |
| Phase 2: RL Survival          | ✅ Done        | Forage + poison survival                             | Probes pass + `limbic_v1.ckpt`                         |
| Phase 3: RL Combat            | ✅ Done        | Fight/flight w/o forgetting forage                   | Probes pass + `limbic_lstm_v2_final.zip`               |
| Phase 4: Reward Tuning        | ✅ Done        | Strength priority, gather/eat decoupled              | LSTM v4: 74 steps, 7.3% defend, `limbic_lstm_v4_final.zip` |
| Phase 5: Group Dynamics       | ✅ Done        | Groups, flanking, cohesion, food sharing + reciprocity | LSTM v8: lifespan 105.7, attack 0.3%, eat 31.9% |
| Phase 5.5: Settlement         | ✅ Done        | Settlement metrics, dense-patch map, shared stash, capacity cap | 5 settlement metrics logged; v16 closed deposit loop (4.6%) |
| Phase 5.6: Power Growth       | ✅ Done        | TRAIN action, qi field, hazard unify, multiplicative DEFEND, damage obs | v16: TRAIN 31.3%, resistance 0.268, age 210.7 |
| Phase 6a: Sect Scaffold       | ✅ Done        | 3 sects, home regions, inter-sect combat rewards, sect obs | SectConfig + SectRegistry + DEFAULT_SECTS in `sect.py` |
| Phase 6b: Generations         | ✅ Done        | Aging, age-death, reproduction, trait inheritance    | `Agent.spawn_from_parents()`, `_try_reproduce()` wired |
| Phase 7: Movable Resources    | 🚫 Cut (MVP)  | **Cut for June 30 demo.** Post-Katja.                | —                                                      |
| Phase 8: LLM Culture          | 🔲 Next        | LLM selection bias + mutation modifier at generation boundaries | Cultural drift measurable across generations |

**Current test suite:** 165 passed, 3 skipped (as of 2026-04-27)
**Latest checkpoint:** `checkpoints/limbic_lstm_v16/limbic_lstm_v16_final.zip` — 1.5M steps, warm-start from v15 (weight surgery for obs 263→264). Lifespan 210.7 (age), DEPOSIT 4.6%↑↑, STEAL 2.7% emerged, DEFEND collapsed to 0.9% (model prefers WALK_AWAY 4.4%↑). Open issues for v17: WITHDRAW retrieval not actually moving items; strength regression (0.727→0.473).

> **Design principle:** No named groups, no top-down abstractions until the behavior they represent
> is observed emerging from the simulation. Code follows behavior, not the other way around.

---

## Phase 5: Group Dynamics ✅ COMPLETE

All group mechanics shipped through LSTM v8. Key results:
- Lifespan: **105.7** (peak 115.0 at step 1.74M) — all-time record
- Attack rate: 1.7% → **0.3%** (near-zero aggression)
- Eat rate: 22% → **31.9%** (food sharing confirmed active)
- Collaborate rate: **6.2%**
- 8-directional flanking (Chebyshev), coordinated attack bonus, food sharing with reciprocity memory all implemented

---

## Phase 5.5: Settlement Benchmark

> Goal: detect whether group clustering is becoming persistent settlement behavior, not just
> temporary proximity. All sub-phases are strictly sequenced — instrument first, then pressure.

### Why Settlement Before Sects
Sects need a home region to make sense. Before adding sect structure, we need to confirm agents
*can* settle — i.e., they return to the same region, maintain a stash, and forage outward.
If they don't settle naturally by Phase 5.5, we cannot build sects on top of that behavior.

### 5.5a — Settlement Metrics (instrument only, no game changes)
Add to `MultiAgentEnv` step logging:
- `stash_fill_rate`: items deposited / items gathered per episode
- `stash_withdraw_rate`: items withdrawn / items deposited per episode  
- `avg_dist_from_stash`: mean Chebyshev distance of each agent from its own stash over episode
- `revisit_entropy`: how concentrated each agent's grid visits are (low entropy = settled)
- `group_persistence`: avg ticks a group stays together without dissolving

Log these per-episode to `logs/`. Dashboard widget in `logs/stage_history.js` optional.

### 5.5b — Dense-Patch Map
Create one environment layout in `config/envs/dense_patch.yaml`:
- Two high-food regions in opposite corners (2×food spawn density vs default)
- Sparse food everywhere else
- Same hazards as default

Train one eval run (no warm-start needed, use LSTM v8 frozen) to measure settlement baseline
against the new metrics. Do not launch a new training session yet.

### 5.5c — Shared Stash Improvements
After baseline is measured, improve group-stash interaction:
- Group members can withdraw from any group member's stash (not just their own)
- Foraging-outward reward: small bonus when agent returns to stash with food after being ≥5 tiles away
- Group-eat bonus: small bonus when two group members eat within the same 5-tick window near a stash

**Obs space note:** 5.5c does NOT change obs size (261 floats). Stash channel already present.
Any obs-size change triggers a partial-transfer warm-start — document it explicitly.

---

## Phase 5.6: Power Growth

> Only start after 5.5c is stable and settlement metrics are positive.

### Qi Training Action
- Add `train` to action space (obs size increases — requires partial warm-start from v8)
- Training near qi tile: `strength += 0.01 * qi_density_at_tile` per tick
- Training anywhere else: `strength += 0.002` per tick (background growth)
- Training costs the tick (opportunity cost vs foraging)

### Power Score
```
power = 0.4 * strength + 0.3 * qi_reserve + 0.2 * poison_resistance + 0.1 * flame_resistance
```
Log `power` separately from `lifespan`. Power growth should not substitute for survival.

### Hazard Unification
Merge poison and flame into one generic hazard system:
```yaml
# config/default.yaml
hazards:
  - id: poison_pool
    damage_per_tick: 0.02
    resistance_stat: poison_resistance
    resistance_growth_rate: 0.001   # per tick of survivable exposure
  - id: flame_vent
    damage_per_tick: 0.03
    resistance_stat: flame_resistance
    resistance_growth_rate: 0.0008
```
Remove duplicate poison/flame code paths. One `apply_hazard(agent, hazard_def)` function handles both.
Curriculum: expose agents at low concentration first, increase over training.

---

## Fast Lane — Active Tickets

> Coding tasks only. Run `pytest` before and after each ticket. No ticket is done until tests are green.
> **Obs space rule:** Any ticket that changes the 261-float obs vector must note the new size and
> plan for partial warm-start from v8 (new dims init to zero, policy retrained for 500K+ steps).

### Phase 5.5 — Settlement Benchmark ✅ COMPLETE

| ID | Task | Status | Depends On |
| -- | ---- | ------ | ---------- |
| `settlement-metrics` | Per-episode logging: stash_fill_rate, stash_withdraw_rate, avg_dist_from_stash, revisit_entropy, group_persistence. | ✅ Done | — |
| `dense-patch-map` | `config/envs/dense_patch.yaml` with 2× food density in 2 corners. Baseline eval run (random policy). | ✅ Done | `settlement-metrics` |
| `shared-stash` | Group withdraw, foraging-outward reward, stash capacity cap (`STASH_MAX_ITEMS=20`), action masking for full stash. | ✅ Done | `dense-patch-map` |
| `settlement-training-v9` | **Cut for MVP.** Map + metrics exist; retraining skipped. Bring in v8 checkpoint for eval. | 🚫 Cut (MVP) | `shared-stash` |

### Phase 5.6 — Power Growth ✅ COMPLETE (v10 → v16)

| ID | Task | Status | Depends On |
| -- | ---- | ------ | ---------- |
| `hazard-unify` | Merge poison + flame into one `apply_hazard(agent, hazard_def)` function. YAML-driven. No obs change. | 🔲 Pending | `settlement-training-v9` |
| `qi-train-action` | Add `train` action to action space. Strength growth from qi tiles. **Obs size changes — partial warm-start required.** | 🔲 Pending | `hazard-unify` |
| `power-score` | Add `power` metric to logging: `0.4*strength + 0.3*qi + 0.2*poison_res + 0.1*flame_res`. Log separately from lifespan. | 🔲 Pending | `qi-train-action` |
| `power-training-v10` | Train LSTM v10 warm-start (partial transfer, new action dim). Watch power growth vs lifespan tradeoff. | 🔲 Pending | `power-score` |

### Phase 6a — Sect Scaffold ✅ COMPLETE

| ID | Task | Status | Depends On |
| -- | ---- | ------ | ---------- |
| `sect-scaffold` | `SectConfig` + `SectRegistry` + `DEFAULT_SECTS` (Iron Fang/Jade Lotus/Shadow Root) in `sect.py`. Home regions (horizontal stripes of 30×30). Sect obs + sect-level metrics in `CombatEnv`. | ✅ Done | `settlement-metrics` |
| `inter-sect-combat` | Sect-aware attack bonus vs enemy sect (`REWARD_INTER_SECT_DEFEAT_BONUS`), penalty vs own sect (`REWARD_SAME_SECT_ATTACK_PENALTY`). | ✅ Done | `sect-scaffold` |
| `viewer-territory` | Per-sect agent colors in `viewer/viewer.js`. Show sect home regions on grid. | 🔲 Pending | `sect-scaffold` |

### Phase 6b — Generations ✅ COMPLETE

| ID | Task | Status | Depends On |
| -- | ---- | ------ | ---------- |
| `aging` | Age counter per agent. Death by age threshold. Age-death in `tick()`. | ✅ Done | `inter-sect-combat` |
| `reproduction` | `Agent.spawn_from_parents()`, `_try_reproduce()` in `CombatEnv`. `inherit_value()` midpoint + Gaussian noise. Lamarckian poison resistance. | ✅ Done | `aging` |

### Phase 6c — v17: Emergence Pressure Test 🔲 NEXT

> **Hypothesis:** Resistance reward and high deposit reward have been training wheels.
> Removing them while introducing a common-enemy boss should make agents *discover*
> strength acquisition, defensive cultivation, and group cohesion as survival strategies
> rather than as reward-shaped habits. v17 = combined experiment. Fallback: revert to v16.

| ID | Task | Status | Depends On |
| -- | ---- | ------ | ---------- |
| `v17-instr-friendly-steal` | Tag each `steal()` as `pure_steal` vs `friendly_steal` (owner ∈ thief's group). Per-episode counters in env info. | ✅ Done | — |
| `v17-instr-collab-vs-life` | Per-episode log: `(collaborate_count, ticks_alive)` for focal agent. Dashboard scatter + correlation. | ✅ Done | — |
| `v17-instr-flee-cause` | On WALK_AWAY, log `own_strength`, `nearest_enemy_strength`, `own_health`. Answers: strength differential or reflex? | ✅ Done | — |
| `v17-instr-bank-vs-granary` | Split WITHDRAW telemetry into `bank_withdrawals` (own) vs `granary_withdrawals` (group). Track `items_per_withdraw`. | ✅ Done | — |
| `v17-fix-withdraw` | Diagnose v16's WITHDRAW=4.5% / 0-items-retrieved bug. Mask + redirect now require an actual stash at the agent's position (own or group-mate's). | ✅ Done | `v17-instr-bank-vs-granary` |
| `v17-reward-resist-zero` | `REWARD_RESISTANCE_GAIN_SCALE: 5.0 → 0.0`. Test if cultivation arc holds without training wheels. | ✅ Done | — |
| `v17-reward-deposit-down` | `REWARD_DEPOSIT_PER_ITEM: 0.05 → 0.02`. Less aggressive shaping, still discoverable. | ✅ Done | — |
| `v17-boss-monster` | Separate `Monster` / `BossMonster` / `MonsterRegistry` (not Agent flag — clean polymorphism for future monster kinds). 5× HP, ~2× strength, aggressive heuristic that hunts nearest agent. Permadeath, no natural death. On death: drops shared loot stash via new `Stash.participants` field — every agent who landed ≥1 attack can WITHDRAW. CombatEnv has `enable_boss=False` default; pass `True` for v17 train. | ✅ Done | — |
| `v17-train` | Warm-start from `checkpoints/limbic_lstm_v16/limbic_lstm_v16_final.zip` (same obs 264, action 15, arch). **v16's learned TRAIN habit (31.3%) is load-bearing — without inherited cultivation competence, the boss instantly wipes a village of squishy newborns and the emergence experiment can't run.** 1.5M steps, same hyperparams as v16, pass `enable_boss=True`. Only fall back to v16_init / from-scratch if v16's baked-in deposit-hoarding refuses to unlearn. | 🔲 Pending | All above |
| `v17-eval` | 2000-step eval seed=42; compare full action breakdown + new metrics vs v16. Decision gate: keep v17 or revert to v16. | 🔲 Pending | `v17-train` |

### Deferred Ideas Backlog (revisit before each major training run)

User-flagged ideas to revisit *before locking in v18+ training*. Do not let
these die quietly.

| Idea | Source | Why deferred | Trigger to revisit |
| ---- | ------ | ------------ | ------------------ |
| **Predictive sneak-attack / steal** — agents that anticipate near-term death switch strategy to opportunistic theft or unprovoked attack | v17 design discussion ("if you think you'll die soon, then go for sneak attacks") | Requires predictive value head or explicit horizon-aware heuristic. Touches reward + obs schema. | After v17 results — if survival is brittle, this becomes survival lever |
| **Personality / "elation of conquest"** — deterministic per-agent modulation of attack appetite based on personality traits and recent kill streak | v17 design discussion ("agents need more personality before they choose combat") | Belongs in limbic-system / mood layer; needs design pass. | Phase 7+ when limbic system gets revisited |
| **IPPO migration** — multi-agent PPO so all 10 agents learn simultaneously instead of one focal | journal Entry 21, v16 retrospective | Architectural lift; explicitly post-Katja. | After June 30 demo |
| **Perception step** — each agent observes its neighborhood and notes other agents' actions; future obs channel | v17 design discussion ("perception will let friendly_steal become meaningful") | Needs new obs channel + per-agent observation buffer. | Concurrent with mid-June LLM gating decision |
| **LLM transmission (Mechanism 2)** — agent-to-agent linguistic signal | journal + Anthropic doc review | Heavyweight; gated on Mechanism 1 success. | Mid-June check-in once v17/v18 results land |
| **Sect-vs-sect dedicated training** — 3 sect envs trained in lockstep with shared policy, then evaluated head-to-head | refactor_plan + sect tickets | Multi-env infra not yet integrated with focal-agent loop. | After v17 gives a stable single-env policy |
| **Predictive value head / health prediction** — auxiliary target for "will I be alive in N ticks?" | v17 discussion offshoot | Needs SB3 policy override; non-trivial. | Same trigger as predictive sneak-attack |
| **Bank vs granary A/B test** — run v17 with bank-only stash semantics vs granary-only, see which produces richer collab | v17 discussion (user: "is there a way to have both kinds and see what gives interesting results") | Currently both coexist (`withdraw` vs `withdraw_group`); A/B requires config flag + 2 training runs. | After v17 shows whether shared stash is even being used |
| **Continuous qi-spend per strike** — replace tiered ATTACK/ATTACK_QI/ATTACK_BURST with a continuous (or bucketed-fine) qi amount per strike, so the policy interpolates rather than picks from 3 fixed slots | v18 design discussion (user: "we want to think of a continuous scale instead of discrete for qi in the future") | True hybrid action space (Discrete + Box) breaks SB3 `MlpLstmPolicy` / `MaskablePPO`; needs custom ActionDistribution + Policy (~3-5 days, fragile) OR migration to RLlib (~1-2 weeks). Cheap path = bucket into 5-7 discrete tiers (~half-day, 95% of value). Cost map captured in the v18 implementation chat. | Bundle with RLlib + CTDE migration — that path gives hybrid actions for free. Until then, just add more discrete tiers if 3 proves too coarse. |
| **Multi-box / cloud training scaling** — distribute training across two devboxes or move to a single beefier cloud box for runs >10M steps | v18 design discussion (user: "is there a way to scale across two dev boxes... when should I hit this with more muscle than smarts?") | DIY multi-box is painful (gradient sync, env shard coordination, network faults). Single beefier cloud box (e.g. 64-core Hetzner / EC2 c7i.16xlarge) is the easy answer for 10–50M steps. For 50M+ steps use Ray RLlib on a small cluster — also unblocks CTDE. | Trigger: any planned run >10M steps, OR when v18 wall-time on the 16-core devbox exceeds 24h. |
| **CTDE / MAPPO migration** — Centralized Training Decentralized Execution: agents see global state during training but execute on local obs only; replaces single-focal PPO with true multi-agent learning | v18 design discussion (user: "I am interested in learning about CTDE") | Requires migrating from SB3 PPO → MAPPO/QMIX (likely PettingZoo + RLlib). ~1-2 week port. Pays off bigger than IPPO for cooperation tasks. | After Mechanism-1 cultural transmission lands AND v18 shows that policy needs richer coordination than PPO can give. Bundle with RLlib migration above. |

### Phase 7 — Movable Resources ⛔ CUT FOR MVP

| ID | Task | Status | Depends On |
| -- | ---- | ------ | ---------- |
| `haul-action` | Add `haul` action only if logs show: stable home regions + spatially uneven resources + returning > consuming in place. Otherwise skip. | ⬜ Conditional | `aging` |

---

## Slow Lane — Coaching Topics

> Theory and curriculum questions. Take these to ChatGPT/Gemini — not this repo.

- Settlement vs roaming: what reward structures encourage one vs the other?
- Group reward attribution: how to credit individual agents for group outcomes
- Qi training opportunity cost: how to set the reward so training is worth it but doesn't dominate
- When to introduce dominance/subordination axis (what observable behavior earns it)
- When to introduce LLM culture pressure — what signal triggers the prompt?
- Genetic drift vs learned drift: how to measure which is driving differentiation
- Population-based outer loop timing: when does PPO inner loop plateau enough to warrant outer loop?

---

## Not Now (do not implement until explicitly unlocked)

| Feature | Unlocked by |
| ------- | ----------- |
| Barter / crafting economy | Never in limbic phase |
| Sect secrets / martial knowledge transfer | Phase 8 LLM culture |
| Symbolic diplomacy | Post-generation stability |
| Procreation mechanics | Phase 6b aging scaffold |
| Special poison-trained subpopulation | Hazard unification + sect scaffold |
| Complex resource hauling | `haul-action` conditional gate |
| 3D rendering / animation | Separate project |

---