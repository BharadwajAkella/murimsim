# MurimSim Design Journal

A running log of interesting discussions, decisions, and tradeoffs made during
development. Intended as raw material for a future blog post.

Covers all training runs v1 through v13.

---

## Entry 1 — The Vision: Emergent Behavior Over Hand-Coded Rules

**Context:** Before a line of code was written, a design principle was established:
no named abstractions (sects, alliances, groups) until the behavior they represent
is *observed*. A "sect" would only exist in the code once agents were already
clustering by behavior. This is the founding philosophy of the project.

**The three-layer architecture:**
- **RL limbic layer** — a small LSTM trained with PPO. Pure survival instinct.
  Handles the fast timescale (every tick). Trained first, frozen during simulation.
- **Deterministic individual model** — personality params (aggression, courage, loyalty)
  that filter the RL logits. Same inputs always produce same outputs. Assigned at birth.
- **LLM culture layer** — observes sect-level outcomes per generation, outputs cultural
  pressure that modifies trait selection and mutation. The slowest timescale.

**Tradeoff:** Each layer is validated independently before adding the next. This is
slower than building everything at once, but prevents the failure mode where you can't
tell which layer is responsible for a behavior.

**Interesting note:** The architecture is deliberately "biology-flavored." The RL layer
is called the "limbic system" — instinct. The culture layer is equivalent to epigenetics.
This framing helped make design decisions: instinct doesn't reason, culture doesn't
micromanage.

---

## Entry 2 — Two-Lane Development (v1→v3)

**Discussion:** The project was trying to do too much in one place — building
features, running training, and learning RL theory all in the same session. Token
burn was high and progress was slow.

**Decision:** Split into two lanes:
- **Fast Lane** (Copilot CLI): coding sessions and training runs only. Every
  session ships something.
- **Slow Lane** (ChatGPT / Gemini): RL theory, coaching, long-document review.

**Tradeoff:** Coaching slows feature velocity. Separating the two means you
sometimes implement before fully understanding — but the simulation itself
becomes the teacher. Watching a replay often teaches more than an hour of theory.

**Interesting note:** The "slow lane" insight was that the simulation is a learning
instrument for the *developer*, not just a test bed. The emergent behaviors you
observe directly teach you what the reward function is actually optimizing.

**Session rule encoded into the repo:** Every Copilot CLI session is either a coding
session or a training session. Prefix conventions (`build:`, `fix:`, `coach:`, `plan:`)
were added to the project instructions. No coaching conversations in the fast lane.

---

## Entry 3 — The Gather/Eat Reward Bug (v2→v3)

**Problem:** LSTM v2 showed a puzzling behavioral pattern — agents were eating from
the ground instead of gathering into inventory first. Gather rate dropped from
20% (v1) to 12% (v2). The agent looked like it was being irrational.

**Root cause:** Two reward signals were fighting each other:
- `PENALTY_STARVATION_APPROACH`: fired when hunger exceeded a threshold,
  creating urgency to fix hunger *immediately*
- `REWARD_HUNGER_RELIEF_SCALE`: rewarded reducing hunger per step

Together they made eating from the ground (instant hunger relief) more attractive
than gathering (delayed hunger relief via inventory). The agent wasn't broken —
it was perfectly optimizing the *actual* reward function, which had an unintended
interaction.

**Fix:** Removed `PENALTY_STARVATION_APPROACH`. Replaced with
`PENALTY_EMPTY_INV_FOOD_NEARBY`: only fires when inventory is empty AND food is
nearby. This creates incentive to gather proactively without creating urgency to
eat immediately.

**Key lesson:** When an agent does something that looks stupid, check the reward
function first. The agent is always doing the right thing for the reward it received.

**v3 result:** Gather rate recovered to 14.8%, eat rate dropped from 43% to 23.8%.
The fix confirmed working.

---

## Entry 4 — Removing the Sect Scaffold (v3)

**Background:** Early in development, a full sect scaffold was built: `SectConfig`,
`SectRegistry`, three named sects with distinct trait distributions. It seemed like
the right foundation for the multi-sect competition that was the end goal.

**Decision:** Remove it entirely.

**Reasoning:** The sect system was encoding the answer before the question had been
asked. "Iron Fang agents are strong" predetermines what makes them different. The
goal is to *discover* why groups diverge, not to declare it in advance.

**New rule:** "Code follows behavior, not the other way around." The sect scaffold
would be rebuilt later — but only after observing natural clustering. When agents
spontaneously stay in certain regions or avoid certain others, *that* is when you add
the vocabulary to describe it.

**Broader principle:** Premature abstraction is the enemy of emergence. Every
hand-coded "because agents of type X do Y" prevents discovering that agents of type X
actually do Z under pressure.

---

## Entry 5 — Group Dynamics: Sociability Trait + COLLABORATE/WALK_AWAY (v4→v5)

**Discussion:** The question was how to add group behavior without encoding strategies
explicitly. Two approaches were considered:
1. Hard-code group tactics (follow leader, share food equally, etc.)
2. Add personality traits and encounter choices, let strategy emerge

**Decision:** Approach 2. Added `sociability` trait [0,1] to each agent. Added two new
actions: `COLLABORATE` (form a group with the nearest agent) and `WALK_AWAY` (move one
step away). Group formation is bilateral — the focal agent chooses COLLABORATE, but the
partner "consents" based on their static sociability trait (≥ 0.5 threshold).

**Key design choice:** COLLABORATE is a one-time group formation signal. It doesn't
make agents move together — they still act independently. The group tag only means:
(1) no friendly fire, (2) `in_group=1.0` in obs vector, (3) small formation bonus.

**Tradeoff:** This means "collaboration" is a weak mechanic in v5. Groups form but
don't actually help each other survive. This was intentional — the mechanic needs to
exist before it can be given meaning. Combat strength pooling and resource sharing
came in subsequent tickets.

**v5 result:** Collaborate at 3.2%, attack dropped from 6.8% to 2.4%. Agents were
already learning that being in a group meant not attacking group members — that alone
shifted combat behavior.

---

## Entry 6 — Group Combat Mechanics: Flanking + Cohesion (v5→v6)

**Discussion:** Groups existed but provided no mechanical combat advantage. A solo
high-strength agent was just as good in combat as a coordinated group. The question
was how to make coordination genuinely useful.

**Options considered:**
- A: Flanking attack bonus (damage × 1.2 per ally adjacent to target)
- B: Damage split (incoming damage split across shielding allies)
- C: Both

**Decision:** A only. Flanking bonus, no damage split. The damage split was removed
after initial implementation because it made being in a group punishing when defending
— group members would share damage even when they didn't want to fight.

**Formula:** `damage *= (1 + 0.20 × N_flanking_allies)`, clamped to max 0.5.

**Also added:** Group cohesion reward — `+0.02` per live group member within 3 tiles
(Chebyshev distance). This continuous per-tick reward pulls agents toward each other
without scripting the movement.

**Coordinate system fix:** All combat proximity checks were switched from Manhattan
distance to Chebyshev (8-directional) to match the movement model. The game allows
diagonal movement, so diagonal attacks should be valid too. Consistency matters.

**v6 result:** Collaborate 4.6% (up), attack 1.7% (down from 2.4%). Group mechanics
shifting behavior but lifespan unchanged — groups forming but not yet providing clear
survival advantage.

---

## Entry 7 — Food Sharing with Reciprocity Memory (v7→v8)

**Feature:** Agents with enough food automatically share with hungry group members.
Not a manual action — a background mechanic triggered by hunger threshold.

**Design choice:** Made sharing probabilistic, not automatic. Base probability: 50%.
If the recipient has recently helped the sharer (within last 100 steps), probability
jumps to 85%. This is a simple model of reciprocal altruism.

**Why probabilistic?** Deterministic sharing removes the interesting tradeoff. With
50% base sharing, agents with more social connections (more potential recipients who
remember them) survive better — that's the emergent pressure.

**Implementation:** `_help_received[recipient][helper] = step_count`. This dictionary
resets every episode. Any "help" (food share, group coverage in combat) updates the
memory. Share probability checked against reciprocity window (100 steps).

**v8 result:** Lifespan 105.7 (new all-time best, +20 vs v7). Peak 115.0 at step 1.74M.
Attack rate near zero (0.3%). Eat rate jumped 22% → 32% — strongest signal that food
sharing was actually happening. Agents were eating food that came from group members.

**Interesting observation:** v7's collaborate rate (7.1%) was the highest before food
sharing. In v8 it dropped to 6.2% — possibly because food sharing provided a survival
fallback that reduced the urgency of group formation for combat defense alone.

---

## Entry 8 — Settlement Mechanics: Stash, Dense Maps, Foraging Outward (v8→v9)

**Observation:** Agents were surviving but not building. No one was using the stash.
The mechanic existed (deposit/withdraw) but deposit had a qi cost that made it
unattractive relative to keeping food in inventory.

**Decision:** Remove qi cost from deposit. Deposit now only requires food ≥ 1.

**Also added:** Three stash-oriented reward signals:
1. `REWARD_FORAGE_OUTWARD`: deposit bonus when agent traveled ≥ 5 tiles from stash
   before depositing. Rewards the "forage outward, bring back" behavior.
2. `REWARD_GROUP_WITHDRAW_BONUS`: for stash owner when a group member withdraws from it.
3. `REWARD_STASH_PROXIMITY`: per-tick bonus when hungry and near own stash.

**Dense-patch map:** Added clustered food spawning (corner patches, 60% fill)
separated by sparse background (2% fill). First settlement eval showed stash
completely unused on both map types, but lifespan jumped +65% on dense-patch.

**Key finding:** Map pressure helped survival and group persistence but did NOT
trigger stash usage. The mechanic needed explicit reward signals to be learned.
Confirms that some behaviors need scaffolding before they can emerge.

---

## Entry 9 — The Anti-Cooperative Stash Reward Bug (v9)

**Problem:** v9 showed a behavioral regression. DEPOSIT collapsed from 3.1% → 0.5%.
WALK_AWAY spiked from 4.8% → 11.2%. Group cohesion worse than v8.

**Root cause:** `REWARD_STASH_PROXIMITY = 0.01/tick` when hungry and near own stash.
On the default map, each agent deposits at different tiles, so stashes are dispersed.
The proximity reward pulled each agent toward *their individual stash location* — away
from the group. REST rate jumped (6%→13%) as agents sat near their stash instead of
foraging.

**The reward was correct in intent but anti-cooperative in effect.** It said "be near
food when hungry" but created individual anchoring rather than group stashing.

**Fix:** Set `REWARD_STASH_PROXIMITY = 0.0`. Disabled entirely.

**Broader lesson:** Rewards that reference individual ownership create selfish
anchoring. Rewards that reference group resources create cooperative pressure. The
distinction matters even when the underlying mechanic is the same stash.

---

## Entry 10 — Hazard Unification: Formula Change with Real Consequences (v9→v10)

**Hazard unification:** Multiple code paths handled poison damage differently.
Unified into `Agent.apply_hazard(resistance_stat, raw_damage)`.

**Important formula fix:** The old eat-poison formula was subtractive:
`damage = potency - resistance`. This meant immunity at `resistance ≥ 0.4` — far too
easy, since starting resistance was already 0.05-0.30. Changed to multiplicative:
`damage = potency × (1 - resistance)`. Now true immunity requires resistance = 1.0.

**Why this matters:** With the old formula, agents quickly became immune to poison
just by existing near it. There was no reason to build resistance past ~0.4. With the
new formula, every point of resistance matters and full immunity is a genuine
achievement. This makes the cultivation arc real.

---

## Entry 11 — The TRAIN Action + Qi Influence Field (v10→v11)

**The TRAIN action:** Added explicit `TRAIN` action that grows `strength`. Rate
interpolates between `TRAIN_RATE_ANYWHERE = 0.002` and `TRAIN_RATE_QI_TILE = 0.01`
based on qi field influence — a 5× multiplier for training near qi sources.

**Qi influence field:** Each qi source tile gets a random value [20,100]. Influence
radiates via `max(0, val - 10 × chebyshev_dist)`. Multiple sources stack additively,
creating "qi nexus" zones at overlaps. This means the best training spots are where
qi sources overlap — a geographic incentive without hardcoding.

**Why a continuous field instead of binary on/off qi tile?**
Binary: "on qi tile → fast training" creates a single hotspot everyone fights over.
Continuous field: gradient that pulls agents toward better training spots without
forcing them all to the exact same tile. More natural clustering behavior.

**Power score added as eval metric:**
`power = 0.4×strength + 0.3×qi_drain_resistance + 0.2×poison_resistance + 0.1×flame_resistance`.
Not used in reward — just logged. Gives a single number for "cultivation level."

---

## Entry 12 — Sects, Aging, and Reproduction (v11)

**Sects returned** — correctly this time. After observing attack rates drop from 6.8%
(v5) to 0.3% (v8), group identity was clearly emerging. The vocabulary was now earned.

**Design:** Three sects with distinct home regions (y-bands on the 30×30 grid). No
forced trait differences — the environmental pressure of having a home zone creates
differentiation over time. The sects are a *label* for what agents discover, not a
prescription for what they should do.

**Aging:** Natural death at `max_age = 3000` world ticks (~600 env steps). At current
lifespan of ~105 steps, age death is rare. But it's the prerequisite for reproduction.

**Reproduction:** When an agent dies of old age and ≥2 survivors exist, a new agent
spawns inheriting traits from two parents via `inherit_value(a, b, rng, sigma=0.05)`:
midpoint + Gaussian noise, clamped [0,1].

**Lamarckian resistance inheritance:** Poison resistance inherits at the parent's
*acquired* value, not the spawn range. An agent that survived many poison tiles passes
that immunity to their children. This violates strict Darwinian evolution but matches
the "cultivation arc" narrative — a master who survived poison teaches their student
to be resistant. It also creates faster convergence toward useful resistances.

---

## Entry 13 — The V11 Oscillation Problem and LR Decay (v11→v12)

**Problem:** v11's lifespan spiked and collapsed repeatedly:
- Step 38%: 115.0 → collapsed
- Step 91%: 144.0 → collapsed
- Step 94%: 151.2 (all-time peak) → collapsed to 72.8
- Step 97%: 105.0 → collapsed

**Root cause:** Fixed `LR = 3e-4` and `clip = 0.2` — both too large for late-stage
fine-tuning. When the policy finds a good strategy, a large update step overshoots it.

**This is a classic RL pathology:** The same mechanism that drives improvement
(large gradient steps) destroys fine-grained strategies at late training. The best
checkpoint was at 94% completion — the policy had the strategy but hadn't destroyed
it yet.

**Fix:** Linear decay schedules. LR decays 1e-4 → 1e-5 (10× reduction). Clip range
decays 0.15 → 0.05. The policy explores aggressively early but makes only small
updates at the end, locking in good strategies instead of oscillating past them.

**v12 result:** Stable lifespan 83.7 with no collapse. Never hit v11's peak of 151
but also never crashed. The LR decay traded peak performance for consistency.

---

## Entry 14 — Health System Overhaul (v12→v13)

**Observation:** Watching the v11 replay, agent_0 had 24 health and kept gathering
without ever eating. Turned out food only reduced hunger — it never restored health.
REST was the only healing mechanic, and resting when starving was counterproductive.

**Root insight:** The agent wasn't broken — the mechanics were. When you watch a
replay and the agent looks stupid, check the mechanics first.

**Overhaul:**
- Food now restores 0.05 health per eat (nutrition as medicine)
- Starvation is gradual — health drains escalating above hunger=0.80, not instant
  death at 1.0
- New `hunger_resistance` trait [0.1–0.5] — heritable, mitigates drain and strength
  penalty when starving
- `effective_strength` = strength penalised by hunger — hungry fighters hit weaker

**Tradeoff:** Delayed death gives agents more time to find food, but the penalty
signal is weaker. The agent needs to learn *before* hitting the threshold, not after.

---

## Entry 15 — The Eat-Farming Problem (v12 analysis)

**Problem:** v12 EAT rate was 34% — the highest single action. Despite the health
overhaul giving more survival time, lifespan barely changed (83.7 vs v11's 85.6).

**Root cause:**
- At lifespan=83.7, EAT=34%: agents ate **28.5 items** per episode
- GATHER=15.3%: agents gathered only **12.8 items** per episode
- Eating 2.2× more than gathering → inventory depleted → starvation anyway

The `REWARD_HEALTH_RECOVERY_SCALE = 0.30` fired every time health improved — even
when health was already near 1.0. Agents "farmed" small recovery rewards by eating
constantly.

**Key lesson:** If an action has a positive reward, the agent will do it as often as
possible regardless of context. Rewards need context gates, not just magnitude tuning.

**Fix:** Gate: health recovery reward only fires when `health < 0.70`. No reward for
eating when already healthy. Also raised gather reward (0.10 → 0.15) so gathering
outweighs the eat-farming incentive.

---

## Entry 16 — Emergent Behavior vs. Spoon-Feeding (v13 design)

**Desired end-state:** Agents collaborate, form parties, take turns filling a shared
stash, train near it. The question: reward these strategies explicitly, or let them
emerge from mechanics?

**Key distinction:**
- **Reward the strategy** = spoon-feeding. Agent learns the shortcut without
  understanding why. Brittle — breaks on novel maps.
- **Fix the mechanics** = emergence. If training near a stash is genuinely the best
  survival strategy, the agent discovers it under mortality pressure.

**Decisions:**
- ❌ No explicit "train near stash" reward
- ❌ No explicit "two agents training together" reward
- ❌ Remove group cohesion reward — let survival pressure create it
- ✅ Training drains hunger 1.5× faster — stash proximity becomes mechanically optimal
- ✅ Inventory cap at 3 items — forces stash use, creates natural division of labor
- ✅ Fold resistances into defense_power in combat — cultivation arc has payoff
- ✅ Reward resistance growth delta — hazard traversal worth doing proactively

**On group behavior in RL:** Purely emergent group behavior is very hard — the
literature (QMIX, MAPPO) all uses explicit coordination mechanisms. The goal is to
put the scaffolding in the *mechanics*, not the *reward*. A group sharing a stash
is mechanically better off than a solo agent — that's the signal.

---

## Entry 17 — Clustered Food Maps as Collaboration Pressure (v13 design)

**Insight:** Uniform food distribution means every agent has roughly equal access —
no geographic pressure to collaborate. Clustered maps create resource scarcity zones
that make group defense mechanically optimal.

**Why this is RL-idiomatic:** Called procedural environment generation. OpenAI used
this in the hide-and-seek paper where agents invented tool use — the environmental
complexity created the evolutionary pressure, not the reward. Key: use as a
*distribution* (40% clustered / 60% uniform), not a fixed map.

**Fixed map vs. distribution:**
- Fixed map → agents memorize tile positions. Strategy is map-specific.
- Distribution of maps → agents learn the *principle* of cluster defense and cooperation.

**Why clusters push collaboration:**
- A group holding a food cluster survives longer than solo agents in deserts
- Stash near a cluster becomes genuinely valuable (surplus to deposit, safe to train)
- One gatherer + one trainer near the same cluster is more efficient than two agents doing both poorly
- Splitting up means someone crosses a desert hungry — real survival cost

**Cluster size tradeoff:** Too small = competition over scarce food, agents fight not
cooperate. Too large = no scarcity, no pressure. Target: feeds 3-4 agents comfortably,
creating cooperative surplus rather than zero-sum competition.

**Design note:** Cluster centers placed near qi tiles creates geography that naturally
co-locates food, qi training, and stash placement — all ingredients for the desired
emergent behavior, without explicitly rewarding any of it.

---

## Running Decisions Log

| Decision | Rationale |
|----------|-----------|
| Traits in [0,1] only | Display scaling only — no 0-100 scale in code |
| Seeded RNG everywhere | Determinism is sacred — same seed = byte-identical replay |
| YAML-driven resources | Adding resource type = zero code changes |
| No named abstractions until behaviors observed | Code follows behavior, not the other way |
| Gradual starvation (not instant death) | Gives policy time to learn avoidance before penalty |
| Gate health recovery at health < 0.70 | Prevents eat-farming while keeping intent |
| Inventory cap 3 items | Forces stash dependency, creates division-of-labor pressure |
| No group cohesion reward | Emergence over spoon-feeding |
| Resistances fold into defense_power | Makes cultivation arc meaningful in combat |
| Clustered maps at 40% of episodes | Environmental pressure > reward pressure for collaboration |
| Training drains hunger 1.5× faster | Stash proximity becomes mechanically optimal |
| Lamarckian resistance inheritance | Faster convergence + narrative fit |
| LR/clip linear decay schedule | Prevents late-training policy oscillation |
| Hazard formula multiplicative not subtractive | True immunity requires resistance=1.0 |
| REWARD_STASH_PROXIMITY = 0 | Individual stash anchoring is anti-cooperative |
| Chebyshev over Manhattan distance | Consistent with 8-directional movement |
| Damage split removed from combat | Punished group members for being nearby |

---

## Training History Summary

| Version | Key Change | Lifespan | Notable |
|---------|-----------|----------|---------|
| MLP v1-3 | Baseline survival | ~60-70 | Initial mechanics |
| LSTM v1-2 | LSTM memory, combat | ~70-90 | Gather bug introduced in v2 |
| v3 | Reward fix (gather vs eat) | 55.2 | Gather recovered to 14.8% |
| v4 | Strength reward, unified train script | 73.9 | Defend doubled to 7.3% |
| v5 | Sociability, COLLABORATE/WALK_AWAY | 94.1 | Attack dropped to 2.4% |
| v6 | Flanking bonus, cohesion reward | 93.7 | Collaborate 4.6% |
| v7 | Coordinated attack reward | 85.5 | Cut short at 600k (config bug) |
| v8 | Food sharing + reciprocity | 105.7 | All-time best; attack near zero 0.3% |
| v9 | Shared stash, dense-patch map | ~54-65 | WALK_AWAY spiked (stash proximity bug) |
| v10 | Stash fix, hazard unify, TRAIN action | ~85 | Power score added |
| v11 | Qi field, reproduction, aging, sects | 85.6 | Oscillation; peak 151.2 at 94% |
| v12 | Health overhaul, LR decay | 83.7 | Stable but eat-farming emerged (EAT 34%) |
| v13 | Eat fix, mechanics overhaul, clusters | TBD | In progress |
