# MurimSim Design Journal

A running log of interesting discussions, decisions, and tradeoffs made during
development. Intended as raw material for a future blog post.

---

## Entry 1 — Two-Lane Development

**Discussion:** The project was trying to do too much in one place — building
features, running training, and learning RL theory all in the same session. Token
burn was high and progress was slow.

**Decision:** Split into two lanes:
- **Fast Lane** (Copilot CLI): coding sessions and training runs only. Every
  session ships something.
- **Slow Lane** (ChatGPT / Gemini): RL theory, coaching, long-document review.

**Tradeoff:** Coaching slows feature velocity. Separating the two means you
sometimes implement before fully understanding — but the simulation itself
becomes the teacher. Watching a replay often teaches more than an hour of
theory.

**Interesting note:** The "slow lane" insight was that the simulation is a
learning instrument for the *developer*, not just a test bed. The emergent
behaviors you observe directly teach you what the reward function is actually
optimizing.

---

## Entry 2 — Design Principle: Code Follows Behavior

**Discussion:** Early in the project, "sects" (named martial arts factions) were
built as a first-class abstraction — SectConfig, SectRegistry, per-sect trait
profiles. This was torn out after v3 because it was premature.

**Decision:** No named abstractions until the behavior they represent is
observed. If agents aren't forming durable alliances yet, there's no point
coding an "alliance" system.

**The rule that emerged:** *Code follows behavior, not the other way around.*

**Later pivot:** Sects were added back in Phase 6a — but this time as a
spawning/territory mechanic only (home region bands), not as a behavioral
scaffold. The behavioral differentiation (iron_fang fights more, jade_lotus
heals more) is supposed to emerge from the combination of home region pressure
+ inter-sect combat rewards, not from hardcoded trait presets.

**Tradeoff:** This discipline creates frustrating gaps ("there's clearly sect
behavior happening but we have no code to track it"). The payoff is that when
you do add the abstraction, it's describing something real.

---

## Entry 3 — The Gather/Eat Reward Interaction Bug (v2 → v3)

**Problem observed:** After LSTM v2, gather rate had dropped from ~20% to ~12%.
Agents were eating food off the ground instead of gathering into inventory first.

**Root cause:** Two reward signals were fighting each other:
- `PENALTY_STARVATION_APPROACH = -0.08` fired when `hunger > 0.7`, creating
  urgency to fix hunger *immediately*
- `REWARD_HUNGER_RELIEF_SCALE = 0.20` rewarded eating
- Eating from the ground is faster than gather→eat

So the agent learned: when hungry, eat directly from the tile. Gather became
suboptimal even though it's the survival-rational strategy.

**Fix:** Removed `PENALTY_STARVATION_APPROACH`. Replaced with
`PENALTY_EMPTY_INV_FOOD_NEARBY = -0.06`: only fires when inventory is *empty*
AND food is nearby. This incentivizes gathering proactively without creating
urgency to eat immediately.

**Key lesson:** When two reward signals both fire on the same event (being
hungry), the agent finds the path that satisfies both — even if that path
undermines the intended behavior. The fix is to decouple the trigger conditions.

---

## Entry 4 — Reward Signal Design: Potential-Based vs. Direct

**Discussion (Phase 5):** As more mechanics were added, several reward choices
came up. The key distinction:

- **Direct reward** (`+X for doing action Y`): Simple but dangerous. Agent will
  do Y as much as possible regardless of context.
- **Potential-based reward** (`+X × Δ(state_value)`): Rewards *improvement*, not
  the action itself. Ng et al. (1999) proved this never changes the optimal
  policy.

**Examples in the codebase:**
- `REWARD_TRAIN_STRENGTH_SCALE × Δstrength` — potential-based. Agent only gets
  the reward when it actually gets stronger.
- `REWARD_GROUP_FORMATION = 0.05` — direct. Fires once when group forms, which
  is fine since it only fires once.
- `REWARD_STASH_PROXIMITY = 0.01/tick` — direct per-tick. This one caused
  serious problems (see Entry 6).

**Takeaway:** Per-tick direct rewards are the most dangerous category. They
create constant gradient pressure that overwhelms other signals. Use them
sparingly and gate them tightly.

---

## Entry 5 — Group Dynamics and Emergence

**Phase 5 goal:** Get agents to form and maintain groups. The mechanisms:
- `sociability` trait [0,1] — determines willingness to collaborate
- `COLLABORATE` action (Discrete(12)): forms a group between focal + nearest
  willing agent
- `WALK_AWAY` action (Discrete(13)): one step away from group
- Group cohesion reward: `+0.02/tick` per live group member within Chebyshev 3

**Results from v5-v8:** Collaborate rate grew from 0% → 3.2% → 4.6% → 6.2%.
Attack rate dropped from 6.8% → 2.4% → 1.7% → 0.3%. Groups are forming and
reducing combat.

**Food sharing with reciprocity (v8 breakthrough):**
- When a group member has `hunger > 0.85`, other members have 50% chance to
  share food — raised to 85% if recipient previously helped the sharer within
  100 steps
- This created a social memory mechanic without LLM involvement
- v8 lifespan: 105.7 (peak 115.0) — new all-time record at the time
- Eat rate jumped from 22% → 32%, confirming food sharing was active

**Interesting observation:** The food sharing mechanic is *not* an agent action
— it runs automatically in the environment step loop based on proximity and
reciprocity. The agent doesn't choose to share. This is closer to a social
physics rule than an RL behavior. Whether this is the right design is debatable.

---

## Entry 6 — The Stash Proximity Reward Trap (v9 regression)

**Problem:** Added `REWARD_STASH_PROXIMITY = 0.01/tick` — fires when the agent
is hungry (>0.5) and within 3 tiles of their own stash. Intent: reward agents
for staying near their food supply while training.

**Result:** WALK_AWAY spiked from 4.8% → 11.2%. DEPOSIT collapsed from
3.1% → 0.5%. Agents started walking *away from groups* to sit next to their
individual stash locations.

**Root cause:** Each agent deposited at a different tile, so stashes were
dispersed across the map. The proximity reward created a centrifugal force —
each agent was pulled to a different spot. This directly counteracted the
cohesion reward.

**Fix:** Set `REWARD_STASH_PROXIMITY = 0.0` (disabled). The lesson: proximity
rewards only help if all agents share the same target location. With individual
stashes, it's anti-cooperative by construction.

**Broader principle:** A reward that helps *one agent* can hurt *the group*. In
multi-agent settings, every reward signal needs to be evaluated for its effect
on group dynamics, not just individual behavior.

---

## Entry 7 — The Stash Is Unused (Eval Finding)

**Problem discovered during eval:** After implementing shared-stash mechanics
(free deposit, group withdraw, foraging-outward bonus), ran a frozen v8 model
eval and found: **stash fill rate = 0.000 on both default and dense-patch maps**.
Agents never deposit.

**Why this mattered:** The stash system was designed to enable the "deposit food,
train near stash" behavior that represents the desired emergent collaboration.
If no one deposits, the whole system is inert.

**Cause:** v8 was trained before deposit was free (it previously cost qi). The
policy learned "DEPOSIT = bad" (costs a resource for no apparent benefit) and
never updated because DEPOSIT was never rewarded.

**Resolution:** Need training on the new free-deposit system. Dense-patch map
helps because clustered food creates surplus pressure, making deposit the
rational strategy.

**Lesson:** Changing a mechanic mid-training doesn't fix existing behavior. The
policy has already learned to avoid the action. You need to retrain from a
checkpoint (v9 warm-started from v8) and provide enough reward gradient for the
new behavior to develop.

---

## Entry 8 — Hazard System Unification

**Problem:** Two separate code paths handled hazard damage:
- `apply_traversal_effects()` for walking into poison/flame tiles
- Poison eating in `eat()` using a different formula (`potency - resistance`)

The traversal code used multiplicative damage `(potency × (1 - resistance))`
but the eat code used subtractive (`potency - resistance`). The docstring for
both said "immunity at resistance=1.0" but only multiplicative achieves that.

**Fix:** Unified both into `Agent.apply_hazard(resistance_stat, raw_damage)`
— single function, multiplicative formula, shared resistance/intake logic. Both
traversal and eating now call the same function.

**Side effect:** Hazard IDs were hardcoded as `("poison", "flame")` in two
places. Changed to derive from `world.hazard_ids` — any resource in the YAML
with `effect: negative` is automatically a hazard. This is the data-driven
design principle: adding a new hazard type = YAML change only, zero code changes.

---

## Entry 9 — Qi Cultivation and Training System

**Design:** Added a TRAIN action (Discrete(14)) that grows `strength` toward 1.0
at a rate that depends on the local qi influence field.

**Qi influence field mechanic:**
- Each qi tile gets a random value [20, 100] on world generation
- Influence radiates outward: `max(0, val - 10 × chebyshev_distance)`
- Multiple sources **stack** (additive) — creates qi nexus zones near overlapping
  sources
- Field is normalized [0, 1] and stored per-cell

**Training rate:** Interpolates linearly between `TRAIN_RATE_ANYWHERE = 0.002`
and `TRAIN_RATE_QI_TILE = 0.01` based on qi_field_value at the agent's position.
Result: **5× faster training at a qi nexus** vs. training in the open.

**Desired emergent behavior:** Agents should discover that training near a qi
nexus + a food source (stash) is the optimal cultivation strategy. This creates
geographic "power leveling zones" that agents should compete over and defend.

**No explicit reward for location choice** — the strength delta reward
(`REWARD_TRAIN_STRENGTH_SCALE = 2.0`) is the same regardless of where you train.
The location advantage comes through the *rate* of strength gain, not through
a location-specific reward.

---

## Entry 10 — Health System Overhaul (v12)

**Problem observed in v11 replay:** Agent_0 had 24 health but kept gathering
without ever eating. Health wasn't recovering. Turned out food only reduced
hunger — REST was the only healing mechanic, and resting when starving
counteracted hunger relief.

**Key insight:** *When you watch a replay and the agent looks stupid, check the
mechanics first.* The agent had learned the optimal strategy for the actual
system, which had no food-health link.

**Overhaul decisions:**
- Food now restores `FOOD_HEALTH_RESTORE = 0.05` health per eat
- Starvation is now gradual: health drains `(hunger - 0.80) × 0.10 × (1 - hunger_resistance)` per tick above the threshold
- New `hunger_resistance` trait [0.1, 0.5]: mitigates both health drain and
  strength penalty. Heritable.
- `effective_strength` property: `strength × (1 - excess_hunger × (1 - hunger_resistance))` — hungry fighters hit weaker

**Tradeoff:** Delayed death gives agents more time to find food, which is good.
But a weaker immediate penalty means the agent needs to learn avoidance *before*
the threshold, not react after.

---

## Entry 11 — The Eat-Farming Problem (v12 diagnosis)

**Problem observed:** v12 trained with 34% EAT rate — highest single action.
At lifespan ~84 steps, agents were eating 28.5 items but only gathering 12.8 —
eating 2.2× more than they gathered. Lifespan barely improved despite the health
overhaul giving more survival time.

**Root cause:** `REWARD_HEALTH_RECOVERY_SCALE = 0.30` fired every time health
improved from eating — *even when health was already near 1.0*. The agent
discovered it could farm small health recovery rewards by eating constantly,
depleting inventory, then starving anyway.

**Key lesson:** *Rewards need context gates, not just magnitude tuning.* The
intent was "eat when hurt" but the agent learned "eat always" because the reward
was unbounded.

**Fix:** Gate the health recovery reward: only fires when `health_prev < 0.70`.
Also raised `REWARD_FOOD_GATHERED_SCALE` (0.10 → 0.15) so gathering outweighs
the eat incentive.

---

## Entry 12 — Emergent Behavior vs. Spoon-Feeding

**Core question:** Should we reward "train near stash" and "work together"
directly, or let it emerge?

**Key distinction made:**
- **Reward the strategy** = spoon-feeding. Agent learns the shortcut (be near
  stash → reward) without understanding why. Brittle — breaks on novel maps.
- **Fix the mechanics** = emergence. If training near a stash is genuinely the
  best survival strategy, the agent discovers it under mortality pressure.

**Decisions:**
- ❌ No explicit "train near stash" reward
- ❌ No explicit "two agents training together" reward
- ❌ Remove group cohesion reward — let survival pressure create it
- ✅ Make TRAIN drain hunger faster (1.5×) — creates mechanical need for food
  access while training
- ✅ Inventory cap at 3 items — forces stash trips, creates natural division of
  labor (one gathers, one trains, stash is the bridge)

**On group behavior:** Purely emergent group behavior is hard without any
scaffolding — the RL literature (QMIX, MAPPO) all use explicit coordination
mechanisms. The goal is to put the scaffolding in the *mechanics*, not the
*reward*.

---

## Entry 13 — Resistances as Defensive Power

**Problem:** Resistances (poison, flame, qi_drain) were disconnected from PvP
combat. A cultivator with maxed resistances fought identically to a fresh agent.

**Decision:** Fold resistances into `defense_power` concept for combat:
```
attack_power  = effective_strength
defense_power = effective_strength × 0.5 + avg(resistances) × 0.5
```

**Also:** Reward resistance *growth delta* directly. Currently only strength
gains are rewarded; agents avoid hazards rather than building immunity. Adding
a `REWARD_RESISTANCE_GAIN_SCALE` turns hazard traversal into a deliberate
cultivation choice.

**Design intention:** This creates a "cultivation arc" — an agent that spent
time near flame tiles becomes genuinely hard to kill. The investment in
resistance-building has a payoff in combat. This is the murim fiction working
as game design.

---

## Entry 14 — Clustered Food Maps and Environmental Pressure

**Question:** Will training on clustered maps (food in 2-4 Gaussian blobs
separated by deserts) spoil the policy or help collaboration?

**Answer: Completely RL-idiomatic** — called procedural environment generation.
OpenAI used this in the hide-and-seek paper where agents invented tool use
without any tool-related reward. The key is using a *distribution* (40% clustered
/ 60% uniform), not a fixed map. Fixed maps → agents memorize positions.
Distribution → agents learn strategy.

**Why clusters help collaboration emerge:**
- A group holding a food cluster survives longer than solo agents in deserts
- Stash near a cluster is genuinely valuable — deposit surplus, train nearby
- Desert traversal is costly — splitting up has real survival cost
- One agent gathering + one training near the same cluster is naturally more
  efficient than two agents doing both poorly

**Cluster size is critical:** Too small = zero-sum competition. Too large = no
scarcity. Target: a cluster that comfortably feeds 3-4 agents to create
cooperative surplus.

**Design note:** Placing cluster centers near qi tiles creates geography that
naturally co-locates food, qi training, and stash placement — all the ingredients
for the desired emergent behavior, without explicitly rewarding any of it.

---

## Running Decisions Log

| Decision | Rationale |
|----------|-----------|
| All traits in [0,1] | Display scaling only — no 0-100 scale in code |
| Seeded RNG everywhere | Determinism is sacred — same seed = byte-identical replay |
| YAML-driven resources | Adding a new resource type = zero code changes |
| Code follows behavior | No named abstractions until behaviors appear |
| Remove `PENALTY_STARVATION_APPROACH` | Was competing with inventory reward, causing direct-eat |
| Potential-based rewards for growth | Never changes optimal policy (Ng et al. 1999) |
| `REWARD_STASH_PROXIMITY = 0.0` | Anti-cooperative on default map (dispersed stashes) |
| Free deposit (no qi cost) | Qi cost was blocking stash usage entirely |
| Gradual starvation (not instant death) | Gives policy time to learn avoidance before penalty |
| Gate health recovery at health < 0.70 | Prevents eat-farming while keeping reward intent |
| Inventory cap 3 items | Forces stash dependency, creates division-of-labor pressure |
| No group cohesion reward | Emergence over spoon-feeding — mechanics create it |
| No "train near stash" reward | Same principle — mechanics (hunger drain during TRAIN) create it |
| Resistances fold into defense_power | Makes cultivation arc meaningful in combat |
| Clustered maps at 40% of episodes | Environmental pressure > reward pressure for collaboration |
| TRAIN drains hunger 1.5× | Stash proximity becomes mechanically optimal, not rewarded |
| Chebyshev distance for combat/cohesion | Consistent with 8-directional movement |
| Multiplicative hazard formula | Immunity only at resistance=1.0, matches docstring intent |
| `apply_hazard()` unified function | Single code path for all hazard damage |
| Qi influence field stacks additively | Creates nexus zones worth competing over |
| `hunger_resistance` trait | Heritable — creates evolutionary pressure for resilience |

---

## Training History Reference

| Version | Steps | Lifespan | Key Behavior | Notes |
|---------|-------|----------|--------------|-------|
| MLP v1 | 500k | ~30 | Basic survival | First working model |
| MLP v2 | 1M | ~45 | Avoids hazards | Resistance reward added |
| MLP v3 | 1M | ~52 | Less eat-farming | Gather/eat reward fix |
| LSTM v1 | 1M | ~60 | Memory-based gathering | First LSTM |
| LSTM v2 | 1.5M | ~85 | Combat basics | CombatEnv introduced |
| LSTM v3 | 1.5M | 55.2 | Gather fixed | Reward decoupling |
| LSTM v4 | 2M | 73.9 | Defend 7.3% | Strength as priority 2 |
| LSTM v5 | 2M | 94.1 | Collaborate 3.2% | Sociability + COLLABORATE action |
| LSTM v6 | 2M | 93.7 | Collaborate 4.6% | Group combat (flanking), cohesion |
| LSTM v7 | 600k* | 85.5 | Collaborate 7.1% | *Config bug, cut short |
| LSTM v8 | 2M | 105.7 | Eat 31.9%, attack 0.3% | Food sharing + reciprocity |
| LSTM v9 | 2M | ~54** | WALK_AWAY 11.2% | Stash proximity reward regression |
| LSTM v10 | — | — | — | Killed before completing |
| LSTM v11 | 2M | 85.6 | TRAIN 7.5% | Qi field, TRAIN action, aging |
| LSTM v12 | 1M | 83.7 | EAT 34.1% | Health overhaul — eat-farming |
| LSTM v13 | — | — | — | Killed at start — tuning in progress |

*v7 ran only 600k due to config bug (training.yaml had 600k instead of 2M)
**v9 lifespan appears lower because metric scales with n_agents; real performance ~same as v8
