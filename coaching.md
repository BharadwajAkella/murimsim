# MurimSim RL Coaching Notes

> A running log of RL concepts covered in this project.
> Written for someone who has watched ML videos but never actually trained anything.
> Examples refer to MurimSim so you can always ground the theory in something you've seen run.

---

## Stage 1 & 2 — How Does an Agent Actually Learn?

### The Core Idea: Rewards

An agent learns by receiving a **reward signal** after taking actions. That's it. No one tells it *why* something was good or bad — it just gets a number, and over millions of attempts, it figures out what tends to produce high numbers.

In MurimSim, rewards look like:
- `+0.1` for moving toward food
- `-0.02` for stepping on poison
- `-0.01` just for being alive (existence cost — encourages efficiency)
- `+big number` for surviving a long time

The agent doesn't know what "poison" means. It just notices that certain sequences of actions reliably end in negative numbers and learns to avoid them.

---

### The Credit Assignment Problem

Here's the hard part. Say your agent survives for 300 ticks and then dies. Which of those 300 actions caused the death? Was it tick 299 (walked into poison)? Or tick 47 (chose to go left instead of right, ending up in a bad region of the map)?

This is called the **credit assignment problem** — assigning blame (or credit) to past actions based on future outcomes.

The naive fix: **sum up all the rewards from every step that led to the outcome**, and give each step a share of that total.

```
G_t = r_t + r_(t+1) + r_(t+2) + ... + r_T
```

This is called the **return** (G). It says: "the value of being in this state and taking this action is the sum of everything that happened afterward."

**MurimSim intuition:** If Zhang Wei trained, ate, trained, then walked into poison and died — the "train" actions at ticks 1 and 3 still get penalized because they're part of a trajectory that ended badly. That feels unfair, but with enough rollouts the signal averages out correctly.

---

### Making It Smarter: The Advantage Function

Pure returns have a problem: some states are just *inherently* good or bad regardless of what action you take. If your agent is healthy with full food and no enemies nearby, it's going to get high returns no matter what it does. The return doesn't tell you if the *action* was smart — just if the *situation* was good.

The fix is the **advantage function**:

```
A(s, a) = Q(s, a) - V(s)
```

Where:
- `Q(s, a)` = "how much return do I get if I'm in state `s` and take action `a`?"
- `V(s)` = "how much return do I *expect* on average from state `s`, regardless of action?"
- `A(s, a)` = "was this action **better or worse than expected**?"

**MurimSim intuition:** If Zhang Wei is at full health with food nearby, V(s) is already high — any decent action will do fine. If he *still* finds a clever path to survive 50 extra ticks, the advantage of that specific action is positive. If he wastes time and dies soon despite the favorable state, the advantage is negative. The advantage **zeroes out the luck** and isolates the quality of the decision.

---

### PPO: How the Policy Actually Updates

PPO (Proximal Policy Optimization) is the algorithm we used to train the monks in MurimSim (starting from the limbic system in `murimsim/rl/`).

It trains **two networks simultaneously**:

```
Observation → Shared layers → Actor  (outputs: probabilities over actions)
                           → Critic (outputs: expected future return V(s))
```

**The Critic** learns to predict V(s) — "given what I see right now, how much reward will I accumulate?" It's essentially a quality estimator for situations.

**The Actor** uses the Critic's predictions to compute advantages, then nudges action probabilities:
- Actions with positive advantage → do more of this
- Actions with negative advantage → do less of this

**The "Proximal" part** is the key innovation of PPO: it doesn't let the policy update too drastically in one step. If you change the policy too aggressively, you can accidentally destroy behavior that was already working well. PPO clips the update ratio to stay within a small range — "improve, but not so fast you forget everything."

**MurimSim intuition:** After 300k training steps, our monk had learned "move toward food when hungry." A big unclipped update might overwrite that with "always move north" if a few lucky north-movement episodes happened to give high reward. PPO's clipping keeps the good behaviors stable while slowly improving the rest.

---

## Stage 3 — Is This Really an MDP? (Spoiler: Not Quite)

### What Is an MDP?

A **Markov Decision Process** is the mathematical framework that RL is built on. It has a crucial assumption called the **Markov property**:

> "The future depends only on the present state — not on the history of how you got here."

Formally: `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, ..., s_t, a_t)`

In plain English: if you give me the full current state of the world, the past is irrelevant.

**MurimSim intuition:** If you gave the agent a perfect snapshot of the entire 30×30 grid — every resource tile, every agent, their exact health and positions — then yes, the Markov property holds. The agent doesn't need to know *how* those resources got there or *why* that enemy agent is at that position.

---

### Our Problem: Partial Observability

In MurimSim, each agent only sees a **5×5 window** around itself. It cannot see:
- What's happening in the rest of the 30×30 grid
- Where enemies moved two ticks ago
- Whether the poison zone to the north is shrinking or growing

This makes it a **POMDP** (Partially Observable MDP). The agent's observation is not the full state — it's a *noisy, incomplete slice* of it.

This breaks the Markov property. "What should I do given what I currently see?" is not always answerable without knowing recent history. For example:

- At tick 100: agent sees empty tile to the north
- At tick 101: agent sees empty tile to the north (but it was poison at tick 99, now respawned)

The *same observation* at tick 101 could mean two different things depending on history. A purely Markov agent can't tell the difference.

---

### The Fix: Give the Agent Memory (LSTMs)

*(This discussion was started but not completed — picking up here.)*

One solution is to give the agent a **recurrent** architecture — specifically an LSTM (Long Short-Term Memory network).

Instead of just processing the current observation, an LSTM maintains a **hidden state** that it carries from tick to tick. Think of it as the agent's "working memory."

```
tick 1: obs_1 → LSTM → hidden_1 → action_1
tick 2: obs_2 → LSTM(hidden_1) → hidden_2 → action_2
tick 3: obs_3 → LSTM(hidden_2) → hidden_3 → action_3
```

The hidden state lets the agent implicitly remember things like:
- "There was poison two tiles north two ticks ago"
- "An enemy has been moving toward me consistently"
- "I've been in this area for 20 ticks and food keeps spawning here"

**MurimSim intuition:** Without memory, an agent that sees "empty tile" acts the same whether it just spawned there or has been exploring for 100 ticks. With an LSTM, the hidden state can encode "I've been here a while, food is not coming from this direction" — leading to exploration behavior that a memoryless agent can't produce.

**Why we haven't implemented it yet:** LSTMs complicate training significantly. The PPO rollout buffer needs to store hidden states. Truncated backpropagation through time (BPTT) is needed. It's worth doing — but we parked it to add more simulation features first (stashes, multi-resource types, M4 improvements).

---

## Stage 3.5 — The Great Evolutionary vs PPO Debate

### The Problem We Identified

Early in training, rewards in MurimSim are **sparse and delayed**. The agent might wander randomly for hundreds of ticks before accidentally finding food. PPO needs *contrast* — it needs to see some good trajectories and some bad trajectories to compute useful advantage estimates. If everything is equally bad (because the agent hasn't discovered any good behavior yet), PPO barely moves.

This is when we discussed using **evolutionary algorithms** for a "Phase 1" before PPO.

### Evolution First, PPO Second (The Hybrid Plan)

The idea from the GPT discussion:

**Phase 1 — Evolution (Explore):**
- Run many random policies in parallel
- Score entire episodes with a fitness function: `F = 0.6*survival + 0.3*food + 0.1*stat_gain`
- Keep the best policies ("elites"), mutate them, discard the rest
- Repeat until basic survival behavior emerges

**Phase 2 — PPO (Polish):**
- Take the best evolved policy weights as initialization
- Fine-tune with PPO using proper advantage estimation
- PPO is now working with an agent that already knows how to survive — the reward signal is dense enough to compute meaningful advantages

**The intuition:** Evolution is like throwing a hundred monks into the training ground and seeing which ones accidentally discover food. PPO is like taking the best monk and giving them a personal coach who explains exactly which decisions were wise.

### Why We Went PPO-First Anyway

After discussion, we decided **PPO first** for the limbic system because:

1. Our reward function isn't *purely* sparse — there's a living cost each tick, small bonuses for moving toward food, etc. Dense enough for PPO to get traction.
2. The fitness function for evolution (`survival + food + stats`) is actually close to what PPO's reward is already doing step-by-step. No need for two separate systems.
3. Evolution adds complexity (population management, mutation operators) that we'd have to maintain forever.

**The one-liner we agreed on:** Evolution is the right tool when the *only* signal is death. We have richer signals, so PPO works from the start.

The evolutionary approach remains valid for future experiments — particularly for the multi-sect specialization work described in `plan_for_rl_arch.md`.

---

---

## Stage 4 — Memory: Teaching an Agent to Remember

### The Goldfish Problem

Look at this from our actual replay (`combat_42.jsonl`, agent_3, ticks 20–40):

```
tick 20: [4,21] → move_n
tick 25: [2,21] → move_n   ← just came from here
tick 37: [4,21] → move_n   ← back where it started at tick 20
```

In 142 ticks alive, agent_3 visited only **39 unique positions** but revisited the same tiles **103 times**.
The agent has 1-tick memory. Every tick it wakes up with no idea where it's been.

**Why this is structural, not fixable with more training:**
The agent only sees a 5×5 window. Two situations can produce *identical* observations:
- Situation A: just arrived, food is 3 steps north (keep going)
- Situation B: been circling 20 ticks, food to the north is already eaten (turn around)

A memoryless policy physically cannot tell the difference — same input always produces same output.

---

### The Fix: A Learned Notepad (LSTM)

An LSTM gives the agent a **hidden state** `h` — a vector of floats carried tick to tick:

```
Standard (memoryless):   obs_t → [net] → action
LSTM (with memory):      obs_t + h_(t-1) → [net] → action + h_t → (saved for next tick)
```

The network learns *what to write* and *what to read* from the notepad. You don't program it.
- Poison zone locations → long memory (danger persists)
- Food tile locations → shorter memory (food respawns)
- Enemy positions → medium memory (fades as enemy moves away)

**The three LSTM gates (the notepad's rules):**

```python
f = sigmoid(W_f · [h_prev, obs])   # Forget gate: what to erase (0=erase, 1=keep)
i = sigmoid(W_i · [h_prev, obs])   # Input gate:  what new info to write
c̃ = tanh   (W_c · [h_prev, obs])   # Candidate:   the actual new content
o = sigmoid(W_o · [h_prev, obs])   # Output gate: what to read for this decision

cell = f * cell_prev + i * c̃       # Update whiteboard
h    = o * tanh(cell)               # Read from whiteboard → hidden state out
```

All W matrices are learned. The LSTM learns forget/write/read rules from survival outcomes alone.

---

### The Training Loop (Recurrent PPO)

Two distinct phases each iteration:

**Phase 1 — Collect (N ticks, no weight changes):**
```
Each tick: (obs_t, h_(t-1)) → LSTM → (action_probs, value_estimate, h_t)
Store: obs_t, h_(t-1), action_taken, reward_t, value_estimate_t
```

**Phase 2 — Compute advantages (math, no weight changes):**
```
Walk backward: G_t = reward_t + 0.99 * G_(t+1)
Advantage:     A_t = G_t - value_estimate_t
("was this step better or worse than expected?")
```

**Phase 3 — Update (K epochs, actual learning):**
```
For each 64-step chunk:
  1. Re-run LSTM from stored h at chunk start → new action_probs, new values
  2. Actor Δ:  push probability of actions with A_t > 0 up, A_t < 0 down (clipped ≤10%)
  3. Critic Δ: push value predictions toward actual returns G_t
  4. LSTM Δ:   gradients flow back through actor+critic → gates shift to remember
               things that lead to better decisions
```

**The key insight on LSTM updates:** The LSTM has no separate loss. Its weights update
because *forgetting useful information would have made the actor worse*, and that gradient
flows backward through time (Truncated BPTT — only last ~16 steps to avoid vanishing gradients).

**Why truncated?** Gradients shrink exponentially when backpropagating through many time steps.
For very long-range dependencies (>100 ticks), LSTMs struggle — Transformers handle this better
(a future topic).

---

### Exploration Baselines (Memoryless Agent)

Measured from `combat_42.jsonl` — these are our targets to beat with the LSTM agent:

| Agent | Lifespan | Unique tiles | Revisit rate | Action entropy |
|-------|----------|-------------|--------------|----------------|
| agent_0 | 800 | 227 | 0.60 | 1.28 |
| agent_8 | 663 | 235 | **0.51** | **1.69** |
| agent_3 | 782 | 188 | 0.63 | 1.64 |
| avg | ~777 | ~193 | 0.63 | 1.55 |

Note: agent_8 (lowest revisit rate, highest entropy) also shows best exploration.
An LSTM agent should shift the whole table: lower revisit rate, more unique tiles, longer survival.

---

## Stage 4 — Complete ✅

### LSTM vs MLP: Before/After Results

LSTM v1 trained for 600k steps using `RecurrentPPO` (sb3-contrib), warm-started from MLP v3 trunk (8/20 layers transferred, LSTM gates random init).

| Metric | MLP v3 (Before) | LSTM v1 (After) | Delta |
|---|---|---|---|
| Avg lifespan | 534.6 ticks | 441.4 ticks | -17.4% 😬 |
| Avg unique tiles | 193.5 | 160.6 | -17.0% |
| Avg revisit rate | 0.631 | 0.630 | ≈ same |
| Avg action entropy | 2.251 | 2.561 | +13.8% ✅ |
| North movement bias | **44.7%** | 12.8% | **cured** ✅ |
| Gather rate | 2.4% | **20.1%** | **+8x** ✅ |
| Combat actions | 0% | 9.9% (attack+defend) | new behavior ✅ |

**Interpretation:**
- The north-movement bias is gone — LSTM learned directional diversity
- Gather rate jumped 8x — memory lets it plan "I need to stockpile"
- Action entropy increased — more diverse behavior overall
- But lifespan dropped 17% — LSTM is still learning; 600k steps is not enough for a model that starts with random memory gates
- Revisit rate barely changed — exploration shaping (Stage 5) is the next lever

**Key takeaway:** The LSTM learned *better behaviors* (gathering, combat) but shorter survival. This is typical of warm-starting with random gates — the memory module destabilizes the MLP trunk initially. More training or a longer warm-up would likely recover lifespan.

---

## Stage 5 — Reward Shaping ✅

### The Problem: Gather Rate Was Only 2.4%

The agent *could* gather. The action existed and had a small reward (+0.05). But eating gave +0.20 immediately. Due to **temporal discounting** (γ = 0.99), rewards further in the future are worth less:

```
Value of reward R arriving t steps later = γᵗ × R
+0.20 eat now  = 0.20
+0.20 from stashed food 50 ticks later = 0.99⁵⁰ × 0.20 ≈ 0.12
```

The agent is mathematically rational — eating now IS better. This is the **sparse/delayed reward** problem.

### What Is Reward Shaping?

Adding intermediate rewards to guide an agent toward behaviors that *lead to* good outcomes, without waiting for the final payoff. You're illuminating the path.

**The catch:** naive shaping can be gamed. If gather = +0.50 and food respawns, the agent becomes a farmer glued to one tile, looping gather→wait→gather forever.

### Potential-Based Shaping (the safe version)

Due to Ng et al. (1999). Instead of rewarding the action, reward **progress toward a goal**:

```
shaping_reward = φ(new_state) - φ(old_state)
```

where φ (phi) is a **potential function** — a measure of how "good" a state is.

Example: φ = stash_fullness + (1 - hunger)

- Gather food: φ goes up → positive reward ✅
- Stand still: φ stays same → zero reward ✅ (can't be farmed)
- Stash fills up: φ can't increase further → reward drops to zero ✅

**Formal guarantee:** potential-based shaping never changes the *optimal* policy, only how fast the agent finds it.

### What We Implemented (LSTM v2 — 1.5M steps)

Three changes on top of LSTM v1:
1. **Gather reward doubled**: 0.05 → 0.10
2. **Inventory security potential**: reward Δ(food_in_hand / 5) × 0.12
3. **Starvation proximity penalty**: −0.08 per unit hunger above 0.70
4. **Survival-gated exploration**: explore bonus × (1 − hunger)

### 3-Way Results: MLP v3 → LSTM v1 → LSTM v2

```
                        MLP v3     LSTM v1    LSTM v2
                       baseline  no shaping  Stage 5
────────────────────────────────────────────────────
Avg lifespan (ticks)    534.6      441.4      449.6   ▲1.9%
Avg unique tiles        193.5      160.6      164.8   ▲2.6%
Avg revisit rate        0.631      0.630      0.629   ≈ same
Action entropy          2.251      2.561      2.394   more focused

North movement bias      44.7%     12.8%       1.9%   ✅ nearly eliminated
Gather rate               2.4%     20.1%      12.1%   ⚠️  dropped!
Eat rate                 15.2%     35.4%      43.4%   starvation shaping working
Defend                    0.0%      3.7%       8.7%   ✅ combat behavior evolved
```

### What Went Wrong (and Why It's a Great Lesson)

**The gather rate dropped from 20% → 12%** between v1 and v2. The starvation penalty backfired.

Here's why: the starvation penalty made agents fear hunger so much that they switched to eating *immediately from the ground* whenever hungry, rather than gathering into inventory first. In our env, eating from the ground gives `+0.20 × hunger_relief` *right now*. Gathering gives `+0.10 + Δφ_inventory` — spread across future ticks when inventory food gets used.

The starvation penalty increased the *urgency* to resolve hunger — but the *fastest* way to resolve hunger is still `eat`, not `gather`. So the agent optimised even harder for immediate eating.

**This is a classic reward interaction bug**: two well-intentioned signals that *individually* make sense, but *together* push the agent toward the wrong behavior. The starvation penalty and inventory shaping are fighting each other.

**The fix** (not yet implemented): decouple the signals. Instead of penalising hunger level, penalise *running out of inventory* when food is available nearby. That would make gathering the rational response, not eating.

### Exploration Shaping — Why It's Different

Exploration bonus (reward per new tile) is **not** potential-based because φ (unique tiles) only ever goes up. An agent that ignores food and sprints across the map could exploit this forever.

The fix used here: multiply exploration reward by `(1 − hunger)`. A starving agent gets zero exploration reward. A well-fed agent gets full reward. This is not potential-based either, but it's **bounded** (max one new-tile reward per tile, and tiles are finite) and **survival-safe** (can't explore at the cost of dying).

---

## Where We Are

**Completed:**
- ✅ Rewards and returns (Stage 1/2)
- ✅ Advantage function and PPO (Stage 1/2)
- ✅ MDP framing and partial observability (Stage 3)
- ✅ Evo vs PPO decision (Stage 3.5)
- ✅ Limbic system trained (v1→v3 checkpoints)
- ✅ Multi-agent combat environment (M4 sim features)
- ✅ Stage 4 theory + implementation: LSTM, gates, recurrent PPO, BPTT
- ✅ LSTM v1 trained (600k steps) — before/after comparison done
- ✅ Stage 5: Reward shaping theory + implementation + 3-way comparison

**Open question for next session:** fix the gather vs eat interaction — decouple starvation urgency from inventory shaping so they don't fight each other.

**Planned curriculum order (teacher's choice):**
1. ✅ **Stage 4:** Recurrent policies
2. ✅ **Stage 5:** Reward shaping — potential-based shaping, interaction bugs, gated exploration
3. **Stage 6:** Multi-agent dynamics — cooperative vs competitive, team credit assignment
4. **Stage 7:** Population-based training and specialization — the 3-sect experiment

---

## Quick Reference Glossary

| Term | Plain English | MurimSim Example |
|---|---|---|
| **Reward** | A number given after each action | +0.1 for moving toward food |
| **Return G** | Sum of all future rewards from this step | Total reward Zhang Wei accumulates until death |
| **Value V(s)** | Expected return from a given state | "Being at full health near food is worth ~5.2 reward" |
| **Advantage A** | Was this action better than expected? | Moving toward food when starving = high advantage |
| **Policy** | The function that maps observations to actions | The neural network inside each monk |
| **Actor** | The part of the network that picks actions | Outputs "60% move north, 20% eat, ..." |
| **Critic** | The part that estimates V(s) | Outputs "this situation is worth 3.7" |
| **PPO** | Stable policy gradient algorithm | What we used to train limbic_v1→v3 |
| **MDP** | Framework assuming full state visibility | What RL assumes; what we wish we had |
| **POMDP** | MDP with partial visibility | What MurimSim actually is (5×5 window) |
| **LSTM** | Recurrent network with working memory | The solution to the POMDP problem; not yet implemented |
| **Markov property** | Future depends only on present, not history | Broken by our 5×5 observation window |
