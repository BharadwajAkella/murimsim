# MurimSim — SOTA Positioning Doc (DRAFT)

Working file. Walking the three closest prior works paragraph-by-paragraph, then refining.

---

## What MurimSim claims

Strategic emergence under survival pressure, **reproducible** across seeds, with cultural knowledge **inheritable across generations** via collective LLM-mediated transmission in a multi-agent world with identity-aware agents.

One-line pitch (working draft):
> *Park et al. show LLM agents produce believable individual social behavior. Voyager shows a single LLM agent can lifelong-learn. Cook et al. show RL populations can implicitly accumulate culture across generations. MurimSim is the missing combination: multi-agent populations that accumulate culture across generations through an LLM transmission medium, with reproducible strategic emergence under survival pressure.*

---

## Prior work 1 — Park et al., Generative Agents (UIST 2023)

### What they do
25 LLM-driven agents in a Sims-like sandbox (Smallville). Each agent has a memory stream (natural language record of all experiences), reflection (synthesis of memories into higher-level insights), and retrieval (pulling relevant memories to plan). LLM is called constantly — every plan, every action, every reflection.

### Env + what they measure
Smallville sandbox. Two evaluations: (1) **controlled** — interview agents to test coherence, ablate components to show each matters; (2) **end-to-end** — let sim run, observe emergent social behavior.

### Their claim
Believable individual human behavior + three emergent social dynamics: information diffusion (news spreads), relationship formation (dyadic memory of past interactions), and coordination on a goal (the Valentine's Day party example). Existence proof, not benchmark.

### MurimSim deltas
- **Goal:** Park aims for *believability* (HCI). MurimSim aims for *strategic emergence under pressure* (multi-agent RL). Different axis — not "more believable," differently scoped.
- **Substrate:** Park is LLM-every-step (expensive). MurimSim is two-tier — numerical RL state during generations, LLM only at generation boundaries. Quantifiable cost delta.
- **Memory unit:** Park's memory is *individual* (per-agent stream). MurimSim's cultural layer is *collective* (sect-level, transmitted across generations).
- **Generations:** Park has none. MurimSim's core mechanic is generational drift.
- **Reproducibility:** Park does not claim it. MurimSim explicitly claims same-seed → same-emergence.

### Behavior comparison table

| Park behavior | MurimSim equivalent | Mechanism in MurimSim |
|---|---|---|
| Information diffusion | Cultural knowledge spread | LLM at gen boundary processes interaction logs |
| Relationship formation | Pairwise parameter maps | Updated by RL-env interaction frequency |
| Coordination | Sect-coordinated survival behavior | RL with sect-level reward shaping; sect identity in observations |

### Methodology lift
Borrow Park's two-layer evaluation: **controlled ablations** (RL-only, RL+personality, full three-layer — show each is necessary) + **end-to-end run** (cultural drift across N generations, quantitative).

---

## Architecture decisions captured during Park reading

### Cultural transmission mechanism
Two candidates considered:

**Mechanism 1 — Programmed transmission.** Parent → offspring or sect-elder → next generation. Deterministic, clean, testable.

**Mechanism 2 — Parametrized social transmission.** Agents have secretiveness/brag-tendency parameters. During a generation, agents accumulate cheap structured interaction logs (no LLM calls). At generation boundary, LLM processes logs + parameters and decides which knowledge propagates to whom in next gen. Produces emergent information-flow patterns.

**Decision (locked April 27):** MVP starts with Mechanism 1. Mechanism 2 is stretch, decision deferred to mid-June based on remaining time. Named "greed at end-of-project" risk; protected against by deferring decision rather than committing now.

### Coordination prerequisite
Sect-level outcomes must feed individual rewards or sects are costumes. Reward shaping must explicitly tie individual survival to sect performance for coordination to emerge from RL.

### Relationships in v1
Pairwise parameter maps (trust, interaction frequency). Sacrifice Park-style relationship richness for reproducibility + cost. If reviewer pushes: "we deliberately traded relationship richness for reproducibility and scale."

### Open question (parked)
Long-term in-character conversations between agents. Not MVP. Revisit post-Katja.

---

## Prior work 2 — Voyager (Wang et al., 2023)

### What they do
First LLM-powered embodied **lifelong learning** agent in Minecraft. Single agent. Three components: (1) automatic curriculum that maximizes exploration, (2) ever-growing skill library of executable code, (3) iterative prompting with environment feedback, execution errors, and self-verification. Uses GPT-4 blackbox queries — no fine-tuning. Code as the action space (not low-level motor commands), so skills are temporally extended, interpretable, compositional.

### Env + what they measure
Minecraft (open-ended, no fixed end goal). Metrics: unique items obtained, distance traveled, tech tree milestones unlocked, generalization to new Minecraft worlds. Reports 3.3x more unique items, 2.3x longer distances, up to 15.3x faster tech tree vs prior SOTA (ReAct, Reflexion, AutoGPT).

### Their claim
A single LLM agent can lifelong-learn in an open-ended world by accumulating reusable skills as code, without fine-tuning the model. Skills compose and alleviate catastrophic forgetting.

### MurimSim deltas
- **Single-agent vs multi-agent.** Voyager: one agent. MurimSim: populations.
- **Lifelong vs generational.** Voyager accumulates within one lifetime. MurimSim accumulates *across* lifetimes (cultural inheritance, not skill memory).
- **LLM-only vs hybrid.** Voyager rejects RL on principle. MurimSim uses RL as survival floor + LLM only at generation boundaries.
- **Solitary vs transmission.** Voyager has no concept of teaching another agent. MurimSim's central mechanic *is* transmission.

### Positioning
Voyager and MurimSim answer different questions — "can one LLM agent learn forever?" vs "can populations culturally inherit and drift?" Not competitors; different research programs. Voyager is the single-agent lifelong baseline; MurimSim is multi-agent generational.

---

## Deferred extensions / future work (NOT MVP)

Captured ideas that are out of scope for paternity-leave MVP but worth preserving.

### "God agent" — Voyager-style omniscient cultural seed
A single immortal agent that lifelong-learns Voyager-style, holds visibility into all population learning, and occasionally interacts with mortal agents to seed insights (e.g., handing poison-immunity hint to a sect that's struggling in a high-poison region).

**Why interesting:** Bridges Voyager (single-agent lifelong) + Park (social interaction) + Cook (cultural accumulation) in one system. Provides a non-human observer/oracle. Genuinely novel — no prior work does this.

**Why deferred:** Adds a fourth layer before the three-layer is built. Doesn't serve the MVP claim (reproducible inter-sect alliance under survival pressure). Risks breaking reproducibility. Dilutes the Voyager differentiation (multi-agent generational vs single-agent lifelong).

**Where it belongs:** Future-work section of the writeup. Strong candidate as the Katja collaboration ask — "here's working three-layer cultural drift; a natural extension is a Voyager-style omniscient cultural seed."

---

## Prior work 3 — Cook et al., Artificial Generational Intelligence: Cultural Accumulation in RL (NeurIPS 2024)

Authors: Jonathan Cook, Chris Lu, Edward Hughes, **Joel Z. Leibo**, Jakob Foerster. Oxford + Google DeepMind.

**Note:** Leibo is a possible second outreach target alongside Katja. This paper is his recent work in this exact space.

### What they do
Achieve **implicit** cultural accumulation in RL agents — accumulation that emerges from balancing social learning with independent learning, rather than from hand-crafted imitation steps. Two regimes:
- **In-context accumulation:** frozen policy weights θ, internal state φ accumulates across episodes within a meta-RL trained agent.
- **In-weights accumulation:** parameters θ are the substrate — each generation trained from random init, prior generation's policy serves as oracle/teacher signal.

Pure RL throughout. S5 / GRU / CNN architectures. No LLM anywhere.

### Env + what they measure
Three minimal abstract envs:
- **Memory Sequence** — sequence memorization (cultural-accumulation analog from Cornish et al. 2017)
- **Goal Sequence** — open-ended adaptation of Goal Cycle (Bhoopchand et al. 2023)
- **TSP** — traveling salesman

Metric: returns achieved across G generations vs returns from a single lifetime of equivalent total compute. Show cumulative improvement.

### Their claim
Cultural accumulation in RL is achievable *without* hand-crafted imitation, by training setups that balance social learning with independent learning. They explicitly position against iterated policy distillation and expert iteration as too hand-crafted. Implicit > explicit, in their view.

### MurimSim deltas
- **LLM transmission medium vs pure RL.** They accumulate culture *implicitly* in policy parameters or internal state. MurimSim accumulates culture *explicitly* in natural-language-summarized cultural state at generation boundaries.
- **Minimal abstract envs vs rich multi-agent world.** Memory Sequence / TSP have no named affordances, no concepts of "poison immunity" or "alliance" — just gradients on returns. MurimSim's environment supports identifiable, legible discoveries.
- **Implicit-only vs hybrid.** Cook et al. argue implicit is the right ideal. MurimSim deliberately departs from this (see design thesis below).
- **No social structure.** Cook et al. agents are interchangeable. MurimSim has sects, sect identity in observations, sect-level cultural state.

### RL paradigm comparison (mechanical, not motivational)
The shared motivation (cultural accumulation across generations) is what makes Cook et al. the closest prior work. The RL mechanisms are structurally different — these are independent deltas from the cultural-layer delta.

| Axis | Cook et al. | MurimSim |
|---|---|---|
| **Agent count per generation** | Single-agent meta-RL | Multi-agent population |
| **Generations** | Sequential training runs (one then next) | Co-existing populations |
| **Environment** | Abstract tasks (Memory Sequence, Goal Sequence, TSP) | Rich shared world (terrain, hazards, resources, other agents) |
| **Architecture** | S5 / GRU + CNN/feedforward, sequence-model heavy | Standard PPO actor-critic, sequence story lives at gen boundary |
| **Reward** | Task return (one scalar per episode) | Survival + sect-level shaping (coordination has gradient pressure) |
| **Social learning** | Between generations only — observe teacher policy | Within *and* between generations — agents see each other as agents |
| **Identity** | Interchangeable agents | Sect identity in observations |
| **Group structure** | None | Sects with sect-level cultural state |

**Bottom line:** Cook does single-agent meta-RL on abstract tasks with sequential generation training and observation-based social learning. MurimSim does multi-agent PPO in a rich shared world with identity-aware agents and sect-level reward shaping, with co-existing populations. Two independent deltas — RL paradigm + cultural transmission medium.

### The danger
This is the closest prior work — a careless reviewer will say "isn't this just Cook et al. with an LLM?" The positioning must make clear *why* the LLM medium changes what's possible: interpretability, qualitative curation of rare events, symbolic concepts as cultural units.

---

## Design thesis — why MurimSim deliberately departs from Cook et al.

Cook et al.'s purist position: hand-crafted transmission constrains the search space; let accumulation emerge from balanced imitation/innovation. This is how human culture works — no transmission committee.

**MurimSim's counter-position:** Pure implicit accumulation has a known failure mode — *valuable rare events get lost to noise.* A sect that discovers poison immunity once and dies before transmitting it loses the discovery to the next generation. In Cook's pure-RL setup this is fine, because there's no "concept" of poison immunity to lose — only gradients that may or may not reproduce the behavior. In a richer multi-agent world with named affordances, losing identifiable discoveries is a real loss.

LLM-mediated transmission performs **qualitative curation** at the generation boundary — preserving rare-but-valuable events as legible cultural knowledge rather than letting them die in gradient noise. Within-generation: RL does what RL is good at (behavior under pressure). Between generations: LLM does what LLM is good at (qualitative evaluation, summarization, symbolic preservation).

**Counter to "implicit is purer":** Cook et al.'s policy-parameter accumulation is *also* a hand-crafted choice (architectures, observation encodings). Every transmission medium imposes structure. The question is which structure matches the phenomenon. For studying multi-agent cultural drift in environments with named affordances, an LLM medium fits the phenomenon better than gradient-only accumulation.

**The MurimSim claim:**

> *Cultural accumulation in rich multi-agent environments benefits from a qualitative transmission medium (LLM at generation boundaries) that preserves rare-but-valuable discoveries which pure-policy implicit accumulation loses to noise. We trade some implicitness for legibility and curation — a deliberate design choice, not a regression toward hand-crafted imitation.*

### Post-MVP ablation to support this thesis
Three conditions on the same MurimSim env:
1. **Mechanism 1** — programmed parent→offspring transmission (close to what Cook critiques)
2. **Pure implicit** — Cook-style, no LLM, behavioral copying / observation-based imitation between generations
3. **Mechanism 2 hybrid** — LLM-curated transmission at generation boundaries (MurimSim's position)

Measure: rare-but-valuable discovery preservation rate across G generations. Hypothesis: hybrid > programmed > pure implicit.

This is a paper-worthy claim. **Out of MVP scope** — but the empirical case for the design choice. MVP just demonstrates the system works; the paper later argues *why* via this ablation.

---

## What MurimSim claims that none of the three claim

Each claim is tagged with which prior work it differentiates from. Where multiple, MurimSim sits at the intersection none of them cover.

### Capability claims

**1. Generational cultural drift in a multi-agent population via LLM transmission medium.**
- Differs from **Park** (no generations, individual memory only)
- Differs from **Voyager** (single-agent, no inheritance, no transmission)
- Differs from **Cook** (generational and multi-pop-style, but no LLM medium — pure policy/internal-state accumulation)
- *Nobody combines all three: multi-agent + generational + LLM-mediated.*

**2. Reproducible strategic emergence under survival pressure.**
- Differs from **Park** (existence proofs of social behavior; reproducibility not claimed)
- Differs from **Voyager** (proficiency metrics on Minecraft, not reproducibility of emergent group dynamics)
- Differs from **Cook** (return curves across seeds, but not "same-seed → same emergent structure" in a rich world)
- *MurimSim explicitly claims same-seed → same inter-sect alliances and same drift patterns.*

**3. Identity-aware agents with sect-level cultural state.**
- Differs from **Park** (individual memory streams, no group structure)
- Differs from **Cook** (interchangeable agents, no group identity)
- *Sect identity in observations + sect-level inherited culture is unique to MurimSim among these three.*

### Architectural claims

**4. LLM-cheap two-tier architecture.**
- Differs from **Park** (LLM-every-step, expensive)
- Differs from **Voyager** (LLM-every-decision, expensive)
- Differs from **Cook** (no LLM, but doesn't get LLM benefits either)
- *RL handles within-generation behavior; LLM only at generation boundaries. Quantifiable cost delta vs Park/Voyager; capability delta vs Cook.*

**5. Qualitative curation thesis.**
- Direct response to **Cook**'s implicit-is-purer position.
- *Argues LLM-mediated transmission preserves rare-but-valuable named-concept discoveries (poison immunity as a concept, not just a behavioral pattern) which pure-policy implicit accumulation loses to noise. Defended via the post-MVP three-condition ablation.*

### Method claim (separable from MurimSim itself)

**6. Recurrent partial loops for credit assignment.**
- Independent contribution. Sample intermediate states during a generation, branch into alternative-strategy vs Markov branches to identify causally pivotal decisions. Sharpens PPO advantage estimates.
- Benchmarkable on toy env independent of MurimSim — separate paper potential.

---

## Outreach implications

- **Katja Hofmann (MSR):** Multi-agent RL, game AI, agents in rich environments. Positioning emphasizes claims 1, 2, 3, 4 — the multi-agent generational architecture. Demo focused on reproducible inter-sect alliance.
- **Joel Leibo (DeepMind):** Co-author on Cook et al., long history in cooperative AI / multi-agent RL. Positioning emphasizes claims 1 and 5 — the explicit departure from Cook's implicit-only position, defended as a deliberate design choice. The post-MVP ablation is the natural collaboration ask.
- **God-agent extension** is the future-work hook for either conversation.
