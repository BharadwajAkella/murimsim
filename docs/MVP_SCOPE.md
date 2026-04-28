# MurimSim — MVP Scope for Katja Demo (June 30, 2026)

> Hard deadline: demo-ready by June 30 for outreach to Dr. Katja Hofmann (MSR) and Joel Leibo (DeepMind).
> Anything not on the MVP path is post-Katja work, regardless of how interesting.

---

## What "MVP-ready" means

A screen recording (NOT a full replay viewer polish pass) showing:

1. **Three sects** with distinct cultural identities co-existing in a shared world
2. **Generation cycle working** — agents age, die, reproduce; traits inherited
3. **LLM cultural layer active** — at generation boundaries, applying selection bias + mutation modifier
4. **Reproducible emergence** — same seed → same inter-sect dynamics; different personality params → predictably different alliance/conflict patterns
5. **Repo writeup pinned** — README explains the three claims with links to positioning.md

That's it. No viewer polish. No power growth. No movable resources. No settlement experiments beyond the metrics that come for free.

---

## In scope (must ship by June 30)

| From plan.md | Why it's on the path |
|---|---|
| `settlement-metrics` | Pure instrumentation, no risk, useful measurement throughout |
| `shared-stash` | Substrate for sects — without collective resources, sects are cosmetic |
| `sect-scaffold` (Phase 6a) | Three sects with biome home regions and sect-level metrics |
| `viewer-territory` | Per-sect colors so the demo recording is legible |
| `inter-sect-combat` | In-group vs out-group dynamics — required for emergent alliance behavior |
| `aging` (Phase 6b) | Death by age threshold — required for generations |
| `reproduction` (Phase 6b) | Trait inheritance — the substrate for cultural drift |
| Phase 8 LLM culture (Mechanism 1) | The LLM-at-boundary layer that makes cultural drift visible and curated |

## Out of scope (post-Katja, even if tempting)

| From plan.md | Why it's cut |
|---|---|
| `dense-patch-map` | Useful eval scenario, not blocking. Default world is fine for demo. |
| `settlement-training-v9` | Full retraining run for settlement-specific tuning. Sect training run will exercise these mechanics anyway. |
| Phase 5.6 entirely (power growth, qi training, hazard unify) | Interesting but not on the claim. Skip until post-Katja. |
| Phase 7 (movable resources, haul action) | Conditional on settlement logs. Skip. |
| Mechanism 2 (parametrized social transmission) | Stretch only. Defer decision to mid-June based on remaining time. |
| Replay viewer polish | Screen recording with overlay text is sufficient for the demo. |
| Recurrent partial loops (credit assignment) | Separate research contribution. Toy-env paper post-Katja. |

---

## The "greed at end of project" rule

Locked April 27: **anything not on this list does not get added during paternity leave**, regardless of how cool it sounds. Mechanism 2 is the one explicitly-deferred decision; everything else stays cut.

If a new idea surfaces, it goes into `positioning.md` under "Deferred extensions / future work" — not into the code.

---

## Schedule alignment with Linear milestones

| Linear milestone | Target date | Maps to |
|---|---|---|
| RL survival layer working | May 25 | Already done (LSTM v8) — milestone retroactively satisfied |
| Personality + LLM cultural layer integrated | June 8 | `sect-scaffold` + `aging` + `reproduction` + Phase 8 Mechanism 1 |
| Reproducible emergent behavior demonstrated | June 22 | Run multi-seed sweeps, document reproducibility |
| Demo recording + Katja outreach sent | June 30 | Screen recording + repo polish + email sent |

---

## What ships in the demo recording

A 3–5 minute screen recording showing:

1. **Open with the claim** — overlay text explaining the three-layer architecture
2. **Show the sim running** — three sects, color-coded, in their home regions
3. **Highlight one emergent behavior** — e.g., two sects forming an alliance against the third, or a sect culturally drifting toward poison resistance after losses
4. **Reproducibility shot** — same seed produces same alliance pattern; different personality params shift it
5. **Generation transitions** — show the LLM cultural layer's selection bias output, show traits drift across generations
6. **Close with the ask** — "happy to walk through the architecture and discuss collaboration"

Nothing fancy. Honest, concrete, demoable.

---

## What goes in the outreach email

Pinned at top of repo as `OUTREACH_EMAIL.md`. Three paragraphs:

1. Who you are (Microsoft engineer, paternity leave research project)
2. What MurimSim is (one-line pitch from positioning.md), what's in the demo, link to repo + recording
3. Why them specifically — for Katja: multi-agent RL, agents in rich envs; for Leibo: explicit follow-up to Cook et al. with deliberate departure on the implicit-vs-curated question. Concrete ask: 30 min to discuss.

Draft these emails in the last week of June, after the demo is recorded.
