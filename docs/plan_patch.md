# Patch for plan.md — Add at the top, before "Vision & Scope"

Add this section as the very first thing in `plan.md`. It anchors every Copilot session in strategic context.

---

```markdown
## ⚓ MVP Anchor — Read Before Every Session

> This project has a hard demo deadline of **June 30, 2026** for outreach to Dr. Katja Hofmann (MSR) and Joel Leibo (DeepMind).
> Strategic context lives in `positioning.md`. MVP scope cuts live in `MVP_SCOPE.md`. Read both before making design decisions.

### The three claims this demo must demonstrate

1. **Generational cultural drift in a multi-agent population via LLM transmission medium** — combination not present in Park et al., Voyager, or Cook et al.
2. **Reproducible strategic emergence under survival pressure** — same seed → same inter-sect dynamics
3. **Identity-aware agents with sect-level cultural state** — sect identity in observations, sect-level inherited culture

### Architecture commitments (do not revisit without explicit unlock)

- **LLM is called only at generation boundaries**, not within generations. This is a deliberate cost + reproducibility delta vs Park et al. (LLM-every-step) and a capability delta vs Cook et al. (no LLM).
- **Sect identity in agent observations from day one of sect work.** Cook et al. agents are interchangeable; ours are not.
- **Sect-level outcomes feed individual rewards.** Without this, sects are cosmetic. Coordination must have gradient pressure.
- **Cultural transmission Mechanism 1 first** (programmed parent→offspring + sect-elder transmission). Mechanism 2 (parametrized social transmission with secretiveness/brag-tendency, resolved by LLM at boundary) is **stretch**, decision deferred to mid-June based on remaining time.
- **Relationships in v1 are pairwise parameter maps** (trust, interaction frequency). Park-style rich relationships are deliberately out of scope.

### What's cut from the plan for MVP

See `MVP_SCOPE.md` for the full list. Headlines:

- Phase 5.5b/5.5c (`dense-patch-map`, `settlement-training-v9`) — useful, not blocking. Skip.
- Phase 5.6 entirely (power growth, qi training, hazard unify) — post-Katja.
- Phase 7 (movable resources, haul action) — post-Katja.
- Replay viewer polish — screen recording with overlay is sufficient.
- Mechanism 2 — stretch only. Default to deferral.

### MVP path — the only Fast Lane that matters until June 30

1. `settlement-metrics` — pure instrumentation, no game changes
2. `shared-stash` — substrate for sects (without collective resources, sects are cosmetic)
3. `sect-scaffold` (Phase 6a) — three sects, biome home regions, sect-level metrics
4. `viewer-territory` — per-sect colors for demo legibility
5. `inter-sect-combat` — in-group vs out-group dynamics
6. `aging` (Phase 6b) — death by age threshold
7. `reproduction` (Phase 6b) — trait inheritance
8. **Phase 8 Mechanism 1 LLM culture** — selection bias + mutation modifier at generation boundaries
9. **Reproducibility sweeps** — multi-seed runs, document same-seed → same emergence
10. **Demo recording + repo polish + outreach email**

### The "greed at end of project" rule

Self-named risk. New ideas during paternity leave go into `positioning.md` under "Deferred extensions / future work" — not into the code. Mechanism 2 is the only deliberately-deferred decision; everything else stays cut.

---
```

# Optional second patch — update Progress Tracker

In the Progress Tracker section near the bottom of plan.md, replace the row sequence with the MVP-aligned sequence. Specifically, mark these as "Skip for MVP":

- Phase 5.5b dense-patch-map → 🟡 Skip for MVP (post-Katja)
- Phase 5.5c shared-stash → ✅ MVP path (substrate for sects)
- Phase 5.6 Power Growth → 🟡 Skip for MVP (post-Katja)
- Phase 7 Movable Resources → 🟡 Skip for MVP (post-Katja)

# Optional third patch — Fast Lane reorganization

Above the existing Fast Lane tables, add a "🎯 MVP Path (June 30 deadline)" section with only the 8–10 tasks that ship the demo. Move everything else under a "🟡 Post-Katja Backlog" header so Copilot doesn't accidentally pick those up first.

This protects against the failure mode where Copilot, given the full Fast Lane, picks the most technically interesting ticket rather than the one on the critical path.
