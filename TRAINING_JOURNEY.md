# MurimSim — Training Journey

A canonical, AI-readable record of every training run, what was implemented in
the run, what we observed in eval, and what decision came out of it. New runs
are appended at the bottom.

This file is the *single source of truth* for "what version did what." Plan
docs and session checkpoints can disappear; this one stays in the repo.

## How to read this doc

Every run gets the same template:

```
### <run-id> — <one-line headline>
- Date:           YYYY-MM-DD
- Phase:          <phase number / name>
- Parent ckpt:    <warm-start source or "cold">
- Steps / seed:   <total env steps> / seed <n>
- Config:         <config file path>
- Ckpt dir:       checkpoints/<dir>/

What changed (env / reward / arch):
  • bullet
  • bullet

What we observed (eval — det = deterministic argmax, stoch = stochastic):
  • metric: value         comment

Decision:
  - kept / discarded / warm-start source for next
```

### Glossary
- **det** — deterministic eval; the trained policy always picks its argmax action.
- **stoch** — stochastic eval; actions are sampled from the policy distribution.
- **life_reward** — mean total reward per completed life (survival score).
- **reciprocity** — dyadic reciprocity, correlation of `affinity[A→B]` with `affinity[B→A]` across all pairs (1.0 = mutual, 0.0 = one-sided).
- **collab_picks** — number of times agents picked the COLLABORATE social action.
- **trade_propose / accept / reject_picks** — masked-on social-action picks for trade lane.
- **trades_executed_cum** — successful trade swaps (PROPOSE → ACCEPT → swap).
- **gifts_executed_cum** — successful GIFT transfers (giver's inventory drops, receiver's grows).

---

## Era 1 — SB3 LSTM single-policy survival (`limbic_lstm_v1` … `v20`)

Documented in `WHAT_WE_BUILT.md`. Covers:
- Phase 1 (world + foraging)
- Phase 2 (poison + qi drain)
- Phase 3 (combat + monsters)
- Phase 5 (sociability + COLLABORATE / WALK_AWAY)
- Food sharing + reciprocity (v8 = peak 115.0 reward at 1.74M)
- Shared stash + dense-patch settlement map (v9)

Key inheritance: the limbic v8/v9 family established the survival baseline
*before* the IPPO multi-agent rewrite. Not re-evaluated in this doc.

---

## Era 2 — IPPO joint-action multi-agent (`ippo_v20a` … `ippo_v32`)

The current architecture: per-agent independent PPO with a shared recurrent
encoder, **joint body+social action heads** in the same step. This is where
all Phase 7+ behaviour (courtship, gifting, trade) lives.

### ippo_v20a / v20b — IPPO baseline (FF and recurrent)
- Phase: rewrite from SB3 to in-house IPPO
- Steps:    500k each, cold
- Ckpt dir: `checkpoints/ippo_v20a_ff/`, `ippo_v20b_recurrent/`

What changed:
  • Replaced SB3 single-policy LSTM with per-agent IPPO + shared encoder.
  • Added joint body+social action dispatch.

Decision: kept v20b_recurrent as the spine for everything that followed.

---

### ippo_v21a / v21b / v21c — boss + carry-cost iterations
- Phase 6 — combat + reward shaping
- Steps:    ~600k–1M each
- Ckpt dirs: `ippo_v21a_*`, `ippo_v21b_*`, `ippo_v21c_*`

What changed:
  • v21a: re-introduced boss monster from limbic era.
  • v21b: dropped formation bonus to test natural cooperation.
  • v21c: added inventory carry cost (`--enable-carry-cost`). This becomes
    the standard flag for all subsequent runs because without it agents
    hoard resources indefinitely.

Decision: v21c (carry cost on) became the new survival baseline.

---

### ippo_v22a / v22b — multi-arena round-robin
What changed:
  • Round-robin arena mix (resource-dense / scarce / poison-heavy / dense-patch).
  • Trains a single policy that has to generalise across maps.

Decision: kept; arena mix is now standard.

---

### ippo_v23a / v23b — fictitious self-play
What changed:
  • FSP: occasionally roll out against frozen older versions of self.
  • Tests in `test_v23_fsp.py`.

Decision: kept; helps stabilise long training runs.

---

### ippo_v24a / v24b — joint action head (the big architectural change)
What changed:
  • Body + social actions are now sampled jointly in one step (previously
    serial). Required reworking obs, masks, and reward credit assignment.
  • All later runs assume joint heads.

Decision: kept; this is the architecture used for all Phase 7+ work.

---

### ippo_v25a / v25b — formation bonus gate
What changed:
  • Made the formation bonus opt-in via `--enable-formation-bonus`. Default OFF.
  • Confirmed cooperation survives without artificial reinforcement.

Decision: kept; formation bonus stays OFF by default.

---

### ippo_v26_courtship — Phase 7 baseline (courtship, marriage, birth)
- Phase:    7
- Parent:   v25b (warm-start via `expand_checkpoint_v25_to_v26.py`)
- Steps:    1M
- Config:   `config/v26_courtship.yaml`
- Ckpt dir: `checkpoints/ippo_v26_courtship/`

What changed:
  • New social actions: PROPOSE_MARRIAGE, ACCEPT_MARRIAGE, REJECT_MARRIAGE.
  • Birth events: married agents in adjacent cells with full inventory
    spawn a baby that inherits averaged parent traits ± noise.
  • Per-agent partner state, marriage cooldown.
  • Obs space grew (encoder 304 → 316). Old ckpts must be expanded via
    `scripts/expand_checkpoint_v25_to_v26.py`.

What we observed:
  • Marriages and births fire at modest rates.
  • Reciprocity remained ~1.0.

Decision: kept; courtship is now standard. This is the substrate for
hereditary work.

---

### ippo_v26_courtship_long — extended courtship training
- Steps: 2M (extension of `_init`)
- Decision: kept as backup baseline.

---

### ippo_v27_gift / v28_gift_balanced — Phase 8c GIFT (limbic / social lane)
- Phase:    8c
- Parent:   v26_courtship_long
- Steps:    600k each

What changed (v27):
  • New `SocialAction.GIFT`. A gives 1 inventory unit to co-located B.
  • `_resolve_gift` in `multi_env`, dispatch in `ippo_env`.
  • Reward shaping: receiver got a flat scalar (later removed).
  • Eval surfaces `gift_picks`, `gifts_executed_cum`, `gift_value_cum`.

Eval v27 (5000 steps, det):
  • gift_picks:        5,690
  • gifts_executed:    5,559        ~98% success rate
  • mean_life_reward:  2.37
  • dyadic_reciprocity: 0.9999

What changed (v28):
  • Diminishing-returns on receiver utility (more food less valuable).
  • Carry cost enabled.

Eval v28 (5000 steps, det):
  • gifts_executed:    2,334        −58% vs v27
  • mean_life_reward:  2.35         flat
  • reciprocity:       1.000        flat
  • completed_lives:   430          flat

Interpretation: diminishing utility cut excessive gifting in half without
hurting survival or bonds. v28 became canonical for Phase 8c.

Decision: kept v28 as the next warm-start source.

---

### ippo_v29_gift_body — moving GIFT to body lane (architectural fix)
- Phase:    8c.2
- Parent:   v28
- Steps:    600k

What changed:
  • GIFT moved from social lane → body lane. Semantically GIFT is a physical
    action (hand off an item), not an emotional one.
  • Body action space grew. Warm-start expand of v28's policy weights
    handled the new body output dim.

Eval v29 (5000 steps, det):
  • gifts_executed_cum: 0            collapsed (policy hadn't relearned)
  • mean_life_reward:  0.20          regression vs v28's 2.35
  • reciprocity:       0.404         regression vs v28's 1.000

Interpretation: relocating GIFT broke both the gift loop AND the COLLAB
dynamic that depended on it. Survival regression is real, not noise.

Decision: kept v29 *only* as a problem state to fix in the next ticket.
Not a deployment ckpt.

---

### ippo_v30_bilateral_collab — bilateral COLLABORATE (Phase 8c.3 fix)
- Phase:    8c.3
- Parent:   v29
- Steps:    600k

What changed:
  • COLLABORATE became bilateral: a group only forms if BOTH agents pick
    COLLABORATE in the same tick (previously unilateral).
  • Removes the loophole where a single agent could "claim" a partner.

Eval v30 (5000 steps, det):
  • reciprocity:       0.404 → **0.939**     mutual consent restored
  • mean_life_reward:  0.20 → **0.55**       +175% vs v29 (partial recovery)
  • groups_formed_cum: 233 (rate-limited but real)

Interpretation: bilateral consent makes co-formation the proxy for
mutuality. Survival recovers but doesn't return to v28's 2.35 — the
GIFT-in-body change still costs us, but the floor is acceptable.

Decision: kept; v30 is the canonical pre-trade baseline.

---

### ippo_v31_trade — Phase 8d.1 TRADE scaffold (cold)
- Phase:    8d.1
- Parent:   v30
- Steps:    600k

What changed:
  • Three new social actions: PROPOSE_TRADE, ACCEPT_TRADE, REJECT_TRADE.
  • Multi-offer inbox: `dict[receiver_idx, list[TradeOffer]]` with
    TTL of 1 tick. Receiver picks the highest-value offer when accepting.
  • Heuristic `_compose_trade_offer` proposes an offer based on inventory
    surplus / deficit. (Learned head deferred — no signal until trades fire.)
  • Symmetric `+0.2 affinity` bump on successful swap. **No scalar reward.**
  • Eval surfaces `trade_propose/accept/reject_picks`, `trades_executed_cum`,
    `trades_rejected_cum`, `trade_value_cum`.

Eval v31 (5000 steps, det, 4×6):
  • mean_life_reward:    0.55 → 0.94      improvement vs v30
  • trade_propose_picks: present
  • trade_accept_picks:  0
  • trades_executed_cum: 0                ACCEPT never won argmax

Interpretation: scaffold works (proposals fire) but the policy hasn't
discovered the ACCEPT side. Social entropy is too low — needs warm-up.

Decision: kept as warm-start source for v32.

---

### ippo_v32_trade_warm — Phase 8d.1b long warm-up
- Phase:    8d.1b
- Parent:   v31
- Steps:    2M
- Seed:     32
- Knob:     `--ent-coef 0.03` (3× default to fight social-head collapse)
- Ckpt dir: `checkpoints/ippo_v32_trade_warm/` (33 checkpoints)

What changed:
  • Same code as v31, just longer training and elevated entropy coefficient.

Training trajectory:
  • Social entropy `sent` jumped from 0.017 → 0.10–0.13 by step ~250k.
  • Late collapse to 0 around step 1.3M (the higher ent-coef bought us
    1M useful steps before exploration died).
  • `mean_life_reward` climbed 0.55 → 0.94 → 1.6 → ~2.65 across iterations.

Per-ckpt deterministic eval (3000 steps × 4 × 6):
  • Best = `iter_000400.pt`: life_reward 2.91, reciprocity 1.00.
  • All ckpts: trades_executed = 0 in argmax (same blockage as v31).

Stochastic eval on iter_400 (5000 steps):
  • mean_life_reward:    2.65
  • dyadic_reciprocity:  0.99
  • trade_propose_picks: 1,283
  • trade_accept_picks:  21
  • trades_executed_cum: **10**       *first non-zero successful trades*
  • trade_value_cum:     46.5
  • trade_reject_picks:  645          (399 landed on real offers)

Interpretation: the trade machinery works end-to-end — under stochastic
sampling the policy explores ACCEPT enough to fire 10 successful swaps.
But under deterministic argmax, ACCEPT_TRADE is never the top action
inside its narrow legal window (TTL=1, range=1, asked-resource present).
The `+0.2` affinity bump is the only carrot, and it isn't loud enough.

Decision: **kept**. `iter_000400.pt` is the canonical Phase 8d.1 baseline.

---

### ippo_v33_trade_shaped — Phase 8d.2 small explicit trade reward (REMOVED)
- Phase:    8d.2 (reverted)
- Parent:   v32 iter_400
- Steps:    1M, seed 33

What changed:
  • Added `TRADE_REWARD_BOTH = 0.05` constant. Both proposer and receiver
    received +0.05 reward on a successful swap.

Eval v33 (5000 steps, stoch):
  • trades_executed_cum: 10 → 22         doubled vs v32
  • trade_value_cum:     46.5 → 106
  • mean_life_reward (best ckpt iter_200, det): 3.66       (best so far)
  • reciprocity:        1.0
  • trade_accept_picks (det): **0**       still no argmax trades

Decision: **discarded**. The reward shaping doubled stochastic trade rate
but failed the argmax goal. Useful proof that the mechanism worked but
the magnitude was insufficient.

---

### ippo_v34_trade_shaped_strong — Phase 8d.2 stronger explicit trade reward (REMOVED)
- Phase:    8d.2 (reverted)
- Parent:   v33 iter_200
- Steps:    800k, seed 34

What changed:
  • Bumped `TRADE_REWARD_BOTH` from 0.05 → 0.20.

Eval v34 (3000 steps, det, 4×6, best ckpt iter_260):
  • trade_accept_picks:   3       *first time argmax picks ACCEPT_TRADE*
  • trades_executed_cum:  3       100% accept→execute success rate
  • trade_value_cum:      12.0
  • mean_life_reward:     1.46    regression vs v33's 3.66
  • dyadic_reciprocity:   0.65    swung 0.37–0.95 across ckpts (unstable)

Interpretation: argmax goal achieved but at the cost of survival and
relationship stability. The bigger trade reward distorted the body lane —
agents stopped eating reliably and traded with whoever was nearby.

Decision: **discarded by user**. Quote: *"If there is not enough use out
of trading, then agents not trading is the right thing to do. Our shaping
should be to make resources more useful not forcing trades… Semantically
trading probably isn't a limbic-system thing anyway."*

The TRADE_REWARD_BOTH constant was removed from the codebase. Both v33
and v34 checkpoint directories were deleted (62 MB).

The trade *machinery* (PROPOSE / ACCEPT / REJECT / multi-offer inbox /
TTL / value-scored receiver) remains in the env — it's available for
the LLM lane in a future phase.

---

## Current canonical baseline

| field | value |
|-------|-------|
| Ckpt  | `checkpoints/ippo_v32_trade_warm/ippo_joint_recurrent_iter_000400.pt` |
| Phase | 8d.1 (trade scaffold present, no shaping) |
| life_reward (det)  | 2.91 |
| reciprocity (det)  | 1.00 |
| trades_executed (det) | 0 (acceptable — trade is intentionally limbic-unfriendly) |
| Tests | 435 passed, 6 skipped |

---

## Open work (queued in session todos)

- `8e-hereditary-stash-inheritance` — babies inherit a fraction of parent stash
- `8e-hereditary-skill-bias` — babies inherit averaged parent traits
- `infra-ent-coef-schedule` — linear ent-coef anneal (fix the v32-style late collapse)
- `infra-eval-derived-metrics` — `trade_success_rate` field in eval JSON

## Cancelled lines of work

- All trade-reward-shaping tickets (`8d3-*`, `8d4-learned-trade-head`).
  Reason: trading isn't limbic; pressure to trade should come from
  resource scarcity, not from shaping. Trade may resurface in the LLM lane.

---

## Template for new entries

When adding a new run, copy the block below, fill it in, and append above
"Current canonical baseline".

```
### ippo_vXX_<short-name> — <one-line headline>
- Phase:    <8e / 9 / etc>
- Parent:   <warm-start source>
- Steps:    <N>, seed <n>
- Config:   <path>
- Ckpt dir: checkpoints/<dir>/

What changed:
  • bullet

Eval (5000 steps, det / stoch, 4×6):
  • metric: value     comment

Interpretation:
  • bullet

Decision:
  - kept / discarded / warm-start source for next
```
