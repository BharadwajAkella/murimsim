#!/usr/bin/env bash
# scripts/run_v25_chain.sh — v25 = v24 minus REWARD_GROUP_FORMATION.
#
# Tests whether cooperation survives without the 0.05 shaping bonus.
# Same arena/agent/carry-cost setup as v24 — only the formation bonus
# is gated off via --disable-formation-bonus.
#
# Run: nohup bash scripts/run_v25_chain.sh > logs/v25_chain.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs

ARENA_MIX="base:1,arena_minion:2,arena_boss:1"
COMMON_ARGS=(
  --total-steps 1500000
  --rollout-length 128
  --n-envs 4
  --n-agents 10
  --arena-mix "$ARENA_MIX"
  --enable-carry-cost
  --disable-formation-bonus
  --seed 25
  --no-wandb
)

echo "=== [v25a_ff] starting at $(date -u +%FT%TZ) ==="
WANDB_DISABLED=1 python3 -m scripts.train_ippo_joint \
  "${COMMON_ARGS[@]}" \
  --checkpoint-dir checkpoints/ippo_v25a_ff \
  > logs/v25a_ff.log 2>&1
echo "=== [v25a_ff] done at $(date -u +%FT%TZ) ==="

echo "=== [v25b_rec] starting at $(date -u +%FT%TZ) ==="
WANDB_DISABLED=1 python3 -m scripts.train_ippo_joint_recurrent \
  "${COMMON_ARGS[@]}" \
  --checkpoint-dir checkpoints/ippo_v25b_rec \
  > logs/v25b_rec.log 2>&1
echo "=== [v25b_rec] done at $(date -u +%FT%TZ) ==="

echo "=== chain complete at $(date -u +%FT%TZ) ==="
