#!/usr/bin/env bash
# v22 multi-arena training chain: FF then recurrent.
# Mix: 1 base (30x30) + 2 arena_minion (12x12, 2 minions) + 1 arena_boss (16x16, boss).
# Carry cost stays enabled (was the survival winner in v21).
set -euo pipefail

cd "$(dirname "$0")/.."

ARENA_MIX="base:1,arena_minion:2,arena_boss:1"
COMMON_ARGS=(
  --total-steps 1500000
  --n-envs 4
  --n-agents 4
  --rollout-length 128
  --seed 22
  --checkpoint-interval 50
  --no-wandb
  --enable-carry-cost
  --arena-mix "$ARENA_MIX"
)

mkdir -p logs/v22a_ff logs/v22b_rec

echo "[v22] launching FF run..."
python3 -m scripts.train_ippo \
  "${COMMON_ARGS[@]}" \
  --checkpoint-dir checkpoints/ippo_v22a_ff \
  > logs/v22a_ff/train.log 2>&1

echo "[v22] FF complete; launching recurrent run..."
python3 -m scripts.train_ippo_recurrent \
  "${COMMON_ARGS[@]}" \
  --checkpoint-dir checkpoints/ippo_v22b_rec \
  > logs/v22b_rec/train.log 2>&1

echo "[v22] all runs complete at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
