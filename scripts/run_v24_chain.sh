#!/usr/bin/env bash
# scripts/run_v24_chain.sh — sequential v24 joint-action IPPO trainings.
#
# v24 = body+social action head split. v23 single-head ckpts are
# action-shape incompatible, so v24 trains pure self-play (no FSP).
# This is intentional — first run isolates the head-split effect.
# v25 will reintroduce FSP using a v24 frozen baseline.
#
# Two runs:
#   v24a_ff:  feedforward joint
#   v24b_rec: recurrent joint
#
# Same arena_mix and carry_cost as v23 for apples-to-apples (modulo
# the missing FSP anchor — flagged as a confound in journal).
#
# Run: nohup bash scripts/run_v24_chain.sh > logs/v24_chain.log 2>&1 &
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
  --seed 24
  --no-wandb
)

echo "=== [v24a_ff] starting at $(date -u +%FT%TZ) ==="
WANDB_DISABLED=1 python3 -m scripts.train_ippo_joint \
  "${COMMON_ARGS[@]}" \
  --checkpoint-dir checkpoints/ippo_v24a_ff \
  > logs/v24a_ff.log 2>&1
echo "=== [v24a_ff] done at $(date -u +%FT%TZ) ==="

echo "=== [v24b_rec] starting at $(date -u +%FT%TZ) ==="
WANDB_DISABLED=1 python3 -m scripts.train_ippo_joint_recurrent \
  "${COMMON_ARGS[@]}" \
  --checkpoint-dir checkpoints/ippo_v24b_rec \
  > logs/v24b_rec.log 2>&1
echo "=== [v24b_rec] done at $(date -u +%FT%TZ) ==="

echo "=== chain complete at $(date -u +%FT%TZ) ==="
