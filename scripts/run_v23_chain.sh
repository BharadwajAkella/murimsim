#!/usr/bin/env bash
# scripts/run_v23_chain.sh — sequential v23 IPPO trainings (FSP enabled).
#
# Two runs:
#   v23a_ff:  feedforward, 4 trained + 6 frozen (v22a_ff)
#   v23b_rec: recurrent,    4 trained + 6 frozen (v22b_rec)
#
# Both use:
#   - 10 total slots / env (matches single-agent PPO setup that killed the boss)
#   - n_policy_agents=4 (only 4 trained — others are frozen baseline teammates)
#   - arena_mix base+arena_minion×2+arena_boss  (2 minion arenas weight forces combat)
#   - carry_cost on
#
# Run: nohup bash scripts/run_v23_chain.sh > logs/v23_chain.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs

ARENA_MIX="base:1,arena_minion:2,arena_boss:1"
COMMON_ARGS=(
  --total-steps 1500000
  --rollout-length 128
  --n-envs 4
  --n-agents 10
  --n-policy-agents 4
  --arena-mix "$ARENA_MIX"
  --enable-carry-cost
  --seed 23
  --no-wandb
)

echo "=== [v23a_ff] starting at $(date -u +%FT%TZ) ==="
WANDB_DISABLED=1 python3 -m scripts.train_ippo \
  "${COMMON_ARGS[@]}" \
  --frozen-ckpt checkpoints/ippo_v22a_ff/ippo_iter_000732.pt \
  --checkpoint-dir checkpoints/ippo_v23a_ff \
  > logs/v23a_ff.log 2>&1
echo "=== [v23a_ff] done at $(date -u +%FT%TZ) ==="

echo "=== [v23b_rec] starting at $(date -u +%FT%TZ) ==="
WANDB_DISABLED=1 python3 -m scripts.train_ippo_recurrent \
  "${COMMON_ARGS[@]}" \
  --frozen-ckpt checkpoints/ippo_v22b_rec/ippo_recurrent_iter_000732.pt \
  --checkpoint-dir checkpoints/ippo_v23b_rec \
  > logs/v23b_rec.log 2>&1
echo "=== [v23b_rec] done at $(date -u +%FT%TZ) ==="

echo "=== v23 chain complete at $(date -u +%FT%TZ) ==="
