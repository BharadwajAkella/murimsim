#!/usr/bin/env bash
# v21 chain launcher — runs v21b (boss + legacy_stash) then v21c (+ carry cost)
# after v21a completes. Designed to be invoked detached via setsid+nohup so it
# survives session shutdown.
#
# Schedule (each pair runs concurrently; pairs are sequential):
#   1. wait for v21a_ff_boss + v21a_rec_boss (already running)
#   2. launch v21b_ff + v21b_rec    (--enable-boss)
#   3. launch v21c_ff + v21c_rec    (--enable-boss --enable-carry-cost)
#
# All commits / code paths are cumulative: v21b inherits boss, v21c inherits
# boss + legacy_stash + adds carry cost.

set -uo pipefail

LOG=/home/ravakella/murimsim/logs/v21_chain.log
cd /home/ravakella/murimsim

log() { echo "[$(date -Iseconds)] $*" >> "$LOG"; }

wait_for_pair() {
    local name="$1" dir_a="$2" dir_b="$3"
    log "waiting for $name pair to finish (final ckpt at iter ~700)"
    while true; do
        # Both training scripts write ippo_iter_000700.pt or ippo_recurrent_iter_000700.pt
        # at the end. We just poll for absence of both processes touching those dirs.
        if ! pgrep -af "checkpoint-dir checkpoints/$dir_a" > /dev/null \
           && ! pgrep -af "checkpoint-dir checkpoints/$dir_b" > /dev/null; then
            log "$name pair finished"
            return 0
        fi
        sleep 60
    done
}

launch_pair() {
    local label="$1"
    local ff_dir="$2"
    local rec_dir="$3"
    shift 3
    local extra_flags=("$@")

    mkdir -p "logs/${ff_dir}" "logs/${rec_dir}"

    log "launching ${label}-ff (extra: ${extra_flags[*]:-})"
    setsid nohup python3 -m scripts.train_ippo \
        --total-steps 1500000 --n-envs 4 --n-agents 4 \
        --rollout-length 128 --seed 21 \
        --checkpoint-dir "checkpoints/${ff_dir}" \
        --checkpoint-interval 50 --no-wandb \
        "${extra_flags[@]}" \
        > "logs/${ff_dir}/train.log" 2>&1 < /dev/null &
    disown
    local pid_ff=$!
    log "${label}-ff PID=$pid_ff"

    log "launching ${label}-rec (extra: ${extra_flags[*]:-})"
    setsid nohup python3 -m scripts.train_ippo_recurrent \
        --total-steps 1500000 --n-envs 4 --n-agents 4 \
        --rollout-length 128 --hidden-dim 128 --pre-lstm-dim 128 --seed 21 \
        --checkpoint-dir "checkpoints/${rec_dir}" \
        --checkpoint-interval 50 --no-wandb \
        "${extra_flags[@]}" \
        > "logs/${rec_dir}/train.log" 2>&1 < /dev/null &
    disown
    local pid_rec=$!
    log "${label}-rec PID=$pid_rec"
}

log "==== v21 chain launcher started (PID $$) ===="

# Stage 1: wait for v21a (already launched manually).
wait_for_pair "v21a" "ippo_v21a_ff_boss" "ippo_v21a_rec_boss"

# Stage 2: v21b — boss + legacy_stash (legacy_stash is unconditional; only flag is boss).
launch_pair "v21b" "ippo_v21b_ff" "ippo_v21b_rec" --enable-boss
sleep 30
wait_for_pair "v21b" "ippo_v21b_ff" "ippo_v21b_rec"

# Stage 3: v21c — boss + legacy_stash + carry cost.
launch_pair "v21c" "ippo_v21c_ff" "ippo_v21c_rec" --enable-boss --enable-carry-cost
sleep 30
wait_for_pair "v21c" "ippo_v21c_ff" "ippo_v21c_rec"

log "==== v21 chain complete: all 6 runs finished ===="
