#!/usr/bin/env bash
# Phase 2: 16-step expansion sweep.
#
# The Tier-1 extension showed a strong signal at 16 steps (rejection gives
# -0.424 FID vs vanilla). This sweep confirms that finding is robust across
# (metric, tau, cap) and finds the best 16-step config for FID-50K validation.
#
# Configurations (total: 17 new configs, or 21 with the optional block):
#
#   Block A — max_prob grid at 16 steps:
#       tau in {0.3, 0.5, 0.7} x cap in {0.1, 0.2, 0.3, 0.5}
#       = 12 configs (1 already done in Tier-1: tau=0.5, cap=0.2) => 11 new
#
#   Block B — other metrics at 16 steps, tau=0.5, cap in {0.1, 0.2, 0.5}:
#       entropy (3) + margin (3) = 6 new configs
#
#   Block C (optional) — ultra-aggressive step counts, max_prob, tau=0.5, cap=0.2:
#       vanilla at steps in {8, 12} (2 configs)
#       rejection at steps in {8, 12} (2 configs)
#       = 4 new configs
#
# Resumable: skips any config whose NPZ is already present.
#
# Env vars:
#   RUN_AGGRESSIVE_STEPS=1  also run block C (default: 0, i.e., skip)
#   ALL standard vars from the other sweeps.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPT_CKPT="${GPT_CKPT:-$ROOT_DIR/weights/arpg_300m.pt}"
VQ_CKPT="${VQ_CKPT:-$ROOT_DIR/weights/vq_ds16_c2i.pt}"
SWEEP_DIR="${SWEEP_DIR:-/content/sweep-16step}"
NUM_SAMPLES="${NUM_SAMPLES:-10000}"
PER_PROC_BATCH_SIZE="${PER_PROC_BATCH_SIZE:-32}"
CFG_SCALE="${CFG_SCALE:-5.0}"
GLOBAL_SEED="${GLOBAL_SEED:-0}"
RUN_AGGRESSIVE_STEPS="${RUN_AGGRESSIVE_STEPS:-0}"

if [[ ! -f "$GPT_CKPT" ]]; then
    echo "Missing checkpoint: $GPT_CKPT" >&2
    exit 1
fi
if [[ ! -f "$VQ_CKPT" ]]; then
    echo "Missing VQ checkpoint: $VQ_CKPT" >&2
    exit 1
fi

VANILLA_DIR="$SWEEP_DIR/vanilla"
REJECTION_DIR="$SWEEP_DIR/rejection"
LOGS_DIR="$SWEEP_DIR/logs"
mkdir -p "$VANILLA_DIR" "$REJECTION_DIR" "$LOGS_DIR"

# Count configs based on whether block C is requested.
if [[ "$RUN_AGGRESSIVE_STEPS" == "1" ]]; then
    TOTAL_CONFIGS=21
else
    TOTAL_CONFIGS=17
fi

echo "======================================================================"
echo "Phase 2: 16-step expansion sweep"
echo "  Output root:       $SWEEP_DIR"
echo "  Samples / config:  $NUM_SAMPLES"
echo "  Batch size:        $PER_PROC_BATCH_SIZE"
echo "  Aggressive steps:  $([[ "$RUN_AGGRESSIVE_STEPS" == "1" ]] && echo "YES (block C)" || echo "no")"
echo "  Total configs:     $TOTAL_CONFIGS"
echo "======================================================================"

CONFIGS_DONE=0
CONFIGS_SKIPPED=0
CONFIGS_FAILED=0
CONFIG_IDX=0

# -------- helpers --------
rejection_npz_exists() {
    local steps="$1" tau="$2" cap="$3" metric="$4"
    compgen -G "$REJECTION_DIR/ARPG-L-*-step-${steps}-*-mode-rejection-metric-${metric}-tau-${tau}-cap-${cap}.npz" > /dev/null
}

vanilla_npz_exists() {
    local steps="$1"
    for f in $(compgen -G "$VANILLA_DIR/ARPG-L-*-step-${steps}-*.npz" 2>/dev/null); do
        if [[ "$f" != *"-mode-"* ]]; then
            return 0
        fi
    done
    return 1
}

run_vanilla() {
    local steps="$1"
    CONFIG_IDX=$((CONFIG_IDX + 1))
    echo ""
    echo "------------------------------------------------------------------"
    echo "[$CONFIG_IDX/$TOTAL_CONFIGS] vanilla / step=$steps"
    echo "------------------------------------------------------------------"
    if vanilla_npz_exists "$steps"; then
        echo "  SKIP (NPZ exists)"
        CONFIGS_SKIPPED=$((CONFIGS_SKIPPED + 1)); return
    fi
    local start_ts=$(date +%s)
    torchrun --standalone --nnodes=1 --nproc_per_node=1 sample_c2i_ddp.py \
        --gpt-model ARPG-L --gpt-ckpt "$GPT_CKPT" --vq-ckpt "$VQ_CKPT" \
        --sample-schedule arccos --cfg-scale "$CFG_SCALE" --step "$steps" \
        --num-fid-samples "$NUM_SAMPLES" --per-proc-batch-size "$PER_PROC_BATCH_SIZE" \
        --sample-dir "$VANILLA_DIR" --global-seed "$GLOBAL_SEED" --no-compile \
        --rejection-mode none
    local rc=$?
    local elapsed=$(( $(date +%s) - start_ts ))
    if [[ $rc -eq 0 ]]; then echo "  DONE in ${elapsed}s"; CONFIGS_DONE=$((CONFIGS_DONE+1)); else echo "  FAILED (rc=$rc)" >&2; CONFIGS_FAILED=$((CONFIGS_FAILED+1)); fi
}

run_rejection() {
    local steps="$1" tau="$2" cap="$3" metric="$4"
    CONFIG_IDX=$((CONFIG_IDX + 1))
    echo ""
    echo "------------------------------------------------------------------"
    echo "[$CONFIG_IDX/$TOTAL_CONFIGS] rejection / step=$steps tau=$tau cap=$cap metric=$metric"
    echo "------------------------------------------------------------------"
    if rejection_npz_exists "$steps" "$tau" "$cap" "$metric"; then
        echo "  SKIP (NPZ exists)"
        CONFIGS_SKIPPED=$((CONFIGS_SKIPPED + 1)); return
    fi
    local log_json="$LOGS_DIR/log-step${steps}-tau${tau}-cap${cap}-${metric}.json"
    local start_ts=$(date +%s)
    torchrun --standalone --nnodes=1 --nproc_per_node=1 sample_c2i_ddp.py \
        --gpt-model ARPG-L --gpt-ckpt "$GPT_CKPT" --vq-ckpt "$VQ_CKPT" \
        --sample-schedule arccos --cfg-scale "$CFG_SCALE" --step "$steps" \
        --num-fid-samples "$NUM_SAMPLES" --per-proc-batch-size "$PER_PROC_BATCH_SIZE" \
        --sample-dir "$REJECTION_DIR" --global-seed "$GLOBAL_SEED" --no-compile \
        --log-json "$log_json" \
        --rejection-mode rejection \
        --confidence-metric "$metric" --rejection-threshold "$tau" --max-reject-rate "$cap"
    local rc=$?
    local elapsed=$(( $(date +%s) - start_ts ))
    if [[ $rc -eq 0 ]]; then echo "  DONE in ${elapsed}s"; CONFIGS_DONE=$((CONFIGS_DONE+1)); else echo "  FAILED (rc=$rc)" >&2; CONFIGS_FAILED=$((CONFIGS_FAILED+1)); fi
}

# =============================================================================
# BLOCK A — max_prob grid at 16 steps (12 configs, 1 already in Tier-1 dir)
# =============================================================================
echo ""
echo "### Block A: max_prob grid at 16 steps ###"
for TAU in 0.3 0.5 0.7; do
    for CAP in 0.1 0.2 0.3 0.5; do
        run_rejection 16 "$TAU" "$CAP" max_prob
    done
done

# =============================================================================
# BLOCK B — entropy and margin at 16 steps (6 configs)
# =============================================================================
echo ""
echo "### Block B: entropy & margin at 16 steps, tau=0.5 ###"
for METRIC in entropy margin; do
    for CAP in 0.1 0.2 0.5; do
        run_rejection 16 0.5 "$CAP" "$METRIC"
    done
done

# =============================================================================
# BLOCK C — ultra-aggressive step counts (4 configs, OPTIONAL)
# =============================================================================
if [[ "$RUN_AGGRESSIVE_STEPS" == "1" ]]; then
    echo ""
    echo "### Block C: ultra-aggressive step counts (8 and 12 steps) ###"
    for STEPS in 8 12; do
        run_vanilla "$STEPS"
        run_rejection "$STEPS" 0.5 0.2 max_prob
    done
fi

echo ""
echo "======================================================================"
echo "16-step sweep sampling complete."
echo "  Completed: $CONFIGS_DONE"
echo "  Skipped (already present): $CONFIGS_SKIPPED"
echo "  Failed: $CONFIGS_FAILED"
echo "  Output root: $SWEEP_DIR"
echo "======================================================================"
