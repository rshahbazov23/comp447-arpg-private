#!/usr/bin/env bash
# Phase 3 — FID-50K validation of the Phase 2 winner.
#
# Replaces the noisy FID-10K estimates with paper-grade FID-50K numbers for
# the headline configs. NPZs are written directly to Drive (slower but
# survives Colab disconnects).
#
# Configurations (5 runs):
#
#   PRIORITY (headline numbers):
#     1. vanilla, 16 steps, FID-50K        - matched-step baseline
#     2. margin tau=0.5 cap=0.5, 16 steps  - the headline winner
#
#   BONUS (strengthens the paper):
#     3. vanilla, 32 steps, FID-50K        - "approaches 32-step quality" claim
#     4. vanilla, 8 steps, FID-50K         - aggressive baseline
#     5. margin tau=0.5 cap=0.5, 8 steps   - validate the 17.6% claim at 50K
#
# Resumable: skips configs whose NPZ already exists.
#
# Env vars:
#   GPT_CKPT, VQ_CKPT
#   PHASE3_DIR             output root (must be on Drive for persistence)
#   NUM_SAMPLES            default 50000
#   PER_PROC_BATCH_SIZE    default 64 (A100 has plenty of VRAM)
#   CFG_SCALE              default 5.0
#   GLOBAL_SEED            default 0
#   RUN_BONUS              "1" to run configs 3-5 (default), "0" to skip

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPT_CKPT="${GPT_CKPT:-$ROOT_DIR/weights/arpg_300m.pt}"
VQ_CKPT="${VQ_CKPT:-$ROOT_DIR/weights/vq_ds16_c2i.pt}"
PHASE3_DIR="${PHASE3_DIR:-/content/drive/MyDrive/ARPG-assets/results/pilot-20260421/phase3-fid50k}"
NUM_SAMPLES="${NUM_SAMPLES:-50000}"
PER_PROC_BATCH_SIZE="${PER_PROC_BATCH_SIZE:-64}"
CFG_SCALE="${CFG_SCALE:-5.0}"
GLOBAL_SEED="${GLOBAL_SEED:-0}"
RUN_BONUS="${RUN_BONUS:-1}"

# Headline rejection config from Phase 2:
METRIC="margin"
TAU="0.5"
CAP="0.5"

if [[ ! -f "$GPT_CKPT" ]]; then
    echo "Missing checkpoint: $GPT_CKPT" >&2
    exit 1
fi
if [[ ! -f "$VQ_CKPT" ]]; then
    echo "Missing VQ checkpoint: $VQ_CKPT" >&2
    exit 1
fi

VANILLA_DIR="$PHASE3_DIR/vanilla"
REJECTION_DIR="$PHASE3_DIR/rejection"
LOGS_DIR="$PHASE3_DIR/logs"
mkdir -p "$VANILLA_DIR" "$REJECTION_DIR" "$LOGS_DIR"

if [[ "$RUN_BONUS" == "1" ]]; then
    TOTAL=5
else
    TOTAL=2
fi

echo "======================================================================"
echo "Phase 3 - FID-50K validation"
echo "  Output root:    $PHASE3_DIR"
echo "  Samples/config: $NUM_SAMPLES"
echo "  Batch size:     $PER_PROC_BATCH_SIZE"
echo "  Bonus runs:     $([[ "$RUN_BONUS" == "1" ]] && echo "YES (configs 3-5)" || echo "no (only 1-2)")"
echo "  Total configs:  $TOTAL"
echo "  Headline rej:   metric=$METRIC tau=$TAU cap=$CAP"
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
    echo "[$CONFIG_IDX/$TOTAL] vanilla / step=$steps / FID-50K"
    echo "------------------------------------------------------------------"
    if vanilla_npz_exists "$steps"; then
        echo "  SKIP (NPZ already in $VANILLA_DIR)"
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
    echo "[$CONFIG_IDX/$TOTAL] rejection / step=$steps tau=$tau cap=$cap metric=$metric / FID-50K"
    echo "------------------------------------------------------------------"
    if rejection_npz_exists "$steps" "$tau" "$cap" "$metric"; then
        echo "  SKIP (NPZ already in $REJECTION_DIR)"
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
# PRIORITY — the two runs the progress report MUST have
# =============================================================================
echo ""
echo "### Priority runs (headline numbers) ###"
run_vanilla 16
run_rejection 16 "$TAU" "$CAP" "$METRIC"

# =============================================================================
# BONUS — strengthens the paper, optional but recommended
# =============================================================================
if [[ "$RUN_BONUS" == "1" ]]; then
    echo ""
    echo "### Bonus runs (compute-efficiency claim + ultra-aggressive validation) ###"
    run_vanilla 32
    run_vanilla 8
    run_rejection 8 "$TAU" "$CAP" "$METRIC"
fi

echo ""
echo "======================================================================"
echo "Phase 3 sampling complete."
echo "  Completed: $CONFIGS_DONE"
echo "  Skipped (already present): $CONFIGS_SKIPPED"
echo "  Failed: $CONFIGS_FAILED"
echo "  Output root: $PHASE3_DIR"
echo "======================================================================"
