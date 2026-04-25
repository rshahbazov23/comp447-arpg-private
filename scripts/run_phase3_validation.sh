#!/usr/bin/env bash
# Phase 3 — FID-50K validation of the Phase 2 winner.
#
# Replaces the noisy FID-10K estimates with paper-grade FID-50K numbers.
#
# IMPORTANT I/O DESIGN:
#   PNGs (50,000 small files per config) -> LOCAL disk during sampling.
#   Final NPZ (one big file) -> sync to DRIVE after each config completes.
#   This avoids the ~30x slowdown from writing 50K PNGs directly to Drive.
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
# Resumable: checks Drive (the persistent location) for existing NPZs.
#
# Env vars:
#   GPT_CKPT, VQ_CKPT
#   DRIVE_DIR              persistent NPZ output root (Drive path)
#   LOCAL_DIR              fast scratch for PNGs (default: /content/phase3-local)
#   NUM_SAMPLES            default 50000
#   PER_PROC_BATCH_SIZE    default 64
#   CFG_SCALE              default 5.0
#   GLOBAL_SEED            default 0
#   RUN_BONUS              "1" to run configs 3-5 (default), "0" to skip
#   KEEP_LOCAL_PNGS        "1" to keep PNG folders on local disk after NPZ
#                          is built (default 0 = delete to save disk space)

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPT_CKPT="${GPT_CKPT:-$ROOT_DIR/weights/arpg_300m.pt}"
VQ_CKPT="${VQ_CKPT:-$ROOT_DIR/weights/vq_ds16_c2i.pt}"
DRIVE_DIR="${DRIVE_DIR:-${PHASE3_DIR:-/content/drive/MyDrive/ARPG-assets/results/pilot-20260421/phase3-fid50k}}"
LOCAL_DIR="${LOCAL_DIR:-/content/phase3-local}"
NUM_SAMPLES="${NUM_SAMPLES:-50000}"
PER_PROC_BATCH_SIZE="${PER_PROC_BATCH_SIZE:-64}"
CFG_SCALE="${CFG_SCALE:-5.0}"
GLOBAL_SEED="${GLOBAL_SEED:-0}"
RUN_BONUS="${RUN_BONUS:-1}"
KEEP_LOCAL_PNGS="${KEEP_LOCAL_PNGS:-0}"

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

# Drive directories (persistent)
DRIVE_VANILLA="$DRIVE_DIR/vanilla"
DRIVE_REJECTION="$DRIVE_DIR/rejection"
DRIVE_LOGS="$DRIVE_DIR/logs"
mkdir -p "$DRIVE_VANILLA" "$DRIVE_REJECTION" "$DRIVE_LOGS"

# Local directories (scratch, fast I/O)
LOCAL_VANILLA="$LOCAL_DIR/vanilla"
LOCAL_REJECTION="$LOCAL_DIR/rejection"
LOCAL_LOGS="$LOCAL_DIR/logs"
mkdir -p "$LOCAL_VANILLA" "$LOCAL_REJECTION" "$LOCAL_LOGS"

if [[ "$RUN_BONUS" == "1" ]]; then
    TOTAL=5
else
    TOTAL=2
fi

echo "======================================================================"
echo "Phase 3 - FID-50K validation"
echo "  Drive (persistent):   $DRIVE_DIR"
echo "  Local (scratch PNGs): $LOCAL_DIR"
echo "  Samples/config:       $NUM_SAMPLES"
echo "  Batch size:           $PER_PROC_BATCH_SIZE"
echo "  Bonus runs:           $([[ "$RUN_BONUS" == "1" ]] && echo "YES (configs 3-5)" || echo "no (only 1-2)")"
echo "  Total configs:        $TOTAL"
echo "  Headline rej:         metric=$METRIC tau=$TAU cap=$CAP"
echo "======================================================================"

CONFIGS_DONE=0
CONFIGS_SKIPPED=0
CONFIGS_FAILED=0
CONFIG_IDX=0

# -------- Drive existence checks (resumability) --------
rejection_npz_exists_on_drive() {
    local steps="$1" tau="$2" cap="$3" metric="$4"
    compgen -G "$DRIVE_REJECTION/ARPG-L-*-step-${steps}-*-mode-rejection-metric-${metric}-tau-${tau}-cap-${cap}.npz" > /dev/null
}

vanilla_npz_exists_on_drive() {
    local steps="$1"
    for f in $(compgen -G "$DRIVE_VANILLA/ARPG-L-*-step-${steps}-*.npz" 2>/dev/null); do
        if [[ "$f" != *"-mode-"* ]]; then
            return 0
        fi
    done
    return 1
}

# -------- Sync newly-created local NPZ to Drive --------
sync_npz_to_drive() {
    local sub="$1"   # "vanilla" or "rejection"
    local local_dir="$LOCAL_DIR/$sub"
    local drive_dir="$DRIVE_DIR/$sub"
    mkdir -p "$drive_dir"

    for npz in "$local_dir"/*.npz; do
        [[ -f "$npz" ]] || continue
        local basename=$(basename "$npz")
        if [[ ! -f "$drive_dir/$basename" ]]; then
            echo "  Syncing NPZ to Drive: $basename"
            local sync_start=$(date +%s)
            cp "$npz" "$drive_dir/$basename"
            local sync_elapsed=$(( $(date +%s) - sync_start ))
            echo "  Synced in ${sync_elapsed}s"
        fi
    done
}

# -------- Sync rejection logs and heatmaps to Drive --------
sync_logs_to_drive() {
    cp "$LOCAL_LOGS"/*.json "$DRIVE_LOGS"/ 2>/dev/null || true
    cp "$LOCAL_LOGS"/*_heatmap.png "$DRIVE_LOGS"/ 2>/dev/null || true
}

# -------- Optional cleanup: remove the 50K PNGs to save local disk --------
maybe_cleanup_local_pngs() {
    local sub="$1"
    if [[ "$KEEP_LOCAL_PNGS" == "1" ]]; then
        return
    fi
    for d in "$LOCAL_DIR/$sub"/ARPG-L-*/; do
        [[ -d "$d" ]] || continue
        rm -rf "$d"
    done
}

# =============================================================================
# Run helpers
# =============================================================================

run_vanilla() {
    local steps="$1"
    CONFIG_IDX=$((CONFIG_IDX + 1))
    echo ""
    echo "------------------------------------------------------------------"
    echo "[$CONFIG_IDX/$TOTAL] vanilla / step=$steps / FID-50K"
    echo "------------------------------------------------------------------"
    if vanilla_npz_exists_on_drive "$steps"; then
        echo "  SKIP (NPZ already in Drive)"
        CONFIGS_SKIPPED=$((CONFIGS_SKIPPED + 1)); return
    fi

    local start_ts=$(date +%s)
    torchrun --standalone --nnodes=1 --nproc_per_node=1 sample_c2i_ddp.py \
        --gpt-model ARPG-L --gpt-ckpt "$GPT_CKPT" --vq-ckpt "$VQ_CKPT" \
        --sample-schedule arccos --cfg-scale "$CFG_SCALE" --step "$steps" \
        --num-fid-samples "$NUM_SAMPLES" --per-proc-batch-size "$PER_PROC_BATCH_SIZE" \
        --sample-dir "$LOCAL_VANILLA" --global-seed "$GLOBAL_SEED" --no-compile \
        --rejection-mode none
    local rc=$?
    local elapsed=$(( $(date +%s) - start_ts ))

    if [[ $rc -eq 0 ]]; then
        echo "  Sampling DONE in ${elapsed}s"
        sync_npz_to_drive vanilla
        maybe_cleanup_local_pngs vanilla
        CONFIGS_DONE=$((CONFIGS_DONE+1))
    else
        echo "  FAILED (rc=$rc)" >&2
        CONFIGS_FAILED=$((CONFIGS_FAILED+1))
    fi
}

run_rejection() {
    local steps="$1" tau="$2" cap="$3" metric="$4"
    CONFIG_IDX=$((CONFIG_IDX + 1))
    echo ""
    echo "------------------------------------------------------------------"
    echo "[$CONFIG_IDX/$TOTAL] rejection / step=$steps tau=$tau cap=$cap metric=$metric / FID-50K"
    echo "------------------------------------------------------------------"
    if rejection_npz_exists_on_drive "$steps" "$tau" "$cap" "$metric"; then
        echo "  SKIP (NPZ already in Drive)"
        CONFIGS_SKIPPED=$((CONFIGS_SKIPPED + 1)); return
    fi

    local log_json="$LOCAL_LOGS/log-step${steps}-tau${tau}-cap${cap}-${metric}.json"
    local start_ts=$(date +%s)
    torchrun --standalone --nnodes=1 --nproc_per_node=1 sample_c2i_ddp.py \
        --gpt-model ARPG-L --gpt-ckpt "$GPT_CKPT" --vq-ckpt "$VQ_CKPT" \
        --sample-schedule arccos --cfg-scale "$CFG_SCALE" --step "$steps" \
        --num-fid-samples "$NUM_SAMPLES" --per-proc-batch-size "$PER_PROC_BATCH_SIZE" \
        --sample-dir "$LOCAL_REJECTION" --global-seed "$GLOBAL_SEED" --no-compile \
        --log-json "$log_json" \
        --rejection-mode rejection \
        --confidence-metric "$metric" --rejection-threshold "$tau" --max-reject-rate "$cap"
    local rc=$?
    local elapsed=$(( $(date +%s) - start_ts ))

    if [[ $rc -eq 0 ]]; then
        echo "  Sampling DONE in ${elapsed}s"
        sync_npz_to_drive rejection
        sync_logs_to_drive
        maybe_cleanup_local_pngs rejection
        CONFIGS_DONE=$((CONFIGS_DONE+1))
    else
        echo "  FAILED (rc=$rc)" >&2
        CONFIGS_FAILED=$((CONFIGS_FAILED+1))
    fi
}

# =============================================================================
# PRIORITY runs (headline numbers)
# =============================================================================
echo ""
echo "### Priority runs (headline numbers) ###"
run_vanilla 16
run_rejection 16 "$TAU" "$CAP" "$METRIC"

# =============================================================================
# BONUS runs
# =============================================================================
if [[ "$RUN_BONUS" == "1" ]]; then
    echo ""
    echo "### Bonus runs ###"
    run_vanilla 32
    run_vanilla 8
    run_rejection 8 "$TAU" "$CAP" "$METRIC"
fi

echo ""
echo "======================================================================"
echo "Phase 3 sampling complete."
echo "  Completed: $CONFIGS_DONE"
echo "  Skipped (already on Drive): $CONFIGS_SKIPPED"
echo "  Failed: $CONFIGS_FAILED"
echo "  Drive root: $DRIVE_DIR"
echo "======================================================================"
