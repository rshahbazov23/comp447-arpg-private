#!/usr/bin/env bash
# Tier-1 extension to the Phase 1 pilot. Answers three questions cheaply:
#
#   (A) WIDER CAP. Does the mechanism work if we loosen the rejection cap?
#       3 configs: max_prob + tau=0.5 + cap in {0.05, 0.3, 0.5}
#
#   (B) REFINEMENT ABLATION. Does post-hoc re-decoding work where rejection didn't?
#       2 configs: max_prob + refinement_k in {0.1, 0.2}
#
#   (C) STEP-COUNT SENSITIVITY. Does rejection help at different step counts?
#       4 configs: vanilla and max_prob+tau=0.5+cap=0.2 at step counts {16, 64}
#
# Total: 9 configs. ~2.75 hours on A100 (sampling ~2h, FID eval ~45min).
#
# Resumable: skips configs whose NPZ already exists.
#
# Env vars (all optional):
#   GPT_CKPT, VQ_CKPT            paths to the two checkpoints
#   TIER1_DIR                    output root (default: /content/tier1-extension)
#   NUM_SAMPLES                  per-config sample count (default: 10000)
#   PER_PROC_BATCH_SIZE          batch size (default: 32)
#   CFG_SCALE                    classifier-free scale (default: 5.0)
#   GLOBAL_SEED                  run seed (default: 0)

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPT_CKPT="${GPT_CKPT:-$ROOT_DIR/weights/arpg_300m.pt}"
VQ_CKPT="${VQ_CKPT:-$ROOT_DIR/weights/vq_ds16_c2i.pt}"
TIER1_DIR="${TIER1_DIR:-/content/tier1-extension}"
NUM_SAMPLES="${NUM_SAMPLES:-10000}"
PER_PROC_BATCH_SIZE="${PER_PROC_BATCH_SIZE:-32}"
CFG_SCALE="${CFG_SCALE:-5.0}"
GLOBAL_SEED="${GLOBAL_SEED:-0}"

if [[ ! -f "$GPT_CKPT" ]]; then
    echo "Missing checkpoint: $GPT_CKPT" >&2
    exit 1
fi
if [[ ! -f "$VQ_CKPT" ]]; then
    echo "Missing VQ checkpoint: $VQ_CKPT" >&2
    exit 1
fi

VANILLA_DIR="$TIER1_DIR/vanilla"
REJECTION_DIR="$TIER1_DIR/rejection"
REFINEMENT_DIR="$TIER1_DIR/refinement"
LOGS_DIR="$TIER1_DIR/logs"
mkdir -p "$VANILLA_DIR" "$REJECTION_DIR" "$REFINEMENT_DIR" "$LOGS_DIR"

echo "======================================================================"
echo "Tier-1 pilot extension"
echo "  Output root: $TIER1_DIR"
echo "  Samples/config: $NUM_SAMPLES, batch: $PER_PROC_BATCH_SIZE"
echo "  9 configs across 3 experiments (A, B, C)"
echo "======================================================================"

CONFIGS_DONE=0
CONFIGS_SKIPPED=0
CONFIGS_FAILED=0
CONFIG_IDX=0

# Helper: check if a rejection NPZ matching (steps, tau, cap, metric) exists.
rejection_npz_exists() {
    local steps="$1"
    local tau="$2"
    local cap="$3"
    local metric="$4"
    local pattern="ARPG-L-*-step-${steps}-*-mode-rejection-metric-${metric}-tau-${tau}-cap-${cap}.npz"
    compgen -G "$REJECTION_DIR/$pattern" > /dev/null
}

# Helper: check if a refinement NPZ matching (steps, k, metric) exists.
refinement_npz_exists() {
    local steps="$1"
    local k="$2"
    local metric="$3"
    local pattern="ARPG-L-*-step-${steps}-*-mode-refinement-metric-${metric}-k-${k}.npz"
    compgen -G "$REFINEMENT_DIR/$pattern" > /dev/null
}

# Helper: check if a vanilla NPZ matching (steps) exists.
vanilla_npz_exists() {
    local steps="$1"
    local pattern="ARPG-L-*-step-${steps}-*.npz"
    # Only match NPZs WITHOUT -mode- suffix (pure vanilla).
    for f in $(compgen -G "$VANILLA_DIR/$pattern" 2>/dev/null); do
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
    echo "[$CONFIG_IDX/9] vanilla / step=$steps"
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
    local steps="$1"
    local tau="$2"
    local cap="$3"
    local metric="$4"
    CONFIG_IDX=$((CONFIG_IDX + 1))
    echo ""
    echo "------------------------------------------------------------------"
    echo "[$CONFIG_IDX/9] rejection / step=$steps tau=$tau cap=$cap metric=$metric"
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

run_refinement() {
    local steps="$1"
    local k="$2"
    local metric="$3"
    CONFIG_IDX=$((CONFIG_IDX + 1))
    echo ""
    echo "------------------------------------------------------------------"
    echo "[$CONFIG_IDX/9] refinement / step=$steps k=$k metric=$metric"
    echo "------------------------------------------------------------------"
    if refinement_npz_exists "$steps" "$k" "$metric"; then
        echo "  SKIP (NPZ exists)"
        CONFIGS_SKIPPED=$((CONFIGS_SKIPPED + 1)); return
    fi

    local start_ts=$(date +%s)
    torchrun --standalone --nnodes=1 --nproc_per_node=1 sample_c2i_ddp.py \
        --gpt-model ARPG-L --gpt-ckpt "$GPT_CKPT" --vq-ckpt "$VQ_CKPT" \
        --sample-schedule arccos --cfg-scale "$CFG_SCALE" --step "$steps" \
        --num-fid-samples "$NUM_SAMPLES" --per-proc-batch-size "$PER_PROC_BATCH_SIZE" \
        --sample-dir "$REFINEMENT_DIR" --global-seed "$GLOBAL_SEED" --no-compile \
        --rejection-mode refinement \
        --confidence-metric "$metric" --refinement-k "$k"
    local rc=$?
    local elapsed=$(( $(date +%s) - start_ts ))
    if [[ $rc -eq 0 ]]; then echo "  DONE in ${elapsed}s"; CONFIGS_DONE=$((CONFIGS_DONE+1)); else echo "  FAILED (rc=$rc)" >&2; CONFIGS_FAILED=$((CONFIGS_FAILED+1)); fi
}

# =============================================================================
# EXPERIMENT A — Wider cap grid (3 configs)
# =============================================================================
echo ""
echo "### Experiment A: wider cap grid (max_prob, tau=0.5) ###"
for CAP in 0.05 0.3 0.5; do
    run_rejection 32 0.5 "$CAP" max_prob
done

# =============================================================================
# EXPERIMENT B — Refinement ablation (2 configs)
# =============================================================================
echo ""
echo "### Experiment B: refinement ablation (max_prob) ###"
for K in 0.1 0.2; do
    run_refinement 32 "$K" max_prob
done

# =============================================================================
# EXPERIMENT C — Step-count sensitivity (4 configs)
# =============================================================================
echo ""
echo "### Experiment C: step-count sensitivity ###"
for STEPS in 16 64; do
    run_vanilla "$STEPS"
    run_rejection "$STEPS" 0.5 0.2 max_prob
done

echo ""
echo "======================================================================"
echo "Tier-1 extension sampling complete."
echo "  Completed: $CONFIGS_DONE"
echo "  Skipped (already present): $CONFIGS_SKIPPED"
echo "  Failed: $CONFIGS_FAILED"
echo "  Output root: $TIER1_DIR"
echo "======================================================================"
