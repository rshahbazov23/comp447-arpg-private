#!/usr/bin/env bash
# Phase 1 pilot sweep: 1 vanilla baseline + 18 rejection configs at 32 steps.
#
# Grid (per proposal §4):
#   tau   in {0.3, 0.5, 0.7}
#   cap   in {0.1, 0.2}
#   metric in {max_prob, entropy, margin}
#
# Sampling only — no FID evaluation here. Run scripts/eval_pilot_sweep.py
# afterwards to compute FID-10K on every generated NPZ.
#
# RESUMABLE: if the NPZ for a given config already exists, that config is
# skipped. Safe to re-run if Colab disconnects mid-sweep.
#
# Env vars (all optional):
#   GPT_CKPT              path to arpg_300m.pt   (default: weights/arpg_300m.pt)
#   VQ_CKPT               path to vq_ds16_c2i.pt (default: weights/vq_ds16_c2i.pt)
#   PILOT_DIR             output root            (default: /content/pilot-YYYYMMDD)
#   NUM_SAMPLES           per-config sample count (default: 10000)
#   STEPS                 decoding steps         (default: 32)
#   PER_PROC_BATCH_SIZE   batch size             (default: 32)
#   CFG_SCALE             classifier-free scale  (default: 5.0)
#   GLOBAL_SEED           run seed               (default: 0)

set -uo pipefail   # NOT -e: we continue past individual-config failures.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPT_CKPT="${GPT_CKPT:-$ROOT_DIR/weights/arpg_300m.pt}"
VQ_CKPT="${VQ_CKPT:-$ROOT_DIR/weights/vq_ds16_c2i.pt}"
PILOT_DIR="${PILOT_DIR:-/content/pilot-$(date +%Y%m%d)}"
NUM_SAMPLES="${NUM_SAMPLES:-10000}"
STEPS="${STEPS:-32}"
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

VANILLA_DIR="$PILOT_DIR/vanilla"
REJECTION_DIR="$PILOT_DIR/rejection"
LOGS_DIR="$PILOT_DIR/logs"
mkdir -p "$VANILLA_DIR" "$REJECTION_DIR" "$LOGS_DIR"

echo "======================================================================"
echo "Phase 1 pilot sweep"
echo "  Output root: $PILOT_DIR"
echo "  Samples/config: $NUM_SAMPLES   Steps: $STEPS   Batch: $PER_PROC_BATCH_SIZE"
echo "  Total configs: 1 vanilla + 18 rejection = 19"
echo "======================================================================"

# ---------------------------------------------------------------------------
# Helper: check if a config already has a finished NPZ — if so, skip.
# ---------------------------------------------------------------------------
npz_exists() {
    local dir="$1"
    local count
    count=$(ls "$dir"/*.npz 2>/dev/null | wc -l)
    [[ "$count" -gt 0 ]]
}

# ---------------------------------------------------------------------------
# 1/19: Vanilla baseline at 32 steps
# ---------------------------------------------------------------------------
CONFIGS_DONE=0
CONFIGS_SKIPPED=0
CONFIGS_FAILED=0
CONFIG_IDX=0

run_config() {
    local label="$1"
    local sample_dir="$2"
    local log_json="$3"
    shift 3
    local extra_args=("$@")

    CONFIG_IDX=$((CONFIG_IDX + 1))
    echo ""
    echo "------------------------------------------------------------------"
    echo "[$CONFIG_IDX/19] $label"
    echo "------------------------------------------------------------------"

    # Skip if an NPZ is already present for this config.
    mkdir -p "$sample_dir"
    if npz_exists "$sample_dir"; then
        # Verify one of the existing NPZs matches the expected config signature.
        # A config can have multiple NPZs only across seeds (not our case), so
        # the presence of any .npz in its sample_dir is a sufficient proxy here
        # when we're using a per-config sample_dir for vanilla, and when the
        # folder_name is unique-per-config for rejection. We'll check more
        # precisely inside the run itself, but for now the simple check works.
        local existing
        existing=$(ls "$sample_dir"/*.npz | head -1)
        # For rejection runs, all 18 share the same $REJECTION_DIR. So the
        # "exists" check above is coarse. Do a stricter check using the label.
        if [[ "$sample_dir" == "$VANILLA_DIR" ]]; then
            echo "  SKIP (NPZ exists): $existing"
            CONFIGS_SKIPPED=$((CONFIGS_SKIPPED + 1))
            return 0
        fi
    fi

    local start_ts end_ts
    start_ts=$(date +%s)

    torchrun --standalone --nnodes=1 --nproc_per_node=1 sample_c2i_ddp.py \
        --gpt-model ARPG-L \
        --gpt-ckpt "$GPT_CKPT" \
        --vq-ckpt "$VQ_CKPT" \
        --sample-schedule arccos \
        --cfg-scale "$CFG_SCALE" \
        --step "$STEPS" \
        --num-fid-samples "$NUM_SAMPLES" \
        --per-proc-batch-size "$PER_PROC_BATCH_SIZE" \
        --sample-dir "$sample_dir" \
        --global-seed "$GLOBAL_SEED" \
        --no-compile \
        ${log_json:+--log-json "$log_json"} \
        "${extra_args[@]}"

    local rc=$?
    end_ts=$(date +%s)
    local elapsed=$((end_ts - start_ts))

    if [[ $rc -eq 0 ]]; then
        echo "  DONE in ${elapsed}s"
        CONFIGS_DONE=$((CONFIGS_DONE + 1))
    else
        echo "  FAILED (rc=$rc) after ${elapsed}s" >&2
        CONFIGS_FAILED=$((CONFIGS_FAILED + 1))
    fi
    return $rc
}

# Vanilla at 32 steps (needed for matched-step comparison).
run_config "vanilla / 32 steps" \
    "$VANILLA_DIR" \
    "" \
    --rejection-mode none

# ---------------------------------------------------------------------------
# Stricter skip for rejection configs (all share $REJECTION_DIR so we need to
# check by the config-specific folder_name prefix).
# ---------------------------------------------------------------------------
rejection_npz_exists() {
    local tau="$1"
    local cap="$2"
    local metric="$3"
    local pattern="ARPG-L-*-mode-rejection-metric-${metric}-tau-${tau}-cap-${cap}.npz"
    compgen -G "$REJECTION_DIR/$pattern" > /dev/null
}

# ---------------------------------------------------------------------------
# 18 rejection configs: tau x cap x metric
# ---------------------------------------------------------------------------
for TAU in 0.3 0.5 0.7; do
    for CAP in 0.1 0.2; do
        for METRIC in max_prob entropy margin; do
            LABEL="rejection / tau=$TAU cap=$CAP metric=$METRIC"
            LOG_JSON="$LOGS_DIR/log-tau${TAU}-cap${CAP}-${METRIC}.json"

            CONFIG_IDX=$((CONFIG_IDX + 1))
            echo ""
            echo "------------------------------------------------------------------"
            echo "[$CONFIG_IDX/19] $LABEL"
            echo "------------------------------------------------------------------"

            if rejection_npz_exists "$TAU" "$CAP" "$METRIC"; then
                echo "  SKIP (NPZ exists for this config)"
                CONFIGS_SKIPPED=$((CONFIGS_SKIPPED + 1))
                continue
            fi

            mkdir -p "$REJECTION_DIR"
            start_ts=$(date +%s)

            torchrun --standalone --nnodes=1 --nproc_per_node=1 sample_c2i_ddp.py \
                --gpt-model ARPG-L \
                --gpt-ckpt "$GPT_CKPT" \
                --vq-ckpt "$VQ_CKPT" \
                --sample-schedule arccos \
                --cfg-scale "$CFG_SCALE" \
                --step "$STEPS" \
                --num-fid-samples "$NUM_SAMPLES" \
                --per-proc-batch-size "$PER_PROC_BATCH_SIZE" \
                --sample-dir "$REJECTION_DIR" \
                --global-seed "$GLOBAL_SEED" \
                --no-compile \
                --log-json "$LOG_JSON" \
                --rejection-mode rejection \
                --confidence-metric "$METRIC" \
                --rejection-threshold "$TAU" \
                --max-reject-rate "$CAP"

            rc=$?
            end_ts=$(date +%s)
            elapsed=$((end_ts - start_ts))
            if [[ $rc -eq 0 ]]; then
                echo "  DONE in ${elapsed}s"
                CONFIGS_DONE=$((CONFIGS_DONE + 1))
            else
                echo "  FAILED (rc=$rc) after ${elapsed}s" >&2
                CONFIGS_FAILED=$((CONFIGS_FAILED + 1))
            fi
        done
    done
done

echo ""
echo "======================================================================"
echo "Pilot sweep sampling complete."
echo "  Completed: $CONFIGS_DONE"
echo "  Skipped (already present): $CONFIGS_SKIPPED"
echo "  Failed: $CONFIGS_FAILED"
echo "  Output root: $PILOT_DIR"
echo ""
echo "Next: run 'scripts/eval_pilot_sweep.py --pilot-dir $PILOT_DIR ...'"
echo "======================================================================"
