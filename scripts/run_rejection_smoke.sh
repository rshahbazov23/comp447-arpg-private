#!/usr/bin/env bash
# Quick smoke test for generate_with_rejection: 16 samples with debug asserts on.
# Expected runtime: ~60-90s on A100.
#
# Produces:
#   - samples/rejection-smoke/... /NNNNNN.png (16 images)
#   - samples/rejection-smoke/log.json (per-step tracker output)
#   - samples/rejection-smoke/log_heatmap.png (16x16 rejection heatmap)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPT_CKPT="${GPT_CKPT:-$ROOT_DIR/weights/arpg_300m.pt}"
VQ_CKPT="${VQ_CKPT:-$ROOT_DIR/weights/vq_ds16_c2i.pt}"
SAMPLE_DIR="${SAMPLE_DIR:-$ROOT_DIR/samples/rejection-smoke}"
LOG_JSON="${LOG_JSON:-$SAMPLE_DIR/log.json}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

# Rejection knobs
THRESHOLD="${THRESHOLD:-0.5}"
MAX_REJECT_RATE="${MAX_REJECT_RATE:-0.2}"
CONFIDENCE_METRIC="${CONFIDENCE_METRIC:-max_prob}"
GLOBAL_SEED="${GLOBAL_SEED:-0}"

if [[ ! -f "$GPT_CKPT" ]]; then
    echo "Missing checkpoint: $GPT_CKPT" >&2
    exit 1
fi
if [[ ! -f "$VQ_CKPT" ]]; then
    echo "Missing VQ checkpoint: $VQ_CKPT" >&2
    exit 1
fi

mkdir -p "$SAMPLE_DIR"

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" sample_c2i_ddp.py \
    --gpt-model ARPG-L \
    --gpt-ckpt "$GPT_CKPT" \
    --vq-ckpt "$VQ_CKPT" \
    --sample-schedule arccos \
    --cfg-scale 5.0 \
    --step 64 \
    --num-fid-samples 16 \
    --per-proc-batch-size 16 \
    --sample-dir "$SAMPLE_DIR" \
    --global-seed "$GLOBAL_SEED" \
    --no-compile \
    --rejection-mode rejection \
    --confidence-metric "$CONFIDENCE_METRIC" \
    --rejection-threshold "$THRESHOLD" \
    --max-reject-rate "$MAX_REJECT_RATE" \
    --debug \
    --log-json "$LOG_JSON" \
    "$@"

echo ""
echo "Smoke test complete."
echo "  - images + NPZ under: $SAMPLE_DIR"
echo "  - rejection log:      $LOG_JSON"
echo "  - rejection heatmap:  ${LOG_JSON%.json}_heatmap.png"
