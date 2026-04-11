#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPT_CKPT="${GPT_CKPT:-$ROOT_DIR/weights/arpg_300m.pt}"
VQ_CKPT="${VQ_CKPT:-$ROOT_DIR/weights/vq_ds16_c2i.pt}"
SAMPLE_DIR="${SAMPLE_DIR:-$ROOT_DIR/samples/arpgl-baseline}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
PER_PROC_BATCH_SIZE="${PER_PROC_BATCH_SIZE:-8}"
NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-50000}"
CFG_SCALE="${CFG_SCALE:-5.0}"
STEP="${STEP:-64}"
SAMPLE_SCHEDULE="${SAMPLE_SCHEDULE:-arccos}"

if [[ ! -f "$GPT_CKPT" ]]; then
    echo "Missing checkpoint: $GPT_CKPT" >&2
    exit 1
fi

if [[ ! -f "$VQ_CKPT" ]]; then
    echo "Missing VQ checkpoint: $VQ_CKPT" >&2
    exit 1
fi

EXTRA_ARGS=()
if [[ "${ARPG_NO_COMPILE:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--no-compile)
fi
EXTRA_ARGS+=("$@")

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" sample_c2i_ddp.py \
    --gpt-model ARPG-L \
    --gpt-ckpt "$GPT_CKPT" \
    --vq-ckpt "$VQ_CKPT" \
    --sample-schedule "$SAMPLE_SCHEDULE" \
    --cfg-scale "$CFG_SCALE" \
    --step "$STEP" \
    --num-fid-samples "$NUM_FID_SAMPLES" \
    --per-proc-batch-size "$PER_PROC_BATCH_SIZE" \
    --sample-dir "$SAMPLE_DIR" \
    "${EXTRA_ARGS[@]}"
