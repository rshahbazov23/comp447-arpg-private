#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GUIDED_DIFFUSION_DIR="${GUIDED_DIFFUSION_DIR:-$ROOT_DIR/external/guided-diffusion}"
REFERENCE_NPZ="${REFERENCE_NPZ:-$ROOT_DIR/eval/VIRTUAL_imagenet256_labeled.npz}"
SAMPLE_DIR="${SAMPLE_DIR:-$ROOT_DIR/samples/arpgl-baseline}"
CFG_SCALE="${CFG_SCALE:-5.0}"
CFG_SCHEDULE="${CFG_SCHEDULE:-linear}"
SAMPLE_SCHEDULE="${SAMPLE_SCHEDULE:-arccos}"
STEP="${STEP:-64}"
GLOBAL_SEED="${GLOBAL_SEED:-0}"

SAMPLE_NPZ="${SAMPLE_NPZ:-$SAMPLE_DIR/ARPG-L-arpg_300m-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-${CFG_SCALE}-cfg-schedule-${CFG_SCHEDULE}-sample-schedule-${SAMPLE_SCHEDULE}-step-${STEP}-seed-${GLOBAL_SEED}.npz}"

if [[ ! -f "$REFERENCE_NPZ" ]]; then
    echo "Missing reference batch: $REFERENCE_NPZ" >&2
    exit 1
fi

if [[ ! -f "$SAMPLE_NPZ" ]]; then
    echo "Missing sample batch: $SAMPLE_NPZ" >&2
    exit 1
fi

if [[ ! -d "$GUIDED_DIFFUSION_DIR/evaluations" ]]; then
    echo "Missing guided-diffusion checkout: $GUIDED_DIFFUSION_DIR" >&2
    exit 1
fi

cd "$GUIDED_DIFFUSION_DIR/evaluations"
python evaluator.py "$REFERENCE_NPZ" "$SAMPLE_NPZ"
