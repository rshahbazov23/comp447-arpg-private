#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="${WEIGHTS_DIR:-$ROOT_DIR/weights}"
EVAL_DIR="${EVAL_DIR:-$ROOT_DIR/eval}"
EXTERNAL_DIR="${EXTERNAL_DIR:-$ROOT_DIR/external}"
GUIDED_DIFFUSION_DIR="${GUIDED_DIFFUSION_DIR:-$EXTERNAL_DIR/guided-diffusion}"

mkdir -p "$WEIGHTS_DIR" "$EVAL_DIR" "$EXTERNAL_DIR"

download_file() {
    local url="$1"
    local output_path="$2"

    if [[ -f "$output_path" ]]; then
        echo "Skipping existing file: $output_path"
        return
    fi

    echo "Downloading $output_path"
    curl -L --fail --progress-bar "$url" -o "$output_path"
}

download_file "https://huggingface.co/hp-l33/ARPG/resolve/main/arpg_300m.pt" "$WEIGHTS_DIR/arpg_300m.pt"
download_file "https://huggingface.co/FoundationVision/LlamaGen/resolve/main/vq_ds16_c2i.pt" "$WEIGHTS_DIR/vq_ds16_c2i.pt"
download_file "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz" "$EVAL_DIR/VIRTUAL_imagenet256_labeled.npz"

if [[ -d "$GUIDED_DIFFUSION_DIR/.git" ]]; then
    echo "Skipping existing repo: $GUIDED_DIFFUSION_DIR"
else
    git clone --depth 1 https://github.com/openai/guided-diffusion.git "$GUIDED_DIFFUSION_DIR"
fi
