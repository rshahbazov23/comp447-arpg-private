#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python - <<'PY'
import torch
from models.arpg import ARPG_models

print("Available models:", ", ".join(sorted(ARPG_models.keys())))
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the vanilla ARPG-L baseline run.")
PY
