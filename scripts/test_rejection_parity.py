"""Parity test for the rejection mechanism.

This is the hard gate: generate_with_rejection(threshold=0.0, max_reject_rate=0.0)
must produce output IDENTICAL to generate() under the same RNG seed. If this fails,
the cache/position plumbing in generate_with_rejection is broken.

Additional checks:
  - Determinism: two runs of generate_with_rejection with same generator match
  - Extreme threshold: threshold=1.01 (nothing ever confident enough) must not
    crash and must still produce seq_len unique positions (final-step override)
  - No NaN / no invalid token ids in output

Usage (on a GPU machine):
    python scripts/test_rejection_parity.py \
        --gpt-ckpt weights/arpg_300m.pt \
        --vq-ckpt weights/vq_ds16_c2i.pt
"""
import argparse
import os
import sys

import torch

# Make `models` importable when run from repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from models.arpg import ARPG_models  # noqa: E402


def load_model(ckpt_path: str, device, dtype):
    model = ARPG_models["ARPG-L"](vocab_size=16384, num_classes=1000).to(device=device, dtype=dtype)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model" in ckpt:
        weights = ckpt["model"]
    elif "module" in ckpt:
        weights = ckpt["module"]
    elif "state_dict" in ckpt:
        weights = ckpt["state_dict"]
    else:
        weights = ckpt
    model.load_state_dict(weights, strict=False)
    model.eval()
    return model


def run_vanilla(model, condition, gen_seed: int, device, **kwargs):
    gen = torch.Generator(device=device).manual_seed(gen_seed)
    return model.generate(condition, generator=gen, **kwargs)


def run_rejection(model, condition, gen_seed: int, device, threshold, max_reject_rate, **kwargs):
    gen = torch.Generator(device=device).manual_seed(gen_seed)
    return model.generate_with_rejection(
        condition,
        generator=gen,
        threshold=threshold,
        max_reject_rate=max_reject_rate,
        confidence_metric="max_prob",
        debug=True,
        **kwargs,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-ckpt", type=str, required=True)
    parser.add_argument("--vq-ckpt", type=str, default=None, help="unused, kept for symmetry with run scripts")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--num-iter", type=int, default=16, help="short for speed; full is 64")
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--order-seed", type=int, default=42)
    parser.add_argument("--class-seed", type=int, default=7)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "parity test requires CUDA"
    device = torch.device("cuda", 0)
    dtype = torch.bfloat16

    print(f"Loading ARPG-L from {args.gpt_ckpt} ...")
    model = load_model(args.gpt_ckpt, device, dtype)

    # Fixed class labels (deterministic)
    torch.manual_seed(args.class_seed)
    condition = torch.randint(0, 1000, (args.num_samples,), device=device)

    common_kwargs = dict(
        guidance_scale=args.guidance_scale,
        cfg_schedule="linear",
        sample_schedule="arccos",
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        seq_len=256,
        num_iter=args.num_iter,
    )

    # ----------------------------------------------------------------------
    # Checkpoint 1: Parity — threshold=0, max_reject=0 must match vanilla.
    # ----------------------------------------------------------------------
    print("\n[1/4] Parity test (threshold=0, max_reject_rate=0) ...")
    out_vanilla = run_vanilla(model, condition, args.order_seed, device, **common_kwargs)
    out_rej_zero = run_rejection(
        model, condition, args.order_seed, device,
        threshold=0.0, max_reject_rate=0.0, **common_kwargs,
    )
    if torch.equal(out_vanilla, out_rej_zero):
        print("  PASS: outputs are bit-identical")
    else:
        diff = (out_vanilla != out_rej_zero).sum().item()
        total = out_vanilla.numel()
        print(f"  FAIL: {diff}/{total} positions differ")
        print(f"  first 10 diffs at indices: "
              f"{(out_vanilla != out_rej_zero).nonzero()[:10].tolist()}")
        sys.exit(1)

    # ----------------------------------------------------------------------
    # Checkpoint 2: Determinism — two rejection runs with same generator match.
    # ----------------------------------------------------------------------
    print("\n[2/4] Determinism test ...")
    out_rej_a = run_rejection(
        model, condition, args.order_seed, device,
        threshold=0.5, max_reject_rate=0.2, **common_kwargs,
    )
    out_rej_b = run_rejection(
        model, condition, args.order_seed, device,
        threshold=0.5, max_reject_rate=0.2, **common_kwargs,
    )
    if torch.equal(out_rej_a, out_rej_b):
        print("  PASS: rejection is deterministic")
    else:
        diff = (out_rej_a != out_rej_b).sum().item()
        print(f"  FAIL: {diff} positions differ between two identical runs")
        sys.exit(1)

    # ----------------------------------------------------------------------
    # Checkpoint 3: Extreme threshold (impossible confidence) — must not crash.
    #   All positions should still be committed via the final-step override.
    # ----------------------------------------------------------------------
    print("\n[3/4] Extreme threshold (tau=1.01) ...")
    try:
        out_extreme = run_rejection(
            model, condition, args.order_seed, device,
            threshold=1.01, max_reject_rate=0.2, **common_kwargs,
        )
    except Exception as e:
        print(f"  FAIL: crashed with {type(e).__name__}: {e}")
        sys.exit(1)
    # Check: no mask-token poison, all values in [0, vocab_size)
    if (out_extreme < 0).any() or (out_extreme >= 16384).any():
        print("  FAIL: output contains invalid token ids (mask-token poison?)")
        sys.exit(1)
    if out_extreme.shape[-1] != 256:
        print(f"  FAIL: expected 256 tokens per sample, got {out_extreme.shape[-1]}")
        sys.exit(1)
    print("  PASS: extreme threshold handled (all 256 positions committed, no invalid tokens)")

    # ----------------------------------------------------------------------
    # Checkpoint 4: Moderate threshold — sanity check on produced tokens.
    # ----------------------------------------------------------------------
    print("\n[4/4] Moderate threshold (tau=0.5, cap=0.2) sanity ...")
    if torch.isnan(out_rej_a.float()).any():
        print("  FAIL: NaN in output")
        sys.exit(1)
    if (out_rej_a < 0).any() or (out_rej_a >= 16384).any():
        print("  FAIL: invalid token ids")
        sys.exit(1)
    if out_rej_a.shape != out_vanilla.shape:
        print(f"  FAIL: shape mismatch {out_rej_a.shape} vs {out_vanilla.shape}")
        sys.exit(1)
    print(f"  PASS: output shape {tuple(out_rej_a.shape)}, tokens in valid range")

    print("\nAll parity checks passed.")


if __name__ == "__main__":
    main()
