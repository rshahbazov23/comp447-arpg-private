"""Unit test for the empty-input_ids forward path used by generate_with_refinement.

The refinement pass calls `forward_shared(empty_ids, freqs_cis_refine, num_refine)`
with `empty_ids.shape = (2N, 0)` so that Pass-1 is a no-op (doesn't grow the cache)
and Pass-2 queries attend to the existing cache. That relies on several edge cases:
  - nn.Embedding on a (B, 0) input tensor
  - Attention on a zero-length sequence
  - apply_rotary_emb on a zero-length tensor
  - update_kv_cache concatenating a zero-length tensor

This test verifies the path works end-to-end on a fresh (uncached) model and on
a model with an active cache, producing finite logits of the expected shape.

Usage:
    python scripts/test_empty_ids_forward.py --gpt-ckpt weights/arpg_300m.pt
"""
import argparse
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from models.arpg import ARPG_models  # noqa: E402


def load_model(ckpt_path: str, device, dtype):
    model = ARPG_models["ARPG-L"](vocab_size=16384, num_classes=1000).to(device=device, dtype=dtype)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    weights = ckpt.get("model", ckpt.get("module", ckpt.get("state_dict", ckpt)))
    model.load_state_dict(weights, strict=False)
    model.eval()
    return model


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-ckpt", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--num-refine", type=int, default=8)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "test requires CUDA"
    device = torch.device("cuda", 0)
    dtype = torch.bfloat16

    print(f"Loading ARPG-L from {args.gpt_ckpt} ...")
    model = load_model(args.gpt_ckpt, device, dtype)
    N = args.num_samples
    num_refine = args.num_refine

    # ---- Case 1: empty forward with a FULL cache (simulates end-of-generate state) ----
    print("\n[1/2] Empty-ids forward with populated cache ...")
    # Populate cache by running vanilla generate() first.
    torch.manual_seed(0)
    condition = torch.randint(0, 1000, (N,), device=device)
    gen = torch.Generator(device=device).manual_seed(42)
    _ = model.generate(
        condition,
        guidance_scale=5.0,
        num_iter=16,   # short for speed
        temperature=1.0,
        generator=gen,
    )
    # After generate, setup_kv_cache(False) is called internally → cache is reset.
    # Re-enable cache and re-populate by running again, then STOP before cleanup.
    # Simpler: mimic the refinement flow by inlining the loop up to cache-alive state.
    model.setup_kv_cache(enable=True)
    freqs_cis_ = model.freqs_cis.unsqueeze(0).to(device)
    cond = model.preprocess_condition(condition.clone(), cond_drop_prob=0.0)

    # One step of content processing: put class + a few tokens in the cache.
    input_ids = torch.cat([cond, torch.full_like(cond, model.none_conds_id)], dim=0)
    num_pred = 4
    random_positions = torch.arange(1, 1 + num_pred, device=device, dtype=torch.long)
    cls_pos = torch.arange(0, 1, device=device, dtype=torch.long)
    freqs_cis = torch.cat([
        freqs_cis_[:, cls_pos, ...],
        freqs_cis_[:, random_positions, ...],
    ], dim=1)
    _ = model.forward_shared(input_ids, freqs_cis, num_pred)

    # Now attempt the refinement-style empty-ids forward.
    query_positions = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80], device=device, dtype=torch.long)[:num_refine]
    freqs_cis_refine = freqs_cis_[:, query_positions, ...]
    empty_ids = torch.empty((2 * N, 0), dtype=torch.long, device=device)
    try:
        logits = model.forward_shared(empty_ids, freqs_cis_refine, num_refine)
    except Exception as e:
        print(f"  FAIL: crashed with {type(e).__name__}: {e}")
        sys.exit(1)

    expected_shape = (2 * N, num_refine, 16384)
    if tuple(logits.shape) != expected_shape:
        print(f"  FAIL: expected shape {expected_shape}, got {tuple(logits.shape)}")
        sys.exit(1)
    if not torch.isfinite(logits.float()).all():
        print("  FAIL: non-finite values in logits (NaN/inf)")
        sys.exit(1)
    print(f"  PASS: logits shape {tuple(logits.shape)}, all finite")

    model.setup_kv_cache(enable=False)

    # ---- Case 2: full generate_with_refinement end-to-end ----
    print("\n[2/2] generate_with_refinement end-to-end ...")
    gen = torch.Generator(device=device).manual_seed(42)
    try:
        out = model.generate_with_refinement(
            condition,
            guidance_scale=5.0,
            num_iter=16,
            temperature=1.0,
            refinement_k=0.1,
            confidence_metric="max_prob",
            generator=gen,
        )
    except Exception as e:
        print(f"  FAIL: crashed with {type(e).__name__}: {e}")
        sys.exit(1)

    if out.shape != (N, 256):
        print(f"  FAIL: expected shape {(N, 256)}, got {tuple(out.shape)}")
        sys.exit(1)
    if (out < 0).any() or (out >= 16384).any():
        print("  FAIL: output contains invalid token ids")
        sys.exit(1)
    print(f"  PASS: output shape {tuple(out.shape)}, tokens in valid range")

    # ---- Case 3: refinement_k=0 must be an exact no-op (return vanilla output) ----
    print("\n[3/3] refinement_k=0 short-circuit ...")
    gen = torch.Generator(device=device).manual_seed(42)
    out_vanilla = model.generate(
        condition, guidance_scale=5.0, num_iter=16, temperature=1.0, generator=gen,
    )
    gen = torch.Generator(device=device).manual_seed(42)
    out_k0 = model.generate_with_refinement(
        condition, guidance_scale=5.0, num_iter=16, temperature=1.0,
        refinement_k=0.0, generator=gen,
    )
    if torch.equal(out_vanilla, out_k0):
        print("  PASS: refinement_k=0 is a pure no-op vs vanilla")
    else:
        diff = (out_vanilla != out_k0).sum().item()
        print(f"  FAIL: {diff} positions differ between refinement_k=0 and vanilla")
        sys.exit(1)

    print("\nAll empty-ids path tests passed.")


if __name__ == "__main__":
    main()
