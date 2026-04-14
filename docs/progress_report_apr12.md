# Progress Report: Confidence-Guided Token Rejection for ARPG
**Date:** April 12, 2026  
**Team:** Riad Shahbazov, Omer Maras, Mohamad Alomar  
**Course:** COMP447 Deep Unsupervised Learning, Spring 2026

---

## 1. Project Summary

We are implementing a confidence-guided token rejection mechanism for ARPG (Li et al., 2026), a visual autoregressive model that generates image tokens in random order with parallel decoding. In each decoding step, multiple tokens are predicted independently against the KV cache. Our modification evaluates prediction confidence after each step and defers low-confidence tokens to later steps where richer context is available, rather than committing uncertain predictions that may pollute the cache for all subsequent steps.

This is an inference-time modification only -- no retraining or architecture changes.

---

## 2. Work Completed (Milestone 1: Baseline Reproduction)

### 2.1 Codebase Setup

The official ARPG repository provides model definitions and a sampling script (`sample_c2i_ddp.py`) but does not include a self-contained reproduction workflow. The README lists raw commands for training and sampling but assumes the user will manually download checkpoints, set up the evaluator, and wire everything together. We built a complete, reproducible pipeline around the existing code so that any team member can go from a fresh clone to a verified FID number in a single session.

**Reproduction scripts.** We created five shell scripts that automate the full workflow:

- `scripts/download_baseline_assets.sh` -- Downloads the three external assets required for evaluation: the ARPG-L pretrained checkpoint (`arpg_300m.pt`, 320M parameters), the LlamaGen VQ tokenizer (`vq_ds16_c2i.pt`) used to decode discrete tokens back into pixel space, and OpenAI's ImageNet-256 reference batch (`VIRTUAL_imagenet256_labeled.npz`) which provides the ground-truth statistics for FID computation. It also clones OpenAI's `guided-diffusion` repository, which contains the standard FID/IS/precision/recall evaluator used by ADM and adopted as the evaluation protocol in the ARPG paper. Total download is approximately 2.5 GB. The script skips files that already exist, making reruns instant.

- `scripts/check_repro_env.sh` -- Verifies that PyTorch can see CUDA and that the ARPG model classes import correctly. This catches environment issues (missing GPU driver, broken dependency) before a long sampling run wastes compute time.

- `scripts/run_arpgl_smoke.sh` -- Generates 1,000 samples (instead of the full 50,000) as a quick sanity check. This takes approximately 8 minutes on a single A100 and verifies the entire sampling pipeline end-to-end: model loading, checkpoint parsing, class-conditional generation with classifier-free guidance, VQ decoding, PNG saving, and NPZ packaging. We use this before every full run to catch configuration errors early.

- `scripts/run_arpgl_full.sh` -- Generates the full 50,000 samples required for FID-50K evaluation. Uses the exact hyperparameters from the ARPG paper: arccos schedule, cfg-scale 5.0, 64 decoding steps, temperature 1.0. All parameters are configurable via environment variables, which will be important when we run experiments with modified decoding in later milestones.

- `scripts/eval_arpgl_baseline.sh` -- Runs the OpenAI guided-diffusion evaluator on the generated NPZ against the ImageNet reference batch. Reports FID, Inception Score, sFID, precision, and recall. The evaluator requires TensorFlow and scipy, which is why we maintain a separate `requirements-eval.txt` to keep the sampling environment clean.

**Dependency management.** We split dependencies into two requirement files because the sampling environment (PyTorch + CUDA) and the evaluation environment (TensorFlow + scipy) can conflict. `requirements-repro.txt` covers the runtime dependencies for model loading and sampling (transformers, einops, Pillow, tqdm, numpy). `requirements-eval.txt` covers the FID evaluator. On Colab, PyTorch and TensorFlow coexist in the same runtime, so this separation is less critical, but it keeps the setup clean for bare-metal Linux machines.

**Colab notebook.** Since our local machines run macOS (no CUDA), we set up a Google Colab Pro notebook (`notebooks/arpgl_baseline_colab.ipynb`) as our primary compute environment. The notebook mounts Google Drive at session start and symlinks the `weights/`, `eval/`, and `external/` directories into the cloned repo. This means the ~2.5 GB of model weights and reference data are downloaded once to Drive and persist across Colab sessions -- reconnecting only requires re-cloning the lightweight repo code (~1 minute) instead of re-downloading everything. Generated results (NPZ files) are also backed up to Drive automatically after sampling completes.

### 2.2 Code Patches

Two small fixes were applied to `sample_c2i_ddp.py`:

1. **`build_class_schedule()` function** -- the original code exhausts class labels when `total_samples > num_fid_samples` in multi-GPU or padded runs. This function tiles the 1000 ImageNet class labels to cover the padded sample count.
2. **`--no-compile` flag** -- added an option to disable `torch.compile`, which can cause issues on certain GPU drivers and Colab environments. Does not affect generated output.

### 2.3 Baseline Reproduction

**Configuration:**
- Model: ARPG-L (320M parameters)
- Hardware: NVIDIA A100-SXM4-40GB (Google Colab Pro)
- Sampling: arccos schedule, cfg-scale 5.0, 64 steps, temperature 1.0, top-k 0, top-p 1.0
- Evaluation: FID-50K against OpenAI's ImageNet-256 reference batch (ADM protocol)

**Results:**

| Metric | Paper (Table 2) | Ours | Match? |
|--------|----------------|------|--------|
| FID (lower is better) | 2.37 | 2.42 | Yes |
| IS (higher is better) | 293.7 | 309.6 | Yes |
| Precision (higher is better) | 0.82 | 0.815 | Yes |
| Recall (higher is better) | 0.55 | 0.563 | Yes |

All metrics are within expected variance. The small FID difference (2.37 vs 2.42) is attributable to single-GPU vs 8-GPU sampling, random seed differences, and torch.compile being disabled. Our IS is higher than the paper reports. **The vanilla ARPG-L baseline is successfully reproduced.**

---

## 3. Repository Structure

```
ARPG-main/
  models/
    arpg.py              # ARPG model definition (Pass-1 + Pass-2 decoder)
    vq_model.py           # LlamaGen VQ tokenizer
  sample_c2i_ddp.py       # Sampling script (patched)
  train_c2i.py            # Training script (not used -- we use pretrained weights)
  scripts/
    download_baseline_assets.sh
    run_arpgl_smoke.sh
    run_arpgl_full.sh
    eval_arpgl_baseline.sh
    check_repro_env.sh
  notebooks/
    arpgl_baseline_colab.ipynb
  docs/
    ARPG_L_BASELINE.md     # Step-by-step reproduction runbook
  requirements-repro.txt
  requirements-eval.txt
```

---

## 4. Next Steps

### Milestone 2: Confidence-Guided Token Rejection (Target: April 20)

The core implementation work involves modifying the `generate()` method in `models/arpg.py`:

1. **Confidence scoring** -- after each Pass-2 forward pass, compute confidence for each predicted token using three metrics:
   - Max softmax probability
   - Predictive entropy: H(p) = -sum(p_i * log(p_i))
   - Top-1 / top-2 margin

2. **Cache admission control** -- tokens below threshold tau are NOT committed to the Pass-1 KV cache. They enter a deferred queue instead of becoming context for future steps.

3. **Deferred token management** -- maintain RoPE positional encodings for deferred tokens so they retain their original spatial position when re-queued. Redistribute deferred tokens to future steps within the fixed schedule budget.

4. **Refinement-pass ablation** -- a simpler variant where vanilla ARPG runs normally, then the K% least confident tokens are re-decoded in a final pass with the full cache. This tests whether the benefit comes from preventing error propagation vs. post-hoc correction.

### Milestone 3: Pilot Experiments (Target: April 27)

- Run rejection on ARPG-L at 32 steps with coarse grid: tau in {0.3, 0.5, 0.7}, max rejection rate in {10%, 20%}, all three confidence metrics (18 configurations)
- Evaluate with FID-10K to identify promising settings

### Milestone 4: Final Experiments (Target: May 3)

- Take 3-5 best configurations from pilot, run full FID-50K at step counts {16, 32, 64}
- Run refinement-pass ablation at K in {10%, 20%} for best confidence metric

---

## 5. Timeline Status

| Activity | Deadline | Status |
|----------|----------|--------|
| Set up codebase and reproduce ARPG-L baseline | April 10 | **Done** (completed April 12) |
| Implement confidence scoring and rejection mechanism | April 20 | Upcoming |
| Run pilot experiments and identify promising settings | April 27 | Upcoming |
| Project progress presentation | April 28-30 | Upcoming |
| Submit progress report | May 3 | Upcoming |
