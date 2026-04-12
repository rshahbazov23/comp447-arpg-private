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

- Cloned the official ARPG repository and set up the reproduction environment
- Created helper scripts for the full reproduction pipeline:
  - `scripts/download_baseline_assets.sh` -- downloads ARPG-L checkpoint (320M params), LlamaGen VQ tokenizer, and ImageNet-256 reference batch (~2.5 GB total)
  - `scripts/run_arpgl_smoke.sh` -- 1k-sample quick sanity check
  - `scripts/run_arpgl_full.sh` -- full 50k-sample baseline run
  - `scripts/eval_arpgl_baseline.sh` -- FID evaluation using OpenAI's guided-diffusion evaluator
  - `scripts/check_repro_env.sh` -- environment verification
- Created pip requirement files (`requirements-repro.txt`, `requirements-eval.txt`)
- Set up a Google Colab Pro notebook (`notebooks/arpgl_baseline_colab.ipynb`) with Google Drive integration for persistent storage of weights and results

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
