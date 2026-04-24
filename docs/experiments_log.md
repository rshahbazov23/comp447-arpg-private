# Experiments Log — COMP547 Token Rejection Project

**Cumulative record of every experimental run.** Updated as new experiments complete. Used as the source of truth when writing the progress report and final paper.

**Team:** Riad Shahbazov, Ömer Maraş, Mohamad Alomar
**Compute environment:** Google Colab Pro, NVIDIA A100-SXM4-40GB
**Model:** ARPG-L (320M parameters), pretrained checkpoint `arpg_300m.pt`
**Evaluator:** OpenAI `guided-diffusion` (ADM protocol)
**Reference batch:** ImageNet-1K 256×256, `VIRTUAL_imagenet256_labeled.npz`

---

## Phase 0 — Vanilla ARPG-L Baseline Reproduction

**Date completed:** April 12, 2026
**Milestone:** 1 (due April 10)

### Configuration

| Setting | Value |
|---------|-------|
| Model | ARPG-L |
| Steps | 64 |
| Samples | 50,000 (FID-50K) |
| Sample schedule | arccos |
| CFG schedule | linear |
| CFG scale | 5.0 |
| Temperature | 1.0 |
| top-k | 0 |
| top-p | 1.0 |
| Precision | bf16 |
| `torch.compile` | disabled |
| Seed | 0 |

### Results

| Metric | Paper (Li et al. 2026) | Our reproduction | Match? |
|--------|-------------------------|-------------------|--------|
| FID-50K ↓ | 2.37 | **2.42** | ✓ |
| Inception Score ↑ | 293.7 | **309.6** | ✓ (exceeded) |
| Precision ↑ | 0.82 | **0.815** | ✓ |
| Recall ↑ | 0.55 | **0.563** | ✓ |
| sFID ↓ | (not in paper table 2) | 5.89 | — |

**Conclusion:** Vanilla baseline successfully reproduced. Small FID gap (0.05) attributed to single-GPU vs 8-GPU sampling and seed differences. IS slightly above paper's number.

---

## Phase 1 — Pilot Sweep at 32 Steps

**Date completed:** April 22, 2026 (approximate)
**Milestone:** 3 (due April 27)

### Setup

- 19 configurations total: 1 vanilla + 18 rejection
- 10,000 samples per config (FID-10K)
- 32 decoding steps
- τ ∈ {0.3, 0.5, 0.7} × max_reject_rate ∈ {0.1, 0.2} × metric ∈ {max_prob, entropy, margin}

### Key results

| Config | FID-10K | Δ vs vanilla |
|--------|---------|--------------|
| Vanilla (32 steps) | **4.846** | — |
| Best rejection (max_prob, τ=0.5, cap=0.2) | **4.830** | **−0.016 (−0.33%)** |
| Best rejection (entropy, τ=0.5, cap=0.2) | ~4.855 | ~+0.009 |
| Best rejection (margin, τ=0.5, cap=0.2) | ~4.904 | ~+0.058 |
| Worst rejection (all at cap=0.1) | ~5.01–5.05 | +0.16 to +0.20 |

**Full results CSV:** `/content/drive/MyDrive/ARPG-assets/results/pilot-20260421/results.csv`

### Key findings

1. **Cap always binding.** At every (τ, cap) combination tested, the `max_reject_rate` cap was reached, meaning τ had zero differentiating effect in this grid. All rejections came from "top-k by confidence" rather than "below threshold."
2. **Metric ordering:** max_prob > entropy > margin (ranked by FID at matched (τ, cap)).
3. **cap=0.1 hurts, cap=0.2 roughly matches vanilla.** The mechanism doesn't significantly help at 32 steps regardless of τ or metric.
4. **Marginal improvement (−0.33%)** is within sampling noise. Cannot be distinguished from no effect at 10K samples.

**Verdict:** 32-step regime is not where the mechanism adds value. Question: is the mechanism fundamentally broken, or is 32 steps the wrong regime to test?

---

## Phase 1b — Tier-1 Pilot Extension

**Date completed:** April 24, 2026
**Purpose:** Decide whether to continue with the rejection idea or pivot to a different approach.

### Setup

Three targeted experiments:

- **Experiment A:** max_prob, τ=0.5, 32 steps, cap ∈ {0.05, 0.3, 0.5} — wider cap grid
- **Experiment B:** refinement ablation at 32 steps, max_prob, K ∈ {0.1, 0.2}
- **Experiment C:** step-count sensitivity — vanilla + rejection (max_prob, τ=0.5, cap=0.2) at steps ∈ {16, 64}

9 new configs, 10,000 samples each.

### Experiment A results (wider cap, 32 steps, max_prob, τ=0.5)

| cap | FID-10K | Δ vs vanilla (4.846) |
|-----|---------|----------------------|
| 0.05 | 4.846 | 0.000 |
| 0.10 | 5.010 | +0.164 |
| 0.20 | 4.830 | −0.016 |
| 0.30 | 4.899 | +0.053 |
| 0.50 | 4.959 | +0.113 |

**Conclusion:** Non-monotonic and noisy. Cap tuning at 32 steps does not produce meaningful improvement. Dead end.

### Experiment B results (refinement ablation, 32 steps, max_prob)

| K | FID-10K | Δ vs vanilla | Δ vs best rejection (4.830) |
|---|---------|--------------|------------------------------|
| 0.10 | 4.945 | +0.098 | +0.114 |
| 0.20 | 5.057 | +0.211 | +0.227 |

**Conclusion:** Refinement actively hurts. **Important negative result** — confirms that the benefit of the rejection idea, if any, must come from *during-generation intervention*, not *post-hoc correction*. This validates the proposal's hypothesis about error propagation through the cache.

### Experiment C results (step-count sensitivity, max_prob, τ=0.5, cap=0.2)

| Steps | Vanilla FID-10K | Rejection FID-10K | Δ |
|-------|------------------|--------------------|---|
| **16** | **6.212** | **5.788** | **−0.424** 🟢 |
| 32 | 4.846 | 4.830 | −0.016 |
| 64 | 4.858 | 4.850 | −0.007 |

**Conclusion:** **BREAKTHROUGH.** At 16 steps, rejection reduces FID by 0.424 (~7% relative improvement). The effect monotonically increases as the decoding regime becomes more aggressive (fewer steps = more parallel tokens per step = more uncertainty for the model).

**Full results CSV:** `/content/drive/MyDrive/ARPG-assets/results/pilot-20260421/tier1-extension/tier1-results.csv`
**Combined CSV:** `/content/drive/MyDrive/ARPG-assets/results/pilot-20260421/tier1-extension/combined-results.csv`

### Phase 1b synthesis

The mechanism **does work**, but only in the aggressive-decoding regime. At 32 and 64 steps the model is already confident enough that rejection finds nothing meaningful to defer. At 16 steps the model is under genuine uncertainty pressure and rejection has signal to exploit.

The refinement ablation gives a clean secondary finding: intervention has to happen DURING generation, not after. Correcting low-confidence tokens post-hoc makes things worse, likely because the cache has already been polluted by those tokens' representations through Pass-1 KV propagation.

**Revised project story:**
> "Confidence-guided token rejection provides substantial FID improvement in the aggressive-decoding regime (16 steps, 7% FID reduction), where parallel-decoding uncertainty is highest. The benefit is specific to during-generation intervention — a controlled ablation (post-hoc refinement) actively hurts quality, confirming that preventing uncertain tokens from entering the KV cache is what matters."

---

## Phase 2 — 16-Step Expansion Sweep *(PENDING)*

**Target date:** Start April 24–25, 2026
**Purpose:** Confirm the 16-step signal is robust, find the best 16-step config, and prepare for FID-50K validation.

### Planned configurations

| Block | Description | # configs |
|-------|-------------|-----------|
| A | max_prob grid: τ ∈ {0.3, 0.5, 0.7} × cap ∈ {0.1, 0.2, 0.3, 0.5} at 16 steps (minus the already-done τ=0.5, cap=0.2) | 11 |
| B | Other metrics at 16 steps, τ=0.5, cap ∈ {0.1, 0.2, 0.5}: entropy (3) + margin (3) | 6 |
| C (optional) | Ultra-aggressive: vanilla + max_prob+τ=0.5+cap=0.2 at 8 and 12 steps | 4 |

Total: 17 configs (21 with the optional ultra-aggressive block).

**Script:** `scripts/run_16step_sweep.sh`
**Notebook:** `notebooks/16step_sweep_colab.ipynb`

### Expected deliverables

1. Best 16-step config identified (winner by FID-10K).
2. Evidence the 16-step effect is robust across metrics/τ/cap (not an artefact of one lucky config).
3. Expected trend: effect may strengthen at 8–12 steps (if ultra-aggressive block is run).
4. Winning config → FID-50K validation → progress report headline number.

---

## Phase 3 — FID-50K Validation on Top Configs *(PENDING)*

**Target date:** April 27 – May 1, 2026
**Purpose:** Replace the 10K-sample FID estimates with paper-grade 50K-sample numbers for the top 1-3 configs identified in Phase 2.

Matched-wall-clock comparison: rejection at 16 steps vs vanilla at some equivalent step count for a fair compute-vs-quality tradeoff claim.

---

## Phase 4 — Progress Report *(DUE MAY 3)*

Deliverable for the May 3 milestone.

---

## Raw artefact index

| Location | Contents |
|----------|----------|
| `pilot-20260421/results.csv` on Drive | Phase 1 pilot, 19 rows (32 steps) |
| `pilot-20260421/logs/` on Drive | Per-config rejection JSON logs + heatmaps (Phase 1) |
| `pilot-20260421/tier1-extension/tier1-results.csv` | Phase 1b, 9 rows |
| `pilot-20260421/tier1-extension/combined-results.csv` | Merged Phase 1 + Phase 1b, 28 rows |
| `pilot-20260421/tier1-extension/logs/` | Per-config logs + heatmaps (Phase 1b) |

---

## Rolling list of conclusions

*Every entry is a finding we're reasonably confident about at the time it was added. Update if later evidence overturns it.*

1. **[April 12]** Vanilla ARPG-L reproduces paper results to within reasonable seed variance.
2. **[April 22]** At 32 steps, the `max_reject_rate` cap dominates — τ has no visible effect because the cap always binds.
3. **[April 22]** max_prob is the best of the three confidence metrics at 32 steps.
4. **[April 24]** Post-hoc refinement (K=0.1, 0.2) consistently hurts quality — intervention must happen DURING generation.
5. **[April 24]** Rejection strongly benefits the aggressive regime: at 16 steps, FID drops 0.424 (6.212 → 5.788). At 32+ steps, the effect is indistinguishable from noise.
6. **[April 24]** The project's primary claim is now compute-efficiency in the low-step regime, not general FID improvement at standard step counts.

*(Next planned updates after the 16-step sweep.)*
