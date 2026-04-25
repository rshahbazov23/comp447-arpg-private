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

## Phase 2 — 16-Step Expansion Sweep

**Date completed:** April 25, 2026
**Compute used:** ~4 hours on A100 (sampling) + ~90 min (FID eval) = ~5.5 hours total
**Configs run:** 22 (block A + B + C all completed, including aggressive 8/12-step block)

### Setup

- 10,000 samples per config (FID-10K)
- Same arccos schedule, CFG 5.0, temperature 1.0, seed 0 as previous phases

### Block A results — max_prob (τ × cap) grid at 16 steps

| FID-10K | cap=0.1 | cap=0.2 | cap=0.3 | cap=0.5 |
|---------|---------|---------|---------|---------|
| τ=0.3 | 5.917 | 5.788 | 5.583 | **5.451** |
| τ=0.5 | 5.917 | 5.788 | 5.583 | **5.451** |
| τ=0.7 | 5.917 | 5.788 | 5.584 | **5.451** |

Δ vs vanilla (FID 6.212):

| Δ | cap=0.1 | cap=0.2 | cap=0.3 | cap=0.5 |
|---|---------|---------|---------|---------|
| τ=0.3 | −0.296 | −0.424 | −0.629 | **−0.762** |
| τ=0.5 | −0.296 | −0.424 | −0.629 | **−0.762** |
| τ=0.7 | −0.296 | −0.424 | −0.629 | **−0.762** |

**Findings:**

1. **τ has zero effect** — the table is *literally identical* across τ rows (deviations of ≤0.001 are within FID estimator stochasticity). The cap is binding everywhere, even at cap=0.5. This means at 16 steps, ARPG is uncertain about more than half of its parallel predictions every step.
2. **Cap is the dominant lever** — FID improves monotonically with cap, from 5.917 (cap=0.1) to 5.451 (cap=0.5). Linear trend.
3. **No diminishing returns visible up to cap=0.5** — running cap > 0.5 might continue to improve quality (open question for follow-up).

### Block B results — Metric comparison at 16 steps, τ=0.5

| Metric | cap=0.1 | cap=0.2 | cap=0.5 | Best Δ vs vanilla |
|--------|---------|---------|---------|--------------------|
| max_prob | 5.917 | 5.788 | 5.451 | −0.762 |
| entropy | 5.966 | 5.803 | 5.709 | −0.503 |
| **margin** | 5.923 | 5.781 | **5.424** | **−0.789** ⬅ |

**Findings:**

1. **The metric ranking changes with cap.** At low cap, max_prob and margin are roughly tied (both better than entropy). At cap=0.5, **margin pulls ahead** as the absolute winner. Entropy is consistently the weakest metric.
2. **The single best config is `margin × τ=0.5 × cap=0.5` → FID 5.424 (−12.69%)**. This becomes the candidate for FID-50K validation.

### Block C results — Step-count sensitivity (max_prob, τ=0.5, cap=0.2)

| Steps | Vanilla FID-10K | Rejection FID-10K | Δ |
|-------|------------------|--------------------|---|
| 8 | 13.357 | 11.011 | **−2.346 (−17.6%)** |
| 12 | 8.120 | 7.161 | **−0.959 (−11.8%)** |
| 16 | 6.212 | 5.788 | −0.424 (−6.8%) |
| 32 | 4.846 | 4.830 | −0.016 |
| 64 | 4.858 | 4.850 | −0.007 |

**Findings:**

1. **Monotonic, dramatic scaling law:** the FID benefit of rejection grows smoothly as step count decreases, from ~0% at 64 steps to **−17.6%** at 8 steps. This is a clean trend across nearly an order of magnitude in step count.
2. **At 8 and 12 steps, the gain is large enough to dwarf any sampling noise** — confidence in the effect is very high.
3. Note: this column uses `max_prob, cap=0.2`, not the best (margin, cap=0.5). The 8/12-step numbers might improve further with the better config — open question.

### Robustness

**18 / 18 rejection configs at 16 steps beat vanilla.** No exceptions. The effect is robust to all (τ, cap, metric) combinations tested.

### Phase 2 ranked summary (top 5 at 16 steps)

| Rank | metric | τ | cap | FID | Δ |
|------|--------|---|-----|-----|---|
| 1 | margin | 0.5 | 0.5 | 5.424 | −0.789 |
| 2 | max_prob | 0.7 | 0.5 | 5.451 | −0.762 |
| 3 | max_prob | 0.3 | 0.5 | 5.451 | −0.762 |
| 4 | max_prob | 0.5 | 0.5 | 5.451 | −0.762 |
| 5 | max_prob | 0.5 | 0.3 | 5.583 | −0.629 |

### Phase 2 synthesis

The 16-step result from Phase 1b was not a fluke — it is robust, scales monotonically with regime aggressiveness, and is improvable with looser caps and the right metric (margin, not max_prob). The project now has:

- **A clean compute-efficiency claim:** rejection delivers a 12.69% FID reduction at 16 steps and up to 17.6% at 8 steps.
- **A monotonic scaling law** linking step count to rejection benefit.
- **A surprising τ-irrelevance finding** — confidence threshold is not the right knob; rejection rate cap is.
- **A clean negative ablation** (Phase 1b) showing intervention must happen during generation, not post-hoc.

**Full results CSV:** `/content/drive/MyDrive/ARPG-assets/results/pilot-20260421/phase2-16step/16step-results.csv`
**Combined CSV (all phases):** `/content/drive/MyDrive/ARPG-assets/results/pilot-20260421/phase2-16step/combined-results-v2.csv`

---

## Phase 3 — FID-50K Validation on Top Configs *(PENDING)*

**Target date:** April 27 – May 1, 2026
**Purpose:** Replace the 10K-sample FID estimates with paper-grade 50K-sample numbers, for headline numbers in the progress report.

### Planned configs

Based on Phase 2 winner ranking, the FID-50K runs to do:

| # | Config | Steps | Purpose |
|---|--------|-------|---------|
| 1 | margin, τ=0.5, cap=0.5 | 16 | Validate the headline winner |
| 2 | vanilla | 16 | Matched-step baseline |
| 3 | margin, τ=0.5, cap=0.5 | 8 | Validate the ultra-aggressive regime |
| 4 | vanilla | 8 | Matched-step baseline |

Optional:
- vanilla at 32 steps with FID-50K for the matched-quality claim (we already have vanilla at 64 with FID-50K from Phase 0)
- margin, τ=0.5, cap=0.5 at 12 steps for the full step-count curve

### Compute estimate

Each config: ~2 hours for 50K samples on A100 + ~5 min FID. 4 configs → ~9 hours. Realistic to split across 2 Colab sessions over 2 days.

---

## Phase 4 — Progress Report *(DUE MAY 3)*

Deliverable for the May 3 milestone.

---

## Raw artefact index

All paths relative to `/content/drive/MyDrive/ARPG-assets/results/`.

| Location | Contents |
|----------|----------|
| `pilot-20260421/results.csv` | Phase 1 pilot, 19 rows (32 steps) |
| `pilot-20260421/logs/` | Per-config rejection JSON logs + heatmaps (Phase 1) |
| `pilot-20260421/tier1-extension/tier1-results.csv` | Phase 1b, 9 rows |
| `pilot-20260421/tier1-extension/combined-results.csv` | Merged Phase 1 + Phase 1b, 28 rows |
| `pilot-20260421/tier1-extension/logs/` | Per-config logs + heatmaps (Phase 1b) |
| `pilot-20260421/phase2-16step/16step-results.csv` | Phase 2, 22 rows |
| `pilot-20260421/phase2-16step/combined-results-v2.csv` | Merged Phase 1 + 1b + 2, 49 unique rows |
| `pilot-20260421/phase2-16step/logs/` | Per-config logs + heatmaps (Phase 2) |

---

## Rolling list of conclusions

*Every entry is a finding we're reasonably confident about at the time it was added. Update if later evidence overturns it.*

1. **[April 12]** Vanilla ARPG-L reproduces paper results to within reasonable seed variance.
2. **[April 22]** At 32 steps, the `max_reject_rate` cap dominates — τ has no visible effect because the cap always binds.
3. **[April 22]** max_prob is the best of the three confidence metrics at 32 steps.
4. **[April 24]** Post-hoc refinement (K=0.1, 0.2) consistently hurts quality — intervention must happen DURING generation.
5. **[April 24]** Rejection strongly benefits the aggressive regime: at 16 steps, FID drops 0.424 (6.212 → 5.788). At 32+ steps, the effect is indistinguishable from noise.
6. **[April 24]** The project's primary claim is now compute-efficiency in the low-step regime, not general FID improvement at standard step counts.
7. **[April 25]** **τ is irrelevant in the aggressive regime** — at 16 steps with max_prob, the (τ × cap) grid shows literally identical FIDs across τ ∈ {0.3, 0.5, 0.7}. The cap is binding even at cap=0.5, meaning ARPG is uncertain about more than half its parallel predictions every step at 16 steps. **The cap is the only meaningful knob.** Conclusion 2 from April 22 generalizes: τ has no effect anywhere in our explored range.
8. **[April 25]** **The metric ranking depends on the regime.** At 32 steps and at 16 steps with low cap, max_prob is best. **At 16 steps with cap=0.5, margin overtakes max_prob** (5.424 vs 5.451). This contradicts conclusion 3 from April 22 in the high-cap regime — margin is best when the rejection rate is high. Entropy is consistently worst.
9. **[April 25]** **Cap improves FID monotonically up through 0.5** at 16 steps. No diminishing returns visible yet. Open question: does cap > 0.5 keep helping?
10. **[April 25]** **Rejection benefit scales monotonically with regime aggressiveness** (Block C of Phase 2). Δ vs vanilla: −0.007 at 64 steps → −0.016 at 32 → −0.424 at 16 → −0.959 at 12 → −2.346 at 8. This is the strongest scaling result in the project and directly supports the compute-efficiency narrative.
11. **[April 25]** **The mechanism is robust across the entire (τ, cap, metric) grid at 16 steps** — 18 / 18 rejection configs beat vanilla. No exceptions in the explored space.
12. **[April 25]** Headline result for the progress report: **margin, τ=0.5, cap=0.5 at 16 steps reduces FID-10K from 6.212 → 5.424 (−12.69%)**. Pending FID-50K validation in Phase 3.
