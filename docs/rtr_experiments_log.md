# RTR Experiments Log — Random Token Rejection

**Cumulative record of experimental runs after the Phase 5 random-deferral discovery.**

This log covers the experiments run *after* the progress report (submitted May 3, 2026), once Phase 5's overnight sweep revealed that **random** selection outperforms confidence-guided rejection. These runs ground the final paper, where the method is renamed **Random Token Rejection (RTR)** and confidence-guided variants are relegated to ablation.

For Phase 0–5 (vanilla baseline reproduction through the original confidence-guided experiments + Phase 5 random-deferral discovery), see [`docs/experiments_log.md`](experiments_log.md).

**Team:** Riad Shahbazov, Ömer Maraş, Mohamad Alomar
**Compute environment:** Google Colab Pro, NVIDIA A100-SXM4-40GB
**Model:** ARPG-L (320M parameters), pretrained checkpoint `arpg_300m.pt`
**Evaluator:** OpenAI `guided-diffusion` (ADM protocol)
**Reference batch:** ImageNet-1K 256×256, `VIRTUAL_imagenet256_labeled.npz`

---

## RTR Phase 1 — Cap Sweep (random selection)

**Date completed:** May 13–14, 2026
**Compute used:** ~6 hours on A100 (sampling + FID eval), session elapsed 6.00 h
**Configs run:** 42 (random × 7 caps × 2 step counts × 3 seeds), **0 failures**
**Notebook:** [`notebooks/cap_sweep_random_colab.ipynb`](../notebooks/cap_sweep_random_colab.ipynb)
**Drive output:** `MyDrive/ARPG-assets/results/final-paper/cap-sweep-random/`

### Purpose

The Phase 5 random-deferral result used ρ=0.5 because that was the Phase 2 winner for the (now-deprecated) margin variant. Before running the §3 main table at FID-50K with RTR, we needed to identify the actual optimum ρ\* for *random* selection.

### Setup

| Setting | Value |
|---|---|
| Selection rule | random |
| Confidence threshold τ | 2.0 (unreachable — forces cap branch) |
| Step counts | 16, 8 |
| Caps ρ | 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 |
| Seeds | 0, 1, 2 |
| Samples per config | 10,000 (FID-10K) |
| NPZ retention | not kept on Drive (free up space) |
| Other params | arccos, CFG 5.0, temperature 1.0, top-k 0, top-p 1.0, bf16 |

### Per-seed FID-10K — 16 steps

| ρ | seed 0 | seed 1 | seed 2 | **mean** | std |
|---|---|---|---|---|---|
| 0.1 | 5.9275 | 6.0352 | 5.8790 | 5.9472 | 0.0800 |
| 0.2 | 5.7127 | 5.7751 | 5.8088 | 5.7655 | 0.0489 |
| 0.3 | 5.5064 | 5.6113 | 5.4724 | 5.5300 | 0.0726 |
| 0.4 | 5.3322 | 5.2677 | 5.1965 | 5.2655 | 0.0679 |
| 0.5 | 5.1644 | 5.1684 | 5.1966 | 5.1765 | 0.0175 |
| 0.6 | 5.0800 | 5.1080 | 5.0791 | **5.0890** | 0.0164 |
| 0.7 | 5.1485 | 5.0278 | 5.0750 | **5.0838** | 0.0608 |

### Per-seed FID-10K — 8 steps

| ρ | seed 0 | seed 1 | seed 2 | **mean** | std |
|---|---|---|---|---|---|
| 0.1 | 12.1117 | 12.0708 | 11.8943 | 12.0256 | 0.1156 |
| 0.2 | 10.6261 | 10.6824 | 10.9454 | 10.7513 | 0.1707 |
| 0.3 | 9.8598 | 9.8320 | 9.8778 | 9.8565 | 0.0231 |
| 0.4 | 8.4243 | 8.4035 | 8.4433 | 8.4237 | 0.0199 |
| 0.5 | 7.4098 | 7.4975 | 7.3992 | 7.4355 | 0.0540 |
| 0.6 | 6.4425 | 6.6908 | 6.7219 | 6.6184 | 0.1531 |
| 0.7 | 6.0434 | 6.1078 | 6.1491 | **6.1001** | 0.0532 |

### Findings

1. **16-step optimum: ρ ∈ {0.6, 0.7} (saturated).** Means are 5.0890 vs 5.0838 — a gap of 0.005 FID, far below the seed std (~0.05). The 16-step regime is fully saturated by ρ=0.6; pushing higher gives no further improvement.

2. **8-step optimum: ρ = 0.7.** Mean 6.1001 (n=3). The 8-step FID curve is monotonically decreasing across the swept range (12.03 → 6.10) but slowing markedly toward ρ=0.7 — likely near saturation.

3. **Random outperforms margin across the entire cap range.** Direct comparison to the Phase 2 margin τ=0.5 cap sweep at 16 steps (single seed):

| ρ | margin (n=1) | random (n=3) | Δ |
|---|---|---|---|
| 0.1 | 5.923 | 5.9472 | +0.024 |
| 0.2 | 5.781 | 5.7655 | −0.015 |
| 0.3 | — | 5.5300 | — |
| 0.5 | 5.424 | 5.1765 | **−0.247** |
| 0.6 | 5.421 | 5.0890 | **−0.332** |
| 0.7 | 5.778 | 5.0838 | **−0.694** |

Random doesn't just beat margin at the headline ρ=0.5 — its advantage *grows* as ρ increases. At ρ=0.7, margin actively crashes (5.778) while random saturates cleanly (5.0838). The stable-cap regime is much wider for random than for confidence-guided rejection.

4. **Cap is the only meaningful knob.** Combined with the Phase 1/2/5 finding that τ is empirically dead, RTR has *one* hyperparameter: ρ. With the optimum localized at ρ ≈ 0.6–0.7 across both step counts.

### Recommendation

**ρ\* = 0.7** for the §3 main table. Matches the 8-step optimum exactly, is within FID-10K noise of the 16-step optimum (5.0838 vs 5.0890 at ρ=0.6), and gives the cleanest single-value method description.

---

## RTR Phase 2 — Step-Varying Cap Schedule

**Date completed:** May 14, 2026
**Compute used:** ~4.3 hours on A100, session elapsed 4.31 h
**Configs run:** 30 (5 schedules × 2 step counts × 3 seeds), **0 failures**
**Notebook:** [`notebooks/cap_schedule_random_colab.ipynb`](../notebooks/cap_schedule_random_colab.ipynb)
**Code changes:** [`models/arpg.py`](../models/arpg.py) extended with `max_reject_rate_end` parameter (linear interpolation between `max_reject_rate` and `max_reject_rate_end` across decoding steps); [`sample_c2i_ddp.py`](../sample_c2i_ddp.py) adds `--max-reject-rate-end` CLI flag. Both backward-compatible.
**Parity test:** [`scripts/test_cap_schedule_parity.py`](../scripts/test_cap_schedule_parity.py) — verifies `max_reject_rate_end=None` is bit-identical to the prior single-cap behaviour.
**Drive output:** `MyDrive/ARPG-assets/results/final-paper/cap-schedule-random/`

### Purpose

Following the structural-deferral intuition: early decoding steps have a small KV cache, so deferring more there gives larger context gains in the re-attempt; late steps have a near-full cache, so deferred tokens end up in the forced-accept final step with no further gain. The hypothesis says: a *decaying* ρ schedule (high early, low late) should beat constant ρ; a *growing* ρ schedule (low early, high late) should be catastrophically worse.

### Setup

| Setting | Value |
|---|---|
| Selection rule | random |
| Step counts | 8, 16 |
| Seeds | 0, 1, 2 |
| Samples per config | 10,000 (FID-10K) |
| Schedule type | linear interpolation between ρ_start and ρ_end |
| NPZ retention | not kept on Drive |

#### Schedules

| Label | ρ_start | ρ_end | Note |
|---|---|---|---|
| `constant-0.7`   | 0.7  | 0.7  | Control = RTR Phase 1 optimum |
| `decay-mild`     | 0.8  | 0.5  | Mild aggressive→conservative |
| `decay-aggr`     | 0.9  | 0.3  | Aggressive decay |
| `decay-extreme`  | 0.95 | 0.2  | Pushing it |
| `reverse-grow`   | 0.3  | 0.7  | Sanity check — opposite direction |

### Per-seed FID-10K — 8 steps

| Schedule | seed 0 | seed 1 | seed 2 | **mean** | std |
|---|---|---|---|---|---|
| constant-0.7 | 6.0434 | 6.1078 | 6.1490 | 6.1001 | 0.0532 |
| **decay-mild** | 5.9340 | 5.9494 | 6.0672 | **5.9835** | 0.0729 |
| decay-aggr | 6.1315 | 6.1870 | 6.1518 | 6.1568 | 0.0281 |
| decay-extreme | 6.3965 | 6.3326 | 6.2985 | 6.3425 | 0.0498 |
| reverse-grow | 9.2222 | 9.0383 | 9.2893 | 9.1833 | 0.1299 |

### Per-seed FID-10K — 16 steps

| Schedule | seed 0 | seed 1 | seed 2 | **mean** | std |
|---|---|---|---|---|---|
| **constant-0.7** | 5.1485 | 5.0279 | 5.0750 | **5.0838** | 0.0608 |
| decay-mild | 5.1011 | 5.0826 | 5.1924 | 5.1254 | 0.0588 |
| decay-aggr | 5.2278 | 5.0966 | 5.0687 | 5.1310 | 0.0849 |
| decay-extreme | 5.2544 | 5.2079 | 5.1650 | 5.2091 | 0.0447 |
| reverse-grow | 5.4697 | 5.4793 | 5.4670 | 5.4720 | 0.0064 |

### Deltas vs `constant-0.7` control

| Schedule | 8-step Δ | 16-step Δ |
|---|---|---|
| **decay-mild** | **−0.1165** | +0.0416 |
| decay-aggr | +0.0567 | +0.0472 |
| decay-extreme | +0.2424 | +0.1253 |
| **reverse-grow** | **+3.0832** | +0.3882 |

### Findings

1. **Mild decay helps at 8 steps, hurts at 16 steps.** `decay-mild` (0.8 → 0.5) beats `constant-0.7` by 0.1165 FID at 8 steps — ~2× the seed std (0.05–0.07), statistically distinguishable but modest in absolute terms. At 16 steps every schedule is slightly worse than constant.

2. **Aggressive decay is counterproductive at both step counts.** `decay-aggr` (0.9 → 0.3) and `decay-extreme` (0.95 → 0.2) are *worse* than constant at both step counts. The sweet spot is mild decay — preserving a relatively high average cap is more important than front-loading aggressively. With aggressive decay, the late-step cap drops too low and the forced-accept final step gets flooded with all the deferred tokens at once, with no context bandwidth left to absorb them.

3. **`reverse-grow` is catastrophic — clean mechanistic confirmation.** Deferring conservatively early (ρ=0.3) and aggressively late (ρ=0.7) inverts the structural-deferral story. At 8 steps the FID jumps from 6.10 to 9.18 (+50% relative). At 16 steps from 5.08 to 5.47 (+8% relative). The 8-step asymmetry between forward decay (−0.12) and reverse growth (+3.08) is the cleanest ablation of the underlying mechanism: early-step deferral creates large cache growth between attempts; late-step deferral pushes tokens into the forced-accept final step with no further context gain. This becomes the central mechanistic figure for §4 of the final paper.

4. **16 steps is fully saturated.** The Phase 1 cap-saturation finding (ρ=0.6 ≈ ρ=0.7) extends to step-varying schedules — once you're at the plateau, no schedule can lift you off it. Schedules only matter in the aggressive (8-step) regime, and even then the improvement is modest.

5. **Internal consistency check.** The 8-step `constant-0.7` schedule values (6.0434 / 6.1078 / 6.1490) match the RTR Phase 1 cap-sweep 8-step ρ=0.7 values (6.0434 / 6.1078 / 6.1491) to within rounding. The 16-step `constant-0.7` schedule values match Phase 1 16-step ρ=0.7 similarly. Determinism under fixed seed confirmed end-to-end.

### Recommendation

**Stay with constant ρ=0.7 for the §3 main table.** Two reasons:

1. **Simplicity.** Single-knob method ("defer ρ fraction per step uniformly at random") is cleaner to describe than "linear decay from ρ_start to ρ_end."
2. **Magnitude.** `decay-mild` gains 0.1165 FID at 8 steps and is *worse* at 16 steps. Whatever absolute improvement the schedule produces is small relative to the random-vs-margin gap and within the headline noise envelope.

The schedule sweep is most valuable as an **ablation figure in §4**: the `reverse-grow` catastrophic result is the cleanest mechanistic evidence in the project that early-step deferral is the active ingredient.

---

## Synthesis — what locks the §3 main table

| Decision | Value | Source |
|---|---|---|
| Selection rule | random | Phase 5 Item 1 |
| Cap ρ | 0.7 | RTR Phase 1 |
| Schedule | constant | RTR Phase 2 |
| Step counts | 8, 12, 16, 24, 32 | Plan |
| Seeds | 0, 1, 2, 3, 4 | Plan (n=5) |
| Samples | 50,000 (FID-50K) | Plan |

### Rolling list of conclusions (RTR extension to the main experiments log)

21. **[May 13–14]** Multi-seed random cap-sweep at FID-10K confirms random outperforms margin at every cap. The advantage *grows* with ρ — at ρ=0.7, random=5.08 vs margin=5.78 (margin actively crashes while random saturates cleanly). Random's stable cap range is wider than confidence-guided. Conclusion 16 from the main log is strengthened: random is not merely "as good as" confidence-guided — it dominates across the entire deployable cap range.

22. **[May 13–14]** **RTR optimum is ρ\*=0.7**, essentially tied with ρ=0.6 at 16 steps (gap = 0.005 FID, within noise) and the clear minimum at 8 steps. The 8-step curve is still descending slowly toward ρ=0.7 but its marginal returns are flat. Conclusion 17 from the main log (cap saturates at ρ≈0.6 for margin) generalises to random with the saturation point shifted slightly higher.

23. **[May 14]** Step-varying ρ has *small marginal value*. Mild decay (0.8→0.5) is the only schedule that beats constant at 8 steps (−0.12 FID); at 16 steps no schedule helps. Aggressive decay (0.9→0.3, 0.95→0.2) actively hurts. **The sweet spot is mild decay, not aggressive.** A single fixed ρ=0.7 is the right operational choice.

24. **[May 14]** **`reverse-grow` is catastrophic — the cleanest mechanistic confirmation in the project.** At 8 steps, reverse-grow (0.3→0.7) gives FID 9.18 vs constant-0.7's 6.10 (+3.08, +50% relative). At 16 steps, 5.47 vs 5.08 (+0.39, +8% relative). The asymmetry between forward decay (mild improvement) and reverse growth (catastrophic) is the cleanest evidence in the project that the active mechanism is *early-step deferral with subsequent cache growth*, not late-step deferral into the forced-accept final step. This becomes the mechanistic figure in §4 of the final paper.

25. **[May 14]** **For the §3 main table:** random selection, constant ρ=0.7, n=5 seeds across step counts {8, 12, 16, 24, 32}, FID-50K. Total: 39 new FID-50K runs (vanilla + RTR at each (step, seed) cell that isn't already done from prior phases). The schedule sweep contributes the reverse-grow ablation figure to §4 as mechanistic evidence; the cap sweep contributes a cap-saturation curve to §4.1.

---

## Raw artefact index

All paths relative to `/content/drive/MyDrive/ARPG-assets/results/`.

| Location | Contents |
|----------|----------|
| `final-paper/cap-sweep-random/results.csv` | RTR Phase 1, 42 rows (FID-10K) |
| `final-paper/cap-sweep-random/summary.csv` | Per-(step, cap) mean ± std table |
| `final-paper/cap-sweep-random/cap_saturation.png` | Cap-saturation curves (two-step plot) |
| `final-paper/cap-sweep-random/logs/` | Per-config sampling + evaluator logs |
| `final-paper/cap-sweep-random/rejection-logs/` | Per-config rejection tracker JSON + spatial heatmaps |
| `final-paper/cap-sweep-random/samples/grids/` | 42 qualitative 8-class grids |
| `final-paper/cap-sweep-random/samples/individual/` | 42 × 8 = 336 individual class PNGs |
| `final-paper/cap-schedule-random/results.csv` | RTR Phase 2, 30 rows (FID-10K) |
| `final-paper/cap-schedule-random/summary.csv` | Per-(step, schedule) mean ± std table |
| `final-paper/cap-schedule-random/schedule_comparison.png` | Schedule bar charts (two-step plot) |
| `final-paper/cap-schedule-random/logs/` | Per-config sampling + evaluator logs |
| `final-paper/cap-schedule-random/rejection-logs/` | Per-config rejection tracker JSON + heatmaps |
| `final-paper/cap-schedule-random/samples/grids/` | 30 qualitative 8-class grids |
| `final-paper/cap-schedule-random/samples/individual/` | 30 × 8 = 240 individual class PNGs |
