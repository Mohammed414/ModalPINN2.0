
# ModalPINN EnKF Audit — Part 2
## Rebuilding the pressure-only Ensemble Kalman Filter so that it assimilates

**Branch** `enkf-repairs` · Re = 100 · 32 wall pressure taps · q = 16 · 201 cycles

---

## 1. What was wrong

Part 1 established the observation operator (wall-normal probe extrapolation) and a
validated scalar frequency handle (time dilation). The audit then found that the Stage D
filter had a **Kalman gain fraction of 2.4 x 10<sup>-4</sup>**. The filter was ignoring the
measurements: Stage D was a free run wearing a filter's clothing, and its failure to
reconstruct the wake was therefore *not* evidence about observability.

Four repairs were specified. All four are implemented, each behind a flag so the old
behaviour stays reachable, and each verified numerically rather than asserted.

---

## 2. Fix 1 — per-tap bias removal

Stage D removed a single scalar per member, `c = mean(p_meas - p_pred)`. Phase 1 showed
78–83% of innovation variance is a **theta-dependent** static bias that no scalar can absorb.

Stage D2 estimates a fixed 32-vector `b` over a **forecast-only** window of 61 cycles
(6.1 t.u. = 1.006 shedding periods), then holds it fixed. The window length was chosen so
that the oscillation averages out: leakage of the unsteady signal into `b` is **8.4%** of the
unsteady RMS at 61 cycles (versus 81% at 20 cycles). `b` uses tap data and the model's own
output only — no truth.

| quantity | value |
|---|---|
| spin-up residual RMS, before | 0.1022 |
| after per-tap `b` | **0.0197** |
| after global scalar (Stage D) | 0.1007 |
| `b` spread over taps | 0.0987 (range −0.179 … +0.207) |
| innovation static fraction, Stage D | 88.0% |
| innovation static fraction, per-tap | **4.0%** |

**The "estimate it continuously" claim, tested rather than asserted.** Running
`--bias-mode continuous` re-estimates the full per-tap offset from each cycle's own
measurement. Result: innovation RMS **4.0 x 10<sup>-17</sup>** and Kalman correction norm
**3.1 x 10<sup>-16</sup>** — machine zero. It absorbs exactly the signal the filter exists to
assimilate. Holding `b` fixed is necessary, and now measured.

---

## 3. Fix 2 — sigma_p from the unsteady signal

Stage D used `sigma_p = 0.3 x std(tap_p) = 0.1015`. But `std(tap_p) = 0.338` is **98% static**
theta-variation of mean Cp, which after Fix 1 is not part of the innovation at all. The
correct scale is the unsteady tap RMS, **0.0472**.

| sigma_p | gain fraction | NIS / n_taps | reproducible across seeds? |
|---|---|---|---|
| 0.0072 | 0.80 | **0.96** | no — runaway gamma, clips in 2/4 seeds |
| 0.0143 | 0.52 | 0.39 | no — clips in 2/4 seeds |
| 0.0236 | 0.30 | 0.18 | — |
| **0.0472** | **0.12** | 0.05 | **yes — 10/10 seeds, zero clips** |

**Which diagnostic I trusted, and why.** NIS is now meaningful again (static bias is only 4%
of innovation, so it is no longer circular) and it points at sigma_p ≈ 0.0072, where
NIS/n_taps = 0.96 is textbook. I did **not** take it. At that setting the gamma trajectory is
seed-unstable and runs into the clip, and — decisively — the *shuffled negative control* also
runs to the clip, which means the apparent confidence is not coming from the data. I chose the
mandated unsteady-RMS value 0.0472, which gives gain 0.12, is reproducible across 10 seeds, and
separates cleanly from both controls. **Gain fraction plus cross-seed reproducibility plus
control separation was the criterion; NIS alone was not.**

---

## 4. Fix 3 — multi-direction ensemble

Perturbations are the discrete curl of a low-pass-filtered random streamfunction, so
incompressibility is exact by construction, not by projection.

- **Divergence:** max |div| of the perturbation field alone = **4.4 x 10<sup>-16</sup>** (8.9 x
  10<sup>-16</sup> at the larger amplitude 0.05 tested during calibration). Perturbed members sit at
  **4.4 x 10<sup>-15</sup>** against an unperturbed baseline of 2.8 x 10<sup>-15</sup> — a factor
  1.6 higher, but both are O(10<sup>-15</sup>), i.e. floating-point round-off on this grid rather
  than a physical divergence. The perturbation adds no divergence beyond round-off, as the discrete
  curl construction requires; it does not *reduce* the solver's own baseline round-off.
- **Stability:** all members stepped forward for the full 201 cycles, no blow-up, max|u| = 1.41.
- **Directions:** n_eff (participation ratio of the state anomaly spectrum)
  **1.013 → 3.29** initial, **5.86** sustained. Phase jitter alone put 99.3% of anomaly energy
  in one direction.
- **Amplitude:** 0.04, calibrated so ensemble tap spread (0.0426) matches the unsteady tap RMS
  it must explain (0.0472).

**A latent bug found on the way.** Stage D's `BASE_IC_TIME = 310.0` is the *first* snapshot in
`spinup_snapshots.npz`, so every negative jitter clamped to index 0: only **8 of 16 members were
distinct**, 6 byte-identical. Base time moved to 316.0 (window centre).

**Multiplicative inflation was not sufficient — and this is a physical statement.** All members
live on the same attractor, so forecast dynamics contract transverse to it and anomalies decay
with nothing to regenerate them. Measured: alpha = 1.00 / 1.02 / 1.05 / 1.10 gives median gain
0.0039 / 0.0041 / 0.0043 / **0.0047**. Inflation can only rescale anomalies that still exist.
**Additive** divergence-free model error (amplitude 0.010/cycle) was required and lifts the
median gain to **0.12**. This is also the honest representation of a model we know is biased
(Phase 1: IBM momentum leak, ~20% surface-pressure amplitude error).

---

## 5. Fix 4 — gamma augmentation: the observability result

gamma initialised at **1.0 ± 0.07** — the solver's own rate, never seeded near the
truth-consistent gamma* = 0.885. Clipped to [0.7, 1.3]; gamma-anomaly spread floor 0.02 to
prevent parameter collapse.

**gamma is not swamped by dimensionality.** K is computed row-wise, so gamma's update depends
on cov(gamma, tap pressure), not on the 18844:1 entry ratio. It receives signal: the gamma row
of K is non-zero throughout and gamma moves every cycle.

**But the answer is negative.** Across 10 seeds:

| | gamma at end | vs gamma* = 0.885 | clip hits |
|---|---|---|---|
| nominal (n=10) | 0.925 ± 0.057 | 0.059 | **0** |
| shuffled time (n=8) | 0.954 ± 0.190 | 0.148 | 550 |
| scrambled sensors (n=8) | 0.965 ± 0.119 | 0.120 | 38 |

Welch t-test on gamma: **p = 0.69** vs shuffled, **p = 0.41** vs scrambled. The mean drift is in
the right direction (0.956 → 0.925) and nominal lands closest to gamma*, but it is **not
statistically distinguishable from the controls**. Reporting this as "the filter found the
frequency" would be over-reading a null.

Two things *are* significant and worth keeping:

1. Nominal gamma **variance** is significantly tighter than shuffled (Levene **p = 0.017**) —
   real pressure constrains gamma more than scrambled pressure does, even though it does not
   localise it.
2. Nominal never hit the clip (0 hits) while shuffled hit it 550 times. The controls diverge;
   the nominal run does not.

**Verdict on gamma:** with 32 wall taps, q = 16, and a 14 t.u. assimilation window, the shedding
rate is **weakly constrained but not identified**. The ensemble covariance between gamma and tap
pressure is too noisy at this ensemble size to localise a 13% frequency error.

---

## 6. The filter now assimilates — and it reconstructs the wake

| | Stage D | Stage D2 |
|---|---|---|
| gain fraction (median) | 2.4 x 10<sup>-4</sup> | **0.121** (500x) |
| innovation static fraction | 88% | **4.0%** |
| ensemble n_eff | 1.013 | **5.86** |
| ensemble tap spread (final) | 0.0011 (collapsing) | **0.0168** (sustained) |

Stage F2 scores against the **withheld** truth on the common window (cycles 61–200):

| run | E_u | E_v | mean abs phase error |
|---|---|---|---|
| free run | 0.130 | 0.528 | 0.618 rad |
| Stage D (original) | 0.136 (+4.5%) | 0.566 (+7.2%) | 0.705 rad |
| **Stage D2 (repaired)** | **0.098 (−24.5%)** | **0.242 (−54.1%)** | **0.057 rad** |
| D2 shuffled control | 0.156 (+20.0%) | 0.462 | 1.451 rad |
| D2 scrambled control | 0.145 (+11.0%) | 0.610 | 0.121 rad |

The original filter made the field **worse** than not assimilating at all. The repaired filter
**halves the cross-wake velocity error** and cuts phase error by **10.8x**. Both negative
controls degrade E_u relative to the free run.

**Negative controls on innovation** (the well-posed statistic, n = 10/8/8): nominal
0.0155 ± 0.0017, shuffled 0.0622 ± 0.0024, scrambled 0.0467 ± 0.0022 —
**p = 4 x 10<sup>-15</sup>** and **1 x 10<sup>-13</sup>**. The filter is tracking the real
pressure sequence and the real sensor layout, not fitting noise.

---

## 7. The answer on observability

**The wake is observable from 32 wall pressure taps.** Cross-wake velocity error drops 54% below
a free run against withheld truth, phase error drops 10.8x, and both negative controls
fail — that is a genuine reconstruction, not a filter tuned into agreement.

**The Stage D conclusion was an artefact of filter mis-specification, not a property of the
measurement.** With a gain fraction of 2 x 10<sup>-4</sup> the earlier experiment could not have
detected observability regardless of whether it was there.

**Relevance to the ModalPINN failure.** The original hypothesis was that wall pressure carries
too little information about the far wake, so the oscillating modes collapse. That hypothesis is
**not supported**: an estimator that actually uses the same 32 taps recovers the wake. The
ModalPINN mode collapse is therefore more likely a property of *its own* training signal — a
near-zero oscillating field satisfying the pointwise NS residual nearly as well as the true one —
than a fundamental information deficit at the sensors.

**What remains unresolved.** The shedding *frequency* is not identifiable at this ensemble size
(Section 5). Since ModalPINN takes omega_0 as given, this does not affect the ModalPINN
diagnosis, but it does bound what a pressure-only filter can self-calibrate.

---

## 8. Caveats

1. Additive model-error inflation is a **tuned** quantity (0.010/cycle). It was set by matching
   ensemble tap spread to the unsteady tap RMS, an observation-space criterion — no truth was
   used — but it is a knob, and the gain fraction depends on it (0.048 → 0.60 over
   amplitude 0.005 → 0.040).
2. sigma_p = 0.0472 is **not** the NIS-optimal value. I chose reproducibility over NIS and said
   so; a reader who prefers NIS-consistency should read the sigma_p sweep in the JSON summary.
3. The forward model retains the characterised IBM bias (~20% surface pressure amplitude,
   omega_s 13% fast). The per-tap `b` absorbs its static part; its dynamic part is not corrected.
4. q = 16 is small. The gamma null in particular may be an ensemble-size limitation rather than
   a genuine unobservability — a q = 64 run is the obvious next test.
5. Stage F2 compares on cycles 61–200 because Stage D2 spends 61 cycles in the forecast-only
   bias window; all runs are scored on that same window, so the comparison is like-for-like.
6. The gamma spread floor (0.02) is an artificial re-inflation. Without it gamma freezes; with
   it, the reported gamma spread is a floor, not a posterior width.

---

## 9. Files

**New code:** `estimator/enkf2.py`, `estimator/ensemble_init.py`,
`experiments/stage_d2_enkf_repaired.py`, `evaluation/stage_f2_evaluate_repaired.py`
**New results:** `experiments/stage_d2_enkf_{nominal,shuffled,scrambled_sensors}.npz`,
`experiments/stage_f2_evaluation.npz`, `experiments/stage_d2_summary.json`
**Figures:** `figures/stage_d2_diagnostics.png`, `figures/stage_d2_ablation_controls.png`

No Stage D file was modified or overwritten. `estimator/enkf.py` is untouched; the repaired
filter is a new module.
