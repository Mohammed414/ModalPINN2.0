# R9 production run — what happened

Run: `runs/R9_extracted/R9_TRUST_street_noBVF_noK0_noPhase_pressure_only_Re100_Nm3_Nint50000_Nmes5000_WL25_Ntap32_FSBC_FIBC_rho0p6_cap0p12_seed0_20260814`
Trained 2026-08-13/14 on Colab (Tesla T4). Method developed in
`R9_wake_rescue/` (see `REPORT.md` there); this document covers the
production run only.

---

## 1. Headline

**Best run in the project on every region and every velocity metric, but it
missed the testbed's prediction by a factor of two in the far wake, for a
reason that is now measured and fixable.**

| region | metric | R7 (prior best) | **R9** | change |
|---|---|---|---|---|
| near-cylinder | E_u | 0.0839 | **0.0443** | −47% |
| near-cylinder | E_v | 0.1538 | **0.0988** | −36% |
| near-cylinder | E_p | 0.0256 | **0.0190** | −26% |
| near-wake | E_u | 0.2234 | **0.1868** | −16% |
| near-wake | E_v | 0.9565 | **0.7569** | −21% |
| near-wake | E_p | 0.4946 | **0.3805** | −23% |
| far-wake | E_u | 0.3436 | **0.2736** | −20% |
| far-wake | E_v | 1.0016 | **0.7881** | −21% |
| far-wake | E_p | 1.9843 | **1.6777** | −15% |
| whole domain | E_v | 0.7772 | **0.6110** | −21% |

(Both columns are each run's own `regional_evaluation.txt`, standard protocol.)

Two things deserve separate emphasis:

- **near-cylinder E_u = 0.0443 with NO warm start.** R7 needed a warm start
  from R3 to reach 0.0839. R9 beat that from scratch, i.e. the trust ansatz
  removed the need for the warm-start crutch.
- **far-wake E_v = 0.7881 is the first sub-1.0 far-wake value with a real
  margin** (21% below "no wake predicted"). R4 also technically dipped below
  1.0 (0.9964) but by 0.4%, which is noise.

**But:** the R9_wake_rescue testbed predicted ≲0.40 far-wake, and the
untrained analytic street prior alone scores 0.3965. The trained network is
therefore **2× worse than its own prior** in the far field. Section 6
explains why.

---

## 2. What R9 changed, mechanically

Every k≥1 mode network output was replaced by a bounded correction around a
closed-form von Kármán street derived from the 32 taps:

    q_k(x,y) = S_k(x,y) + (ρ|S_k(x,y)| + c) · tanh(net_k(x,y)),   k ≥ 1
    q_0(x,y) = net_0(x,y)                                        (unchanged)

with ρ = 0.6, c = 0.12. Because |tanh| ≤ 1, q_k = 0 is unreachable wherever
|S_k| > c/(1−ρ) — the dead-wake solution that every prior run converged to is
removed from the search space rather than penalised.

Loss terms: **physics residual + tap loss + BCs only.** BVF, K0Loss,
CV1Loss, PhaseLoss, CausalWeighting, HardSym all OFF; no warm start. This
was deliberate — if the wake lived, credit had to be unambiguous.

The prior actually used (saved in the run folder as `street_prior_used.npz`,
all derived from the taps + classical relations):

| quantity | value | source |
|---|---|---|
| ω₀ | 1.035745 | nonlinear sinusoid fit to tap-integrated lift |
| Γ | 2.527051 | Kármán drag relation inverted on measured C_D |
| U_c | 0.820501 | street self-advection self-consistency |
| C_D (pressure) | 0.989054 | tap integration |
| x_f | 1.2 | tap p₁ pattern match |
| r₀ | 0.4 | tap p₁ pattern match |
| phase | 0.731227 | image-vortex surface pressure vs tap harmonics |
| amp_scale | 0.742694 | closed-form ↔ numeric street calibration |
| scale_p | 1.238612 | linearized-Bernoulli pressure calibration |
| tap_p1_corr | 0.744698 | quality of the tap-pattern match |
| cf_corr_vs_numeric | 0.999028 | closed form vs numeric street agreement |

---

## 3. The run

| | |
|---|---|
| GPU | Tesla T4 (14.7 GB) |
| collocation points | 50,000 × 5 multigrid sets, resampled every 200 iters |
| tap measurements | 32 taps (`--PressureOnly --SparseData --NTaps 32`) |
| network | width 25/mode, 2 hidden layers, Nmodes=3 → **k = 0, 1, 2** |
| priors on | freestream BC, fluctuation inlet BC, TrustStreet |
| L-BFGS | ftol 1e-12, **44,955 function evaluations**, **31,818 s** |
| Adam polish | 106 s, loss 1.927e-04 → 2.188e-04 (Adam made it slightly worse) |
| termination | `CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH`, Tit=42,414, Tnf=44,955, N=36,918 params, projg 1.373e-04 — **see provenance note below** |
| peak host memory | 13.9 GB after session init |

**Provenance note on the termination line.** scipy's L-BFGS-B prints its
iteration table and termination message from the Fortran layer straight to
file descriptor 1, which bypasses the script's Python `Tee` — so `out.txt`
contains **no** `CONVERGENCE`, `At iterate`, or `Tit` lines (verified:
`grep -c` returns 0 for all three, in this run and in R3/R5). The values
above were transcribed from the live Colab console output, not from the run
log, and are therefore not reproducible from the archived files. What *is*
in `out.txt` is the per-evaluation `Loss:` trace written by the Python
callback: 44,955 lines, matching Tnf exactly.

Loss trajectory: 0.3579 → 2.47e-02 (100 evals) → 4.16e-03 (1k) → 4.29e-04
(10k) → 2.43e-04 (30k) → **1.927e-04** (44,955). Final boundary residual
2.457e-09.

**Note on `Nmodes 3`:** in this codebase that means three network outputs,
k = 0, 1, 2 — there is **no k=3 mode in this run**, unlike the testbed
(NK=3 → k=0..3).

The optimizer effort is worth flagging, compared on the one quantity that
*is* archived for every run — the number of `Loss:` lines in `out.txt`, i.e.
function evaluations:

| run | function evaluations | final objective |
|---|---|---|
| R5 | 4,574 | (different loss composition) |
| R3 | 15,452 | (different loss composition) |
| **R9** | **44,955** | 1.927e-04 |

R9 kept finding descent for ~3× as many evaluations as R3 and ~10× R5 —
consistent with the trivial minimum being walled off rather than merely
penalised. Two caveats: the final objective values are **not** comparable
across these runs (R3/R5 include BVF/K0/CV1 terms that R9 drops), and R5's
*termination reason* is unknown — the project's own R6 notes record that
R5's early stop "was never actually diagnosed either way", and since scipy's
message never reached `out.txt` it cannot be recovered now. An earlier draft
of this report asserted R5 ended in `ABNORMAL_TERMINATION_IN_LNSRCH`; that
was unsupported and has been removed.

---

## 4. What the reconstruction actually looks like

### Per-harmonic (v modes, network one-sided vs truth two-sided, converted)

| region | k=1 rel err | k=1 amp ratio | k=2 rel err | k=2 amp ratio |
|---|---|---|---|---|
| near-cylinder | 0.648 | 0.500 | 2.514 | 1.812 |
| near-wake | 0.790 | 0.255 | 0.901 | 1.070 |
| far-wake | 0.747 | 0.287 | **0.503** | 0.659 |

So the fundamental is present everywhere but at **25–50% of true
amplitude**, and the second harmonic is *over*-amplified near the cylinder
(1.8×) while being the best-reconstructed harmonic in the far wake (0.503).

### Kinematics — correct

| | truth | R9 |
|---|---|---|
| streamwise wavenumber (k=1) | 1.427 | 1.376 |
| wavelength | 4.40 D | 4.56 D (+3.6%) |
| implied convection speed | 0.726 U∞ | 0.752 U∞ (+3.6%) |
| k=1 complex correlation (wake) | — | 0.936 |

The wake's *geometry* and *phase* are right to within ~4%; the deficit is
almost purely in amplitude. That is a much better failure mode than a
wrong-wavelength or out-of-phase wake.

### Mean flow (k=0, no analytic prior — pure network)

R9 reproduces the reversed-flow recirculation bubble (centreline minimum
≈ −0.18 vs truth ≈ −0.17) which **R7 misses entirely**. Downstream it
under-recovers: at x = 8 R9 gives ū ≈ 0.35 against truth 0.76.

### Vorticity structure — the qualitative shortfall

In the animation (`figures/anim_w_truth_vs_R9.gif`) truth shows compact
alternating vortex cores detaching and convecting; R9 shows two largely
*continuous* shear layers with only weak lumpiness where cores belong. The
street is wavy rather than rolled up. Given that the kinematics are right
and k=1 amplitude is ~29%, the most likely cause is the amplitude deficit
itself (a weak fundamental cannot concentrate vorticity into cores) plus
truncation at k=2. This is a qualitative reading of one figure, not a
measurement.

---

## 5. Fixes forced by the run (all caught before wasting GPU time)

The notebook's smoke-test gate earned its cost three times:

1. **`DRIFT_CHECK_EVERY` patch target missing** — the smoke cell was
   inherited from R8's builder and patched a constant that exists only in
   `R8/src`. R9 is based on `src/pressure_only`, so the assertion fired
   immediately. Fixed by copying the file unpatched.
2. **`--LBFGSFtol` did not exist in the R9 base** — an R6-era flag that was
   never merged into `src/pressure_only`. The run command would have died on
   an argparse error *after* a ~40-minute smoke test. Ported the flag, and
   wired it through to `declare_LBFGS` (the base tree never passed it), plus
   `disp=True` so the scipy termination message is logged at all.
3. **The post-training diagnostics hung >10 min** — after `End of training`,
   the script evaluates `Loss_int_mode_wrap` for the first time; TF 1.14
   builds and optimizes that (street-enlarged) graph on first `sess.run`, and
   the smoke cell's stall guard killed it. Two changes: a new
   `--SkipDiagnostics` flag for the gate, and — more importantly — **the
   model save was moved to immediately after training.** In the original
   code the save sat ~70 lines *after* that diagnostics block, so a hang or
   runtime death there would have destroyed a completed 9-hour run's weights.
   That landmine was live in every previous run.

In the actual production run the hang recurred as expected; the run was
interrupted manually **after** the safety save, losing only
`Convergence_history.pickle` and the mode-shape PNGs (the loss trace is
recoverable from `out.txt`, which flushes per line — 44,955 entries).

---

## 6. Why it fell short of the testbed's ≲0.40 — the trust radius was too loose

With ρ = 0.6 and c = 0.12, the *lowest* amplitude the ansatz permits is
(1−ρ)|S_k| − c = 0.4|S_k| − 0.12. Measured against the street's own far-wake
amplitude:

| x/D | \|S_v1\| | permitted floor | truth \|v1\| |
|---|---|---|---|
| 1.5 | 0.292 | 0.005 (55% of points can still reach 0) | 0.334 |
| 3.5 | 0.411 | 0.044 | 0.541 |
| 5.5 | 0.396 | 0.039 | 0.522 |
| 7.5 | 0.381 | 0.033 | 0.489 |

The floor sits at **~7% of the true amplitude**, so the network was legally
free to shrink the wake to near-nothing — and it used most of that freedom,
settling just above the floor. The cage kept it off *exactly* zero (which is
why the wake exists at all) but left the door wide open. There is also a
genuine hole around x ≈ 1.5, where the formation ramp makes |S| small enough
that 55% of points can still reach zero.

**Conclusion: the mechanism worked as designed; the hyperparameter was
wrong.** The ansatz was never tested at a radius tight enough to matter in
the far field.

### Recommended R9b (not yet run)

- ρ ≈ 0.25–0.3, c ≈ 0.02–0.03 → far-wake floor ≈ 0.3 instead of 0.04, while
  still permitting the near-field correction the network clearly does well.
- Screen ρ ∈ {0.2, 0.3, 0.45} in the local testbed (~1 h) before spending GPU.
- Optionally compose with a warm start for k=0 only (R7's advantage), now
  that R9 shows the ansatz doesn't need it.
- Nmodes ≥ 4 if the goal is rolled-up cores rather than a wavy sheet.

---

## 7. Corrections to earlier claims in this project

Recorded because they were stated before being checked:

1. **"k≥2 is worse than predicting zero."** True in the *testbed*
   (near-wake k=2 err 2.185); **false for the production run**, where
   far-wake k=2 err = 0.503 is the best harmonic in the run. The production
   run also has no k=3 mode at all (Nmodes=3 → k=0,1,2).
2. **"Only R9 clears the no-wake line."** False — R4 also dipped below 1.0
   on both metrics (0.9668 / 0.9964). The defensible claim is *margin*:
   21–24% for R9 vs 0.4–3% for R4.
3. **"R9 sustains ~half the true amplitude."** Measured 29% (far-wake band
   mean 0.152 vs 0.519), not ~50%.
4. **Testbed reference numbers (0.64 / 0.40)** come from the `trust` arm
   *rerun after the image-vortex phase fix* (`arm_trust_v2.json`); the first
   pre-fix run scored E_v > 1 with its street phase ~π off and is superseded.

---

## 8. Analysis-tooling caveat

The figures and per-harmonic tables were produced with a numpy
reimplementation of the forward pass (`R9/analysis/eval_runs.py`), validated
against the official TF evaluator on all 30 region × quantity pairs of R7
and R9:

- **velocity: median 0.59%, worst 5.59% deviation** (worst case R9
  near-cylinder E_v, where the truth norm is small); ≤2.92% excluding the
  near-cylinder region.
- **pressure near the cylinder does NOT reproduce** — 58% off for R7, 84%
  for R9. Ruled out: ω mismatch, gauge offset, float32/complex64 precision.
  Cause unidentified.

Consequently **no pressure figure or pressure claim in this report comes
from that module** — every E_p value above is from the official
`evaluate_regions.py` output. The caveat is documented at the top of
`eval_runs.py`.

---

## 9. Files

Produced in this folder (nothing outside `R9/` and `runs/R9_extracted/`
was modified):

- `src/street_prior.py` — taps → closed-form street prior (numpy only)
- `src/NN_functions.py` — `street_modes_k` + the trust wrap
- `src/ModalPINN_VortexShedding.py` — `--TrustStreet/--StreetPrior/--TrustRho/--TrustCap`, `--SkipDiagnostics`, `--LBFGSFtol`, pre-diagnostics safety save
- `src/evaluate_regions.py` — trust-aware restore (must load the prior or it scores the wrong function)
- `notebooks/R9_trust_street_32taps.ipynb` — Colab notebook with the smoke gate
- `analysis/eval_runs.py`, `analysis/make_figures.py`, `analysis/make_gif.py`
- `figures/fig1_campaign_progress.png` — every run E3→R9 on one axis
- `figures/fig2_v_field_snapshots.png` — truth / R7 / R9 at one phase
- `figures/fig3_amplitude_and_trust_region.png` — amplitude profile + the ρ diagnosis
- `figures/fig4_mean_flow_centreline.png` — mean flow (no prior at k=0)
- `figures/anim_v_truth_vs_R9.gif`, `figures/anim_w_truth_vs_R9.gif`
- `runs/R9_extracted/.../regional_evaluation.txt`, `out.txt`, `street_prior_used.npz`, `DNN2_75_75_3_tanh.pickle`

---

## 10. One-paragraph verdict

R9 ported the trust-street ansatz into the production TF1 codebase and ran
it at full scale on 32 pressure taps plus physics, with no warm start and no
auxiliary loss terms. It ran 44,955 L-BFGS function evaluations over 8.8 h —
~3× R3 and ~10× R5, the only cross-run optimizer figure the logs preserve —
and produced the best reconstruction in the project: every region improved
15–47% over R7, the near-cylinder fit beat R7's warm-started result from
scratch, and the far-wake fluctuation error fell below the "no wake"
threshold by a real margin for the first time. The wake it produces has the
right wavelength, phase, and convection speed (within 4%) but only ~29% of
the true amplitude, because the trust radius ρ = 0.6 permitted amplitudes
down to ~7% of truth — a hyperparameter error, not a mechanism failure. The
untrained analytic prior still beats the trained network in the far field,
which keeps the honest headline as: *tap-derived classical theory
reconstructs the far wake; the network supplies the mean flow and near
field.* R9b with ρ ≈ 0.25–0.3 is the obvious next run.
