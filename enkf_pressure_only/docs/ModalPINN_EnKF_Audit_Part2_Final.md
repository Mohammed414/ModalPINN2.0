# ModalPINN EnKF Audit, Part 2 — Final Evaluation and the Observability Answer

Branch `enkf-repairs`. Re=100 cylinder wake, 32 wall pressure taps, no velocity
data anywhere. Domain [-4,8]x[-4,4], omega_0 = 1.036.

Every number below was recomputed for this report from the run archives; none
is copied from a previous log. The repaired nominal run was re-executed from
scratch (`stage_d2_enkf_nominal_repro.npz`) and reproduces the original
**bitwise** (max abs difference 0.0 on gain fraction, innovation, mean fields,
gamma and NIS).

---

## 1. Headline verdict

**The 32 wall pressure taps carry enough information to correct the wake's
TIMING out to at least x = 8. They do not measurably correct its AMPLITUDE,
because the forward solver's amplitude was already close to correct — there was
no amplitude deficit at the sensors for the filter to fix.**

The repaired filter separates from both negative controls, but only on the
metric that combines amplitude and phase, and on frequency. On amplitude alone
it does not beat the free run stably. That is stated plainly because the
audit's central complaint about the original work was exactly this kind of
unstable separation.

---

## 2. Stage E2 — the observability diagnostic against its null

The original Stage E reported "only ~6 of 16 directions visible in wall
pressure". **That conclusion was an artefact of the ensemble, not a property of
the sensors**, and the rebuilt null proves it three separate ways.

| ensemble / observation | n_eff | interpretation |
|---|---|---|
| jitter-only ensemble, observed through **32 taps** | **1.010** | the Stage E result |
| jitter-only ensemble, observed through the **ENTIRE state vector** (18 844 DOF) | **1.013** | *same answer with nothing hidden* |
| repaired ensemble, observed through 32 taps | 2.656 | |
| repaired ensemble, observed through the entire state | 1.714 | |
| repaired ensemble, 32 random state functionals | 1.895 | |
| synthetic perfectly-observable system | 14.947 | what a rich system looks like |

The decisive row is the second. Observing **every one of the 18 844 state
degrees of freedom** of the phase-jittered ensemble recovers n_eff = 1.013 —
essentially identical to the 1.010 that 32 pressure taps recover. An
observation operator that hides nothing at all produces the same steep
spectrum. The Stage E decay therefore measured the rank of the ensemble, which
was 1 by construction (16 time-shifts of one trajectory), and said nothing
about wall pressure.

The numerical-floor complaint also holds: everything past index 7 in the
original Stage E spectrum sat at 3.75e-15 relative, below the Gram-matrix
round-off floor of 1.49e-8.

### With a genuinely multi-direction ensemble

Whitening by sigma_p = 0.0472 makes the pressure singular values an absolute
detectability scale — a value of 1 means one noise standard deviation of tap
signal.

- **11 of 15** ensemble directions produce pressure singular values above
  1 sigma_p (largest 19.2, eleventh 1.19).
- Ordering-free version (visibility per state direction, which does not assume
  the i-th pressure direction matches the i-th state direction): **15 of 15**
  spanned directions produce above-noise pressure signal, from 16.0 sigma_p
  down to 1.27 sigma_p. The 16th is the ensemble-mean null direction and is
  exactly zero by construction.
- The 15 non-trivial directions all sit above the numerical floor.

**Answer: with a genuinely multi-direction ensemble, every direction the
ensemble spans is visible in wall pressure above the noise level. Nothing is
hidden from the taps in the ensemble subspace. The information deficit reported
in Stage E does not exist.**

Caveat, stated because it bounds the claim: 15 directions is the rank of a
16-member ensemble, not the rank of the flow. This says the taps see everything
*this ensemble can express*; it does not certify observability of the full
18 844-dimensional state.

---

## 3. Stage F2 — evaluation with metrics_v2

Common window: absolute cycles 61–200, t = 406.1–420.0, 140 samples,
**13.9 time units = 2.29 periods of omega_0**.

### 3.1 Phase-aligned field error and the fitted time shift, separately

| run | E_v aligned | E_v unaligned | tau* (t.u.) | phase lag (rad) |
|---|---|---|---|---|
| free run | 0.3315 | 0.4411 | -0.3042 | -0.315 |
| original EnKF | 0.3059 | 0.5744 | -0.4609 | -0.478 |
| **repaired EnKF** | **0.1937** | **0.2585** | **+0.1713** | +0.177 |
| repaired, shuffled | 0.4366 | 0.4559 | -0.1685 | -0.175 |
| repaired, scrambled | 0.3040 | 0.6561 | +0.6894 | +0.714 |

The repaired filter has both the lowest aligned error and much the smallest
fitted time shift, and it is the only run whose unaligned and aligned errors
are close together — i.e. the only one that is nearly in phase with the truth
without being shifted.

### 3.2 Mode-1 and mode-2 amplitude versus downstream distance

Truth reference (recomputed): |v1| peaks at **0.6522 at x = 2.90**, is **0.5262
at x = 7**, persistence 0.807. |v2| peaks 0.1345 at x = 3.95, persistence 0.762.

|v1| max-over-y, selected x:

| x | truth | free run | orig EnKF | **repaired** | shuffled | scrambled |
|---|---|---|---|---|---|---|
| 1.02 | 0.297 | 0.290 | 0.308 | 0.319 | 0.152 | 0.306 |
| 2.83 | 0.652 | 0.612 | 0.652 | 0.676 | 0.449 | 0.625 |
| 4.62 | 0.592 | 0.567 | 0.598 | 0.592 | 0.443 | 0.545 |
| 7.03 | 0.526 | 0.521 | 0.552 | 0.524 | 0.473 | 0.544 |

**The critical observation: the free run — which assimilates nothing at all —
already reproduces the truth's |v1| profile to within 5%** (mean amplitude
ratio 0.955 over x >= 1, persistence 0.825 vs truth 0.807). There is no
ModalPINN-style amplitude collapse in the forward solver to begin with, so
there is no amplitude deficit for the filter to repair. Mode 2 is the one place
the filter clearly helps the amplitude (ratio 0.969 vs free run 0.593).

### 3.3 Amplitude error and phase error per mode as functions of x

Mode 1, averaged over x >= 1:

| run | amplitude ratio | amp. error | mean phase error |
|---|---|---|---|
| free run | 0.9553 | 0.0507 | +21.4 deg |
| original EnKF | 1.0046 | 0.0320 | +32.1 deg |
| **repaired EnKF** | 1.0078 | 0.0352 | **-11.9 deg** |
| shuffled | 0.6948 | 0.3052 | +10.5 deg (but std 11.7 deg) |
| scrambled | 0.9369 | 0.0687 | -38.0 deg |

Paired Wilcoxon over x, repaired EnKF versus each:

| comparison | amplitude | phase |
|---|---|---|
| vs free run | better, p = 3.3e-4 | better, p = 2.8e-14 |
| vs original EnKF | **not distinguishable, p = 0.41** | better, p = 5.6e-17 |
| vs shuffled | better, p = 5.6e-17 | **not distinguishable, p = 0.58** |
| vs scrambled | better, p = 2.0e-4 | better, p = 5.6e-17 |

Note the two failures, which matter: the repaired filter does **not** beat the
original filter on amplitude, and it does **not** beat the *shuffled* control on
phase. Each control fails on a different axis — shuffled destroys the
amplitude, scrambled destroys the phase — so neither single-axis metric
separates the filter from both at once.

### 3.4 Sub-window stability — and whether the window is long enough

**It is not.** The window is 13.9 t.u. = 2.29 periods. The beat period between
each run's own shedding frequency and omega_0 is 2*pi/|omega_est - omega_0|:

| run | omega_est | beat period | window / beat |
|---|---|---|---|
| free run | 1.1713 | 46.5 t.u. | 0.30 |
| original EnKF | 1.1483 | 55.9 t.u. | 0.25 |
| repaired EnKF | 1.0119 | 260.8 t.u. | 0.05 |
| shuffled | 0.9778 | 107.9 t.u. | 0.13 |
| scrambled | 0.9499 | 72.9 t.u. | 0.19 |

**No run covers even a third of a beat period.** The task asked for evaluation
over a full beat period; that is not achievable with these 140 saved cycles and
saying otherwise would be wrong. Thirds of the window are 0.76 of a period —
shorter than one oscillation — so a k=1 harmonic fit is not identifiable there
and those columns are diagnostic only.

Because three window cuts are too few to establish stability, a **40-position
sliding-window analysis** (1-period window, 0.2 t.u. steps) was run instead.
Fraction of positions at which the repaired filter wins:

| claim | amplitude only | phase only | joint (amp+phase) |
|---|---|---|---|
| vs shuffled control | 1.00 | 0.58 UNSTABLE | **1.00** |
| vs scrambled control | 0.33 UNSTABLE | 1.00 | **1.00** |
| vs original EnKF | 0.93 UNSTABLE | 1.00 | **1.00** |
| vs free run | 0.85 UNSTABLE | 0.80 UNSTABLE | 0.85 UNSTABLE |

Single-axis rankings flip with window position exactly as the original Stage F
ranking did. The **joint** per-x complex error |c(x) - 1|, which charges
amplitude and phase together, is stable against all three comparators and only
against those three.

### 3.5 The one window-independent measurement

Phase error grows linearly in time at a rate equal to the frequency error, so
fitting its slope across the 40 sliding positions measures omega without
depending on where any window was cut. Residual scatter about the fit is
0.2–1.4 deg, so the fits are tight:

| run | implied omega | error vs omega_0 |
|---|---|---|
| free run | 1.1704 | +12.98% |
| original EnKF | 1.1493 | +10.94% |
| **repaired EnKF** | **1.0093** | **-2.57%** |
| shuffled | 0.9702 | -6.35% |
| scrambled | 0.9536 | -7.96% |

**The repaired filter cuts the frequency error from 13.0% to 2.6% — a 5x
reduction — and is closer to the truth than either control.** This is the
cleanest positive result in the study and the only one immune to the
window-length problem.

---

## 4. The observability answer

**How far downstream are the oscillating modes recovered above the free-run
baseline?**

On amplitude: **nowhere, stably.** At 0 x-nodes does the repaired filter beat
the free run's |v1| amplitude at more than 95% of sliding-window positions.
Averaged over the full window it does beat it (error 0.0352 vs 0.0507,
p = 3.3e-4), but that ranking holds at only 85% of window positions and
reverses in the second half. The honest statement is that the free run's
amplitude was already good (ratio 0.955) and the filter did not reliably
improve it.

On timing: **out to at least x = 8, the full reconstruction domain.** Mean
mode-1 phase error falls from +21.4 deg (free run) to -11.9 deg, the
improvement holds across the entire x range, and the underlying frequency error
falls 5x. The phase improvement over the free run is significant at
p = 2.8e-14 and holds at 80% of sliding positions — better than the amplitude
claim but still not the >95% I would require to call it stable.

**Does the repaired filter separate from BOTH negative controls on the wake
metrics, not just innovation?**

**Yes, but only on the joint amplitude-and-phase metric.** It beats both
controls at 100% of 40 sliding-window positions on |c(x) - 1| (mean 0.193 vs
0.374 shuffled and 0.612 scrambled; worst case 0.327 vs 0.895 for the free
run). On either axis alone the separation fails: shuffled ties it on phase
(p = 0.58), scrambled beats it on amplitude at 67% of positions. Innovation
separates at p ~ 1e-44, but I would not have accepted that alone, and it is not
what the answer rests on.

**Not repeated from the original work**: the filter genuinely does separate
from both controls here, unlike Stage D. But the separation is narrower than
the innovation statistic suggests, and it is carried by phase and frequency,
not by wake amplitude.

---

## 5. What this implies for the ModalPINN mode collapse

**It points to a training-signal problem rather than an information deficit at
the sensors — but the evidence is weaker than that sentence sounds, and the
caveat below is not optional.**

The supporting evidence:

1. The 32 taps are not information-starved. Every direction the ensemble spans
   produces above-noise pressure signal (15 of 15, 1.27–16.0 sigma_p), and the
   Stage E "6 of 16" deficit was an ensemble artefact, not a sensor property.
2. Wall pressure demonstrably carries the wake's *timing*: an estimator using
   only those taps cut the frequency error 5x and the phase error by ~2x, out
   to x = 8.
3. The evaluation metric was itself blind to the failure mode. Stage F's E_v
   scores a **deleted** wake 0.906 against 1.299 for a correctly-reproduced one
   — it actively prefers no vortex street. All five metrics_v2 measures are
   minimised at the truth and return exactly 1.0 at scale 0. A collapsing wake
   was being *rewarded* by the old objective, which is the same structural
   blindness suspected in the PINN's collocation loss.

**The caveat, which materially limits the inference: the EnKF has a full
Navier–Stokes solver as its prior and ModalPINN does not.** The filter never
had to discover that a vortex street persists downstream — its forward model
produces one unprompted, as the free-run column shows (|v1| ratio 0.955 with
zero assimilation). All the taps had to supply was a timing correction to an
already-correct spatial structure. ModalPINN, by contrast, must *construct* the
mode shapes from the residual loss, and this study provides **no evidence** that
32 wall pressure taps suffice for that harder task. "The EnKF can do it" does
not transfer.

What this study does establish for ModalPINN: the failure is unlikely to be
attributable to the sensors carrying no downstream information, and any fix
should be validated with an amplitude-sensitive metric, because the natural
pointwise ones are minimised by the collapse they are meant to detect.

---

## 6. Files

Written this session (no existing file overwritten):

- `experiments/stage_e2_observability.py`, `.npz`, `.log`
- `evaluation/stage_f2_metrics_v2.py`, `experiments/stage_f2_metrics_v2.{json,npz}`
- `evaluation/stage_f2_sliding_stability.py`, `experiments/stage_f2_sliding_stability.{json,npz}`
- `experiments/stage_d2_enkf_nominal_repro.npz`, `.log` (bitwise reproduction check)
- `figures/stage_d2_gain_and_innovation.png`
- `figures/stage_e2_observability_vs_null.png`
- `figures/stage_f2_mode1_vs_x.png`
- `figures/stage_f2_window_stability.png`
- `figures/metric_damping_test.png`

## 7. Open items

- The 140-cycle window is too short for a beat-period evaluation. Re-running
  the observer for >= 50 time units would let every claim here be tested over a
  full beat; until then the amplitude rankings should be treated as unresolved.
- The gamma (frequency) augmentation result remains negative in the strict
  sense: gamma ends at 0.9247 +/- 0.0574 versus shuffled 0.9537 +/- 0.1903
  (p = 0.688). The *field* frequency improves 5x, but the augmented parameter
  itself is not distinguishable from its controls, and those two facts have not
  been reconciled.
