# R7 — Is the full alternating EKI-ModalPINN method worth building?

Companion to `R7_EKI_ModalPINN_observability.ipynb`. Every number below is produced by that
notebook; nothing here is asserted independently of it.

## Answer

**Probability that the full method meaningfully improves downstream wake reconstruction: ~25%.**

Not because the Kalman update fails — it works, and cleanly. Because the diagnostic says the
method as proposed corrects the wrong quantity.

## The four measured results

| # | Question | Result |
|---|---|---|
| 1 | Does the EKI machinery work? | Yes. Global α₁ recovered 0.904 from a corrupted 0.5 (target 1.0). Expected; bears on nothing that was in doubt. |
| 2 | How far downstream can 32 taps constrain wake **amplitude**? | Sensitivity crosses the noise floor at **x/D = 6.14**. Per-band sensitivity falls **50.6×** from x/D=1.5 to x/D=5.5. |
| 3 | Is the **phase gradient** observable? | Yes, cleanly — δk_x recovered to 0.016 from a corrupted 0.60. Per-unit sensitivity **1.27×** the nearest amplitude band. Scaled by the checkpoint's real deficits, wall signal is **56× noise** (wavenumber) vs **51× noise** (amplitude) — ratio **1.10×**. |
| 4 | Is the absorption defect real? | Yes. ≥20% of an imposed amplitude correction is given back by a subsequent shape update. Lower bound (3-dof surrogate). |

Band recovery, each corrupted to 0.5 with target 1.0:

| band | centre | recovered | verdict |
|---|---|---|---|
| a₁ | x/D = 1.5 | **0.9998** | fully recovered |
| a₂ | x/D = 3.5 | **0.9884** | recovered |
| a₃ | x/D = 5.5 | **0.6844** | only 37% of the gap closed |

The predicted ordering is confirmed. The taps see the near wake essentially perfectly, the mid
wake well, and the far wake poorly.

## Why 25% and not higher

The observability result is *better* than the pessimistic hypothesis: usable amplitude
information extends to about 6 diameters, not 1–2. If downstream amplitude were the binding
constraint, a per-band EKI correction would be well-posed over most of the reconstruction domain
and the answer would be optimistic.

It is not the binding constraint. From §1.7 of the notebook, the trained checkpoint has:

- peak |v̂₁| = **0.037** against a true **0.652** — a 17× amplitude deficit;
- streamwise wavenumber **0.201** against a true **1.581** — **13%** of the correct value.

The second number is the important one. A wake that is merely too weak has the right spatial
structure at the wrong scale, and an amplitude correction fixes it. A wake with the wrong
wavenumber has **no vortex street at all** — the phase is nearly flat in x, so there is nothing
for α₁(x) to scale up. Multiplying an absent structure by a larger number leaves it absent.

So the method's correction parameters (per-band amplitudes) address a symptom that is real but
secondary, while the parameter that addresses the actual defect (δk_x) is the one the proposal
treats as a minor addition. That mismatch, not any failure of the filter, is what caps the
estimate.

Note that this argument does **not** rest on the phase gradient being more observable than
amplitude — measured against the checkpoint's actual deficits it is only 1.10× stronger at the
wall, which is not a meaningful margin. It rests on the fact that an amplitude correction has
nothing to act on when the phase is flat.

## What would change the answer

**Upward, to ~60%, if the parameterisation is reordered.** Make δk_x(x) — a spatially varying
wavenumber correction — the primary state, with amplitude bands secondary. This is a change to
the proposal, not a rejection of it: the EKI machinery, the observation operator, and the band
construction all carry over unchanged.

The argument for this is **physical, not observability-based**, and it is worth being precise
because an earlier draft of this memo got it wrong. Both deficits are strongly detectable at the
wall — 51× noise for amplitude, 56× for wavenumber, a ratio of only 1.10×. Observability does
*not* select between them. The reason to prioritise the wavenumber is §1.7: a near-flat phase
means there is no vortex street present for an amplitude factor to scale. Fixing amplitude first
would make an absent structure larger, not correct.

**Downward, to ~10%, if the absorption defect is not addressed.** The alternating scheme's loss
sees only the product α₁·q̂₁. The measured ≥20% give-back is a lower bound from a 3-dof
surrogate; a full 11,403-weight network has far more freedom to reshape q̂₁ at constant tap
misfit. The obvious fix — constraining the mode-shape norm — was tested and **does not work**
(it produced a −27% "absorption", i.e. the constrained optimiser inflated the shape rather than
preserving it). A working fix needs to be designed and tested, not assumed.

## Three ways this estimate could be wrong

1. **The linearised observation operator.** Wall pressure is obtained by solving the
   mean-flow-linearised pressure Poisson equation. It validates at |corr| = 1.000 against the
   true mode-1 wall pressure, so it is sound at Re=100 where the wake is weakly nonlinear — but a
   strongly nonlinear coupling would change the downstream reach.
2. **The base state.** Observability was measured around the **true** modal field, which is the
   right choice for "what can the sensors see in principle." Gains around the *collapsed* PINN
   field would differ, and are what the method would actually encounter in its first iterations.
3. **The surrogate in §5.** It demonstrates the absorption mechanism but not its magnitude in the
   full method. If the true absorption is near-total, the estimate drops regardless of the
   parameterisation.

## Recommendation

Do not build the full alternating method as specified. Instead:

1. Re-run this notebook's §4 with a **spatially varying** δk_x(x) on the same band basis, to
   confirm the phase gradient stays observable when it is allowed to vary downstream. This is a
   one-cell change and costs minutes.
2. If it does, rebuild the proposal around phase-gradient correction with amplitude secondary.
3. Design and test a fix for the absorption defect before committing to the alternating loop.
   The norm constraint tested here is not it.

The cheapest decisive experiment remains the one from the Part 2 EnKF audit: test the ModalPINN
loss directly for whether it prefers a deleted wake to a mistimed one. If it does, that is the
mechanism generating the flat-phase failure documented above, and no amount of downstream
correction will hold against a training objective that is actively removing the structure.
