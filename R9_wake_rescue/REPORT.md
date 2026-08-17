# R9_wake_rescue — Report

**Goal:** reconstruct the unsteady Re=100 cylinder wake from ONLY the 32
wall pressure taps + physics. No PDE solver, no POD, no EnKF, no reuse of
R1–R8's loss terms. Reference CFD used only for tap extraction, one clearly
marked diagnostic (contrast measurement), and final validation.

**Outcome: the wake is reconstructed.** Far-wake E_v drops from ~1.00
(every prior attempt; 1.0 = predicting no wake at all) to **0.40**, and
near-wake E_v from ~0.96 to **0.64**, using a street-anchored trust-region
ansatz built entirely from tap-derived scalars and classical physics.
Result is stable to tap noise (1%, 5%), to halving the sensor count (16
taps), and across network seeds (far-wake E_v 0.376–0.384 everywhere).

---

## 1. Why every previous run failed (now measured, not hypothesized)

The project's hypothesis was that a dead wake nearly satisfies the
pointwise NS residual. Measured on a family of progressively-killed wakes
(`src/contrast_ratios.py`, truth used diagnostically):

- The k≥1 momentum residual in the wake changes only **~5x** between the
  true field and the observed collapse profile (dead within ~1 D). At the
  weights used in R1–R8 that is invisible next to the tap/BC terms.
- Deeper: the steady base flow with zero oscillation is an EXACT NS
  solution at Re=100. The optimizer isn't "failing" — it is correctly
  finding a genuine spurious minimum that only the 32 surface taps oppose,
  locally.

Loss-landscape probe (`src/landscape_probe.py`): under the baseline
objective the dead solution scores **430x better** than the true field.
Under the amplitude-normalized / RZIF / lift-anchored objectives the
ordering flips (truth 4–18x better than dead). The landscape can be fixed —
but see §3: fixing the ordering alone was not enough.

## 2. What the taps give for free (`src/tap_anchors.py`)

From the 32 tap signals alone: omega0 = 1.0357 (0.02% off the project's
1.036), CL k=1 harmonic |L1| = 0.143 (amplitude & phase), pressure drag
CD_p = 0.989, all 32 per-tap harmonic coefficients k=0..3.

## 3. Controlled arms (testbed: faithful PyTorch ModalPINN, `src/modal_pinn.py`)

The testbed reproduces the failure exactly (near-wake E_v = 0.996 from
taps+physics only) — validity gate passed.

| arm | near-wake E_v | far-wake E_v | verdict |
|---|---|---|---|
| baseline (ModalPINN loss) | 1.00 | 0.99 | dead (control ✓) |
| + relative residual | 1.09 | 1.01 | dead |
| + RZIF (mean-linearized k=1) | 1.14 | 1.00 | dead |
| + lift-anchored CV budgets | 1.07 | 1.01 | dead |
| + hard shift-reflect symmetry | 0.99 | 0.99 | dead |
| + all combined | 1.21 | 1.05 | dead |
| street pretraining init (soft) | 1.03 | 1.00 | dies again |
| **trust ansatz around street** | **0.64** | **0.40** | **alive** |

Lesson: re-ordering the loss landscape (better objectives) or starting
near a live wake (soft init) is insufficient — gradient descent still
walks back into the dead basin. What works is **removing the dead basin
from the search space**.

## 4. The winner: analytic-street trust ansatz

### 4.1 The prior (`src/analytic_street.py`) — classical physics, taps-only
Von Kármán point-vortex street, Lamb–Oseen cores with viscous growth,
Milne-Thomson image vortices (cylinder surface = streamline), potential
dipole for the mean. Parameters, all from taps + classical relations:
- spacing/advection: a = 2π·Uc/ω₀ with Uc = 1 − Γ/(√8·a) (self-consistent)
- circulation Γ = 2.53 from the Kármán drag formula = measured CD_p/0.75
- formation point/core size from the tap k=1 pressure pattern (shape only)
- temporal phase from the same pattern — the image system is what makes
  the surface pressure pattern orientation-aware (without images the fit
  picks a phase ~π off; with them the phase error vs truth is ≤0.2 rad
  past x≈3, and the standalone street already hits far-wake E_v = 0.40,
  |v1| pattern correlation 0.99).

### 4.2 The ansatz (`src/modal_pinn.py::TrustModalPINN`)
k=0: free network (street has no boundary layer — mean flow is learned).
k≥1: q_k(x,y) = S_k(x,y) + (ρ|S_k| + c)·(tanh a_k + i·tanh b_k),
ρ=0.6, c=0.06. The network corrects the street by a bounded amount;
**q_k = 0 is outside the search space** wherever the street is alive.
Trained with the plain ModalPINN loss (physics + taps + BC) — no exotic
terms needed once the ansatz is right.

### 4.3 Robustness (`src/robustness.py`, full pipeline re-derivation per case)
| perturbation | far-wake E_v |
|---|---|
| clean 32 taps | 0.400 |
| 1% tap noise | 0.382 |
| 5% tap noise | 0.382 |
| 16 taps | 0.382 |
| seed 1 / seed 2 | 0.384 / 0.376 |

### 4.4 Autoencoder variant (user-requested, `src/decoder_family.py`)
Decoder trained on the analytic street family (never CFD/solver data),
latent inverted from taps + divergence penalty. Recovers plausible street
parameters but lands phase-ambiguous (near-wake E_v 1.12): the tap k=1
pattern alone, through a generic learned decoder, cannot resolve the
half-period ambiguity that the image-vortex physics resolves. Negative
result worth keeping: physics-structured prior > learned prior here.

## 4.5 Attribution: how much is the analytic prior vs the network?

Measured per mode and region (`results/attribution_prior_vs_network.csv`;
`net_move_rel` = ||q_trained - S_prior|| / ||S_prior||):

| region | mode | network's move | E(prior alone) | E(trained) |
|---|---|---|---|---|
| far-wake x>3 | v1 | 0.12 | **0.316** | 0.359 |
| far-wake x>3 | v2 | 0.05 | 0.677 | 0.693 |
| near-wake 0<x<3 | v1 | 0.42 | 0.838 | **0.621** |
| near-cyl r<0.75 | v1 | 0.85 | 1.536 | **1.024** |
| all | k=0 mean | no prior (free net) | 1.914 (street's own mean, near-cyl) | **0.859** |

Read honestly:
- **The far-wake oscillation is essentially the analytic solution.** The
  network moves it 5-12%, and the prior alone is marginally BETTER there
  (0.316 vs 0.359). The headline far-wake E_v ~ 0.40 is attributable to the
  Karman relations fed by tap measurements, NOT to network learning.
- **The near wake / formation region is where the network earns its place**
  (moves the prior 42%, improves E_v 0.84 -> 0.62) - exactly where the
  idealized street is wrong (finite formation length, no boundary layer).
- **Near the cylinder the network overrides the prior almost entirely**
  (85% move, at/near the trust bound), correctly.
- **The mean flow is 100% network** (no prior is supplied at k=0) and beats
  the street's own crude mean near the cylinder by >2x.

So the correct description of this method is a **hybrid**: an analytic
kinematic skeleton for the oscillating far field + a network for the mean
flow and near field, with the network structurally forbidden from deleting
the skeleton. It is neither "a PINN that learned the wake" nor "just an
analytic formula".

Additional finding from the same table: the **higher harmonics are not
reconstructed** - near-wake E for v2/v3 is 2.19/5.09 (worse than predicting
zero). k=1 carries most of the fluctuation energy so the regional totals
still improve, but any claim about k>=2 mode shapes is unsupported.

## 5. Honest limitations

- Far-wake E_p is poor (2.8) — the ansatz's pressure gauge/mean pressure
  far downstream is weakly constrained; velocity is the deliverable here.
- Only the k=1 harmonic is reconstructed; k=2,3 mode shapes are worse than
  zero in the near wake (see 4.5). The reconstruction is effectively
  "mean flow + fundamental", not a full 3-harmonic field.
- Credit split is uneven: the far-wake result is the analytic prior's, not
  the network's (see 4.5). State this explicitly in any write-up.
- Near-cylinder E_u (~0.86) is worse than R7's warm-started 0.084 — R7
  bought that with a checkpoint from R3; this testbed trains from scratch
  with a smaller net and shorter budget. The two are composable (see §6).
- Small-scale testbed: width 40 vs 75, 3k vs 50k collocation points,
  ~2.8k iterations vs ~15k. Numbers should improve at production scale.
- ρ, c (trust radii) were set a priori, not tuned; the a-priori values
  worked across all robustness cases.
- The trust ansatz asserts the wake exists — justified here by measured
  finite lift oscillation at the taps. It would be the wrong prior for a
  flow with no periodic shedding (that assertion is checkable from taps).
- Harmonic-balance normalization: this testbed's convolution terms were
  written from the product-to-sum identity directly (the k=0/k≥1
  normalization factors the R6 audit flagged in the TF1 code are correct
  here by construction).

## 6. Recommendation for the production R9 run

Port `TrustModalPINN` to the TF1 codebase (a contained change to
`out_nn_modes_uv/p`):
1. Fit the street from the taps exactly as in `src/analytic_street.py`
   (runs in seconds, pure numpy — reuse the saved `street_fit.npz`).
2. Wrap the existing mode networks: q_k = S_k + (0.6|S_k| + 0.06)·tanh(·)
   for k≥1, evaluated at the collocation points (street values can be
   precomputed per point set — no autograd through the street needed if
   the envelope derivatives are cached, see `modes_and_derivs`).
3. Keep the standard loss (physics + taps + FSBC). Drop BVF/K0/PhaseLoss.
4. Optionally warm-start k=0 from R3's checkpoint for the near-cylinder
   fit (composable: k=0 warm start + k≥1 trust anchor address different
   regions' errors).
Expected: far-wake E_v ≤ 0.4 at production scale, vs 1.0 for R1–R8.

## 7. File map

- `src/common.py` — loaders + the standard regional-eval protocol
- `src/tap_anchors.py` → `cache/tap_anchors.npz` — tap-derived quantities
- `src/contrast_ratios.py` → `results/contrast_ratios.csv` + `_NOTES.md`
- `src/modal_pinn.py` — testbed (ModalPINN ansatz, candidate losses,
  TorchStreet, TrustModalPINN)
- `src/run_arm.py` → `results/arm_*.json/.npz/.pt` — controlled arms
- `src/landscape_probe.py` → `results/landscape_probe.json`
- `src/analytic_street.py` → `cache/street_fit.npz` — the analytic prior
- `src/eval_street.py` → `results/street_standalone.json`
- `src/phase_sweep.py` → `results/phase_sweep.json` (superseded by the
  image-vortex phase anchor, kept for the record)
- `src/decoder_family.py` → `results/decoder_inv.json` — autoencoder arm
- `src/robustness.py` → `results/robustness_*.json`
- `src/final_figures.py` → `figures/fig1..3`
- All large regenerable intermediates in `cache/` (gitignore-worthy).

*Every training input in this folder derives from `taps_32.npz` and
classical physics; `data/fixed_cylinder_atRe100` appears only in
evaluation code paths. Produced 2026-08-13 in the Claude Science session.*
