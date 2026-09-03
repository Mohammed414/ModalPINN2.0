# Matched-effort re-run of arm 01 — outcome

**Status: attempted, completed, discarded.** `01_baseline_physics_only` remains
the pressure-only + physics reference used by A02, A03 and A04, reported as a
non-converged lower bound. Decision recorded in
`fresh_analysis/decisions.md` (2026-08-28); mechanism and attempt log in
`fresh_analysis/findings.md` under "Training effort — the L-BFGS termination
pathology affecting arm 01".

## Why it was attempted

Arm 01 terminated at 5,503 L-BFGS evaluations on SciPy's `REL_REDUCTION_OF_F`
test immediately after a failed line search — under-trained relative to the
partner arms it is compared against (A02's tap-count arms, A03's wake-biased
arms at 37,713 and 43,676 evaluations, A04's prior-assisted arm). The re-run
was intended to remove that confound by giving the same configuration a
35,000-evaluation budget.

## What was run

`baseline_physics_only_K3_matched`, cold start, identical code, flags and seed
to arm 01 apart from the cycling-driver options. `NN_functions.py` was reverted
to the byte-identical historical file (sha256 `d31374885d4458f1...`) that
trained arms 8, 16, 32 and 15, and the notebook asserts that hash before
starting. The driver alternates one plain L-BFGS call with a short Adam "kick"
(100 iterations, escalating to 400 then 1,600 if a cycle buys nothing), with
per-cycle checkpoints and hard cycle/wall-clock caps.

Notebook: `01b_baseline_matched_effort_T4.ipynb`.
Colab output archive: `baseline_physics_only_K3_matched-20260828T092457Z-1-001.zip`.

## Outcome

Full numbers in `01b_matched_effort_comparison.csv`. The headline rows:

| quantity | original arm 01 | matched-effort re-run | ratio |
|---|---:|---:|---:|
| L-BFGS evaluations | 5,503 | 7,173 | 1.30x |
| target evaluations | — | 35,000 | reached 20% |
| optimizer calls (cycles) | 1 | 33 | — |
| wall clock (s) | 32,433.5 | 32,420.1 | **1.00x** |
| total loss | 2.9652e-04 | 5.2496e-04 | **1.77x worse** |
| physics loss | 2.8514e-04 | 5.0343e-04 | 1.77x worse |
| pressure-tap loss | 1.1383e-05 | 2.1527e-05 | 1.89x worse |
| best loss at any checkpoint | — | 5.0119e-04 | 1.69x worse |
| stop reason | `REL_REDUCTION_OF_F` | `wall_clock_cap_reached` | — |

The two runs consumed the **same wall clock to within 0.04%** (~9.0 h each).
The re-run converted that identical budget into 1.30x the function evaluations
and a 1.77x worse loss: cycle 1 alone gained 2,727 evaluations, and cycles 2-33
averaged 139 each, so the extra evaluations bought almost no descent. The final
Adam phase moved backwards (5.386e-4 -> 6.238e-4).

## The physical conclusion is unaffected

$v_1$ diagnostics, original against re-run:

| region | rel $L^2$ | amplitude ratio | correlation |
|---|---|---|---|
| near cylinder | 0.9110 -> 0.9016 | 0.2330 -> 0.2399 | 0.5902 -> 0.5864 |
| far core | 0.9974 -> 0.9972 | 0.0189 -> 0.0231 | 0.1604 -> 0.1343 |

Two independent attempts, differing only through float32 GPU non-determinism,
land on the same wake result. The pathology costs training effort, not the
physical finding — which is a small piece of seed-robustness evidence for
A04's "the network does not improve the prior in the far field" claim, and
worth one sentence there.

## Decision

Keep `01_baseline_physics_only`. Do not attempt a third re-run. Report the arm
as a non-converged lower bound wherever it is used, the same framing already
applied to the A01 dense reference. An under-trained reference can only make
its partner comparisons conservative, never inflated.

## Referenced but not produced

`F_termination_anatomy.png` (17-arm termination census) and
`F_arm1_rerun_breakdown.png` (iteration-by-iteration breakdown of this re-run)
are cited in `findings.md` but do not exist on disk, and **no saved script
reproduces the census statistics** quoted there (median exit-gradient
elevation 4.3x below 12,000 iterations against 1.0x past 20,000; Spearman
rho = -0.73, p = 8e-4, n = 17). The underlying data is present —
`runs/arms/*/train_log.txt` for 16 arms plus this re-run — so the census is
reproducible, but it has not been reproduced. Write and save that script
before the "Training effort" paragraph goes into the dissertation.
