# Completed runs

Seven arms done, nine queued. All at `--Nmodes 4` (k = 0,1,2,3), cold start, `--Tmax 9`,
`Nint 50000`, `Nmes 5000`, `WidthLayer 25`, `Seed 0`, freestream inlet prior on.
Every one exited 0 with L-BFGS converged to a flat loss tail inside the 50,000-evaluation cap.

| folder | what it changes from arm 1 | wake amp | wake corr | lift phase err |
|---|---|---|---|---|
| `01_baseline_physics_only`      | — (the reference)                          | 0.01895 | 0.16036 | -4.80° |
| `02_wall_vorticity_flux`        | `--BVF`, fluctuation inlet BC **on**       | 0.08945 | 0.10809 | -6.35° |
| `03_karman_prior_fluct_on`      | `--V1RadialTrust`, fluctuation BC **on**   | 0.80878 | 0.97159 | -5.65° |
| `04_paper_sparse_probes`        | drops `--PressureOnly` → +40 velocity probes | **0.86578** | **0.99193** | -5.88° |
| `07_wake_biased_grid`           | `--WakeBiasedGridSampling`                 | 0.25404 | 0.10256 | -6.53° |
| `15_karman_prior_fluct_off`     | `--V1RadialTrust` only                     | 0.83875 | 0.97658 | **-5.56°** |
| `16_prior_no_inlet_bc_no_adam`  | prior, **no** inlet BC at all, `--SkipAdam` | 0.82667 | 0.97470 | -5.68° |

Reference values: analytical prior alone, no network = 0.8082. Anything below 0.10 is collapsed.

## Which files to trust

| file | trust |
|---|---|
| `v1.json` → `far-core`, `far-wake` | **yes** — the inlet ramp is 9.4e-14 at x=3 |
| `v1.json` → `near-cylinder`, `near-wake`, `whole-domain` | **no** — these span the ramp, which both v1 and physics evaluators force on regardless of training |
| `regions.json` | **yes** — the only evaluator that handles the inlet flags correctly |
| `physics.json` | present only for arms 03, 15, 16 (prior-active). Absent by design elsewhere, not a failure |
| `training_run/DNN2_100_100_4_tanh.pickle` | the checkpoint; layer shape confirms k = 0,1,2,3 |
| `street_prior_used.npz` | the exact prior that run used (arms 15 onward only) |

## Still queued

05 dense reference · 06 wake-biased random · 08 taps=8 · 09 taps=16 ·
10 prior + wake-biased grid · 11-13 prior + noise 1/5/10% · 14 BVF fluctuation off

`zips/` holds the original Drive downloads, renamed to match.
