# Branch `enkf-repairs` — change summary

Generated 2026-08-11T21:38:54Z. Nothing is committed; review before merging.

Base: main @ f313780 (Add enkf_pressure_only Stages B-F: solver, EnKF, observability, evaluation)

## Read this first

Nothing here is committed. `main` is untouched: every original Stage A–F `.npz` is intact and
remains the evidence base for the Part 1 audit. No existing result file was overwritten.

**What to review, in order:**

1. `estimator/ns_solver.py` — the ONLY modified existing file. Adds `sample_pressure(...,
   method='wall_probe')` (quadratic normal-direction extrapolation, reads 0 solid cells) as the
   new default; `method='bilinear'` retains the original behaviour for comparison.
2. `estimator/enkf2.py` + `estimator/ensemble_init.py` — the repaired filter: per-tap bias
   removal, sigma_p from the unsteady signal, divergence-free multi-direction ensemble,
   shedding-rate (gamma) augmentation.
3. `evaluation/metrics_v2.py` — evaluation metrics that are not minimised by deleting the wake.
   `evaluation/METRICS_V2_NOTES.md` explains the design.
4. `experiments/stage_d2_summary.json` and `experiments/stage_f2_metrics_v2.json` — the numbers.

**Headline results** (all independently recomputed from the saved `.npz` files):

| quantity | before | after |
|---|---|---|
| Kalman gain fraction | 2.4e-4 | 0.121 |
| static innovation fraction | 0.88 | 0.040 |
| ensemble effective directions | 1.01 | 5.86 |
| E_v vs free run (cycles 61–200) | +7.2% (worse) | −54.1% |
| shedding-frequency error | 13.0% | 2.6% |

**Two findings that qualify the above, and should not be dropped when summarising:**

- The free run, assimilating nothing, already reproduces the truth mode-1 amplitude profile to
  within 5%. There was no amplitude deficit for assimilation to fix; what the taps demonstrably
  fix is timing, not amplitude.
- The original Stage E claim ("only ~6 of 16 directions visible in wall pressure") is refuted:
  observing the entire 18,844-DOF state of that same ensemble gives effective dimension 1.013
  versus 1.010 from 32 taps. It measured the ensemble, not the sensors.

**Known defect, not fixed here:** the binary-masking immersed boundary zeroes velocity inside the
cylinder before the pressure projection but not after, so no-slip is not enforced — fluid speed
inside the "solid" reaches 47% of freestream. This causes the ~20% surface-pressure over-prediction,
Cd = 1.55–1.61 (literature 1.32–1.36), and the 13% frequency bias. Grid refinement does not converge
it away (observed order p = 0.40). Fixing it properly (sharp-interface IBM or body-fitted grid) is
the highest-value remaining forward-model improvement.

**Duplicate reports:** `docs/ModalPINN_EnKF_Audit_Part2{,_Final}.{md,pdf}` are sub-agent drafts and
are near-duplicates of each other. The authoritative version is the Part 2 PDF delivered in the
session (independently verified numbers); delete the drafts if you prefer a single copy.

## Modified source files (1)

- enkf_pressure_only/estimator/ns_solver.py

## New files (73)

- enkf_pressure_only/CHANGES_enkf_repairs.md
- enkf_pressure_only/_view_001_frequency_parameterization.png
- enkf_pressure_only/docs/ModalPINN_EnKF_Audit_Part2.md
- enkf_pressure_only/docs/ModalPINN_EnKF_Audit_Part2.pdf
- enkf_pressure_only/docs/ModalPINN_EnKF_Audit_Part2_Final.md
- enkf_pressure_only/docs/ModalPINN_EnKF_Audit_Part2_Final.pdf
- enkf_pressure_only/estimator/enkf2.py
- enkf_pressure_only/estimator/ensemble_init.py
- enkf_pressure_only/evaluation/METRICS_V2_NOTES.md
- enkf_pressure_only/evaluation/_truth_grid_cache.npz
- enkf_pressure_only/evaluation/metrics_v2.py
- enkf_pressure_only/evaluation/plot_metrics_v2_baseline.py
- enkf_pressure_only/evaluation/run_metrics_v2_baseline.py
- enkf_pressure_only/evaluation/stage_f2_evaluate_repaired.py
- enkf_pressure_only/evaluation/stage_f2_metrics_v2.log
- enkf_pressure_only/evaluation/stage_f2_metrics_v2.py
- enkf_pressure_only/evaluation/stage_f2_sliding_stability.py
- enkf_pressure_only/evaluation/validate_metrics_v2.py
- enkf_pressure_only/experiments/_blockage_probe.py
- enkf_pressure_only/experiments/_gridrefine_probe.py
- enkf_pressure_only/experiments/frequency_baseline_lift.npz
- enkf_pressure_only/experiments/frequency_followups_raw.json
- enkf_pressure_only/experiments/frequency_gamma_switch.npz
- enkf_pressure_only/experiments/frequency_limits_raw.json
- enkf_pressure_only/experiments/frequency_parameterization.json
- enkf_pressure_only/experiments/frequency_parameterization.npz
- enkf_pressure_only/experiments/frequency_parameterization.py
- enkf_pressure_only/experiments/frequency_parameterization_raw.json
- enkf_pressure_only/experiments/metric_validation.npz
- enkf_pressure_only/experiments/metric_validation_ownfreq.npz
- enkf_pressure_only/experiments/metric_validation_v2.npz
- enkf_pressure_only/experiments/metric_validation_v3.npz
- enkf_pressure_only/experiments/metric_validation_v4.npz
- enkf_pressure_only/experiments/metrics_v2_baseline.json
- enkf_pressure_only/experiments/metrics_v2_baseline.npz
- enkf_pressure_only/experiments/metrics_v2_baseline_v2.json
- enkf_pressure_only/experiments/metrics_v2_baseline_v2.npz
- enkf_pressure_only/experiments/run_frequency_followups.py
- enkf_pressure_only/experiments/run_frequency_limits.py
- enkf_pressure_only/experiments/run_frequency_parameterization.py
- enkf_pressure_only/experiments/sensor_model_blockage.npz
- enkf_pressure_only/experiments/sensor_model_comparison.json
- enkf_pressure_only/experiments/sensor_model_comparison.npz
- enkf_pressure_only/experiments/sensor_model_gridrefine.npz
- enkf_pressure_only/experiments/stage_d2_enkf_nominal.log
- enkf_pressure_only/experiments/stage_d2_enkf_nominal.npz
- enkf_pressure_only/experiments/stage_d2_enkf_nominal_repro.log
- enkf_pressure_only/experiments/stage_d2_enkf_nominal_repro.npz
- enkf_pressure_only/experiments/stage_d2_enkf_repaired.py
- enkf_pressure_only/experiments/stage_d2_enkf_scrambled_sensors.log
- enkf_pressure_only/experiments/stage_d2_enkf_scrambled_sensors.npz
- enkf_pressure_only/experiments/stage_d2_enkf_shuffled.log
- enkf_pressure_only/experiments/stage_d2_enkf_shuffled.npz
- enkf_pressure_only/experiments/stage_d2_summary.json
- enkf_pressure_only/experiments/stage_e2_observability.log
- enkf_pressure_only/experiments/stage_e2_observability.npz
- enkf_pressure_only/experiments/stage_e2_observability.py
- enkf_pressure_only/experiments/stage_f2_evaluation.npz
- enkf_pressure_only/experiments/stage_f2_metrics_v2.json
- enkf_pressure_only/experiments/stage_f2_metrics_v2.npz
- enkf_pressure_only/experiments/stage_f2_sliding_stability.json
- enkf_pressure_only/experiments/stage_f2_sliding_stability.npz
- enkf_pressure_only/figures/frequency_parameterization.png
- enkf_pressure_only/figures/metric_damping_test.png
- enkf_pressure_only/figures/metric_validation.png
- enkf_pressure_only/figures/metrics_v2_baseline.png
- enkf_pressure_only/figures/sensor_model_fix.png
- enkf_pressure_only/figures/stage_d2_ablation_controls.png
- enkf_pressure_only/figures/stage_d2_diagnostics.png
- enkf_pressure_only/figures/stage_d2_gain_and_innovation.png
- enkf_pressure_only/figures/stage_e2_observability_vs_null.png
- enkf_pressure_only/figures/stage_f2_mode1_vs_x.png
- enkf_pressure_only/figures/stage_f2_window_stability.png
