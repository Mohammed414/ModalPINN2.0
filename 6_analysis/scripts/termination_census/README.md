# Verification code — matched-effort re-run of arm 01

Scripts used to check the completed Colab run, diagnose why it (and the
original arm 01) stopped early, rule out tap count as the cause, and produce
the report figures. Run in order from a directory containing (or able to
create) `zx/baseline_physics_only_K3_matched/` — the extracted re-run
archive.

| script | purpose |
|---|---|
| `01_locate_and_extract_run_zip.sh` | Find and unzip the downloaded Colab output. |
| `02_compare_run_metrics.py` | Original vs. re-run: L-BFGS evaluations, final loss, checkpoint history. |
| `03_compare_v1_wake_metrics.py` | Confirms the wake (v1) result is unchanged — the pathology affects training effort, not the physical conclusion. |
| `04_parse_cycle_log_and_lbfgs_mechanism.py` | Per-cycle progress in the re-run; shows the last two accepted iterations before a dying L-BFGS call are numerically identical, and every call ends on the same scipy line-search warning. |
| `05_two_run_divergence.py` | Original arm 01 vs. re-run's first call: bit-identical for ~99 iterations, then diverge via float32 GPU non-determinism; gradient blow-up at each exit. |
| `06_all_arms_death_census.py` | The decisive test across all 17 project arms: tap count does not predict run length (two 32-tap arms differing only in collocation sampling ran 7-8x longer); Spearman correlation between stopping iteration and exit-gradient blow-up (rho = -0.73, p = 8e-4). |
| `07_check_warmstart_prior_usage.py` | Verification that `--RestoreModel` warm-starting had never actually been used in any prior arm (an earlier claim to the contrary, caught by review, was corrected). |
| `08_figure_arm1_rerun_breakdown.py` | Renders `F_arm1_rerun_breakdown.png`. Requires `figure_common.py` from `6_analysis/` on the path. |
| `09_figure_termination_anatomy.py` | Renders `F_termination_anatomy.png`. Same dependency. |

Both figure scripts import `figure_common` (house plotting style: `new_figure`,
`save_figure`, `check_text_overlaps`, `COLORS`) from the project's
`6_analysis/` directory — not included here, as it is
shared project infrastructure rather than part of this specific
investigation.
