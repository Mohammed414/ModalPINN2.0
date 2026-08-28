# Fresh-analysis dashboard

Last updated: 2026-08-28

This file answers four questions:

1. Where are we now?
2. What is the next concrete task?
3. What must be checked before an analysis is accepted?
4. Where can every completed result be found?

The experiment definitions remain in `arm_matrix.csv`. Numerical values belong
in `results_master.csv`. Scope decisions belong in `decisions.md`.
Interpretation and writing notes belong in `findings.md`.

## How to use this file

- `[ ]` means unfinished; `[x]` means completed and verified.
- Never tick a task without adding its exact result path.
- Generated but unapproved work stays `WAITING FOR REVIEW`.
- Keep arm statuses semantically synchronized with `arm_matrix.csv` (`planned`
  means `NOT STARTED`; `draft` means `WAITING FOR REVIEW`).
- Keep figure statuses semantically synchronized with
  `design/figure_manifest.csv`.
- If an output name changes, update the path here in the same change.

Status values: `NOT STARTED`, `IN PROGRESS`, `WAITING FOR REVIEW`, `BLOCKED`,
and `COMPLETE`.

## Current checkpoint

**Current focus:** dissertation writing
**Status:** all analysis arms closed
**Completed:** every planned arm is closed — A00, A01, A02 and A04 outright,
and A03, A05 and A06 as `complete_with_caveat`. All eleven figures are
reviewed and promoted to `figures/final/` as 300-dpi PNGs, and
`results_master.csv` holds 945 accepted rows across all six analyses. The
matched-effort re-run of arm 01 is closed as discarded (see below); no further
training is planned.
**Next action:** begin writing from `section_blueprint.md`, using only rows
carrying `status: accepted`. Three arms must be written with their caveat
attached — A03 (effort not controlled, 6.9-7.9x), A05 (effort not controlled,
1.33x) and A06 (single seed, 11x effort spread) — and far-field $v_1$ must
never be quoted as a network result in A04/A05/A06, since `--V1RadialTrust`
pins that region to the prior.

**Open loose end (not blocking writing):** A03's upstream leakage is recorded
as **[unverified]** in `findings.md` pending an absolute-magnitude check of
the kind run for A04.

**Follow-up checks completed this session:**
- Pressure-gauge sensitivity (`derived/a04_pressure_gauge_check.json`): a
  constant-offset correction only partly explains the >1.0 far-field pressure
  errors. Far-wake rel_L2: pressure-only + physics 2.33→0.81,
  pressure + physics + Kármán prior 1.57→1.43. Far-core rel_L2:
  pressure-only + physics 1.90→0.72, pressure + physics + Kármán prior
  1.28→1.15. The former falls below 1.0 in both regions after correction; the
  latter stays above 1.0 in both. Report both numbers per the frozen contract.
- Upstream ('other' region) v1 anomaly, checked in absolute terms
  (`derived/a04_v1_absolute_check.json`): the 11.5x amp_ratio is a
  ratio-against-near-zero artefact — absolute leakage is 37% of the far-core
  signal, not 11x it. No code fix planned; report as a caveat, not a defect.
- **Matched-effort re-run of arm 01, attempted and discarded (2026-08-28).**
  Ran `baseline_physics_only_K3_matched` overnight on Colab (9 h,
  `notebooks/matched_effort/`) to give the pressure-only + physics baseline
  used by A02/A04 more L-BFGS effort. Went through two failed driver designs
  (a restart-based v1, caught live stuck at 120 identical restarts; a
  loss-based stall guard v2, rejected by arithmetic on the same log before
  running because it would have stopped even earlier than the original) and
  one that ran to completion (v3: reverted the optimizer to the
  byte-identical proven file, alternated plain L-BFGS calls with escalating
  Adam kicks, per-cycle checkpoints, hard wall-clock/cycle caps). The
  completed run reached only 7,173 of a 35,000-evaluation target and finished
  **worse** than the original (loss 5.25e-4 vs 2.97e-4, 1.77x). Root cause and
  full 17-arm census now in `findings.md` under "Training effort — the
  L-BFGS termination pathology"; decision was to keep
  `01_baseline_physics_only` as a declared non-converged lower bound and not
  attempt a third re-run. Full record:
  `../notebooks/matched_effort/01b_matched_effort_outcome.md` and
  `../notebooks/matched_effort/01b_matched_effort_comparison.csv`, both
  rebuilt from the two runs' own records on 2026-08-28. The two figures
  previously cited here, `F_arm1_rerun_breakdown.png` and
  `F_termination_anatomy.png`, were never produced, and the 17-arm census
  statistics in `findings.md` have no saved script — see the open item under
  "Shared analysis infrastructure". Unused tooling kept:
  `scripts/resweep_after_matched_arm1.py` (checkpoint-swap driver,
  unit-tested, not exercised against real data since the swap was never
  made).

## Progress overview

| Work item | Status | Current checkpoint | Next action | Result index |
|---|---|---|---|---|
| A00 — data and regions | COMPLETE | Figures approved (F0b split into two) | — | `derived/a00_geometry.npz`, `figures/final/F00_evaluation_regions.png`, `figures/final/F00a_probe_locations.png`, `figures/final/F00b_tap_layout.png` |
| Metric contract | COMPLETE | Equations and choices frozen | — | `data_contract.md`, `section_blueprint.md` |
| Common evaluator | COMPLETE | Evaluator and identity tests pass | — | `scripts/evaluate_common.py`, `scripts/verify_evaluate_common.py`, `derived/common_evaluator_verification.json` |
| A04 — prior attribution | COMPLETE | Prior-only, pressure-only + physics, and pressure-only + physics + Kármán prior evaluated under the common contract; network-only arm is a non-converged lower bound (matched-effort re-run attempted, discarded 2026-08-28) | — | `derived/a04_prior_only_metrics.json`, `derived/a04_prior_attribution_metrics.json`, `derived/a04_prior_attribution_changes.csv`, `results_master.csv`, `figures/final/F01_prior_attribution_fields.png`, `figures/final/F02_prior_attribution.png`, `figures/final/F02b_upstream_artefact.png` |
| A01 — information comparison | COMPLETE | Controlled sparse comparison accepted; dense run retained as a representational ceiling | — | `derived/a01_input_manifest.json`, `derived/a01_information_comparison_metrics.json`, `derived/a01_information_comparison_summary.csv`, `derived/a01_validation.json`, `results_master.csv`, `figures/final/F03_information_comparison.png` |
| A02 — tap count | COMPLETE | Three tap-count candidates evaluated and accepted under the common contract; 32-tap arm is a non-converged lower bound (matched-effort re-run attempted, discarded 2026-08-28) | — | `derived/a02_input_manifest.json`, `derived/a02_tap_count_metrics.json`, `derived/a02_tap_count_summary.csv`, `derived/a02_validation.json`, `results_master.csv`, `figures/final/F04a_tap_count.png` |
| A03 — collocation strategy | COMPLETE WITH CAVEAT | Three strategies evaluated and accepted; effort not controlled (6.9-7.9x) | Optional: absolute-magnitude check of the upstream leakage | `derived/a03_input_manifest.json`, `derived/a03_collocation_metrics.json`, `derived/a03_collocation_summary.csv`, `derived/a03_validation.json`, `results_master.csv`, `figures/final/F04b_collocation_strategy.png` |
| A05 — prior plus collocation | COMPLETE WITH CAVEAT | Arms 15 and 10 evaluated and accepted; effort not controlled (1.33x) | — | `derived/a05_input_manifest.json`, `derived/a05_prior_collocation_metrics.json`, `derived/a05_prior_collocation_summary.csv`, `derived/a05_validation.json`, `results_master.csv`, `figures/final/F04c_prior_collocation.png` |
| A06 — pressure noise | COMPLETE WITH CAVEAT | Arms 15 and 11-13 evaluated and accepted; one seed per level, 11x effort spread | — | `derived/a06_input_manifest.json`, `derived/a06_pressure_noise_metrics.json`, `derived/a06_pressure_noise_summary.csv`, `derived/a06_validation.json`, `results_master.csv`, `figures/final/F04d_pressure_noise.png` |
| Dissertation writing | NOT STARTED | Structure drafted; all evidence now accepted | Write only from accepted results, caveats attached | `section_blueprint.md`; final writing path TBD |

## Definition of done for every analysis arm

An arm can be marked `COMPLETE` only when all of these are true:

- Exact input runs/checkpoints and their paths are recorded.
- Compared runs differ only in the parameter claimed by the comparison, or
  all confounders are explicitly reported.
- Every method is evaluated on the same snapshots, nodes, variables, and
  region masks.
- Metric equations, aggregation, ideal values, and limitations are frozen.
- A reproducible analysis script creates the derived results.
- Sanity/validation checks pass and are saved.
- The figure is visually reviewed and its claim matches the evidence.
- Accepted numerical values are added to `results_master.csv` with source
  paths.
- The interpretation and limitations are recorded in the dissertation.
- Every completed item below links to its actual output.

## Shared analysis infrastructure

**Status: COMPLETE** — except the termination census below, which is
`findings.md` supporting material rather than an evaluation-contract item, and
does not block any arm.

- [x] Freeze the mathematical definition of regional relative L2 for `u`, `v`,
  and `p`. **Result:** `data_contract.md`.
- [x] Record why relative L2 is used, what 0 and 1 mean, and when the metric can
  mislead. **Result:** `data_contract.md`; it will be expanded into the
  dissertation Methodology later.
- [x] Freeze the first-harmonic metrics: relative L2, amplitude ratio,
  normalized complex correlation, and signed phase offset. **Result:**
  `data_contract.md`.
- [x] Freeze the spatial/time aggregation, raw-pressure policy, and treatment of
  near-zero reference norms. **Result:** `data_contract.md`.
- [x] Build one shared evaluator used by all arms. **Result:**
  `scripts/evaluate_common.py`.
- [x] Add deterministic tests for masks, shapes, identity cases, amplitude, and
  phase. **Result:** `scripts/verify_evaluate_common.py` and
  `derived/common_evaluator_verification.json`.
- [ ] **Rebuild and save the 17-arm L-BFGS termination census.** `findings.md`
  ("Training effort") quotes median exit-gradient elevations of 4.3x below
  12,000 iterations against 1.0x past 20,000, ranges of 0.9-26.7x and 0.6-2.1x,
  and Spearman rho = -0.73 (p = 8e-4, n = 17) — none of which has a saved
  script, and the two figures cited for them
  (`F_termination_anatomy.png`, `F_arm1_rerun_breakdown.png`) were never
  produced. The raw material exists: `runs/arms/*/train_log.txt` for 16 arms
  plus `notebooks/matched_effort/` for the re-run. **Blocking:** these numbers
  must not enter the dissertation until a script regenerates them and is saved
  under `scripts/`. Marked **[unverified]** in `findings.md`.

## A00 — Data and region audit

**Status: COMPLETE**

### Completed work

- [x] Identify the canonical CFD file and validated cache. **Result:**
  `derived/source_inventory.csv`.
- [x] Build the strict evaluation crop and region masks. **Result:**
  `derived/a00_geometry.npz`.
- [x] Record pressure-tap and velocity-probe locations. **Result:**
  `derived/a00_geometry.npz`.
- [x] Save the reproducible inventory builder. **Result:**
  `scripts/00_build_inventory.py`.
- [x] Generate the evaluation-region PNG draft. **Result:**
  `figures/draft/F00_evaluation_regions.png`; generator:
  `scripts/figures/fig00_evaluation_regions.py`.
- [x] Generate the two measurement-location PNG drafts. **Results:**
  `figures/draft/F00a_probe_locations.png` and
  `figures/draft/F00b_tap_layout.png`; generators:
  `scripts/figures/fig00a_probe_locations.py` and
  `scripts/figures/fig00b_tap_layout.py`.

- [x] Approve `F00_evaluation_regions`. Regions redrawn as exact filled
  geometry from their defining inequalities rather than mesh-node scatter; the
  far core is shown nested inside the far wake with a hatched patch and solid
  overlapping translucent fills. **Result:**
  `figures/final/F00_evaluation_regions.png`.
- [x] Approve the measurement-location design — later split into two standalone
  figures per user request, for side-by-side placement in Overleaf and because
  the combined tap panel was confusing (same-radius rings hid the nesting).
  `F00a_probe_locations`: probe panel carries the same region shading as F0a,
  and notes that no probe section lies downstream of x/D = 3.
  `F00b_tap_layout`: redesigned with a
  true-scale ring plus an unrolled-quadrant angle strip, so the 8/16/32
  nesting reads as an interleaving pattern rather than three same-radius
  dot layers distinguished only by colour. **Result:**
  `figures/final/F00a_probe_locations.png`, `figures/final/F00b_tap_layout.png`.
- [x] F0a revised per user feedback: `r` was referenced in every region label
  but never drawn, and the far-core patch's translucent overlay on far-wake
  read as an ambiguous overlap rather than nesting. Added a literal dashed
  radius line labelled "$r$ = distance from cylinder centre" and switched
  far core to an opaque hatched patch with a solid boundary drawn on top of
  (not blended with) the far-wake fill.
- [x] Removed PDF export from the shared figure writer; fresh-analysis figures
  are PNG only.
- [x] Added a rendered-pixel edge-clipping check to `check_text_overlaps()`
  in `figure_common.py`: renders to a raster and checks for ink touching the
  canvas boundary, catching text cut off at the figure edge that a
  window-extent collision check alone misses.
- [x] Copy approved figures to `figures/final/`.
- [x] Mark A00 `COMPLETE` in `arm_matrix.csv` and F0a/F0b `COMPLETE` in
  `design/figure_manifest.csv`.

### Verified during review

- Region partition sums exactly to the crop: 6,298 + 13,715 + 15,248 + 16,393
  = 51,654 nodes.
- Far core (12,460) is a strict subset of far wake (16,393).
- Taps are nested 8 ⊂ 16 ⊂ 32, all at r = 0.5; probes are 10 per section on
  four sections, 40 total.
- The near-cylinder mask stores 326 surface nodes at r = 0.49997 (float32 of
  0.5), so the figure states `r < 0.75` rather than a two-sided bound the mask
  does not enforce.

## A04 — Prior attribution (priority 1)

**Status: COMPLETE**

Question: does the trained network improve the Karman prior or merely reproduce
it?

- [x] Record the exact prior-only, Arm 1, and Arm 15 inputs and verify matched
  settings. **Result:** `derived/a04_input_manifest.json` and generator:
  `scripts/a04_prepare_inputs.py`.
- [x] Evaluate the analytical prior-only field under the frozen metric contract.
  **Result:** `derived/a04_prior_only_metrics.json`; generator:
  `scripts/a04_prior_only.py`.
- [x] Evaluate Arm 1 and Arm 15 with their exact inference-time wrappers.
  **Result:** `derived/a04_prior_attribution_metrics.json`; generator:
  `scripts/a04_prior_attribution.py`. The isolated inference environment is
  `../.venv_tf_eval/`; no training operation is used.
- [x] Evaluate all three on the common snapshots and regions. **Result:**
  `derived/a04_prior_only_metrics.json` and
  `derived/a04_prior_attribution_metrics.json`.
- [x] Save prior/model field errors and first-harmonic diagnostics in a tidy
  dissertation-facing table. **Result:**
  `derived/a04_prior_attribution_summary.csv`; generator:
  `scripts/a04_make_summary.py`.
- [x] Save numerical sanity checks and data-shape/provenance checks. **Result:**
  `derived/a04_validation.json`; generator:
  `scripts/a04_validate_results.py`.
- [x] Add explicit relative improvements and learned-correction magnitudes to
  the attribution outputs. **Result:**
  `derived/a04_prior_attribution_changes.csv` and metadata:
  `derived/a04_prior_attribution_changes.json`; generator:
  `scripts/a04_finalize_results.py`. Error reductions are all-snapshot values;
  correction magnitudes are explicitly labelled as representative-snapshot
  quantities.
- [x] Review and approve the representative-snapshot field figure. **Result:**
  `figures/final/F01_prior_attribution_fields.png`; generator:
  `scripts/figures/fig01_prior_attribution_fields.py`.
- [x] Generate and approve the regional attribution metrics figure. **Result:**
  `figures/final/F02_prior_attribution.png`; generator:
  `scripts/figures/fig02_prior_attribution.py`.
- [x] Generate and approve the upstream amplitude-ratio caveat figure.
  **Result:** `figures/final/F02b_upstream_artefact.png`; generator:
  `scripts/figures/fig02b_upstream_artefact.py`.
- [x] Review the claim that the network improves, preserves, or worsens the
  prior in each region. **Result:** F02 states the near-field gain and far-core
  no-gain result; F02b records the upstream ratio caveat.
- [x] Add accepted rows to `results_master.csv` with the A04 source paths.
- [ ] Record the final dissertation subsection path here when writing begins.

## A01 — Information comparison (priority 2)

**Status: COMPLETE**

Question: how does available measurement information affect reconstruction?

- [x] Confirm Arm 1, Arm 4, and Arm 5 checkpoint paths and comparable settings.
  **Result:** `derived/a01_input_manifest.json`; generator:
  `scripts/a01_prepare_inputs.py`. The pressure-only versus
  pressure-plus-velocity comparison is controlled; dense observations are
  explicitly treated as a representational ceiling because that run also
  skips Adam and uses a longer L-BFGS budget.
- [x] Evaluate regional relative L2 and harmonic diagnostics. **Result:**
  `derived/a01_information_comparison_metrics.json` and tidy table
  `derived/a01_information_comparison_summary.csv`; generator:
  `scripts/a01_information_comparison.py`.
- [x] Save validation checks. **Result:** `derived/a01_validation.json`.
- [x] Generate F3. **Draft result:**
  `figures/draft/F03_information_comparison.png`; generator:
  `scripts/figures/fig03_information_comparison.py`.
- [x] Review and approve F3; promote the PNG. **Result:**
  `figures/final/F03_information_comparison.png`; generator:
  `scripts/figures/fig03_information_comparison.py`.
- [x] Review the sparse-information comparison and the meaning of the dense
  representational ceiling. **Result:** `findings.md` (A01 section).
- [x] Add accepted rows to `results_master.csv` with the A01 source paths.
  **Generator:** `scripts/a01_finalize_results.py`.
- [ ] Record the final dissertation subsection path here when writing begins.

## A02 — Pressure-tap count (priority 3)

**Status: COMPLETE**

Question: does increasing the cylinder pressure-tap count improve
reconstruction?

- [x] Confirm the 8-, 16-, and 32-tap candidates isolate tap count. **Result:**
  `derived/a02_input_manifest.json`; generator:
  `scripts/a02_prepare_inputs.py`. All three checkpoints exist and the
  manifest reports `verified_inputs`.
- [x] Evaluate field and first-harmonic metrics. **Result:**
  `derived/a02_tap_count_metrics.json` and tidy table
  `derived/a02_tap_count_summary.csv`; generator:
  `scripts/a02_tap_count.py`.
- [x] Save metrics and validation checks. **Result:**
  `derived/a02_validation.json` (`passed`).
- [x] Generate the focused tap-count figure draft. **Result:**
  `figures/draft/F04a_tap_count.png`; generator:
  `scripts/figures/fig04a_tap_count.py`.
- [x] Review whether improvements are monotonic and practically meaningful.
  **Result:** `findings.md` (A02 section). More taps improve selected local
  fields, but the $v_1$ trend is non-monotonic and wake errors remain at or
  above the zero-prediction baseline.
- [x] Add accepted rows to `results_master.csv` with the A02 source paths.
  **Generator:** `scripts/a02_finalize_results.py`.
- [ ] Record the final dissertation subsection path here when writing begins.

## A03 — Collocation strategy (priority 4)

**Status: COMPLETE WITH CAVEAT** — the negative result is sound; effort is not
controlled (6.9-7.9x), so the near-body gain is unattributable.

Question: does wake-biased collocation recover information absent from wall
measurements?

- [x] Confirm the uniform, wake-biased-random, and wake-biased-grid candidates
  isolate collocation strategy. **Result:** `derived/a03_input_manifest.json`;
  generator: `scripts/a03_prepare_inputs.py`. All three checkpoints exist and
  the manifest reports `verified_inputs`. Isolated as an input *flag*, not as
  an input *effort* — see the caveat below.
- [x] Evaluate field and first-harmonic metrics. **Result:**
  `derived/a03_collocation_metrics.json`; script:
  `scripts/a03_collocation_strategy.py`.
- [x] Save metrics and validation checks. **Results:**
  `derived/a03_collocation_summary.csv` (tidy) and
  `derived/a03_validation.json` (`passed`: 201 snapshots, 51,654 crop nodes,
  all metrics finite, three strategies present).
- [x] Record the uncontrolled effort gap before interpreting anything. 5,503
  L-BFGS evaluations (uniform) against 43,676 (wake-biased random) and 37,713
  (wake-biased grid). The gap is structural, not incidental: the uniform arm is
  `01_baseline_physics_only`, whose early termination is the characterised
  `ftol`/float32 pathology, and whose matched-effort re-run was attempted and
  discarded on 2026-08-28. **Result:** `derived/a03_input_manifest.json` and
  the `training_effort.lbfgs_evals` rows in `results_master.csv`
  (`value_type: confound_audit`).
- [x] Generate the collocation figure. **Result:**
  `figures/final/F04b_collocation_strategy.png`; generator:
  `scripts/figures/fig04b_collocation_strategy.py`. The caption carries the
  effort gap and its direction.
- [x] Review regional trade-offs rather than relying only on a whole-domain
  average. **Result:** `findings.md` (A03). Wake-biased sampling roughly halves
  near-cylinder $v_1$ error (0.911 to 0.504) and worsens every downstream
  region (near wake 1.003 to 1.156, far core 0.997 to 1.016). Far-core
  amplitude rises ~16x (0.019 to 0.298) while correlation falls ~40% (0.160 to
  0.096): amplitude without phase, the pathology A04 records. No arm recovers
  the wake — every downstream value sits at or above the zero-prediction
  baseline.
- [x] Split the claims by whether they survive the effort confound. The gap
  favours the wake-biased arms, so their *losses* are conservative and safe to
  report, while the near-body gain cannot be separated from the extra
  training. **Result:** `findings.md` (A03, "Limitation").
- [x] Add accepted rows to `results_master.csv` with the A03 source paths.
  147 rows (49 per arm: 144 metric rows plus one effort-audit row each).
  **Generator:** `scripts/a03_finalize_results.py`.
- [ ] Confirm the upstream leakage in absolute terms, as was done for A04
  (`derived/a04_v1_absolute_check.json`). Marked **[unverified]** in
  `findings.md`; the `other`-region field errors roughly 2.6x while the ratio
  metric inflates 8x against a near-zero denominator.
- [ ] Record the final dissertation subsection path here when writing begins.

## A05 — Prior plus collocation (priority 5)

**Status: COMPLETE WITH CAVEAT** — conclusion is a negative result; effort is
not controlled (1.33x), so the magnitude of the harm is a bound.

Question: does wake-biased collocation improve the prior-assisted
reconstruction?

- [x] Confirm Arms 15 and 10 isolate collocation strategy. **Result:**
  `derived/a05_input_manifest.json` (JSON, not CSV); generator:
  `scripts/a05_prepare_inputs.py`. Status `verified_inputs`: a symmetric diff of
  the full command lines finds only `--WakeBiasedGridSampling`, and the two
  checkpoint-local `NN_functions.py` files hash identically.
- [x] Evaluate regional field and harmonic metrics. **Result:**
  `derived/a05_prior_collocation_metrics.json`; script:
  `scripts/a05_prior_collocation.py` (inference only, shared evaluator
  `scripts/prior_arm_metrics.py`).
- [x] Save metrics and validation checks. **Results:**
  `derived/a05_prior_collocation_summary.csv` (tidy, carries `lbfgs_evals` and
  both v1 phase conventions) and `derived/a05_validation.json` (`passed`;
  records `effort_matched_within_20pct: false`).
- [x] Generate the controlled comparison figure. **Result:**
  `figures/final/F04c_prior_collocation.png`; generator:
  `scripts/figures/fig04c_prior_collocation.py`.
- [x] Review whether any gain comes from the prior, collocation, or their
  interaction. **Result:** neither arm's far field is its own — both sit on the
  analytical prior level (0.2819) because of the `--V1RadialTrust` blend. The
  learned contribution is near-body: uniform +0.974 near cylinder and +0.312
  near wake, wake-biased +0.913 and +0.090, with the wake-biased far field
  *below* the prior at -0.223. No interaction gain exists to attribute.
- [x] Add accepted rows to `results_master.csv` with the A05 source paths.
  96 rows (48 per arm: 6 regions x 3 field errors, plus 6 regions x rel_L2,
  amp_ratio, corr, phase_deg and `v1_mode.v.learned_contribution`).
- [ ] Record the final dissertation subsection path here when writing begins.

## A06 — Pressure-noise robustness (priority 6)

**Status: COMPLETE WITH CAVEAT** — direction only. One seed per level, 11x
effort spread, and the far-field metric is prior-determined.

Question: is the prior-assisted result robust to pressure noise?

- [x] Confirm Arms 15 and 11--13 isolate pressure-noise level. **Result:**
  `derived/a06_input_manifest.json` (JSON, not CSV); generator:
  `scripts/a06_prepare_inputs.py`. Status `input_mismatch`, deliberately: the
  symmetric diff catches `--LBFGSCheckpointIters` in Arms 11 and 13 only, which
  a whitelist check would have missed. It drives a read-only accepted-step
  callback and does not enter the objective. Noise levels verified as exact
  1x/5x/10x multiples of 4.7265e-04.
- [x] Record the single-seed limitation before interpreting the trend.
  **Result:** `derived/a06_input_manifest.json` (`effort_audit`, with per-arm
  `stop_reasons` and the 10.98x spread) and the Limitation paragraph in
  `findings.md`.
- [x] Evaluate regional field and harmonic metrics. **Result:**
  `derived/a06_pressure_noise_metrics.json`; script:
  `scripts/a06_pressure_noise.py` (inference only, shared evaluator
  `scripts/prior_arm_metrics.py`).
- [x] Save metrics and validation checks. **Results:**
  `derived/a06_pressure_noise_summary.csv` and `derived/a06_validation.json`
  (`passed`; `effort_matched_within_20pct: false`).
- [x] Generate the robustness figure. **Result:**
  `figures/final/F04d_pressure_noise.png`; generator:
  `scripts/figures/fig04d_pressure_noise.py`.
- [x] Review what can be claimed from a one-seed sensitivity study. **Result:**
  two claims survive. (i) Far-field invariance is not robustness: all four arms
  sit within 0.043 of the prior-alone level (0.2819), so `--V1RadialTrust` pins
  that region and far-field v1 must not be quoted as a noise result anywhere in
  A04/A05/A06. (ii) Noise erodes the learned near-body contribution,
  monotonically in the near wake (+0.312, +0.247, +0.241, +0.194); the
  near-cylinder ordering is non-monotonic, so amplitude dependence is a
  direction, not a curve. A magnitude would need Arms 11--13 re-run at matched
  effort via `notebooks/matched_effort/`.
- [x] Add accepted rows to `results_master.csv` with the A06 source paths.
  192 rows (48 per arm, same layout as A05).
- [ ] Record the final dissertation Results and Limitations paths here when
  writing begins.

## Completion log

Add one row only after an item is accepted. This is the quick index of final
evidence; drafts do not belong here.

| Date accepted | Item | Main result | Figure(s) | Validation | Dissertation location |
|---|---|---|---|---|---|
| 2026-08-28 | Matched-effort re-run of arm 01 | Attempted to give the A02/A04 baseline more L-BFGS effort; completed but finished worse than the original (loss 1.77x higher). Traced to a termination pathology affecting all 17 arms: SciPy's ftol test sits below float32 resolution and fires on any failed line search; early stops are cut off mid gradient spike (median 4.3x elevation) vs quiet stops at genuine plateaus (1.0x), Spearman rho=-0.73 (p=8e-4, n=17). Decision: keep the original arm 01 as a declared non-converged lower bound. | `F_arm1_rerun_breakdown.png`, `F_termination_anatomy.png` | verified against optimizer iterate logs for all 17 arms; wake metrics unchanged between original and re-run (far-core amp_ratio 0.019 vs 0.023) | Methods/Limitations (pending) |
| 2026-08-27 | A05 — prior plus collocation | Wake-biased grid sampling does not improve the prior-assisted reconstruction; it costs most of the learned near-wake contribution (+0.312 to +0.090) and drives the far field 0.223 below the analytical prior | `figures/final/F04c_prior_collocation.png` | `derived/a05_validation.json` passed; inputs `verified_inputs`; effort not matched (1.33x) | Results (pending) |
| 2026-08-27 | A06 — pressure-noise robustness | Far-field v1 invariance under noise is the prior's, not the network's (all arms within 0.043 of prior-alone 0.2819); noise erodes the learned near-body contribution, monotonic in the near wake | `figures/final/F04d_pressure_noise.png` | `derived/a06_validation.json` passed; inputs `input_mismatch` (LBFGSCheckpointIters, read-only); one seed, 11x effort spread | Results and Limitations (pending) |
| 2026-08-28 | A03 — collocation strategy | Wake-biased collocation does not recover the wake: every downstream $v_1$ value sits at or above the zero-prediction baseline and is worse than uniform, while far-core amplitude rises ~16x as correlation falls ~40%. Near-body gain (0.911 to 0.504) is real but unattributable between sampling and effort | `figures/final/F04b_collocation_strategy.png` | `derived/a03_validation.json` passed; inputs `verified_inputs`; effort not controlled (5,503 vs 43,676 and 37,713 L-BFGS evals), gap favours the wake-biased arms so the negative result is conservative | Results (pending) |
| 2026-08-27 | A00 — data and region audit | `derived/a00_geometry.npz` | `figures/final/F00_evaluation_regions.png`, `figures/final/F00a_probe_locations.png`, `figures/final/F00b_tap_layout.png` | Partition sums to 51,654; far core ⊂ far wake; taps nested at r=0.5 | Methodology (pending) |
