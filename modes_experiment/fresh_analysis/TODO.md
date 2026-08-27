# Fresh-analysis dashboard

Last updated: 2026-08-27

This file answers four questions:

1. Where are we now?
2. What is the next concrete task?
3. What must be checked before an analysis is accepted?
4. Where can every completed result be found?

The experiment definitions remain in `arm_matrix.csv`. Numerical values belong
in `results_master.csv`. Scope decisions belong in `decisions.md`.

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

**Current focus:** metric contract  
**Status:** NOT STARTED  
**Completed:** A00 is closed. Data provenance, crop/region masks, sensor
mapping, and all three F0 figures are finished, reviewed, and promoted to
`figures/final/` as 300-dpi PNGs.  
**Next action:** freeze the metric contract in `data_contract.md`, then begin
A04 because prior attribution is the central dissertation question.

## Progress overview

| Work item | Status | Current checkpoint | Next action | Result index |
|---|---|---|---|---|
| A00 — data and regions | COMPLETE | Figures approved (F0b split into two) | — | `derived/a00_geometry.npz`, `figures/final/F00_evaluation_regions.png`, `figures/final/F00a_probe_locations.png`, `figures/final/F00b_tap_layout.png` |
| Metric contract | NOT STARTED | Definitions outlined only — **next task** | Freeze equations, aggregation, ideal values, and limitations | `data_contract.md`, `section_blueprint.md` |
| Common evaluator | NOT STARTED | No fresh evaluator yet | Specify shared input/output contract | planned: `scripts/evaluate_common.py` |
| A04 — prior attribution | NOT STARTED | Planned | Start after metric contract | planned: `derived/a04_*`, F1 and F2 |
| A01 — information comparison | NOT STARTED | Planned | Start after common evaluator | planned: `derived/a01_*`, F3 |
| A02 — tap count | NOT STARTED | Planned | Start after common evaluator | planned: `derived/a02_*`, F4a |
| A03 — collocation strategy | NOT STARTED | Planned | Start after common evaluator | planned: `derived/a03_*`, F4b |
| A05 — prior plus collocation | NOT STARTED | Planned | Start after A04 | planned: `derived/a05_*`, F4c |
| A06 — pressure noise | NOT STARTED | Planned | Start after A04 | planned: `derived/a06_*`, F4d |
| Dissertation writing | NOT STARTED | Structure drafted | Write only from accepted results | `section_blueprint.md`; final writing path TBD |

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

**Status: NOT STARTED**

- [ ] Freeze the mathematical definition of regional relative L2 for `u`, `v`,
  and `p`. **Planned result:** `data_contract.md`.
- [ ] Explain why relative L2 is used, what 0 and 1 mean, and when the metric can
  mislead. **Planned result:** `section_blueprint.md` and dissertation
  Methodology.
- [ ] Freeze the first-harmonic metrics: relative L2, amplitude ratio,
  normalized complex correlation, and phase offset. **Planned result:**
  `data_contract.md`.
- [ ] Define the spatial/time aggregation and treatment of near-zero reference
  norms. **Planned result:** `data_contract.md`.
- [ ] Build one shared evaluator used by all arms. **Planned result:**
  `scripts/evaluate_common.py`.
- [ ] Add small deterministic tests for masks, shapes, and identity cases.
  **Planned result:** `scripts/verify_evaluate_common.py` and its saved report in
  `derived/common_evaluator_verification.json`.

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
  far core is shown nested inside the far wake with a dashed boundary and no
  overlapping translucent fills. **Result:**
  `figures/final/F00_evaluation_regions.png`.
- [x] Approve `F00_measurement_locations` — later split into two standalone
  figures per user request, for side-by-side placement in Overleaf and because
  the combined tap panel was confusing (same-radius rings hid the nesting).
  `F00a_probe_locations`: probe panel carries the same region shading as F0a,
  so the unobserved far core is visible. `F00b_tap_layout`: redesigned with a
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

**Status: NOT STARTED**

Question: does the trained network improve the Karman prior or merely reproduce
it?

- [ ] Record the exact prior-only, Arm 1, and Arm 15 inputs. **Planned result:**
  `derived/a04_input_manifest.csv`.
- [ ] Evaluate all three on the common snapshots and regions. **Planned script:**
  `scripts/a04_prior_attribution.py`.
- [ ] Save prior error, hybrid error, relative improvement, learned-correction
  magnitude, and harmonic diagnostics. **Planned result:**
  `derived/a04_prior_attribution_metrics.csv`.
- [ ] Save numerical sanity checks and data-shape/provenance checks. **Planned
  result:** `derived/a04_validation.json`.
- [ ] Generate the prior-only field figure. **Planned result:**
  `figures/draft/F01_prior_only.png`; planned generator:
  `scripts/figures/fig01_prior_only.py`.
- [ ] Generate the attribution figure. **Planned result:**
  `figures/draft/F02_prior_attribution.png`; planned generator:
  `scripts/figures/fig02_prior_attribution.py`.
- [ ] Review the claim that the network improves, preserves, or worsens the
  prior in each region.
- [ ] Add accepted rows to `results_master.csv` with the A04 source paths.
- [ ] Record the final dissertation subsection path here when writing begins.

## A01 — Information comparison (priority 2)

**Status: NOT STARTED**

Question: how does available measurement information affect reconstruction?

- [ ] Confirm Arm 1, Arm 4, and Arm 5 checkpoint paths and comparable settings.
  **Planned result:** `derived/a01_input_manifest.csv`.
- [ ] Evaluate regional relative L2 and harmonic diagnostics. **Planned script:**
  `scripts/a01_information_comparison.py`.
- [ ] Save metrics. **Planned result:**
  `derived/a01_information_comparison_metrics.csv`.
- [ ] Save validation checks. **Planned result:**
  `derived/a01_validation.json`.
- [ ] Generate F3. **Planned result:**
  `figures/draft/F03_information_comparison.png`; planned generator:
  `scripts/figures/fig03_information_comparison.py`.
- [ ] Review the sparse-information comparison and the meaning of the dense
  representational ceiling.
- [ ] Add accepted rows to `results_master.csv` with the A01 source paths.
- [ ] Record the final dissertation subsection path here when writing begins.

## A02 — Pressure-tap count (priority 3)

**Status: NOT STARTED**

Question: does increasing the cylinder pressure-tap count improve
reconstruction?

- [ ] Confirm Arms 8, 9, and 1 isolate 8, 16, and 32 taps. **Planned result:**
  `derived/a02_input_manifest.csv`.
- [ ] Evaluate field and first-harmonic metrics. **Planned script:**
  `scripts/a02_tap_count.py`.
- [ ] Save metrics and validation checks. **Planned results:**
  `derived/a02_tap_count_metrics.csv` and `derived/a02_validation.json`.
- [ ] Generate the tap-count figure. **Planned result:**
  `figures/draft/F04a_tap_count.png`; planned generator:
  `scripts/figures/fig04a_tap_count.py`.
- [ ] Review whether improvements are monotonic and practically meaningful.
- [ ] Add accepted rows to `results_master.csv` with the A02 source paths.
- [ ] Record the final dissertation subsection path here when writing begins.

## A03 — Collocation strategy (priority 4)

**Status: NOT STARTED**

Question: does wake-biased collocation recover information absent from wall
measurements?

- [ ] Confirm Arms 1, 6, and 7 isolate collocation strategy. **Planned result:**
  `derived/a03_input_manifest.csv`.
- [ ] Evaluate field and first-harmonic metrics. **Planned script:**
  `scripts/a03_collocation_strategy.py`.
- [ ] Save metrics and validation checks. **Planned results:**
  `derived/a03_collocation_metrics.csv` and `derived/a03_validation.json`.
- [ ] Generate the collocation figure. **Planned result:**
  `figures/draft/F04b_collocation_strategy.png`; planned generator:
  `scripts/figures/fig04b_collocation_strategy.py`.
- [ ] Review regional trade-offs rather than relying only on a whole-domain
  average.
- [ ] Add accepted rows to `results_master.csv` with the A03 source paths.
- [ ] Record the final dissertation subsection path here when writing begins.

## A05 — Prior plus collocation (priority 5)

**Status: NOT STARTED**

Question: does wake-biased collocation improve the prior-assisted
reconstruction?

- [ ] Confirm Arms 15 and 10 isolate collocation strategy. **Planned result:**
  `derived/a05_input_manifest.csv`.
- [ ] Evaluate regional field and harmonic metrics. **Planned script:**
  `scripts/a05_prior_collocation.py`.
- [ ] Save metrics and validation checks. **Planned results:**
  `derived/a05_prior_collocation_metrics.csv` and
  `derived/a05_validation.json`.
- [ ] Generate the controlled comparison figure. **Planned result:**
  `figures/draft/F04c_prior_collocation.png`; planned generator:
  `scripts/figures/fig04c_prior_collocation.py`.
- [ ] Review whether any gain comes from the prior, collocation, or their
  interaction.
- [ ] Add accepted rows to `results_master.csv` with the A05 source paths.
- [ ] Record the final dissertation subsection path here when writing begins.

## A06 — Pressure-noise robustness (priority 6)

**Status: NOT STARTED**

Question: is the prior-assisted result robust to pressure noise?

- [ ] Confirm Arms 15 and 11--13 isolate pressure-noise level. **Planned
  result:** `derived/a06_input_manifest.csv`.
- [ ] Record the single-seed limitation before interpreting the trend.
  **Planned result:** `derived/a06_input_manifest.csv`.
- [ ] Evaluate regional field and harmonic metrics. **Planned script:**
  `scripts/a06_pressure_noise.py`.
- [ ] Save metrics and validation checks. **Planned results:**
  `derived/a06_pressure_noise_metrics.csv` and `derived/a06_validation.json`.
- [ ] Generate the robustness figure. **Planned result:**
  `figures/draft/F04d_pressure_noise.png`; planned generator:
  `scripts/figures/fig04d_pressure_noise.py`.
- [ ] Review what can be claimed from a one-seed sensitivity study.
- [ ] Add accepted rows to `results_master.csv` with the A06 source paths.
- [ ] Record the final dissertation Results and Limitations paths here when
  writing begins.

## Completion log

Add one row only after an item is accepted. This is the quick index of final
evidence; drafts do not belong here.

| Date accepted | Item | Main result | Figure(s) | Validation | Dissertation location |
|---|---|---|---|---|---|
| 2026-08-27 | A00 — data and region audit | `derived/a00_geometry.npz` | `figures/final/F00_evaluation_regions.png`, `figures/final/F00a_probe_locations.png`, `figures/final/F00b_tap_layout.png` | Partition sums to 51,654; far core ⊂ far wake; taps nested at r=0.5 | Methodology (pending) |
