# 6_analysis — what the runs mean

This is the clean workspace where the dissertation analysis was rebuilt step by
step. It deliberately did not inherit results, figures, conclusions, or arm
definitions from the earlier work: every number here was recomputed from
`4_runs/` and checked before being accepted.

`results_master.csv` and `findings.md` are the numbers the dissertation quotes.
`evaluator/` holds `modalpinn_eval.py`, which rebuilds a trained arm's fields
exactly as it was trained, and the parsed DNS cache `dns_raw.npz`.

## Working rule

We agree the question, inputs, metric, and planned visual first. Only then do we
write the analysis script and run it. A result is not added to
`results_master.csv` until its source and interpretation have been reviewed.

## Files

- `section_blueprint.md` — the proposed Methodology, Results, and Discussion
  structure, written before any new computation.
- `TODO.md` — the execution checklist and checkpoint dashboard. Completed tasks
  always include the location of their result.
- `figure_blueprint.md` — visual wireframes for each planned figure.
- `analysis_matrix.csv` — a fresh plan of analyses/arms; status starts as `planned`.
- `results_master.csv` — an empty long-form results table. Each row will carry
  one metric, its region, source, and review status.
- `data_contract.md` — the data paths, variables, units, and preprocessing we
  agree to use.
- `decisions.md` — a dated record of decisions and changes in scope.
- `findings.md` — running notes on what the numbers appear to mean, per
  analysis. Raw material for the Results and Discussion sections, not finished
  prose. Speculative items are marked `[unverified]`.
- `design/` — the shared style guide, figure manifest, and one reviewed design
  note per figure.
- `figure_common.py` — the single source of shared typography, colours, sizes,
  axes, and export behaviour.
- `derived/` — machine-readable outputs produced by analysis scripts; never
  hand-edited.
- `derived/verified_results.csv` — the single filter-friendly table containing
  every accepted ModalPINN and Gappy POD result, with method, region, metric,
  units, phase convention, provenance, and caveat columns.
- `derived/verified_results_audit.json` — automated checks for the consolidated
  table. `status: passed` is required before numbers are used in writing.
- `derived/v1_phase_correction_audit.csv` — an explicit old-versus-corrected
  audit for first-harmonic metrics affected by the time-origin convention.
- `figures/final/` — generated figures only; the figure scripts write here directly.
- `scripts/` — data and metric builders; `scripts/figures/` contains short
  figure-specific generators that load prepared data.

## Proposed sequence

1. Freeze the research question and the exact dataset/snapshot split.
2. Agree the data and metric contracts.
3. Fill the arm matrix without running anything.
4. Sketch each figure and decide what claim it is allowed to support.
5. Build one small analysis script at a time.
6. Inspect the numbers and figure together, then mark the result reviewed.
7. Draft the corresponding dissertation subsection only after the evidence is
   accepted.

No training run is implied by this folder. The first pass can be evaluation-only
using existing data and saved checkpoints, if that is what we decide.
