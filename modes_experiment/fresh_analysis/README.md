# Fresh analysis workspace

This is a clean workspace for rebuilding the dissertation analysis step by step.
It deliberately does not copy results, figures, conclusions, or arm definitions
from the parent experiment. The existing `modes_experiment` files remain
untouched and are used only as references when we explicitly choose them.

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
- `arm_matrix.csv` — a fresh plan of analyses/arms; status starts as `planned`.
- `results_master.csv` — an empty long-form results table. Each row will carry
  one metric, its region, source, and review status.
- `data_contract.md` — the data paths, variables, units, and preprocessing we
  agree to use.
- `decisions.md` — a dated record of decisions and changes in scope.
- `design/` — the shared style guide, figure manifest, and one reviewed design
  note per figure.
- `figure_common.py` — the single source of shared typography, colours, sizes,
  axes, and export behaviour.
- `derived/` — machine-readable outputs produced by analysis scripts; never
  hand-edited.
- `figures/draft/` and `figures/final/` — generated figures only.
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
