# F0a — Evaluation regions

## Purpose

Define every spatial mask before regional errors are reported. The figure is a
clean geometry schematic rather than a mesh-density plot; node counts come from
the audited machine-readable masks.

## Design decisions

- Near cylinder, near wake, far wake, and other are shown as exact filled
  geometry from the inequalities used by the evaluation masks.
- Far core is shown as an opaque hatched subset because it is nested inside far
  wake rather than being another partition member.
- The solid cylinder remains white; the near-cylinder fluid annulus is visible.
- Exact inequalities and node counts are printed directly inside the relevant
  region.
- A labelled radius defines the symbol `r` used in the inequalities.

Output: `figures/final/F00_evaluation_regions.png`. Status: **complete**.
