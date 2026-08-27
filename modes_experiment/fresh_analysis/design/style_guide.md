# Dissertation figure style

This directory contains figure decisions and wireframes only. Generated files
belong in `../figures/`; scientific calculations belong in `../scripts/`.

## Shared implementation

All presentation settings live in `../figure_common.py`. Figure-specific scripts
must not redefine fonts, method colours, dimensions, panel-label placement, or
export settings. They should load already-derived data and contain only layout,
marks, and figure-specific annotations.

## Fixed visual rules

- Full-width figures are 178 mm wide; single-column figures are 85 mm wide.
- Use DejaVu Sans with STIX sans-serif mathematics.
- Export a 300-dpi PNG at the declared physical figure width.
- Use the same semantic colour for a method everywhere.
- Use colour-blind-safe colours and never use a rainbow colormap.
- Side-by-side scalar fields share limits and colormaps.
- Signed errors and corrections use a zero-centred diverging scale.
- Magnitudes use a perceptually uniform sequential scale.
- Use direct labels when they are clearer than a legend.
- Do not place `(a)`, `(b)`, or other LaTeX panel labels inside the image; add
  them in LaTeX when a multi-panel figure is assembled.
- Keep backgrounds white, axes thin, and decoration minimal.
- State the evaluated variable, region, normalization, and units in the caption.

## Semantic method colours

| method | colour role |
|---|---|
| CFD reference | black |
| pressure-only reference | neutral grey |
| sparse velocity probes | blue |
| dense observation ceiling | green |
| analytical Kármán prior | vermillion |
| prior plus network | purple |

These roles are encoded in `figure_common.py`; hex values should not be copied
into individual plotting scripts.
