# F0b — Measurement locations

## Purpose

Show the information supplied to the sparse experiments without mixing the
domain-scale velocity probes with the cylinder-scale pressure taps.

## Design decisions

- The velocity probes and pressure taps are two independent single-column PNGs
  that can be arranged in LaTeX.
- The probe figure shows the actual nearest CFD nodes, the requested sections,
  and that no section lies downstream of `x/D=3`.
- The tap figure combines a true-scale cylinder ring with one unrolled
  quadrant, making the nested 8/16/32 interleaving explicit.

Neither image contains `(a)/(b)` labels; LaTeX adds them if the figures are
used as subfigures. Outputs: `figures/final/F00a_probe_locations.png` and
`figures/final/F00b_tap_layout.png`. Status: **complete**.
