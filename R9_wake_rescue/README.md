# R9_wake_rescue — creative taps+physics-only wake reconstruction

Fully isolated folder (same policy as R6/R7/R8: nothing outside this folder is
modified). Everything here was produced in the Claude Science session of
2026-08-13.

## The legitimacy rule (hard constraint on every method here)

Every reconstruction method in this folder may see ONLY:

1. **The 32 wall tap pressure signals** (`data/sensor_indices/taps_32.npz`) —
   including anything derivable from them alone (shedding frequency, lift/drag
   time series via surface-pressure integration, harmonic amplitudes/phases).
2. **Physics**: the incompressible Navier-Stokes equations at Re=100, boundary
   conditions, and known *structural* facts about the Karman limit cycle that
   follow from the equations (spatio-temporal shift-reflect symmetry,
   mean-flow marginal stability / RZIF, classical analytic vortex-street
   solutions). No PDE time-stepping solver is used anywhere.

The reference CFD (`data/fixed_cylinder_atRe100`) is used ONLY:
- to extract the 32 tap signals (already done, `taps_32.npz`),
- to build *diagnostic* fields for the phase-0 contrast-ratio measurement
  (clearly marked; never trained on),
- for final validation with the standard regional-error protocol
  (identical bins + rel-L2 formula as `src/pressure_only/evaluate_regions.py`).

Banned per user instruction: POD (any basis), EnKF / any data assimilation
with a solver, any PDE time-stepper, and re-use of R1–R8's failed loss terms
(BVF, K0Loss, CV1, PhaseLoss, causal weighting) as the proposed fix.

## Layout
- `src/` — all code
- `cache/` — regenerable intermediates (gitignore-worthy, large)
- `figures/`, `results/` — outputs
