# Pressure-only EnKF flow reconstruction — design deliverable (pre-implementation)

Status: Stage A data prep done and validated. Everything below Stage A is
**design only** — no forward solver / EnKF code has been written yet. This
document is the "first deliverable" requested before any substantial
computation, per the experiment spec.

## 0. What Stage A already established (facts, not proposals)

Inspected `data/fixed_cylinder_atRe100` and `data/sensor_indices/taps_*.npz`
directly (see `evaluation/build_reference_truth.py`, executed and its output
checked below).

- Raw CFD file: Re=100, "Ur"=1e6 (i.e. effectively infinite reduced
  velocity → rigid fixed cylinder, matches the filename), Nt=201 snapshots,
  dt=0.1, t∈[400,420] (≈3.3 shedding periods), N_nodes=82,872 on an
  unstructured mesh spanning a huge far-field domain (x∈[-40,120],
  y∈[-60,60]) — the CFD's own domain is far larger than the region of
  interest ModalPINN (and this experiment) actually works in.
- Cropped to the ROI used throughout this project's ModalPINN work
  (x∈[-4,8], y∈[-4,4], cylinder r_c=0.5 at origin — identical to
  `src/pressure_only/ModalPINN_VortexShedding.py`'s `geom`): 51,654 nodes.
- All 4 tap sets (4/8/16/32) sit exactly on r=r_c (checked to 1e-3), times
  are strictly monotonic with dt=0.1, pressure values come from the same
  raw CFD file (same gauge as the full field — no cross-file gauge
  ambiguity in the *data*; the gauge issue is between the *future NS
  observer's own pressure* and this data, see §5).
- omega_0=1.036 (the project's documented constant) is well supported by
  the data: a 3-mode (k=0,1,2) harmonic least-squares fit at this frequency
  reconstructs the raw u field to **1.3% relative RMS residual** — strong
  independent confirmation of both the frequency and the adequacy of a
  3-mode truncation (also validates why ModalPINN itself uses Nmodes=3).
- Produced and visually sanity-checked (`figures/sanity_modal_truth.png`):
  Mtrue_u0 shows the expected mean recirculation bubble, Mtrue_v1
  magnitude shows the expected elongated fundamental-mode wake envelope,
  Mtrue_p2 shows near-wall/near-wake concentration of the 2·omega_0
  pressure harmonic — all physically correct patterns.

Two output files now exist under `enkf_pressure_only/data/`:

- `tap_observations.npz` (small, committed) — **the only file the
  estimator may load**: tap_x/y/times/p for n_taps∈{4,8,16,32}, Re, r_c,
  x_c, y_c, domain, omega_0.
- `reference_truth_modal.npz` (small, committed) — evaluation-only: gx,gy
  grid + Mtrue_u0/u1/u2, v0/v1/v2, p0/p1/p2.
- `reference_truth_full.npz` (110MB, gitignored, regenerable) —
  evaluation-only: ref_x, ref_y, ref_times, ref_cu, ref_cv, ref_cp (raw,
  untruncated scattered CFD snapshots), for Stage F metrics that must not
  be contaminated by the 3-mode truncation itself.

An enforced leakage guard is implemented and tested
(`estimator/_leakage_guard.py`): importing `estimator` monkeypatches
`np.load` to raise `LeakageError` on any attempt to open a withheld-truth
filename, unless inside `evaluation`'s `with allow_truth_access():`.
Verified: blocks the leak, allows it inside the context manager, and
re-blocks after the context exits.

## 1. Exact state vector

```
x = [u_i, v_i]  for i in INTERIOR ACTIVE fluid grid points only
```

"Interior active" excludes (a) grid points fixed by a Dirichlet BC
(inflow, top/bottom far-field) and (b) grid points inside/immediately at
the immersed cylinder. This is exactly the solver's own native unknown
set — the same DOFs the momentum/projection solve iterates over — not an
enlarged or reduced representation. Pressure is **not** part of x; it is
always re-diagnosed from the current velocity field via the projection
solve (see §2), both during the forecast and immediately after every
analysis update (needed to re-evaluate h(x) at the next cycle).

Rationale for excluding BC/solid DOFs from x rather than including them
and projecting afterward: it makes the "state stays physically valid after
an affine EnKF combination" requirement (§9) structural rather than a
post-hoc patch for the two BC-DOF classes — only the immersed-boundary
forcing region needs an explicit post-analysis check, not the far-field/
inlet boundary.

## 2. Independent NS solver — the major implementation dependency

**Checked and confirmed: no incompressible-NS/CFD/FEM package is
installed in this environment.** Explicitly checked: FEniCS/dolfin/
dolfinx, Firedrake, PyFR, py-pde, fluidsim, PhiFlow, FiPy, scikit-fem, and
system-level OpenFOAM/gmsh — none present, and none are lightweight
`pip install`s (FEniCSx in particular typically wants conda and a sizeable
environment). Per the task's own instruction, this is flagged as the
**critical implementation dependency**: a solver must be hand-written.

Proposed method (textbook, not novel — the risk is implementation
correctness, not the numerical method itself):

- **Grid**: uniform Cartesian grid over the same ROI as the truth
  (x∈[-4,8], y∈[-4,4]), target spacing dx=dy≈0.1 (Nx≈121, Ny≈81,
  ≈9,800 points) — deliberately coarser than the hidden truth's
  unstructured mesh, per the spec's explicit allowance.
- **Cylinder**: immersed boundary, direct-forcing method (Fadlun et al.
  2000 style) rather than a body-fitted curvilinear grid. Chosen over a
  body-fitted O-grid because it keeps the momentum/Poisson equations in
  plain Cartesian finite-difference form (far fewer places for a metric-
  term algebra bug to hide) at the cost of only an approximate no-slip
  condition — a well-precedented, heavily validated tradeoff for exactly
  this Re=100–200 benchmark. Direct forcing (not feedback forcing) is
  chosen specifically to avoid an extra spring/damping gain that is
  fiddly to tune and can destabilize.
- **Time integration**: fractional-step (Chorin/Kim–Moin) projection:
  1. Predictor: explicit 2nd-order (Adams–Bashforth) advection +
     implicit viscous diffusion (Crank–Nicolson via sparse solve) → u*.
  2. Apply IBM direct forcing near the cylinder to drive u* toward
     no-slip.
  3. Pressure-Poisson solve for the projection potential φ enforcing
     div(u)=0.
  4. Correct: u = u* − Δt·∇φ; diagnostic pressure p = φ/Δt.
- **Boundary conditions**: Dirichlet freestream (u=1,v=0) on inflow
  (x=Lxmin) and top/bottom (y=±Lymax) — matching ModalPINN's own
  `FreestreamBC` convention at the same domain edges; simple Neumann
  (zero-gradient) outflow at x=Lxmax for the first version, upgraded to a
  convective/Orlanski condition if Stage B shows reflection artifacts.
- **Performance-critical detail**: the discrete Laplacian for the pressure
  Poisson solve depends only on the (static) grid/geometry, never on the
  flow state. It is **factorized once** (`scipy.sparse.linalg.splu`) at
  solver setup and reused — as a fast triangular solve — for every
  substep of every ensemble member's every forecast. This is what makes
  the EnKF's repeated-forecast cost tractable (see §11).

## 3. How wall pressure is obtained from the state

After each forecast step's projection solve, the diagnostic pressure field
p lives on the Cartesian grid's fluid cells. Wall pressure is **not** read
directly off any solid/forcing-region cell (those are not well-defined
physical fluid pressure in a direct-forcing IBM). Instead: for each tap,
locate the nearest fluid-cell neighbors along the local outward normal
direction from the cylinder surface and extrapolate to r=r_c — the
standard "wall probe" approach for extracting surface loads from an
immersed-boundary solve.

## 4. Tap interpolation

Bilinear interpolation (via the wall-probe extrapolation of §3) evaluated
at each tap's exact (x,y). All 32 (or 4/8/16) taps interpolated in one
vectorized call per forecast step — this is `h(x)`.

## 5. Pressure gauge treatment

Exactly as specified: at each assimilation time,

```
c = mean_over_taps(p_measured − p_predicted)
p_predicted* = p_predicted + c
r = p_measured − p_predicted*
```

This removes only the arbitrary additive constant that differs between
the observer's own pressure-Poisson gauge (pinned however the discrete
Poisson solve's null space is resolved — e.g. mean-zero over the domain)
and the raw CFD file's gauge. It does **not** touch the spatially-varying
part (front-stagnation-high vs. base-low pressure), which is the
physically meaningful signal the filter is meant to use. Applied
identically during Stage E's observability diagnostic (the pressure
anomaly matrix Y is unaffected by a constant gauge shift anyway, since Y
is built from deviations from the ensemble mean — the gauge correction
only matters for the innovation, not for Y itself).

## 6. Ensemble size

q=16 for the first minimal trustworthy test (Stage D), per the spec's
suggested starting point. Swept over {8,16,24,32} in Stage G once the
nominal case is trustworthy.

## 7. Assimilation interval

Every available measurement in the nominal test: Δt_assim = 0.1 (matches
the tap data's native sampling — all 201 timesteps). The solver's own
internal timestep will be smaller (Δt_solver ≈ 0.01–0.02, i.e. 5–10
substeps per assimilation cycle) for numerical stability; this is a
solver-internal detail, not the assimilation interval. Swept to every
2nd/4th/8th measurement in Stage G.

## 8. Inflation / noise assumptions

- Multiplicative inflation α on the forecast anomalies X ← αX, default
  α=1.0 (off) for the first run, configurable; a small sweep (α∈
  [1.0,1.1] typical) tuned only against observation-space diagnostics
  (innovation magnitude/NIS/ensemble pressure spread/divergence), never
  against withheld field error, per the spec.
- R = σ_p² I. The tap "measurements" are deterministic CFD values with no
  noise added by nature — a literally-zero R risks an ill-conditioned
  K if the ensemble's pressure spread YYᵀ also gets small, so the nominal
  synthetic test uses a small explicitly-documented noise floor,
  σ_p = 1% of the RMS tap-pressure fluctuation amplitude (computed from
  the tap data itself, a property of the observation, not of the hidden
  full field — not a leak). Deliberately-larger noise is a separate,
  later robustness axis (Stage G item 2), not this default.

## 9. Incompressibility / BC validity after the analysis update

- **Divergence-free**: div is linear, so div=0 is a genuine linear
  subspace (not merely affine). Since (a) every forecast ensemble member
  is divergence-free to machine precision (direct sparse solve, not an
  iterative one with a tolerance) and (b) the standard EnKF analysis
  update for each member is x_f + K(y − y_f) with K built entirely from
  the forecast anomaly matrix X, the correction K(·) is by construction a
  linear combination of divergence-free anomaly columns, hence itself
  divergence-free — so xbar_a and every member's analysis state are
  divergence-free automatically, **no extra projection needed**. This
  will still be checked numerically post-hoc (compute discrete divergence
  of analysis states, confirm it's at solver machine-precision level), as
  requested.
- **Far-field/inlet BC**: not part of x at all (§1) — trivially preserved,
  nothing to enforce.
- **No-slip at the cylinder**: direct-forcing IBM enforces no-slip via a
  forcing *term* during time-stepping, not a hard linear constraint on the
  state DOFs, so an analysis correction could in principle leave a small
  residual at forcing-region points (anomalies there are small in
  practice, since every ensemble member already approximately satisfies
  no-slip, but not exactly zero). Mitigation: explicitly re-zero velocity
  at forcing-region grid points immediately after every analysis update
  (cheap, well-justified, exactly the "appropriate projection step" the
  spec allows for) — belt-and-braces, since the next forecast step's IBM
  forcing would clean this up anyway even without it.

## 10. Information flow

```
 data/fixed_cylinder_atRe100 ─────┐   (raw hidden CFD truth)
 data/sensor_indices/taps_*.npz ──┤
                                   ▼
              evaluation/build_reference_truth.py   [ONLY script touching raw CFD]
                                   │
            ┌──────────────────────┴───────────────────────┐
            ▼                                               ▼
  data/tap_observations.npz               data/reference_truth_{modal,full}.npz
  (x,y,t,p for 4/8/16/32 taps,             (Mtrue_* modal truth, gx/gy grid,
   Re, r_c, domain, omega_0)                ref_cu/cv/cp raw scattered truth)
            │                                               │
            ▼                                               │
  ┌────────────────────────────────────┐                    │
  │  estimator/  (leakage-guarded)      │                    │
  │  - data_interface.TapObservations   │                    │
  │  - forward NS solver (IBM Cartesian)│                    │
  │  - EnKF (Experiments 0–3)           │                    │
  │  → observer state history, tap      │                    │
  │    predictions, innovations, etc.   │                    │
  │    (saved to disk, e.g. npz/json)   │                    │
  └────────────────┬─────────────────────                   │
                    │  observer run finished, state saved     │
                    ▼                                          │
  ┌───────────────────────────────────────────────────────────┘
  │  evaluation/  (only place `allow_truth_access()` is used)
  │  loads observer's saved state history + reference_truth_*.npz
  │  → E_u(t)/E_v(t), vorticity error, phase error, plots
  └───────────────────────────────────────────────────────────
```

The `estimator` package can physically only reach
`tap_observations.npz` through `data_interface.TapObservations`; any
direct `np.load` of a truth filename anywhere in the process raises
`LeakageError` unless inside `evaluation`'s context manager.

## 11. Estimated computational cost (pre-implementation estimate)

Grid ≈9,800 points → state dim ≈18–19k. Per solver substep: vectorized
numpy predictor (~ms) + one sparse triangular solve reusing the
precomputed pressure-Poisson factorization (~ms). At ~5–10 substeps per
0.1-time-unit assimilation interval: roughly 50–100ms per member per
cycle. With q=16 and 201 assimilation cycles (serial, not yet
parallelized across members): **very roughly 3–7 minutes** for the entire
nominal Stage D run; Stage B's spin-up to a periodic limit cycle (10–20
shedding periods, ≈60–120 time units) for one trajectory: **roughly
1–2 minutes**. These are rough pre-implementation estimates, not
measurements — will be replaced with profiled numbers once Stage B runs.
The main risk to this estimate is the timestep needed for stability
turning out smaller than assumed (cell Reynolds/CFL near the immersed
boundary), or Poisson conditioning being worse than expected.

## 12. Known mathematical/numerical obstacles

- **IBM no-slip is approximate**, not exact — near-wall resolution error
  could shift the observer's *natural* (un-assimilated) Strouhal number
  away from omega_0≈1.036 even with the right Re. This is exactly what
  Stage B's validation is for; qualitatively wrong physics (no periodic
  shedding at all, or a wildly different frequency) blocks proceeding, per
  the spec. A quantitatively small mismatch is expected and acceptable —
  it's part of why an *independent* forward model is more interesting
  than reading off the truth.
- **Outflow reflection risk**: a plain Neumann outflow at x=Lxmax can
  partially reflect vortical/pressure structures. Watch for artifacts in
  Stage B; upgrade to a convective outflow condition if seen.
- **Ensemble collapse / spurious covariance risk**: q=16–24 against a
  state dimension of ~18–19k is a large ratio imbalance, a known source
  of spurious long-range sample covariance in EnKF. Mitigated by
  inflation (as specified) but the spec does not call for spatial
  localization; if Stage E's representer fields (P_xy for a given tap)
  show unphysical long-range influence, localization (e.g. Gaspari–Cohn
  tapering) is a natural follow-up, flagged here rather than
  pre-emptively implemented.
- **Perturbed-observation EnKF sampling noise**: adds Monte Carlo noise on
  top of the true analysis update, more noticeable at q=16. The spec
  explicitly allows deferring an ETKF/square-root variant; flagged as a
  follow-up if Stage D results look noisier than the observation-space
  diagnostics alone would suggest.
- **Domain-size mismatch**: the raw CFD's actual computational domain is
  vastly larger (x∈[-40,120], y∈[-60,60]) than the ROI (x∈[-4,8],
  y∈[-4,4]) both the truth-extraction and the observer use. The truth's
  far-field BCs were effectively applied much farther away than the
  observer's hard boundaries at y=±4/x=8/-4. This is an existing,
  accepted limitation shared with all of this project's prior ModalPINN
  work on the same ROI, not something new introduced here — documented
  for completeness.
- **3-mode truncation residual**: Mtrue_* reconstructs the raw field to
  ~1.3% RMS residual — excellent but not exact. Primary Stage F field-
  error metrics (E_u(t), E_v(t), vorticity error) will use the
  **untruncated** `reference_truth_full.npz`, not Mtrue_*, to avoid
  circularity with ModalPINN's own modal assumption. Mtrue_* is used for
  wake-amplitude/mode-shape diagnostics that mirror this project's
  existing ModalPINN evaluation conventions.

## Stage plan (unchanged from spec, restated for tracking)

A. Data + truth/estimator separation — **done, validated above**.
B. Independent NS solver implementation + validation (own periodic
   Kármán street, frequency check against omega_0).
C. Free-run wrong-phase control.
D. Minimal EnKF, 32 taps, q=16.
E. Empirical observability/detectability diagnostic.
F. Withheld full-field evaluation + negative controls (shuffled/scrambled
   pressure).
G. Robustness sweeps (sensor count, noise, assimilation interval,
   ensemble size, inflation, model mismatch, coarser grid) — only after
   A–F are trustworthy.

Every stage prints explicit PASS/FAIL checks and saves enough state
(config as JSON, fixed seeds) to reproduce the result, per the spec.
