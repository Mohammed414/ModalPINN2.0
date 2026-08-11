# Stages B-F results (first pass)

Status: A (data/separation) through F (withheld evaluation) have all run
at least once and produced honest, unmassaged results. Stage G
(robustness sweeps) intentionally not started yet, per the spec's own
staging discipline ("only after A-F are trustworthy") — see the open
question at the end of this document for why F isn't fully trustworthy
yet.

## Stage B — solver validation: PASS

`experiments/stage_b_validate_solver.py`, config: Nx=120, Ny=80 (dx=dy=0.1),
dt=0.005, T=400.

- No NaN/blowup. Divergence-free to machine precision in the interior
  (max|div u| = 2.2e-15 away from domain/immersed boundaries).
- Symmetric initial condition (impulsive freestream start) stays laminar
  until t~200-250, then self-excited symmetry breaking grows into a
  sustained, constant-amplitude limit cycle by t~300 — textbook behavior
  for an unperturbed symmetric start, not a numerical artifact.
- Shedding frequency (nonlinear sinusoid fit, not a raw FFT bin — the
  ~20-time-unit records used elsewhere give poor bin resolution):
  omega_lift=1.171, omega_tap0=1.145, both ~10-13% above omega_0=1.036.
  Grid refinement (dx=0.2 -> dx=0.1) moved the estimate from 1.181 -> 1.171,
  i.e. *converging toward* 1.036 as resolution improves — consistent with
  the bias being discretization/immersed-boundary error, not a wrong
  attractor. Documented in DESIGN.md as an accepted, expected limitation
  of the direct-forcing binary-masking IBM at this resolution.
- A real implementation bug was found and fixed during this stage: masking
  solid-region velocity to exactly zero *after* the pressure correction
  (rather than only before, as input to the Poisson RHS) silently
  reintroduced O(1) local divergence error at the fluid/solid interface —
  caught via a max|div(u)| sanity check (was ~0.66, now ~1e-15 interior).
  See the comment left in `estimator/ns_solver.py::step()`.

## Stage C — free-run control: done

`experiments/stage_c_free_run.py`. Initial condition: a single snapshot of
the Stage-B solver's own saturated limit cycle (`spin_up_solver.py`,
t=310 internal clock) — never touches the reference CFD. Run forward 201
steps at Delta_t=0.1 (matching the tap dataset), no assimilation.

## Stage D — minimal EnKF: done, with one real tuning finding

`experiments/stage_d_enkf.py`, q=16, ensemble ICs = time-jittered
(+/-0.4 time units, ~7.5% of one period) snapshots around the same base
phase Stage C used — see the important caveat below about what this
ensemble construction actually spans.

**Bug found and fixed during this stage**: at the very first assimilation
cycle, freshly-initialized ensemble members hadn't had `solver.step()`
called yet, so `solver.p` was still the zero-initialized array from
`__init__` — every member predicted identically zero pressure, making the
Kalman gain exactly zero for that cycle. Fixed by also saving/restoring a
dynamically-consistent pressure snapshot (not just u,v) for every initial
condition (`spin_up_solver.py` now saves `p` too).

**R calibration (observation-space only, not against truth)**: a naive
sigma_p = 1% of raw tap-pressure std caused the ensemble's pressure spread
to collapse ~10x within 8 cycles and NIS to blow up to ~30,000 (expected
value for a 32-dim observation is ~32). Root cause: with q=16 < n_taps=32,
the ensemble pressure-anomaly matrix Y has rank <= 15, so (Y Y^T + R) has
at least 17 directions where Y Y^T contributes ~0 and R alone (tiny)
dominates the inverse — the filter was applying huge, noise-driven
corrections in exactly the directions the ensemble has no real information
about. Swept sigma_p as a fraction of tap-pressure std: 0.3 gives NIS in
the 38-57 range (close to the ideal ~32), stable non-collapsing spread,
and a genuinely-decreasing-but-bounded innovation. Adopted as the nominal
default. This is exactly the kind of observation-space tuning the spec
asks for; withheld field error was never consulted for this choice.

**Negative controls**: shuffled pressure time-order gives similar
innovation magnitude to the nominal run (NIS 38-91 vs. nominal's 38-57) —
individual snapshot *values* of a periodic signal, reordered, still look
plausible instant-to-instant, so this control alone is not very
discriminating (see Stage F). Scrambled sensor identity gives a much
larger, clearly distinguishable innovation (NIS ~900-956, ~20x nominal) —
a good sanity check that the observation operator is genuinely spatially
sensitive to which tap is which.

## Stage E — empirical observability: informative, with an important caveat

`experiments/stage_e_observability.py`, q=16 fresh ensemble, 30-window
forecast-only run (no assimilation), whitened stacked SVD.

Singular value spectrum drops smoothly through mode ~6 (roughly 4-5 orders
of magnitude from mode 0 to mode 6), then falls to numerical noise floor
(~1e-40 and below) for modes 7-15. Interpretation: only ~6-7 of the 16
ensemble directions are meaningfully visible in wall pressure at all under
*this* ensemble.

Representer fields P_xy for two taps (shoulder ~90deg, base ~180deg) show
clean, physically coherent wave-like structure tracing the wake
downstream — not noise. Pressure-velocity correlation clearly has real,
spatially meaningful reach into the wake.

**Important caveat, stated plainly**: the ensemble was built by
time-jittering a *single* deterministic trajectory (chosen specifically
because every member is then trivially physically valid without touching
the reference). To leading order, nearby points in time along one periodic
orbit differ mainly along one tangent ("phase") direction, with
progressively weaker higher-order corrections — so a sharp low-rank
spectrum is close to what this construction would produce *even if the
full state space were richly observable from pressure*. This result is
genuine and worth reporting, but it conflates two different things: (1)
how much of the *true* state space is observable from 32 wall taps, and
(2) how much of *this particular low-diversity ensemble* is observable.
Distinguishing them needs a richer ensemble (e.g. small structured
divergence-free perturbations added on top of the phase spread, or
multiple independent spin-ups from different generic initial conditions)
— flagged as the top follow-up, not yet done.

## Stage F — withheld evaluation: done, honest result is mixed

`evaluation/stage_f_evaluate.py`. Compares free-run / EnKF / shuffled
against `reference_truth_full.npz` (untruncated scattered CFD truth, not
the 3-mode Mtrue_*).

Phase metric note: the first attempt used tap-0's raw pressure for a
Hilbert-transform phase estimate and got a visibly noisy, jagged result —
traced to Stage A's own earlier finding that tap-0's pressure spectrum is
dominated by the *second* harmonic (2*omega_0), not the fundamental,
which breaks a simple instantaneous-phase estimate. Switched to a crude
lift-coefficient proxy built from all 32 measured taps (`sum p*n_y*dtheta`
around the sorted taps) for the true-side phase reference, and each run's
own IBM reaction-force Fy (confirmed clean single-frequency in Stage B)
for the estimate-side phase. Result is much smoother/more interpretable.

Results (mean over the 20-time-unit window):

| | E_u | E_v | mean &#124;dphi&#124; (rad) |
|---|---|---|---|
| free-run | 0.1413 | 0.6333 | 0.7068 |
| EnKF | 0.1317 | 0.5513 | 0.6003 |
| shuffled | 0.1340 | 0.5766 | 0.6242 |

EnKF beats free-run on every metric, consistently across the whole
20-time-unit window (not just on average — E_u(t)/E_v(t) plots show EnKF
at or below free-run almost everywhere). That is a real, if modest
(~7-13%), effect.

**But** EnKF's margin over the *shuffled* negative control is much
smaller than its margin over free-run, and the phase-error time series for
all three runs (free-run, EnKF, shuffled) drift together almost linearly
at nearly the same rate over the 20-time-unit window — a rate consistent
with the ~13% solver-vs-truth frequency mismatch found in Stage B, common
to every run regardless of assimilation. Plausible reading: (1) most of
the EnKF-vs-free-run gain comes from correcting toward realistic pressure
*amplitude/gauge*, which a shuffled-but-still-real-valued control partly
gets for free since shuffling preserves the marginal distribution of tap
values; (2) genuine phase-locking is being fought, cycle after cycle, by
the solver's own ~13% intrinsic frequency bias reasserting itself between
corrections; (3) Stage E's low-rank finding means the ensemble may simply
not span enough independent directions to fix this from 32 pressure taps
under this particular initialization scheme.

Per the spec's own honesty requirement: this is **not** a clean pass on
the phase-synchronization success criterion. E_u/E_v show a real,
consistent, modest win for pressure assimilation over no information at
all — that part is solid. Phase-specific synchronization strictly better
than a "right values, wrong timing" shuffled control is not yet
convincingly demonstrated. Reporting this as-is rather than reaching for a
more flattering metric.

## Open question / top follow-up

Is the weak EnKF-vs-shuffled phase discrimination a genuine physical
statement about pressure-only observability of the fundamental's phase, or
an artifact of the low-diversity (time-jitter-of-one-trajectory) ensemble
Stage E flagged? Distinguishing these requires re-running Stage D/E/F with
a richer initial ensemble (structured divergence-free perturbations, or
several independent spin-ups) before treating the current Stage F verdict
as final. Stage G (sensor count/noise/ensemble-size/inflation sweeps) is
deliberately not started until this is resolved, since sweeping
hyperparameters around a possibly-too-narrow ensemble would risk
optimizing the wrong thing.
