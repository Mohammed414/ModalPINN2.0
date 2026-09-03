# Section blueprint

Writing skeleton. Every slot below names the evidence it is written from, so no
section needs re-derivation at writing time. All analysis is closed as of
2026-08-28; nothing here is waiting on a result.

**Sources.** Numbers: `results_master.csv` (945 rows, all `status: accepted`).
Interpretation: `findings.md`. Frozen definitions: `data_contract.md`. Scope
choices and their reasons: `decisions.md`. Figures: `figures/final/` (13 PNGs,
300 dpi), indexed in `design/figure_manifest.csv`.

**Standing rule.** Far-core $v_1$ must never be quoted as an unconstrained
network result in A04, A05 or A06. For $x\geq3$, $|y|\leq2$,
`--V1RadialTrust` uses the analytical prior plus a learned correction bounded
below 60% of the local prior amplitude. Far-core agreement is therefore a
prior-assisted result and must always be compared with the prior alone.

---

## Methodology

### M1. Research question and comparison

- Question: does a sparse-measurement ModalPINN reconstruct the wake, and if so
  which part of the answer comes from the network rather than the prior?
- Comparison set: prior only, network only, and prior + network, evaluated
  under one protocol. Definitions in `analysis_matrix.csv`.
- Excluded and why: Arms 2, 3, 14, 16 — see `decisions.md`. State the exclusions
  explicitly rather than silently omitting them.
- **Evidence:** `analysis_matrix.csv`, `decisions.md`.

### M2. Dataset and preprocessing

All frozen in `data_contract.md` ("Dataset and physical convention"); transcribe
rather than restate from memory.

- Canonical CFD file `1_data/fixed_cylinder_atRe100`, SHA-256 recorded; analysis
  cache `1_data/flow_cache.npz`, SHA-256 recorded.
- Cache validated against the canonical text file: max field difference
  5.960e-08, coordinate difference 3.812e-06, consistent with float32 storage.
- Re = 100, $U_\infty = 1$, cylinder at origin, $R = 0.5$, $D = 1$. Coordinates
  reported as $x/D$, $y/D$ (numerically equal, since $D = 1$).
- 201 uniformly spaced snapshots, $t = 400$ to $420$, $\Delta t = 0.1$.
- No deletion or imputation; audit confirms no NaN or infinite values.
- Pressure gauge: stored CFD pressure and the reconstruction's trained gauge,
  no post-hoc offset removal. The sensitivity of this choice is quantified in
  `derived/a04_pressure_gauge_check.json` and belongs in Limitations.
- **Evidence:** `data_contract.md`, `derived/source_inventory.csv`.

### M3. Reconstruction methods and the information boundary

The information boundary is the part an examiner will probe hardest — state it
before any result.

- **Analytical prior.** Consumes the 32 cylinder-pressure time series plus fixed
  classical physics and geometry. It does *not* fit interior CFD velocity or
  pressure. Disclose the drag assumption: total drag is inferred from
  tap-integrated pressure drag using the fixed fraction 0.75, a
  prior-construction assumption rather than an evaluation input.
- **Trained reconstruction.** Evaluated from the saved checkpoint with the exact
  inference-time wrappers recorded in each arm's `run_record.json`. Inference
  only; no training operation is used at evaluation.
- **Measurements.** Pressure taps at 8/16/32 uniform angles on $r = 0.5$;
  sparse velocity probes 10 per section at $x/D = -3, 1, 2, 3$ (40 nodes, max
  target-to-node displacement $0.1758D$); dense Arm 5 sees 5,000 sampled
  space-time points and is a representational ceiling, not a matched ablation.
- **Train/evaluation relationship — state plainly.** Evaluation uses all 201
  snapshots and all 51,654 cropped nodes. Most interior nodes are unobserved
  during sparse training, but the evaluation interval is *not* temporally held
  out. This is a full-field reconstruction comparison, not a held-out test set.
  Repeat the point in Limitations.
- **Evidence:** `data_contract.md` ("Measurements", "Prior and checkpoint
  information boundary", "Training and evaluation relationship").

### M4. Metrics and visual diagnostics

Transcribe from `data_contract.md` ("Evaluation-metric contract"). Every metric
needs its definition, ideal value, interpretation, and limitation.

- Regional relative $L^2$ for $u$, $v$, $p$: define the norm, justify
  normalising by the reference-field norm, state the space/time aggregation,
  and explain 0, 1, and values above 1 (above 1 = worse than predicting zero).
- First-shedding-harmonic diagnostics: relative $L^2$, amplitude ratio,
  normalised complex correlation, signed phase offset. Note the amplitude-ratio
  failure mode explicitly — it inflates without bound against a near-zero
  reference, which is what produces A03's 8.0x and A04's 11.5x upstream figures.
- Regions: fixed masks and node counts, shown visually *before* any regional
  result. Near cylinder 13,715 + near wake 15,248 + far wake 16,393 + other
  6,298 = 51,654, an exact partition; far core (12,460) is a nested subset of
  far wake, not a fifth region.
- Out of scope: force, lift and drag metrics (`decisions.md`).
- **Figures:** F00 (regions), F00a (probes), F00b (taps).
- **Evidence:** `data_contract.md`, `derived/a00_geometry.npz`.

### M5. Optimisation and convergence — the termination pathology

New section; the blueprint had no slot for this. It belongs in Methodology
because it explains why several results below carry an effort caveat, and the
reader needs it before those results, not after.

- SciPy's L-BFGS-B convergence test uses `ftol = 1e-12`, below the float32
  resolution of the loss (~1e-10 at these values). It can therefore only be
  satisfied by a line search that moves the loss by exactly zero — a failed
  line search, reported as convergence.
- Census across all 17 project runs: 16 terminate this way; the exception,
  `05_dense_reference`, hit its evaluation cap instead. Runs stopping before
  12,000 iterations exited at median 4.3x their recent gradient level
  (0.9–26.7x, n = 9); runs past 20,000 at 1.0x (0.6–2.1x, n = 8). Spearman
  $\rho = -0.733$, $p = 8.19 \times 10^{-4}$, $n = 17$.
- Consequence: effective optimiser effort is a random variable seeded by
  float32 GPU rounding, not a controlled quantity. A matched-effort re-run of
  arm 01 was attempted and failed — same wall clock to within 0.04%, 1.30x the
  evaluations, 1.77x worse loss — so the confound cannot be removed by
  re-running.
- Tap count is ruled out as the cause: two 32-tap arms differing from the
  baseline only in collocation placement ran 41,171 and 35,313 iterations
  against its 5,081.
- **Figures:** F_termination_anatomy, F_arm1_rerun_breakdown.
- **Evidence:** `findings.md` ("Training effort"),
  `scripts/termination_census/`,
  `notebooks/matched_effort/01b_matched_effort_outcome.md`.

---

## Results

### R1. Reference data and sanity checks

Geometry, region definitions and node counts, measurement locations.

- **Figures:** F00, F00a, F00b. **Evidence:** `findings.md` (A00).

### R2. Prior-only reconstruction

How much of the field the analytical prior already supplies. Prior-alone $v_1$
relative $L^2$: 1.372 near cylinder, 0.696 near wake, 0.285 far wake, 0.282 far
core — i.e. the prior is worst exactly where the sensors are.

- **Evidence:** `derived/a04_prior_only_metrics.json`, `findings.md` (A04).

### R3. Network-only reconstruction

What the network produces without the prior, under the same protocol. Report as
the observed endpoint, not a one-sided bound (M5).

- **Evidence:** `findings.md` (A04, "Training effort"), A02 and A03 rows in
  `results_master.csv`.

### R4. Prior plus network — the division of labour

The central result. Near the body the network contributes strongly (+70.9%
near cylinder, +44.8% near wake); in the far field it does not measurably
improve the prior (−2.5% far wake, −2.7% far core), so the good far-wake
structure is supplied by the prior, not learned from sparse data. Single seed,
so this supports "no improvement", not "reliably worse".

- **Figures:** F01, F02, F02b. **Evidence:** `findings.md` (A04).

### R5. Controlled-parameter studies

Each subsection states its caveat *with* its result, not in a footnote.

| study | arms | status | caveat to state inline |
|---|---|---|---|
| Information content | 1, 4, 5 | accepted | dense arm is a fixed-budget representational reference |
| Tap count | 8, 9, 1 | accepted | effort differs; endpoint comparison only |
| Collocation strategy | 1, 6, 7 | with caveat | effort not controlled, 6.9–7.9x |
| Prior + collocation | 15, 10 | with caveat | effort not controlled, 1.33x |
| Pressure noise | 15, 11–13 | with caveat | one seed, 11x effort spread |

For A03 and A05, state the effort gap without assigning it a favourable
direction. Evaluation count is not an accuracy proxy, so neither gains nor
losses are causal estimates of sampling alone. The shared endpoint result is
that neither tested collocation change recovers the travelling wake.

- **Figures:** F03, F04a, F04b, F04c, F04d. **Evidence:** `findings.md`
  (A01, A02, A03, A05, A06).

---

## Discussion

### D1. Direct answer to the research question

State it with the metric and region attached. The short form: the reconstruction
works near the body, where the measurements are; the far wake is supplied by the
prior, and no configuration tested recovers it from sparse data.

### D2. Attribution

Separate prior, network, and data. Two mechanisms are worth drawing out because
they recur:

- **Amplitude without phase.** Wake-biased collocation raises far-core amplitude
  ~16x while correlation with the true mode falls ~40%, and raises upstream
  leakage 6.6x where the true answer is zero. A reconstruction can look more
  wake-like and be less correct; the metric is right and the eye is wrong.
- **Invariance that belongs to the prior.** Far-field stability under noise
  (A06) is the trust region's, not the network's — all four arms sit within
  0.043 of the prior-alone level.

### D3. Limitations

- Single seed throughout.
- Evaluation interval not temporally held out (M3).
- Effort uncontrolled in A03, A05, A06, and structurally so (M5).
- Amplitude-ratio inflation against near-zero references (A03, A04 upstream).
- Pressure-gauge convention affects far-field pressure errors materially.
- Not addressed by the evidence: force reconstruction, optimizer ablation.

### D4. Implications and next experiment

Only what the accepted results support. The obvious next experiment is the one
the effort pathology blocks: a genuinely effort-matched collocation comparison,
which needs the `ftol`/float32 termination fixed first — arguably the more
valuable contribution.
