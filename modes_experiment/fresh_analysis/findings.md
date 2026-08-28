# Findings notes

Running notes on what the numbers appear to mean, kept so the reasoning is not
lost between sessions. This is raw material for the Results and Discussion
sections, not finished prose.

Rules for this file:

- Numbers live in `results_master.csv` and `derived/*.json`; quote them here
  only to make a point, and always with the region attached.
- Scope and method decisions live in `decisions.md`, not here.
- Anything speculative is marked **[unverified]** so it does not get written up
  as a result by accident.
- If a later run changes a number, update the note and say what changed.

For each new result, keep the same order: **claim**, **evidence** (metric,
region, and source path), **interpretation**, **limitation**, and **figure or
result path**. This keeps a useful scientific notebook separate from finished
dissertation prose.

---

## A00 — data and regions

**Regions are a genuine partition.** near-cylinder + near-wake + far-wake +
other = 51,654, exactly the evaluation crop. Far core (12,460) is a nested
subset of far wake (16,393), not a fifth region — worth stating once in the
methodology so no reader adds them up.

**The near-cylinder mask is one-sided.** 326 surface nodes sit at
r = 0.499975, marginally inside the nominal cylinder radius — this is where the
CFD mesh actually places its wall nodes, not solely a floating-point artefact
(0.5 is exactly representable in float32). A two-sided `0.5 <= r < 0.75` would silently
drop all 326. The stored mask is `r < 0.75`, verified to reproduce the recorded
count of 13,715 exactly. Figure F0a and the methodology both use the one-sided
form. Small, but the kind of thing an examiner notices if the figure and the
text disagree.

---

## Training effort — the L-BFGS termination pathology affecting arm 01

**Claim:** the pressure-only + physics baseline (`01_baseline_physics_only`, the
network-only reference used by A02 and A04) is under-trained relative to its
partner arms, for a reason that is characterised and reproducible rather than
incidental, and this is why it is reported as a non-converged lower bound
wherever it appears below.

**Evidence:** across all 17 arms with a recorded L-BFGS iterate log, 16
terminate on SciPy's `REL_REDUCTION_OF_F` test immediately after a
`more than 10 function and gradient evaluations in the last line search`
warning — i.e. every "convergence" but one is in fact a failed line search
misreported as success, because `ftol = 1e-12` sits below the float32
resolution of the loss (~1e-10 at these values) and can only be satisfied by a
line search that changes the loss by exactly zero. The exception,
`05_dense_reference`, is the one arm that reached its 40,000-evaluation cap
instead.

Measuring the gradient at the moment each run quit, relative to its own
preceding level: runs stopping before 12,000 iterations exited at a median
4.3x elevation (range 0.9-26.7x) — cut off mid transient spike; runs
surviving past 20,000 exited at median 1.0x (0.6-2.1x) — a quiet stop at a
genuine plateau. Spearman rho between stopping iteration and exit-gradient
elevation is −0.73 (p = 8e-4, n = 17). A controlled repeat of arm 01 under identical code, flags and seed
(`baseline_physics_only_K3_matched`, `notebooks/matched_effort/`) was
bit-identical to the original for 99 iterations, then diverged through
float32 GPU non-determinism: its first (cold-start) L-BFGS call died at
2,727 evaluations, at loss 1.177e-3, exit-gradient elevation 26.7x — the
same early/violent regime as the original's single call, which died at
5,503 evaluations, loss 2.97e-4, exit-gradient elevation 6.5x. Because the
repeat's cycling driver (see "Attempt log" below) kept alternating short
Adam bursts with fresh L-BFGS calls after that first death, it accumulated
7,173 evaluations in total and *more* effort than the original, not less —
but each subsequent L-BFGS call also died almost immediately (median 139
evaluations per cycle over cycles 2-33), so the extra evaluations bought
almost no extra descent: final loss 5.25e-4, 1.77x **worse** than the
original's 2.97e-4. Both attempts give the same wake result (far-core $v_1$
rel_L2 0.997 vs 0.997, amp_ratio 0.023 vs 0.019) — so the pathology affects
training effort, not the physical conclusion.

**Interpretation:** effective L-BFGS-B effort in this project is a random
variable, not a controlled quantity, because the failure time is seeded by
float32 GPU rounding. Warm-starting or restart schemes cannot fix it — a
restart from unchanged weights is deterministic (same gradient, same first
step, same line-search outcome), which is why 120 consecutive restarts on the
first repeat attempt returned the identical loss to sixteen digits every
time. The only lever that helped, tried and rejected, was re-rolling the cold
start.

**Limitation:** the uniform-vs-wake-biased pattern (2,727-5,503 evaluations
for uniform sampling at 32 taps against 37,713-43,676 for the two wake-biased
variants) is suggestive with n=2 per side, not established; see A03's
caveat. This section does not claim collocation placement causes the
termination difference, only that it correlates in the arms observed so far.

**Figure/result:** full numbers in
`../notebooks/matched_effort/01b_matched_effort_outcome.md` and
`../notebooks/matched_effort/01b_matched_effort_comparison.csv`, both rebuilt
2026-08-28 directly from the two runs' `run_record.json`,
`training_loss_summary.json` and `arm_summary.json`.

**The census is reproducible, and it reproduces exactly.** The scripts were
recovered on 2026-08-28 and are saved as `scripts/termination_census/`
(`06_all_arms_death_census.py` is the census itself). Re-run against the 16
arm logs plus the re-run, they return the quoted values to the digit:
Spearman rho = -0.733, p = 8.19e-04, n = 17; runs stopping before 12,000
iterations exit at median 4.3x (range 0.9-26.7x, n = 9), runs surviving past
20,000 at median 1.0x (0.6-2.1x, n = 8); the line-search warning is present at
termination in 16 of 17 runs, the exception being `05_dense_reference`, which
hit its evaluation cap. The two figures now exist:
`figures/final/F_termination_anatomy.png` and
`figures/final/F_arm1_rerun_breakdown.png`.

One caveat carried over from the recovery: the terminal blow-up ratio depends
mildly on how the baseline window is sliced. The census (and both figures, and
every number quoted here) uses an array-index window, giving 6.5x for arm 01
and 26.7x for the re-run. `05_two_run_divergence.py` slices by iteration
number instead and prints 5.7x and 25.5x for the same two runs; its output
carries a note saying so. The conclusion does not turn on the choice --
rho = -0.733 against -0.721 across the 17 runs -- but only the census
definition should be quoted.

The wall-clock comparison is worth stating alongside the effort one: the two
runs consumed the **same wall clock to within 0.04%** (32,433.5 s against 32,420.1 s). The
re-run turned an identical budget into 1.30x the evaluations and a 1.77x worse
loss, which is a sharper statement of the pathology than "more effort, worse
result".

### Attempt log — the matched-effort re-run, in order

This is the chronological record of what was tried between the Colab launch
and the decision to discard the re-run, kept so the reasoning is not repeated
by accident.

1. **Launch attempt 1 (restart-based driver).** The first matched-effort
   notebook re-declared a fresh L-BFGS call every time SciPy returned, up to
   a 35,000-evaluation target, guarded against dead restarts by checking how
   many evaluations a restart consumed. Live Colab output showed this guard
   never firing: four consecutive restarts at the very start of the pasted
   log sat at identical loss `1.17682e-03`, each "converging" after only 25
   function evaluations, and a later checkpoint ("restart 203") reported
   7,777 evaluations spent after 1.8 h with the most recent restart still
   gaining only 25. A second, fuller paste confirmed it: **120 consecutive
   restarts, each gaining exactly 25 evaluations, at the identical loss
   `0.0011768235126510262` to sixteen significant digits.**

2. **Root cause 1 — the guard measured the wrong thing.** It checked
   "did this restart gain close to zero evaluations", but each restart was
   burning a normal-looking 25 evaluations on a line search that was doomed
   from the start, so the check never tripped and the notebook would have
   restarted for the full 8 h budget without making progress.

3. **Fix v2 — loss-based stall guard.** Rewritten to check actual loss
   improvement rather than evaluations consumed, and unit-tested against
   mocked optimizer behaviour before being trusted.

4. **v2 rejected before running, by re-reading the same pasted log.**
   Arithmetic on the log showed the stall began right after the *first*
   optimizer call ended (~evaluation 2,702). A stop-on-stall guard placed
   there would have ended training at roughly 2,777 evaluations — **fewer**
   than the 5,503 of the original arm 01 it was meant to replace. Restarting
   can never fix this kind of failure: a cold restart from unchanged weights
   is deterministic (same gradient, same first step, same line-search
   outcome), which is exactly why all 120 restarts in step 1 returned
   identical results. The weights have to actually move.

5. **Fix v3 — revert the optimizer, alternate short Adam bursts instead of
   restarting.** `NN_functions.py` was reverted to the byte-identical,
   already-proven historical file (sha256 `d31374885d4458f1...`, the file
   that trained arms 8, 16, 32 and 15); the notebook's own verification cell
   asserts this hash and refuses to start otherwise. Matched-effort logic
   moved into the main script as a cycle loop: one plain L-BFGS call, then a
   short Adam "kick" (100 iterations, escalating to 400 then 1,600 if a
   cycle buys nothing), with every cycle checkpointed to disk and hard caps
   on cycle count and wall clock so a disconnect could not lose the run. The
   shipped loop text was executed verbatim against six mocked failure
   scenarios (the observed stall, escalation, wall-clock cap, cycle cap, and
   a lucky long first call) before being trusted; one initial test failure
   was traced to an off-by-one in the test harness itself, not the notebook.

6. **Warm-starting considered and dropped.** `--RestoreModel` (loading arm
   01's finished weights as the new run's starting point) was proposed as a
   lower-risk alternative, then found to have **never been exercised in this
   project** — all 16 arms record `warm_started: false` — so it was
   downgraded to an untested code path needing a smoke test first, and
   dropped once step 8 below made clear it would start deeper into the
   float32 resolution floor than the state that was already dying early.

7. **The v3 run executed overnight and completed cleanly.**
   `baseline_physics_only_K3_matched`: 33 cycles, 32 Adam kicks, correct exit
   on `wall_clock_cap_reached` after 32,420 s (~9.0 h). The machinery worked
   exactly as designed.

8. **Outcome: worse than the arm it was meant to replace.** 7,173 L-BFGS
   evaluations against a 35,000 target (cycle 1 alone gained 2,727; cycles
   2-33 averaged 139 each). Final total loss 5.25e-4 against the original's
   2.97e-4 (1.77x worse); pressure-tap loss 2.15e-5 against 1.14e-5 (1.89x
   worse); best loss reached at any checkpoint, 5.01e-4, still 1.69x worse.
   The final Adam phase moved backwards (5.386e-4 -> 6.238e-4). Wake metrics
   were essentially unchanged from the original (see item 9 above under
   "Evidence"), so the physical conclusion was unaffected — only training
   effort was lost. **Decision: keep `01_baseline_physics_only`, discard the
   re-run except as evidence, do not attempt a third time.**

9. **Post-hoc mechanism analysis (this findings section, "Evidence" above)**
   traced the failure to the ftol/float32 pathology and the 17-arm
   gradient-blow-up census, and ruled out tap count as the cause: two other
   32-tap arms differing from arm 01 only in collocation placement
   (`arm_06_wake_biased_random`, `07_wake_biased_grid`) ran 43,676 and
   37,713 evaluations — 7-8x arm 01's mileage at the identical tap count —
   which is direct evidence against "32 taps breaks the optimizer."

**Tooling produced along the way, not needed in the end but kept:**
`scripts/resweep_after_matched_arm1.py` — a `--check`/`--swap`/`--run`/
`--restore` driver that would have repointed A01/A02/A03/A04 at new arm-01
weights via a filesystem symlink swap and re-run their evaluation scripts,
unit-tested against a scratch checkpoint tree. Not exercised against real
data because the decision in item 8 was to keep the original checkpoint.


---

## A04 — prior attribution

The central question: does the trained network improve the Kármán prior or
just reproduce it? The labels below are descriptive rather than internal arm
numbers, so they can be reused in figures and dissertation text.

### The main result: a division of labour along the wake

v1 relative L2 (lower better; 1.0 = no better than predicting zero):

| region | pressure-only + physics | Kármán prior alone | pressure + physics + Kármán prior | network's effect |
|---|---|---|---|---|
| near-cylinder | 0.911 | 1.372 | 0.513 | +62.6% |
| near-wake | 1.003 | 0.696 | 0.441 | +36.7% |
| far-wake | 0.998 | 0.285 | 0.290 | −1.6% |
| far-core | 0.997 | 0.282 | 0.285 | −1.1% |

Two things to say in the writing:

1. **In the far wake the network does not measurably improve the prior for this
   configuration and metric.** The far-core change is −1.1% (far-wake −1.6%),
   so the hybrid is marginally worse than the analytical prior alone. Because
   this is a single seed, it supports “no improvement” rather than a claim that
   the prior-assisted network is reliably worse. The good far-wake structure is
   therefore supplied primarily by the prior, not learned from the sparse data.
2. **Near the body the network contributes strongly** (+62.6% near-cylinder,
   +36.7% near-wake). This is consistent with the 32 wall-pressure taps being
   directly informative near the cylinder, while the far-field street prior is
   least accurate there (near-cylinder v1 error 1.372, worse than predicting
   zero). The comparison does not prove that the taps alone cause the gain;
   physics and the learned network are part of the same trained method.

The pressure-only + physics method confirms the collapse rather than merely underperforming:
far-core amplitude ratio 0.019, correlation 0.160, rel L2 0.997. Note near-wake
at 1.003, i.e. *worse* than zero — consistent with the earlier wavenumber
finding that the failure is wrong phase structure, not only missing amplitude.

### Upstream region: a ratio artefact, not a defect

Headline number is amplitude ratio 11.5 in `other` (x < 0), which invites
"11x too much oscillation upstream". In absolute terms:

- true upstream |v1| RMS/node: 0.0056 (physically near zero — correct, there is
  no shedding upstream of a cylinder)
- hybrid predicted: 0.0647
- for scale, hybrid far-core signal: 0.1736

So the leakage is ~37% of the far-core signal magnitude, not 11x anything. The
11.5 is a near-zero denominator.

Decision was to report as a caveat, not fix (see `decisions.md`): the trust
region deliberately starts at x = 3.0, so upstream is unconstrained by design,
and widening it would suppress amplitude without adding information.

**Consequence for the writing:** do not lead with whole-domain numbers. `other`
is 12.2% of nodes with an 11x ratio, so it distorts any naive domain-wide
average. Report regionally.

### Pressure: not just a gauge offset

Far-field pressure rel L2 exceeds 1.0 for both methods. Removing a best-fit
constant offset (per region, per method — the sensitivity metric `data_contract.md`
permits as secondary):

| region | pressure-only + physics: raw → corrected | pressure + physics + Kármán prior: raw → corrected |
|---|---|---|
| far-wake | 2.333 → 0.806 | 1.565 → 1.429 |
| far-core | 1.901 → 0.724 | 1.277 → 1.155 |

The pressure-only + physics method drops below 1.0 in both regions; the
prior-assisted method stays above 1.0 in both. So the offset explains most of
the pressure-only method's apparent deficit but little of the prior-assisted
method's — a genuine far-field pressure error survives correction. Report the
raw value as primary (contract) with this as the stated sensitivity.

### Loose ends from A04

- The pressure-only + physics arm above (`01_baseline_physics_only`) is a
  non-converged lower bound — see "Training effort — the L-BFGS termination
  pathology" above. Its numbers can only get better with more training, so
  the network's measured contribution (+62.6% near-cylinder, +36.7%
  near-wake) is a floor, not a ceiling; the true division of labour may
  favour the network somewhat more than shown.
- The upstream leakage suggests an upstream damping condition or a broader
  trust region as future work. **[unverified]** — never tested in any arm.
- Why the prior-assisted method's pressure error is *less* offset-explained
  than the pressure-only method's is not understood. **[unverified]** —
  plausibly the prior's Bernoulli pressure anchor fixes the level, but this has
  not been checked.
- Everything here is single-seed (Seed 0, n = 1 per configuration), so
  differences of a few percent cannot be separated from initialisation
  scatter. This applies to the −1.1% far-core result: it supports "no
  improvement", not "reliably slightly worse".

### Cross-reference with A01

A01 reaches the far wake by adding velocity probes; A04 reaches it by adding
the analytical prior. The prior gets further (far-core $v_1$ 0.285 versus
0.373) and needs no extra sensors, but loses badly near the cylinder where its
assumed form does not hold. See the comparison table in the A01 section — the
two analyses should be written up as one argument about where wake information
can come from, not as two separate sensitivity studies.

### Figures carrying these claims

- `F01_prior_attribution_fields.png` — representative snapshot fields
- `F02_prior_attribution.png` — the division of labour (the crossover)
- `F02b_upstream_artefact.png` — the upstream ratio artefact, both framings

---

## A01 — information comparison

**Question:** how much does additional measurement information improve the
reconstruction? The controlled comparison is pressure-only + physics versus
pressure + velocity probes + physics. Dense observations are shown only as a
representational ceiling: that run also uses a different optimizer budget and
is not a controlled sensor-information comparison.

### Finding: velocity probes close the wake-reconstruction gap

**Evidence:** whole-domain relative $L^2$ errors are $(u,v,p)=(0.353,0.804,0.562)$
for pressure-only + physics and $(0.047,0.202,0.128)$ after adding velocity
probes. For the first shedding harmonic $v_1$, the regional relative $L^2$
errors for pressure-only versus pressure + velocity probes are:

| region | pressure-only + physics | pressure + velocity probes + physics |
|---|---:|---:|
| near-cylinder | 0.911 | 0.360 |
| near-wake | 1.003 | 0.295 |
| far-core | 0.997 | 0.373 |

**Interpretation:** the velocity probes provide information that pressure-only
training cannot infer reliably, especially for the oscillatory wake field. The
near-wake harmonic correlation rises from 0.082 to 0.999 and the amplitude
ratio rises from 0.054 to 0.992, showing that the improvement is structural,
not merely a reduction in overall magnitude.

### The dense arm did not converge — it is a lower bound on the ceiling

The dense run stopped on its **function-evaluation** cap, not on its own
convergence test. Its log ends with
`STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT` at Tnf = 40,001 against
`--LBFGSMaxfun 40000`. Note it was the evaluation budget that bound, not the
iteration budget: Tit = 37,924, still under `--LBFGSMaxit 40000`. It was also
launched with `--SkipAdam`, so the reported field is the L-BFGS endpoint at a
fixed budget.

This is worth stating positively rather than as a caveat: a non-converged dense
run is a **lower bound** on what full-field observation could achieve, so every
comparison against it is conservative. The gap between probes and dense would
only widen with more budget, never narrow.

### The ceiling is closer than expected: 40 probes get most of the way

On far-core $v_1$ relative $L^2$, dense observations reach 0.292 against 0.373
for 40 velocity probes — only about 22% better, despite dense having the entire
field rather than 40 point measurements. Since the dense number is a
non-converged lower bound, the true gap is somewhat larger, but the order of
magnitude holds.

**Why this matters for the writing:** it is a design result, not just a
sensitivity check. Sparse velocity probing recovers most of the reconstruction
quality of full-field observation, which is the practically relevant statement
for anyone instrumenting a real experiment.

### Cross-reference with A04: the prior beats the probes in the far wake

A01 and A04 answer the same question — where does far-wake structure come
from? — by two different routes, and they should be read together:

All three rows are trained networks on 32 pressure taps, differing only in what
is added:

| route to far-core $v_1$ | rel $L^2$ | extra sensors needed |
|---|---:|---|
| nothing added (pressure + physics) | 0.997 | — (fails) |
| + 40 velocity probes (A01) | 0.373 | 40 probes |
| + analytical Kármán prior (A04) | 0.285 | none |

(For reference, the prior with *no* network scores 0.282 in this region — see
the A04 section. The 0.285 above is the trained prior-assisted arm, which is
the like-for-like comparison against the probe arm.)

**The analytical prior outperforms 40 velocity probes in the far wake, with no
additional instrumentation.** Both are ways of injecting the wake structure that
wall pressure cannot supply; the prior injects it as an assumption, the probes as
data. Stated together this is a stronger claim than either section makes alone.

The honest counterweight, which belongs in the same paragraph: the prior only
wins where its assumed form is correct. In near-cylinder it scores 1.372, worse
than predicting zero, where the probes reach 0.360. So this is not "priors beat
sensors" — it is that each supplies information in the region where the other
cannot.

**Limitation:** this is one seed per method. The dense-observation curve is a
non-converged ceiling (see above), not a fair third controlled arm, because its
optimizer/training budget differs from the other two.

**Figure/result:** `figures/final/F03_information_comparison.png`, generated by
`scripts/figures/fig03_information_comparison.py`; numerical source:
`derived/a01_information_comparison_metrics.json` and
`derived/a01_information_comparison_summary.csv`.

---

## A02 — pressure-tap count

**Question:** does increasing the number of cylinder pressure taps improve the
reconstruction? This is a controlled comparison of pressure-only + physics
models with 8, 16, and 32 uniformly spaced taps. All three use the same
network and training settings; only the tap count changes as an input setting.

### Finding: more taps improve selected local fields, not the unobserved wake

**Evidence:** near-cylinder relative $L^2$ errors for $(u,v,p)$ are, for 8,
16, and 32 taps respectively:

| quantity | 8 taps | 16 taps | 32 taps |
|---|---:|---:|---:|
| $u$ | 0.069 | 0.053 | 0.043 |
| $v$ | 0.143 | 0.114 | 0.141 |
| $p$ | 0.085 | 0.019 | 0.019 |

For the first shedding harmonic $v_1$, the near-cylinder errors are 0.771,
0.709, and 0.911, while the near-wake errors are 1.044, 1.060, and 1.003.
The far-core errors are 0.992, 0.992, and 0.997. The zero-prediction baseline
is 1.0 for this relative-$L^2$ metric.

**Interpretation:** additional wall pressure information improves the local
mean/pressure fields, especially from 8 to 16 taps, but it does not provide
the missing downstream information needed to reconstruct the oscillatory wake.
The $v_1$ trend is non-monotonic: 16 taps is best near the cylinder, whereas 32
taps is worse on this harmonic metric. This does not mean 32 taps worsens every
field variable; it improves $u$ and $p$ locally.

**Limitation:** each tap count has one seed, so the non-monotonic harmonic
ordering should not be interpreted as evidence that 32 taps are intrinsically
harmful. The result establishes that more wall-pressure taps alone are not a
reliable substitute for wake-sensitive information such as velocity probes or
an appropriate prior.

**Limitation (training):** the 32-tap arm is the same non-converged
`01_baseline_physics_only` checkpoint discussed under "Training effort — the
L-BFGS termination pathology" (5,503 evaluations against 27,130 for 8 taps
and 21,868 for 16 taps). The $u$/$p$ improvements above are unaffected —
an under-trained arm can only make 32 taps look worse, so those hold as
lower bounds. The $v_1$ non-monotonicity just above is the one claim that
does depend on 32 taps being fairly trained, and should not be read as a
tap-count effect for that reason.

**Figure/result:** `figures/final/F04a_tap_count.png`, generated by
`scripts/figures/fig04a_tap_count.py`; numerical sources:
`derived/a02_tap_count_metrics.json` and
`derived/a02_tap_count_summary.csv`.

---

## A03 — collocation strategy

**Question:** does wake-biased collocation recover information absent from the
wall measurements? Three pressure-only + physics arms at 32 taps, differing in
where the interior collocation points are placed: uniform
(`01_baseline_physics_only`), wake-biased random (`arm_06_wake_biased_random`),
and wake-biased regular grid (`07_wake_biased_grid`). All three checkpoints
exist and the manifest reports `verified_inputs`
(`derived/a03_input_manifest.json`).

Read this arm together with A05, which asks the same sampling question in the
*prior-assisted* setting. A03 is the network-only version of it.

### Finding: wake-biased sampling moves the error, it does not remove it

**Evidence:** first-harmonic $v_1$ relative $L^2$ error by region (lower
better; 1.0 = no better than predicting zero):

| region | uniform | wake-biased random | wake-biased grid |
|---|---:|---:|---:|
| near cylinder | 0.9110 | 0.5039 | 0.5064 |
| near wake | 1.0028 | 1.1559 | 1.1325 |
| far wake | 0.9976 | 1.0312 | 1.0175 |
| far core | 0.9974 | 1.0156 | 1.0062 |

Near the body the wake-biased arms roughly halve the $v_1$ error (0.911 →
0.504), and the supporting diagnostics agree that this is a real local gain:
near-cylinder amplitude ratio rises 0.233 → 0.711 and complex correlation
0.590 → 0.935. Downstream every wake-biased number is *worse* than uniform,
and all six downstream values sit at or above 1.0.

The mean-field errors move the other way, mildly: far-core $u$ improves
0.583 → 0.479 and whole-domain $u$ 0.353 → 0.304, monotonically in the order
uniform → random → grid.

**Interpretation:** the two wake-biased variants are near-identical (grid
marginally ahead of random everywhere), so this is an effect of *biasing*
the sampling toward the wake, not of the particular sampling pattern. Biasing
buys a substantial near-body first-harmonic gain and a small mean-field gain,
and pays for both downstream. Nothing here recovers the wake: the shedding
mode remains at or worse than the zero-prediction baseline in every region
beyond the cylinder, in every arm.

### Finding: downstream, the wake-biased arms buy amplitude and lose phase

**Evidence:** far-core $v_1$ diagnostics, which decompose the rel $L^2$ above:

| quantity | uniform | wake-biased random | wake-biased grid |
|---|---:|---:|---:|
| amplitude ratio (ideal 1) | 0.0189 | 0.2975 | 0.2540 |
| complex correlation (ideal 1) | 0.1604 | 0.0963 | 0.1026 |
| rel $L^2$ (ideal 0) | 0.9974 | 1.0156 | 1.0062 |

**Interpretation:** wake-biased collocation raises the far-core shedding
amplitude by roughly 16x (0.019 → 0.298) while the correlation with the true
mode *falls* by about 40% (0.160 → 0.096). The reconstruction becomes more
visibly wake-like and less correct at the same time, which is why the error
metric worsens even as the amplitude deficit closes. This is the same
wrong-phase-structure pathology recorded in A04 for the pressure-only + physics
method — near-wake error above 1.0 driven by phase rather than missing
amplitude — and A03 shows that adding interior points where the wake is does
not fix it, it feeds it.

This matters for how the figures are read. An eyeball comparison of
reconstructed wake fields would rank the wake-biased arms *higher*; the metric
ranks them lower, and the metric is right. Worth one sentence in the
Discussion.

### Limitation: effort is not controlled, and cannot be

**Evidence:** L-BFGS evaluations by arm — uniform 5,503, wake-biased random
43,676, wake-biased grid 37,713: a 7.9x and 6.9x gap. Recorded per arm in
`results_master.csv` as `training_effort.lbfgs_evals`
(`value_type: confound_audit`).

The arms differ in exactly one input flag, so the comparison is controlled as
an *input*; it is not controlled as an *effort*. This is not a fixable
oversight. The uniform arm is `01_baseline_physics_only`, whose early
termination is the characterised `ftol`/float32 pathology documented under
"Training effort" above, and the matched-effort re-run mounted against it
(2026-08-28) ended with *more* evaluations and a *worse* loss. There is no
version of this comparison at matched effort available from the existing arm
set, and no reason to expect a third attempt to produce one.

**What this costs and what it does not.** The gap favours the wake-biased
arms, which splits the findings cleanly:

- **Conservative, and safe to report.** Every result in which the wake-biased
  arms are *worse* — all six downstream $v_1$ values, the near-wake and
  far-wake field $v$ errors, the upstream leakage below — survives the
  confound, because a wake-biased arm would have had to lose *in spite of*
  7-8x the optimizer mileage. The headline negative claim is therefore sound.
- **Unattributable, and must be hedged.** The near-body $v_1$ gain (0.911 →
  0.504) and the mean-field $u$ gains cannot be assigned to sampling rather
  than to effort. Report them as observed differences between arms, never as
  "wake-biased collocation improves the near-body reconstruction".

Note the symmetry with A05: there the *losing* arm was the less-trained one,
and here the losing arms are the better-trained ones. Both point the same way
— no evidence that wake-biased collocation helps — and for opposite reasons,
which is worth a line in the Discussion because two independent confound
directions agreeing is stronger than either arm alone.

Secondary limitations: single seed per arm, as everywhere in this project;
and the two wake-biased arms are not independent evidence of each other, being
the same idea implemented two ways.

### Upstream leakage: larger than A04's, and not only a ratio artefact

**Evidence:** in `other` (upstream/off-axis), $v_1$ rel $L^2$ goes 1.043 →
7.955 / 7.964 and amplitude ratio 1.215 → 8.008 / 8.004.

The ratio inherits the near-zero-denominator caveat established for A04's
11.5x figure — true upstream $|v_1|$ is physically almost zero, so any ratio
against it inflates. But unlike A04, the *field* errors move too: upstream
$v$ rel $L^2$ 0.122 → 0.322 and upstream $p$ 0.197 → 0.434, both roughly
2.6x and both computed against non-degenerate denominators. **[unverified]**
The likely reason is that these arms carry no `--V1RadialTrust` blend to
suppress upstream oscillation, so concentrating interior points downstream
leaves the upstream region both unconstrained and unpenalised; confirming that
would need an absolute-magnitude check of the kind run for A04
(`derived/a04_v1_absolute_check.json`), which has not been done for A03.

Report as a caveat on whole-domain and `other` summaries, consistent with the
A04 decision.

**Figure/result:** `figures/final/F04b_collocation_strategy.png`, generated by
`scripts/figures/fig04b_collocation_strategy.py`; numerical sources
`derived/a03_collocation_metrics.json` and
`derived/a03_collocation_summary.csv`; validation `derived/a03_validation.json`
(`passed`); rows registered by `scripts/a03_finalize_results.py`.

---

## A05 — prior plus collocation

**Question:** does wake-biased collocation improve the prior-assisted
reconstruction? Arm 15 (prior, uniform interior sampling) against Arm 10
(prior, wake-biased regular grid). A symmetric diff of the recorded command
lines reports a single difference, `--WakeBiasedGridSampling`, and the two
checkpoint-local `NN_functions.py` files hash identically
(`derived/a05_input_manifest.json`, status `verified_inputs`).

### Finding: wake-biased grid sampling does not help, and downstream it hurts

**Evidence:** first-harmonic $v_1$ relative $L^2$ error by region, with the
analytical prior evaluated alone as the reference every prior-assisted arm must
beat to have contributed anything:

| region | prior alone | uniform collocation | wake-biased grid |
|---|---:|---:|---:|
| near cylinder | 1.3722 | 0.3986 | 0.4596 |
| near wake | 0.6958 | 0.3838 | 0.6059 |
| far core | 0.2819 | 0.2895 | 0.5047 |
| far wake | 0.2850 | 0.2922 | 0.5137 |

Expressed as the learned contribution (prior error minus arm error, positive
meaning the network improved on the analytical field): uniform collocation
gains $+0.974$ near the cylinder, $+0.312$ in the near wake, and $-0.008$ in
the far core. Wake-biased grid gains $+0.913$, $+0.090$, and $-0.223$.

Near-cylinder field errors are essentially unchanged between the two arms
($u$ 0.059 vs 0.054, $v$ 0.065 vs 0.077, $p$ 0.017 vs 0.016), so the
difference is confined to the wake.

**Interpretation:** concentrating collocation points in the wake did not
recover wake information. It left the near-body fit intact, cost most of the
learned near-wake contribution ($+0.312 \to +0.090$), and drove the far field
*below* the analytical prior it was blended with ($-0.223$). The mechanism is
visible in the third row of the table: with the $v_1$ radial trust active, the
far field of a well-behaved prior-assisted arm sits on the prior level, and
the wake-biased arm departs from it in the wrong direction.

**Limitation:** the arms are not effort-matched — 34,643 L-BFGS evaluations
(uniform) against 26,129 (wake-biased), a 1.33x spread, the closest match in
the arm set but not a controlled budget. The losing arm is also the
less-trained one, so the *magnitude* of the harm is an upper bound on the
sampling effect and a lower bound on the arm's achievable quality. What is not
at risk is the negative direction of the claim: there is no evidence that
wake-biased grid sampling improves the prior-assisted reconstruction, since a
gain would have had to appear in spite of the effort deficit, and none did.

**Figure/result:** `figures/final/F04c_prior_collocation.png`, generated by
`scripts/figures/fig04c_prior_collocation.py`; numerical sources:
`derived/a05_prior_collocation_metrics.json`,
`derived/a05_prior_collocation_summary.csv`, and
`derived/a04_prior_only_metrics.json` for the prior-alone column.

---

## A06 — pressure-noise robustness

**Scope limit, stated first because it bounds every claim below:** all four
arms train with `--V1RadialTrust --StreetPrior`, and no physics-only noise arm
exists. A06 therefore describes the noise tolerance of the prior-assisted
hybrid only, and says nothing about how the pressure-only configuration of A01
or A02 responds to tap noise.

**Question:** is the prior-assisted result robust to pressure noise? Arm 15
(clean taps) against Arms 11, 12 and 13 at `--Noise` 4.7265e-04, 2.3633e-03
and 4.7265e-03, i.e. the 1 %, 5 % and 10 % levels, verified to be exact 1x/5x/10x
multiples of the same base level. Evaluation is always against clean CFD.

The input manifest reports `input_mismatch` rather than `verified_inputs`:
besides `--Noise`, Arms 11 and 13 carry `--LBFGSCheckpointIters` which Arms 15
and 12 do not. That flag drives a read-only accepted-step callback that copies
the packed parameter vector and writes five `.npy` files; it does not enter the
objective, the data, or the search direction, and is recorded rather than
assumed away (`derived/a06_input_manifest.json`).

### Finding: far-field invariance is the prior's, not the network's

**Evidence:** the four arms' far-core $v_1$ errors are 0.2895, 0.3249, 0.2776
and 0.2765 for 0 %, 1 %, 5 % and 10 % noise. The analytical prior alone scores
0.2819 in the same region. Every arm is within 0.043 of the prior level, and
the two highest-noise arms are *closer* to the truth than the clean arm is.

**Interpretation:** downstream of the trust radius the $v_1$ radial blend pins
the field to the analytical Kármán street, so tap noise cannot degrade it — and
equally, agreement there is not evidence that the reconstruction tolerates
noise. The point is sharper than the blend alone: `street_prior_used.npz` is
byte-identical across all four arms (sha256 `18e905159e09af39`), i.e. the prior
parameters ($\Gamma = 2.527$, $U_c = 0.821$, $\omega = 1.036$, phase 0.731,
amp_scale 0.743) were fitted beforehand on clean data and passed in unchanged,
while `--Noise` perturbs only the tap-measurement term. Far downstream the
arms therefore share a fixed analytical field by construction, and their
agreement there is close to tautological. Far-field $v_1$ must not be quoted as a robustness result for any
prior-assisted arm. This applies to A04 and A05 as well: no conclusion in those
sections should rest on far-field $v_1$.

### Finding: noise erodes the learned near-body contribution

**Evidence:** the learned contribution in the near wake (prior error minus arm
error) falls monotonically with noise: $+0.312$, $+0.247$, $+0.241$, $+0.194$.
Near-cylinder $v_1$ error rises from 0.3986 (clean) to 0.6999, 0.6545 and
0.7896, and near-cylinder field errors follow: $v$ 0.065 (clean) against 0.109,
0.098 and 0.133; $u$ 0.059 against 0.074, 0.063 and 0.094.

**Interpretation:** noise degrades precisely the part of the field the network
is responsible for, and leaves untouched the part the prior dictates. The
near-wake ordering is monotonic in noise amplitude; the near-cylinder ordering
is not (the 5 % arm is better than the 1 % arm on every near-body quantity),
so the amplitude dependence is a direction rather than a dose-response curve.

**Limitation:** one seed per level, and effort is not controlled — 34,643
evaluations for the clean arm against 12,047, 5,376 and 3,156 for 5 %, 1 % and
10 %, an 11x spread, all four stopped by SciPy's ftol test on float32 rounding
noise rather than at convergence. The noisy arms all trained less, which biases
the comparison toward "noise hurts", so no magnitude should be read off these
numbers. One internal check argues the near-wake trend is not purely
truncation: the 5 % arm has 2.24x more optimizer effort than the 1 % arm and
still contributes slightly less ($+0.2405$ against $+0.2470$). A quantitative
noise sensitivity would need Arms 11--13 re-run at matched effort
(`notebooks/matched_effort/`); the qualitative claim above does not.

**Figure/result:** `figures/final/F04d_pressure_noise.png`, generated by
`scripts/figures/fig04d_pressure_noise.py`; numerical sources:
`derived/a06_pressure_noise_metrics.json`,
`derived/a06_pressure_noise_summary.csv`, and
`derived/a04_prior_only_metrics.json` for the prior-alone reference.

### Cross-cutting note added by A05 and A06

Both sections needed the prior-alone column to be interpretable, and both found
the far field prior-determined. The practical consequence for the remaining
write-up: for any arm trained with `--V1RadialTrust`, report the learned
contribution relative to the prior rather than the raw regional error, or the
prior's accuracy will be read as the network's.
