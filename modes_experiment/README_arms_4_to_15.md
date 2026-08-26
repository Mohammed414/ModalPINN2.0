# K = 3 series — arms 4 to 15

Twelve runs at the source paper's mode truncation (`--Nmodes 4`, k = 0,1,2,3), extending arms 1-3.
**Every arm changes exactly one thing from Arm 1.** Held fixed everywhere:

    --Tmax 9 --Nint 50000 --Nmes 5000 --multigrid --Ngrid 5 --NgridTurn 200
    --WidthLayer 25 --Nmodes 4 --Seed 0
    --FreestreamBC ON, --FluctuationInletBC OFF
    cold start, Adam kept

## The boundary formulation, and why it changed

The source paper uses **one** prior dictionary: cylinder no-slip, `f_BC = tanh[gamma(r - r_c)]`
with `h = 0`, its Eq. 18, read at source. It has **no inlet condition of any kind**. Both inlet
flags in this codebase are project additions.

| flag | evidence | decision |
|---|---|---|
| `--FreestreamBC` | E3 (off) 0.00779 against E3F (on) 0.01852, identical otherwise: **+137.7%** | **kept on** |
| `--FluctuationInletBC` | R2 (off) 0.15027 against R3 (on) 0.01718 on a BVF arm, identical otherwise, both converged: **-88.6%** | **dropped** |

Both act only upstream. The blending ramp is `0.5(1 - tanh(3(x + 2)))`: 0.9975 at x = -3, 0.0025
at x = -1, 6e-06 at the cylinder, 0.0 in the wake. Neither reaches the wake directly, which is
what makes the R2/R3 result notable — a constraint acting three diameters upstream cost 88.6% of
the far-core amplitude, by changing the optimisation path rather than the physics.

Arms 1, 4-9 already had the fluctuation damping off. Arms 10-13 were rebuilt with it off; arms 14
and 15 are new, supplying the fluct-off counterparts of arms 2 and 3. Each new arm asserts in-cell
that the flag is absent and that the log does not report it active, so a mis-edit fails loudly.

**Arms 2 and 3 are kept as they are.** They ran with the flag on and are the only arms in the
series that connect to R3 and R15, including the 0.80878 the report currently quotes. Arms 14 and
15 add the missing setting rather than replacing them.

## The arms

| # | notebook | sensors | remedy | collocation | tap noise | change from Arm 1 |
|---|---|---|---|---|---|---|
| 4 | `4_paper_sparse_probes` | 32 taps + 40 probes | none | uniform | 0 | drop `--PressureOnly` |
| 5 | `5_dense_reference` | dense, 5000 pts | none | uniform | 0 | drop `--PressureOnly --SparseData` |
| 6 | `6_wake_biased_random` | 32 taps | none | wake-biased random | 0 | `--WakeBiasedSampling` |
| 7 | `7_wake_biased_grid` | 32 taps | none | wake-biased grid | 0 | `--WakeBiasedGridSampling` |
| 8 | `8_taps_08` | 8 taps | none | uniform | 0 | `--NTaps 8` |
| 9 | `9_taps_16` | 16 taps | none | uniform | 0 | `--NTaps 16` |
| 10 | `10_prior_wake_biased_grid` | 32 taps | Kármán prior | wake-biased grid | 0 | prior + sampler |
| 11 | `11_prior_noise_01pct` | 32 taps | Kármán prior | uniform | 1% | prior + noise |
| 12 | `12_prior_noise_05pct` | 32 taps | Kármán prior | uniform | 5% | prior + noise |
| 13 | `13_prior_noise_10pct` | 32 taps | Kármán prior | uniform | 10% | prior + noise |
| 14 | `14_bvf_no_fluct_inlet` | 32 taps | wall vorticity flux | uniform | 0 | `--BVF` (KMAX=3 targets) |
| 15 | `15_prior_no_fluct_inlet` | 32 taps | Kármán prior | uniform | 0 | prior only |

Arm 15 is the reference for arms 10-13: same remedy, same boundary formulation, uniform sampling,
no noise. Arm 1 is the reference for arms 4-9 and 14.

## Reference values

Arms 1-3, run, `--Nmodes 4`, cold, `--Tmax 9`:

| arm | fluct inlet BC | far-core k=1 amplitude | phase corr |
|---|---|---|---|
| 1 baseline physics-only | off | 0.01895 | 0.16036 |
| 2 wall vorticity flux | on | 0.08945 | 0.10809 |
| 3 Kármán prior | on | 0.80878 | 0.97159 |
| analytical prior alone, no network | — | 0.8082 | — |

Older runs at k = 0,1,2, for the BVF comparison:

| run | fluct inlet BC | far-core k=1 amplitude |
|---|---|---|
| R2 | off | 0.15027 |
| R3 | on | 0.01718 |
| R4 | on | 0.00991 |

R2 is the highest BVF number on record, 1.68x Arm 2. Arm 14 tests whether it reproduces at the
paper's truncation.

## What each arm is for

**Arm 4** is the paper's own sparse configuration. Its Run 5 (Table 1) trained on 30 surface
pressure points plus 4 sections x 10 interior velocity probes at x = -3, 1, 2, 3, at N* = 4, and
reports a validation loss of 1.6e-3 with unsteady forces recovered to NRMSE 9.8e-4 and 6.1e-3.
That is a working unsteady reconstruction, so this arm is expected to revive the wake. It uses 32
taps rather than 30 and this project's collocation settings, so it is a test of the mechanism
rather than a byte-exact replication.

**Arm 5** measures the representability ceiling at the correct truncation. E1 reached 0.9946 but
at Nmodes 3 with 12,306 parameters per field against 21,608 here.

**Arms 6 and 7** fill a real gap. All three wake-focused runs in the project (R12w, R14, R15) had
the Kármán prior active; none tested focused collocation on a bare pressure-only baseline. The
identifiability finding predicts it cannot help, since collocation points enforce the PDE and do
not measure anything. Arm 7 against Arm 6 isolates equidistant spacing from random placement with
the four zones and their fractions held identical (30% whole domain, 35% formation x in [0,3], 25%
far wake x in [3,8], 10% cylinder annulus).

**Arms 8 and 9** give the network a tap-count curve. The Gappy POD sweep shows the linear baseline
improving with tap count to saturation at 10-12 taps (E_v 0.09861 at n=4, 0.00768 at n=10, 0.00637
at n=32); every ModalPINN run in the project used 32 taps, so no comparable curve exists.

**Arms 11-13** close an asymmetry: the Gappy POD baseline is tested at 0/1/5/10% noise, and no
ModalPINN run in the project has ever used noisy data. Levels are calibrated to the measured tap
pressure fluctuation RMS of 0.047265, the same normalisation the Gappy sweep uses.

## Caveats to record with the results

**The prior is derived from CLEAN taps in arms 11-13.** `street_prior.py` has no noise option, so
the Kármán prior is fitted to noiseless DNS tap pressures while training sees noisy ones. This
measures robustness of the training, not of the whole pipeline.

**Arm 5 is not the paper's dense run.** It trains on 5,000 scattered u,v,p points (`Nmes 5000`),
the same data configuration as E1, at `--Nmodes 4`. The paper's dense runs used 8,000-50,000
collocation points at different widths.

**`--Nmodes 4` also raises the parameter count** from 12,306 to 21,608 per field, since width is
`WidthLayer * Nmodes`. Arms differing from a k=0,1,2 reference differ in capacity as well as mode
count.

**`evaluate_physics_uniform.py` is skipped on prior-off arms** (4-9, 14). It applies the v1 trust
wrap with no opt-out, so on a prior-off checkpoint it would rebuild a function the network never
trained. Those arms report regional reconstruction errors instead.

## Known evaluator defect, and what it does and does not affect

`evaluate_v1_smoke.py` and `evaluate_physics_uniform.py` both **hardcode the inlet treatment on**:

    out_nn_modes_uv(..., freestream_target=0.0, damp_fluctuations=True, ...)

with no command-line flag to disable it. Every arm 4-15 trains with the fluctuation inlet damping
**off**, so both evaluators rebuild the velocity field with a treatment the network never trained
with. `evaluate_regions.py` is the only one that handles this correctly — it reads the flags and
otherwise infers them from the run-directory name.

**Which numbers are affected.** Both treatments act through the ramp `0.5(1 - tanh(3(x + 2)))`,
which is 1.00 at x = -4, 2.5e-03 at x = -1, 6.1e-06 at x = 0 and 9.4e-14 at x = 3. So:

| metric | region | affected? |
|---|---|---|
| `far-core` (x >= 3, \|y\| <= 2) | downstream | **no** — ramp is 9.4e-14 |
| `far-wake` (x >= 3) | downstream | **no** |
| `near-wake` (0 <= x < 3) | spans the ramp tail | **yes, partly** — quote with care |
| `near-cylinder` (r < 0.75) | includes x < -0.75 | **yes** |
| `whole-domain` | includes the inlet | **yes** |
| everything in `physics.json` | uniform sample over the domain | **yes** |
| everything in `regions.json` | — | **no**, that evaluator is correct |

The headline far-core amplitude and phase are sound. Do not quote `near-cylinder` or
`whole-domain` from `v1.json`, treat `near-wake` as approximate, and use `regions.json` rather than
`physics.json` for field errors. `force_error.py` sidesteps the defect entirely: the wall ring spans
\|x\| <= 0.55 where the ramp is <= 1.7e-04, and it rebuilds the field independently in numpy.

## Guards added before launch

Each notebook now fails loudly rather than producing a misleading result:

| guard | what it catches |
|---|---|
| `assert rc==0` after training | a nonzero training exit that still left a checkpoint |
| `assert 'Fluctuation Inlet BC : True' not in log` | the flag creeping back in |
| `assert 'Freestream BC : True' in log` | the freestream prior silently absent |
| `assert '--FluctuationInletBC' not in cmd` | a mis-edited command line |
| `EVAL_FAILURES` + `assert not EVAL_FAILURES` | an evaluator that failed, leaving an incomplete `arm_summary.json`. The summary is written **first**, so nothing is lost — the assertion only flags it |
| checkpoint filename vs `DNN2_100_100_4` | wrong mode count |

Prior-active arms (10-13, 15) now also copy `street_prior_used.npz` into the Drive result folder, so
the exact prior each run used is retained rather than living only in the ephemeral Colab working
directory.

**L-BFGS tolerance:** these arms use the codebase default `ftol = 1e-12`, not the `2.22e-16` the
older E3F/R2 runs set. Arms 1-3 also ran at 1e-12, so the series is internally matched, but it is
not the older runs' tolerance. All four completed arms converged to a flat loss tail using 11-49% of
the 50,000 iteration cap, so the difference had no effect in practice.
