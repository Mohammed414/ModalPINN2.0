# K = 3 experiment series

## Layout

| folder | role |
|---|---|
| `experiment/` | **what ran** — `notebooks/` (the 16 arm notebooks), `code/` (the exact source they used, including the `--KMAX` fix), `bvf_targets/` (the verification that the fix is behaviour-preserving) |
| `runs/` | **what came out** — `arms/`, one folder per arm: weights, checkpoints, `run_record.json`, `train_log.txt` |
| `fresh_analysis/` | **what it means** — the live analysis workspace. `results_master.csv` and `findings.md` are the numbers the dissertation quotes |
| `baselines/` | **what to compare against** — `prior_only_evaluation/` (the analytical Kármán prior alone, 0.8082) and `gappy_pod_final/` (the frozen Gappy POD diagnostic) |
| `figures/` | **evaluation machinery** — `modalpinn_eval.py` evaluates a trained arm exactly as it was trained; `dns_raw.npz` is the parsed DNS cache. `viva/build.py` imports both |

Loose files at this level: `arm_matrix.csv` (the 16 arms) and
`arms_master_results.csv` (their headline metrics). Those two are the current
tables — earlier partial summaries were archived, see below.

Dissertation-facing figures are **not** here — they are in `ModalPINN2.0/results/figures/`.

Three training runs at the **source paper's mode truncation**, using the **original ModalPINN
schedule unchanged**. Purpose: establish whether the pressure-only wake collapse depends on the
truncation, and how the two remedies behave once the truncation is correct.

## Why

Every run in the existing project trained **k = 0,1,2**. The flag `--Nmodes 3` counts the zero
frequency, so it means mean-plus-two-harmonics. The paper's `N = 3` truncation means
**k = 0,1,2,3** and requires `--Nmodes 4`. These three arms run the paper's truncation.

## The three arms

Run them in this order. Each is a separate 9-hour job.

| # | notebook | adds | reference point from the existing project |
|---|---|---|---|
| 1 | `1_baseline_physics_only_K3_T4.ipynb` | nothing — physics residual only | **E3F**, far-core k=1 amplitude 0.01852 |
| 2 | `2_boundary_vorticity_flux_K3_T4.ipynb` | `--BVF` Lighthill wall-flux loss | **R2**, far-core k=1 amplitude 0.15027 |
| 3 | `3_karman_prior_K3_T4.ipynb` | `--V1RadialTrust` tap-derived Kármán prior | see the caveat below |

Arm 1 is the one that answers the truncation question, so run it first.

**Caveat on arm 3.** R15 reached 0.83166 but is **not** a matched control: it was warm-started
from R10, used `Nint 25000`, `Ngrid 1/1000`, and skipped Adam. So compare arm 3 against the
analytical prior alone (0.8082) and against arm 1, not against R15.

## Settings — identical across all three except the added flag

    --Tmax 9 --Nint 50000 --Nmes 5000 --multigrid --Ngrid 5 --NgridTurn 200
    --WidthLayer 25 --Nmodes 4 --SparseData --PressureOnly --NTaps 32 --Seed 0
    --FreestreamBC

**All cold-started.** No `--RestoreModel` anywhere. Weights are initialised fresh, as E3F and R2
were.

**Adam kept.** L-BFGS runs to convergence (cap 50000, ftol 1e-12 — the codebase default; the older E3F/R2 runs set 2.22e-16, so this series stops on a looser tolerance, though all three arms converged to a flat loss tail well inside it), then Adam consumes the rest
of the 9-hour budget at lr 1e-5. This is the schedule E3F and R2 used. `--SkipAdam` appears in no
training command, and each notebook asserts `AdamTmax = Tmax-(t2-t0)` is present in the source
before training starts — so a source substitution that silently dropped the Adam phase fails the
check rather than quietly changing the experiment.

Arms 2 and 3 additionally pass `--FluctuationInletBC`, as their counterparts R2 and R15 did.

## The capacity caveat, stated once

`layers = [2, WidthLayer*Nmodes, WidthLayer*Nmodes, Nmodes]`, so the mode count also sets the
width:

| Nmodes | k trained | layers | parameters per field |
|---|---|---|---|
| 3 | 0,1,2 | [2, 75, 75, 3] | 12,306 |
| 4 | 0,1,2,3 | [2, 100, 100, 4] | 21,608 |

A factor of **1.76**. This is inherent to the paper's architecture, not a choice made here — but it
means arm 1 differs from E3F in *two* ways: the extra mode and the extra capacity. If arm 1 still
collapses, the truncation question is settled regardless. If it does not, a capacity-matched
follow-up (`--Nmodes 4 --WidthLayer 19`, 12,776 parameters) would be needed to say which factor
caused it.

## The one code change, verified

`experiment/code/bvf_targets.py` is a modified copy. The original hardcodes `KMAX = 2`, so the wall-flux
target `G` it builds contains only k = 0,1,2 harmonics. A 4-mode network has a k = 3 mode, so
training it against a k = 0..2 target would penalise that mode's wall flux toward zero. The copy
here exposes `--KMAX`.

**Regression test.** Called with `--KMAX 2` it reproduces the project's original
`bvf_targets_Ntap32_seed0.npz` to a maximum absolute difference of **1.1e-14** across all 17 shared
keys — float64 round-off. The only added key is `kmax` metadata. The change is provably
behaviour-preserving.

**What the k = 3 harmonic actually contributes** (`experiment/bvf_targets/bvf_kmax_verification.json`):

| harmonic | wall-flux RMS |
|---|---|
| k=0 | 1.184 |
| k=1 | 0.161 |
| k=2 | 0.0205 |
| k=3 | 0.0021 |

k = 3 carries **0.017%** of the oscillating wall-flux energy and changes the reassembled target by
**0.125% rms**. So the original k = 0..2 target was not materially wrong — the inconsistency is
real but small. The corrected target is used because it costs nothing and removes the objection.
One genuine gain: admitting the third harmonic improves the per-tap temporal fit R² from 0.99972
to 0.99998.

`ModalPINN2.0/src/pressure_only/bvf_targets.py` is **untouched**.

## The one deviation from the original invocation

`--SkipDiagnostics --ExitAfterSafetySave` are passed. These skip the legacy post-training plotting
**after** weights are saved; training itself is untouched. The evaluation cells replace those plots
with the regional metrics the dissertation quotes. Without this, plotting runs after a 9-hour
budget is already spent and risks losing the run to a Colab timeout.

## Prerequisites on Drive

    MyDrive/ModalPINN_data/fixed_cylinder_atRe100        the Re=100 DNS file
    MyDrive/ModalPINN_results/R10_v1radial_smoke/...     optional; the prior is derived from the
                                                        taps if this is absent

Results go to `MyDrive/ModalPINN_results/K3_series/<arm_tag>/`.

## What each arm saves

    training_run/        weights, checkpoints, manifest
    run_record.json      command line, wall time, L-BFGS eval count, whether Adam ran, cold-start flag
    arm_summary.json     the v1 / physics / regional metrics together
    v1.json              far-core k=1 amplitude ratio, relative L2, correlation
    regions.json         full u,v,p regional reconstruction error
    train_log.txt        complete stdout

The checkpoint filename encodes the layer shape — `DNN2_100_100_4_tanh.pickle` for Nmodes 4 — and
each notebook asserts it matches, so the truncation that trained is independently confirmed rather
than assumed.

## Project reference values

| quantity | far-core k=1 amplitude ratio |
|---|---|
| E3, collapsed physics-only (no freestream BC) | 0.00779 |
| E3F, baseline + freestream BC | 0.01852 |
| R2, BVF | 0.15027 |
| R15, prior-active (different schedule — not a matched control) | 0.83166 |
| analytical Kármán prior alone, no network | 0.8082 |

## One known limitation of the evaluation

`evaluate_physics_uniform.py` applies the v1 trust wrap unconditionally — it has no opt-out. For
arms 1 and 2 (prior off) that evaluator is skipped rather than run with a wrap the network never
trained with, which would rebuild a different function and report a meaningless residual. The
regional metrics from `evaluate_regions.py` are the comparable numbers for those arms. Patching
that evaluator is a separate change and was not made here.

## Archived out of this folder

| what | where | why |
|---|---|---|
| `runs/INDEX.md`, `runs/arms_1_to_16_results.csv`, `runs/ARMS_4_7_15_RESULTS.md`, `runs/arms_4_7_15.png` | `archive/modalpinn2.0_archive/superseded_summaries/` | mid-experiment snapshots. `INDEX.md` still said "seven arms done, nine queued" and the CSV covered 7 of 16. `arms_master_results.csv` and `fresh_analysis/results_master.csv` supersede them. |
| `FORCE_ERROR.md`, `force_error_summary.csv`, `code/force_error.py` | `archive/modalpinn2.0_archive/force_analysis_out_of_scope/` | force and drag/lift analysis was removed from scope on 2026-08-27 (`fresh_analysis/decisions.md`). Nothing in the report or the live analysis reads them. The per-arm force columns remain in `arm_matrix.csv`. |
| `experiment/notebooks/matched_effort/*.zip` | `archive/modalpinn2.0_archive/matched_effort_zips/` | the raw Colab downloads. `matched_effort/zx/` is the extracted copy, and six scripts read its `train_log.txt`. |
