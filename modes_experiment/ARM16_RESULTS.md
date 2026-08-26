# Arm 16 — Kármán prior at the paper's boundary formulation, L-BFGS only

`--Nmodes 4`, cold start, 32 taps, `Nint 50000`, uniform sampling, **no inlet boundary condition
of either kind**, `--SkipAdam`, `--LBFGSFtol 2.22e-16`. Exit code 0.

## It ran as intended

| | |
|---|---|
| wall time | 3.46 h (not the 1.5 h estimated) |
| L-BFGS evaluations | 12,516 of a 50,000 cap — **converged on its own**, flat tail at 1.016e-03 |
| log confirms | `Freestream BC : False`, `Fluctuation Inlet BC : False`, `Skipping Adam training` |
| checkpoint | `DNN2_100_100_4_tanh.pickle` — layer shape confirms k = 0,1,2,3 |

## Headline: the prior does not need either inlet BC

Far-core k=1 transverse mode:

| | amplitude | relative L2 | phase corr |
|---|---|---|---|
| arm 3 — both inlet BCs, Adam ran | 0.80878 | 0.29477 | 0.97159 |
| **arm 16 — no inlet BC, no Adam** | **0.82667** | **0.26830** | **0.97470** |
| analytical prior alone, no network | 0.8082 | — | — |

Arm 16 is **+2.21% on amplitude, 8.98% lower relative L2, +0.32% on phase** against arm 3 — better
on all three, with two boundary conditions removed and 5.4 fewer hours of training. Against the
analytical prior alone it is +2.29%.

By region:

| region | amplitude | relative L2 | phase corr |
|---|---|---|---|
| near-cylinder | 0.80302 | 0.59621 | 0.80741 |
| near-wake | 0.98701 | 0.43788 | 0.92400 |
| far-wake | 0.82784 | 0.27203 | 0.97334 |
| far-core | 0.82667 | 0.26830 | 0.97470 |
| whole-domain | 0.88611 | 0.35254 | 0.93890 |

## But the mean flow is much worse

Full-field relative errors, from `evaluate_regions.py`:

| region | arm 3 E_u / E_v / E_p | arm 16 E_u / E_v / E_p |
|---|---|---|
| near-cylinder | 0.0593 / 0.0752 / **0.0161** | 0.2650 / 0.2594 / **0.1430** |
| near-wake | 0.1586 / 0.4803 / 0.2865 | 0.3475 / 0.7744 / 0.5144 |
| far-wake | 0.2001 / 0.4719 / 1.0361 | 0.1766 / 0.4387 / **3.2873** |
| other (upstream/off-axis) | 0.0467 / 0.1617 / 0.2663 | 0.4745 / 0.7646 / 0.6212 |
| whole domain | 0.1592 / 0.3768 / 0.2597 | 0.3149 / 0.5290 / 0.7195 |

Near-cylinder pressure error is **8.9x worse** (0.0161 to 0.1430); whole-domain pressure **2.8x
worse**; upstream u error **10.2x worse** (0.0467 to 0.4745). The upstream degradation is the
expected direct consequence of removing the inlet freestream prior — that is the region it acted
on. Far-wake pressure at 3.29 is worse than useless.

**So the two questions have opposite answers.** The k=1 oscillating mode — the quantity the
dissertation is about — is indifferent to the inlet formulation and to Adam. The mean flow and
pressure field are substantially worse without them. Which matters depends on the claim being made.

## Evaluator audit — a real bug, and it does not affect the headline

`evaluate_v1_smoke.py` **hardcodes both inlet treatments on**, with no CLI flag:

    out_nn_modes_uv(..., freestream_target=0.0, damp_fluctuations=True, ...)

against the trainer's

    freestream_target_v = 0. if args.FreestreamBC else None
    damp_fluct          = bool(args.FluctuationInletBC)

So for arm 16 the evaluator rebuilt the velocity field with an inlet treatment the network never
trained with. The same mismatch applies in reverse to arms 1-3 (arm 1 trained with the fluctuation
damping off; the evaluator forced it on).

**Why the far-core numbers are still valid.** Both treatments act through the ramp
`0.5(1 - tanh(3(x + 2)))`, which is 1.00 at x = -4, 2.5e-03 at x = -1, and **9.4e-14 at x = 3**.
Far-core, far-wake and near-wake all sit downstream of that, so the forced treatment is numerically
absent there. The affected quantities are `near-cylinder` and `whole-domain` in
`v1.json`, which include x < -1 — those two rows should not be quoted for any arm.

The three evaluators disagree with each other:

| script | inlet treatment | correct? |
|---|---|---|
| `evaluate_v1_smoke.py` | hardcoded ON, no flag | **no** |
| `evaluate_physics_uniform.py` | hardcoded ON, no flag | **no** |
| `evaluate_regions.py` | reads `--FreestreamBC` / `--FluctuationInletBC`, else infers from the run-directory name | **yes** |

`evaluate_regions.py` inferred correctly for arm 16: the run directory
`ModalPINN_2026_08_24-18_25_05__681_Ponly_Ntap32_V1RAD_rho0p6_x3p0` contains neither `FSBC` nor
`FIBC`, so it evaluated with both off, matching training. The regional table above is therefore
sound.

**`evaluate_physics_uniform.py` has the same hardcoding plus an unconditional prior trust wrap**,
which is why it is skipped on prior-off arms in this series. For arm 16 it ran, but its residual
should be treated as approximate for the same inlet reason.

## What this settles, and what it does not

**Settled:** the prior's wake recovery does not depend on either inlet boundary condition, and does
not depend on the Adam phase. A pure-paper formulation is viable for the k=1 result.

**Not settled:** arm 16 removed two things at once. Because the outcome was positive on the
oscillating mode, that ambiguity does not matter for the k=1 claim — neither removal hurt it. It
does matter for the mean-flow degradation, which cannot be attributed to the BC removal or to the
missing Adam phase from this run alone.

**Also unmeasured:** whether the *baseline* and the *BVF* arm behave the same way without the inlet
BCs. E3 (both off, k = 0,1,2) reached 0.00779 against E3F's 0.01852, so for the bare baseline the
freestream prior was worth +137.7%. Nothing here contradicts that.
