# Arms 4, 7 and 15: results

Three arms completed at the source paper's truncation (`--Nmodes 4`), all cold-started, all on the
E3F/R2 schedule (`--Tmax 9`, L-BFGS to convergence then Adam), freestream inlet prior on,
fluctuation inlet damping off. All three exited 0 and L-BFGS converged to a flat loss tail well
inside the 50,000-evaluation cap (24.8%, 75.4% and 69.3% respectively).

## The reported "error" was my guard misfiring, not a run failure

Arms 4 and 7 ended on

    AssertionError: evaluation failed for ['physics'] - the summary is incomplete

That is a **false alarm caused by my own check**, and both runs are sound. `physics.json` is absent
by design on prior-off arms: `evaluate_physics_uniform.py` applies the v1 trust wrap with no opt-out,
so on a prior-off checkpoint it would rebuild a function the network never trained. The notebooks
therefore skip it (`if pf:` — `pf` is empty without the prior), which is documented behaviour. My
`EVAL_FAILURES` guard could not distinguish "skipped on purpose" from "crashed" and flagged the
deliberate skip.

Verified: every arm's `physics.json` presence matches whether it is prior-active — arms 4 and 7
absent, arm 15 present. Nothing crashed. The guard has been corrected in all twelve notebooks to
count physics as required only when the arm is prior-active, so the eight still queued will not
raise it.

## Arm 4 — interior velocity probes: the mechanism test passes

32 wall taps plus 40 interior velocity probes, i.e. the source paper's own sparse configuration,
never previously run in this project.

| | far-core k=1 amplitude | phase correlation |
|---|---|---|
| arm 1, taps only | 0.01895 | 0.16036 |
| **arm 4, taps + probes** | **0.86578** | **0.99193** |
| analytical prior alone | 0.8082 | — |

**45.7x the collapsed baseline's amplitude, and the only arm to beat the analytical prior on its
own merits (+7.12%).** Near-wake amplitude is 0.99194 at correlation 0.99972 — essentially exact.

This is the causal confirmation of the identifiability argument. The claim was that the 32 taps have
an exactly zero Jacobian with respect to wake amplitude because they all sit at r=0.5, so no amount
of optimisation can recover the wake from them. Adding sensors that *do* see the wake revives it,
with nothing else changed and no prior injected. The mechanism moves from a rank calculation to a
measurement.

Field errors are the best of any arm: whole-domain E_u 0.0466, E_v 0.2023, E_p 0.1285, against arm
3's 0.1592 / 0.3768 / 0.2597.

## Arm 7 — focused collocation does not rescue the collapse

Wake-biased grid sampling, no prior. This was genuinely unmeasured: all three wake-focused runs in
the project's history had the prior active.

| | far-core amplitude | phase correlation | relative L2 |
|---|---|---|---|
| arm 1, uniform collocation | 0.01895 | 0.16036 | 0.99744 |
| **arm 7, wake-biased grid** | **0.25404** | **0.10256** | **1.00621** |

Amplitude rises 13.4x, but **the phase is lost** — correlation falls from 0.160 to 0.103 — and the
relative L2 exceeds 1.0, meaning the reconstruction is *worse than predicting zero everywhere*. The
arm produces oscillation of roughly the right magnitude in the wrong place and the wrong phase.

This is the predicted result, stated sharply: collocation points do not measure anything. Moving
them into the wake changes where the PDE residual is enforced, not what the sensors can see. A
decayed field satisfies the residual almost as well as the true field, so concentrating residual
enforcement produces amplitude without structure.

**The pair (arm 4, arm 7) is the strongest evidence in the project for the identifiability claim.**
Adding information revives the wake; redistributing physics enforcement does not.

## Arm 15 — the fluctuation inlet flag, isolated

Arm 15 differs from arm 3 in exactly one flag, both with Adam, both cold, uniform sampling.

| | arm 3 (flag on) | arm 15 (flag off) |
|---|---|---|
| far-core amplitude | 0.80878 | **0.83875** (+3.71%) |
| far-core phase correlation | 0.97159 | **0.97658** |
| near-cylinder E_p | 0.0161 | 0.0171 |
| whole-domain E_p | **0.2597** | 0.3767 |
| lift amplitude error | -6.4% | **-4.8%** |
| lift phase error | -5.65° | **-5.56°** |
| drag mean error | **-0.11%** | +0.24% |

Dropping the flag improves the wake by 3.7% and both lift metrics, and costs whole-domain pressure.
Arm 15 is the correct reference for arms 10-13 when they land, and it is now the project's best
prior-arm number.

Note this **separates the two changes that arm 16 confounded**: arm 16 removed both inlet conditions
*and* Adam and reached 0.82667. Arm 15 removes only the fluctuation flag and reaches 0.83875, so the
fluctuation flag alone accounts for the gain and the freestream prior is not what was holding the
wake back.

## All seven completed arms

| arm | wake amp | wake corr | lift amp err | lift phase err | drag mean err |
|---|---|---|---|---|---|
| 1 baseline physics-only | 0.01895 | 0.16036 | -11.8% | -4.80° | +0.68% |
| 2 wall vorticity flux | 0.08945 | 0.10809 | -3.4% | -6.35° | +0.31% |
| 3 Kármán prior (fluct on) | 0.80878 | 0.97159 | -6.4% | -5.65° | -0.11% |
| **4 taps + 40 probes** | **0.86578** | **0.99193** | -6.0% | -5.88° | +1.30% |
| **7 wake-biased grid** | 0.25404 | **0.10256** | -2.3% | -6.53° | +0.60% |
| **15 prior (fluct off)** | **0.83875** | 0.97658 | -4.8% | **-5.56°** | +0.24% |
| 16 prior, no inlet BC, no Adam | 0.82667 | 0.97470 | -5.0% | -5.68° | -6.83% |

**The force finding holds across all seven.** Lift phase error stays in a 1.7° band (-4.80° to
-6.53°) while wake amplitude spans 0.019 to 0.866 — a 46x range. Arm 7 makes the point most
sharply: its wake reconstruction is worse than predicting zero, and its lift amplitude error is the
smallest of all seven at -2.3%. Wake-field accuracy and force accuracy are separate quantities with
separate information requirements.

## Caveats

* `physics.json` exists only for arms 3, 15 and 16. Use `regions.json` for field errors throughout —
  it is the only evaluator that handles the inlet flags correctly.
* Do not quote `near-cylinder`, `near-wake` or `whole-domain` from any `v1.json`: those regions
  include the inlet ramp, which both v1 and physics evaluators force on regardless of training.
  `far-core` and `far-wake` are unaffected (ramp is 9.4e-14 at x=3).
* Arm 4 changes the sensor set, so it is not comparable to arms 1-3 on sensor count. It is the
  paper's sparse configuration, not a pressure-only arm.
