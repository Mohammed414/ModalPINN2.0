# K = 3 series: results

Three arms at the source paper's truncation (`--Nmodes 4`, k = 0,1,2,3), original ModalPINN
schedule unchanged (`--Tmax 9`, L-BFGS to convergence then Adam), all cold-started. All three
exited 0 at 9.01 h wall time with `DNN2_100_100_4_tanh.pickle` weights, confirming the mode count
independently of the flag.

## The headline: the collapse survives the paper's truncation

| arm | far-core k=1 amplitude | k=0..2 reference | change |
|---|---|---|---|
| physics only | **0.01895** | E3F 0.01852 | +2.3% |
| + wall vorticity flux (BVF) | 0.08945 | **R3** 0.01718 | **+420.7%** |
| + Kármán prior | 0.80878 | prior alone 0.80820 | +0.1% |

The pressure-only baseline recovers **1.9% of the true oscillating amplitude** in the far core at
k = 0,1,2,3 — statistically indistinguishable from the 1.85% it recovered at k = 0,1,2. The missing
mode was never the cause of the collapse. That closes the question that motivated this series.

This is a direct measurement, not an inference: arm 1 and E3F share cold start, `--Tmax 9`,
`Nint 50000`, `Ngrid 5/200` and the Adam phase, and differ only in `--Nmodes` (and the parameter
count that rides along with it, 21,608 against 12,306 per field).

## The prior is the only remedy that works, and it adds nothing to the prior

| arm | near-cylinder | near-wake | far-wake | far-core |
|---|---|---|---|---|
| physics only | 0.233 | 0.054 | 0.022 | 0.019 |
| + wall vorticity flux | 0.500 | 0.120 | 0.104 | 0.089 |
| + Kármán prior | 0.688 | 0.965 | 0.810 | 0.809 |

BVF doubles the near-cylinder amplitude and lifts the far core by about 4.7x over baseline, but at
0.089 it is still below the 0.10 collapse threshold — a real improvement that does not reconstruct
the wake.

The prior arm reaches 0.809, and **the analytical prior alone reaches 0.808**. Training against 32
taps plus physics moved the far-core amplitude by +0.1%. The wake in that arm is the prior's, not
the network's. This reproduces at the correct truncation the qualified finding the report already
carries.

## The Adam phase, measured

| arm | L-BFGS | after Adam | verdict |
|---|---|---|---|
| physics only | 3.79 h / 5,503 evals -> 3.000e-04 | 2.965e-04 | improved 1.2% |
| + wall vorticity flux | 6.22 h / 24,307 evals -> 6.521e-05 | 1.078e-04 | **worsened 65.3%** |
| + Kármán prior | 1.48 h / 6,056 evals -> 1.207e-03 | 1.146e-03 | improved 5.1% |

All three L-BFGS phases converged (flat loss tail, well inside the 50,000 cap). Adam then consumed
the rest of the 9-hour budget and contributed 1.2% and 5.1% in two arms while **degrading the BVF
arm by 65%** — and the saved checkpoint is the post-Adam state, so the BVF result reported above is
from the worse of the two available states. This repeats the pattern already visible in E3
(post-Adam worse) and is the strongest argument in the project for `--SkipAdam`, or for
checkpoint selection on a validation metric.

## What was verified, not assumed

- **Mode count**: every checkpoint is `DNN2_100_100_4`, the k=0..3 layer shape.
- **BVF target**: the log records `bvf_targets_Ntap32_seed0_KMAX3.npz`, so the arm trained against
  a wall-flux target reassembled to k = 0..3 rather than the project's k = 0..2 file.
- **Cold start**: `warm_started: false`, `restore_model: null` in all three loss summaries.
- **Flags**: `Nmodes 4`, `Freestream BC: True` in all three; `Fluctuation Inlet BC: True` in the
  BVF and prior arms only, matching R2 and R15.
- **No numerical failures**: no NaN, no Inf, no traceback in any log. The only warnings are
  TensorFlow deprecation notices and one benign L-BFGS line-search message.

## Two caveats

**The BVF arm's reference is R3, not R2 (corrected).** Arm 2 passes `--FluctuationInletBC`, so its
matched k=0,1,2 counterpart is **R3** (FIBC + BVF, 0.01718), not R2 (BVF only, 0.15027). R2 differs
in the inlet condition as well as the truncation. Against the matched reference the BVF arm
**improves by +420.7%**, not the −40.5% first reported here. Both R3 and this arm converged (flat
L-BFGS tails at 1.405e-04 and 6.521e-05), so the comparison is sound on that axis, though R3's
L-BFGS ran 8.9 h against 6.2 h here.

**The fluctuation inlet BC is harmful on the BVF arm and harmless on the prior arm.** R2 vs R3 is
the project's only clean FIBC pair — identical but for that flag, both converged — and adding it
cost **−88.6%** (0.15027 to 0.01718). Yet the prior arm carries FIBC and still reaches 0.809. So
FIBC interacts with the remedy rather than acting uniformly, and any BVF result must state whether
it was on.

**The physics evaluator ran in its original form.** `evaluate_physics_uniform.py` applies the v1
trust wrap unconditionally, so it was run only for the prior arm, where the wrap is correct. The
two prior-off arms have no uniform NS-residual number; their regional reconstruction errors
(`regions.json`) are the comparable metric.
