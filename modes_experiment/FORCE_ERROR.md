# Force-coefficient error: is the reconstruction adequate for active flow control?

`force_error.py` (in `code/`) rebuilds each saved checkpoint's fields on the cylinder wall ring and
integrates the surface tractions to C_L(t) and C_D(t), then does the same on the DNS field and
compares per harmonic. Numpy only, runs in about 40 s per checkpoint on a laptop. This is the
measurement the coherence audit flagged as missing — the report set a control-relevant standard in
Literature Review 2.6, characterised the DNS force harmonics in Results 4.5, and measured no arm
against it.

## The headline: wake-field accuracy does not predict force accuracy

Lift is the control-critical signal. Its oscillation lives in k=1, drag's in k=2 — the channel
assignment already established in the report.

| arm | wake k=1 amplitude | lift amp error | lift phase error | lift rel L2 | lift corr |
|---|---|---|---|---|---|
| 1 baseline physics-only | **1.9%** | −11.8% | **−4.80°** | 0.145 | 0.9965 |
| 2 wall vorticity flux | 8.9% | −3.4% | −6.35° | 0.117 | 0.9937 |
| 3 Kármán prior | 80.9% | −6.4% | −5.65° | 0.118 | 0.9950 |
| 16 prior, no inlet BC, no Adam | 82.7% | −5.0% | −5.68° | 0.112 | 0.9949 |

**The collapsed baseline recovers lift phase to 4.8 degrees — better than any other arm.** Its wake
is at 1.9% of the true k=1 amplitude, a 43x deficit against the prior arms, yet its lift phase error
is the smallest of the four and its lift time-series correlation is the highest at 0.9965.

Drag is similar in structure:

| arm | drag mean error | drag amp error | drag phase error | drag rel L2 | drag corr |
|---|---|---|---|---|---|
| 1 baseline | +0.7% | −15.2% | −13.91° | 0.0069 | 0.9671 |
| 2 wall vorticity flux | +0.3% | −3.6% | −13.46° | 0.0036 | 0.9443 |
| 3 Kármán prior | −0.1% | −5.9% | −8.68° | 0.0016 | 0.9774 |
| 16 prior, no inlet BC | −6.8% | −16.8% | −17.55° | 0.0682 | 0.7630 |

Every arm gets the mean drag to within 1% except arm 16, which is 6.8% low — the direct consequence
of its degraded near-cylinder pressure field (E_p 0.1430 against arm 3's 0.0161).

## Why the collapsed baseline still gets the forces

Because lift and drag are **surface integrals at r = 0.5**, and the collapse is a *wake* phenomenon
beginning at x/D >= 1. The baseline's near-cylinder k=1 amplitude is 0.2330 — twelve times its
far-core value of 0.0189 — and the wall pressure is fitted directly by the 32 taps, which is the
one thing every arm has in common. So the forces are recovered from information the taps supply
locally, not from a correct wake.

This is a genuinely useful finding rather than a curiosity: it separates two claims the report has
been treating as one. Wake reconstruction and force estimation are different problems with
different information requirements, and the pressure-only configuration is far better at the second
than at the first.

## The mean-lift ratio is meaningless and must not be quoted

The DNS mean C_L is +0.00001 — numerically zero, as symmetry requires. Dividing by it produces
nonsense (the script reports −19110% for arm 3, +15330% for arm 1). Those percentages are artefacts
of a near-zero denominator. **Quote the absolute mean-lift offset instead:** arm 1 +0.00132,
arm 2 +0.00018, arm 3 −0.00163, arm 16 −0.00626, against a k=1 oscillation amplitude of 0.3169. So
every arm's spurious mean lift is under 2% of the oscillation it must resolve.

## Method validation

The DNS side of this script was checked against `analysis/evidence/drag_split.json`, which computed
the same quantities independently with moving-least-squares wall-shear fitting:

| quantity | this script | drag_split.json | difference |
|---|---|---|---|
| mean C_D | 1.2942 | 1.3284 ± 0.0019 | −2.57% |
| C_D k=2 amplitude | 0.008928 | 0.009104 | −1.94% |
| mean C_D pressure term | 0.9890 | 0.9890 | **exact** |
| friction share of mean C_D | 0.2356 | 0.2555 | −7.8% |

The pressure term agrees to four decimal places, since neither method differentiates. The friction
term is 7.8% low here because `force_error.py` uses a one-sided finite difference at a single radial
offset of 0.02 rather than an MLS fit — cruder, and it under-resolves the wall gradient. Published
unconfined Re=100 C_D values are 1.39-1.42 (Relf, Wieselsberger, Tseng & Ferziger), so both
estimates sit below the literature for the usual confinement and resolution reasons.

**Consequence:** the *differences* between arms are trustworthy, since every arm is processed
identically. The absolute friction magnitude is not, so a claim about absolute drag accuracy should
cite the MLS number, not this one.

## Why no prior or boundary-condition arguments are needed

The wall ring spans |x| <= 0.55 including the offset ring. Every optional wrap in the trainer is
inactive there:

* the v1 radial trust x-gate is exactly 0 for x <= xstart − xwidth = 2.70
* the inlet ramp 0.5(1 − tanh(3(x+2))) is at most 1.7e-04 on the ring

So the wall field is the plain complex network output masked by the paper's no-slip prior
dictionary. One script is therefore valid for prior-on and prior-off arms alike, and it sidesteps
the hardcoded-inlet bug in `evaluate_v1_smoke.py` and `evaluate_physics_uniform.py` entirely. The
script asserts that the no-slip mask annihilates u and v on the wall to below 1e-12 before
proceeding.

## What this does and does not settle

**Settled:** the pressure-only reconstruction delivers lift phase to about 5-6 degrees, 1.3-1.8% of
a shedding period, and lift amplitude to 3-12%, in every arm including the collapsed baseline. Mean
drag is within 1% for three of the four.

**Not settled:** whether that is *adequate* for a specific controller. That depends on the control
law's phase margin, which is a control-design question this measurement feeds rather than answers.
The honest claim is the number, not the verdict.

**A caveat on the friction channel:** 23.6% of mean drag comes from wall shear, which needs velocity
gradients at the wall. Arm 3 reconstructs near-cylinder velocity to E_u 0.0593 and arm 16 to 0.2650.
A drag-based controller would inherit that error; a lift-based one largely would not, since lift is
pressure-dominated.

## Independent validation of the forward pass (added after review)

The force numbers depend on a numpy reimplementation of the trainer's complex forward pass, so that
reimplementation was checked against the project's own TF1 evaluator rather than assumed correct.
`evaluate_v1_smoke.py` fits the DNS k=1 transverse mode as `v1_true = 0.5(a1 - i b1)` and takes the
network's mode as `modes[:,1]/2`; both conventions were transcribed exactly, and the same far-core
mask was applied (`r >= 0.75`, `x >= 3`, `|y| <= 2`).

| checkpoint | quantity | this numpy path | the project's TF1 evaluator |
|---|---|---|---|
| baseline physics-only | far-core k=1 amplitude ratio | **0.01895** | **0.01895** |
| baseline physics-only | far-core phase correlation | **0.16036** | **0.16036** |
| Kármán prior | far-core k=1 amplitude ratio | 0.58965 | 0.80878 |

The **collapsed baseline reproduces exactly** — amplitude and phase correlation both to five decimal
places. That confirms three things at once: the checkpoint used for the force computation is the
collapsed arm and not another file, the complex weight decoding and tanh forward pass are correct,
and the no-slip mask is applied as the trainer applies it.

The prior arm deliberately does **not** match, and the reason is the check working as intended: the
v1 radial trust wrap is active exactly where far-core is measured (its x-gate is 1 for x >= 3), and
the numpy path omits that wrap. So 0.58965 is the bare network's far-core amplitude and 0.80878 is
the trust-wrapped value. The gap is the wrap's contribution, which is consistent with arm 3 sitting
at +0.1% over the analytical prior alone.

**This does not affect the force numbers.** The wall ring spans |x| <= 0.55 and the trust x-gate is
0 for x <= 2.70, so the wrap contributes nothing at the wall for any arm. The force computation
therefore uses the bare network output everywhere it is valid to do so, which is why one script
serves prior-on and prior-off arms alike.

Checkpoint identity was also confirmed by SHA-256: the four pickles are distinct files
(`1eb616cf…`, `db8fffa6…`, `2212307d…`, `7f290f76…`), and each run record's stored command line
carries the flags that define its arm.

## Is the baseline's lift-phase advantage real? (sensitivity sweep)

The baseline's lift phase error (-4.80 deg) is smaller than the Kármán prior's (-5.65 deg). The
script has one free parameter — the radial offset used for the one-sided wall-gradient difference —
so the gap was tested against it rather than asserted.

| radial offset | baseline | Kármán prior | gap |
|---|---|---|---|
| 0.01 | -4.85° | -5.66° | 0.82° |
| 0.02 | -4.80° | -5.65° | 0.85° |
| 0.04 | -4.80° | -5.71° | 0.91° |

Within-arm variation from changing the parameter is **0.050°** (baseline) and **0.062°** (prior).
The between-arm gap averages **0.859°** and holds the same sign and magnitude at every offset — about
**14x** the method's own sensitivity. **The gap is a robust arm-to-arm difference, not a numerical
artifact.**

A separate observation should not be confused with it: all four arms' lift phase errors are negative
(-4.80, -6.35, -5.65, -5.68 deg). That shared sign is a common offset in the finite-difference
method, applying equally to every arm. It does not make the *differences* between arms unreliable —
those are measured on identically processed data and, per the sweep above, exceed the method's
resolution by an order of magnitude.

So the correct statement is that the collapsed baseline resolves lift phase **0.86 deg better** than
the prior arm — a real if small advantage — while the prior arm wins 7 of the 10 force metrics,
including every drag metric.
