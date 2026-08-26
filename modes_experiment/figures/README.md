# Figure set - ModalPINN arm study (k = 0,1,2,3)

Generated from `arms_master_results.csv`, which is built from the 15 arm folders
under `runs/arms/`. Every number in every figure is read from that CSV.

## Reproducing

```
cd figures
python fig1_collapse.py      # writes fig1_collapse.png + .pdf
python fig2_information.py
python fig3_robustness.py
python fig4_force_blind.py
python fig5_cost.py
```

Each script is standalone and imports shared colours, regime rules and the data
load from `fig_common.py`. Edit `fig_common.py` to change the palette or the
regime thresholds across the whole set at once; edit an individual script to
change one figure. Output is 300 dpi PNG plus vector PDF.

## The regime classification

Applied in `fig_common.regime()` and used for colour throughout:

| regime | rule | arms |
|---|---|---|
| collapsed | far-core amplitude < 0.15 | 1, 2, 8, 9, 14 |
| amplitude, no phase | relative L2 >= 1.0 | 6, 7 |
| recovered | otherwise | 3, 4, 10, 11, 12, 13, 15, 16 |

The middle category matters: arms 6 and 7 produce a non-trivial oscillating
amplitude (0.30 and 0.25) but a relative L2 above 1.0
(1.016, 1.006), meaning the reconstruction is worse than
predicting zero everywhere. Amplitude alone would rank them as partial
successes; they are not.

## What each figure claims

**fig1_collapse** - Wall pressure alone never recovers the oscillating wake.
Panel a ranks all 15 arms by far-core amplitude; panel b separates genuine
recovery from amplitude with the wrong phase.

**fig2_information** - The information has to come from the data, not the
optimiser. Panel a: 8, 16 and 32 taps are all collapsed. Panel b: the sampler
changes little; the prior changes everything.

**fig3_robustness** - Once wake structure is supplied, the reconstruction
tolerates sensor noise to 10% and is insensitive to the boundary-condition and
optimiser variants tested.

**fig4_force_blind** - Integrated forces do not diagnose the wake. Lift phase
error varies by 1.6 degrees across a 43x span in wake amplitude. Only arms
1, 2, 3 and 16 have force diagnostics; no other arm is implied.

**fig5_cost** - Optimisation effort does not explain the collapse. Every arm
terminated on its own convergence test, none reached the 50,000-iteration cap,
and the arms that spent the most L-BFGS effort are not the recovered ones.

## Caveats that belong in every caption

- **n = 1 per configuration, all at seed 0.** Run-to-run scatter is unmeasured,
  so differences of a few percent cannot be separated from initialisation
  variance. The 46x baseline-to-probe gap is far outside any plausible scatter;
  the 3-4% gaps between recovered arms are not.
- **Arm 5 (dense reference) is absent** - still training at the time of writing.
- Noise is tested only with the prior active, so the probe arm has no noise
  counterpart.
- Arm 10's whole-domain NS residual is 0.129,
  around 50x the other prior-active arms, while its far-core residual is the
  lowest of any arm. Not yet explained; flagged rather than smoothed.

## Field-based figures (added after the scalar set)

**fig6_flow_fields** - the collapse and the recovery as actual fields: DNS
reference against the collapsed pressure-only arm and the recovered prior arm,
in mean streamwise velocity, k=1 mode magnitude, and an instantaneous
transverse velocity snapshot.

**fig7_decay** - the collapse quantified along the wake. Panel a is |v1|
against streamwise distance; the pressure-only arm falls under 10% of DNS by
x/D = 1.25, i.e. within one diameter of the cylinder. Panel b is the centreline
phase, whose slope is 2*pi/wavelength:

| arm | streamwise wavelength | error vs DNS |
|---|---|---|
| DNS reference | 4.59 D | - |
| velocity probes (4) | 4.78 D | +4.2% |
| pressure + prior (15) | 4.87 D | +6.2% |
| wake-biased grid (7) | 78 D | near-flat phase |
| pressure only (1) | 172 D | near-flat phase |

This is the mechanism behind figure 1b. Arms 6 and 7 do not produce a noisy
street - they produce a nearly STANDING disturbance, a phase that barely
advances with x, so it cancels against the travelling DNS wave over any
extended region. That is why their relative L2 exceeds 1.0 while their
amplitude looks non-trivial.

### Requirements for these two

```
python parse_dns.py          # once: writes dns_raw.npz (182 MB) from the
                             # Boudina Re=100 DNS text file
python build_decay_data.py   # writes decay_profiles.json
python fig6_flow_fields.py
python fig7_decay.py
```

`modalpinn_eval.py` reimplements the ModalPINN forward pass in NumPy from the
saved weight pickles, including the analytic Karman street prior and the v1
radial trust wrap, so prior-active arms are evaluated exactly as trained. It
was validated against the stored far-core amplitude ratios on the same 12,460
nodes the project's own evaluator uses:

| arm | stored | reimplementation | ratio |
|---|---|---|---|
| 1 | 0.0189 | 0.0202 | 1.06 |
| 4 | 0.8658 | 0.8648 | 1.00 |
| 7 | 0.2540 | 0.2716 | 1.07 |
| 15 | 0.8388 | 0.8565 | 1.02 |
| 13 | 0.8576 | 0.8750 | 1.02 |

### One field-level finding not visible in any scalar

The recovered prior arm carries oscillation UPSTREAM of the cylinder and
off-axis, where physically there should be almost none (visible at the left and
outer edges of row 3 in fig6). The far-core metric samples only x >= 3,
|y| <= 2, so it cannot see this. It is a direct argument for the
fluctuation-inlet-BC variant, which exists to suppress exactly that.
