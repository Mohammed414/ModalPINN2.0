# Notes on contrast_ratios.csv (read before quoting the numbers)

Produced by `src/contrast_ratios.py` (diagnostic family: truth modal fields
with k>=1 modes damped downstream of x=1 by exp(-s(x-1)); "dead" = hard kill).

**Caveat on A_dom / A_wake (standard-loss contrast ~150-170x):** these look
respectable but are inflated by a diagnostic artifact. Most of that contrast
sits in the k=0 equation: killing the k>=1 modes removes the Reynolds-stress
divergence sum_m u_m . grad u_{-m} while the diagnostic family keeps the TRUE
mean flow fixed, so the k=0 residual jumps. A real optimizer is not held to
the true mean - it quenches exactly this term by drifting the mean toward the
steady base flow (which is an exact NS solution at Re=100 with zero
oscillation). The trainable-relevant number for the standard loss is
`A_k1_wake` (k>=1 momentum residual in the wake): contrast only ~5x between
true and the observed collapse profile (s=1), ~43x even for a hard kill -
i.e. the standard pointwise loss is structurally near-blind to the failure
mode, confirming the project hypothesis.

**B (relative residual), D (mean-linearized RZIF residual):** contrasts
300-1300x and ~1000x resp. at practical eps; both grow monotonically with
collapse severity. These are the strong trainable signals.

**C (lift-anchored k=1 CV budgets):** mean-square contrast 11x understates
its value: station-wise, a dead wake misses the measured lift budget by
~2|L1| at EVERY station x_s in [1.5, 7], while the true field sits at
0.28-0.87 |L1| - and that true-field residual is dominated by second-order
finite-difference quadrature error on scattered->grid interpolated fields,
which does not apply to the autodiff/quadrature implementation used in
training. C is the only term that ties far-wake amplitude to a MEASURED
scalar, so it is retained as an anchor despite the modest raw ratio.
