# Does R9 transfer beyond the cylinder? (airfoil analysis)

Tested numerically 2026-08-13, not argued from theory. Verdict up front:
**the mechanism transfers, the prior does not — and the blocker is physical,
not a coding problem.**

## Component-by-component audit

| component | transfers? | evidence |
|---|---|---|
| Trust-region ansatz `q_k = S_k + (rho\|S_k\|+c)tanh(net)` | **Yes, fully** | nothing in it references geometry; it needs only *some* nonzero prior field `S_k` |
| No-slip prior mask (`f_BC5`) | Yes, mechanical | currently `tanh(5*(r-R))`; any signed-distance function to the body works |
| omega_0 from tap-integrated lift | **Yes** | body-agnostic: any surface with taps gives a lift time series |
| Forces from taps on a non-circular body | **Yes, verified** | replacing theta-weights with arc-length + true outward normal recovers CL to **0.38%** vs exact Kutta-Joukowski on a Joukowski airfoil (t/c=0.108, camber 4.1%, alpha=5 deg). Caveat: numerical CD came out -0.10 instead of the d'Alembert 0, i.e. the sharp trailing-edge pressure singularity contaminates the *drag* integral at ~9% of CL — relevant below |
| Far-field street geometry under a conformal map | **Yes, asymptotically** | Joukowski `z = zeta + c^2/zeta` is the identity at large radius: position shift/r = 0.116 at r=3, 0.028 at r=6, 0.007 at r=12; metric stretch `dz/dzeta` in [0.88,1.12] at r=3 falling to [0.993,1.007] at r=12. So a street built in the circle plane maps to a valid airfoil-plane street, with a <=12% local correction inside ~3 chords |
| **Amplitude anchor (Gamma)** | **NO — this is the blocker** | see below |
| **Premise that a periodic street exists** | **NO at design conditions** | see below |

## Blocker 1: the amplitude anchor does not transfer

The cylinder gets `Gamma = 2.53` by inverting the **von Karman drag relation**
against tap-measured pressure drag. That works because a cylinder is a bluff
body whose drag *is* the wake's momentum deficit. For an airfoil near design
incidence, pressure drag is small, dominated by different physics (induced +
viscous), and — per the numerical check above — the drag integral is the one
most polluted by the trailing-edge singularity. The anchor is invalid there.

Obvious replacement: anchor `Gamma` on the measured **lift fluctuation**
`|L1|`, which any body's taps give directly. **Tested, and it does not work
as implemented**: the k=1 momentum-budget route gives `|L1|_model ~ 1.5` at
the same Gamma=2.53 that produces the good wake, while the cylinder's
measured `|L1| = 0.071` — a factor ~21 discrepancy, i.e. inverting it would
return Gamma ~ 0.12 instead of 2.53. Either the budget's normalization/slab
quadrature is wrong or the linear-in-Gamma scaling assumption fails. Until
that is resolved there is **no validated body-agnostic amplitude anchor**.
(Consistent with the earlier finding that this budget's station-wise residual
sat at 0.28-0.87|L1| even for the TRUE field.)

## Blocker 2 (the important one): an airfoil at design conditions has no street

The prior asserts a periodic Karman street. An attached-flow airfoil does not
have one — there is no alternating vortex row to anchor. The prior would be
structurally wrong, not merely imprecise, and the trust region would then
*prevent* the network from reaching the correct (non-street) answer: the same
cage that saved the cylinder becomes a trap.

The assertion is checkable from the taps before committing (this is the
saving grace): a clean single-frequency lift oscillation with finite |L1| ->
street prior is justified; broadband or near-zero |L1| -> it is not.

Where the method DOES apply beyond the cylinder:
- bluff bodies generally (square/rectangular cylinders, plates normal to
  flow, cylinder pairs): same street topology, and the drag anchor stays
  valid because they are momentum-deficit wakes
- **post-stall / high-incidence airfoils**, which shed a genuine
  bluff-body-like street
- low-Re airfoils where the wake is periodic

## Recommended path for an airfoil case (in order)

1. **Check the premise first.** Integrate the taps -> CL(t); confirm a
   dominant single frequency and finite |L1|. If absent, stop — this method
   is the wrong tool.
2. **Generalize the tap->force step** (small, verified): arc-length weights +
   true outward normal instead of theta-based. Keep the drag integral away
   from the trailing-edge singularity, or use lift only.
3. **Solve the amplitude anchor properly** (the real research step). Options:
   fix the k=1 momentum budget normalization and re-validate on the cylinder
   FIRST (where the answer is known); or anchor on the near-wake formation
   region rather than a global force; or make Gamma a trainable scalar with
   the taps constraining it (a 1-parameter fit is far weaker than an
   unconstrained wake field, so the dead-wake trap likely stays closed).
4. **Build the prior in the circle plane and map it** for a Joukowski-type
   section (validated above), or construct the street directly in the airfoil
   plane with image vortices replaced by a panel/vortex-lattice satisfaction
   of the no-through-flow condition.
5. **Re-validate the dead-wake exclusion**: confirm on the new geometry that
   the baseline still fails and the trust ansatz still revives — the R9
   testbed's validity gate, re-run.

Bottom line: R9 is a *bluff-body-wake* method whose mechanism is general.
Porting it to an airfoil is a genuine research step (blocked on the amplitude
anchor), not a configuration change — and it is only meaningful for a shedding
airfoil in the first place.
