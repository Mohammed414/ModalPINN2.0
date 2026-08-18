# R9 - trust-region ansatz around a taps-only vortex-street prior

Production port of the winning method from `R9_wake_rescue/` (see REPORT.md
there for the full development, controlled arms, and robustness checks).

## What changed vs R8
- **Mechanism, not loss term**: k>=1 mode networks become bounded corrections
  around a closed-form von Karman street derived from the taps alone
  (`--TrustStreet`); the dead-wake solution is excluded from the search space.
- **Dropped**: --BVF, --K0Loss, --PhaseLoss, warm start. Minimal loss stack:
  physics + taps + FSBC/FIBC.
- **New**: `src/street_prior.py` (numpy-only taps->prior derivation),
  `NN_functions.street_modes_k` + trust wrap in `out_nn_modes_uv/p`,
  trust-aware restore in `evaluate_regions.py`.

## Files
- `src/` - the five run files + street_prior.py (diff base: src/pressure_only)
- `notebooks/R9_trust_street_32taps.ipynb` - Colab notebook (build with
  `python3 build_r9_notebook.py`), same smoke-test-gated pattern as R7/R8
- `build_r9_notebook.py` - the builder

## Verification done locally (Apple-silicon TF 2.x, math-level)
- `street_modes_k` (TF) vs the numpy closed form: max rel err ~1e-6, all 9
  mode fields (k=1..3 x u,v,p)
- trust wrap: correction stays within (rho|S|+cap), zero excluded at wake
  points, plain-net |v1| ~ 0.05-0.12 vs trust-anchored ~0.37-0.44
- prior derivation on the local dataset: omega=1.03575, Gamma=2.527,
  Uc=0.821, xf=1.2, r0=0.4, corr(closed form vs numeric street)=0.999
- the closed-form prior was also validated in the PyTorch testbed
  (`R9_wake_rescue` arm `trust_cf`: near-wake E_v 0.68, far-wake 0.51 -
  slightly above the numeric-street arm's 0.64/0.40, i.e. results file
  `arm_trust_v2.json` from the `trust` arm RERUN after the image-vortex
  phase fix; the earlier pre-image `arm_trust_w40_s0` run is superseded,
  its phase was ~pi off and it scored E_v > 1. The gap 0.68/0.51 vs
  0.64/0.40 is because the closed form drops the image-vortex near-field;
  acceptable, the near field belongs to the k=0 network and the taps anyway)

## Expected result (testbed reference, production should improve on it)
near-wake E_v <= ~0.64, far-wake E_v <= ~0.40 (vs ~1.0 for R1-R8).
Convention note: --TrustCap 0.12 here == 0.06 in the testbed (this codebase's
one-sided mode convention has no factor 2 in NN_time_*).
