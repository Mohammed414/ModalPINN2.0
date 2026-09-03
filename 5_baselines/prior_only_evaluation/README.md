# Prior-only Kármán evaluation

This folder contains the standalone evaluation of arm 15's taps-only Kármán
prior, without neural-network training.

## Files

- `evaluate_prior_only.py` — NumPy evaluator.
- `prior_only_modal_metrics.json` — results on the cached Boudina DNS data.
- `prior_only_modal_fields.png` — DNS/prior/error field comparison.
- `street_prior_used.npz` — a copy of the exact arm-15 prior parameters.

The evaluator uses:

- Original prior parameters: `../../4_runs/15_karman_prior_fluct_off/street_prior_used.npz`
- Local copy used for the saved result: `street_prior_used.npz`
- DNS cache: `../../../GappyPOD/data/flow_cache.npz`
- Prior implementation: `../../src/R9/src/street_prior.py`

The training code and original dataset were not modified or copied.
