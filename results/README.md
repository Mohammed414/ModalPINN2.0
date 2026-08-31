# Results

Everything behind the dissertation, in one self-contained folder.
Nothing here reads a file outside it.

## Start here

**`all_results.xlsx`** — 15 sheets, every number, filterable. Open Contents.
*Headline* has the values the argument rests on. **All results** is the sheet to
filter: all 1,155 ModalPINN and Gappy POD rows in one table, each carrying the
configuration of its arm (taps, collocation, prior, noise). *Phase correction*
holds the 132 superseded pre-correction values for reference. The *Verification*
sheet is generated live by `code/verify.py` on every rebuild and currently reads
25 passed, 0 failed.

**`data/verified_results.csv`** — the same 1,155 rows as a plain CSV, if you
would rather work outside Excel. Its audit is
`data/verified_results_audit.json`.

## What is in each folder

| | |
|---|---|
| `data/results_master.csv` | the 945 accepted ModalPINN metrics. One row = one metric, one region, one arm. |
| `data/verified_results.csv` | all 1,155 ModalPINN and Gappy POD results in one filter-ready schema. |
| `data/v1_phase_correction_audit.csv` | the 132 old-versus-corrected values affected by the first-harmonic phase-origin bug. |
| `data/verified_results_audit.json` | automated cross-analysis, provenance, phase-convention, dataset-hash, and derived-value checks. |
| `data/analysis/` | the per-analysis files those rows were read from — A01 information, A02 tap count, A03 collocation, A04 prior attribution, A05 prior+collocation, A06 noise, A07 wavelength. |
| `data/gappy/` | the Gappy POD baseline: 210 metrics and the report-facing values. |
| `data/geometry/` | mesh coordinates, region masks, sensor positions, one representative snapshot. |
| `data/census_traces.npz` | the L-BFGS loss and gradient trace of all 17 training runs. |
| `data/dataset_extract.npz` | the slice of the CFD record the dataset figure draws. |
| `figures/` | the 20 figures, 300 dpi PNG. |
| `code/figures/` | one script per figure. Run any of them: `python3 code/figures/<name>.py`. |
| `code/analysis/` | the scripts that produced the metrics. Kept for provenance — re-running these needs the TensorFlow checkpoints, which are not in this folder. |
| `code/vendor/` | third-party modules the figures import, copied verbatim. |
| `code/verify.py` | checks every number against the file it came from. |
| `code/make_workbook.py` | rebuilds `all_results.xlsx`, regenerating the Verification sheet from a live `verify.py` run. |

## Regenerating

```
python3 code/figures/fig02_prior_attribution.py   # any one figure
python3 code/verify.py                            # check every number
python3 code/make_workbook.py                     # rebuild the spreadsheet
```

Needs `numpy`, `matplotlib`, `scipy`, `xlsxwriter`.

## Two things to know

**The evaluator was corrected and re-audited on 2026-08-31.** The first-harmonic fit is now
rotated back to absolute time before it is compared with a saved network mode.
Amplitude ratios and correlations are unaffected; `v1_mode` relative errors and
phases changed. The numbers in this folder and the dissertation are the
corrected ones. Full time-domain field errors were unaffected.

**The 36 learned-contribution rows now verify.** The consolidated audit
re-derives each one as prior-only relative error minus the corresponding arm's
relative error in the same region; the maximum discrepancy is recorded in
`data/verified_results_audit.json` and must be below $10^{-12}$.
