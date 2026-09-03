# Final GappyPOD diagnostic

This folder is the dissertation-facing GappyPOD calculation copied into the
ModalPINN2.0 repository for the Results chapter. It replaces the broader
exploratory workflow now archived at
`archive/gappy_pod_history/5_fresh_analysis_GappyPOD/` with one matched
diagnostic for the ModalPINN study.

## Question

Can the same 32 uniformly spaced cylinder-pressure measurements used by the
pressure-only ModalPINN identify the full flow when a POD basis supplies the
spatial structure?

This is an in-sample reconstruction diagnostic, not a generalisation test. The
POD mean and basis use all 201 CFD snapshots because the ModalPINN is also
trained and evaluated over that complete time record. GappyPOD therefore tests
coefficient identifiability under matched information, not prediction of an
unseen flow.

## Frozen configuration

- Dataset: `1_data/flow_cache.npz`, resolved automatically by `run_analysis.py`.
  It is a hardlink to `1_data/flow_cache.npz`, so it is one copy
  on disk rather than two, and it is git-ignored. The original GappyPOD
  workspace it came from is now at `archive/GappyPOD/`.
- Evaluation crop: `-4 < x < 8`, `-4 < y < 4` (51,654 nodes).
- State: unscaled joint vector `[u; v; p]` on the crop.
- POD library and reconstruction interval: all 201 snapshots.
- Retained rank: 6 (three complete travelling-wave mode pairs).
- Sensors: the exact 32 uniform target angles and nearest wall nodes used by
  `ModalPINN`'s `cut_simu_cylinder_only` loader.
- Solve: centred, unregularised least squares.
- Regions and relative-L2 definition: identical to the ModalPINN fresh
  analysis.
- Pressure noise: additive zero-mean Gaussian noise with the exact ModalPINN
  standard deviations `0`, `4.7265e-4`, `2.3633e-3`, and `4.7265e-3`.
- Noise seed: 0. One standard-normal pressure perturbation is reused at all
  three non-zero levels, scaled by the requested standard deviation, matching
  the single-seed dose design of the ModalPINN runs.

The folder implements only this frozen comparison so its outputs can be traced
directly into the report without carrying the exploratory study alongside it.

## Run

From the `ModalPINN2.0` repository root in the shared workspace:

```bash
python3 5_baselines/gappy_pod_final/run_analysis.py
```

In another checkout, point the script to the CFD cache explicitly:

```bash
GAPPYPOD_FLOW_CACHE=/path/to/flow_cache.npz \
  python3 5_baselines/gappy_pod_final/run_analysis.py
```

The script writes:

- `results/configuration.json` — complete frozen inputs and audit values;
- `results/metrics.csv` — clean/noisy field and first-harmonic metrics;
- `results/summary.json` — concise headline values used by the report.

## Dissertation figures

Run the three scripts in `scripts/figures/` to produce the focused PNG files in
`figures/final/`. The representative field figure can be regenerated from the
included `results/representative_snapshot.npz` even when the external CFD cache
is unavailable.

- `G01_clean_reconstruction.png` — one clean CFD snapshot, its rank-6
  GappyPOD reconstruction, and the absolute vertical-velocity error;
- `G02_clean_method_comparison.png` — the matched far-core vertical-velocity
  error for pressure-only ModalPINN, prior-assisted ModalPINN, dense ModalPINN,
  and GappyPOD;
- `G03_noise_sensitivity.png` — GappyPOD error under the four pressure-noise
  levels used by the ModalPINN study.

`results/chapter4_values.csv` is the compact plotting table for the two metric
figures. `results/representative_snapshot.npz` is the plot-ready data behind
the field figure. The comparison figure is diagnostic: GappyPOD receives a POD
basis built from the complete CFD record, so it demonstrates coefficient
identifiability within a supplied subspace rather than an independently
trained competitor outperforming ModalPINN.
