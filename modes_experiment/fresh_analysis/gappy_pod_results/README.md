# GappyPOD result-section package

This is the deliberately small GappyPOD package retained beside the ModalPINN
fresh analysis for dissertation writing. It contains only the final figures and
the values/provenance needed to support them. The analysis implementation and
plot-ready field data remain in `GappyPOD/gappy_pod_final` at local commit
`b5fc513`.

## Files retained here

- `figures/G01_clean_reconstruction.png`: representative clean vertical-
  velocity snapshot, reconstruction, and absolute error;
- `figures/G02_clean_method_comparison.png`: clean far-core comparison against
  pressure-only, prior-assisted, and dense ModalPINN results;
- `figures/G03_noise_sensitivity.png`: GappyPOD far-core error at the four
  pressure-noise levels;
- `results/chapter4_values.csv`: exact values plotted in the two metric figures;
- `results/result_manifest.json`: frozen configuration, headline results, and
  the interpretation boundary for the comparison.

## Interpretation boundary

The POD mean and rank-6 basis were built from all 201 CFD snapshots also being
reconstructed. GappyPOD is therefore an in-sample coefficient-identifiability
diagnostic: it asks whether 32 pressure taps can recover the state when the
correct low-dimensional spatial subspace is supplied. It is not evidence that
GappyPOD generalises to unseen snapshots, and it is not a like-for-like claim
that GappyPOD independently outperforms ModalPINN.
