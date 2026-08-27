# Section blueprint (draft)

This is a writing and analysis outline, not a set of accepted results. We will
fill the evidence columns together.

## Methodology

### M1. Research question and comparison

- Question: what exactly should the reconstruction demonstrate?
- Comparison: prior only, neural network only, prior plus neural network, or
  another set agreed in `arm_matrix.csv`.
- Evidence needed: a precise input/output definition and no hidden CFD fields.

### M2. Dataset and preprocessing

- Dataset file and version: `TBD`.
- Snapshot/time range: `TBD`.
- Mesh/domain/cylinder-wall treatment: `TBD`.
- Variables and nondimensionalisation: `TBD`.
- Train/evaluation split: `TBD`.

### M3. Reconstruction methods

- Analytical prior: `TBD`.
- ModalPINN or other learned model: `TBD`.
- What information each method is allowed to use: `TBD`.

### M4. Metrics and visual diagnostics

- Field metrics: regional relative L2 for u, v, and p. The methodology must
  define the L2 norm, justify normalization by the reference-field norm, state
  how space and time are aggregated, and explain the interpretation of 0, 1,
  and values above 1.
- First-shedding-harmonic diagnostics: relative L2, amplitude ratio, normalized
  complex correlation, and global phase offset. Each metric must include its
  mathematical definition, ideal value, interpretation, and limitation.
- Regions: fixed definitions and node counts recorded in the data contract and
  shown visually before any regional result is presented.
- Force, lift, and drag metrics are outside the current scope.

## Results

### R1. Reference data and sanity checks

Show the geometry, representative snapshots, and basic temporal/modal content.

### R2. Prior-only reconstruction

Answer: how much of the field is already supplied by the analytical prior?

### R3. Network-only reconstruction

Answer: what does the network produce under the same evaluation protocol without
the prior?

### R4. Prior plus network

Answer: where does the network change the prior, and does that improve the field?

### R5. Controlled-parameter studies

Evaluate only comparisons with a defensible matched reference:

- pressure-tap count: Arms 8, 9, and 1;
- collocation strategy: Arms 1, 6, and 7;
- prior plus wake-biased collocation: Arms 15 and 10;
- pressure noise: Arms 15 and 11--13.

Do not infer an Adam effect from Arm 16 because optimizer, boundary-condition,
and L-BFGS settings changed together.

## Discussion

### D1. Direct answer to the research question

State what is learned, with the agreed metric and region.

### D2. Attribution

Separate what comes from the analytical prior, what comes from the network, and
what comes from the data/snapshot library.

### D3. Limitations

Record dataset scope, single-seed effects, metric limitations, and any
unmeasured comparison. State explicitly that force reconstruction and optimizer
ablation are not addressed by the current evidence.

### D4. Implications and next experiment

Only include implications supported by the accepted results.
