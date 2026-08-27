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

- Field metrics: `TBD`.
- Modal/amplitude/phase metrics: `TBD`.
- Force metrics, if included: `TBD`.
- Regions and denominators: `TBD`.

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

### R5. Sensitivity/robustness (only if justified)

Possible variables: taps, noise, phase, region, or training configuration.

### R6. Forces and derived quantities (only if justified)

Keep this separate from field reconstruction because an accurate force does not
automatically establish an accurate wake field.

## Discussion

### D1. Direct answer to the research question

State what is learned, with the agreed metric and region.

### D2. Attribution

Separate what comes from the analytical prior, what comes from the network, and
what comes from the data/snapshot library.

### D3. Limitations

Record dataset scope, single-seed effects, metric limitations, and any
unmeasured comparison.

### D4. Implications and next experiment

Only include implications supported by the accepted results.
