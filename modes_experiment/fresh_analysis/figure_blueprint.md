# Figure blueprint (draft wireframes)

These are proposed layouts only. No figure should be generated until its claim,
inputs, and metric are approved.

## F0 — Data, sensors, and evaluation regions

```text
┌─────────────────────────────────────┐
│ full domain + cylinder + sensors    │
├──────────────────┬──────────────────┤
│ named regions    │ region node count│
└──────────────────┴──────────────────┘
```

Purpose: establish what data the analysis uses and make every regional metric
spatially interpretable. Each region must have a distinct outline or light
transparent fill, a direct label, and its node count. Overlapping regions must
be made explicit rather than visually hidden.

## F1 — Prior-only field reconstruction

```text
┌──────────────┬──────────────┬──────────────┐
│ reference    │ prior        │ error        │
├──────────────┼──────────────┼──────────────┤
│ reference    │ prior        │ error        │
└──────────────┴──────────────┴──────────────┘
```

Purpose: show what the prior supplies before any neural-network correction.

## F2 — Prior versus trained reconstruction

```text
┌──────────────────────────────┐
│ metric by region: prior/NN   │
├──────────────────────────────┤
│ field map: prior → trained   │
└──────────────────────────────┘
```

Purpose: identify where training helps, leaves the prior unchanged, or makes it
worse.

## F3 — Main comparison plot

```text
┌──────────────────┬──────────────────┐
│ amplitude/error  │ phase/correlation│
├──────────────────┼──────────────────┤
│ method summary   │ region summary   │
└──────────────────┴──────────────────┘
```

Purpose: compare only the methods that are defined in the final arm matrix.

## F4 — Optional robustness figure

```text
metric
  │  ● ● ●  method A
  │  ■ ■ ■  method B
  └──────────────────── parameter
```

Purpose: compare only controlled parameters with a matched reference: tap
count, collocation strategy, prior-plus-collocation, or pressure noise. Force
metrics and optimizer ablations are outside the current figure plan.
