# Figure blueprint (draft wireframes)

These are proposed layouts only. No figure should be generated until its claim,
inputs, and metric are approved.

## F0 — Data and geometry sanity check

```text
┌──────────────────┬──────────────────┐
│ mesh/domain      │ cylinder + taps  │
├──────────────────┼──────────────────┤
│ snapshot example │ time/phase check │
└──────────────────┴──────────────────┘
```

Purpose: establish what data the analysis is using.

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

Purpose: include only a sensitivity study that answers a stated research
question.

## F5 — Optional force/derived-quantity figure

```text
┌──────────────────┬──────────────────┐
│ force time series│ harmonic metrics │
└──────────────────┴──────────────────┘
```

Purpose: keep force validation visibly separate from field validation.
