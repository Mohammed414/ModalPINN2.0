# metrics_v2 — how to use it, and what it says about the pre-repair runs

## Why E_v was replaced

Stage F's `E_v` fails two ways, both reproduced numerically in
`validate_metrics_v2.py`:

1. **It tracks phase, not skill.** All runs shed ~13% fast, so each sweeps
   through phase alignment during a 20 t.u. window. The old ranking reverses
   between window halves: `E_v` full window says EnKF (0.4603) beats free run
   (0.5181); second half says free run (0.6501) beats EnKF (0.7846).
2. **It is minimised by deleting the wake.** Against the truth delayed by a
   quarter period (structure and amplitude *perfect*, only the timing wrong),
   `E_v` = **0.9058 at s = 0** (no wake at all) versus **1.2991 at s = 1**
   (the correct field). Its minimum over s ∈ [0, 1.4] is at **s = 0**.

## What replaces it

| function | what it answers |
|---|---|
| `phase_aligned_field_error` | how wrong is the STRUCTURE, with timing fitted out (`E_aligned`) and reported (`tau_opt`) |
| `modal_metrics` | **primary.** per-mode complex amplitude vs `reference_truth_modal.npz`; mode-1 amplitude profile vs downstream x; amplitude and phase split per x |
| `leakage_attenuation` | how much a frequency-offset run's ω₀ amplitude is damped by the finite window |
| `Ev_old` | the broken metric, kept only for side-by-side plots |

Alignment method: a **sub-sample time shift by cubic interpolation**, not a
complex phase rotation. The runs are not monochromatic at ω₀ (they shed at
ω_s ≈ 1.14–1.17), so one rotation cannot align k=1 and k=2 at once, and the
k=0 mean must not shift at all. A time shift acts correctly on every harmonic
simultaneously and leaves the mean untouched. The complex rotation *is* used —
per mode, in `modal_metrics`, where it is **reported** as `psi_opt`, its
correct role.

## Damping test (mandatory; all pass)

`F_s(t) = mean_t F + s·(F(t) − mean_t F)`, phase held fixed.

Base A = truth delayed T/4 (correct answer known: minimum must be at s = 1):

| metric | s = 0 | s = 1 | argmin | |
|---|---|---|---|---|
| `E_v` (old) | 0.9058 | 1.2991 | **s = 0.0** | **FAIL** |
| phase-aligned `E_v` | 0.9058 | 0.0000 | s = 1.0 | PASS |
| modal k1 `amp_rel` | 1.0000 | 0.0040 | s = 1.0 | PASS |
| modal k1 cplx aligned | 1.0000 | 0.0056 | s = 1.0 | PASS |
| `|v1|(x)` profile err | 1.0000 | 0.0043 | s = 1.0 | PASS |
| modal k2 `amp_rel` | 1.0000 | 0.0341 | s = 1.0 | PASS |

The fitted shift recovers the imposed one: τ* = −1.5162 against an imposed
−T/4 = −1.5162, and stays correct for every s ≥ 0.1.

On the real runs (free run, EnKF) every new metric also scores s = 0 as its
worst value (amplitude metrics return exactly 1.0 there). Their minima sit at
s ≈ 1.1–1.2 rather than 1.0 — see below; that is a real frequency error in the
runs, not a metric defect.

## Read modal amplitudes at ω₀ together with the leakage factor

A run oscillating at ω_s fitted onto a basis at ω₀ over a window T has its k=1
coefficient damped by |sinc(Δω·T/2)|. Verified: free run ω_s = 1.17083
(+13.01%), predicted attenuation over T = 20 is **0.7234**, measured peak-amplitude
ratio **0.7184**. This is not a metric artefact — ModalPINN asserts its modes
*at* ω₀ and energy elsewhere genuinely is not in the mode — but the penalty
scales with window length, so:

* compare ω₀-referenced modal amplitudes **only between runs on the same window**;
* always read them alongside `modal_metrics(..., omega=dominant_omega(...))`,
  which removes the frequency error and isolates pure amplitude error.
  Refitting at each run's own ω_s moves the damping-test minimum back to
  s = 0.9 for both the free run and the EnKF.

## Pre-repair baseline — and an honest negative result

`experiments/metrics_v2_baseline_v2.json`, full 20 t.u. window:

| run | ω_est | leak | E_aligned | τ* | k1 amp deficit @ω₀ | @own ω | peak \|v1\| | persistence |
|---|---|---|---|---|---|---|---|---|
| free run | 1.17083 | 0.7234 | 0.5170 | +0.0418 | 0.1990 | 0.1006 | 0.5181 | 0.8088 |
| EnKF | 1.14456 | 0.8148 | 0.4389 | −0.1639 | 0.0876 | 0.1145 | 0.5898 | 0.8281 |
| shuffled | 1.15745 | 0.7717 | 0.4678 | −0.0255 | 0.1484 | 0.0978 | 0.5529 | 0.8172 |

Truth: peak |v̂₁| = 0.6522 at x = 2.90, 0.5262 at x = 7, persistence 0.8069.

**None of these three runs exhibits the ModalPINN failure mode.** All have
persistence 0.81–0.83 against the truth's 0.807 — their wakes persist
downstream; their deficits are frequency and a uniform ~10–20% amplitude
shortfall, not wake collapse. The new metrics were validated on a synthetic
wake-deleted field precisely because no existing run supplies one.

**Window sensitivity — the EnKF's apparent win does not survive.**
`E_aligned` does rank EnKF first in all three windows where it is defined
(full 0.4389, half-1 0.1966, half-2 0.2142), which is a genuine improvement
over the old metric's reversal. But the amplitude deficit at ω₀ ranks the
EnKF first only on the full window; on halves and thirds the free run or the
shuffled control wins. Once the frequency error is removed the picture is
window-stable and unambiguous: the deficit is 0.099–0.101 (free run),
0.106–0.132 (EnKF), 0.089–0.105 (shuffled) — the **EnKF is worst in five of
six windows, and never beats the shuffled negative control**. So the EnKF's
full-window advantage at ω₀ comes from its smaller frequency error
(+10.5% vs +13.0%, leakage 0.8148 vs 0.7234), not from better wake amplitude.
Any claim that the pre-repair EnKF reconstructs the wake better than a
sensor-shuffled control is not supported.

## Files

* `metrics_v2.py` — the metrics (all take run arrays as arguments)
* `validate_metrics_v2.py` — damping test → `experiments/metric_validation*.npz`, `figures/metric_validation.png`
* `run_metrics_v2_baseline.py` — baseline → `experiments/metrics_v2_baseline_v2.{json,npz}`
* `plot_metrics_v2_baseline.py` → `figures/metrics_v2_baseline.png`
* `_truth_grid_cache.npz` — regenerable resample of the truth onto the modal grid (gitignorable)

Applying these to a repaired run:

```python
truth = M.load_truth()
res = M.evaluate_run(truth, u_hist, v_hist, times_absolute, config, label='repaired')
```

`times_absolute` must be the 400.0-based clock (`tap_times_true`), not
`exp_times`; using the 0-based clock offsets every modal phase by
exp(i·k·ω₀·400).
