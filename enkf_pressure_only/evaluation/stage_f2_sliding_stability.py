"""
STAGE F2b: sliding-window stability of every ranking claim.

The halves/thirds table in stage_f2_metrics_v2.py answers "does the ranking
survive ONE particular re-cut?".  That is too weak: with 3 usable cuts, a
ranking can look stable by luck.  This script slides a window of length
W_PERIODS * 2*pi/omega_0 across the common window in steps of one
assimilation cycle and recomputes, at EVERY position:

    * the per-x k=1 amplitude ratio |c(x)| and phase error arg c(x)
    * the mean over x >= 1 of each

A claim "run A beats run B" is then reported as the FRACTION of sliding
positions at which it holds.  A claim that holds at 100% of positions is a
result; one that holds at 60% is a coin flip dressed as a finding, and it is
labelled as such here rather than in a footnote.

The phase drift is the point of this file.  The observer solver sheds at
omega_s = 1.1707 (13.0% fast), so an unassimilated run's phase error against
the truth GROWS LINEARLY with time; a run whose frequency has been corrected
holds a roughly constant phase error.  A single window cannot tell those two
apart -- it reports one number -- but the slope of phase error versus window
centre can, and that slope is the frequency error in disguise:

    d(phase error)/dt = omega_est - omega_0

so it is a direct, window-position-independent measurement of the one thing
the amplitude metrics are blind to.

No number here is tuned.  Truth is read only through metrics_v2.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import estimator  # noqa: F401
import metrics_v2 as M

EXP_DIR = os.path.join(ROOT, 'experiments')
K0 = 61
W_PERIODS = 1.0          # sliding window length, in periods of omega_0
STEP_CYCLES = 2          # 0.2 t.u.

RUNS = [
    ('free_run', 'stage_c_free_run_control.npz', 'u_hist', 'v_hist', 'solver_config', True),
    ('enkf_d1', 'stage_d_enkf_nominal.npz', 'u_mean_hist', 'v_mean_hist', 'config', True),
    ('enkf_d2', 'stage_d2_enkf_nominal.npz', 'u_mean_hist', 'v_mean_hist', 'config', False),
    ('d2_shuffled', 'stage_d2_enkf_shuffled.npz', 'u_mean_hist', 'v_mean_hist', 'config', False),
    ('d2_scrambled', 'stage_d2_enkf_scrambled_sensors.npz', 'u_mean_hist', 'v_mean_hist', 'config', False),
]


def main():
    truth = M.load_truth()
    T0 = 2 * np.pi / truth.omega_0
    gx = truth.gx
    xsel = gx >= 1.0

    V = {}
    times = None
    for label, fname, uk, vk, ck, cut in RUNS:
        d = np.load(os.path.join(EXP_DIR, fname), allow_pickle=True)
        cfg = json.loads(str(d[ck]))
        t = d['tap_times_true'].astype(float)
        u, v = d[uk], d[vk]
        if cut:
            u, v, t = u[K0:], v[K0:], t[K0:]
        _, Vg = M.project_run(u, v, cfg, truth.gx, truth.gy)
        V[label] = Vg
        times = t
        print('projected %s: %s' % (label, Vg.shape))

    n = len(times)
    wlen = int(round(W_PERIODS * T0 / (times[1] - times[0])))
    starts = list(range(0, n - wlen + 1, STEP_CYCLES))
    print('window = %.2f t.u. (%d samples, %.2f periods); %d positions'
          % (wlen * (times[1] - times[0]), wlen, W_PERIODS, len(starts)))

    amp_mae = {l: [] for l in V}
    ph_mean = {l: [] for l in V}
    ph_mae = {l: [] for l in V}
    amp_x = {l: [] for l in V}
    centres = []
    for s0 in starts:
        sl = slice(s0, s0 + wlen)
        centres.append(0.5 * (times[s0] + times[s0 + wlen - 1]))
        for l in V:
            m = M.modal_metrics(times[sl], V[l][sl], truth, 'v', K=1)[1]
            a = m['amp_ratio_x']; p = m['phase_err_x']
            ok = xsel & np.isfinite(a)
            amp_mae[l].append(np.mean(np.abs(1.0 - a[ok])))
            ph_mean[l].append(np.mean(p[ok]))
            ph_mae[l].append(np.mean(np.abs(p[ok])))
            amp_x[l].append(a)
    centres = np.array(centres)
    for dct in (amp_mae, ph_mean, ph_mae):
        for l in dct:
            dct[l] = np.array(dct[l])
    for l in amp_x:
        amp_x[l] = np.array(amp_x[l])

    # ---- phase drift: unwrap over window centres, fit a slope -------------
    drift = {}
    for l in V:
        ph = np.unwrap(ph_mean[l])
        A = np.stack([centres - centres.mean(), np.ones_like(centres)], axis=1)
        coef, *_ = np.linalg.lstsq(A, ph, rcond=None)
        resid = ph - A @ coef
        drift[l] = dict(
            slope_rad_per_tu=float(coef[0]),
            implied_omega=float(truth.omega_0 + coef[0]),
            intercept_rad=float(coef[1]),
            resid_std_deg=float(np.degrees(resid.std())),
            total_drift_deg=float(np.degrees(ph[-1] - ph[0])),
            mean_abs_phase_deg=float(np.degrees(ph_mae[l].mean())),
            max_abs_phase_deg=float(np.degrees(np.abs(ph).max())))
        print('%-13s phase slope %+.5f rad/t.u. -> omega %.5f (direct %.5f); '
              'drift over window %+.1f deg; |phase| mean %.1f deg'
              % (l, coef[0], drift[l]['implied_omega'], truth.omega_0 + coef[0],
                 drift[l]['total_drift_deg'], drift[l]['mean_abs_phase_deg']))

    # ---- pairwise claim stability ----------------------------------------
    pairs = [('enkf_d2', b) for b in ('free_run', 'enkf_d1', 'd2_shuffled', 'd2_scrambled')]
    pairs += [('enkf_d1', 'free_run')]
    claims = {}
    print('\n%-28s %14s %14s %14s' % ('claim (A beats B)', 'amp: frac pos',
                                      'phase: frac pos', 'verdict'))
    for a, b in pairs:
        fa = float(np.mean(amp_mae[a] < amp_mae[b]))
        fp = float(np.mean(ph_mae[a] < ph_mae[b]))
        claims['%s_vs_%s' % (a, b)] = dict(
            amp_frac_positions=fa, phase_frac_positions=fp,
            amp_mean_A=float(amp_mae[a].mean()), amp_mean_B=float(amp_mae[b].mean()),
            phase_mean_A_deg=float(np.degrees(ph_mae[a].mean())),
            phase_mean_B_deg=float(np.degrees(ph_mae[b].mean())),
            amp_stable=bool(fa > 0.95 or fa < 0.05),
            phase_stable=bool(fp > 0.95 or fp < 0.05))
        verdict = []
        verdict.append('AMP %s' % ('yes' if fa > 0.95 else ('no' if fa < 0.05 else 'UNSTABLE')))
        verdict.append('PH %s' % ('yes' if fp > 0.95 else ('no' if fp < 0.05 else 'UNSTABLE')))
        print('%-28s %14.3f %14.3f   %s' % ('%s > %s' % (a, b), fa, fp, ', '.join(verdict)))

    # ---- how far downstream is the amplitude recovered, at EVERY position -
    # fraction of sliding positions at which run beats free_run at each x
    frac_better_x = {}
    for l in V:
        if l == 'free_run':
            continue
        e = np.abs(1.0 - amp_x[l]); b = np.abs(1.0 - amp_x['free_run'])
        with np.errstate(invalid='ignore'):
            frac_better_x[l] = np.nanmean((e < b).astype(float), axis=0)
        f = frac_better_x[l]
        strong = xsel & np.isfinite(f) & (f > 0.95)
        print('%-13s beats free_run on |v1|(x) at >95%% of positions at %d x-nodes'
              ' (x up to %.2f)' % (l, strong.sum(),
                                   gx[strong].max() if strong.any() else float('nan')))

    out = os.path.join(EXP_DIR, 'stage_f2_sliding_stability.npz')
    jout = os.path.join(EXP_DIR, 'stage_f2_sliding_stability.json')
    for p in (out, jout):
        if os.path.exists(p):
            raise SystemExit('refusing to overwrite %s' % p)
    np.savez_compressed(
        out, centres=centres, gx=gx, wlen=wlen, w_periods=W_PERIODS,
        **{'amp_mae_%s' % l: amp_mae[l] for l in V},
        **{'ph_mean_%s' % l: ph_mean[l] for l in V},
        **{'ph_mae_%s' % l: ph_mae[l] for l in V},
        **{'amp_x_%s' % l: amp_x[l] for l in V},
        **{'frac_better_x_%s' % l: frac_better_x[l] for l in frac_better_x})
    with open(jout, 'w') as f:
        json.dump(dict(w_periods=W_PERIODS, n_positions=len(starts),
                       window_tu=float(wlen * (times[1] - times[0])),
                       omega_0=truth.omega_0, drift=drift, claims=claims), f, indent=1)
    print('Wrote %s\nWrote %s' % (out, jout))


if __name__ == '__main__':
    main()
