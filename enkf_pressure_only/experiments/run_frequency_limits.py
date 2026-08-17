"""(4) Stability ceiling on gamma, and (5) does the OBSERVED tap signal (not
just the lift) track gamma?  The filter never sees Fy -- it sees 32 wall taps --
so the frequency handle has to be visible in the observable.  The tap-0 fit in
the earlier pass was amplitude-starved (A~5e-4); here the highest-variance tap
is used and a 2-harmonic model is fitted, because wall pressure carries a
strong 2*omega (drag-frequency) component that contaminates a single-tone fit.
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from frequency_parameterization import (  # noqa: E402
    ScalableCylinderFlowSolver, max_div_interior)
from estimator.data_interface import TapObservations  # noqa: E402


def fit_two_harmonic(t, f, w0):
    def model(tt, off, A1, p1, A2, p2, w):
        return (off + A1 * np.cos(w * tt + p1) + A2 * np.cos(2 * w * tt + p2))
    A0 = (f.max() - f.min()) / 2
    popt, pcov = curve_fit(model, t, f,
                           p0=[f.mean(), A0, 0.0, A0 / 3, 0.0, w0], maxfev=100000)
    resid = f - model(t, *popt)
    return dict(omega=float(abs(popt[5])),
                sigma_omega=float(np.sqrt(np.diag(pcov))[5]),
                A1=float(abs(popt[1])), A2=float(abs(popt[3])),
                r2=float(1 - np.var(resid) / np.var(f)))


def main():
    snap = np.load(os.path.join(HERE, 'spinup_snapshots.npz'))
    cfg = json.loads(str(snap['solver_config']))
    u0 = snap['u'][0].copy(); v0 = snap['v'][0].copy()
    dt_nom = cfg['dt']
    out = {}

    def build(g, Re=None):
        s = ScalableCylinderFlowSolver(
            Nx=cfg['Nx'], Ny=cfg['Ny'], Lxmin=cfg['Lxmin'], Lxmax=cfg['Lxmax'],
            Lymin=cfg['Lymin'], Lymax=cfg['Lymax'], x_c=cfg['x_c'],
            y_c=cfg['y_c'], r_c=cfg['r_c'],
            Re=(cfg['Re'] if Re is None else Re), dt=dt_nom * g)
        s.u = u0.copy(); s.v = v0.copy(); s.t = 0.0
        return s

    # ---- (4) stability ceiling: short runs, watch for blow-up ----------
    stab = []
    for g in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        s = build(g)
        n = int(round(40.0 / dt_nom))
        bad = False; peak = 0.0
        for k in range(n):
            s.step()
            if k % 200 == 0:
                mx = float(np.nanmax(np.abs(s.u)))
                peak = max(peak, mx)
                if not np.isfinite(mx) or mx > 20.0:
                    bad = True
                    break
        row = dict(gamma=g, dt=dt_nom * g, blew_up=bad, peak_absu=peak,
                   max_div=float('nan') if bad else max_div_interior(s),
                   t_reached=float((k + 1) * dt_nom))
        stab.append(row)
        print('[stab] gamma=%.2f dt=%.5f  blew=%s peak|u|=%.3f div=%.2e'
              % (g, dt_nom * g, bad, peak, row['max_div']), flush=True)
        if bad:
            break
    out['stability'] = stab

    # ---- (5) tap-observable frequency vs gamma -------------------------
    obs = TapObservations(n_taps=32)
    every = int(round(0.05 / dt_nom))
    tap_rows = []
    for g in [0.85, 0.885, 0.95, 1.00, 1.10]:
        s = build(g)
        n = int(round(140.0 / dt_nom))
        ts, taps = [], []
        for k in range(n):
            s.step()
            if k % every == 0:
                ts.append((k + 1) * dt_nom)
                taps.append(s.sample_pressure(obs.tap_x, obs.tap_y).copy())
        ts = np.array(ts); taps = np.array(taps)
        m = ts > 20.0
        tp = taps[m]; tt = ts[m]
        var = tp.var(axis=0)
        jbest = int(np.argmax(var))
        f2 = fit_two_harmonic(tt, tp[:, jbest], 1.17 * g)
        # also the leading POD-like mode of the unsteady tap field
        Xc = tp - tp.mean(axis=0)
        U_, S_, Vt_ = np.linalg.svd(Xc, full_matrices=False)
        f2m = fit_two_harmonic(tt, U_[:, 0] * S_[0], 1.17 * g)
        row = dict(gamma=g, best_tap=jbest,
                   tap_unsteady_rms=float(np.sqrt(np.mean(Xc ** 2))),
                   tap_mean_l2=float(np.linalg.norm(tp.mean(axis=0))),
                   best_tap_omega=f2['omega'], best_tap_sigma=f2['sigma_omega'],
                   best_tap_A1=f2['A1'], best_tap_A2=f2['A2'], best_tap_r2=f2['r2'],
                   mode1_omega=f2m['omega'], mode1_r2=f2m['r2'],
                   mode1_energy_frac=float(S_[0] ** 2 / np.sum(S_ ** 2)))
        tap_rows.append(row)
        print('[tap] gamma=%.3f tap%d  omega=%.5f (r2=%.5f)  mode1_omega=%.5f '
              '(r2=%.5f, %.1f%% energy)  rms=%.5f'
              % (g, jbest, f2['omega'], f2['r2'], f2m['omega'], f2m['r2'],
                 100 * row['mode1_energy_frac'], row['tap_unsteady_rms']), flush=True)
    out['tap_frequency'] = tap_rows

    with open(os.path.join(HERE, 'frequency_limits_raw.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print('wrote frequency_limits_raw.json', flush=True)


if __name__ == '__main__':
    main()
