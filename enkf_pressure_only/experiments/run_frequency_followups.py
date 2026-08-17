"""Follow-up experiments for the frequency-parameterization study.

(1) Reynolds sweep redone properly: the first pass seeded curve_fit with a fixed
    guess and used a 20-t.u. burn-in.  Changing Re from a Re=100 limit cycle is a
    genuine physical transient (the limit-cycle amplitude has to re-equilibrate),
    so both the record and the burn-in are lengthened and the nonlinear fit is
    seeded from the FFT peak of the burned-in record.
(2) Mid-run gamma switch: the EnKF would perturb gamma every cycle, so the
    frequency response to a gamma change must be immediate, not a slow
    re-equilibration.  Measured with a Hilbert instantaneous frequency.
(3) Observation-operator sanity: does time dilation distort the wall-pressure
    tap signal the filter actually sees?
"""
import json
import os
import sys

import numpy as np
from scipy.signal import hilbert

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from frequency_parameterization import (  # noqa: E402
    ScalableCylinderFlowSolver, fit_sinusoid, max_div_interior, run_case)
from estimator.data_interface import TapObservations  # noqa: E402


def fft_peak(t, f):
    ff = f - f.mean()
    n = len(ff)
    dt = float(np.mean(np.diff(t)))
    F = np.abs(np.fft.rfft(ff * np.hanning(n)))
    fr = np.fft.rfftfreq(n, dt)
    return 2 * np.pi * fr[np.argmax(F)]


def main():
    snap = np.load(os.path.join(HERE, 'spinup_snapshots.npz'))
    cfg = json.loads(str(snap['solver_config']))
    u0 = snap['u'][0].copy()
    v0 = snap['v'][0].copy()
    dt_nom = cfg['dt']
    out = {}

    # ---------- (1) Reynolds, done properly ----------------------------
    C_rows = []
    for Re in [60.0, 80.0, 100.0, 130.0, 180.0]:
        s = ScalableCylinderFlowSolver(
            Nx=cfg['Nx'], Ny=cfg['Ny'], Lxmin=cfg['Lxmin'], Lxmax=cfg['Lxmax'],
            Lymin=cfg['Lymin'], Lymax=cfg['Lymax'], x_c=cfg['x_c'], y_c=cfg['y_c'],
            r_c=cfg['r_c'], Re=Re, dt=dt_nom)
        s.u = u0.copy(); s.v = v0.copy(); s.t = 0.0
        T_obs, t_burn = 400.0, 200.0     # long: Re change is a real transient
        n = int(round(T_obs / dt_nom)); every = int(round(0.05 / dt_nom))
        ts, Fy = [], []
        for k in range(n):
            s.step()
            if k % every == 0:
                ts.append((k + 1) * dt_nom); Fy.append(s.force_on_body()[1])
        ts = np.array(ts); Fy = np.array(Fy)
        m = ts > t_burn
        w_guess = fft_peak(ts[m], Fy[m])
        fit = fit_sinusoid(ts[m], Fy[m], w_guess)
        nh = int(m.sum()) // 2
        f1 = fit_sinusoid(ts[m][:nh], Fy[m][:nh], fit['omega'])
        f2 = fit_sinusoid(ts[m][nh:], Fy[m][nh:], fit['omega'])
        row = dict(Re=Re, omega=fit['omega'], sigma_omega=fit['sigma_omega'],
                   A=fit['A'], r2=fit['r2'], w_fft=float(w_guess),
                   omega_h1=f1['omega'], omega_h2=f2['omega'],
                   max_div_interior=max_div_interior(s))
        C_rows.append(row)
        print('[C2] Re=%.0f  w_fft=%.4f  omega=%.6f  A=%.5f  r2=%.6f  '
              'h1-h2=%.2e' % (Re, w_guess, fit['omega'], fit['A'], fit['r2'],
                              abs(f1['omega'] - f2['omega'])), flush=True)
    out['reynolds_v2'] = C_rows

    # ---------- (2) mid-run gamma switch -------------------------------
    s = ScalableCylinderFlowSolver(
        Nx=cfg['Nx'], Ny=cfg['Ny'], Lxmin=cfg['Lxmin'], Lxmax=cfg['Lxmax'],
        Lymin=cfg['Lymin'], Lymax=cfg['Lymax'], x_c=cfg['x_c'], y_c=cfg['y_c'],
        r_c=cfg['r_c'], Re=cfg['Re'], dt=dt_nom)
    s.u = u0.copy(); s.v = v0.copy(); s.t = 0.0
    T_pre, T_post = 80.0, 80.0
    g_post = 0.885
    every = int(round(0.05 / dt_nom))
    ts, Fy, gam = [], [], []
    t_obs = 0.0
    for k in range(int(round(T_pre / dt_nom))):
        s.dt = dt_nom * 1.0
        s.step(); t_obs += dt_nom
        if k % every == 0:
            ts.append(t_obs); Fy.append(s.force_on_body()[1]); gam.append(1.0)
    for k in range(int(round(T_post / dt_nom))):
        s.dt = dt_nom * g_post
        s.step(); t_obs += dt_nom
        if k % every == 0:
            ts.append(t_obs); Fy.append(s.force_on_body()[1]); gam.append(g_post)
    ts = np.array(ts); Fy = np.array(Fy); gam = np.array(gam)

    an = hilbert(Fy - Fy.mean())
    inst_w = np.gradient(np.unwrap(np.angle(an)), ts)
    # windowed fits either side of the switch, skipping 5 t.u. around it
    pre_m = (ts > 30.0) & (ts < T_pre - 2.0)
    post_early = (ts > T_pre + 2.0) & (ts < T_pre + 12.0)
    post_late = (ts > T_pre + 40.0)
    fpre = fit_sinusoid(ts[pre_m], Fy[pre_m], 1.17)
    fpe = fit_sinusoid(ts[post_early], Fy[post_early], 1.036)
    fpl = fit_sinusoid(ts[post_late], Fy[post_late], 1.036)
    out['gamma_switch'] = dict(
        g_post=g_post, T_pre=T_pre,
        omega_pre=fpre['omega'], omega_post_early=fpe['omega'],
        omega_post_late=fpl['omega'],
        inst_w_pre=float(np.median(inst_w[pre_m])),
        inst_w_post_1period=float(np.median(
            inst_w[(ts > T_pre + 0.5) & (ts < T_pre + 6.5)])),
        inst_w_post_late=float(np.median(inst_w[post_late])),
        amp_pre=fpre['A'], amp_post_late=fpl['A'])
    print('[switch] omega pre=%.5f  post(2-12 t.u.)=%.5f  post(late)=%.5f'
          % (fpre['omega'], fpe['omega'], fpl['omega']), flush=True)
    print('[switch] inst-w pre=%.5f  first-period-after=%.5f  late=%.5f'
          % (out['gamma_switch']['inst_w_pre'],
             out['gamma_switch']['inst_w_post_1period'],
             out['gamma_switch']['inst_w_post_late']), flush=True)
    np.savez_compressed(os.path.join(HERE, 'frequency_gamma_switch.npz'),
                        t_observer=ts, Fy=Fy, gamma=gam, inst_omega=inst_w)

    # ---------- (3) observation-operator sanity under gamma ------------
    obs = TapObservations(n_taps=32)
    tap_rows = []
    for g in [0.85, 0.885, 1.00, 1.10]:
        s = ScalableCylinderFlowSolver(
            Nx=cfg['Nx'], Ny=cfg['Ny'], Lxmin=cfg['Lxmin'], Lxmax=cfg['Lxmax'],
            Lymin=cfg['Lymin'], Lymax=cfg['Lymax'], x_c=cfg['x_c'], y_c=cfg['y_c'],
            r_c=cfg['r_c'], Re=cfg['Re'], dt=dt_nom * g)
        s.u = u0.copy(); s.v = v0.copy(); s.t = 0.0
        n = int(round(120.0 / dt_nom))
        ts2, taps = [], []
        for k in range(n):
            s.step()
            if k % every == 0:
                ts2.append((k + 1) * dt_nom)
                taps.append(s.sample_pressure(obs.tap_x, obs.tap_y).copy())
        ts2 = np.array(ts2); taps = np.array(taps)
        m2 = ts2 > 20.0
        tp = taps[m2]
        mean_p = tp.mean(axis=0)
        unsteady_rms = float(np.sqrt(np.mean((tp - mean_p) ** 2)))
        ft = fit_sinusoid(ts2[m2], tp[:, 0], 1.17 * g)
        tap_rows.append(dict(gamma=g, tap_unsteady_rms=unsteady_rms,
                             tap_mean_l2=float(np.linalg.norm(mean_p)),
                             tap0_omega=ft['omega'], tap0_A=ft['A'],
                             tap0_r2=ft['r2'],
                             mean_p=mean_p.tolist()))
        print('[taps] gamma=%.3f  unsteady_rms=%.5f  |mean_p|=%.4f  '
              'tap0_omega=%.5f  tap0_A=%.5f' %
              (g, unsteady_rms, np.linalg.norm(mean_p), ft['omega'], ft['A']),
              flush=True)
    out['tap_signal'] = tap_rows

    with open(os.path.join(HERE, 'frequency_followups_raw.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print('wrote frequency_followups_raw.json', flush=True)


if __name__ == '__main__':
    main()
