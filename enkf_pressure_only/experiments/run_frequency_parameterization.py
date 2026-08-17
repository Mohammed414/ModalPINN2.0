"""Driver for the frequency-parameterization sweeps.  Writes
experiments/frequency_parameterization.npz + .json (NEW files; nothing existing
is touched)."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from frequency_parameterization import run_case, fit_sinusoid, OMEGA_TRUTH  # noqa: E402

T_OBS_BASE = 220.0
T_OBS_SWEEP = 140.0
T_BURN = 20.0


def main():
    snap = np.load(os.path.join(HERE, 'spinup_snapshots.npz'))
    cfg = json.loads(str(snap['solver_config']))
    u0 = snap['u'][0].copy()
    v0 = snap['v'][0].copy()

    results = {}

    # ---- baseline, long record ----------------------------------------
    print('[base] gamma=1 long record', flush=True)
    base, tb, Fyb, Fxb = run_case(cfg, u0, v0, gamma=1.0,
                                  T_obs=T_OBS_BASE, t_burn=T_BURN)
    print('   omega=%.6f +-%.1e r2=%.6f' % (base['omega_eff'], base['sigma_omega'], base['r2']), flush=True)
    results['baseline'] = base
    np.savez_compressed(os.path.join(HERE, 'frequency_baseline_lift.npz'),
                        t_observer=tb, Fy=Fyb, Fx=Fxb,
                        config=json.dumps(cfg))

    # ---- A: time dilation ---------------------------------------------
    gammas = [0.70, 0.80, 0.85, 0.885, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.35, 1.50]
    A_rows = []
    for g in gammas:
        r, ts, Fy, Fx = run_case(cfg, u0, v0, gamma=g, T_obs=T_OBS_SWEEP,
                                 t_burn=T_BURN, w0_guess=1.17 * g)
        A_rows.append(r)
        print('[A] gamma=%.3f  dt=%.5f  omega_eff=%.6f  omega_internal=%.6f  '
              'A=%.5f  r2=%.6f  div=%.2e  blew=%s'
              % (g, r['dt_solver'], r['omega_eff'], r['omega_solver_clock'],
                 r['A'], r['r2'], r['max_div_interior'], r['blew_up']), flush=True)
    results['time_dilation'] = A_rows

    # ---- B: freestream rescaling at FIXED Re ---------------------------
    # U_inf = s, nu = s/Re_ref  (solver Re argument = Re_ref/s) keeps
    # Re = U*D/nu = Re_ref.  Exact dynamic similarity predicts omega ~ s.
    B_rows = []
    for s_u in [0.85, 0.885, 0.95, 1.00, 1.10]:
        r, ts, Fy, Fx = run_case(cfg, u0, v0, gamma=1.0, Re=cfg['Re'] / s_u,
                                 U_inf=s_u, T_obs=T_OBS_SWEEP, t_burn=T_BURN,
                                 w0_guess=1.17 * s_u)
        r['s_U'] = s_u
        B_rows.append(r)
        print('[B] U=%.3f Re_arg=%.1f  omega=%.6f  A=%.5f  r2=%.6f  div=%.2e  blew=%s'
              % (s_u, cfg['Re'] / s_u, r['omega_eff'], r['A'], r['r2'],
                 r['max_div_interior'], r['blew_up']), flush=True)
    results['freestream_fixed_Re'] = B_rows

    # ---- C: Reynolds number (physically real, U fixed at 1) ------------
    C_rows = []
    for Re in [60.0, 80.0, 100.0, 130.0, 180.0]:
        r, ts, Fy, Fx = run_case(cfg, u0, v0, gamma=1.0, Re=Re,
                                 T_obs=T_OBS_SWEEP, t_burn=T_BURN, w0_guess=1.15)
        C_rows.append(r)
        print('[C] Re=%.0f  omega=%.6f  A=%.5f  r2=%.6f  div=%.2e  blew=%s'
              % (Re, r['omega_eff'], r['A'], r['r2'],
                 r['max_div_interior'], r['blew_up']), flush=True)
    results['reynolds'] = C_rows

    with open(os.path.join(HERE, 'frequency_parameterization_raw.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('wrote frequency_parameterization_raw.json', flush=True)


if __name__ == '__main__':
    main()
