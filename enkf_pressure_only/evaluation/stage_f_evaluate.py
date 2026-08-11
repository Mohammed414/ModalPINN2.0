"""
STAGE F: withheld full-field evaluation. Only script (besides
build_reference_truth.py) that touches the ground truth, and only after
every observer run (Stage C free-run, Stage D EnKF + negative controls)
has already completed and saved its state history to disk.

Compares, at every one of the 201 assimilation instants:
  - free-run (no assimilation) control
  - EnKF (correct pressure sequence)
  - shuffled-pressure negative control
against the withheld reference_truth_full.npz (raw, untruncated scattered
CFD snapshots -- NOT the 3-mode Mtrue_* truth, to avoid circularity with
any modal assumption).

Metrics: E_u(t), E_v(t) relative L2 field error (solver's own Cartesian
velocity field, interpolated onto the truth's native scattered mesh
points); a phase-synchronization metric via the Hilbert-transform
instantaneous phase of each run's OWN predicted tap-0 pressure signal vs.
the true measured tap-0 pressure signal.
"""
import json
import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import hilbert

import estimator  # for CylinderFlowSolver's grid helper only, guard installed
from estimator._leakage_guard import allow_truth_access
from estimator.ns_solver import CylinderFlowSolver

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(HERE, '..', 'experiments')
FIG_DIR = os.path.join(HERE, '..', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


def grid_from_config(c):
    """Rebuild just the coordinate arrays (cheap) without a full solver
    instance's Poisson factorization -- avoids re-factorizing per call."""
    dx = (c['Lxmax'] - c['Lxmin']) / c['Nx']
    dy = (c['Lymax'] - c['Lymin']) / c['Ny']
    x_centers = c['Lxmin'] + (np.arange(c['Nx']) + 0.5) * dx
    y_centers = c['Lymin'] + (np.arange(c['Ny']) + 0.5) * dy
    return x_centers, y_centers


def velocity_at_centers(u, v):
    u_c = 0.5 * (u[:, :-1] + u[:, 1:])
    v_c = 0.5 * (v[:-1, :] + v[1:, :])
    return u_c, v_c


def interp_field_to_points(field, x_centers, y_centers, px, py):
    interp = RegularGridInterpolator((y_centers, x_centers), field,
                                      method='linear', bounds_error=False, fill_value=np.nan)
    return interp(np.stack([py, px], axis=-1))


def relative_l2_error_series(u_hist, v_hist, x_centers, y_centers,
                              ref_u, ref_v, ref_x, ref_y, r_c, x_c, y_c):
    """u_hist/v_hist: (Nt, Ny, Nx+1)/(Nt, Ny+1, Nx) MAC arrays.
    ref_u/ref_v: (Nt, Npts) truth at (ref_x, ref_y). Excludes truth points
    within 1.5*dx of the cylinder (immersed-boundary stair-step region,
    where the solver's representation is least trustworthy by construction,
    not because we're hiding unfavorable error)."""
    dx = x_centers[1] - x_centers[0]
    keep = np.hypot(ref_x - x_c, ref_y - y_c) > (r_c + 1.5 * dx)
    px, py = ref_x[keep], ref_y[keep]

    Nt = u_hist.shape[0]
    Eu = np.empty(Nt); Ev = np.empty(Nt)
    for k in range(Nt):
        u_c, v_c = velocity_at_centers(u_hist[k], v_hist[k])
        u_est = interp_field_to_points(u_c, x_centers, y_centers, px, py)
        v_est = interp_field_to_points(v_c, x_centers, y_centers, px, py)
        u_true = ref_u[k][keep]; v_true = ref_v[k][keep]
        valid = np.isfinite(u_est) & np.isfinite(v_est)
        Eu[k] = np.linalg.norm(u_est[valid] - u_true[valid]) / np.linalg.norm(u_true[valid])
        Ev[k] = np.linalg.norm(v_est[valid] - v_true[valid]) / np.linalg.norm(v_true[valid])
    return Eu, Ev


def phase_series(signal):
    analytic = hilbert(signal - signal.mean())
    return np.angle(analytic)


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def main():
    with allow_truth_access():
        truth = np.load(os.path.join(HERE, '..', 'data', 'reference_truth_full.npz'))
        ref_x, ref_y = truth['ref_x'], truth['ref_y']
        ref_cu, ref_cv = truth['ref_cu'], truth['ref_cv']  # (Nt, Npts)

    runs = {}
    free = np.load(os.path.join(EXP_DIR, 'stage_c_free_run_control.npz'))
    runs['free_run'] = dict(u=free['u_hist'], v=free['v_hist'], Fy=free['Fy_hist'],
                             config=json.loads(str(free['solver_config'])))

    enkf = np.load(os.path.join(EXP_DIR, 'stage_d_enkf_nominal.npz'))
    runs['enkf'] = dict(u=enkf['u_mean_hist'], v=enkf['v_mean_hist'], Fy=enkf['Fy_mean_hist'],
                         config=json.loads(str(enkf['config'])))

    shuf = np.load(os.path.join(EXP_DIR, 'stage_d_enkf_shuffled.npz'))
    runs['shuffled'] = dict(u=shuf['u_mean_hist'], v=shuf['v_mean_hist'], Fy=shuf['Fy_mean_hist'],
                             config=json.loads(str(shuf['config'])))

    tap_p_measured = free['tap_p_measured']  # true measured, same for all runs
    x_c, y_c, r_c = 0., 0., 0.5

    # TRUE phase reference: a crude lift-coefficient PROXY built directly from
    # the measured taps (sum p*n_y*dtheta around the sorted taps) -- NOT the
    # tap-0 raw pressure, which Stage A found is dominated by the 2nd harmonic
    # (2*omega_0), not the fundamental, and would give a meaningless Hilbert
    # phase. This proxy stays on the estimator-permitted side (only tap_x/y/p),
    # it's simply a different (fundamental-dominant) combination of the same
    # legitimately-available tap measurements, not a truth-side signal.
    obs_taps = np.load(os.path.join(HERE, '..', 'data', 'tap_observations.npz'))
    tx, ty = obs_taps['tap_x_32'], obs_taps['tap_y_32']
    theta = np.arctan2(ty, tx)
    order = np.argsort(theta)
    ny_sorted = np.sin(theta[order])
    dtheta = 2 * np.pi / len(theta)
    Cl_proxy_true = -np.sum(tap_p_measured[:, order] * ny_sorted[None, :], axis=1) * dtheta

    results = {}
    for name, run in runs.items():
        c = run['config']
        x_centers, y_centers = grid_from_config(c)
        Eu, Ev = relative_l2_error_series(run['u'], run['v'], x_centers, y_centers,
                                           ref_cu, ref_cv, ref_x, ref_y, r_c, x_c, y_c)
        phi_est = phase_series(run['Fy'])
        phi_true = phase_series(Cl_proxy_true)
        dphi = wrap(phi_est - phi_true)
        results[name] = dict(Eu=Eu, Ev=Ev, dphi=dphi)
        print('%-10s: mean E_u=%.4f mean E_v=%.4f  mean|dphi|=%.4f rad  final|dphi|=%.4f rad'
              % (name, Eu.mean(), Ev.mean(), np.abs(dphi).mean(), abs(dphi[-1])))

    out_path = os.path.join(EXP_DIR, 'stage_f_evaluation.npz')
    np.savez_compressed(out_path, **{
        f'{name}_{k}': v for name, r in results.items() for k, v in r.items()
    })
    print('Wrote %s' % out_path)

    # ---- verdict ----
    print()
    print('=' * 70)
    print('STAGE F VERDICT')
    print('=' * 70)
    eu_enkf, eu_free = results['enkf']['Eu'].mean(), results['free_run']['Eu'].mean()
    ev_enkf, ev_free = results['enkf']['Ev'].mean(), results['free_run']['Ev'].mean()
    dphi_enkf_final = abs(results['enkf']['dphi'][-1])
    dphi_free_final = abs(results['free_run']['dphi'][-1])
    dphi_shuf_final = abs(results['shuffled']['dphi'][-1])
    print('E_u: EnKF=%.4f  free-run=%.4f  (EnKF better: %s)' % (eu_enkf, eu_free, eu_enkf < eu_free))
    print('E_v: EnKF=%.4f  free-run=%.4f  (EnKF better: %s)' % (ev_enkf, ev_free, ev_enkf < ev_free))
    print('final |dphi|: EnKF=%.4f  free-run=%.4f  shuffled=%.4f rad' %
          (dphi_enkf_final, dphi_free_final, dphi_shuf_final))
    print('=' * 70)

    # ---- figures ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t = np.arange(201) * 0.1
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    colors = dict(free_run='C0', enkf='C1', shuffled='C2')
    for name in ['free_run', 'enkf', 'shuffled']:
        axs[0].plot(t, results[name]['Eu'], label=name, color=colors[name])
        axs[1].plot(t, results[name]['Ev'], label=name, color=colors[name])
        axs[2].plot(t, results[name]['dphi'], label=name, color=colors[name])
    axs[0].set_title('E_u(t)'); axs[0].set_xlabel('t'); axs[0].legend()
    axs[1].set_title('E_v(t)'); axs[1].set_xlabel('t'); axs[1].legend()
    axs[2].set_title('phase error dphi(t) [rad]'); axs[2].set_xlabel('t'); axs[2].legend()
    axs[2].axhline(0, color='gray', ls=':')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'stage_f_evaluation.png'), dpi=130)
    print('Figure written to figures/stage_f_evaluation.png')


if __name__ == '__main__':
    main()
