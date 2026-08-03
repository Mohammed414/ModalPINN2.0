# -*- coding: utf-8 -*-
"""
Phase 1 of the Lighthill boundary-vorticity-flux (BVF) plan (see bvf.md).

Builds the measured target g(theta, t) = (1/R) d p/d theta at the cylinder
wall, from the same pressure taps used by --PressureOnly training. Offline,
numpy-only: no TensorFlow training or GPU involved.

Pipeline: per-tap temporal harmonic fit (k=0,1,2) -> per-mode azimuthal
Fourier fit across taps -> analytic d/dtheta -> reassembled time-domain
target on a dense wall grid.
"""

import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')  # must happen before Load_train_data_desync's unconditional `import matplotlib.pyplot`

_HERE = os.path.dirname(os.path.abspath(__file__))
# Insert src/ first, then _HERE (src/pressure_only) on top of it, so that
# for any module name present in both (e.g. Load_train_data_desync.py, which
# exists in both as different files), the pressure_only copy wins - src/ is
# only there to supply text_flow.py / reactions_process.py, which pressure_only
# doesn't duplicate.
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, _HERE)
try:
    import Load_train_data_desync as ltd
except ImportError:
    # Local machine missing TensorFlow and/or running a scipy new enough to
    # have dropped scipy.integrate.trapz - fall back to harmless dev stubs
    # (see _local_dev_stubs/) so the real tap-extraction logic still runs.
    sys.path.insert(0, os.path.join(_HERE, '_local_dev_stubs'))
    import Load_train_data_desync as ltd

RE = 100.
R_C = 0.5
OMEGA_0 = 1.036
KMAX = 2


def geom_default():
    Lxmin, Lxmax = -4., 8.
    Lymin, Lymax = -4., 4.
    x_c, y_c, r_c = 0., 0., R_C
    return (Lxmin, Lxmax, Lymin, Lymax, x_c, y_c, r_c)


def temporal_harmonic_fit(times, p_t, omega_0, kmax=KMAX):
    """
    Least-squares fit of p(t) ~ a0 + sum_k [ak cos(k w0 t) + bk sin(k w0 t)].
    Returns {k: complex p_hat_k} matching NN_functions.NN_time_p's convention
    (p(t) = Re{sum_k p_hat_k * exp(+i k w0 t)}), plus fit R^2.
    """
    cols = [np.ones_like(times)]
    for k in range(1, kmax + 1):
        cols.append(np.cos(k * omega_0 * times))
        cols.append(np.sin(k * omega_0 * times))
    A = np.stack(cols, axis=1)
    coeffs, _, _, _ = np.linalg.lstsq(A, p_t, rcond=None)
    pred = A @ coeffs
    ss_res = np.sum((p_t - pred) ** 2)
    ss_tot = np.sum((p_t - np.mean(p_t)) ** 2)
    r2 = 1. - ss_res / ss_tot if ss_tot > 0 else 1.0

    p_hat = {0: complex(coeffs[0], 0.)}
    for k in range(1, kmax + 1):
        a_k = coeffs[1 + 2 * (k - 1)]
        b_k = coeffs[2 + 2 * (k - 1)]
        # Re{(a - i b) e^{i k w0 t}} = a cos(k w0 t) + b sin(k w0 t)
        p_hat[k] = complex(a_k, -b_k)
    return p_hat, r2


def azimuthal_fourier_fit(thetas, values, M=8):
    """Least-squares fit of complex `values(thetas)` ~ sum_{m=-M}^{M} c_m e^{i m theta}."""
    ms = np.arange(-M, M + 1)
    A = np.exp(1j * np.outer(thetas, ms))
    c, _, _, _ = np.linalg.lstsq(A, values, rcond=None)
    return ms, c


def eval_azimuthal(ms, c, theta):
    return np.sum(c[None, :] * np.exp(1j * np.outer(theta, ms)), axis=1)


def eval_azimuthal_dtheta(ms, c, theta):
    return np.sum((1j * ms)[None, :] * c[None, :] * np.exp(1j * np.outer(theta, ms)), axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--DataFile', type=str, default='Data/fixed_cylinder_atRe100')
    parser.add_argument('--NTaps', type=int, default=32)
    parser.add_argument('--Seed', type=int, default=0)
    parser.add_argument('--stdNoise', type=float, default=0.0)
    parser.add_argument('--M', type=int, default=8)
    parser.add_argument('--Ntheta', type=int, default=64)
    parser.add_argument('--Ntime', type=int, default=24)
    parser.add_argument('--Out', type=str, default=None)
    parser.add_argument('--NoPlot', action='store_true', default=False)
    args = parser.parse_args()

    np.random.seed(args.Seed)
    geom = geom_default()

    print('Reading real dataset and extracting %d cylinder taps...' % args.NTaps)
    times, data_cyl, _ = ltd.read_cut_simulation_data_exp_point_and_cylinder(
        args.DataFile, geom, n_taps=args.NTaps)
    x_cyl, y_cyl, _, _, p_cyl = data_cyl  # each [Nt, NTaps]

    x_tap = x_cyl[0, :]
    y_tap = y_cyl[0, :]
    theta_tap = np.arctan2(y_tap, x_tap)

    if args.stdNoise > 0:
        for j in range(p_cyl.shape[1]):
            p_cyl[:, j] = ltd.addNoise(p_cyl[:, j], args.stdNoise)
        print('Added Gaussian noise, std=%.4f' % args.stdNoise)

    p_hat_taps = {k: np.zeros(args.NTaps, dtype=complex) for k in range(KMAX + 1)}
    r2_taps = np.zeros(args.NTaps)
    for j in range(args.NTaps):
        p_hat, r2 = temporal_harmonic_fit(times, p_cyl[:, j], OMEGA_0, kmax=KMAX)
        for k in range(KMAX + 1):
            p_hat_taps[k][j] = p_hat[k]
        r2_taps[j] = r2

    print('Temporal harmonic fit R^2 per tap: min=%.5f mean=%.5f max=%.5f'
          % (r2_taps.min(), r2_taps.mean(), r2_taps.max()))

    azimuthal_ms = {}
    azimuthal_c = {}
    for k in range(KMAX + 1):
        ms, c = azimuthal_fourier_fit(theta_tap, p_hat_taps[k], M=args.M)
        azimuthal_ms[k] = ms
        azimuthal_c[k] = c
        dominant = ms[np.argsort(-np.abs(c))[:3]]
        print('Mode k=%d azimuthal fit: dominant |c_m| at m=%s' % (k, dominant.tolist()))

    theta_grid = np.linspace(0., 2 * np.pi, args.Ntheta, endpoint=False)
    T_period = 2 * np.pi / OMEGA_0
    t_grid = 400. + np.linspace(0., T_period, args.Ntime, endpoint=False)

    g_hat_k = {}
    for k in range(KMAX + 1):
        dphat_dtheta = eval_azimuthal_dtheta(azimuthal_ms[k], azimuthal_c[k], theta_grid)
        g_hat_k[k] = dphat_dtheta / R_C

    G = np.zeros((args.Ntheta, args.Ntime))
    for it, t in enumerate(t_grid):
        val = np.zeros(args.Ntheta, dtype=complex)
        for k in range(KMAX + 1):
            val = val + g_hat_k[k] * np.exp(1j * k * OMEGA_0 * t)
        G[:, it] = np.real(val)

    x_wall = R_C * np.cos(theta_grid)
    y_wall = R_C * np.sin(theta_grid)

    out_path = args.Out or ('bvf_targets_Ntap%d_seed%d.npz' % (args.NTaps, args.Seed))
    np.savez(
        out_path,
        theta_grid=theta_grid, t_grid=t_grid, x_wall=x_wall, y_wall=y_wall, G=G,
        g_hat_k0=g_hat_k[0], g_hat_k1=g_hat_k[1], g_hat_k2=g_hat_k[2],
        theta_tap=theta_tap, r2_taps=r2_taps,
        p_hat_tap_k0=p_hat_taps[0], p_hat_tap_k1=p_hat_taps[1], p_hat_tap_k2=p_hat_taps[2],
        c_k0=azimuthal_c[0], c_k1=azimuthal_c[1], c_k2=azimuthal_c[2], ms=azimuthal_ms[0],
    )
    print('Saved targets to', out_path)

    if not args.NoPlot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, KMAX + 1, figsize=(15, 4))
        # Match arctan2's (-pi, pi] branch so the fitted curve and the tap
        # scatter points visually line up (the fit itself is periodic and
        # correct on either branch - this is purely for a legible plot).
        theta_dense = np.linspace(-np.pi, np.pi, 400)
        for k in range(KMAX + 1):
            phat_dense = eval_azimuthal(azimuthal_ms[k], azimuthal_c[k], theta_dense)
            axes[k].plot(theta_dense, phat_dense.real, label='Re fit')
            axes[k].plot(theta_dense, phat_dense.imag, label='Im fit')
            axes[k].scatter(theta_tap, p_hat_taps[k].real, marker='o', s=15, label='Re taps')
            axes[k].scatter(theta_tap, p_hat_taps[k].imag, marker='x', s=15, label='Im taps')
            axes[k].set_title('p_hat mode k=%d' % k)
            axes[k].set_xlabel('theta')
            axes[k].legend(fontsize=7)
        plt.tight_layout()
        plot_path = 'bvf_targets_diagnostic_Ntap%d_seed%d.png' % (args.NTaps, args.Seed)
        plt.savefig(plot_path)
        print('Saved diagnostic plot to', plot_path)


if __name__ == '__main__':
    main()
