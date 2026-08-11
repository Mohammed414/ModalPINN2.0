"""
STAGE A (data prep): build the withheld reference-truth datasets from the raw
CFD flow file and the existing tap-index files.

This is the ONLY script in the whole enkf_pressure_only/ project that is
allowed to read data/fixed_cylinder_atRe100 (the full CFD field) directly.
Everything else -- the estimator, the NS forward model, the EnKF -- must
only ever see the "estimator-facing" tap dataset produced here
(enkf_pressure_only/data/tap_observations.npz), never the truth files.

Produces two output files under enkf_pressure_only/data/:

1. tap_observations.npz  (small, committed to git)
   Keys: tap_x, tap_y, tap_times, tap_p            -- for each of NTaps in {4,8,16,32}
         Re, r_c, x_c, y_c, domain, omega_0
   This is the ONLY file the estimator (STAGE B onward) is permitted to load.

2. reference_truth_modal.npz  (small, committed to git)
   Keys: gx, gy                                     -- regular evaluation grid
         Mtrue_u0/u1/u2, Mtrue_v0/v1/v2, Mtrue_p0/p1/p2  -- 3-mode (k=0,1,2)
           harmonic least-squares fit of the raw CFD field at omega_0,
           interpolated onto (gx, gy). Mode 0 is real; modes 1,2 are complex
           (matching ModalPINN's own Re{sum_k f_hat_k exp(i k omega_0 t)}
           convention, so f(t) = f0 + Re[f1 exp(i w0 t)] + Re[f2 exp(i 2 w0 t)]).
         Re, r_c, x_c, y_c, domain, omega_0, omega_0_fit_note

3. reference_truth_full.npz  (large, GITIGNORED, regenerate on demand)
   Keys: ref_x, ref_y, ref_times, ref_cu, ref_cv, ref_cp
   Raw, un-truncated instantaneous CFD snapshots at the cropped mesh nodes,
   for Stage F metrics that must not be contaminated by the 3-mode
   truncation itself (E_u(t)/E_v(t), vorticity fields, recirculation
   length, wake amplitude at exact x-stations).

Usage:
    cd enkf_pressure_only/evaluation
    python3 build_reference_truth.py
"""
import os
import numpy as np
from scipy.interpolate import griddata

from _fast_flow_parser import load_flow

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
DATA_DIR = os.path.join(HERE, '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

RAW_FLOW_FILE = os.path.join(ROOT, 'data', 'fixed_cylinder_atRe100')
FLOW_CACHE = os.path.join(HERE, '_flow_cache.npz')  # gitignored, regenerable

TAP_FILES = {
    4: os.path.join(ROOT, 'data', 'sensor_indices', 'taps_04.npz'),
    8: os.path.join(ROOT, 'data', 'sensor_indices', 'taps_08.npz'),
    16: os.path.join(ROOT, 'data', 'sensor_indices', 'taps_16.npz'),
    32: os.path.join(ROOT, 'data', 'sensor_indices', 'taps_32.npz'),
}

# Matches src/pressure_only/ModalPINN_VortexShedding.py's geom exactly, so the
# EnKF observer trains/evaluates over the identical region of interest.
Lxmin, Lxmax = -4., 8.
Lymin, Lymax = -4., 4.
X_C, Y_C, R_C = 0., 0., 0.5
OMEGA_0 = 1.036

GRID_NX, GRID_NY = 161, 107  # ~0.075 spacing in both directions


def build_tap_observations():
    print('=== Building tap_observations.npz ===')
    out = dict(Re=100.0, r_c=R_C, x_c=X_C, y_c=Y_C,
               domain=np.array([Lxmin, Lxmax, Lymin, Lymax]), omega_0=OMEGA_0)
    for n_taps, path in TAP_FILES.items():
        d = np.load(path)
        x, y, times, pressure = d['x'], d['y'], d['times'], d['pressure']
        r = np.hypot(x, y)
        assert np.allclose(r, R_C, atol=1e-3), \
            'tap radius check failed for n_taps=%d: r range [%f, %f]' % (n_taps, r.min(), r.max())
        assert np.all(np.diff(times) > 0), 'tap times not monotonic for n_taps=%d' % n_taps
        out['tap_x_%d' % n_taps] = x
        out['tap_y_%d' % n_taps] = y
        out['tap_times_%d' % n_taps] = times
        out['tap_p_%d' % n_taps] = pressure
        print('  n_taps=%d: x/y on cylinder OK (r in [%.6f, %.6f]), '
              'times monotonic OK, pressure shape %s' % (n_taps, r.min(), r.max(), pressure.shape))
    out_path = os.path.join(DATA_DIR, 'tap_observations.npz')
    np.savez_compressed(out_path, **out)
    print('Wrote %s' % out_path)


def harmonic_fit(times, F, omega_0):
    """F: (Nt, Npts). Least-squares fit F(t) ~ a0 + a1 cos(wt) + b1 sin(wt)
    + a2 cos(2wt) + b2 sin(2wt), vectorised over all points at once.
    Returns f0 (Npts, real), f1 (Npts, complex), f2 (Npts, complex) such that
    F(t) ~= f0 + Re[f1 exp(i w t)] + Re[f2 exp(i 2 w t)]."""
    w = omega_0
    Phi = np.stack([
        np.ones_like(times),
        np.cos(w * times), np.sin(w * times),
        np.cos(2 * w * times), np.sin(2 * w * times),
    ], axis=1)  # (Nt, 5)
    coeffs, *_ = np.linalg.lstsq(Phi, F, rcond=None)  # (5, Npts)
    a0, a1, b1, a2, b2 = coeffs
    f0 = a0
    f1 = a1 - 1j * b1
    f2 = a2 - 1j * b2
    return f0, f1, f2


def build_reference_truth():
    print('=== Building reference_truth_modal.npz + reference_truth_full.npz ===')
    Re, Ur, times, X, Y, U, V, p = load_flow(RAW_FLOW_FILE, cache=FLOW_CACHE)
    print('Raw flow loaded: Re=%.0f Ur=%.1f Nt=%d N_nodes=%d' % (Re, Ur, len(times), len(X)))

    crop = (X > Lxmin) & (X < Lxmax) & (Y > Lymin) & (Y < Lymax)
    Xc, Yc = X[crop], Y[crop]
    Uc, Vc, Pc = U[:, crop], V[:, crop], p[:, crop]
    print('Cropped to region of interest: %d nodes' % len(Xc))

    # --- full (untruncated) reference, evaluation-only, gitignored ---
    full_path = os.path.join(DATA_DIR, 'reference_truth_full.npz')
    np.savez_compressed(full_path, ref_x=Xc, ref_y=Yc, ref_times=times,
                         ref_cu=Uc, ref_cv=Vc, ref_cp=Pc)
    print('Wrote %s (%.1f MB)' % (full_path, os.path.getsize(full_path) / 1e6))

    # --- 3-mode harmonic fit at native node locations ---
    print('Fitting 3-mode (k=0,1,2) harmonic model at omega_0=%.4f ...' % OMEGA_0)
    u0, u1, u2 = harmonic_fit(times, Uc, OMEGA_0)
    v0, v1, v2 = harmonic_fit(times, Vc, OMEGA_0)
    p0, p1, p2 = harmonic_fit(times, Pc, OMEGA_0)

    # sanity: reconstruction residual
    recon_u = u0[None, :] + (u1[None, :] * np.exp(1j * OMEGA_0 * times)[:, None]).real \
        + (u2[None, :] * np.exp(1j * 2 * OMEGA_0 * times)[:, None]).real
    resid = np.sqrt(np.mean((recon_u - Uc) ** 2)) / np.sqrt(np.mean(Uc ** 2))
    print('  3-mode reconstruction relative RMS residual (u): %.4f' % resid)

    # --- interpolate coefficient fields onto a regular grid ---
    gx = np.linspace(Lxmin, Lxmax, GRID_NX)
    gy = np.linspace(Lymin, Lymax, GRID_NY)
    GX, GY = np.meshgrid(gx, gy, indexing='xy')
    pts = np.stack([Xc, Yc], axis=1)
    grid_pts = np.stack([GX.ravel(), GY.ravel()], axis=1)

    # mask grid points inside the solid cylinder -> NaN (no fluid there)
    inside = np.hypot(GX - X_C, GY - Y_C) < R_C

    def to_grid(field_native, is_complex):
        if is_complex:
            re = griddata(pts, field_native.real, grid_pts, method='linear').reshape(GX.shape)
            im = griddata(pts, field_native.imag, grid_pts, method='linear').reshape(GX.shape)
            g = re + 1j * im
        else:
            g = griddata(pts, field_native, grid_pts, method='linear').reshape(GX.shape)
        g = g.astype(np.complex64 if is_complex else np.float32)
        g[inside] = np.nan
        return g

    out = dict(
        gx=gx, gy=gy,
        Mtrue_u0=to_grid(u0, False), Mtrue_u1=to_grid(u1, True), Mtrue_u2=to_grid(u2, True),
        Mtrue_v0=to_grid(v0, False), Mtrue_v1=to_grid(v1, True), Mtrue_v2=to_grid(v2, True),
        Mtrue_p0=to_grid(p0, False), Mtrue_p1=to_grid(p1, True), Mtrue_p2=to_grid(p2, True),
        Re=100.0, r_c=R_C, x_c=X_C, y_c=Y_C,
        domain=np.array([Lxmin, Lxmax, Lymin, Lymax]), omega_0=OMEGA_0,
    )
    modal_path = os.path.join(DATA_DIR, 'reference_truth_modal.npz')
    np.savez_compressed(modal_path, **out)
    print('Wrote %s (%.2f MB)' % (modal_path, os.path.getsize(modal_path) / 1e6))


if __name__ == '__main__':
    build_tap_observations()
    build_reference_truth()
    print('\nDone. tap_observations.npz is estimator-facing (permitted).')
    print('reference_truth_modal.npz and reference_truth_full.npz are evaluation-only (forbidden to the estimator).')
