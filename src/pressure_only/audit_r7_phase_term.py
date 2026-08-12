"""
Phase-0 audit for the proposed new loss term, "Loss_phase" (dispersion-
consistency): does it actually discriminate a mistimed/flat-phase wake from
a correct one, and - critically, since this is the entire point of adding
it - does its residual stay large (not vanish) as amplitude collapses,
unlike K0Loss/CV1Loss?

Physical basis (validated against the real CFD field before being proposed -
see chat record / PROJECT_LOG): for x/D >= ~4, the true field's local mode-1
wavenumber k_x = Im[conj(q1)*dq1/dx]/|q1|^2 tracks -c*omega_0/u0(x,y) (u0 =
the mean streamwise velocity) with c = 0.891 +/- 0.021 - a tight, physically
sensible convection-deficit factor. Breaks down inside the recirculation
bubble (x < ~3), where u0 changes sign and "convection speed" isn't
well-defined - windowed out below for exactly that reason.

Formula (self-referential - only ever reads the network's OWN k=0 and k=1
modes, never privileged full-field truth):
    k_x_est(x,y)    = Im[conj(q1)*dq1/dx] / (|q1|^2 + eps)
    k_x_target(x,y) = -c * omega_0 / u0(x,y)      (c: single fitted/learnable scalar)
    r_phase          = k_x_est - k_x_target
    Loss_phase        = mean(r_phase^2) over windowed wake sample points

Three checks, all local/numpy, no TF:
  1. TRUE field (via the same RBF-interpolant machinery audit_r5_losses.py
     already uses): confirms the formula recovers a sensible c and a low
     residual on the field it was calibrated against.
  2. Synthetic MISTIMED and DELETED candidates (reusing
     audit_r7_mistimed_vs_deleted.py's linear-interpolator-scaling trick):
     does Loss_phase stay bounded/high as amplitude -> 0 (the property
     K0/CV1 do NOT have), and does it flag a flat-phase MISTIMED field even
     at full amplitude?
  3. THE ACTUAL R5 CHECKPOINT's own real forward pass (numpy MLP, matching
     the same complex tanh-network math NN_functions.py/out_nn_modes_uv
     uses, and the same reconstruction R7's notebook already validated at
     0.9472 unsteady tap-pressure correlation) - does the term that would
     have been in R5's loss actually flag the specific failure R5 produced?
     This is the most decisive check: not "would this term catch a made-up
     bad case" but "would it have caught THIS one."
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'enkf_pressure_only', 'evaluation'))
import audit_r5_losses as audit  # noqa: E402
from audit_r7_mistimed_vs_deleted import make_scaled_interp_dict  # noqa: E402

OMEGA_0 = audit.OMEGA_0
LXMIN, LXMAX, LYMIN, LYMAX = audit.LXMIN, audit.LXMAX, audit.LYMIN, audit.LYMAX
X_C, Y_C, R_C = audit.X_C, audit.Y_C, audit.R_C
GEOM = [LXMIN, LXMAX, LYMIN, LYMAX, X_C, Y_C, R_C]

X_WINDOW_MIN = 3.0  # exclude the recirculation bubble, per the validated breakdown region
FLOW_CACHE = os.path.join(ROOT, 'enkf_pressure_only', 'evaluation', '_flow_cache.npz')
CKPT_PATH = os.path.join(
    ROOT, 'runs', 'R5_extracted',
    'R5_k0_cv1_pressure_only_Re100_Nm3_Nint50000_Nmes5000_WL25_Ntap32_FSBC_FIBC_BVF_lam0p1_K0_0p025_CV1_0p0075_seed0_20260806',
    'DNN2_75_75_3_tanh.pickle')


# =============================================================================
# The proposed term itself (numpy, works on any (u0, q1, dq1dx) triple)
# =============================================================================

def fit_c(u0, q1, dq1dx, eps):
    """Least-squares fit of the single convection-deficit scalar c from
    k_x_est ~= -c*omega_0/u0, restricted to the caller's already-windowed
    sample points. eps MUST be a fixed absolute floor computed once from the
    TRUE field's own scale (see calibrate_eps) -- NOT recomputed per-call
    from whatever candidate is being evaluated. An earlier version did the
    latter (eps = frac * CURRENT q1's own max) and it was a real bug: since
    q1 -> s*q1 makes BOTH the numerator and eps scale as s^2, the ratio came
    out perfectly scale-invariant, silently reproducing the exact
    amplitude-blindness this term exists to avoid (confirmed empirically:
    Loss_phase was bit-identical across the entire amplitude sweep before
    this fix)."""
    kx_est = np.imag(np.conj(q1) * dq1dx) / (np.abs(q1) ** 2 + eps)
    kx_basis = -OMEGA_0 / u0
    c = np.sum(kx_est * kx_basis) / np.sum(kx_basis ** 2)
    return c, kx_est, kx_basis


def calibrate_eps(q1_true, frac):
    return (frac * np.median(np.abs(q1_true))) ** 2


def loss_phase(u0, q1, dq1dx, c, eps):
    kx_est = np.imag(np.conj(q1) * dq1dx) / (np.abs(q1) ** 2 + eps)
    kx_target = -c * OMEGA_0 / u0
    r = kx_est - kx_target
    return float(np.mean(r ** 2)), kx_est, kx_target


# =============================================================================
# 1 & 2: true field + synthetic candidates, via the RBF machinery
# =============================================================================

def run_true_and_synthetic():
    print('=' * 90)
    print('PART 1: true field calibration + synthetic mistimed/deleted candidates')
    print('=' * 90)
    d = np.load(FLOW_CACHE)
    times, X, Y, U, V = d['times'], d['X'], d['Y'], d['U'], d['V']
    in_box = (X > LXMIN) & (X < LXMAX) & (Y > LYMIN) & (Y < LYMAX)
    X, Y, U, V = X[in_box], Y[in_box], U[:, in_box].astype(np.float64), V[:, in_box].astype(np.float64)

    coeff_u = audit.harmonic_fit_all_nodes(times, U, OMEGA_0, audit.NMODES)
    coeff_v = audit.harmonic_fit_all_nodes(times, V, OMEGA_0, audit.NMODES)
    interps_u = audit.build_mode_interpolators(X, Y, coeff_u, audit.NMODES)
    interps_v = audit.build_mode_interpolators(X, Y, coeff_v, audit.NMODES)

    rng = np.random.RandomState(1)
    n = 400
    sx = rng.uniform(X_WINDOW_MIN, LXMAX - 0.3, n)
    sy = rng.uniform(-2.0, 2.0, n)

    def fields(iu, iv):
        u0 = audit.eval_mode(iu, 0, sx, sy).real
        q1 = audit.eval_mode(iv, 1, sx, sy)
        dq1dx = audit.deriv(iv, 1, sx, sy, 'x')
        return u0, q1, dq1dx

    q1_true_for_eps = audit.eval_mode(interps_v, 1, sx, sy)
    u0_true, q1_true_c, dq1_true_c = fields(interps_u, interps_v)
    iv_ckpt_scale = make_scaled_interp_dict(interps_v, {0: 1 + 0j, 1: 0.057 + 0j, 2: 0.057 + 0j}, audit.NMODES)
    u0_057, q1_057, dq1_057 = fields(interps_u, iv_ckpt_scale)

    print('eps calibration sweep (fixed floor = (frac * median|q1_true|)^2):')
    for frac in (0.01, 0.05, 0.1, 0.2, 0.4):
        eps_try = calibrate_eps(q1_true_for_eps, frac)
        c_try, _, _ = fit_c(u0_true, q1_true_c, dq1_true_c, eps_try)
        L_true_try, _, _ = loss_phase(u0_true, q1_true_c, dq1_true_c, c_try, eps_try)
        L_057, _, _ = loss_phase(u0_057, q1_057, dq1_057, c_try, eps_try)
        print('  frac=%.2f  eps=%.3e  c=%.4f  L(true)=%.4e  L(scale=0.057)/L(true)=%.2fx'
              % (frac, eps_try, c_try, L_true_try, L_057 / L_true_try))

    EPS_FRAC = 0.1
    eps = calibrate_eps(q1_true_for_eps, EPS_FRAC)
    print('\nUsing eps_frac=%.2f (eps=%.4e) for the rest of this run.' % (EPS_FRAC, eps))

    u0, q1, dq1dx = fields(interps_u, interps_v)
    c_fit, kx_est_true, kx_basis_true = fit_c(u0, q1, dq1dx, eps)
    L_true, _, _ = loss_phase(u0, q1, dq1dx, c_fit, eps)
    print('Fitted c (should be close to the 0.891 found in the earlier x=4-7 centerline check): %.4f' % c_fit)
    print('Loss_phase(TRUE field, using its own fitted c) = %.6e  (this is the floor, by construction)' % L_true)

    print()
    print('TIMING sweep (correct amplitude+wavenumber, phase-shifted by dt --')
    print('expected to be INVARIANT: a constant time-shift multiplies q1 by a constant complex')
    print('factor, which does not change how fast phase varies WITH X (the spatial wavenumber) --')
    print('this measures d(phase)/dx, a genuinely different quantity from "when do peaks occur".')
    print('Confirms this term is not meant to (and correctly does not) duplicate the tap loss'
          ' role of fixing absolute timing -- only the tap loss does that.):')
    T = 2 * np.pi / OMEGA_0
    for frac in (0.0, 0.125, 0.25, 0.375, 0.5):
        dt = frac * T
        factor = {0: 1 + 0j, 1: np.exp(-1j * OMEGA_0 * dt), 2: np.exp(-2j * OMEGA_0 * dt)}
        iv_s = make_scaled_interp_dict(interps_v, factor, audit.NMODES)
        u0s, q1s, dq1s = fields(interps_u, iv_s)
        L, _, _ = loss_phase(u0s, q1s, dq1s, c_fit, eps)
        print('  dt=%.3f (%.3fT)  Loss_phase=%.6e  (ratio to true: %.2fx)' % (dt, frac, L, L / L_true))

    print()
    print('DELETED sweep (amplitude scaled toward zero, wavenumber unchanged) -- with the eps bug fixed,')
    print('this should now clearly RISE as scale shrinks, not stay flat:')
    for scale in (1.0, 0.5, 0.2, 0.057, 0.02, 0.002, 0.0):
        factor = {0: 1 + 0j, 1: scale + 0j, 2: scale + 0j}
        iv_s = make_scaled_interp_dict(interps_v, factor, audit.NMODES)
        u0s, q1s, dq1s = fields(interps_u, iv_s)
        L, _, _ = loss_phase(u0s, q1s, dq1s, c_fit, eps)
        tag = '  <- checkpoint-scale' if scale == 0.057 else ''
        print('  scale=%.3f  Loss_phase=%.6e  (ratio to true: %.2fx)%s' % (scale, L, L / L_true, tag))

    print()
    print('WRONG-WAVENUMBER sweep (full amplitude preserved EXACTLY, spatial phase gradient')
    print('compressed by beta -- beta=0.127 matches the actual checkpoint wavenumber ratio')
    print('0.201/1.581 found in the R7 notebook -- this isolates the wavenumber axis from')
    print('the amplitude axis, which the deleted/timing sweeps above cannot do on their own):')
    h = 1e-3

    def phase_compressed_fields(beta):
        q1r = audit.eval_mode(interps_v, 1, sx, sy)
        r, phi = np.abs(q1r), np.angle(q1r)
        q1c = r * np.exp(1j * beta * phi)
        q1p_raw = audit.eval_mode(interps_v, 1, sx + h, sy)
        q1m_raw = audit.eval_mode(interps_v, 1, sx - h, sy)
        q1p = np.abs(q1p_raw) * np.exp(1j * beta * np.angle(q1p_raw))
        q1m = np.abs(q1m_raw) * np.exp(1j * beta * np.angle(q1m_raw))
        dq1c = (q1p - q1m) / (2 * h)
        u0c = audit.eval_mode(interps_u, 0, sx, sy).real
        return u0c, q1c, dq1c

    for beta in (1.0, 0.5, 0.25, 0.127, 0.0):
        u0b, q1b, dq1b = phase_compressed_fields(beta)
        L, kx_est_b, _ = loss_phase(u0b, q1b, dq1b, c_fit, eps)
        amp_check = np.median(np.abs(q1b)) / np.median(np.abs(q1))
        tag = '  <- matches actual R5 checkpoint ratio' if beta == 0.127 else ''
        print('  beta=%.3f  Loss_phase=%.6e  (ratio to true: %.2fx)  median|q1| ratio to true: %.3f (should be ~1.0)%s'
              % (beta, L, L / L_true, amp_check, tag))

    return c_fit, eps


# =============================================================================
# 3: the actual R5 checkpoint, numpy forward pass
# =============================================================================

def neural_net_np(Xin, Wl, bl):
    H = Xin
    for l in range(len(Wl) - 1):
        H = np.tanh(H @ Wl[l] + bl[l])
    return H @ Wl[-1] + bl[-1]


def f_BC5(x, y, fact=5.):
    r = np.hypot(x - X_C, y - Y_C) - R_C
    return np.tanh(fact * r)


def pinn_modes(x, y, Wl, bl):
    """Numpy twin of NN_functions.out_nn_modes_uv/out_nn_modes_p's core
    (f_BC5 masking only - freestream_target/damp_fluctuations only affect
    x < -2, kill_k0_imag only affects Im(mode 0), neither matters for this
    term, which only ever reads Re(mode 0) and mode 1 in the wake x>=3;
    already the same simplification R7's notebook validated at 0.9472
    unsteady tap-pressure correlation)."""
    X = np.stack([x, y], 1).astype(np.complex64)
    return neural_net_np(X, Wl, bl) * f_BC5(x, y)[:, None]


def run_checkpoint():
    print()
    print('=' * 90)
    print('PART 2: the ACTUAL R5 checkpoint - would Loss_phase have caught this failure?')
    print('=' * 90)
    if not os.path.exists(CKPT_PATH):
        print('Checkpoint not found at %s, skipping.' % CKPT_PATH)
        return
    import pickle
    with open(CKPT_PATH, 'rb') as f:
        Wu, bu, Wv, bv, Wp, bp = pickle.load(f)
    print('Loaded checkpoint, layer shapes: %s' % [np.shape(w) for w in Wu])

    rng = np.random.RandomState(1)
    n = 400
    sx = rng.uniform(X_WINDOW_MIN, LXMAX - 0.3, n).astype(np.float32)
    sy = rng.uniform(-2.0, 2.0, n).astype(np.float32)

    h = 1e-3
    u0 = pinn_modes(sx, sy, Wu, bu)[:, 0].real
    q1 = pinn_modes(sx, sy, Wv, bv)[:, 1]
    q1_plus = pinn_modes(sx + h, sy, Wv, bv)[:, 1]
    q1_minus = pinn_modes(sx - h, sy, Wv, bv)[:, 1]
    dq1dx = (q1_plus - q1_minus) / (2 * h)

    print('checkpoint |u0| range: [%.4f, %.4f]  (near-zero u0 in-window would be a problem - checking)'
          % (np.abs(u0).min(), np.abs(u0).max()))
    print('checkpoint |q1| range: [%.4e, %.4e]  (R7 found peak|v1|~0.037 vs true 0.652)'
          % (np.abs(q1).min(), np.abs(q1).max()))

    return u0, q1, dq1dx, sx, sy


def main():
    c_fit, eps = run_true_and_synthetic()
    result = run_checkpoint()
    if result is None:
        return
    u0, q1, dq1dx, sx, sy = result
    # NOTE: eps was calibrated from the TRUE field's median|q1| in Part 1 and is reused
    # unchanged here - the whole point is that the floor is fixed, not re-derived from
    # whatever (possibly collapsed) field is currently being scored.
    L_ckpt, kx_est, kx_target = loss_phase(u0, q1, dq1dx, c_fit, eps)

    print()
    print('Loss_phase(R5 CHECKPOINT, true-field-calibrated c=%.4f, fixed eps=%.4e) = %.6e'
          % (c_fit, eps, L_ckpt))
    print('median |k_x_est| on checkpoint: %.4f   median |k_x_target| (from ckpt own u0): %.4f'
          % (np.median(np.abs(kx_est)), np.median(np.abs(kx_target))))
    print('(true field k_x at x=4-7 was ~1.2-1.6 in magnitude, per the earlier centerline check)')

    print()
    print('=' * 90)
    print('VERDICT')
    print('=' * 90)
    print('Same c and eps used for both Part 1 and the checkpoint, so Loss_phase(checkpoint) IS')
    print('directly comparable in scale to Loss_phase(true)=the Part 1 floor - compare directly.')


if __name__ == '__main__':
    main()
