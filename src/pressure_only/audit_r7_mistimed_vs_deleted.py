"""
R7 decisive experiment: does the ACTUAL ModalPINN training loss (k=0 harmonic
residual + k=1 control-volume residual, the two terms R5/R6 specifically added
to fight wake collapse - see "R5 measured best candidates plan.md" and
audit_r5_losses.py) score a MISTIMED wake (correct amplitude and spatial
wavenumber, wrong phase) worse, better, or about the same as a DELETED wake
(collapsed amplitude, like the actual R5 checkpoint)?

This is the follow-up the R7 memo (notebooks/R7_EKI_observability_memo.md)
named as "the cheapest decisive experiment" and flagged as unresolved by the
existing audit_r5_losses.py, which only ever compared TRUE vs the ACTUAL
CHECKPOINT (which conflates amplitude and phase error) - never TRUE vs a
clean, isolated MISTIMED-only candidate.

Reuses audit_r5_losses.py's already-audited numpy/scipy pipeline VERBATIM
(harmonic_fit_all_nodes, build_mode_interpolators, true_field_k0_residual,
true_field_cv1_residual, build_cv1_boxes, conv_mode_k_np/conv_deriv_k_np) -
nothing here reimplements the physics formulas. What's new is only:

1. Loading via the fast cached parser (enkf_pressure_only) instead of
   text_flow.read_flow's slow pure-Python reader (audit_r5_losses.py's own
   --Mode truefield already does the same harmonic fit + RBF build; this
   just gets there faster, same math).
2. A scaled/rotated interpolator wrapper: RBFInterpolator solves a LINEAR
   system for its weights, so it is exactly linear in the fitted values -
   RBFInterp(c*y) = c*RBFInterp(y) for any constant c (real or, applied to
   the real/imag parts separately as done here, complex). This means a
   "mistimed" (phase-rotated) or "deleted" (amplitude-scaled) candidate mode
   field can be built as a cheap POST-HOC linear combination of the
   already-fitted TRUE field's interpolator outputs - no RBF refit needed,
   which is what makes a full phase x amplitude sweep tractable locally
   (the RBF fit itself is the slow, few-minutes step; this sweep is not).
   Correctness of this shortcut is checked numerically below (factor=1+0j
   must reproduce the unmodified true-field residuals exactly).

Runs entirely locally: no TensorFlow, no Colab, no GPU. Only tests the
k=0/CV1 terms against the TRUE field's own spatial structure - it does NOT
run the actual trained network (see the printed recommendation at the end
for what a Colab follow-up would need to add).

Usage:
    python3 audit_r7_mistimed_vs_deleted.py
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')

sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))
import audit_r5_losses as audit  # noqa: E402
from fast_flow_parser import load_flow  # noqa: E402
FLOW_CACHE = os.path.join(HERE, '_flow_cache.npz')
RAW_FLOW = os.path.join(ROOT, 'data', 'fixed_cylinder_atRe100')

NMODES = audit.NMODES
OMEGA_0 = audit.OMEGA_0
RE = audit.RE
LXMIN, LXMAX, LYMIN, LYMAX = audit.LXMIN, audit.LXMAX, audit.LYMIN, audit.LYMAX
X_C, Y_C, R_C = audit.X_C, audit.Y_C, audit.R_C

T_PERIOD = 2 * np.pi / OMEGA_0
CHECKPOINT_AMP_RATIO = 0.057  # measured in R7 notebook Section 1.7, R5 checkpoint peak |v1| vs truth


def make_scaled_interp_dict(true_interps, factor_by_k, nmodes):
    """Wrap an already-fitted interps dict so eval_mode returns
    factor_by_k[k] * (the true field's own interpolated value) - exact,
    given RBFInterpolator's linearity, no refit."""
    scaled = {}
    for k in range(nmodes):
        re_interp, im_interp = true_interps[k]
        f = factor_by_k.get(k, 1.0 + 0j)

        def re_fn(pts, re_interp=re_interp, im_interp=im_interp, f=f):
            re = re_interp(pts)
            im = im_interp(pts) if im_interp is not None else np.zeros_like(re)
            return ((re + 1j * im) * f).real

        def im_fn(pts, re_interp=re_interp, im_interp=im_interp, f=f):
            re = re_interp(pts)
            im = im_interp(pts) if im_interp is not None else np.zeros_like(re)
            return ((re + 1j * im) * f).imag

        scaled[k] = (re_fn, im_fn)
    return scaled


def load_cfd_fast():
    if os.path.exists(FLOW_CACHE):
        print('Loading pre-parsed flow cache: %s' % FLOW_CACHE)
        d = np.load(FLOW_CACHE)
        times, X, Y, U, V, P = d['times'], d['X'], d['Y'], d['U'], d['V'], d['p']
    else:
        print('No cache found, parsing raw flow file (one-off, ~15s)...')
        _, _, times, X, Y, U, V, P = load_flow(RAW_FLOW, cache=FLOW_CACHE)
    in_box = (X > LXMIN) & (X < LXMAX) & (Y > LYMIN) & (Y < LYMAX)
    return times, X[in_box], Y[in_box], U[:, in_box].astype(np.float64), \
        V[:, in_box].astype(np.float64), P[:, in_box].astype(np.float64)


def main():
    t0 = time.time()
    times, nx, ny, Us, Vs, Ps = load_cfd_fast()
    print('Loaded %d nodes, %d snapshots in %.1fs' % (len(nx), len(times), time.time() - t0))

    print('Fitting %d-mode harmonics...' % NMODES)
    coeff_u = audit.harmonic_fit_all_nodes(times, Us, OMEGA_0, NMODES)
    coeff_v = audit.harmonic_fit_all_nodes(times, Vs, OMEGA_0, NMODES)
    coeff_p = audit.harmonic_fit_all_nodes(times, Ps, OMEGA_0, NMODES)

    print('Building local RBF interpolants (slow step, a few minutes)...')
    t0 = time.time()
    interps_u = audit.build_mode_interpolators(nx, ny, coeff_u, NMODES)
    interps_v = audit.build_mode_interpolators(nx, ny, coeff_v, NMODES)
    interps_p = audit.build_mode_interpolators(nx, ny, coeff_p, NMODES)
    print('done in %.1fs' % (time.time() - t0))

    # sample points: identical construction/seed to audit_r5_losses.py's own
    # --Mode truefield, so results are directly comparable to its printed
    # true-field baseline.
    rng = np.random.RandomState(0)
    n_sample = 300
    sx = rng.uniform(LXMIN + 0.5, LXMAX - 0.5, n_sample)
    sy = rng.uniform(LYMIN + 0.5, LYMAX - 0.5, n_sample)
    r = np.sqrt((sx - X_C) ** 2 + (sy - Y_C) ** 2)
    keep = r > 1.5 * R_C
    sx, sy = sx[keep], sy[keep]
    boxes = audit.build_cv1_boxes()

    # 32 taps, for a tap-fit-error axis alongside the physics residuals
    taps = np.load(os.path.join(ROOT, 'data', 'sensor_indices', 'taps_32.npz'))
    tap_x, tap_y, tap_t, tap_p_true = taps['x'], taps['y'], taps['times'], taps['pressure']

    def tap_fit_rmse(interps_u_s, interps_v_s, interps_p_s):
        # only p matters for PressureOnly measurement loss
        pred = np.zeros_like(tap_p_true)
        for k in range(NMODES):
            ck = audit.eval_mode(interps_p_s, k, tap_x, tap_y)  # (Ntap,)
            pred += np.real(ck[None, :] * np.exp(1j * k * OMEGA_0 * (tap_t - tap_t[0]))[:, None])
        return float(np.sqrt(np.mean((pred - tap_p_true) ** 2)))

    def evaluate(label, factor_by_k):
        iu = make_scaled_interp_dict(interps_u, factor_by_k, NMODES)
        iv = make_scaled_interp_dict(interps_v, factor_by_k, NMODES)
        ip = make_scaled_interp_dict(interps_p, factor_by_k, NMODES)
        res_k0, *_ = audit.true_field_k0_residual(iu, iv, ip, sx, sy, NMODES, OMEGA_0, RE)
        Rx, Ry = audit.true_field_cv1_residual(iu, iv, ip, boxes[0], NMODES, OMEGA_0, RE)
        cv1_mag2 = abs(Rx) ** 2 + abs(Ry) ** 2
        rmse = tap_fit_rmse(iu, iv, ip)
        print('%-32s  k0_mean=%.4e  cv1=%.4e  tap_rmse=%.4e' % (label, res_k0.mean(), cv1_mag2, rmse))
        return res_k0.mean(), cv1_mag2, rmse

    print()
    print('=' * 90)
    print('SANITY CHECK: factor=1 must reproduce the unmodified true field exactly')
    print('=' * 90)
    k0_true, cv1_true, rmse_true = evaluate('TRUE (factor=1, sanity)', {0: 1 + 0j, 1: 1 + 0j, 2: 1 + 0j})

    print()
    print('=' * 90)
    print('MISTIMED sweep: correct amplitude + wavenumber, phase-shifted by dt (fraction of period T=%.3f)' % T_PERIOD)
    print('=' * 90)
    mistimed_results = {}
    for frac in (0.0, 0.125, 0.25, 0.375, 0.5):
        dt = frac * T_PERIOD
        factor = {0: 1 + 0j, 1: np.exp(-1j * OMEGA_0 * dt), 2: np.exp(-2j * OMEGA_0 * dt)}
        mistimed_results[frac] = evaluate('mistimed dt=%.3f (%.3fT)' % (dt, frac), factor)

    print()
    print('=' * 90)
    print('DELETED sweep: amplitude scaled toward zero, wavenumber/phase unchanged')
    print('=' * 90)
    deleted_results = {}
    for scale in (1.0, 0.5, 0.2, CHECKPOINT_AMP_RATIO, 0.02, 0.0):
        factor = {0: 1 + 0j, 1: scale + 0j, 2: scale + 0j}
        deleted_results[scale] = evaluate('deleted scale=%.3f%s' % (scale, ' (actual R5 ckpt)' if scale == CHECKPOINT_AMP_RATIO else ''), factor)

    print()
    print('=' * 90)
    print('VERDICT')
    print('=' * 90)
    worst_mistimed_k0 = max(v[0] for v in mistimed_results.values())
    best_deleted_k0 = min(v[0] for v in deleted_results.values() if v[0] > 0)
    ckpt_deleted_k0 = deleted_results[CHECKPOINT_AMP_RATIO][0]
    worst_mistimed_cv1 = max(v[1] for v in mistimed_results.values())
    ckpt_deleted_cv1 = deleted_results[CHECKPOINT_AMP_RATIO][1]
    fully_deleted_cv1 = deleted_results[0.0][1]

    print('k=0 residual: true=%.4e, mistimed range=[%.4e, %.4e], deleted-at-ckpt-ratio=%.4e, fully-deleted=%.4e'
          % (k0_true, min(v[0] for v in mistimed_results.values()), worst_mistimed_k0,
             ckpt_deleted_k0, deleted_results[0.0][0]))
    print('cv1 residual: true=%.4e, mistimed range=[%.4e, %.4e], deleted-at-ckpt-ratio=%.4e, fully-deleted=%.4e'
          % (cv1_true, min(v[1] for v in mistimed_results.values()), worst_mistimed_cv1,
             ckpt_deleted_cv1, fully_deleted_cv1))
    print()
    if ckpt_deleted_k0 < worst_mistimed_k0 or ckpt_deleted_cv1 < worst_mistimed_cv1:
        print('-> AT LEAST ONE of the two anti-collapse loss terms scores a checkpoint-amplitude')
        print('   DELETED wake as good as or BETTER than a worst-case MISTIMED (full-amplitude,')
        print('   wrong-phase) wake. This would mean the loss can still prefer collapse over a')
        print('   real-but-mistimed wake, even with K0/CV1 active -- consistent with why R5/R6')
        print('   raising LambdaK0 may not fix the underlying problem.')
    else:
        print('-> Both anti-collapse terms score every tested MISTIMED candidate as better than a')
        print('   checkpoint-amplitude DELETED one. The loss does NOT prefer this kind of collapse')
        print('   once K0/CV1 are active -- the collapse mechanism must lie elsewhere (e.g. in how')
        print('   L-BFGS actually navigates the combined, weighted loss landscape during training,')
        print('   not in these terms scoring collapse as globally better).')
    print()
    print('Tap-fit RMSE (PressureOnly measurement loss) at the SAME candidates, for context:')
    print('  true=%.4e  mistimed(dt=T/2)=%.4e  deleted(ckpt ratio)=%.4e  deleted(0)=%.4e'
          % (rmse_true, mistimed_results[0.5][2], deleted_results[CHECKPOINT_AMP_RATIO][2], deleted_results[0.0][2]))


if __name__ == '__main__':
    main()
