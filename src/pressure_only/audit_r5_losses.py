# -*- coding: utf-8 -*-
"""
R5 Phase 0 audit (mandatory before any --K0Loss/--CV1Loss training run) - see
"R5 measured best candidates plan.md".

Two independent checks, run as two separate --Mode values so this script
never needs TensorFlow AND a modern scipy at the same time (the project's
pinned training environment uses scipy==1.3.2, which predates
scipy.interpolate.RBFInterpolator - added in scipy 1.7 - and is needed here
for scalable local interpolation over the CFD mesh's tens of thousands of
scattered nodes; --Mode checkpoint needs TF1.14 but no interpolation at all):

  --Mode truefield  (pure numpy/scipy, no TensorFlow, runs anywhere - the
      pinned Colab env's ancient scipy will NOT work here; run this on a
      normal modern-Python machine)
    Numerically evaluates the k=0 harmonic residual and the k=1
    control-volume residuals (four boxes - see CV1_X_UP/CV1_X_DOWN below)
    on the ACTUAL CFD data (not the network).
    The CFD mesh is unstructured, so this fits smooth local-RBF
    interpolants to each mode's scattered per-node harmonic-fit
    coefficients and differentiates those (central finite differences on
    the interpolant, not the raw scattered data) - validates every
    sign/factor/quadrature in the TF implementation before trusting it to
    shape a 9h GPU run.

  --Mode checkpoint  (needs TensorFlow 1.14 - run in the pinned conda env,
      e.g. via colab-cli, same as every other run in this project)
    Restores an actual trained model (e.g. R3's checkpoint) and evaluates
    the same two losses through the network via autodiff (no mesh, no
    interpolation needed). The plan's own framing: these two numbers (how
    much higher than the true-field values) are themselves a dissertation
    figure, independent of R5's eventual outcome. Also does the lambda
    calibration (Change 2/3's LambdaK0/LambdaCV1 + CV1_NORMALIZERS) and the
    two free diagnostics (wake-region loss vs truncation floor, mean-flow
    recirculation length).

Usage:
    python audit_r5_losses.py --Mode truefield --DataFile Data/fixed_cylinder_atRe100
    python audit_r5_losses.py --Mode checkpoint --RunDir <R3 output folder> --WidthLayer 25 --Nmodes 3
"""
import argparse
import glob
import os
import sys

import numpy as np

# Geometry / physics constants, matching ModalPINN_VortexShedding.py exactly
# (duplicated rather than imported - importing the training script would run
# its argparse/training side effects; see evaluate_regions.py for the same,
# already-established pattern in this codebase).
X_C, Y_C, R_C = 0., 0., 0.5
LXMIN, LXMAX, LYMIN, LYMAX = -4., 8., -4., 4.
GEOM = [LXMIN, LXMAX, LYMIN, LYMAX, X_C, Y_C, R_C]
OMEGA_0 = 1.036
RE = 100.
NMODES = 3  # k=0,1,2 - matches every run's --Nmodes 3 (see the plan's own Nmodes discrepancy note)
D = 2 * R_C

DEFAULT_DATA_FILE = 'Data/fixed_cylinder_atRe100'

# One k=1 control-volume box (the full-wake box) - MUST match
# ModalPINN_VortexShedding.py's CV1_X_UP/CV1_X_DOWN/CV1_YMIN/CV1_YMAX/
# CV1_N_FACE_PTS/CV1_N_AREA_X/Y exactly. Originally six (five paired + one
# full-wake box); progressively cut down to just the full-wake box - first
# for measured wrong-sign sensitivity in two boxes, then purely for
# --CV1Loss's memory cost (a fixed per-box graph-topology cost, confirmed
# independent of quadrature resolution) - see ModalPINN_VortexShedding.py's
# CV1_X_UP comment and PROJECT_LOG.md for the full numbers.
CV1_X_UP = [0.5]
CV1_X_DOWN = [6.]
CV1_YMIN, CV1_YMAX = -2., 2.
CV1_N_FACE_PTS = 64
CV1_N_AREA_X, CV1_N_AREA_Y = 32, 16


def trapz_nodes_weights(a, b, n):
    '''Uniform trapezoid quadrature nodes+weights on [a,b], n points.'''
    nodes = np.linspace(a, b, n)
    w = np.full(n, (b - a) / (n - 1))
    w[0] *= 0.5
    w[-1] *= 0.5
    return nodes, w


def build_cv1_boxes():
    '''numpy twin of ModalPINN_VortexShedding.py's _build_cv1_boxes - MUST
    stay in sync with it (same constants above, same construction).'''
    boxes = []
    for x_up, x_down in zip(CV1_X_UP, CV1_X_DOWN):
        y_face, wy_face = trapz_nodes_weights(CV1_YMIN, CV1_YMAX, CV1_N_FACE_PTS)
        x_face, wx_face = trapz_nodes_weights(x_up, x_down, CV1_N_FACE_PTS)
        xa, wxa = trapz_nodes_weights(x_up, x_down, CV1_N_AREA_X)
        ya, wya = trapz_nodes_weights(CV1_YMIN, CV1_YMAX, CV1_N_AREA_Y)
        Xa, Ya = np.meshgrid(xa, ya, indexing='ij')
        Wa = np.outer(wxa, wya)
        boxes.append(dict(
            left_x=np.full(CV1_N_FACE_PTS, x_up), left_y=y_face, left_w=wy_face, left_n=(-1., 0.),
            right_x=np.full(CV1_N_FACE_PTS, x_down), right_y=y_face, right_w=wy_face, right_n=(1., 0.),
            bottom_x=x_face, bottom_y=np.full(CV1_N_FACE_PTS, CV1_YMIN), bottom_w=wx_face, bottom_n=(0., -1.),
            top_x=x_face, top_y=np.full(CV1_N_FACE_PTS, CV1_YMAX), top_w=wx_face, top_n=(0., 1.),
            area_x=Xa.flatten(), area_y=Ya.flatten(), area_w=Wa.flatten(),
        ))
    return boxes


def conv_mode_k_np(a_list, b_list, k, nmodes):
    '''numpy twin of ModalPINN_VortexShedding.conv_mode_k (raw product, no
    derivative) - a_list,b_list: lists indexed by mode of same-shape complex
    ndarrays.'''
    direct = sum(a_list[l] * b_list[k - l] for l in range(k + 1))
    if k + 1 < nmodes:
        conj_a = sum(a_list[l] * np.conj(b_list[l - k]) for l in range(k + 1, nmodes))
        conj_b = sum(np.conj(a_list[l - k]) * b_list[l] for l in range(k + 1, nmodes))
        return direct + conj_a + conj_b
    return direct


def conv_deriv_k_np(a_list, ad_list, k, nmodes):
    '''numpy twin of the derivative-convolution pattern used by
    loss_int_mode_per_k's f_u_4a/4b + 5a/5b terms (one raw factor, one
    derivative factor) - a_list: all_u or all_v; ad_list: the matching
    derivative array (all_u_x, all_u_y, ...).'''
    total = sum(a_list[l] * ad_list[k - l] for l in range(k + 1))
    if k + 1 < nmodes:
        total = total + sum(a_list[l] * np.conj(ad_list[l - k]) for l in range(k + 1, nmodes))
        total = total + sum(np.conj(a_list[l - k]) * ad_list[l] for l in range(k + 1, nmodes))
    return total


# =============================================================================
# --Mode truefield
# =============================================================================

def harmonic_fit_all_nodes(times, values, omega_0, nmodes):
    '''
    Per-node harmonic least-squares fit: values[t,i] ~= Re(sum_k coeff[k,i] *
    exp(i*k*omega_0*t)), vectorized across all nodes i at once (same harmonic
    convention as bvf_targets.py's per-tap fit, generalized to every mesh
    node instead of just the pressure taps).
    Returns coeff : complex ndarray [nmodes, Nnodes].
    '''
    Nt, Nnodes = values.shape
    cols = [np.ones_like(times)]
    for k in range(1, nmodes):
        cols.append(np.cos(k * omega_0 * times))
        cols.append(np.sin(k * omega_0 * times))
    A = np.stack(cols, axis=1)
    coefs, _, _, _ = np.linalg.lstsq(A, values, rcond=None)
    out = np.zeros((nmodes, Nnodes), dtype=np.complex128)
    out[0, :] = coefs[0, :]
    for k in range(1, nmodes):
        a_k = coefs[1 + 2 * (k - 1), :]
        b_k = coefs[2 + 2 * (k - 1), :]
        # a_k*cos(kwt) + b_k*sin(kwt) = Re(c_k*exp(i*k*w*t)) with c_k = a_k - i*b_k
        out[k, :] = a_k - 1j * b_k
    return out


def build_mode_interpolators(nodes_x, nodes_y, coeff, nmodes):
    '''
    coeff : complex [nmodes, Nnodes]. Returns dict k -> (re_interp, im_interp)
    of scipy.interpolate.RBFInterpolator callables (local/neighbor-based, so
    this scales to tens of thousands of scattered CFD mesh nodes - a global
    RBF fit would be O(Nnodes^2) memory and infeasible at this size).
    '''
    from scipy.interpolate import RBFInterpolator
    pts = np.stack([nodes_x, nodes_y], axis=1)
    interps = {}
    for k in range(nmodes):
        re_interp = RBFInterpolator(pts, coeff[k, :].real, neighbors=50, kernel='thin_plate_spline')
        if k == 0:
            interps[k] = (re_interp, None)
        else:
            im_interp = RBFInterpolator(pts, coeff[k, :].imag, neighbors=50, kernel='thin_plate_spline')
            interps[k] = (re_interp, im_interp)
    return interps


def eval_mode(interps, k, x, y):
    '''x,y : 1D numpy arrays, same length. Returns complex ndarray, mode k.'''
    pts = np.stack([x, y], axis=1)
    re_interp, im_interp = interps[k]
    re = re_interp(pts)
    im = im_interp(pts) if im_interp is not None else np.zeros_like(re)
    return re + 1j * im


def deriv(interps, k, x, y, wrt, h=2e-3):
    if wrt == 'x':
        return (eval_mode(interps, k, x + h, y) - eval_mode(interps, k, x - h, y)) / (2 * h)
    else:
        return (eval_mode(interps, k, x, y + h) - eval_mode(interps, k, x, y - h)) / (2 * h)


def deriv2(interps, k, x, y, wrt, h=2e-3):
    c = eval_mode(interps, k, x, y)
    if wrt == 'xx':
        return (eval_mode(interps, k, x + h, y) - 2 * c + eval_mode(interps, k, x - h, y)) / h ** 2
    else:
        return (eval_mode(interps, k, x, y + h) - 2 * c + eval_mode(interps, k, x, y - h)) / h ** 2


def true_field_k0_residual(interps_u, interps_v, interps_p, sx, sy, nmodes, omega_0, Re):
    '''numpy twin of loss_int_mode_per_k's k=0 slice, evaluated on the
    interpolated true field at sample interior points sx,sy.
    Returns |f_u|^2+|f_v|^2+|div_u|^2 (ndarray, one value per sample point)
    plus the raw f_u,f_v,div_u (for diagnostic printing).'''
    all_u = [eval_mode(interps_u, k, sx, sy) for k in range(nmodes)]
    all_v = [eval_mode(interps_v, k, sx, sy) for k in range(nmodes)]
    all_u_x = [deriv(interps_u, k, sx, sy, 'x') for k in range(nmodes)]
    all_u_y = [deriv(interps_u, k, sx, sy, 'y') for k in range(nmodes)]
    all_v_x = [deriv(interps_v, k, sx, sy, 'x') for k in range(nmodes)]
    all_v_y = [deriv(interps_v, k, sx, sy, 'y') for k in range(nmodes)]
    all_u_xx = [deriv2(interps_u, k, sx, sy, 'xx') for k in range(nmodes)]
    all_u_yy = [deriv2(interps_u, k, sx, sy, 'yy') for k in range(nmodes)]
    all_v_xx = [deriv2(interps_v, k, sx, sy, 'xx') for k in range(nmodes)]
    all_v_yy = [deriv2(interps_v, k, sx, sy, 'yy') for k in range(nmodes)]
    all_p_x = [deriv(interps_p, k, sx, sy, 'x') for k in range(nmodes)]
    all_p_y = [deriv(interps_p, k, sx, sy, 'y') for k in range(nmodes)]

    k = 0
    f_u = 1j * k * omega_0 * all_u[k] + all_p_x[k] - (1. / Re) * (all_u_xx[k] + all_u_yy[k])
    f_u = f_u + conv_deriv_k_np(all_u, all_u_x, k, nmodes)
    f_u = f_u + conv_deriv_k_np(all_v, all_u_y, k, nmodes)

    f_v = 1j * k * omega_0 * all_v[k] + all_p_y[k] - (1. / Re) * (all_v_xx[k] + all_v_yy[k])
    f_v = f_v + conv_deriv_k_np(all_u, all_v_x, k, nmodes)
    f_v = f_v + conv_deriv_k_np(all_v, all_v_y, k, nmodes)

    div_u = all_u_x[k] + all_v_y[k]

    residual = np.abs(f_u) ** 2 + np.abs(f_v) ** 2 + np.abs(div_u) ** 2
    return residual, f_u, f_v, div_u


def true_field_cv1_residual(interps_u, interps_v, interps_p, box, nmodes, omega_0, Re):
    '''numpy twin of ModalPINN_VortexShedding._cv1_box_residual, evaluated on
    the interpolated true field. Returns (Rx, Ry) complex scalars.'''
    xa, ya, wa = box['area_x'], box['area_y'], box['area_w']
    u1_a = eval_mode(interps_u, 1, xa, ya)
    v1_a = eval_mode(interps_v, 1, xa, ya)
    Rx = 1j * omega_0 * np.sum(wa * u1_a)
    Ry = 1j * omega_0 * np.sum(wa * v1_a)

    for face in ['left', 'right', 'bottom', 'top']:
        xf, yf, wf = box[face + '_x'], box[face + '_y'], box[face + '_w']
        nx, ny = box[face + '_n']
        all_u = [eval_mode(interps_u, k, xf, yf) for k in range(nmodes)]
        all_v = [eval_mode(interps_v, k, xf, yf) for k in range(nmodes)]
        p1 = eval_mode(interps_p, 1, xf, yf)
        Qxx = conv_mode_k_np(all_u, all_u, 1, nmodes)
        Qxy = conv_mode_k_np(all_u, all_v, 1, nmodes)
        Qyy = conv_mode_k_np(all_v, all_v, 1, nmodes)
        u1_x = deriv(interps_u, 1, xf, yf, 'x')
        u1_y = deriv(interps_u, 1, xf, yf, 'y')
        v1_x = deriv(interps_v, 1, xf, yf, 'x')
        v1_y = deriv(interps_v, 1, xf, yf, 'y')

        flux_x = Qxx * nx + Qxy * ny
        flux_y = Qxy * nx + Qyy * ny
        visc_x = (1. / Re) * (2. * u1_x * nx + (u1_y + v1_x) * ny)
        visc_y = (1. / Re) * ((u1_y + v1_x) * nx + 2. * v1_y * ny)

        Rx = Rx + np.sum(wf * (flux_x + p1 * nx - visc_x))
        Ry = Ry + np.sum(wf * (flux_y + p1 * ny - visc_y))

    return Rx, Ry


def run_truefield(args):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from text_flow import read_flow  # noqa: E402

    print('Reading real dataset...')
    Re_data, Ur, times, nodes_X, nodes_Y, Us, Vs, Ps = read_flow(args.DataFile)
    nodes_x0, nodes_y0 = nodes_X[0, :], nodes_Y[0, :]
    dx = np.max(np.abs(nodes_X - nodes_X[0:1, :]))
    dy = np.max(np.abs(nodes_Y - nodes_Y[0:1, :]))
    assert dx < 1e-9 and dy < 1e-9, 'Mesh is not static across time.'

    in_box = ((nodes_x0 < LXMAX) & (nodes_x0 > LXMIN) & (nodes_y0 > LYMIN) & (nodes_y0 < LYMAX))
    idx = np.argwhere(in_box)[:, 0]
    nodes_x0, nodes_y0 = nodes_x0[idx], nodes_y0[idx]
    Us_c, Vs_c, Ps_c = Us[:, idx], Vs[:, idx], Ps[:, idx]
    print('Fitting harmonics at %d modes over %d nodes...' % (NMODES, len(idx)))

    coeff_u = harmonic_fit_all_nodes(times, Us_c, OMEGA_0, NMODES)
    coeff_v = harmonic_fit_all_nodes(times, Vs_c, OMEGA_0, NMODES)
    coeff_p = harmonic_fit_all_nodes(times, Ps_c, OMEGA_0, NMODES)

    # Fit quality check (same spirit as bvf_targets.py's per-tap R^2 print).
    recon_u = np.real(sum(coeff_u[k, :][None, :] * np.exp(1j * k * OMEGA_0 * times)[:, None] for k in range(NMODES)))
    ss_res = np.sum((Us_c - recon_u) ** 2)
    ss_tot = np.sum((Us_c - Us_c.mean()) ** 2)
    print('Harmonic fit R^2 (u, whole field): %.5f' % (1 - ss_res / ss_tot))

    print('Building local RBF interpolants (this is the slow step - a few minutes)...')
    interps_u = build_mode_interpolators(nodes_x0, nodes_y0, coeff_u, NMODES)
    interps_v = build_mode_interpolators(nodes_x0, nodes_y0, coeff_v, NMODES)
    interps_p = build_mode_interpolators(nodes_x0, nodes_y0, coeff_p, NMODES)

    # --- k=0 residual, sampled at interior points clear of the cylinder ---
    rng = np.random.RandomState(0)
    n_sample = 300
    sx = rng.uniform(LXMIN + 0.5, LXMAX - 0.5, n_sample)
    sy = rng.uniform(LYMIN + 0.5, LYMAX - 0.5, n_sample)
    r = np.sqrt((sx - X_C) ** 2 + (sy - Y_C) ** 2)
    keep = r > 1.5 * R_C
    sx, sy = sx[keep], sy[keep]
    residual, f_u, f_v, div_u = true_field_k0_residual(interps_u, interps_v, interps_p, sx, sy, NMODES, OMEGA_0, RE)
    print('')
    print('=== k=0 harmonic residual, true field (%d interior sample points) ===' % len(sx))
    print('  mean |f_u|^2+|f_v|^2+|div_u|^2 : %.4e' % residual.mean())
    print('  median                        : %.4e' % np.median(residual))
    print('  max                           : %.4e' % residual.max())
    print('  (compare against the R3-checkpoint value from --Mode checkpoint - expect the')
    print('   checkpoint value roughly an order of magnitude higher, per the diagnostic study)')

    # --- CV1 box residuals ---
    boxes = build_cv1_boxes()
    print('')
    print('=== k=1 control-volume residuals, true field (%d boxes) ===' % len(boxes))
    print('%-8s%-22s%-14s%-14s%-14s' % ('box', 'x range', '|Rx|', '|Ry|', '|Rx|^2+|Ry|^2'))
    normalizers = []
    for j, box in enumerate(boxes):
        Rx, Ry = true_field_cv1_residual(interps_u, interps_v, interps_p, box, NMODES, OMEGA_0, RE)
        mag2 = abs(Rx) ** 2 + abs(Ry) ** 2
        normalizers.append(max(mag2, 1e-12))
        print('%-8d[%.1f, %.1f]        %-14.4e%-14.4e%-14.4e' %
              (j, CV1_X_UP[j], CV1_X_DOWN[j], abs(Rx), abs(Ry), mag2))
    print('')
    print('Suggested CV1_NORMALIZERS starting point (true-field |R_j|^2 - use the')
    print('R3-checkpoint values instead for the actual calibration, per the plan; this')
    print('is only useful as a sanity floor - normalizers this small would make the loss')
    print('blow up on any field that is not already true):')
    print(' ', normalizers)


# =============================================================================
# --Mode checkpoint
# =============================================================================

def run_checkpoint(args):
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    import NN_functions as nnf  # noqa: E402
    from text_flow import read_flow  # noqa: E402

    run_dir_name = os.path.basename(os.path.normpath(args.RunDir))
    use_freestream = 'FSBC' in run_dir_name
    use_fluct_damp = 'FIBC' in run_dir_name
    freestream_target_u = 1.0 if use_freestream else None
    freestream_target_v = 0.0 if use_freestream else None
    print('Restoring with freestream_target=%s, damp_fluctuations=%s (run dir: %s)' %
          (use_freestream, use_fluct_damp, run_dir_name))

    layers = [2, args.WidthLayer * args.Nmodes, args.WidthLayer * args.Nmodes, args.Nmodes]
    pickle_candidates = glob.glob(os.path.join(args.RunDir, 'DNN*_tanh.pickle'))
    assert pickle_candidates, f'No model pickle found in {args.RunDir}'
    w_u, b_u, w_v, b_v, w_p, b_p = nnf.restore_NN(layers, pickle_candidates[0], tf_as_constant=True)
    Nmodes = args.Nmodes

    def fluid_u(x, y):
        return nnf.out_nn_modes_uv(x, y, w_u, b_u, GEOM, freestream_target=freestream_target_u, damp_fluctuations=use_fluct_damp)

    def fluid_v(x, y):
        return nnf.out_nn_modes_uv(x, y, w_v, b_v, GEOM, freestream_target=freestream_target_v, damp_fluctuations=use_fluct_damp)

    def fluid_p(x, y):
        return nnf.out_nn_modes_p(x, y, w_p, b_p)

    def customgrad(fgrad, xgrad):
        one = tf.transpose(0. * xgrad + 1.)
        parts = [tf.complex(tf.gradients(tf.real(fgrad[:, :, k]), xgrad, grad_ys=one)[0],
                             tf.gradients(tf.imag(fgrad[:, :, k]), xgrad, grad_ys=one)[0]) for k in range(Nmodes)]
        return tf.transpose(tf.convert_to_tensor(parts), perm=[2, 1, 0])

    x_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
    y_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

    # ---- k=0 harmonic residual, standalone TF graph (duplicated from
    # ModalPINN_VortexShedding.py's loss_int_mode_per_k - must stay in sync). ----
    all_u = fluid_u(x_tf, y_tf)
    all_v = fluid_v(x_tf, y_tf)
    all_p = fluid_p(x_tf, y_tf)
    all_u_x = customgrad(all_u, x_tf)
    all_u_y = customgrad(all_u, y_tf)
    all_v_x = customgrad(all_v, x_tf)
    all_v_y = customgrad(all_v, y_tf)
    all_p_x = customgrad(all_p, x_tf)
    all_p_y = customgrad(all_p, y_tf)
    all_u_xx = customgrad(all_u_x, x_tf)
    all_u_yy = customgrad(all_u_y, y_tf)
    all_v_xx = customgrad(all_v_x, x_tf)
    all_v_yy = customgrad(all_v_y, y_tf)

    k = 0
    f_u_k0 = tf.complex(0., k * OMEGA_0) * all_u[:, :, k] + all_p_x[:, :, k] - (1. / RE) * (all_u_xx[:, :, k] + all_u_yy[:, :, k])
    f_u_k0 += tf.reduce_sum(tf.convert_to_tensor([all_u[:, :, l] * all_u_x[:, :, k - l] for l in range(k + 1)]), axis=0)
    f_u_k0 += tf.reduce_sum(tf.convert_to_tensor([all_v[:, :, l] * all_u_y[:, :, k - l] for l in range(k + 1)]), axis=0)
    if k + 1 < Nmodes:
        f_u_k0 += tf.reduce_sum(tf.convert_to_tensor([all_u[:, :, l] * tf.conj(all_u_x[:, :, l - k]) for l in range(k + 1, Nmodes)]), axis=0)
        f_u_k0 += tf.reduce_sum(tf.convert_to_tensor([tf.conj(all_u[:, :, l - k]) * all_u_x[:, :, l] for l in range(k + 1, Nmodes)]), axis=0)
        f_u_k0 += tf.reduce_sum(tf.convert_to_tensor([all_v[:, :, l] * tf.conj(all_u_y[:, :, l - k]) for l in range(k + 1, Nmodes)]), axis=0)
        f_u_k0 += tf.reduce_sum(tf.convert_to_tensor([tf.conj(all_v[:, :, l - k]) * all_u_y[:, :, l] for l in range(k + 1, Nmodes)]), axis=0)

    f_v_k0 = tf.complex(0., k * OMEGA_0) * all_v[:, :, k] + all_p_y[:, :, k] - (1. / RE) * (all_v_xx[:, :, k] + all_v_yy[:, :, k])
    f_v_k0 += tf.reduce_sum(tf.convert_to_tensor([all_u[:, :, l] * all_v_x[:, :, k - l] for l in range(k + 1)]), axis=0)
    f_v_k0 += tf.reduce_sum(tf.convert_to_tensor([all_v[:, :, l] * all_v_y[:, :, k - l] for l in range(k + 1)]), axis=0)
    if k + 1 < Nmodes:
        f_v_k0 += tf.reduce_sum(tf.convert_to_tensor([all_u[:, :, l] * tf.conj(all_v_x[:, :, l - k]) for l in range(k + 1, Nmodes)]), axis=0)
        f_v_k0 += tf.reduce_sum(tf.convert_to_tensor([tf.conj(all_u[:, :, l - k]) * all_v_x[:, :, l] for l in range(k + 1, Nmodes)]), axis=0)
        f_v_k0 += tf.reduce_sum(tf.convert_to_tensor([all_v[:, :, l] * tf.conj(all_v_y[:, :, l - k]) for l in range(k + 1, Nmodes)]), axis=0)
        f_v_k0 += tf.reduce_sum(tf.convert_to_tensor([tf.conj(all_v[:, :, l - k]) * all_v_y[:, :, l] for l in range(k + 1, Nmodes)]), axis=0)

    div_u_k0 = all_u_x[:, :, k] + all_v_y[:, :, k]
    loss_k0_tf = nnf.square_norm(f_u_k0) + nnf.square_norm(f_v_k0) + nnf.square_norm(div_u_k0)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    rng = np.random.RandomState(0)
    n_sample = 300
    sx = rng.uniform(LXMIN + 0.5, LXMAX - 0.5, n_sample).astype(np.float32)
    sy = rng.uniform(LYMIN + 0.5, LYMAX - 0.5, n_sample).astype(np.float32)
    r = np.sqrt((sx - X_C) ** 2 + (sy - Y_C) ** 2)
    keep = r > 1.5 * R_C
    sx, sy = sx[keep], sy[keep]
    feed = {x_tf: sx.reshape(-1, 1), y_tf: sy.reshape(-1, 1)}
    k0_vals = sess.run(loss_k0_tf, feed_dict=feed)
    print('')
    print('=== k=0 harmonic residual, R3 checkpoint (%d interior sample points) ===' % len(sx))
    print('  mean : %.4e' % k0_vals.mean())
    print('  median : %.4e' % np.median(k0_vals))
    print('  max : %.4e' % k0_vals.max())

    # ---- k=1 control-volume residuals, standalone TF graph ----
    def conv_mode_k_tf(a, b, kk, nmodes_local):
        direct = tf.reduce_sum(tf.convert_to_tensor([a[:, :, l] * b[:, :, kk - l] for l in range(kk + 1)]), axis=0)
        if kk + 1 < nmodes_local:
            conj_a = tf.reduce_sum(tf.convert_to_tensor([a[:, :, l] * tf.conj(b[:, :, l - kk]) for l in range(kk + 1, nmodes_local)]), axis=0)
            conj_b = tf.reduce_sum(tf.convert_to_tensor([tf.conj(a[:, :, l - kk]) * b[:, :, l] for l in range(kk + 1, nmodes_local)]), axis=0)
            return direct + conj_a + conj_b
        return direct

    def grad_mode1_tf(fgrad, xgrad):
        '''1st derivative of just the k=1 mode slice - see
        ModalPINN_VortexShedding.grad_mode1's docstring for why this must not
        loop over all Nmodes (an earlier all-modes version here OOM-killed
        the training script's own gate smoke test).'''
        one = tf.transpose(0. * xgrad + 1.)
        f1 = fgrad[:, :, 1]
        return tf.complex(tf.gradients(tf.real(f1), xgrad, grad_ys=one)[0], tf.gradients(tf.imag(f1), xgrad, grad_ys=one)[0])

    boxes = build_cv1_boxes()
    print('')
    print('=== k=1 control-volume residuals, R3 checkpoint (%d boxes) ===' % len(boxes))
    print('%-8s%-22s%-14s%-14s%-14s' % ('box', 'x range', '|Rx|', '|Ry|', '|Rx|^2+|Ry|^2'))
    checkpoint_normalizers = []
    for j, box in enumerate(boxes):
        def col(a):
            return tf.constant(a.reshape(-1, 1), dtype=tf.float32)

        xa, ya = col(box['area_x']), col(box['area_y'])
        wa_c = tf.complex(tf.constant(box['area_w'], dtype=tf.float32), 0.)
        u1_a = fluid_u(xa, ya)[0, :, 1]
        v1_a = fluid_v(xa, ya)[0, :, 1]
        Rx = tf.complex(0., OMEGA_0) * tf.reduce_sum(wa_c * u1_a)
        Ry = tf.complex(0., OMEGA_0) * tf.reduce_sum(wa_c * v1_a)
        for face in ['left', 'right', 'bottom', 'top']:
            xf, yf = col(box[face + '_x']), col(box[face + '_y'])
            wf_c = tf.complex(tf.constant(box[face + '_w'], dtype=tf.float32), 0.)
            nx, ny = box[face + '_n']
            nx_c, ny_c = tf.complex(nx, 0.), tf.complex(ny, 0.)
            au = fluid_u(xf, yf)
            av = fluid_v(xf, yf)
            ap = fluid_p(xf, yf)
            p1 = ap[0, :, 1]
            Qxx = conv_mode_k_tf(au, au, 1, Nmodes)[0, :]
            Qxy = conv_mode_k_tf(au, av, 1, Nmodes)[0, :]
            Qyy = conv_mode_k_tf(av, av, 1, Nmodes)[0, :]
            u1_x = grad_mode1_tf(au, xf)[0, :]
            u1_y = grad_mode1_tf(au, yf)[0, :]
            v1_x = grad_mode1_tf(av, xf)[0, :]
            v1_y = grad_mode1_tf(av, yf)[0, :]
            flux_x = Qxx * nx_c + Qxy * ny_c
            flux_y = Qxy * nx_c + Qyy * ny_c
            visc_x = (1. / tf.complex(RE, 0.)) * (2. * u1_x * nx_c + (u1_y + v1_x) * ny_c)
            visc_y = (1. / tf.complex(RE, 0.)) * ((u1_y + v1_x) * nx_c + 2. * v1_y * ny_c)
            Rx = Rx + tf.reduce_sum(wf_c * (flux_x + p1 * nx_c - visc_x))
            Ry = Ry + tf.reduce_sum(wf_c * (flux_y + p1 * ny_c - visc_y))

        Rx_val, Ry_val = sess.run([Rx, Ry])
        mag2 = abs(Rx_val) ** 2 + abs(Ry_val) ** 2
        checkpoint_normalizers.append(max(mag2, 1e-12))
        print('%-8d[%.1f, %.1f]        %-14.4e%-14.4e%-14.4e' %
              (j, CV1_X_UP[j], CV1_X_DOWN[j], abs(Rx_val), abs(Ry_val), mag2))

    print('')
    print('CV1_NORMALIZERS to paste into ModalPINN_VortexShedding.py (R3-checkpoint')
    print('|R_j|^2 per box - per the plan, calibrate so each box contributes roughly')
    print('evenly once --CV1Loss is on, i.e. use these values directly as the')
    print('normalizers so every box starts at O(1) contribution to Loss_cv1):')
    print(' ', checkpoint_normalizers)

    print('')
    print('Suggested LambdaCV1 / LambdaK0: with CV1_NORMALIZERS set as above, Loss_cv1')
    print('itself will be O(%d) at the R3 checkpoint (each box ~O(1)) - pick LambdaCV1' % len(boxes))
    print('so that LambdaCV1*Loss_cv1 is ~10-20%% of the total R3-checkpoint loss (read')
    print("R3's out.txt for the total loss value at convergence) and LambdaK0 similarly")
    print('against the printed mean k=0 residual above.')

    # ---- free diagnostic (a): wake-region pointwise loss vs N=3 truncation floor ----
    print('')
    print('=== Free diagnostic: wake-region pointwise residual vs truncation floor ===')
    t_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
    u_t = nnf.NN_time_uv(x_tf, y_tf, t_tf, w_u, b_u, GEOM, OMEGA_0, freestream_target=freestream_target_u, damp_fluctuations=use_fluct_damp)
    v_t = nnf.NN_time_uv(x_tf, y_tf, t_tf, w_v, b_v, GEOM, OMEGA_0, freestream_target=freestream_target_v, damp_fluctuations=use_fluct_damp)
    p_t = nnf.NN_time_p(x_tf, y_tf, t_tf, w_p, b_p, OMEGA_0)
    u_t_t = tf.gradients(u_t, t_tf)[0]
    v_t_t = tf.gradients(v_t, t_tf)[0]
    u_t_x = tf.gradients(u_t, x_tf)[0]
    u_t_y = tf.gradients(u_t, y_tf)[0]
    u_t_xx = tf.gradients(u_t_x, x_tf)[0]
    u_t_yy = tf.gradients(u_t_y, y_tf)[0]
    v_t_x = tf.gradients(v_t, x_tf)[0]
    v_t_y = tf.gradients(v_t, y_tf)[0]
    v_t_xx = tf.gradients(v_t_x, x_tf)[0]
    v_t_yy = tf.gradients(v_t_y, y_tf)[0]
    p_t_x = tf.gradients(p_t, x_tf)[0]
    p_t_y = tf.gradients(p_t, y_tf)[0]
    f_u_t = u_t_t + (u_t * u_t_x + v_t * u_t_y) + p_t_x - (1. / RE) * (u_t_xx + u_t_yy)
    f_v_t = v_t_t + (u_t * v_t_x + v_t * v_t_y) + p_t_y - (1. / RE) * (v_t_xx + v_t_yy)
    div_t = u_t_x + v_t_y
    pointwise_loss = tf.square(f_u_t) + tf.square(f_v_t) + tf.square(div_t)

    n_wake = 300
    wx = rng.uniform(X_C, X_C + 3 * D, n_wake).astype(np.float32)
    wy = rng.uniform(-2., 2., n_wake).astype(np.float32)
    wt = rng.uniform(0., 20., n_wake).astype(np.float32)
    feed_wake = {x_tf: wx.reshape(-1, 1), y_tf: wy.reshape(-1, 1), t_tf: wt.reshape(-1, 1)}
    wake_loss = sess.run(pointwise_loss, feed_dict=feed_wake)
    print('  mean pointwise NS residual in near-wake region : %.4e (R3 log reported a' % wake_loss.mean())
    print('  1.79e-2 truncation floor for comparison - if these are close, R3 had')
    print('  converged as far as N=3 permits in the wake already)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Mode', required=True, choices=['truefield', 'checkpoint'])
    parser.add_argument('--DataFile', default=DEFAULT_DATA_FILE)
    parser.add_argument('--RunDir', default=None, help='(checkpoint mode) path to a trained run folder')
    parser.add_argument('--WidthLayer', type=int, default=25)
    parser.add_argument('--Nmodes', type=int, default=3)
    args = parser.parse_args()

    if args.Mode == 'truefield':
        run_truefield(args)
    else:
        assert args.RunDir is not None, '--Mode checkpoint requires --RunDir'
        run_checkpoint(args)


if __name__ == '__main__':
    main()
