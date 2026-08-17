"""Numpy re-implementation of the ModalPINN forward pass, for plotting the
downloaded production runs (runs/*/DNN2_75_75_3_tanh.pickle).

Mirrors NN_functions.py exactly:
  neural_net  : H = tanh(H W + b) x2, then linear
  out_nn_modes_uv : fbc5 mask, R9 trust wrap for k>=1 (if street prior given),
                    freestream blend at k=0, inlet damping for k>=1
  out_nn_modes_p  : same minus the fbc5 mask / freestream / damping
  NN_time_*   : q(t) = Re[ sum_k q_k e^{i k w0 t} ]   (ONE-sided, NO factor 2)

Note Nmodes=3 in this codebase means k = 0,1,2 (three network outputs).

Truth modes (R9_wake_rescue/common.fit_modes) use the TWO-sided convention
q = c0 + sum 2Re[c_k e^{ikwt}], so a physical harmonic amplitude is
2|c_k^truth| <-> |q_k^network|. Amplitude comparisons below convert explicitly.

VALIDATION AGAINST THE OFFICIAL EVALUATOR (evaluate_regions.py), all 30
region x quantity pairs of the R7 and R9 production runs:

    quantity          median dev   max dev   where the max sits
    E_u, E_v              0.59 %    5.59 %   R9 near-cylinder (E_v; E_u 4.23 %)
    E_p                   0.55 %   83.56 %   R9 near-cylinder (R7: 57.9 %)

So: VELOCITY is validated to <=5.6 % worst case over all 20 E_u/E_v pairs
(<=2.92 % once the near-cylinder region is excluded - that bound is set by
R9 'other (upstream/off-axis)' E_v, 0.1086 official vs 0.1118 here; the
next largest are far-wake E_v 1.54 % and R7 'other' E_v 1.34 %). The largest
deviations all sit in regions where the truth norm is small, so a fixed
absolute discrepancy shows up as a larger relative one. PRESSURE IS NOT
VALIDATED near the cylinder - both runs show a large discrepancy there, so
this module's p-field must not be used for near-cylinder pressure claims.
Ruled out as causes: omega mismatch (both 1.036), a pressure gauge offset
(removing the mean changes E_p only 0.0349 -> 0.0345), and float32/complex64
precision (complex128 vs complex64 forward passes differ by 1e-5). The cause
is still unidentified; treat p as diagnostic-only.

The deliverable figures use velocity and the parsed regional_evaluation.txt
tables only - none of them plot a pressure field from this module.
"""
import os
import pickle
import sys

import numpy as np

R9_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(R9_ROOT)
sys.path.insert(0, os.path.join(R9_ROOT, 'src'))
sys.path.insert(0, os.path.join(REPO, 'R9_wake_rescue', 'src'))
sys.path.insert(0, os.path.join(REPO, 'src'))   # text_flow (street_prior imports it)

from street_prior import cf_modes_uv  # noqa: E402

X_C, Y_C, R_C = 0.0, 0.0, 0.5
OMEGA_0 = 1.036          # the value the training script's ansatz uses


def neural_net(X, weights, biases):
    H = X
    for W, b in zip(weights[:-1], biases[:-1]):
        H = np.tanh(H @ np.asarray(W) + np.asarray(b))
    return H @ np.asarray(weights[-1]) + np.asarray(biases[-1])


def f_bc5(x, y, fact=5.0):
    r = np.sqrt((x - X_C) ** 2 + (y - Y_C) ** 2) - R_C
    return np.tanh(fact * r)


def f_freestream_weight(x, x_transition=-2.0, gamma=3.0):
    return 0.5 * (1.0 - np.tanh(gamma * (x - x_transition)))


def street_modes(x, y, sp, nk):
    """Closed-form street modes k=1..nk in the NETWORK's one-sided convention
    (includes the amp_scale*2 factor of NN_functions.street_modes_k)."""
    prm = {k: float(sp[k]) for k in
           ('Gamma', 'Uc', 'xf', 'r0', 'omega', 'phase', 'ramp', 'delta')}
    amp = float(sp['amp_scale']) * 2.0
    us, vs = cf_modes_uv(x.astype(np.float64), y.astype(np.float64), prm, nk=nk)
    us = [u * amp for u in us]
    vs = [v * amp for v in vs]
    ps = [-(1.0 - prm['Uc']) * float(sp['scale_p']) * u for u in us]
    return us, vs, ps


def modes_from_weights(x, y, w, b, kind, sp=None, rho=0.6, cap=0.12,
                       freestream=None, damp=False):
    """Returns list of complex mode arrays [k=0..Nmodes-1], each shape (N,)."""
    X = np.stack([x.astype(np.complex128), y.astype(np.complex128)], axis=1)
    out = neural_net(X, w, b)
    nm = out.shape[1]
    fb = f_bc5(x, y).astype(np.complex128)
    wt = f_freestream_weight(x).astype(np.complex128)
    if sp is not None:
        S_all = dict(zip('uvp', street_modes(x, y, sp, nk=nm - 1)))
    modes = []
    for k in range(nm):
        if sp is not None and k >= 1:
            S = S_all[kind][k - 1]
            corr = np.tanh(out[:, k].real) + 1j * np.tanh(out[:, k].imag)
            A = rho * np.abs(S) + cap
            mode = S + A * corr
            if kind != 'p':
                mode = fb * mode
        else:
            mode = out[:, k] if kind == 'p' else fb * out[:, k]
        if k == 0 and freestream is not None:
            mode = wt * complex(freestream) + (1.0 - wt) * mode
        elif k >= 1 and damp and kind != 'p':
            mode = (1.0 - wt) * mode
        modes.append(mode)
    return modes


def reconstruct(modes, times, omega0=OMEGA_0):
    """q(t) = Re[ sum_k q_k e^{i k w0 t} ] -- this codebase's convention."""
    t = np.asarray(times) - times[0]
    F = np.zeros((len(t), len(modes[0])))
    for k, qk in enumerate(modes):
        F += (qk[None, :] * np.exp(1j * k * omega0 * t[:, None])).real
    return F.astype(np.float32)


def load_run(run_dir, trust=False):
    pk = [f for f in os.listdir(run_dir)
          if f.startswith('DNN') and f.endswith('.pickle')][0]
    with open(os.path.join(run_dir, pk), 'rb') as f:
        w_u, b_u, w_v, b_v, w_p, b_p = pickle.load(f)
    sp = None
    if trust:
        cand = [f for f in os.listdir(run_dir) if f.startswith('street_prior')]
        assert cand, f'trust run without street prior: {run_dir}'
        sp = np.load(os.path.join(run_dir, cand[0]))
    return dict(w=(w_u, b_u, w_v, b_v, w_p, b_p), sp=sp)


def eval_run(run, x, y):
    w_u, b_u, w_v, b_v, w_p, b_p = run['w']
    sp = run['sp']
    u = modes_from_weights(x, y, w_u, b_u, 'u', sp=sp, freestream=1.0, damp=True)
    v = modes_from_weights(x, y, w_v, b_v, 'v', sp=sp, freestream=0.0, damp=True)
    p = modes_from_weights(x, y, w_p, b_p, 'p', sp=sp)
    return u, v, p
