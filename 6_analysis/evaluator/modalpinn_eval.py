"""Evaluate a trained ModalPINN checkpoint on an arbitrary (x, y) grid, in NumPy.

The saved pickles hold plain complex64 weight/bias lists, so the forward pass is
reimplemented here rather than reloading TensorFlow. Every function below is a
direct transcription of the corresponding one in the project's NN_functions.py:

    neural_net          -> tanh MLP, 2 -> 100 -> 100 -> Nmodes, complex
    f_BC5               -> tanh(5 * (r - r_c)), the no-slip mask on the cylinder
    f_freestream_weight -> 0.5 * (1 - tanh(3 * (x + 2))), the inlet blend ramp
    out_nn_modes_*      -> mask, then the freestream blend on k=0
    NN_time_*           -> Re( sum_k q_k e^{i k omega_0 t} )

The --V1RadialTrust wrap is reimplemented too (street_modes_k + _v1_trust), so
prior-active arms are evaluated as they were trained. hard_sym,
damp_fluctuations and kill_k0_imag are NOT reimplemented; no arm in this study
sets them, and `assert_flags_supported` refuses to evaluate one that does
rather than silently returning a different field.
"""
import json
import os
import pickle

import numpy as np

# Domain geometry, from the project's --Lxmin/--Lxmax/... defaults.
GEOM = dict(Lxmin=-4.0, Lxmax=8.0, Lymin=-4.0, Lymax=4.0,
            x_c=0.0, y_c=0.0, r_c=0.5)
OMEGA_0 = 1.0392   # shedding frequency used in training (St ~ 0.1654)


def load_weights(pickle_path):
    """Return (w_u, b_u, w_v, b_v, w_p, b_p) as lists of complex arrays."""
    with open(pickle_path, "rb") as f:
        obj = pickle.load(f)
    assert len(obj) == 6, f"expected 6 weight/bias lists, got {len(obj)}"
    return [[np.asarray(a) for a in group] for group in obj]


def neural_net(X, weights, biases):
    """Complex tanh MLP. X is [N, 2] complex; returns [N, Nmodes] complex."""
    H = X
    for W, b in zip(weights[:-1], biases[:-1]):
        H = np.tanh(H @ W + b)
    return H @ weights[-1] + biases[-1]


def f_bc5(x, y, fact=5.0):
    """No-slip mask: zero on the cylinder surface, ->1 away from it."""
    r = np.sqrt((x - GEOM["x_c"]) ** 2 + (y - GEOM["y_c"]) ** 2) - GEOM["r_c"]
    return np.tanh(fact * r)


def f_freestream_weight(x, x_transition=-2.0, gamma=3.0):
    """Inlet blend ramp: ->1 upstream of the transition, ->0 downstream."""
    return 0.5 * (1.0 - np.tanh(gamma * (x - x_transition)))


def smootherstep01(z):
    """C2 smootherstep clamped to [0, 1]: 6z^5 - 15z^4 + 10z^3."""
    zc = np.clip(z, 0.0, 1.0)
    return zc * zc * zc * (zc * (zc * 6.0 - 15.0) + 10.0)


def load_street_params(npz_path):
    """Read the analytic Karman-street prior parameters fitted from the taps."""
    z = np.load(npz_path, allow_pickle=True)
    return {k: float(z[k]) for k in z.files}


def street_modes_k(x, y, sp, k):
    """Closed-form Karman street mode k. Returns (u_k, v_k, p_k) complex."""
    G, Uc, xf, r0 = sp["Gamma"], sp["Uc"], sp["xf"], sp["r0"]
    omega, phase = sp["omega"], sp["phase"]
    ramp, delta = sp["ramp"], sp["delta"]
    amp = sp["amp_scale"] * 2.0
    nu = 1.0 / 100.0
    a = 2.0 * np.pi * Uc / omega
    h = 0.281 * a
    xs = np.asarray(x, np.float64).ravel()
    ys = np.asarray(y, np.float64).ravel()
    env = 0.5 * (1.0 + np.tanh((xs - xf) / ramp))
    rc2 = r0 ** 2 + 4.0 * nu * np.maximum(xs - xf, 0.0) / Uc
    att = np.exp(-(np.pi * k) ** 2 * rc2 / a ** 2)
    u_re = np.zeros_like(xs); u_im = np.zeros_like(xs)
    v_re = np.zeros_like(xs); v_im = np.zeros_like(xs)
    for y_row, s_row, x0 in ((+h / 2.0, -1.0, xf), (-h / 2.0, +1.0, xf + a / 2.0)):
        yp = ys - y_row
        sabs = np.sqrt(yp ** 2 + delta ** 2) - delta
        sgn = -np.tanh(yp / delta)
        Dk = np.exp(-2.0 * np.pi * k * sabs / a)
        ph = -2.0 * np.pi * k * (xs - x0) / a - k * phase
        mag = s_row * G / (2.0 * a) * Dk * att
        b_re, b_im = mag * np.cos(ph), mag * np.sin(ph)
        u_re += sgn * b_re; u_im += sgn * b_im
        v_re += -b_im;      v_im += b_re
    scale = amp * env
    u_k = scale * (u_re + 1j * u_im)
    v_k = scale * (v_re + 1j * v_im)
    p_fac = -(1.0 - Uc) * sp["scale_p"]
    return u_k, v_k, p_fac * u_k


def _v1_trust(x, y, corr_raw, sp, rho=0.60, xstart=3.0, xwidth=0.30,
              ymax=2.0, ywidth=0.20):
    """The --V1RadialTrust wrap applied to v_1 during training.

    v1 = fbc5 * ((1-W)*free + W*(S + rho|S| * z/sqrt(1+|z|^2)))

    W is a C2 gate that is exactly 1 in the trusted downstream core. Inside it
    the network can only make a bounded correction to the analytic street mode,
    so v1 = 0 is outside the search space wherever the street is alive.
    """
    _, S_v, _ = street_modes_k(x, y, sp, 1)
    den = np.sqrt(1.0 + np.abs(corr_raw) ** 2)
    corr_radial = corr_raw / den
    wx = smootherstep01((x - (xstart - xwidth)) / xwidth)
    wy_hi = 1.0 - smootherstep01((y - ymax) / ywidth)
    wy_lo = 1.0 - smootherstep01((-y - ymax) / ywidth)
    W = wx * wy_hi * wy_lo
    eps = 1e-6
    S_mag = np.sqrt(S_v.real ** 2 + S_v.imag ** 2 + eps ** 2) - eps
    trusted = S_v + (rho * S_mag) * corr_radial
    return (1.0 - W) * corr_raw + W * trusted


def modes(x, y, weights, biases, freestream_target=None,
          street_params=None, is_v=False):
    """Mode shapes at (x, y). Returns [N, Nmodes] complex.

    x, y are flat float arrays. freestream_target blends the k=0 mode toward a
    known constant near the inlet (1.0 for u, 0.0 for v); pass None to disable.
    """
    x = np.asarray(x, np.float64).ravel()
    y = np.asarray(y, np.float64).ravel()
    X = np.stack([x, y], axis=1).astype(np.complex128)
    out = neural_net(X, weights, biases)
    # The v1 trust wrap consumes the RAW network output and applies the mask
    # itself, so it must run before the global mask multiply.
    if street_params is not None and is_v and out.shape[1] > 1:
        out[:, 1] = _v1_trust(x, y, out[:, 1].copy(), street_params)
    out = out * f_bc5(x, y)[:, None]
    if freestream_target is not None:
        w = f_freestream_weight(x)
        out[:, 0] = (1.0 - w) * out[:, 0] + w * freestream_target
    return out


def field_at_time(x, y, t, weights, biases, freestream_target=None,
                  omega_0=OMEGA_0, n_modes=None, street_params=None,
                  is_v=False):
    """Real field value at time t: Re( sum_k q_k e^{i k omega_0 t} )."""
    q = modes(x, y, weights, biases, freestream_target,
              street_params=street_params, is_v=is_v)
    K = q.shape[1] if n_modes is None else min(n_modes, q.shape[1])
    ks = np.arange(K)
    return np.real((q[:, :K] * np.exp(1j * ks * omega_0 * t)).sum(axis=1))


def assert_flags_supported(run_record_path):
    """Refuse to evaluate an arm whose flags this module does not reimplement.

    Returns a dict of the flags that DO matter for evaluation.
    """
    with open(run_record_path) as f:
        cmd = json.load(f).get("command") or []
    unsupported = ["--HardSym", "--FluctuationInletBC", "--KillK0Imag"]
    present = [f for f in unsupported if f in cmd]
    assert not present, (
        f"{os.path.basename(os.path.dirname(run_record_path))} uses "
        f"{present}, which modalpinn_eval does not reimplement")
    return dict(freestream=("--FreestreamBC" in cmd),
                prior=("--V1RadialTrust" in cmd))


def grid(nx=240, ny=160):
    """Uniform evaluation grid over the reconstruction domain, cylinder masked."""
    xs = np.linspace(GEOM["Lxmin"], GEOM["Lxmax"], nx)
    ys = np.linspace(GEOM["Lymin"], GEOM["Lymax"], ny)
    XX, YY = np.meshgrid(xs, ys)
    inside = ((XX - GEOM["x_c"]) ** 2 + (YY - GEOM["y_c"]) ** 2
              < GEOM["r_c"] ** 2)
    return XX, YY, inside
