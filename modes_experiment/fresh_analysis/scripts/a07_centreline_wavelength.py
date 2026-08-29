"""Streamwise wavelength of the first harmonic, from the centreline phase gradient.

Chapter 4's mechanism argument rests on the claim that the failed arms produce a
nearly *standing* disturbance rather than a travelling vortex street. The
quantity that shows it is the streamwise wavelength, obtained from the gradient
of the unwrapped phase of $\\hat v_1$ along the wake centreline:

    phi(x) = arg( v1(x, y~0) ),   fitted over 2 <= x <= 8, |y| <= 0.75,
    lambda = 2*pi / |d phi / d x|.

A travelling wave advances phase linearly with x, so a correct reconstruction
returns a wavelength near the DNS value. A standing disturbance has almost no
phase advance, so its fitted wavelength diverges.

The numbers previously quoted for this (DNS 4.59D, and so on) came from
`modes_experiment/figures/decay_profiles.json`, outside the frozen workspace.
This script recomputes them under the frozen contract — same crop, same harmonic
extraction, same inference wrappers — so the chapter can cite a workspace file.

Inference only; no training operation is used.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from evaluate_common import (  # noqa: E402
    OMEGA_0,
    strict_crop_indices,
    temporal_harmonic_coefficients,
)
import a04_prior_attribution as base  # noqa: E402
import a03_collocation_strategy as a03  # noqa: E402

OUT = ROOT / "derived" / "a07_centreline_wavelength.json"

# Centreline strip and fit window, matching the convention the existing draft used.
Y_HALFWIDTH = 0.75
X_MIN, X_MAX = 2.0, 8.0

ARMS = base.ARMS_ROOT
CONFIGS = {
    "pressure_only_physics": (
        ARMS / "01_baseline_physics_only" / "training_run" / "NN_functions.py",
        ARMS / "01_baseline_physics_only" / "training_run" / "DNN2_100_100_4_tanh.pickle",
        None,
    ),
    "pressure_and_velocity_probes": (
        ARMS / "04_paper_sparse_probes" / "training_run" / "NN_functions.py",
        ARMS / "04_paper_sparse_probes" / "training_run" / "DNN2_100_100_4_tanh.pickle",
        None,
    ),
    "wake_biased_random_collocation": (
        ARMS / "arm_06_wake_biased_random" / "training_run" / "NN_functions.py",
        ARMS / "arm_06_wake_biased_random" / "training_run" / "DNN2_100_100_4_tanh.pickle",
        None,
    ),
    "wake_biased_grid_collocation": (
        ARMS / "07_wake_biased_grid" / "training_run" / "NN_functions.py",
        ARMS / "07_wake_biased_grid" / "training_run" / "DNN2_100_100_4_tanh.pickle",
        None,
    ),
    "pressure_only_physics_karman_prior": (
        ARMS / "15_karman_prior_fluct_off" / "training_run" / "NN_functions.py",
        ARMS / "15_karman_prior_fluct_off" / "training_run" / "DNN2_100_100_4_tanh.pickle",
        ARMS / "15_karman_prior_fluct_off" / "street_prior_used.npz",
    ),
}


def fit_wavelength(x, y, v1):
    """Least-squares wavelength from the unwrapped centreline phase gradient."""
    strip = (np.abs(y) <= Y_HALFWIDTH) & (x >= X_MIN) & (x <= X_MAX)
    xs, vs = x[strip], v1[strip]
    order = np.argsort(xs)
    xs, vs = xs[order], vs[order]

    # Bin along x so that mesh clustering does not weight one station heavily,
    # and take the phase of the binned complex mean rather than a mean of phases.
    edges = np.linspace(X_MIN, X_MAX, 61)
    idx = np.digitize(xs, edges) - 1
    xb, pb = [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < 5:
            continue
        xb.append(0.5 * (edges[b] + edges[b + 1]))
        pb.append(np.angle(vs[m].mean()))
    xb, pb = np.asarray(xb), np.unwrap(np.asarray(pb))

    slope, intercept = np.polyfit(xb, pb, 1)
    resid = pb - (slope * xb + intercept)
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((pb - pb.mean()) ** 2).sum())
    return {
        "phase_gradient_rad_per_D": float(slope),
        "wavelength_D": float(2 * np.pi / abs(slope)) if slope != 0 else float("inf"),
        "fit_r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n_bins": int(xb.size),
    }


def main() -> None:
    import tensorflow as tf  # noqa: WPS433
    tf.compat.v1.disable_eager_execution()
    if not hasattr(tf, "real"):
        tf.real = tf.math.real
    if not hasattr(tf, "imag"):
        tf.imag = tf.math.imag

    times, X, Y, U, V, P = base.load_cache(a03.DATA)
    idx = strict_crop_indices(X, Y)
    x, y = X[idx], Y[idx]
    true_v1 = temporal_harmonic_coefficients(V[:, idx], times, OMEGA_0, 3)[1]

    out = {
        "analysis_id": "A07",
        "method": "centreline_phase_gradient",
        "fit_window": {"x_min": X_MIN, "x_max": X_MAX, "y_halfwidth": Y_HALFWIDTH},
        "snapshots": int(times.size),
        "crop_nodes": int(x.size),
        "arms": {},
    }
    out["dns_reference"] = fit_wavelength(x, y, true_v1)
    print("dns", out["dns_reference"], flush=True)

    for name, (module_path, checkpoint, prior_path) in CONFIGS.items():
        tf.compat.v1.reset_default_graph()
        _, placeholders, _, v_modes = base._tf_model(module_path, checkpoint, prior_path, tf)
        x_tf, y_tf, _ = placeholders
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True))
        sess.run(tf.compat.v1.global_variables_initializer())
        v1 = sess.run(
            v_modes,
            feed_dict={x_tf: x.astype(np.float32).reshape(-1, 1),
                       y_tf: y.astype(np.float32).reshape(-1, 1)},
        )[0, :, 1].astype(np.complex128) / 2.0
        sess.close()
        out["arms"][name] = fit_wavelength(x, y, v1)
        print(name, out["arms"][name], flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
