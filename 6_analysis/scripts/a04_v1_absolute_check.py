"""Absolute (not ratio) v1 magnitudes per region, for the 'other'-region check.

amp_ratio = ||predicted v1|| / ||true v1||, both L2-summed over the region's
nodes. When the true amplitude is itself near zero (as expected upstream of
the cylinder, where physically there is no shedding), a small absolute
prediction leakage produces a large RATIO purely from the near-zero
denominator. This script reports the absolute norms directly so that
question can be answered: is the upstream artefact large in absolute terms,
or only large as a ratio against a near-zero reference?
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
FRESH_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from evaluate_common import (  # noqa: E402
    OMEGA_0,
    region_masks,
    strict_crop_indices,
    temporal_harmonic_coefficients,
)
import a04_prior_attribution as base  # noqa: E402

OUT = FRESH_ROOT / "derived" / "a04_v1_absolute_check.json"


def main() -> None:
    import tensorflow as tf  # noqa: WPS433
    tf.compat.v1.disable_eager_execution()
    if not hasattr(tf, "real"):
        tf.real = tf.math.real
    if not hasattr(tf, "imag"):
        tf.imag = tf.math.imag

    times, X, Y, U, V, P = base.load_cache(base.DEFAULT_DATA)
    idx = strict_crop_indices(X, Y)
    x, y = X[idx], Y[idx]
    v_ref = V[:, idx]
    regions = region_masks(x, y)
    true_v1 = temporal_harmonic_coefficients(v_ref, times, OMEGA_0, 3)[1]

    arm1 = base.ARMS_ROOT / "01_baseline_physics_only"
    arm15 = base.ARMS_ROOT / "15_karman_prior_fluct_off"
    configs = {
        "arm1_baseline": (arm1 / "training_run" / "NN_functions.py",
                          arm1 / "training_run" / "DNN2_100_100_4_tanh.pickle",
                          None),
        "arm15_v1_radial_trust": (
            arm15 / "training_run" / "NN_functions.py",
            arm15 / "training_run" / "DNN2_100_100_4_tanh.pickle",
            arm15 / "street_prior_used.npz",
        ),
    }

    out = {}
    for name, (module_path, checkpoint, prior_path) in configs.items():
        tf.compat.v1.reset_default_graph()
        _, placeholders, _, v_modes = base._tf_model(
            module_path, checkpoint, prior_path, tf
        )
        x_tf, y_tf, _ = placeholders
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            allow_soft_placement=True
        ))
        sess.run(tf.compat.v1.global_variables_initializer())
        x_col = x.astype(np.float32).reshape(-1, 1)
        y_col = y.astype(np.float32).reshape(-1, 1)
        v1 = sess.run(v_modes, feed_dict={x_tf: x_col, y_tf: y_col})[0, :, 1]
        v1 = v1.astype(np.complex128) / 2.0
        sess.close()

        out[name] = {}
        for region, mask in regions.items():
            n = int(mask.sum())
            pred_norm = float(np.linalg.norm(v1[mask]))
            true_norm = float(np.linalg.norm(true_v1[mask]))
            out[name][region] = {
                "n": n,
                "pred_l2_norm": pred_norm,
                "true_l2_norm": true_norm,
                "pred_rms_per_node": pred_norm / np.sqrt(n),
                "true_rms_per_node": true_norm / np.sqrt(n),
            }
        print(name, "done", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
