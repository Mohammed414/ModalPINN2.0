"""Zero-mean pressure sensitivity check, as specified in data_contract.md.

The primary A04 metric keeps raw pressure (matches the training loss and the
existing evaluators). Several far-field pressure rel_L2 values exceed 1.0,
which is the signature of a constant gauge offset inflating the error rather
than the pressure *field* being that wrong. This script adds the explicitly
labelled zero-mean sensitivity metric the contract allows, WITHOUT replacing
the primary raw-pressure result: it removes the best-fit constant offset (one
scalar per model, per region, fit by least squares over space and time) and
reports both the offset magnitude and the corrected rel_L2 alongside the raw
one.

This is inference-only and reuses the exact model-building code in
a04_prior_attribution.py so the fields are identical to the primary run.
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
    region_masks,
    relative_l2,
    strict_crop_indices,
)
import a04_prior_attribution as base  # noqa: E402

DEFAULT_OUT = FRESH_ROOT / "derived" / "a04_pressure_gauge_check.json"


def offset_corrected_rel_l2(pred: np.ndarray, ref: np.ndarray,
                            mask: np.ndarray) -> dict:
    """Best-fit constant offset (least squares, space+time) then rel_L2.

    A single scalar c minimising ||pred - c - ref|| over the region/time is
    c = mean(pred - ref) on that mask. Removing it isolates whatever error
    remains once a constant instrument-style bias is discounted.
    """
    p = pred[:, mask]
    r = ref[:, mask]
    raw = relative_l2(p, r, mask=None)
    offset = float(np.mean(p - r))
    corrected = relative_l2(p - offset, r, mask=None)
    return {"raw_rel_L2": raw, "offset": offset, "offset_corrected_rel_L2": corrected}


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
    refs = {"u": U[:, idx], "v": V[:, idx], "p": P[:, idx]}
    regions = region_masks(x, y)

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

    out = {"analysis_id": "A04", "method": "pressure_gauge_sensitivity",
           "status": "verified", "models": {}}
    for name, (module_path, checkpoint, prior_path) in configs.items():
        tf.compat.v1.reset_default_graph()
        _, placeholders, outputs, _ = base._tf_model(
            module_path, checkpoint, prior_path, tf
        )
        x_tf, y_tf, t_tf = placeholders
        _, _, p_t = outputs
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            allow_soft_placement=True
        ))
        sess.run(tf.compat.v1.global_variables_initializer())
        nt, nnode = refs["p"].shape
        pred_p = np.empty((nt, nnode), dtype=np.float32)
        x_col = x.astype(np.float32).reshape(-1, 1)
        y_col = y.astype(np.float32).reshape(-1, 1)
        chunk = 4000
        for start in range(0, nnode, chunk):
            stop = min(start + chunk, nnode)
            sl = slice(start, stop)
            feed_xy = {x_tf: x_col[sl], y_tf: y_col[sl]}
            for it, t in enumerate(times):
                feed = dict(feed_xy)
                feed[t_tf] = np.full((stop - start, 1), t, dtype=np.float32)
                pred_p[it, sl] = sess.run(p_t, feed_dict=feed)[:, 0]
            print(f"{name}: pressure nodes {stop}/{nnode}", flush=True)
        sess.close()

        out["models"][name] = {
            region: offset_corrected_rel_l2(pred_p, refs["p"], mask)
            for region, mask in regions.items()
        }

    DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUT.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps(out, indent=2, allow_nan=False))
    print("Wrote", DEFAULT_OUT)


if __name__ == "__main__":
    main()
