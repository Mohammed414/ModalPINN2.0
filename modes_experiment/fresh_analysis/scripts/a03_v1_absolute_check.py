"""Absolute (not ratio) v1 magnitudes per region for A03's three arms.

A03's `other` region (upstream / off-axis) reports a v1 amplitude ratio of
~8.0 for both wake-biased arms against ~1.2 for uniform. Read naively that is
"8x too much oscillation upstream", but upstream of a cylinder the true v1 is
physically almost zero, so any ratio against it inflates. Exactly this trap
was already found and defused for A04, whose alarming 11.5x turned out to be
leakage worth only 37% of the far-core signal
(`derived/a04_v1_absolute_check.json`).

A03 is not obviously the same story, which is why this check exists. There the
ratio moved alone; here the FIELD errors move too -- upstream v rel_L2 goes
0.122 -> 0.322 and p 0.197 -> 0.434, both against non-degenerate denominators.
So something may be genuinely different, and the question this script answers
is the same one A04 asked: how large is the upstream leakage in absolute
terms, and how does it compare to the signal the arm actually produces
downstream?

Mirrors `a04_v1_absolute_check.py` exactly (same masks, same harmonic
extraction, same normalisation) so the two sets of numbers are comparable.
Inference only; no training operation is used.
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
import a03_collocation_strategy as a03  # noqa: E402

OUT = FRESH_ROOT / "derived" / "a03_v1_absolute_check.json"

# Reuse A03's own arm definitions so this check can never drift from the
# evaluation it is explaining. None of these arms carries a Karman prior, which
# is the hypothesis under test: they have no --V1RadialTrust blend to suppress
# upstream oscillation.
CONFIGS = {
    name: (module_path, checkpoint, None)
    for name, (module_path, checkpoint, _sampling) in a03.CONFIGS.items()
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
    v_ref = V[:, idx]
    regions = region_masks(x, y)
    true_v1 = temporal_harmonic_coefficients(v_ref, times, OMEGA_0, 3)[1]

    out = {}
    for name, (module_path, checkpoint, prior_path) in CONFIGS.items():
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

    # The question the ratio cannot answer: how big is the upstream leakage
    # relative to what the arm actually produces in the wake it is meant to
    # reconstruct? A04's answer for its own arm was 37%.
    for name, per_region in out.items():
        other = per_region.get("other (upstream/off-axis)") or per_region.get("other")
        far_core = per_region.get("far-core")
        if other and far_core and far_core["pred_rms_per_node"] > 0:
            other["leakage_vs_far_core_pred"] = (
                other["pred_rms_per_node"] / far_core["pred_rms_per_node"]
            )
            other["pred_over_true_rms"] = (
                other["pred_rms_per_node"] / other["true_rms_per_node"]
                if other["true_rms_per_node"] > 0 else None
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
