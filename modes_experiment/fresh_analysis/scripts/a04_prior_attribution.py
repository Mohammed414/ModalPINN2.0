"""Evaluate the analytical prior and the two trained A04 checkpoints.

This is an inference-only script.  It restores the saved weights as constants,
uses the checkpoint-local ``NN_functions.py`` forward path, and applies the
same metric contract as ``a04_prior_only.py``.  No optimizer, loss, or training
operation is constructed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
from typing import Dict, Tuple

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
FRESH_ROOT = HERE.parent
REPO_ROOT = FRESH_ROOT.parents[1]
ARMS_ROOT = REPO_ROOT / "modes_experiment" / "runs" / "arms"
DEFAULT_DATA = REPO_ROOT / "data" / "flow_cache.npz"
DEFAULT_OUT = FRESH_ROOT / "derived" / "a04_prior_attribution_metrics.json"

sys.path.insert(0, str(HERE))
from evaluate_common import (  # noqa: E402
    OMEGA_0,
    regional_complex_metrics,
    regional_field_metrics,
    region_masks,
    strict_crop_indices,
    temporal_harmonic_coefficients,
)


GEOM = [-4.0, 8.0, -4.0, 4.0, 0.0, 0.0, 0.5]
LAYERS = [2, 100, 100, 4]


def load_cache(path: pathlib.Path):
    z = np.load(path)
    return (np.asarray(z["times"], dtype=float).reshape(-1),
            np.asarray(z["X"], dtype=float).reshape(-1),
            np.asarray(z["Y"], dtype=float).reshape(-1),
            np.asarray(z["U"], dtype=np.float32),
            np.asarray(z["V"], dtype=np.float32),
            np.asarray(z["p"], dtype=np.float32))


def load_prior(path: pathlib.Path) -> Dict[str, float]:
    z = np.load(path)
    names = ("Gamma", "Uc", "xf", "r0", "omega", "phase", "amp_scale",
             "scale_p", "ramp", "delta")
    return {name: float(z[name]) for name in names}


def load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tf_model(module_path: pathlib.Path, checkpoint: pathlib.Path,
              prior_path: pathlib.Path | None, tf):
    """Build only inference tensors for one checkpoint."""
    module = load_module(module_path, "nn_functions_a04")
    w_u, b_u, w_v, b_v, w_p, b_p = module.restore_NN(
        LAYERS, str(checkpoint), tf_as_constant=True
    )
    x_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
    y_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
    t_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

    common_u = dict(
        freestream_target=1.0,
        damp_fluctuations=False,
        kill_k0_imag=False,
        hard_sym=False,
        trust_rho=0.6,
        trust_cap=0.12,
    )
    common_v = dict(common_u)
    common_v["freestream_target"] = 0.0
    # Arm 15 used the checkpoint-local V1RadialTrust path.  Arm 1 receives
    # the same defaults, but with no prior dictionary, so it remains ordinary.
    radial = None
    if prior_path is not None:
        radial = load_prior(prior_path)
    u_t = module.NN_time_uv(x_tf, y_tf, t_tf, w_u, b_u, GEOM, OMEGA_0,
                            is_v=False, street_params=None, **common_u)
    if radial is None:
        v_t = module.NN_time_uv(x_tf, y_tf, t_tf, w_v, b_v, GEOM, OMEGA_0,
                                is_v=True, street_params=None, **common_v)
    else:
        v_t = module.NN_time_uv(
            x_tf, y_tf, t_tf, w_v, b_v, GEOM, OMEGA_0, is_v=True,
            street_params=None, v1_radial_params=radial,
            v1_trust_rho=0.60, v1_xstart=3.0, v1_xwidth=0.30,
            v1_ymax=2.0, v1_ywidth=0.20, **common_v
        )
    p_t = module.NN_time_p(x_tf, y_tf, t_tf, w_p, b_p, OMEGA_0,
                           street_params=None, kill_k0_imag=False,
                           hard_sym=False, trust_rho=0.6, trust_cap=0.12)

    # The v1 metric is conventionally q=q0+q1 exp(iwt)+c.c.; the network
    # stores the one-sided coefficient used by NN_time_* and therefore needs
    # the same /2 conversion used by notebook 15.
    v_modes = module.out_nn_modes_uv(
        x_tf, y_tf, w_v, b_v, GEOM, is_v=True, street_params=None,
        v1_radial_params=radial, v1_trust_rho=0.60, v1_xstart=3.0,
        v1_xwidth=0.30, v1_ymax=2.0, v1_ywidth=0.20, **common_v
    )
    return module, (x_tf, y_tf, t_tf), (u_t, v_t, p_t), v_modes


def evaluate_model(name: str, module_path: pathlib.Path,
                   checkpoint: pathlib.Path, prior_path: pathlib.Path | None,
                   x: np.ndarray, y: np.ndarray, times: np.ndarray,
                   refs: Dict[str, np.ndarray], regions,
                   tf, chunk: int, snapshot_index: int) -> Dict[str, object]:
    tf.compat.v1.reset_default_graph()
    _, placeholders, outputs, v_modes = _tf_model(
        module_path, checkpoint, prior_path, tf
    )
    x_tf, y_tf, t_tf = placeholders
    u_t, v_t, p_t = outputs
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
        allow_soft_placement=True, log_device_placement=False
    ))
    sess.run(tf.compat.v1.global_variables_initializer())
    nt, nnode = refs["u"].shape
    pred = {key: np.empty((nt, nnode), dtype=np.float32)
            for key in ("u", "v", "p")}
    snapshot = {key: np.empty(nnode, dtype=np.float32)
                for key in ("u", "v", "p")}
    x_col = x.astype(np.float32).reshape(-1, 1)
    y_col = y.astype(np.float32).reshape(-1, 1)
    for start in range(0, nnode, chunk):
        stop = min(start + chunk, nnode)
        sl = slice(start, stop)
        feed_xy = {x_tf: x_col[sl], y_tf: y_col[sl]}
        for it, t in enumerate(times):
            feed = dict(feed_xy)
            feed[t_tf] = np.full((stop - start, 1), t, dtype=np.float32)
            pu, pv, pp = sess.run([u_t, v_t, p_t], feed_dict=feed)
            pred["u"][it, sl] = pu[:, 0]
            pred["v"][it, sl] = pv[:, 0]
            pred["p"][it, sl] = pp[:, 0]
            if it == snapshot_index:
                snapshot["u"][sl] = pu[:, 0]
                snapshot["v"][sl] = pv[:, 0]
                snapshot["p"][sl] = pp[:, 0]
        print(f"{name}: nodes {stop}/{nnode}", flush=True)
    # Direct modal evaluation avoids fitting the network output back in time.
    # Convert the one-sided coefficient to the frozen conventional coefficient.
    mode_feed = {x_tf: x_col, y_tf: y_col}
    v1 = sess.run(v_modes, feed_dict=mode_feed)[0, :, 1].astype(np.complex128) / 2.0
    sess.close()
    return {
        "field_metrics": regional_field_metrics(pred, refs, regions),
        "v1_mode_metrics": regional_complex_metrics(v1, TRUE_V1, regions),
        "metadata": {
            "checkpoint": str(checkpoint),
            "checkpoint_local_nn_functions": str(module_path),
            "prior_for_v1_radial": str(prior_path) if prior_path else None,
            "one_sided_mode_conversion": "network_v1 / 2",
        },
        "snapshot": snapshot,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--Data", type=pathlib.Path, default=DEFAULT_DATA)
    ap.add_argument("--Out", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--Chunk", type=int, default=4000)
    args = ap.parse_args()
    if args.Chunk < 1:
        raise ValueError("--Chunk must be positive")

    # TensorFlow is imported only when this trained-checkpoint evaluator runs.
    import tensorflow as tf  # noqa: WPS433
    tf.compat.v1.disable_eager_execution()
    # TensorFlow 2.21 removed a few TF1 top-level aliases used by the
    # checkpoint-local module.  These aliases select the same math kernels;
    # they do not change the model or its parameters.
    if not hasattr(tf, "real"):
        tf.real = tf.math.real
    if not hasattr(tf, "imag"):
        tf.imag = tf.math.imag

    times, X, Y, U, V, P = load_cache(args.Data)
    idx = strict_crop_indices(X, Y)
    x, y = X[idx], Y[idx]
    refs = {"u": U[:, idx], "v": V[:, idx], "p": P[:, idx]}
    regions = region_masks(x, y)
    global TRUE_V1
    true_modes = temporal_harmonic_coefficients(refs["v"], times, OMEGA_0, 3)
    TRUE_V1 = true_modes[1]
    snapshot_index = times.size // 2

    arm1 = ARMS_ROOT / "01_baseline_physics_only"
    arm15 = ARMS_ROOT / "15_karman_prior_fluct_off"
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
    results = {}
    for name, (module, checkpoint, prior) in configs.items():
        for path in (module, checkpoint):
            if not path.exists():
                raise FileNotFoundError(path)
        results[name] = evaluate_model(
            name, module, checkpoint, prior, x, y, times, refs, regions,
            tf, args.Chunk, snapshot_index
        )

    # Keep a compact representative-state artifact for the field figure; the
    # JSON remains scalar-only and therefore easy to inspect and version.
    snapshot_payload = {
        "x": x.astype(np.float32), "y": y.astype(np.float32),
        "time": np.asarray(times[snapshot_index], dtype=float),
        "u_true": refs["u"][snapshot_index],
        "v_true": refs["v"][snapshot_index],
        "p_true": refs["p"][snapshot_index],
    }
    for name, model_result in results.items():
        for variable, values in model_result.pop("snapshot").items():
            snapshot_payload[f"{name}_{variable}"] = values
    snapshot_path = FRESH_ROOT / "derived" / "a04_snapshot_fields.npz"
    np.savez_compressed(snapshot_path, **snapshot_payload)

    out = {
        "analysis_id": "A04",
        "method": "prior_attribution",
        "status": "verified",
        "data": str(args.Data),
        "snapshots": int(times.size),
        "crop_nodes": int(x.size),
        "regions": {key: int(mask.sum()) for key, mask in regions.items()},
        "metric_contract": str(FRESH_ROOT / "data_contract.md"),
        "snapshot": {"index": int(snapshot_index),
                      "time": float(times[snapshot_index]),
                      "artifact": str(snapshot_path)},
        "models": results,
    }
    args.Out.parent.mkdir(parents=True, exist_ok=True)
    args.Out.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps(out, indent=2, allow_nan=False))
    print("Wrote", args.Out)


if __name__ == "__main__":
    main()
