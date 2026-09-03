"""Evaluate the 8-, 16-, and 32-pressure-tap reconstructions for A02.

This is inference only: saved weights are restored through the checkpoint-local
forward wrappers and evaluated on the common 201-snapshot crop and metrics.
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
ARMS_ROOT = REPO / "4_runs"
DATA = REPO / "1_data" / "flow_cache.npz"
OUT = ROOT / "derived" / "a02_tap_count_metrics.json"
SUMMARY = ROOT / "derived" / "a02_tap_count_summary.csv"
VALIDATION = ROOT / "derived" / "a02_validation.json"

sys.path.insert(0, str(HERE))
import a04_prior_attribution as inference  # noqa: E402
from evaluate_common import (  # noqa: E402
    OMEGA_0,
    region_masks,
    strict_crop_indices,
    temporal_harmonic_coefficients,
)


CONFIGS = {
    "pressure_only_physics_8_taps": (
        ARMS_ROOT / "08_taps_08" / "training_run" / "NN_functions.py",
        ARMS_ROOT / "08_taps_08" / "training_run" / "DNN2_100_100_4_tanh.pickle",
        8,
    ),
    "pressure_only_physics_16_taps": (
        ARMS_ROOT / "09_taps_16" / "training_run" / "NN_functions.py",
        ARMS_ROOT / "09_taps_16" / "training_run" / "DNN2_100_100_4_tanh.pickle",
        16,
    ),
    "pressure_only_physics_32_taps": (
        ARMS_ROOT / "01_baseline_physics_only" / "training_run" / "NN_functions.py",
        ARMS_ROOT / "01_baseline_physics_only" / "training_run" / "DNN2_100_100_4_tanh.pickle",
        32,
    ),
}


def main(chunk: int = 4000) -> None:
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    if not hasattr(tf, "real"):
        tf.real = tf.math.real
    if not hasattr(tf, "imag"):
        tf.imag = tf.math.imag

    times, X, Y, U, V, P = inference.load_cache(DATA)
    idx = strict_crop_indices(X, Y)
    x, y = X[idx], Y[idx]
    refs = {"u": U[:, idx], "v": V[:, idx], "p": P[:, idx]}
    regions = region_masks(x, y)
    inference.TRUE_V1 = temporal_harmonic_coefficients(
        refs["v"], times, OMEGA_0, 3
    )[1]
    snapshot_index = times.size // 2
    results = {}
    snapshot_payload = {
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "time": np.asarray(times[snapshot_index], dtype=float),
        "u_true": refs["u"][snapshot_index],
        "v_true": refs["v"][snapshot_index],
        "p_true": refs["p"][snapshot_index],
    }

    for name, (module, checkpoint, ntaps) in CONFIGS.items():
        for path in (module, checkpoint):
            if not path.exists():
                raise FileNotFoundError(path)
        result = inference.evaluate_model(
            name, module, checkpoint, None, x, y, times, refs, regions,
            tf, chunk, snapshot_index,
        )
        snapshot = result.pop("snapshot")
        for variable, values in snapshot.items():
            snapshot_payload[f"{name}_{variable}"] = values
        result["ntaps"] = ntaps
        results[name] = result

    snapshot_path = ROOT / "derived" / "a02_snapshot_fields.npz"
    np.savez_compressed(snapshot_path, **snapshot_payload)
    output = {
        "analysis_id": "A02",
        "method": "tap_count",
        "status": "verified",
        "data": str(DATA),
        "snapshots": int(times.size),
        "crop_nodes": int(x.size),
        "regions": {key: int(mask.sum()) for key, mask in regions.items()},
        "metric_contract": str(ROOT / "data_contract.md"),
        "input_manifest": str(ROOT / "derived" / "a02_input_manifest.json"),
        "snapshot": {
            "index": int(snapshot_index),
            "time": float(times[snapshot_index]),
            "artifact": str(snapshot_path),
        },
        "models": results,
    }
    OUT.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")

    fields = ["method", "ntaps", "region", "metric_group", "quantity", "metric", "value"]
    with SUMMARY.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for method, result in results.items():
            for region, metrics in result["field_metrics"].items():
                for quantity, value in metrics.items():
                    writer.writerow({
                        "method": method, "ntaps": result["ntaps"],
                        "region": region, "metric_group": "field",
                        "quantity": quantity, "metric": "rel_L2", "value": value,
                    })
            for region, metrics in result["v1_mode_metrics"].items():
                for metric, value in metrics.items():
                    writer.writerow({
                        "method": method, "ntaps": result["ntaps"],
                        "region": region, "metric_group": "v1_mode",
                        "quantity": "v", "metric": metric, "value": value,
                    })

    with SUMMARY.open(newline="") as handle:
        values = [float(row["value"]) for row in csv.DictReader(handle)]
    all_finite = bool(values) and all(np.isfinite(values))
    validation = {
        "analysis_id": "A02",
        "status": "passed" if all_finite and times.size == 201 and x.size == 51654 else "failed",
        "checks": {
            "snapshots_201": bool(times.size == 201),
            "crop_nodes_51654": bool(x.size == 51654),
            "all_metrics_finite": all_finite,
            "three_tap_counts": sorted(r["ntaps"] for r in results.values()) == [8, 16, 32],
        },
        "input_manifest": str(ROOT / "derived" / "a02_input_manifest.json"),
        "summary": str(SUMMARY),
    }
    VALIDATION.write_text(json.dumps(validation, indent=2) + "\n")
    print("Wrote", OUT)
    print("Wrote", SUMMARY)
    print("Wrote", VALIDATION)


if __name__ == "__main__":
    main()
