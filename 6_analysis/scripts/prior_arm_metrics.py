"""Shared inference-only evaluator for the prior-assisted arms (A05, A06).

A05 and A06 are the same computation over different arm sets, so the loop lives
here and the two analysis scripts only declare their arms.  Nothing here trains:
saved weights are restored as constants through each checkpoint's own
``NN_functions.py``, exactly as ``a02_tap_count.py`` and
``a04_prior_attribution.py`` do.

Two departures from the A02/A04 scripts, both deliberate:

1.  **First-harmonic phase convention.**  ``evaluate_common`` returns the CFD
    coefficient in the absolute-time convention used by the network.  The old
    window convention, based on ``tau = t - t0`` with ``t0 = 400``, differs by
    a fixed rotation: under that legacy convention a *perfect* network mode
    scores ``rel_L2 = 0.289`` and ``phase_deg = +16.63`` rather than 0.  The
    legacy values are retained alongside the corrected primary values only as
    an audit trail.
2.  **Optimizer effort travels with the metrics.**  ``lbfgs_evals`` and the
    final losses are copied into the metrics JSON and the tidy CSV, so no table
    downstream can present these arms as if effort were controlled.
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys
from typing import Dict, Mapping, Optional

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[0]
ARMS_ROOT = REPO / "4_runs"
DEFAULT_DATA = REPO / "1_data" / "flow_cache.npz"

sys.path.insert(0, str(HERE))
import a04_prior_attribution as inference  # noqa: E402
from arm_inputs import (  # noqa: E402
    assert_metric_identity,
    phase_rotation_deg,
    rotate_metrics,
    run_facts,
)
from evaluate_common import (  # noqa: E402
    OMEGA_0,
    region_masks,
    strict_crop_indices,
    temporal_harmonic_coefficients,
)

EXPECTED_SNAPSHOTS = 201
EXPECTED_CROP_NODES = 51654

# The forward wrapper in a04_prior_attribution hardcodes the v1 radial trust
# geometry. Every arm evaluated here must have trained with exactly those
# values, or the restored model would be evaluated through a different ansatz
# than it was trained with.
REQUIRED_TRUST_FLAGS = {
    "V1TrustRho": "0.60",
    "V1TrustXStart": "3.0",
    "V1TrustXWidth": "0.30",
    "V1TrustYMax": "2.0",
    "V1TrustYWidth": "0.20",
}


def _check_trust_geometry(name: str, flags: Mapping[str, object]) -> None:
    if "V1RadialTrust" not in flags:
        raise AssertionError("%s did not train with --V1RadialTrust; the prior "
                             "forward path does not apply to it" % name)
    for flag, expected in REQUIRED_TRUST_FLAGS.items():
        observed = flags.get(flag)
        if observed is None or float(observed) != float(expected):
            raise AssertionError(
                "%s trained with --%s %s but the evaluator applies %s"
                % (name, flag, observed, expected))


def evaluate(*, analysis_id: str, method: str, arms: Mapping[str, pathlib.Path],
             extra_columns: Optional[Mapping[str, Mapping[str, float]]] = None,
             data: pathlib.Path = DEFAULT_DATA, chunk: int = 4000) -> Dict[str, object]:
    """Evaluate a set of prior-assisted arms under the common metric contract."""
    import tensorflow as tf  # noqa: WPS433
    tf.compat.v1.disable_eager_execution()
    if not hasattr(tf, "real"):
        tf.real = tf.math.real
    if not hasattr(tf, "imag"):
        tf.imag = tf.math.imag

    derived = ROOT / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    prefix = analysis_id.lower()

    times, X, Y, U, V, P = inference.load_cache(data)
    idx = strict_crop_indices(X, Y)
    x, y = X[idx], Y[idx]
    refs = {"u": U[:, idx], "v": V[:, idx], "p": P[:, idx]}
    regions = region_masks(x, y)

    # --- phase convention -------------------------------------------------
    true_modes = temporal_harmonic_coefficients(refs["v"], times, OMEGA_0, 3)
    v1_absolute = true_modes[1]
    theta_deg = phase_rotation_deg(float(times[0]), OMEGA_0)
    inference.TRUE_V1 = v1_absolute

    snapshot_index = times.size // 2
    snapshot_payload = {
        "x": x.astype(np.float32), "y": y.astype(np.float32),
        "time": np.asarray(times[snapshot_index], dtype=float),
        "u_true": refs["u"][snapshot_index],
        "v_true": refs["v"][snapshot_index],
        "p_true": refs["p"][snapshot_index],
    }

    results: Dict[str, object] = {}
    for name, arm_dir in arms.items():
        facts = run_facts(arm_dir)
        _check_trust_geometry(name, facts["flags"])
        module = pathlib.Path(facts["checkpoint_local_nn_functions"])
        checkpoint = pathlib.Path(facts["checkpoint"])
        prior = pathlib.Path(facts["street_prior"]) if facts["street_prior"] else None
        for path in (module, checkpoint):
            if not path.exists():
                raise FileNotFoundError(path)
        if prior is None:
            raise FileNotFoundError("no street_prior_used.npz in %s" % arm_dir)

        result = inference.evaluate_model(
            name, module, checkpoint, prior, x, y, times, refs, regions,
            tf, chunk, snapshot_index,
        )
        snapshot = result.pop("snapshot")
        for variable, values in snapshot.items():
            snapshot_payload["%s_%s" % (name, variable)] = values

        window_metrics = {}
        for region, metrics in result["v1_mode_metrics"].items():
            assert_metric_identity(metrics)
            window_metrics[region] = rotate_metrics(metrics, theta_deg)
        result["v1_mode_metrics_window_convention"] = window_metrics
        result["effort"] = facts["effort"]
        result["provenance"] = {
            "run_directory": facts["run_directory"],
            "checkpoint_sha256": facts["checkpoint_sha256"],
            "nn_functions_sha256": facts["nn_functions_sha256"],
            "street_prior": facts["street_prior"],
        }
        if extra_columns and name in extra_columns:
            result.update(extra_columns[name])
        results[name] = result

    snapshot_path = derived / ("%s_snapshot_fields.npz" % prefix)
    np.savez_compressed(snapshot_path, **snapshot_payload)

    output = {
        "analysis_id": analysis_id,
        "method": method,
        "status": "verified",
        "data": str(data),
        "snapshots": int(times.size),
        "crop_nodes": int(x.size),
        "regions": {key: int(mask.sum()) for key, mask in regions.items()},
        "metric_contract": str(ROOT / "data_contract.md"),
        "input_manifest": str(derived / ("%s_input_manifest.json" % prefix)),
        "v1_phase_convention": {
            "primary": "absolute time (network convention); a perfect model scores rel_L2 0 and phase 0",
            "legacy": "window tau = t - t0 (superseded audit value); a perfect model scores rel_L2 0.289 and phase +16.63 deg",
            "rotation_deg": theta_deg,
            "t0": float(times[0]),
            "omega_0": OMEGA_0,
        },
        "snapshot": {"index": int(snapshot_index),
                     "time": float(times[snapshot_index]),
                     "artifact": str(snapshot_path)},
        "models": results,
    }
    metrics_path = derived / ("%s_%s_metrics.json" % (prefix, method))
    metrics_path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")

    # --- tidy table -------------------------------------------------------
    summary_path = derived / ("%s_%s_summary.csv" % (prefix, method))
    fields = ["method", "region", "metric_group", "quantity", "metric", "value",
              "lbfgs_evals", "final_total_loss"]
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name, result in results.items():
            effort = result["effort"]
            base = {"method": name, "lbfgs_evals": effort["lbfgs_evals"],
                    "final_total_loss": effort["final_total_loss"]}
            for region, metrics in result["field_metrics"].items():
                for quantity, value in metrics.items():
                    writer.writerow(dict(base, region=region, metric_group="field",
                                         quantity=quantity, metric="rel_L2", value=value))
            for group, key in (("v1_mode", "v1_mode_metrics"),
                               ("v1_mode_window_convention", "v1_mode_metrics_window_convention")):
                for region, metrics in result[key].items():
                    for metric, value in metrics.items():
                        writer.writerow(dict(base, region=region, metric_group=group,
                                             quantity="v", metric=metric, value=value))

    with summary_path.open(newline="") as handle:
        values = [float(row["value"]) for row in csv.DictReader(handle)]
    all_finite = bool(values) and all(np.isfinite(values))
    validation = {
        "analysis_id": analysis_id,
        "status": "passed" if (all_finite
                               and times.size == EXPECTED_SNAPSHOTS
                               and x.size == EXPECTED_CROP_NODES) else "failed",
        "checks": {
            "snapshots_%d" % EXPECTED_SNAPSHOTS: bool(times.size == EXPECTED_SNAPSHOTS),
            "crop_nodes_%d" % EXPECTED_CROP_NODES: bool(x.size == EXPECTED_CROP_NODES),
            "all_metrics_finite": all_finite,
            "arms_evaluated": sorted(results),
            "rel_L2_identity_verified": True,
            "trust_geometry_matches_training": True,
            "distinct_checkpoints": len({result["provenance"]["checkpoint_sha256"]
                                         for result in results.values()}) == len(results),
            "effort_matched_within_20pct": bool(
                max(r["effort"]["lbfgs_evals"] for r in results.values())
                / min(r["effort"]["lbfgs_evals"] for r in results.values()) <= 1.2),
        },
        "input_manifest": str(derived / ("%s_input_manifest.json" % prefix)),
        "summary": str(summary_path),
    }
    validation_path = derived / ("%s_validation.json" % prefix)
    validation_path.write_text(json.dumps(validation, indent=2) + "\n")

    print("Wrote", metrics_path)
    print("Wrote", summary_path)
    print("Wrote", validation_path)
    print("validation:", validation["status"],
          "| effort matched within 20%:",
          validation["checks"]["effort_matched_within_20pct"])
    return output
