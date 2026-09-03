"""Finalize the accepted A04 attribution numbers.

The original metric table remains unchanged and is kept as the source-of-truth
for raw values.  This script creates a second tidy table containing (i) the
percentage reduction from the analytical Karman prior to the prior-assisted
network and (ii) the actual network correction magnitude at the representative
snapshot used by F01.  It also refreshes the A04 rows in ``results_master.csv``
without touching rows belonging to later analyses.
"""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
DERIVED = ROOT / "derived"

PRIOR_JSON = DERIVED / "a04_prior_only_metrics.json"
ATTRIB_JSON = DERIVED / "a04_prior_attribution_metrics.json"
SUMMARY_CSV = DERIVED / "a04_prior_attribution_summary.csv"
SNAPSHOT_NPZ = DERIVED / "a04_snapshot_fields.npz"
CHANGES_CSV = DERIVED / "a04_prior_attribution_changes.csv"
CHANGES_JSON = DERIVED / "a04_prior_attribution_changes.json"
RESULTS_MASTER = ROOT / "results_master.csv"

HYBRID = "arm15_v1_radial_trust"
METHOD_IDS = {
    "prior_only": "karman_prior_only",
    "arm1_baseline": "pressure_only_physics",
    HYBRID: "pressure_only_physics_karman_prior",
}
REGIONS = ["near-cylinder", "near-wake", "far-wake", "far-core", "other", "whole-domain"]
QUANTITIES = ("u", "v", "p")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_prior_field_function():
    """Import the existing NumPy prior implementation instead of duplicating it."""
    path = HERE / "a04_prior_only.py"
    spec = importlib.util.spec_from_file_location("a04_prior_only_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.prior_field


def append_change(rows, *, comparison, region, metric_group, quantity, metric,
                  value, unit, reference, notes):
    rows.append({
        "comparison": comparison,
        "region": region,
        "metric_group": metric_group,
        "quantity": quantity,
        "metric": metric,
        "value": f"{float(value):.12g}",
        "unit": unit,
        "reference": reference,
        "notes": notes,
    })


def main() -> None:
    prior_result = load_json(PRIOR_JSON)
    attrib_result = load_json(ATTRIB_JSON)
    summary_rows = list(csv.DictReader(SUMMARY_CSV.open(newline="")))
    changes = []

    prior = prior_result["field_metrics"]
    hybrid = attrib_result["models"][HYBRID]["field_metrics"]
    prior_v1 = prior_result["v1_mode_metrics"]
    hybrid_v1 = attrib_result["models"][HYBRID]["v1_mode_metrics"]

    # Error reductions are signed: positive means the hybrid has a lower error
    # than the analytical prior, while negative means the network worsened it.
    for region in REGIONS:
        for quantity in QUANTITIES:
            base = float(prior[region][quantity])
            result = float(hybrid[region][quantity])
            append_change(
                changes,
                comparison="network_gain_over_karman_prior",
                region=region,
                metric_group="field",
                quantity=quantity,
                metric="rel_L2_reduction_pct",
                value=100.0 * (base - result) / base,
                unit="percent",
                reference="karman_prior_only",
                notes="Positive values indicate lower relative L2 after adding the network.",
            )

        base = float(prior_v1[region]["rel_L2"])
        result = float(hybrid_v1[region]["rel_L2"])
        append_change(
            changes,
            comparison="network_gain_over_karman_prior",
            region=region,
            metric_group="v1_mode",
            quantity="v",
            metric="rel_L2_reduction_pct",
            value=100.0 * (base - result) / base,
            unit="percent",
            reference="karman_prior_only",
            notes="First-harmonic v1 relative L2 reduction; positive is better.",
        )

        base_amp = float(prior_v1[region]["amp_ratio"])
        result_amp = float(hybrid_v1[region]["amp_ratio"])
        amp_den = abs(base_amp - 1.0)
        append_change(
            changes,
            comparison="network_gain_over_karman_prior",
            region=region,
            metric_group="v1_mode",
            quantity="v",
            metric="amplitude_abs_error_reduction_pct",
            value=100.0 * (amp_den - abs(result_amp - 1.0)) / amp_den if amp_den > 1e-12 else 0.0,
            unit="percent",
            reference="ideal_amplitude_ratio_1",
            notes="Reduction in absolute distance from amplitude ratio 1.",
        )

        base_corr = float(prior_v1[region]["corr"])
        result_corr = float(hybrid_v1[region]["corr"])
        append_change(
            changes,
            comparison="network_gain_over_karman_prior",
            region=region,
            metric_group="v1_mode",
            quantity="v",
            metric="correlation_gain_pp",
            value=100.0 * (result_corr - base_corr),
            unit="percentage_points",
            reference="karman_prior_only",
            notes="Signed increase in normalized complex correlation; ideal value is 1.",
        )

        base_phase = abs(float(prior_v1[region]["phase_deg"]))
        result_phase = abs(float(hybrid_v1[region]["phase_deg"]))
        append_change(
            changes,
            comparison="network_gain_over_karman_prior",
            region=region,
            metric_group="v1_mode",
            quantity="v",
            metric="phase_abs_error_reduction_pct",
            value=100.0 * (base_phase - result_phase) / base_phase if base_phase > 1e-12 else 0.0,
            unit="percent",
            reference="ideal_phase_0_deg",
            notes="Reduction in absolute phase offset from zero degrees.",
        )

    # The trained evaluator retains one representative snapshot for F01.  The
    # correction is measured directly as hybrid minus analytical prior, not
    # inferred from the error metrics.  These values are explicitly marked as
    # snapshot quantities and are not presented as time-averaged corrections.
    snap = np.load(SNAPSHOT_NPZ)
    data_path = Path(attrib_result["data"])
    times = np.asarray(np.load(data_path)["times"], dtype=float).reshape(-1)
    snapshot_index = int(attrib_result["snapshot"]["index"])
    time_value = float(attrib_result["snapshot"]["time"])
    prior_path = REPO / "4_runs" / "15_karman_prior_fluct_off" / "street_prior_used.npz"
    prior_z = np.load(prior_path)
    names = ("Gamma", "Uc", "xf", "r0", "omega", "phase", "amp_scale",
             "scale_p", "ramp", "delta")
    prior_params = {name: float(prior_z[name]) for name in names}
    prior_field = load_prior_field_function()
    x = np.asarray(snap["x"], dtype=float)
    y = np.asarray(snap["y"], dtype=float)
    prior_u, prior_v, prior_p = prior_field(
        x, y, time_value, prior_params, float(times[0])
    )
    prior_snapshot = {"u": prior_u, "v": prior_v, "p": prior_p}
    hybrid_snapshot = {
        "u": np.asarray(snap[f"{HYBRID}_u"], dtype=float),
        "v": np.asarray(snap[f"{HYBRID}_v"], dtype=float),
        "p": np.asarray(snap[f"{HYBRID}_p"], dtype=float),
    }
    reference_snapshot = {
        "u": np.asarray(snap["u_true"], dtype=float),
        "v": np.asarray(snap["v_true"], dtype=float),
        "p": np.asarray(snap["p_true"], dtype=float),
    }
    # The common evaluator defines the same masks used by every A04 metric.
    from evaluate_common import region_masks  # noqa: E402
    masks = region_masks(x, y)
    snapshot_note = f"Representative snapshot index {snapshot_index}, t={time_value:g}; not time-averaged."
    for region in REGIONS:
        mask = np.asarray(masks[region], dtype=bool)
        for quantity in QUANTITIES:
            correction = hybrid_snapshot[quantity][mask] - prior_snapshot[quantity][mask]
            ref = reference_snapshot[quantity][mask]
            prior_values = prior_snapshot[quantity][mask]
            corr_rms = float(np.sqrt(np.mean(correction ** 2)))
            ref_rms = float(np.sqrt(np.mean(ref ** 2)))
            prior_rms = float(np.sqrt(np.mean(prior_values ** 2)))
            append_change(
                changes,
                comparison="learned_correction_snapshot",
                region=region,
                metric_group="learned_correction_snapshot",
                quantity=quantity,
                metric="rms_abs_per_node",
                value=corr_rms,
                unit="field_units",
                reference="hybrid_minus_karman_prior",
                notes=snapshot_note,
            )
            append_change(
                changes,
                comparison="learned_correction_snapshot",
                region=region,
                metric_group="learned_correction_snapshot",
                quantity=quantity,
                metric="relative_to_reference_rms",
                value=corr_rms / ref_rms if ref_rms > 1e-12 else 0.0,
                unit="ratio",
                reference="CFD_snapshot_rms",
                notes=snapshot_note,
            )
            append_change(
                changes,
                comparison="learned_correction_snapshot",
                region=region,
                metric_group="learned_correction_snapshot",
                quantity=quantity,
                metric="relative_to_prior_rms",
                value=corr_rms / prior_rms if prior_rms > 1e-12 else 0.0,
                unit="ratio",
                reference="Karman_prior_snapshot_rms",
                notes=snapshot_note,
            )

    CHANGES_CSV.write_text("")
    with CHANGES_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(changes[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(changes)

    metadata = {
        "analysis_id": "A04",
        "status": "verified",
        "source_metrics": str(SUMMARY_CSV),
        "change_table": str(CHANGES_CSV),
        "formula_error_reduction": "100 * (prior_error - hybrid_error) / prior_error",
        "formula_correction": "hybrid_snapshot - analytical_prior_snapshot",
        "snapshot_index": snapshot_index,
        "snapshot_time": time_value,
        "snapshot_artifact": str(SNAPSHOT_NPZ),
        "rows": len(changes),
    }
    CHANGES_JSON.write_text(json.dumps(metadata, indent=2) + "\n")

    # Refresh only A04 rows so this script remains safe once A01 onward has
    # populated the same master table.
    master_fields = ["analysis_id", "arm_id", "section", "metric", "region",
                     "value", "unit", "reference", "value_type", "source",
                     "status", "notes"]
    existing = []
    if RESULTS_MASTER.exists():
        existing = list(csv.DictReader(RESULTS_MASTER.open(newline="")))
    existing = [row for row in existing if row.get("analysis_id") != "A04"]
    master_rows = []
    summary_source = "6_analysis/derived/a04_prior_attribution_summary.csv"
    for row in summary_rows:
        metric = f"{row['metric_group']}.{row['quantity']}.{row['metric']}"
        master_rows.append({
            "analysis_id": "A04",
            "arm_id": METHOD_IDS[row["method"]],
            "section": "prior_attribution",
            "metric": metric,
            "region": row["region"],
            "value": row["value"],
            "unit": "dimensionless",
            "reference": "CFD",
            "value_type": "raw_metric",
            "source": summary_source,
            "status": "accepted",
            "notes": "201-snapshot common-contract evaluation.",
        })
    changes_source = "6_analysis/derived/a04_prior_attribution_changes.csv"
    for row in changes:
        if row["comparison"] == "network_gain_over_karman_prior":
            arm_id = METHOD_IDS[HYBRID]
            value_type = "derived_change"
        else:
            arm_id = METHOD_IDS[HYBRID]
            value_type = "snapshot_correction"
        master_rows.append({
            "analysis_id": "A04",
            "arm_id": arm_id,
            "section": "prior_attribution",
            "metric": f"{row['metric_group']}.{row['quantity']}.{row['metric']}",
            "region": row["region"],
            "value": row["value"],
            "unit": row["unit"],
            "reference": row["reference"],
            "value_type": value_type,
            "source": changes_source,
            "status": "accepted",
            "notes": row["notes"],
        })
    with RESULTS_MASTER.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=master_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(existing + master_rows)

    print(f"Wrote {CHANGES_CSV} ({len(changes)} rows)")
    print(f"Wrote {CHANGES_JSON}")
    print(f"Wrote {RESULTS_MASTER} ({len(master_rows)} A04 rows)")


if __name__ == "__main__":
    main()
