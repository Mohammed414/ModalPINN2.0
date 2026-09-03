"""Build one audited, filter-friendly results table for the dissertation.

The numerical source remains the per-analysis outputs.  This script enriches
``results_master.csv`` and the matched Gappy POD metrics with human-readable
configuration columns, verifies cross-analysis identities, and records the
first-harmonic time-origin correction explicitly.  It does not train or run a
model.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
WORKSPACE = REPO.parent
DERIVED = ROOT / "derived"
MASTER = ROOT / "results_master.csv"
GAPPY = REPO / "modes_experiment" / "gappy_pod_final" / "results" / "metrics.csv"
GAPPY_CONFIG = GAPPY.parent / "configuration.json"

OUT = DERIVED / "verified_results.csv"
CORRECTION_OUT = DERIVED / "v1_phase_correction_audit.csv"
AUDIT_OUT = DERIVED / "verified_results_audit.json"

ANALYSES = {
    "A01": ("Measurement information", "complete"),
    "A02": ("Pressure-tap count", "complete"),
    "A03": ("Collocation strategy", "complete_with_effort_caveat"),
    "A04": ("Karman-prior attribution", "complete_with_single_run_caveat"),
    "A05": ("Prior plus collocation", "complete_with_effort_caveat"),
    "A06": ("Pressure-noise sensitivity", "complete_with_effort_and_single_seed_caveat"),
    "GPOD": ("Gappy POD identifiability and noise", "verified_in_sample_single_seed"),
}

# family, label, observations, taps, collocation, prior, noise percent
METHODS: Dict[Tuple[str, str], Tuple[str, str, str, object, str, bool, object]] = {
    ("A01", "pressure_only_physics"): ("ModalPINN", "Pressure only + physics", "cylinder pressure", 32, "uniform", False, ""),
    ("A01", "pressure_and_velocity_probes_physics"): ("ModalPINN", "Pressure + velocity probes + physics", "cylinder pressure + 40 interior velocity probes", 32, "uniform", False, ""),
    ("A01", "dense_observations"): ("ModalPINN", "Dense observations + physics", "5000 dense u,v,p samples", "", "uniform", False, ""),
    ("A02", "pressure_only_physics_8_taps"): ("ModalPINN", "Pressure only + physics (8 taps)", "cylinder pressure", 8, "uniform", False, ""),
    ("A02", "pressure_only_physics_16_taps"): ("ModalPINN", "Pressure only + physics (16 taps)", "cylinder pressure", 16, "uniform", False, ""),
    ("A02", "pressure_only_physics_32_taps"): ("ModalPINN", "Pressure only + physics (32 taps)", "cylinder pressure", 32, "uniform", False, ""),
    ("A03", "uniform_collocation"): ("ModalPINN", "Pressure only + physics (uniform collocation)", "cylinder pressure", 32, "uniform", False, ""),
    ("A03", "wake_biased_random_collocation"): ("ModalPINN", "Pressure only + physics (wake-biased random)", "cylinder pressure", 32, "wake-biased random", False, ""),
    ("A03", "wake_biased_grid_collocation"): ("ModalPINN", "Pressure only + physics (wake-biased grid)", "cylinder pressure", 32, "wake-biased grid", False, ""),
    ("A04", "karman_prior_only"): ("Analytical prior", "Karman prior alone", "32 tap-derived prior parameters", 32, "n/a", True, ""),
    ("A04", "pressure_only_physics"): ("ModalPINN", "Pressure only + physics", "cylinder pressure", 32, "uniform", False, ""),
    ("A04", "pressure_only_physics_karman_prior"): ("Hybrid ModalPINN", "Pressure + physics + Karman prior", "cylinder pressure", 32, "uniform", True, ""),
    ("A05", "prior_physics_uniform_collocation"): ("Hybrid ModalPINN", "Prior-assisted (uniform collocation)", "cylinder pressure", 32, "uniform", True, ""),
    ("A05", "prior_physics_wake_biased_grid"): ("Hybrid ModalPINN", "Prior-assisted (wake-biased grid)", "cylinder pressure", 32, "wake-biased grid", True, ""),
    ("A06", "prior_physics_noise_00pct"): ("Hybrid ModalPINN", "Prior-assisted (clean pressure)", "cylinder pressure", 32, "uniform", True, 0.0),
    ("A06", "prior_physics_noise_01pct"): ("Hybrid ModalPINN", "Prior-assisted (1% pressure noise)", "cylinder pressure", 32, "uniform", True, 1.0),
    ("A06", "prior_physics_noise_05pct"): ("Hybrid ModalPINN", "Prior-assisted (5% pressure noise)", "cylinder pressure", 32, "uniform", True, 5.0),
    ("A06", "prior_physics_noise_10pct"): ("Hybrid ModalPINN", "Prior-assisted (10% pressure noise)", "cylinder pressure", 32, "uniform", True, 10.0),
}

FIELDS = [
    "result_id", "analysis_id", "analysis", "analysis_status", "method_family",
    "method_id", "method", "observations", "tap_count", "collocation",
    "prior_active", "noise_percent", "metric_group", "quantity", "metric",
    "region", "value", "unit", "reference", "value_type", "phase_convention",
    "verification", "source", "notes",
]


def read_csv(path: Path) -> List[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric_parts(metric: str) -> Tuple[str, str, str]:
    parts = metric.split(".")
    if len(parts) >= 3:
        return parts[0], parts[1], ".".join(parts[2:])
    if len(parts) == 2:
        return parts[0], "n/a", parts[1]
    return "other", "n/a", metric


def gappy_noise_label(sigma: float) -> object:
    if sigma < 0:
        return "projection ceiling"
    levels = {0.0: 0.0, 0.00047265: 1.0, 0.0023633: 5.0, 0.0047265: 10.0}
    for key, value in levels.items():
        if math.isclose(sigma, key, rel_tol=0.0, abs_tol=1e-12):
            return value
    raise AssertionError("unexpected Gappy POD noise sigma %r" % sigma)


def build_results() -> List[dict]:
    rows = []
    for index, source in enumerate(read_csv(MASTER), start=1):
        analysis_id = source["analysis_id"]
        metadata = METHODS[(analysis_id, source["arm_id"])]
        family, label, observations, taps, collocation, prior, noise = metadata
        group, quantity, metric = metric_parts(source["metric"])
        phase_convention = (
            "absolute time; prediction and CFD reference aligned"
            if group == "v1_mode" else "not applicable"
        )
        analysis, analysis_status = ANALYSES[analysis_id]
        rows.append({
            "result_id": "MP-%04d" % index,
            "analysis_id": analysis_id,
            "analysis": analysis,
            "analysis_status": analysis_status,
            "method_family": family,
            "method_id": source["arm_id"],
            "method": label,
            "observations": observations,
            "tap_count": taps,
            "collocation": collocation,
            "prior_active": prior,
            "noise_percent": noise,
            "metric_group": group,
            "quantity": quantity,
            "metric": metric,
            "region": source["region"],
            "value": source["value"],
            "unit": source["unit"],
            "reference": source["reference"],
            "value_type": source["value_type"],
            "phase_convention": phase_convention,
            "verification": "verified_primary",
            "source": source["source"],
            "notes": source["notes"],
        })

    offset = len(rows)
    for index, source in enumerate(read_csv(GAPPY), start=1):
        sigma = float(source["noise_sigma"])
        projection = sigma < 0
        noise = gappy_noise_label(sigma)
        group = source["quantity"]
        label = ("Rank-6 POD projection ceiling" if projection
                 else "Gappy POD (rank 6)")
        metric = source["metric"]
        unit = "degrees" if metric == "phase_deg" else "dimensionless"
        reference = ("ideal phase 0 deg" if metric == "phase_deg" else
                     "ideal 1" if metric in {"amplitude_ratio", "correlation"}
                     else "CFD")
        phase_convention = (
            "common absolute-time fit; pairwise metric is time-origin invariant"
            if "harmonic" in group else "not applicable"
        )
        analysis, analysis_status = ANALYSES["GPOD"]
        rows.append({
            "result_id": "GP-%04d" % index,
            "analysis_id": "GPOD",
            "analysis": analysis,
            "analysis_status": analysis_status,
            "method_family": "Gappy POD",
            "method_id": "pod_projection" if projection else "gappy_pod_rank6",
            "method": label,
            "observations": "exact coefficients (projection ceiling)" if projection else "32 cylinder pressure taps",
            "tap_count": "" if projection else 32,
            "collocation": "n/a",
            "prior_active": False,
            "noise_percent": noise,
            "metric_group": group,
            "quantity": source["variable"],
            "metric": metric,
            "region": source["region"],
            "value": source["value"],
            "unit": unit,
            "reference": reference,
            "value_type": "projection_ceiling" if projection else "raw_metric",
            "phase_convention": phase_convention,
            "verification": "verified_in_sample",
            "source": "modes_experiment/gappy_pod_final/results/metrics.csv",
            "notes": ("Basis and reconstruction use the same 201 snapshots; this is an in-sample diagnostic. "
                      "Noise levels use one shared-seed realisation."),
        })
    assert len(rows) == offset + len(read_csv(GAPPY))
    return rows


def write_csv(path: Path, fields: Iterable[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_correction_audit() -> List[dict]:
    from arm_inputs import phase_rotation_deg, rotate_metrics
    from evaluate_common import OMEGA_0

    sources = {
        "A01": (DERIVED / "a01_information_comparison_summary.csv", None),
        "A02": (DERIVED / "a02_tap_count_summary.csv", None),
        "A03": (DERIVED / "a03_collocation_summary.csv", None),
        "A04": (DERIVED / "a04_prior_attribution_summary.csv",
                {"arm1_baseline", "arm15_v1_radial_trust"}),
    }
    a04_labels = {
        "arm1_baseline": "Pressure only + physics",
        "arm15_v1_radial_trust": "Pressure + physics + Karman prior",
    }
    theta = phase_rotation_deg(400.0, OMEGA_0)
    output = []
    for analysis_id, (path, selected) in sources.items():
        rows = read_csv(path)
        methods = sorted({row["method"] for row in rows})
        if selected is not None:
            methods = [method for method in methods if method in selected]
        for method in methods:
            for region in sorted({row["region"] for row in rows}):
                metrics = {
                    row["metric"]: float(row["value"])
                    for row in rows
                    if row["method"] == method and row["region"] == region
                    and row["metric_group"] == "v1_mode"
                }
                if not {"rel_L2", "amp_ratio", "corr", "phase_deg"} <= metrics.keys():
                    continue
                corrected = dict(metrics, n=int(metrics.get("n", 0)))
                legacy = rotate_metrics(corrected, theta)
                label = a04_labels.get(method, METHODS.get((analysis_id, method), ("", method, "", "", "", False, ""))[1])
                for metric in ("rel_L2", "phase_deg"):
                    old = legacy[metric]
                    new = corrected[metric]
                    output.append({
                        "analysis_id": analysis_id,
                        "method_id": method,
                        "method": label,
                        "region": region,
                        "metric": metric,
                        "legacy_window_value": old,
                        "corrected_absolute_value": new,
                        "absolute_change": new - old,
                        "correction_reason": "CFD coefficient rotated to the absolute-time convention used by the direct ModalPINN mode",
                        "verification": "recomputed from saved checkpoint; amplitude ratio and correlation unchanged",
                    })
    return output


def value(rows: List[dict], **criteria: object) -> float:
    matches = [row for row in rows if all(str(row[key]) == str(expected)
                                         for key, expected in criteria.items())]
    if len(matches) != 1:
        raise AssertionError("expected one row for %r, found %d" % (criteria, len(matches)))
    return float(matches[0]["value"])


def audit(rows: List[dict], correction_rows: List[dict]) -> dict:
    validation_paths = [DERIVED / ("a%02d_validation.json" % i) for i in range(1, 7)]
    validations = {path.stem: json.loads(path.read_text())["status"]
                   for path in validation_paths}
    finite = all(math.isfinite(float(row["value"])) for row in rows)
    keys = [(row["analysis_id"], row["method_id"], row["metric_group"],
             row["quantity"], row["metric"], row["region"], row["value_type"])
            + (str(row["noise_percent"]),)
            for row in rows]

    # The same pressure-only checkpoint appears under four scientific
    # questions.  Every overlapping primary metric must be identical.
    aliases = {
        "A01": "pressure_only_physics",
        "A02": "pressure_only_physics_32_taps",
        "A03": "uniform_collocation",
        "A04": "pressure_only_physics",
    }
    sets = {}
    for analysis_id, method_id in aliases.items():
        sets[analysis_id] = {
            (row["metric_group"], row["quantity"], row["metric"], row["region"]): float(row["value"])
            for row in rows if row["analysis_id"] == analysis_id
            and row["method_id"] == method_id and row["value_type"] == "raw_metric"
        }
    common = set.intersection(*(set(items) for items in sets.values()))
    baseline_max_diff = max(
        max(sets[aid][key] for aid in sets) - min(sets[aid][key] for aid in sets)
        for key in common
    )

    # Reconstruct every A05/A06 learned-contribution row from its stated
    # formula. This closes the provenance gap in the older packaged workbook:
    # learned contribution = prior-only rel_L2 - arm rel_L2, same region.
    prior_errors = {
        row["region"]: float(row["value"])
        for row in rows
        if row["analysis_id"] == "A04"
        and row["method_id"] == "karman_prior_only"
        and row["metric_group"] == "v1_mode"
        and row["quantity"] == "v" and row["metric"] == "rel_L2"
        and row["value_type"] == "raw_metric"
    }
    learned_rows = [row for row in rows if row["metric"] == "learned_contribution"]
    learned_differences = []
    for learned in learned_rows:
        arm_error = value(
            rows,
            analysis_id=learned["analysis_id"],
            method_id=learned["method_id"],
            metric_group="v1_mode",
            quantity="v",
            metric="rel_L2",
            region=learned["region"],
            value_type="raw_metric",
        )
        expected = prior_errors[learned["region"]] - arm_error
        learned_differences.append(abs(float(learned["value"]) - expected))
    learned_max_diff = max(learned_differences, default=float("inf"))

    config = json.loads(GAPPY_CONFIG.read_text())
    data_path = REPO / config["data_file"]
    key_values = {
        "pressure_only_far_core_v1_relative_L2": value(
            rows, analysis_id="A04", method_id="pressure_only_physics",
            metric_group="v1_mode", quantity="v", metric="rel_L2",
            region="far-core", value_type="raw_metric"),
        "prior_only_far_core_v1_relative_L2": value(
            rows, analysis_id="A04", method_id="karman_prior_only",
            metric_group="v1_mode", quantity="v", metric="rel_L2",
            region="far-core", value_type="raw_metric"),
        "hybrid_far_core_v1_relative_L2": value(
            rows, analysis_id="A04", method_id="pressure_only_physics_karman_prior",
            metric_group="v1_mode", quantity="v", metric="rel_L2",
            region="far-core", value_type="raw_metric"),
        "dense_far_core_field_v_relative_L2": value(
            rows, analysis_id="A01", method_id="dense_observations",
            metric_group="field", quantity="v", metric="rel_L2",
            region="far-core", value_type="raw_metric"),
        "gappy_far_core_field_v_relative_L2": value(
            rows, analysis_id="GPOD", method_id="gappy_pod_rank6",
            metric_group="field", quantity="v", metric="relative_L2",
            region="far-core", value_type="raw_metric", noise_percent=0.0),
        "gappy_far_core_v1_relative_L2": value(
            rows, analysis_id="GPOD", method_id="gappy_pod_rank6",
            metric_group="first_harmonic_v1", quantity="v", metric="relative_L2",
            region="far-core", value_type="raw_metric", noise_percent=0.0),
    }
    checks = {
        "all_a01_to_a06_validations_passed": all(status == "passed" for status in validations.values()),
        "all_values_finite": finite,
        "no_duplicate_primary_keys": len(keys) == len(set(keys)),
        "expected_modal_rows_945": sum(row["analysis_id"] != "GPOD" for row in rows) == 945,
        "expected_gappy_rows_210": sum(row["analysis_id"] == "GPOD" for row in rows) == 210,
        "no_legacy_window_metrics_in_primary_table": not any("window" in row["metric_group"] for row in rows),
        "same_checkpoint_identical_across_a01_to_a04": baseline_max_diff < 1e-12,
        "all_36_learned_contributions_rederived_exactly": (
            len(learned_rows) == 36 and learned_max_diff < 1e-12
        ),
        "phase_correction_rows_present": len(correction_rows) == 132,
        "dataset_hash_matches_gappy_configuration": sha256(data_path) == config["data_sha256"],
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "generated": "2026-08-31",
        "checks": checks,
        "validation_statuses": validations,
        "row_counts": dict(Counter(row["analysis_id"] for row in rows)),
        "same_checkpoint_max_abs_difference": baseline_max_diff,
        "learned_contribution_max_abs_difference": learned_max_diff,
        "first_harmonic_primary_convention": "absolute time; a perfect direct ModalPINN mode scores relative L2 0 and phase 0",
        "superseded_convention": "window tau=t-t0; retained only in v1_phase_correction_audit.csv",
        "key_verified_values": key_values,
        "sources": {
            "modal_results": str(MASTER),
            "gappy_results": str(GAPPY),
            "data_contract": str(ROOT / "data_contract.md"),
            "dataset": str(data_path),
            "dataset_sha256": sha256(data_path),
        },
    }


def main() -> None:
    rows = build_results()
    corrections = build_correction_audit()
    write_csv(OUT, FIELDS, rows)
    correction_fields = [
        "analysis_id", "method_id", "method", "region", "metric",
        "legacy_window_value", "corrected_absolute_value", "absolute_change",
        "correction_reason", "verification",
    ]
    write_csv(CORRECTION_OUT, correction_fields, corrections)
    result = audit(rows, corrections)
    AUDIT_OUT.write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "passed":
        raise AssertionError(json.dumps(result, indent=2))
    print("Wrote", OUT, len(rows), "rows")
    print("Wrote", CORRECTION_OUT, len(corrections), "rows")
    print("Wrote", AUDIT_OUT, result["status"])


if __name__ == "__main__":
    main()
