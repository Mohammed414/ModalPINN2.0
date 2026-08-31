"""Register the accepted A01 information-comparison metrics.

The analysis table remains the numerical source of truth.  This small,
idempotent builder copies its tidy rows into ``results_master.csv`` while
preserving rows from every other analysis.
"""
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "derived" / "a01_information_comparison_summary.csv"
RESULTS_MASTER = ROOT / "results_master.csv"

FIELDS = [
    "analysis_id", "arm_id", "section", "metric", "region", "value",
    "unit", "reference", "value_type", "source", "status", "notes",
]

METHOD_NOTES = {
    "pressure_only_physics":
        "Controlled sparse-data comparison: 32 pressure taps + physics.",
    "pressure_and_velocity_probes_physics":
        "Controlled sparse-data comparison: 32 pressure taps + velocity probes + physics.",
    "dense_observations":
        "Representational ceiling only; dense run uses a different optimizer budget.",
}


def main() -> None:
    with SUMMARY.open(newline="") as handle:
        summary_rows = list(csv.DictReader(handle))

    existing = []
    if RESULTS_MASTER.exists():
        with RESULTS_MASTER.open(newline="") as handle:
            existing = [row for row in csv.DictReader(handle)
                        if row.get("analysis_id") != "A01"]

    rows = []
    source = "modes_experiment/fresh_analysis/derived/a01_information_comparison_summary.csv"
    for row in summary_rows:
        group = row["metric_group"]
        metric = f"{group}.{row['quantity']}.{row['metric']}"
        if row["metric"] == "n":
            unit = "count"
            reference = "region node count"
        elif row["metric"] == "phase_deg":
            unit = "degrees"
            reference = "ideal phase 0 deg"
        elif row["metric"] in {"amp_ratio", "corr"}:
            unit = "dimensionless"
            reference = "ideal 1"
        else:
            unit = "dimensionless"
            reference = "CFD"
        rows.append({
            "analysis_id": "A01",
            "arm_id": row["method"],
            "section": "information_comparison",
            "metric": metric,
            "region": row["region"],
            "value": row["value"],
            "unit": unit,
            "reference": reference,
            "value_type": "raw_metric",
            "source": source,
            "status": "accepted",
            "notes": METHOD_NOTES[row["method"]],
        })

    with RESULTS_MASTER.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(existing + rows)
    print(f"Wrote {RESULTS_MASTER} ({len(rows)} A01 rows; {len(existing) + len(rows)} total rows)")


if __name__ == "__main__":
    main()
