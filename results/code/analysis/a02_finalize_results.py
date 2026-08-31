"""Register the accepted A02 pressure-tap-count metrics."""
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "derived" / "a02_tap_count_summary.csv"
RESULTS_MASTER = ROOT / "results_master.csv"

FIELDS = [
    "analysis_id", "arm_id", "section", "metric", "region", "value",
    "unit", "reference", "value_type", "source", "status", "notes",
]

METHOD_NOTES = {
    "pressure_only_physics_8_taps": "Pressure-only + physics; 8 cylinder pressure taps.",
    "pressure_only_physics_16_taps": "Pressure-only + physics; 16 cylinder pressure taps.",
    "pressure_only_physics_32_taps": "Pressure-only + physics; 32 cylinder pressure taps.",
}


def main() -> None:
    with SUMMARY.open(newline="") as handle:
        summary_rows = list(csv.DictReader(handle))

    existing = []
    if RESULTS_MASTER.exists():
        with RESULTS_MASTER.open(newline="") as handle:
            existing = [row for row in csv.DictReader(handle)
                        if row.get("analysis_id") != "A02"]

    rows = []
    source = "modes_experiment/fresh_analysis/derived/a02_tap_count_summary.csv"
    for row in summary_rows:
        metric_name = row["metric"]
        if metric_name == "n":
            unit, reference = "count", "region node count"
        elif metric_name == "phase_deg":
            unit, reference = "degrees", "ideal phase 0 deg"
        elif metric_name in {"amp_ratio", "corr"}:
            unit, reference = "dimensionless", "ideal 1"
        else:
            unit, reference = "dimensionless", "CFD"
        rows.append({
            "analysis_id": "A02",
            "arm_id": row["method"],
            "section": "tap_count",
            "metric": f"{row['metric_group']}.{row['quantity']}.{metric_name}",
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
    print(f"Wrote {RESULTS_MASTER} ({len(rows)} A02 rows; {len(existing) + len(rows)} total rows)")


if __name__ == "__main__":
    main()
