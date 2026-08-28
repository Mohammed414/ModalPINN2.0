"""Register the accepted A03 collocation-strategy metrics.

A03 is accepted as ``complete_with_caveat``. The three arms isolate the
interior collocation sampling strategy as an input *flag*, but not as an
input *effort*: the uniform arm terminated at 5,503 L-BFGS evaluations while
the two wake-biased arms ran 43,676 and 37,713, a 6.9-7.9x gap. That gap is
structural, not incidental -- see ``findings.md``, "Training effort" -- so it
is recorded per row here rather than left to the reader.

The confound has a direction, which is what makes the arm reportable: extra
effort favours the wake-biased arms, so every result in which they are
*worse* is conservative, and every result in which they are *better* is
unattributable between sampling and effort.
"""
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "derived" / "a03_collocation_summary.csv"
RESULTS_MASTER = ROOT / "results_master.csv"

FIELDS = [
    "analysis_id", "arm_id", "section", "metric", "region", "value",
    "unit", "reference", "value_type", "source", "status", "notes",
]

# L-BFGS evaluations per arm, from derived/a03_input_manifest.json.
LBFGS_EVALS = {
    "uniform_collocation": 5503,
    "wake_biased_random_collocation": 43676,
    "wake_biased_grid_collocation": 37713,
}

METHOD_NOTES = {
    "uniform_collocation": (
        "Pressure-only + physics; 32 taps; uniform interior collocation. "
        "Non-converged lower bound (5,503 L-BFGS evals; matched-effort re-run "
        "attempted and discarded 2026-08-28)."
    ),
    "wake_biased_random_collocation": (
        "Pressure-only + physics; 32 taps; wake-biased random collocation. "
        "7.9x the uniform arm's L-BFGS effort (43,676 evals); effort not controlled."
    ),
    "wake_biased_grid_collocation": (
        "Pressure-only + physics; 32 taps; wake-biased grid collocation. "
        "6.9x the uniform arm's L-BFGS effort (37,713 evals); effort not controlled."
    ),
}


def main() -> None:
    with SUMMARY.open(newline="") as handle:
        summary_rows = list(csv.DictReader(handle))

    existing = []
    if RESULTS_MASTER.exists():
        with RESULTS_MASTER.open(newline="") as handle:
            existing = [row for row in csv.DictReader(handle)
                        if row.get("analysis_id") != "A03"]

    rows = []
    source = "modes_experiment/fresh_analysis/derived/a03_collocation_summary.csv"
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
            "analysis_id": "A03",
            "arm_id": row["method"],
            "section": "collocation_strategy",
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

    # Record the effort gap itself as a first-class row per arm, so a reader of
    # results_master.csv alone can see the confound without opening the manifest.
    for method, evals in LBFGS_EVALS.items():
        rows.append({
            "analysis_id": "A03",
            "arm_id": method,
            "section": "collocation_strategy",
            "metric": "training_effort.lbfgs_evals",
            "region": "n/a",
            "value": str(evals),
            "unit": "count",
            "reference": "uniform_collocation = 5503",
            "value_type": "confound_audit",
            "source": "modes_experiment/fresh_analysis/derived/a03_input_manifest.json",
            "status": "accepted",
            "notes": (
                "Effort is not controlled across A03 arms; the gap favours the "
                "wake-biased arms, so their worse results are conservative and "
                "their better results are unattributable."
            ),
        })

    with RESULTS_MASTER.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(existing + rows)
    print(f"Wrote {RESULTS_MASTER} ({len(rows)} A03 rows; {len(existing) + len(rows)} total rows)")


if __name__ == "__main__":
    main()
