"""Flatten A04 prior/model JSON outputs into a plotting-friendly CSV."""
from __future__ import annotations

import csv
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
PRIOR = ROOT / "derived" / "a04_prior_only_metrics.json"
MODELS = ROOT / "derived" / "a04_prior_attribution_metrics.json"
OUT = ROOT / "derived" / "a04_prior_attribution_summary.csv"


def main() -> None:
    prior = json.loads(PRIOR.read_text())
    models = json.loads(MODELS.read_text())
    rows = []

    def add_field(method, metrics):
        for region, variables in metrics.items():
            for variable, value in variables.items():
                rows.append({"method": method, "region": region,
                             "metric_group": "field", "quantity": variable,
                             "metric": "rel_L2", "value": value})

    def add_v1(method, metrics):
        for region, values in metrics.items():
            for metric in ("rel_L2", "amp_ratio", "corr", "phase_deg"):
                rows.append({"method": method, "region": region,
                             "metric_group": "v1_mode", "quantity": "v",
                             "metric": metric, "value": values[metric]})

    add_field("prior_only", prior["field_metrics"])
    add_v1("prior_only", prior["v1_mode_metrics"])
    labels = {"arm1_baseline": "arm1_baseline",
              "arm15_v1_radial_trust": "arm15_v1_radial_trust"}
    for key, label in labels.items():
        add_field(label, models["models"][key]["field_metrics"])
        add_v1(label, models["models"][key]["v1_mode_metrics"])

    OUT.write_text("")
    with OUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", "region",
                                                     "metric_group", "quantity",
                                                     "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote", OUT, "rows", len(rows))


if __name__ == "__main__":
    main()
