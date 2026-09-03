"""Validate the A04 inference result and cross-check legacy field metrics."""
from __future__ import annotations

import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RESULT = ROOT / "derived" / "a04_prior_attribution_metrics.json"
OUT = ROOT / "derived" / "a04_validation.json"
ARMS = ROOT.parents[1] / "4_runs"


def main() -> None:
    result = json.loads(RESULT.read_text())
    checks = {}
    checks["status_verified"] = result.get("status") == "verified"
    checks["snapshots_201"] = result.get("snapshots") == 201
    checks["crop_nodes_51654"] = result.get("crop_nodes") == 51654
    checks["region_counts"] = result.get("regions") == {
        "near-cylinder": 13715, "near-wake": 15248, "far-wake": 16393,
        "far-core": 12460, "other": 6298, "whole-domain": 51654,
    }
    checks["required_models"] = set(result.get("models", {})) == {
        "arm1_baseline", "arm15_v1_radial_trust"
    }

    # The old summaries used the same trained checkpoints but predate the
    # frozen metric contract.  Shared field quantities should nevertheless
    # agree to numerical precision; this catches a wrong wrapper or checkpoint.
    legacy_dirs = {
        "arm1_baseline": "01_baseline_physics_only",
        "arm15_v1_radial_trust": "15_karman_prior_fluct_off",
    }
    region_map = {
        "near-cylinder": "near-cylinder",
        "near-wake": "near-wake",
        "far-wake": "far-wake",
        "other": "other (upstream/off-axis)",
        "whole-domain": "whole domain",
    }
    max_diffs = {}
    for model, dirname in legacy_dirs.items():
        old_path = ARMS / dirname / "regions.json"
        old = json.loads(old_path.read_text())["regions"]
        diffs = []
        for region, old_region in region_map.items():
            for variable in ("u", "v", "p"):
                diffs.append(abs(float(result["models"][model]["field_metrics"][region][variable])
                                 - float(old[old_region][f"E_{variable}"])))
        max_diffs[model] = max(diffs)
    checks["legacy_field_crosscheck"] = all(value < 5e-4 for value in max_diffs.values())

    finite = []
    for model in result["models"].values():
        for region in model["field_metrics"].values():
            finite.extend(region.values())
        for region in model["v1_mode_metrics"].values():
            finite.extend(value for key, value in region.items() if key != "n")
    checks["all_metrics_finite"] = bool(np.all(np.isfinite(np.asarray(finite, dtype=float))))
    checks["all_pass"] = all(checks.values())
    output = {
        "analysis_id": "A04",
        "status": "passed" if checks["all_pass"] else "failed",
        "checks": checks,
        "max_legacy_field_metric_abs_diff": max_diffs,
        "legacy_files": {model: str(ARMS / dirname / "regions.json")
                         for model, dirname in legacy_dirs.items()},
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
