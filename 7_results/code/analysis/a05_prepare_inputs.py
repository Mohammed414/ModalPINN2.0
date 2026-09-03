"""Audit the matched prior-assisted runs for A05 (collocation strategy).

A05 asks whether wake-biased collocation improves the prior-assisted
reconstruction.  Arm 15 (prior, uniform interior sampling) against arm 10
(prior, wake-biased regular grid); the intended difference is the single flag
``--WakeBiasedGridSampling``.

Usage:
    ../../.venv_tf_eval/bin/python scripts/a05_prepare_inputs.py
(no TensorFlow needed here - this reads run records only)
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
ARMS = ROOT.parents[1] / "4_runs"
OUT = ROOT / "derived" / "a05_input_manifest.json"

sys.path.insert(0, str(HERE))
from arm_inputs import build_manifest  # noqa: E402

RUNS = {
    "prior_uniform_collocation": ARMS / "15_karman_prior_fluct_off",
    "prior_wake_biased_grid": ARMS / "10_prior_wake_biased_grid",
}


def main() -> None:
    manifest = build_manifest(
        analysis_id="A05",
        question="Does wake-biased collocation improve the prior-assisted reconstruction?",
        runs=RUNS,
        intended_change_flags=["WakeBiasedGridSampling"],
        intended_change="interior collocation sampling strategy, with the Karman v1 radial prior active in both arms",
        data_contract=ROOT / "data_contract.md",
        out_path=OUT,
    )
    print("Wrote", OUT)
    print("status:", manifest["status"])
    comparison = manifest["controlled_comparison"]
    print("identical forward code:", comparison["identical_forward_code"])
    print("flag differences:")
    for item in comparison["flag_differences"]:
        print("  ", item["flag"], "=", item["values"])
    if comparison["unintended_flag_differences"]:
        print("UNINTENDED differences present:")
        print(json.dumps(comparison["unintended_flag_differences"], indent=2))
    audit = manifest["effort_audit"]
    print("L-BFGS evals:", audit["lbfgs_evals"],
          "| spread", audit["spread_ratio"], "x",
          "| matched within 20%:", audit["matched_within_20pct"])


if __name__ == "__main__":
    main()
