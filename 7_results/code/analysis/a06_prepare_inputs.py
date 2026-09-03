"""Audit the prior-assisted noise runs for A06 (pressure-noise robustness).

Arm 15 (clean taps) against arms 11, 12, 13 at --Noise 4.7265e-04, 2.3633e-03
and 4.7265e-03, i.e. the 1 %, 5 % and 10 % levels.  The intended difference is
the single flag ``--Noise``.

Two things this manifest is expected to surface rather than hide:

*   ``--LBFGSCheckpointIters`` is present in arms 11 and 13 and absent in arms
    15 and 12.  It drives a read-only accepted-step callback that copies the
    packed parameter vector and writes five ``.npy`` files, so it does not
    change the objective, the data, or the search direction; it does add a
    small per-iteration cost.  Reported as an unintended difference because it
    is one, with that characterisation recorded here rather than assumed away.
*   The optimizer effort spread, which for this group is about 11x
    (34643 / 12047 / 5376 / 3156 evaluations).  All three noisy arms trained
    less than the clean reference, which biases the comparison toward "noise
    hurts".  Read ``effort_audit`` before interpreting any A06 metric.

Usage:
    ../../.venv_tf_eval/bin/python scripts/a06_prepare_inputs.py
(no TensorFlow needed here - this reads run records only)
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
ARMS = ROOT.parents[1] / "4_runs"
OUT = ROOT / "derived" / "a06_input_manifest.json"

sys.path.insert(0, str(HERE))
from arm_inputs import build_manifest  # noqa: E402

# Noise is an absolute standard deviation on the tap pressures; the labels are
# the percentages used in the arm names, and the values are exact multiples of
# the 1 % level (x5 and x10), which is asserted below.
RUNS = {
    "prior_noise_00pct": ARMS / "15_karman_prior_fluct_off",
    "prior_noise_01pct": ARMS / "11_prior_noise_01pct",
    "prior_noise_05pct": ARMS / "12_prior_noise_05pct",
    "prior_noise_10pct": ARMS / "13_prior_noise_10pct",
}
NOISE_LABELS = {
    "prior_noise_00pct": 0.0,
    "prior_noise_01pct": 1.0,
    "prior_noise_05pct": 5.0,
    "prior_noise_10pct": 10.0,
}


def main() -> None:
    manifest = build_manifest(
        analysis_id="A06",
        question="Is the prior-assisted reconstruction robust to pressure noise?",
        runs=RUNS,
        intended_change_flags=["Noise"],
        intended_change="standard deviation of additive noise on the cylinder tap pressures",
        data_contract=ROOT / "data_contract.md",
        out_path=OUT,
    )

    # record the noise level per arm, and check the stated percentages really
    # are 1x / 5x / 10x of the same base level
    levels = {}
    for name, entry in manifest["candidates"].items():
        raw = entry["flags"].get("Noise")
        levels[name] = float(raw) if raw not in (None, True) else 0.0
    base = levels["prior_noise_01pct"]
    for name, expected in (("prior_noise_05pct", 5.0), ("prior_noise_10pct", 10.0)):
        ratio = levels[name] / base
        if abs(ratio - expected) > 1e-3:
            raise AssertionError("%s is %.4gx the 1%% level, expected %gx"
                                 % (name, ratio, expected))
    manifest["noise_levels"] = {
        "flag_values": levels,
        "labels_pct": NOISE_LABELS,
        "units": "absolute standard deviation on tap pressure (nondimensional)",
    }
    OUT.write_text(json.dumps(manifest, indent=2) + "\n")

    print("Wrote", OUT)
    print("status:", manifest["status"])
    comparison = manifest["controlled_comparison"]
    print("identical forward code:", comparison["identical_forward_code"])
    print("noise levels:", levels)
    print("flag differences:")
    for item in comparison["flag_differences"]:
        print("  ", item["flag"], "=", item["values"])
    if comparison["unintended_flag_differences"]:
        print()
        print("UNINTENDED differences (expected: LBFGSCheckpointIters, read-only callback):")
        for item in comparison["unintended_flag_differences"]:
            print("  ", item["flag"], "=", item["values"])
    audit = manifest["effort_audit"]
    print()
    print("L-BFGS evals:", audit["lbfgs_evals"])
    print("effort spread:", audit["spread_ratio"], "x | matched within 20%:",
          audit["matched_within_20pct"])
    if not audit["matched_within_20pct"]:
        print("The arms have unequal evaluation counts, which are not an accuracy proxy.")
        print("Report the effort and seed caveats; do not infer a directional bound.")


if __name__ == "__main__":
    main()
