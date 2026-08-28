"""A06 - is the prior-assisted reconstruction robust to pressure noise?

Arm 15 (clean) against arms 11, 12, 13 at the 1 %, 5 % and 10 % noise levels.
Inference only; evaluation is always against the clean CFD reference.

Read this before writing anything from the output.  All three noisy arms trained
LESS than the clean reference (34643 vs 12047 / 5376 / 3156 L-BFGS evaluations,
an 11x spread), because SciPy's ftol test stops these runs on float32 rounding
noise at an unpredictable iteration rather than at convergence.  The bias
therefore runs one way, and only one of the two possible outcomes is a result:

*   noisy arms as good as the clean arm  ->  robustness holds, as a lower bound.
    Less-trained arms matching a better-trained one cannot be explained away by
    effort, so the claim is safe and no re-runs are needed.
*   noisy arms worse than the clean arm  ->  uninformative.  Noise and
    truncation are not separable here, and arms 11/12/13 would need re-running
    at matched effort (see notebooks/matched_effort/) before A06 can claim
    anything.

The script prints which of the two it is.  It deliberately does not compute a
'noise sensitivity' slope, because a slope fitted across arms of unequal effort
would read as a physical result while partly measuring training length.

Usage:
    cd modes_experiment/fresh_analysis
    ../../.venv_tf_eval/bin/python scripts/a06_prepare_inputs.py
    ../../.venv_tf_eval/bin/python scripts/a06_pressure_noise.py
"""
from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prior_arm_metrics import ARMS_ROOT, evaluate  # noqa: E402

ARMS = {
    "prior_noise_00pct": ARMS_ROOT / "15_karman_prior_fluct_off",
    "prior_noise_01pct": ARMS_ROOT / "arm_11_prior_noise_01pct",
    "prior_noise_05pct": ARMS_ROOT / "arm_12_prior_noise_05pct",
    "prior_noise_10pct": ARMS_ROOT / "arm_13_prior_noise_10pct",
}
NOISE_PCT = {"prior_noise_00pct": 0.0, "prior_noise_01pct": 1.0,
             "prior_noise_05pct": 5.0, "prior_noise_10pct": 10.0}
CLEAN = "prior_noise_00pct"
WAKE_REGIONS = ("near-wake", "far-core")
# A noisy arm counts as "matching" the clean arm if its wake v1 error is no
# more than 2% worse in relative terms. Deliberately loose: the question is
# whether noise causes a qualitative loss of the wake, not a third decimal.
MATCH_TOLERANCE = 0.02


def main() -> None:
    output = evaluate(
        analysis_id="A06", method="pressure_noise", arms=ARMS,
        extra_columns={name: {"noise_pct": pct} for name, pct in NOISE_PCT.items()},
    )
    models = output["models"]
    order = sorted(models, key=lambda name: NOISE_PCT[name])

    print()
    print("effort and noise per arm")
    print("  %-20s %6s %8s %14s %s" % ("arm", "noise%", "evals", "final loss", "stop reason"))
    for name in order:
        effort = models[name]["effort"]
        print("  %-20s %6.1f %8d %14.3e %s"
              % (name, NOISE_PCT[name], effort["lbfgs_evals"],
                 effort["final_total_loss"], effort["stop_reason"]))

    print()
    print("wake first harmonic v1, absolute-time convention (1.0 = zero prediction)")
    print("  %-20s %12s %10s %10s" % ("arm", "region", "rel_L2", "amp_ratio"))
    for name in order:
        for region in WAKE_REGIONS:
            metrics = models[name]["v1_mode_metrics"][region]
            print("  %-20s %12s %10.4f %10.4f"
                  % (name, region, metrics["rel_L2"], metrics["amp_ratio"]))

    print()
    print("field errors, near cylinder and far core")
    print("  %-20s %12s %8s %8s %8s" % ("arm", "region", "u", "v", "p"))
    for name in order:
        for region in ("near-cylinder", "far-core"):
            field = models[name]["field_metrics"][region]
            print("  %-20s %12s %8.4f %8.4f %8.4f"
                  % (name, region, field["u"], field["v"], field["p"]))

    # ---- the decision this script exists to support ----------------------
    clean_evals = models[CLEAN]["effort"]["lbfgs_evals"]
    print()
    print("effort-aware verdict")
    informative = True
    for name in order:
        if name == CLEAN:
            continue
        trained_less = models[name]["effort"]["lbfgs_evals"] < clean_evals
        worse_regions = []
        for region in WAKE_REGIONS:
            noisy = models[name]["v1_mode_metrics"][region]["rel_L2"]
            clean = models[CLEAN]["v1_mode_metrics"][region]["rel_L2"]
            if noisy > clean * (1.0 + MATCH_TOLERANCE):
                worse_regions.append("%s (%.4f vs %.4f)" % (region, noisy, clean))
        if not worse_regions:
            print("  %-20s matches the clean arm in the wake%s -> ROBUSTNESS SUPPORTED"
                  % (name, " despite training less" if trained_less else ""))
        elif trained_less:
            informative = False
            print("  %-20s worse in %s, and it trained less (%d vs %d evals)"
                  % (name, "; ".join(worse_regions),
                     models[name]["effort"]["lbfgs_evals"], clean_evals))
            print("  %-20s -> CONFOUNDED: noise and truncation not separable"
                  % "")
        else:
            print("  %-20s worse in %s with at least as much effort -> noise effect is real"
                  % (name, "; ".join(worse_regions)))

    print()
    if informative:
        print("A06 can be written from these checkpoints: every degradation is either")
        print("absent or attributable to noise rather than to training length.")
    else:
        print("A06 cannot claim a noise effect from these checkpoints. Either narrow the")
        print("claim to the arms marked ROBUSTNESS SUPPORTED, or re-run the confounded")
        print("arms at matched effort with notebooks/matched_effort/ before writing.")


if __name__ == "__main__":
    main()
