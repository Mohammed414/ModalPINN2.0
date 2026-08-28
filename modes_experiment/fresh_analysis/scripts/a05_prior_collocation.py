"""A05 - does wake-biased collocation improve the prior-assisted reconstruction?

Arm 15 (prior, uniform interior sampling) against arm 10 (prior, wake-biased
regular grid).  Inference only.

This is the one comparison in the project whose two arms are already close in
optimizer effort (34643 vs 26129 L-BFGS evaluations, 1.33x), so it can be read
without an effort caveat on the direction - though the residual 1.33x still
favours arm 15 and is printed below.

Usage:
    cd modes_experiment/fresh_analysis
    ../../.venv_tf_eval/bin/python scripts/a05_prepare_inputs.py
    ../../.venv_tf_eval/bin/python scripts/a05_prior_collocation.py
"""
from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prior_arm_metrics import ARMS_ROOT, evaluate  # noqa: E402

ARMS = {
    "prior_uniform_collocation": ARMS_ROOT / "15_karman_prior_fluct_off",
    "prior_wake_biased_grid": ARMS_ROOT / "arm_10_prior_wake_biased_grid",
}
WAKE_REGIONS = ("near-wake", "far-core", "far-wake")


def main() -> None:
    output = evaluate(analysis_id="A05", method="prior_collocation", arms=ARMS)
    models = output["models"]

    print()
    print("optimizer effort (uncontrolled under --Tmax; report alongside every metric)")
    for name, result in models.items():
        effort = result["effort"]
        print("  %-26s %6d evals  final loss %.3e  stop=%s"
              % (name, effort["lbfgs_evals"], effort["final_total_loss"],
                 effort["stop_reason"]))

    print()
    print("first-harmonic v1 in the wake, absolute-time convention (0 is perfect, 1.0 is zero prediction)")
    print("  %-26s %12s %10s %10s" % ("arm", "region", "rel_L2", "amp_ratio"))
    for name, result in models.items():
        for region in WAKE_REGIONS:
            metrics = result["v1_mode_metrics"][region]
            print("  %-26s %12s %10.4f %10.4f"
                  % (name, region, metrics["rel_L2"], metrics["amp_ratio"]))

    print()
    print("regional field errors (rel_L2 against CFD)")
    print("  %-26s %12s %8s %8s %8s" % ("arm", "region", "u", "v", "p"))
    for name, result in models.items():
        for region in ("near-cylinder",) + WAKE_REGIONS:
            field = result["field_metrics"][region]
            print("  %-26s %12s %8.4f %8.4f %8.4f"
                  % (name, region, field["u"], field["v"], field["p"]))

    # direction, stated in the conservative form: the arm with LESS optimizer
    # effort winning is a lower bound; the arm with less effort losing is not
    # informative on its own.
    names = list(models)
    leaner = min(names, key=lambda n: models[n]["effort"]["lbfgs_evals"])
    richer = max(names, key=lambda n: models[n]["effort"]["lbfgs_evals"])
    print()
    print("effort-aware reading")
    print("  %s trained less (%d vs %d evals)."
          % (leaner, models[leaner]["effort"]["lbfgs_evals"],
             models[richer]["effort"]["lbfgs_evals"]))
    for region in WAKE_REGIONS:
        lean = models[leaner]["v1_mode_metrics"][region]["rel_L2"]
        rich = models[richer]["v1_mode_metrics"][region]["rel_L2"]
        if lean < rich:
            verdict = "%s better despite less effort -> safe as a lower bound" % leaner
        else:
            verdict = "%s better, but it also trained more -> effort not excluded" % richer
        print("  %-10s %s" % (region, verdict))


if __name__ == "__main__":
    main()
