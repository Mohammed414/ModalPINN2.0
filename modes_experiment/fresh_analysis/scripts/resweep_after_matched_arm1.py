"""Re-run A01/A02/A03/A04 once the matched-effort Arm 1 checkpoint lands.

Nothing here can run until ``runs/arms/01b_baseline_matched`` exists (i.e. the
Colab session for ``notebooks/matched_effort/01b_baseline_matched_effort_T4.ipynb``
has finished and its output has been unpacked). This script is safe to inspect
and dry-run before that: `--check` reports whether the checkpoint is present
and, once it is, whether it actually met its effort target, without touching
anything.

Why a symlink swap rather than editing ten call sites: a01/a02/a03/a04 (and
their sub-scripts a04_pressure_gauge_check.py, a04_v1_absolute_check.py, ...)
each hardcode the literal path fragment ``01_baseline_physics_only`` rather
than importing a shared constant, so there is no single point to patch. This
script instead points that path at the new checkpoint at the filesystem
level, which lets every existing, already-reviewed script run unmodified.

    python scripts/resweep_after_matched_arm1.py --check     # safe, read-only
    python scripts/resweep_after_matched_arm1.py --swap       # rename + symlink
    python scripts/resweep_after_matched_arm1.py --run        # re-run A01/A02/A03/A04
    python scripts/resweep_after_matched_arm1.py --restore    # undo the swap

``--swap`` renames the original checkpoint directory to
``01_baseline_physics_only_superseded_5503evals`` (nothing is deleted) and
creates ``01_baseline_physics_only`` as a symlink to
``01b_baseline_matched``. ``--restore`` reverses exactly that operation.
``--run`` executes the four evaluation pipelines via subprocess (each does
its own heavy TF import; keeping them as separate processes matches how they
have always been run in this workspace) and prints a before/after diff of
every quoted v1 rel_L2 value already in ``derived/``, so what changed is
visible before anything is copied into results_master.csv or findings.md.
That last step -- deciding what to accept -- is deliberately left manual.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ARMS_ROOT = ROOT.parents[1] / "modes_experiment" / "runs" / "arms"
DERIVED = ROOT / "derived"

ORIGINAL = ARMS_ROOT / "01_baseline_physics_only"
SUPERSEDED = ARMS_ROOT / "01_baseline_physics_only_superseded_5503evals"
MATCHED = ARMS_ROOT / "01b_baseline_matched"

# partner arms this checkpoint must meet or beat, per the matched-effort notebook
PARTNERS = [
    ("A02 8 taps", 27130), ("A02 16 taps", 21868), ("A04 prior arm 15", 34643),
    ("A01 dense reference", 40001), ("A03 wake-biased grid", 37713),
    ("A03 wake-biased random", 43676),
]

# (script, cwd-relative to ROOT) in dependency order; a04 needs three follow-on
# scripts to regenerate its tidy tables after the metrics themselves change
PIPELINE = [
    ("scripts/a01_information_comparison.py", None),
    ("scripts/a02_tap_count.py", None),
    ("scripts/a03_collocation_strategy.py", None),
    ("scripts/a04_prior_attribution.py", None),
    ("scripts/a04_make_summary.py", None),
    ("scripts/a04_finalize_results.py", None),
    ("scripts/a04_pressure_gauge_check.py", None),
    ("scripts/a04_v1_absolute_check.py", None),
]

# metrics files worth diffing before/after, and the v1 rel_L2 path to read
METRICS_FILES = [
    "a01_information_comparison_metrics.json",
    "a02_tap_count_metrics.json",
    "a03_collocation_metrics.json",
    "a04_prior_attribution_metrics.json",
]
REGIONS = ("near-cylinder", "near-wake", "far-core", "far-wake")


def check() -> bool:
    if ORIGINAL.is_symlink():
        print("Already swapped: 01_baseline_physics_only -> %s" % ORIGINAL.resolve().name)
        target = ORIGINAL
    elif not MATCHED.exists():
        print("Not ready: %s does not exist yet." % MATCHED)
        print("Waiting on: Colab run of notebooks/matched_effort/01b_baseline_matched_effort_T4.ipynb,")
        print("then unpack its Drive output into that path.")
        return False
    else:
        target = MATCHED

    record_path = target / "run_record.json"
    if not record_path.exists():
        print("Found %s but it has no run_record.json -- unpack looks incomplete." % target)
        return False
    record = json.loads(record_path.read_text())
    evals = record.get("lbfgs_evals")
    if evals is None:
        print("run_record.json has no lbfgs_evals -- unexpected, inspect manually.")
        return False

    print("Checkpoint found: %s (%s L-BFGS evaluations)" % (target.name, format(evals, ",")))
    print("%-26s %10s %8s" % ("compared against", "evals", "verdict"))
    all_met = True
    for name, partner_evals in PARTNERS:
        met = evals >= partner_evals
        all_met &= met
        print("%-26s %10s %8s" % (name, format(partner_evals, ","), "met" if met else "SHORT"))
    if not all_met:
        print("\nSome partners are not met. Re-running is still useful (A01/A02 are met, which is")
        print("most of what was blocked), but note any SHORT rows as a caveat rather than silently")
        print("treating every comparison as matched.")
    return True


def swap() -> None:
    if ORIGINAL.is_symlink():
        print("Already swapped, nothing to do. Use --restore first if you want to redo it.")
        return
    if not ORIGINAL.exists():
        raise SystemExit("Expected %s to exist (it should be the original arm 1)." % ORIGINAL)
    if not MATCHED.exists():
        raise SystemExit("Expected %s to exist -- run --check first." % MATCHED)
    if SUPERSEDED.exists():
        raise SystemExit("%s already exists; refuse to overwrite, inspect manually." % SUPERSEDED)

    ORIGINAL.rename(SUPERSEDED)
    ORIGINAL.symlink_to(MATCHED.name)
    print("Renamed %s -> %s" % (ORIGINAL.name, SUPERSEDED.name))
    print("Created symlink %s -> %s" % (ORIGINAL.name, MATCHED.name))
    print("Every script that hardcodes '01_baseline_physics_only' now reads the matched-effort weights.")


def restore() -> None:
    if not ORIGINAL.is_symlink():
        print("Nothing to restore: %s is not a symlink." % ORIGINAL)
        return
    if not SUPERSEDED.exists():
        raise SystemExit("Cannot restore: %s is missing." % SUPERSEDED)
    ORIGINAL.unlink()
    SUPERSEDED.rename(ORIGINAL)
    print("Restored %s from %s" % (ORIGINAL.name, SUPERSEDED.name))


def _snapshot_v1(path: Path) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out = {}
    for model, r in data.get("models", {}).items():
        for region in REGIONS:
            v1 = r.get("v1_mode_metrics", {}).get(region)
            if v1:
                out[(model, region)] = round(float(v1["rel_L2"]), 4)
    return out


def run() -> None:
    if not ORIGINAL.is_symlink():
        raise SystemExit("Run --swap first -- the pipelines still point at the old checkpoint.")

    before = {f: _snapshot_v1(DERIVED / f) for f in METRICS_FILES}
    backup_dir = DERIVED / "_pre_resweep_backup"
    backup_dir.mkdir(exist_ok=True)
    for f in METRICS_FILES:
        src = DERIVED / f
        if src.exists():
            shutil.copy2(src, backup_dir / f)

    for script, cwd in PIPELINE:
        print("\n=== running %s ===" % script)
        result = subprocess.run(
            [sys.executable, script], cwd=str(ROOT), capture_output=True, text=True)
        print(result.stdout[-2000:])
        if result.returncode != 0:
            print(result.stderr[-3000:], file=sys.stderr)
            raise SystemExit("%s failed (exit %d) -- stopping before later steps run on stale "
                              "inputs." % (script, result.returncode))

    print("\n=== v1 rel_L2 changes (old checkpoint -> matched-effort checkpoint) ===")
    for f in METRICS_FILES:
        after = _snapshot_v1(DERIVED / f)
        keys = sorted(set(before[f]) | set(after))
        if not keys:
            continue
        print("\n%s" % f)
        for key in keys:
            old_v, new_v = before[f].get(key), after.get(key)
            if old_v == new_v:
                continue
            model, region = key
            print("  %-32s %-14s %8s -> %8s" % (
                model, region, "n/a" if old_v is None else old_v,
                "n/a" if new_v is None else new_v))

    print("\nPre-resweep metrics backed up to %s for comparison." % backup_dir)
    print("Nothing has been written to results_master.csv, arm_matrix.csv, findings.md,")
    print("or figures/final/ -- review the diff above, then update those by hand as with A05/A06.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="report readiness, read-only")
    group.add_argument("--swap", action="store_true", help="point A01/A02/A03/A04 at the new checkpoint")
    group.add_argument("--run", action="store_true", help="re-run the four evaluation pipelines")
    group.add_argument("--restore", action="store_true", help="undo --swap")
    args = parser.parse_args()

    if args.check:
        check()
    elif args.swap:
        swap()
    elif args.run:
        run()
    elif args.restore:
        restore()
