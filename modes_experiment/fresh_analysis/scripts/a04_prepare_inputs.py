"""Prepare and validate the A04 prior-attribution input manifest.

This does not evaluate a field or train a model. It records the exact CFD,
geometry, prior, checkpoints, and run metadata used by the later A04 evaluator,
and explicitly reports which settings are matched versus intentionally
different.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


FRESH_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
ARMS = REPO_ROOT / "modes_experiment" / "runs" / "arms"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_options(command: Iterable[str]) -> Dict[str, Any]:
    """Parse the simple ``--flag value`` command-record convention."""
    values: Dict[str, Any] = {}
    command = list(command)
    i = 0
    while i < len(command):
        token = command[i]
        if not token.startswith("--"):
            i += 1
            continue
        key = token[2:]
        if i + 1 < len(command) and not command[i + 1].startswith("--"):
            values[key] = command[i + 1]
            i += 2
        else:
            values[key] = True
            i += 1
    return values


def load_arm(name: str) -> Dict[str, Any]:
    arm_dir = ARMS / name
    record_path = arm_dir / "run_record.json"
    summary_path = arm_dir / "arm_summary.json"
    checkpoint = arm_dir / "training_run" / "DNN2_100_100_4_tanh.pickle"
    record = json.loads(record_path.read_text())
    summary = json.loads(summary_path.read_text())
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return {
        "name": name,
        "directory": str(arm_dir.relative_to(REPO_ROOT)),
        "record_path": str(record_path.relative_to(REPO_ROOT)),
        "summary_path": str(summary_path.relative_to(REPO_ROOT)),
        "checkpoint_path": str(checkpoint.relative_to(REPO_ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "record": record,
        "summary": summary,
        "options": command_options(record["command"]),
    }


def main() -> None:
    arm1 = load_arm("01_baseline_physics_only")
    arm15 = load_arm("15_karman_prior_fluct_off")
    prior = ARMS / "15_karman_prior_fluct_off" / "street_prior_used.npz"
    prior_copy = REPO_ROOT / "modes_experiment" / "prior_only_evaluation" / "street_prior_used.npz"
    for path in (prior, prior_copy):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(prior) != sha256(prior_copy):
        raise AssertionError("Arm-15 and prior-only prior files differ")

    data = REPO_ROOT / "data" / "fixed_cylinder_atRe100"
    cache = REPO_ROOT / "data" / "flow_cache.npz"
    geometry = FRESH_ROOT / "derived" / "a00_geometry.npz"
    for path in (data, cache, geometry):
        if not path.is_file():
            raise FileNotFoundError(path)

    # These are the settings that must match for an attribution comparison.
    matched_keys = ("SparseData", "PressureOnly", "NTaps", "Seed", "Nmes",
                    "Nint", "multigrid", "Ngrid", "NgridTurn", "WidthLayer",
                    "Nmodes", "FreestreamBC")
    matched: Dict[str, bool] = {}
    for key in matched_keys:
        matched[key] = arm1["options"].get(key) == arm15["options"].get(key)
    if not all(matched.values()):
        raise AssertionError("required A04 settings are not matched: %s" % matched)

    differences = {
        "prior_wrapper": {
            "arm1": "ordinary ModalPINN mode output",
            "arm15": "V1RadialTrust with street_prior_used.npz, rho=0.60, xstart=3.0",
        },
        "optimizer_evaluations": {
            "arm1": arm1["record"].get("lbfgs_evals"),
            "arm15": arm15["record"].get("lbfgs_evals"),
            "interpretation": "training convergence histories are retained as metadata; they are not treated as an A04 control parameter",
        },
    }

    def artifact(path: Path, role: str, *, same_as: str = "") -> Dict[str, str]:
        return {
            "role": role,
            "path": str(path.relative_to(REPO_ROOT.parent)),
            "sha256": sha256(path),
            "same_as": same_as,
        }

    manifest = {
        "analysis_id": "A04",
        "question": "Does the trained network improve the Karman prior or merely reproduce it?",
        "status": "inputs_verified",
        "evaluation_contract": "modes_experiment/fresh_analysis/data_contract.md",
        "artifacts": [
            artifact(data, "canonical CFD reference"),
            artifact(cache, "validated float32 cache"),
            artifact(geometry, "strict crop, regions, and sensor mapping"),
            artifact(REPO_ROOT / arm1["checkpoint_path"], "Arm 1 checkpoint"),
            artifact(REPO_ROOT / arm15["checkpoint_path"], "Arm 15 checkpoint"),
            artifact(prior, "Arm 15 Karman prior", same_as=str(prior_copy.relative_to(REPO_ROOT))),
        ],
        "arms": {
            "arm1": {
                "run_directory": arm1["directory"],
                "run_record": arm1["record_path"],
                "arm_summary": arm1["summary_path"],
                "checkpoint": arm1["checkpoint_path"],
                "command_options": arm1["options"],
            },
            "arm15": {
                "run_directory": arm15["directory"],
                "run_record": arm15["record_path"],
                "arm_summary": arm15["summary_path"],
                "checkpoint": arm15["checkpoint_path"],
                "command_options": arm15["options"],
                "prior": str(prior.relative_to(REPO_ROOT)),
            },
        },
        "matched_settings": matched,
        "intentional_differences": differences,
        "interpretation_guard": "Arm 15 is prior-assisted through its V1 radial trust wrapper; Arm 1 is the matched prior-off baseline. The comparison attributes changes to the prior-assisted function class, not to a pure optimizer ablation.",
    }
    out = FRESH_ROOT / "derived" / "a04_input_manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print("Wrote", out)


if __name__ == "__main__":
    main()
