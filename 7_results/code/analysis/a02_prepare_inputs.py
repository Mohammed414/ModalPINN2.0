"""Audit the matched 8-, 16-, and 32-tap runs for A02."""
from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ARMS = ROOT.parents[1] / "4_runs"
OUT = ROOT / "derived" / "a02_input_manifest.json"

RUNS = {
    "pressure_only_physics_8_taps": ("08_taps_08", 8),
    "pressure_only_physics_16_taps": ("09_taps_16", 16),
    "pressure_only_physics_32_taps": ("01_baseline_physics_only", 32),
}

EXPECTED = {
    "nmodes": 4,
    "width": 25,
    "sparse_data": True,
    "pressure_only": True,
    "freestream_bc": True,
    "seed": "0",
    "nmes": "5000",
    "nint": "50000",
    "multigrid": True,
    "ngrid": "5",
    "ngrid_turn": "200",
}


def command_dict(command):
    flags = {}
    i = 0
    while i < len(command):
        token = command[i]
        if token.startswith("--"):
            key = token[2:]
            if i + 1 < len(command) and not command[i + 1].startswith("--"):
                flags[key] = command[i + 1]
                i += 2
            else:
                flags[key] = True
                i += 1
        else:
            i += 1
    return flags


def main():
    entries = {}
    mismatches = []
    for name, (directory, expected_taps) in RUNS.items():
        path = ARMS / directory
        record_path = path / "run_record.json"
        if not record_path.exists():
            raise FileNotFoundError(record_path)
        record = json.loads(record_path.read_text())
        flags = command_dict(record["command"])
        checkpoint = path / "training_run" / str(record["weights"])
        nn_functions = path / "training_run" / "NN_functions.py"
        settings = {
            "nmodes": record["nmodes"],
            "width": record["width"],
            "ntaps": int(flags["NTaps"]),
            "sparse_data": "SparseData" in flags,
            "pressure_only": "PressureOnly" in flags,
            "freestream_bc": "FreestreamBC" in flags,
            "seed": flags.get("Seed"),
            "nmes": flags.get("Nmes"),
            "nint": flags.get("Nint"),
            "multigrid": "multigrid" in flags,
            "ngrid": flags.get("Ngrid"),
            "ngrid_turn": flags.get("NgridTurn"),
            "adam_ran": bool(record.get("adam_ran", False)),
            "lbfgs_evals": record.get("lbfgs_evals"),
        }
        for key, expected in {**EXPECTED, "ntaps": expected_taps}.items():
            if settings[key] != expected:
                mismatches.append({"method": name, "setting": key,
                                   "expected": expected, "observed": settings[key]})
        entries[name] = {
            "run_directory": str(path),
            "run_label": record["arm"],
            "checkpoint": str(checkpoint),
            "checkpoint_exists": checkpoint.exists(),
            "checkpoint_local_nn_functions": str(nn_functions),
            "nn_functions_exists": nn_functions.exists(),
            "settings": settings,
        }

    manifest = {
        "analysis_id": "A02",
        "status": "verified_inputs" if not mismatches else "input_mismatch",
        "question": "Does increasing the cylinder pressure-tap count improve reconstruction?",
        "data_contract": str(ROOT / "data_contract.md"),
        "candidates": entries,
        "controlled_comparison": {
            "methods": list(RUNS),
            "intended_change": "number of uniformly spaced cylinder pressure taps",
            "interpretation": "All three runs use pressure-only sparse data, the same network/training-size settings, seed 0, FreestreamBC, multigrid settings, and Adam. Only NTaps changes as an input setting; convergence effort is reported but not treated as a controlled parameter.",
        },
        "mismatches": mismatches,
    }
    OUT.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Wrote", OUT)
    print("status:", manifest["status"])
    for name, entry in entries.items():
        print(name, "checkpoint exists =", entry["checkpoint_exists"])
    if mismatches:
        print("mismatches:", json.dumps(mismatches, indent=2))


if __name__ == "__main__":
    main()
