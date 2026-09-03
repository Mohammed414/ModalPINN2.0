"""Audit the matched collocation-strategy runs for A03."""
from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ARMS = ROOT.parents[0] / "4_runs"
OUT = ROOT / "derived" / "a03_input_manifest.json"

RUNS = {
    "uniform_collocation": ("01_baseline_physics_only", "uniform"),
    "wake_biased_random_collocation": ("06_wake_biased_random", "wake_biased_random"),
    "wake_biased_grid_collocation": ("07_wake_biased_grid", "wake_biased_grid"),
}

EXPECTED = {
    "nmodes": 4,
    "width": 25,
    "ntaps": 32,
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
    for name, (directory, sampling) in RUNS.items():
        path = ARMS / directory
        record_path = path / "run_record.json"
        if not record_path.exists():
            raise FileNotFoundError(record_path)
        record = json.loads(record_path.read_text())
        flags = command_dict(record["command"])
        if "WakeBiasedSampling" in flags:
            observed_sampling = "wake_biased_random"
        elif "WakeBiasedGridSampling" in flags:
            observed_sampling = "wake_biased_grid"
        else:
            observed_sampling = "uniform"
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
            "sampling": observed_sampling,
        }
        expected_settings = {**EXPECTED, "sampling": sampling}
        for key, expected in expected_settings.items():
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
        "analysis_id": "A03",
        "status": "verified_inputs" if not mismatches else "input_mismatch",
        "question": "Does wake-biased collocation recover information absent from wall measurements?",
        "data_contract": str(ROOT / "data_contract.md"),
        "candidates": entries,
        "controlled_comparison": {
            "methods": list(RUNS),
            "intended_change": "interior collocation sampling strategy",
            "interpretation": "All three runs use 32 pressure taps, pressure-only sparse data, the same network/training-size settings, seed 0, FreestreamBC, multigrid settings, and Adam. The intended change is uniform versus wake-biased random versus wake-biased grid sampling of interior physics points.",
        },
        "mismatches": mismatches,
    }
    OUT.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Wrote", OUT)
    print("status:", manifest["status"])
    for name, entry in entries.items():
        print(name, "checkpoint exists =", entry["checkpoint_exists"],
              "sampling =", entry["settings"]["sampling"])
    if mismatches:
        print("mismatches:", json.dumps(mismatches, indent=2))


if __name__ == "__main__":
    main()
