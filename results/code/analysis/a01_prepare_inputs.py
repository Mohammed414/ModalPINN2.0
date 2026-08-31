"""Audit the three A01 measurement-information runs before evaluation."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ARMS = ROOT.parents[1] / "modes_experiment" / "runs" / "arms"
OUT = ROOT / "derived" / "a01_input_manifest.json"

RUNS = {
    "pressure_only_physics": ARMS / "01_baseline_physics_only",
    "pressure_and_velocity_probes_physics": ARMS / "04_paper_sparse_probes",
    "dense_observations": ARMS / "05_dense_reference",
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
    for name, path in RUNS.items():
        record_path = path / "run_record.json"
        if not record_path.exists():
            raise FileNotFoundError(record_path)
        record = json.loads(record_path.read_text())
        command = command_dict(record["command"])
        checkpoint = path / "training_run" / str(record["weights"])
        nn_functions = path / "training_run" / "NN_functions.py"
        entries[name] = {
            "run_directory": str(path),
            "run_label": record["arm"],
            "checkpoint": str(checkpoint),
            "checkpoint_local_nn_functions": str(nn_functions),
            "settings": {
                "nmodes": record["nmodes"],
                "width": record["width"],
                "ntaps": command.get("NTaps", 0),
                "sparse_data": bool("SparseData" in command),
                "pressure_only": bool("PressureOnly" in command),
                "freestream_bc": bool("FreestreamBC" in command),
                "seed": command.get("Seed"),
                "nmes": command.get("Nmes"),
                "nint": command.get("Nint"),
                "multigrid": bool("multigrid" in command),
                "ngrid": command.get("Ngrid"),
                "ngrid_turn": command.get("NgridTurn"),
                "adam_ran": bool(record.get("adam_ran", False)),
                "lbfgs_evals": record.get("lbfgs_evals"),
            },
        }

    manifest = {
        "analysis_id": "A01",
        "status": "verified_inputs",
        "question": "How does available measurement information affect reconstruction?",
        "data_contract": str(ROOT / "data_contract.md"),
        "candidates": entries,
        "controlled_comparison": {
            "methods": ["pressure_only_physics", "pressure_and_velocity_probes_physics"],
            "interpretation": "Controlled information comparison: both use 32 pressure taps, sparse data, seed 0, FreestreamBC, and the same network/training-size settings; the added velocity probes are the intended information change.",
        },
        "dense_ceiling": {
            "method": "dense_observations",
            "interpretation": "Representational ceiling, not a single-parameter controlled comparison: it uses dense observations and also skips Adam with a longer L-BFGS budget.",
        },
    }
    OUT.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Wrote", OUT)
    for name, entry in entries.items():
        print(name, "checkpoint exists =", Path(entry["checkpoint"]).exists())


if __name__ == "__main__":
    main()
