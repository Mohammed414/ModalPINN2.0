"""Shared input auditing for the fresh-analysis arms.

Three things every controlled comparison in this workspace needs, implemented
once rather than per analysis:

1.  A *symmetric* flag diff.  ``a02_prepare_inputs.py`` checked a fixed
    whitelist of settings, which cannot notice a flag present in one run and
    absent in another.  :func:`flag_diff` compares the full parsed command
    lines, so an unintended difference is reported instead of passing silently.
2.  Checkpoint provenance.  ``verified_inputs`` should pin the actual weights,
    so :func:`run_facts` records the sha256 of every pickle and of the
    checkpoint-local ``NN_functions.py``.
3.  Optimizer effort as a first-class field.  15 of the 16 arms were stopped by
    SciPy's ftol test, which is below float32 resolution for this loss and
    therefore fires on rounding noise at an unpredictable iteration; the
    exception is ``05_dense_reference``, which reached its 40000-evaluation
    ``maxfun`` cap.  Effort consequently varies several-fold between otherwise
    identical runs (1162 to 43676 L-BFGS evaluations), so it must be read off
    the manifest before any comparison is interpreted.  :func:`effort_audit`
    reports the stop reason actually observed for the arms in hand rather than
    assuming one.

Also holds the phase-convention helper described in :func:`rotate_metrics`.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

CHECKPOINT_NAME = "DNN2_100_100_4_tanh.pickle"


# ---------------------------------------------------------------------------
# command lines
# ---------------------------------------------------------------------------
def parse_flags(command: Sequence[str]) -> Dict[str, object]:
    """Parse a recorded argv into ``{flag_name: value_or_True}``."""
    flags: Dict[str, object] = {}
    i = 0
    command = [str(token) for token in command]
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


def flag_diff(named_flags: Mapping[str, Mapping[str, object]]) -> List[Dict[str, object]]:
    """Return every flag whose value is not identical across all runs.

    Symmetric: a flag missing from one run appears with ``None`` for that run,
    so presence differences are caught as well as value differences.
    """
    keys = sorted({key for flags in named_flags.values() for key in flags})
    differing = []
    for key in keys:
        values = {name: flags.get(key) for name, flags in named_flags.items()}
        if len({json.dumps(value, sort_keys=True) for value in values.values()}) > 1:
            differing.append({"flag": key, "values": values})
    return differing


# ---------------------------------------------------------------------------
# per-run facts
# ---------------------------------------------------------------------------
def sha256(path: pathlib.Path, digits: int = 64) -> Optional[str]:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:digits]


def _lbfgs_exit(train_log: pathlib.Path) -> Dict[str, object]:
    """Read the L-BFGS termination facts out of a training log."""
    if not train_log.exists():
        return {"stop_reason": None, "iterations": None, "projg_at_exit": None,
                "objective_at_exit": None}
    text = train_log.read_text(errors="replace")
    if "EXCEEDS LIMIT" in text:
        reason = "maxfun_cap"
    elif "CONVERGENCE: REL_REDUCTION" in text:
        # NOT convergence in any physical sense: the test is
        # f_k - f_k+1 <= factr*eps = 1e-12 at ftol 1e-12, while the float32
        # loss cannot resolve steps below ~3e-11 near a loss of 3e-4.
        reason = "ftol_rel_reduction"
    elif "ABNORMAL" in text:
        reason = "line_search_failure"
    else:
        reason = "unknown"
    row = re.search(r"^\s*\d+\s+(\d+)\s+(\d+)\s+\S+\s+\S+\s+\S+\s+(\S+)\s+(\S+)\s*$",
                    text, re.M)
    return {
        "stop_reason": reason,
        "iterations": int(row.group(1)) if row else None,
        "projg_at_exit": row.group(3) if row else None,
        "objective_at_exit": row.group(4) if row else None,
    }


def run_facts(arm_dir: pathlib.Path) -> Dict[str, object]:
    """Collect settings, provenance, and optimizer effort for one arm."""
    record_path = arm_dir / "run_record.json"
    if not record_path.exists():
        raise FileNotFoundError(record_path)
    record = json.loads(record_path.read_text())
    training = arm_dir / "training_run"
    checkpoint = training / str(record.get("weights", CHECKPOINT_NAME))
    nn_functions = training / "NN_functions.py"
    loss_summary_path = training / "training_loss_summary.json"
    losses = json.loads(loss_summary_path.read_text()) if loss_summary_path.exists() else {}
    facts = {
        "run_directory": str(arm_dir),
        "run_label": record.get("arm"),
        "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.exists(),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_local_nn_functions": str(nn_functions),
        "nn_functions_exists": nn_functions.exists(),
        "nn_functions_sha256": sha256(nn_functions),
        "street_prior": str(arm_dir / "street_prior_used.npz")
                        if (arm_dir / "street_prior_used.npz").exists() else None,
        "flags": parse_flags(record.get("command", [])),
        "effort": {
            "wall_s": record.get("wall_s"),
            "lbfgs_evals": record.get("lbfgs_evals"),
            "adam_ran": bool(record.get("adam_ran", False)),
            "final_total_loss": losses.get("total_loss"),
            "final_physics_loss": losses.get("physics_loss"),
            "final_measurement_loss": losses.get("pressure_tap_loss"),
            **_lbfgs_exit(arm_dir / "train_log.txt"),
        },
    }
    return facts


def effort_audit(entries: Mapping[str, Mapping[str, object]]) -> Dict[str, object]:
    """Summarise how unequal the optimizer effort is across the arms."""
    evals = {name: entry["effort"]["lbfgs_evals"] for name, entry in entries.items()}
    usable = {name: value for name, value in evals.items() if value}
    spread = (max(usable.values()) / min(usable.values())) if usable else None
    stop_reasons = {name: entry["effort"]["stop_reason"] for name, entry in entries.items()}
    ftol_stopped = [name for name, reason in stop_reasons.items()
                    if reason == "ftol_rel_reduction"]
    if len(ftol_stopped) == len(stop_reasons):
        mechanism = ("All arms here were stopped by SciPy's ftol test, which is below "
                     "float32 resolution for this loss and so fires on rounding noise "
                     "at an unpredictable iteration. ")
    elif ftol_stopped:
        mechanism = ("Arms stopped by SciPy's ftol test (below float32 resolution for "
                     "this loss, so it fires on rounding noise at an unpredictable "
                     "iteration): %s. Other arms stopped otherwise: %s. "
                     % (", ".join(sorted(ftol_stopped)),
                        ", ".join("%s=%s" % (name, reason)
                                  for name, reason in sorted(stop_reasons.items())
                                  if reason != "ftol_rel_reduction")))
    else:
        mechanism = ("No arm here stopped on the ftol test; observed stop reasons: %s. "
                     % ", ".join("%s=%s" % item for item in sorted(stop_reasons.items())))
    return {
        "lbfgs_evals": evals,
        "spread_ratio": round(spread, 2) if spread else None,
        "matched_within_20pct": bool(spread is not None and spread <= 1.2),
        "stop_reasons": stop_reasons,
        "note": (mechanism +
                 "Effort is therefore NOT a controlled "
                 "variable under --Tmax and must be reported alongside every metric. "
                 "Evaluation count is not a reconstruction-quality proxy, so the "
                 "gap does not provide a directional accuracy bound."),
    }


def build_manifest(*, analysis_id: str, question: str, runs: Mapping[str, pathlib.Path],
                   intended_change_flags: Iterable[str], intended_change: str,
                   data_contract: pathlib.Path, out_path: pathlib.Path) -> Dict[str, object]:
    """Audit a set of arms and write the input manifest."""
    entries = {name: run_facts(path) for name, path in runs.items()}
    differing = flag_diff({name: entry["flags"] for name, entry in entries.items()})
    intended = set(intended_change_flags)
    unintended = [item for item in differing if item["flag"] not in intended]
    missing_files = [name for name, entry in entries.items()
                     if not (entry["checkpoint_exists"] and entry["nn_functions_exists"])]
    identical_forward_code = len({entry["nn_functions_sha256"]
                                  for entry in entries.values()}) == 1
    manifest = {
        "analysis_id": analysis_id,
        "status": "verified_inputs" if not unintended and not missing_files
                  else "input_mismatch",
        "question": question,
        "data_contract": str(data_contract),
        "candidates": entries,
        "controlled_comparison": {
            "methods": list(runs),
            "intended_change": intended_change,
            "intended_change_flags": sorted(intended),
            "flag_differences": differing,
            "unintended_flag_differences": unintended,
            "identical_forward_code": identical_forward_code,
        },
        "effort_audit": effort_audit(entries),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


# ---------------------------------------------------------------------------
# first-harmonic phase convention
# ---------------------------------------------------------------------------
def phase_rotation_deg(t0: float, omega: float) -> float:
    """Degrees by which the two v1 phase conventions differ.

    The corrected ``temporal_harmonic_coefficients`` returns absolute-time
    coefficients. This helper quantifies the legacy error: older tables kept
    the truth in ``tau = t - t0`` while the network mode was referenced to
    absolute ``t``. The two differ by ``omega*t0``; under the legacy
    comparison a perfect model reports ``phase_deg = +16.63`` and
    ``rel_L2 = 0.289`` rather than zero.
    """
    return math.degrees((omega * t0) % (2.0 * math.pi))


def rotate_metrics(metrics: Mapping[str, float], delta_deg: float) -> Dict[str, float]:
    """Re-express complex metrics against a truth rotated by ``exp(+i delta)``.

    With ``q`` the reference, ``p`` the prediction, ``a = |p|/|q|``,
    ``c = |<q,p>|/(|p||q|)`` and ``phi = arg(<q,p>)``:

        <q e^{i d}, p> = e^{-i d} <q, p>            ->  phi' = phi - d
        |p - q e^{i d}|^2 / |q|^2 = 1 + a^2 - 2 a c cos(phi')

    ``n``, ``amp_ratio`` and ``corr`` are invariant under the rotation; only
    ``rel_L2`` and ``phase_deg`` move.  Exact, so no re-inference is needed to
    report both conventions.
    """
    a = float(metrics["amp_ratio"])
    c = float(metrics["corr"])
    phi = float(metrics["phase_deg"]) - float(delta_deg)
    phi = (phi + 180.0) % 360.0 - 180.0
    value = 1.0 + a * a - 2.0 * a * c * math.cos(math.radians(phi))
    return {
        "n": int(metrics["n"]),
        "rel_L2": math.sqrt(max(value, 0.0)),
        "amp_ratio": a,
        "corr": c,
        "phase_deg": phi,
    }


def assert_metric_identity(metrics: Mapping[str, float], tol: float = 1e-6) -> None:
    """Check the reported rel_L2 against (amp_ratio, corr, phase_deg).

    Guards the rotation above: if this identity holds on freshly computed
    metrics, the closed form used by :func:`rotate_metrics` is the same
    quantity ``evaluate_common.complex_metrics`` computed.
    """
    rebuilt = rotate_metrics(metrics, 0.0)["rel_L2"]
    if not math.isclose(rebuilt, float(metrics["rel_L2"]), rel_tol=tol, abs_tol=tol):
        raise AssertionError(
            "rel_L2 identity failed: reported %.12g, rebuilt %.12g"
            % (float(metrics["rel_L2"]), rebuilt))
