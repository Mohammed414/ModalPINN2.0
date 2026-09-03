"""Evaluate the analytical Kármán prior under the frozen A04 contract.

This is the prior-only baseline: no TensorFlow checkpoint and no training are
used. The output is intentionally compatible with the later Arm 1/Arm 15
evaluator, so prior attribution compares like with like.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
FRESH_ROOT = HERE.parent
REPO_ROOT = FRESH_ROOT.parents[1]
SRC = REPO_ROOT / "src" / "R9" / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluate_common import (  # noqa: E402
    OMEGA_0,
    regional_complex_metrics,
    regional_field_metrics,
    strict_crop_indices,
    temporal_harmonic_coefficients,
)
from street_prior import cf_modes_uv  # noqa: E402


def load_data(path: pathlib.Path):
    z = np.load(path)
    return (float(z["Re"]), float(z["Ur"]), np.asarray(z["times"]),
            np.asarray(z["X"]), np.asarray(z["Y"]), np.asarray(z["U"]),
            np.asarray(z["V"]), np.asarray(z["p"]))


def prior_field(x: np.ndarray, y: np.ndarray, t: float, prm: Dict[str, float],
                t0: float):
    """NumPy port of the modal prior used by ``evaluate_prior_only.py``."""
    us, vs = cf_modes_uv(x, y, prm, nk=3)
    amp = 2.0 * prm["amp_scale"]
    r2 = x * x + y * y + 1e-12
    a2 = 0.5 ** 2
    u0 = 1.0 - a2 * (x * x - y * y) / r2 ** 2
    v0 = -a2 * 2.0 * x * y / r2 ** 2
    up, vp = u0.copy(), v0.copy()
    pp = -0.5 * ((u0 - prm["Uc"]) ** 2 + v0 ** 2)
    for k, (uk, vk) in enumerate(zip(us, vs), start=1):
        eu = amp * uk
        ev = amp * vk
        ep = -(1.0 - prm["Uc"]) * prm["scale_p"] * eu
        phase_t = np.exp(1j * k * OMEGA_0 * (t - t0))
        up += np.real(eu * phase_t)
        vp += np.real(ev * phase_t)
        pp += np.real(ep * phase_t)
    return up, vp, pp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--Prior", required=True)
    ap.add_argument("--Data", default=str(REPO_ROOT / "data" / "flow_cache.npz"))
    ap.add_argument("--Geometry", default=str(FRESH_ROOT / "derived" /
                                                 "a00_geometry.npz"))
    ap.add_argument("--Out", default=str(FRESH_ROOT / "derived" /
                                           "a04_prior_only_metrics.json"))
    ap.add_argument("--Chunk", type=int, default=4000)
    args = ap.parse_args()

    prior_path = pathlib.Path(args.Prior)
    data_path = pathlib.Path(args.Data)
    geom_path = pathlib.Path(args.Geometry)
    z = np.load(prior_path)
    names = ("Gamma", "Uc", "xf", "r0", "omega", "phase", "amp_scale",
             "scale_p", "ramp", "delta")
    prm = {k: float(z[k]) for k in names}
    _, _, times, X, Y, U, V, P = load_data(data_path)
    X, Y, times = X.reshape(-1), Y.reshape(-1), times.reshape(-1)
    idx = strict_crop_indices(X, Y)
    geom = np.load(geom_path, allow_pickle=True)
    x, y = X[idx], Y[idx]
    if not (np.allclose(x, geom["x"], atol=5e-6) and
            np.allclose(y, geom["y"], atol=5e-6)):
        raise AssertionError("data crop does not match audited A00 geometry")
    regions = {
        "near-cylinder": geom["region_code"] == 1,
        "near-wake": geom["region_code"] == 2,
        "far-wake": geom["region_code"] == 3,
        "far-core": np.asarray(geom["far_core"], dtype=bool),
        "other": geom["region_code"] == 0,
        "whole-domain": np.ones(x.size, dtype=bool),
    }

    pred_u = np.empty_like(U[:, idx], dtype=float)
    pred_v = np.empty_like(V[:, idx], dtype=float)
    pred_p = np.empty_like(P[:, idx], dtype=float)
    chunk = max(1, int(args.Chunk))
    for it, t in enumerate(times):
        for start in range(0, x.size, chunk):
            sl = slice(start, min(start + chunk, x.size))
            pred_u[it, sl], pred_v[it, sl], pred_p[it, sl] = prior_field(
                x[sl], y[sl], float(t), prm, float(times[0])
            )
    predictions = {"u": pred_u, "v": pred_v, "p": pred_p}
    references = {"u": U[:, idx], "v": V[:, idx], "p": P[:, idx]}
    field_metrics = regional_field_metrics(predictions, references, regions)
    pred_modes = temporal_harmonic_coefficients(pred_v, times, OMEGA_0, 3)
    true_modes = temporal_harmonic_coefficients(references["v"], times,
                                                 OMEGA_0, 3)
    v1_metrics = regional_complex_metrics(pred_modes[1], true_modes[1], regions)

    result = {
        "analysis_id": "A04",
        "method": "prior_only",
        "status": "verified",
        "prior": str(prior_path),
        "data": str(data_path),
        "geometry": str(geom_path),
        "snapshots": int(times.size),
        "crop_nodes": int(x.size),
        "prior_parameters": prm,
        "field_metrics": field_metrics,
        "v1_mode_metrics": v1_metrics,
    }
    out = pathlib.Path(args.Out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    print("Wrote", out)


if __name__ == "__main__":
    main()
