#!/usr/bin/env python3
"""Run the final GappyPOD diagnostic matched to the ModalPINN experiment.

The POD basis, pressure solve, and evaluation all use the complete 201-snapshot
record. This is intentional: the calculation tests whether the pressure taps
can identify the state when the spatial subspace is supplied, under the same
in-sample reconstruction setting as ModalPINN. It does not test temporal
generalisation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def resolve_data_file() -> Path:
    """Locate the CFD cache without copying the large dataset into this repo."""

    configured = os.environ.get("GAPPYPOD_FLOW_CACHE")
    if configured:
        return Path(configured).expanduser().resolve()
    candidates = (
        HERE.parents[2] / "GappyPOD" / "data" / "flow_cache.npz",
        HERE.parents[1] / "data" / "flow_cache.npz",
        HERE.parent / "data" / "flow_cache.npz",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DATA_FILE = resolve_data_file()

RANK = 6
N_TAPS = 32
RADIUS = 0.5
OMEGA_0 = 1.036
NOISE_SEED = 0
NOISE_SIGMAS = (0.0, 4.7265e-4, 2.3633e-3, 4.7265e-3)
EPS = 1.0e-30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def uniform_tap_indices(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Reproduce ModalPINN's cylinder-tap selection exactly."""

    radius = np.hypot(x, y)
    candidates = np.flatnonzero((radius - RADIUS) ** 2 < 1.0e-5)
    targets = np.linspace(0.0, 2.0 * np.pi, N_TAPS, endpoint=False)
    chosen = []
    for theta in targets:
        xt = RADIUS * np.cos(theta)
        yt = RADIUS * np.sin(theta)
        local = np.argmin((x[candidates] - xt) ** 2 + (y[candidates] - yt) ** 2)
        chosen.append(int(candidates[local]))
    chosen = np.asarray(chosen, dtype=int)
    if candidates.size != 628:
        raise ValueError(f"Expected 628 wall candidates, found {candidates.size}")
    if np.unique(chosen).size != N_TAPS:
        raise ValueError("Uniform target angles did not produce 32 distinct taps")
    return chosen


def region_masks(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    radius = np.hypot(x, y)
    near_cylinder = radius < 0.75
    near_wake = (~near_cylinder) & (x >= 0.0) & (x < 3.0)
    far_wake = (~near_cylinder) & (x >= 3.0)
    other = ~(near_cylinder | near_wake | far_wake)
    far_core = far_wake & (np.abs(y) <= 2.0)
    masks = {
        "near-cylinder": near_cylinder,
        "near-wake": near_wake,
        "far-wake": far_wake,
        "far-core": far_core,
        "other": other,
        "whole-domain": np.ones(x.size, dtype=bool),
    }
    if sum(int(masks[name].sum()) for name in ("near-cylinder", "near-wake", "far-wake", "other")) != x.size:
        raise AssertionError("ModalPINN region partition does not cover the crop")
    if np.any(far_core & ~far_wake):
        raise AssertionError("Far core must be nested inside far wake")
    return masks


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(prediction - reference)
        / (np.linalg.norm(reference) + EPS)
    )


def temporal_modes(field: np.ndarray, times: np.ndarray, kmax: int = 3) -> np.ndarray:
    """Use the exact least-squares Fourier convention of the ModalPINN audit."""

    tau = times - times[0]
    columns = [np.ones_like(tau)]
    for k in range(1, kmax + 1):
        columns.extend((np.cos(k * OMEGA_0 * tau), np.sin(k * OMEGA_0 * tau)))
    design = np.stack(columns, axis=1)
    coefficients, _, _, _ = np.linalg.lstsq(design, field, rcond=None)
    modes = np.empty((kmax + 1, field.shape[1]), dtype=np.complex128)
    modes[0] = coefficients[0]
    for k in range(1, kmax + 1):
        modes[k] = 0.5 * (
            coefficients[2 * k - 1] - 1j * coefficients[2 * k]
        )
    return modes


def complex_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    norm_p = float(np.linalg.norm(prediction))
    norm_q = float(np.linalg.norm(reference))
    inner = np.vdot(reference, prediction)
    return {
        "relative_L2": float(np.linalg.norm(prediction - reference) / (norm_q + EPS)),
        "amplitude_ratio": float(norm_p / (norm_q + EPS)),
        "correlation": float(abs(inner) / (norm_p * norm_q + EPS)),
        "phase_deg": float(np.degrees(np.angle(inner))),
    }


def evaluate(
    reconstruction: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
    times: np.ndarray,
    masks: dict[str, np.ndarray],
    sigma: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for region, mask in masks.items():
        for variable in ("u", "v", "p"):
            rows.append(
                {
                    "noise_sigma": sigma,
                    "quantity": "field",
                    "variable": variable,
                    "region": region,
                    "metric": "relative_L2",
                    "value": relative_l2(
                        reconstruction[variable][:, mask], truth[variable][:, mask]
                    ),
                }
            )

    reference_v1 = temporal_modes(truth["v"], times)[1]
    reconstructed_v1 = temporal_modes(reconstruction["v"], times)[1]
    for region, mask in masks.items():
        values = complex_metrics(reconstructed_v1[mask], reference_v1[mask])
        for metric, value in values.items():
            rows.append(
                {
                    "noise_sigma": sigma,
                    "quantity": "first_harmonic_v1",
                    "variable": "v",
                    "region": region,
                    "metric": metric,
                    "value": value,
                }
            )
    return rows


def main() -> None:
    with np.load(DATA_FILE) as data:
        times = np.asarray(data["times"], dtype=np.float64)
        x_full = np.asarray(data["X"], dtype=np.float64)
        y_full = np.asarray(data["Y"], dtype=np.float64)
        fields_full = {
            "u": np.asarray(data["U"], dtype=np.float64),
            "v": np.asarray(data["V"], dtype=np.float64),
            "p": np.asarray(data["p"], dtype=np.float64),
        }

    if times.size != 201:
        raise ValueError(f"Expected 201 snapshots, found {times.size}")

    crop = (x_full > -4.0) & (x_full < 8.0) & (y_full > -4.0) & (y_full < 4.0)
    crop_indices = np.flatnonzero(crop)
    x = x_full[crop]
    y = y_full[crop]
    truth = {name: values[:, crop] for name, values in fields_full.items()}
    n_snapshots = times.size
    n_nodes = x.size
    if n_nodes != 51654:
        raise ValueError(f"Expected 51,654 cropped nodes, found {n_nodes}")

    taps_full = uniform_tap_indices(x_full, y_full)
    full_to_crop = np.full(x_full.size, -1, dtype=int)
    full_to_crop[crop_indices] = np.arange(n_nodes)
    taps = full_to_crop[taps_full]
    if np.any(taps < 0):
        raise ValueError("A ModalPINN pressure tap lies outside the evaluation crop")

    state = np.vstack([truth[name].T for name in ("u", "v", "p")])
    mean = state.mean(axis=1, keepdims=True)
    fluctuations = state - mean

    pod_basis, singular_values, _ = np.linalg.svd(fluctuations, full_matrices=False)
    phi = pod_basis[:, :RANK]
    exact_coefficients = phi.T @ fluctuations
    projection = mean + phi @ exact_coefficients

    pressure_offset = 2 * n_nodes
    pressure_rows = pressure_offset + taps
    sampled_basis = phi[pressure_rows]
    matrix_rank = int(np.linalg.matrix_rank(sampled_basis))
    condition_number = float(np.linalg.cond(sampled_basis))
    if matrix_rank != RANK:
        raise ValueError(f"Sampled rank is {matrix_rank}, expected {RANK}")

    centred_pressure = state[pressure_rows] - mean[pressure_rows]
    base_noise_flat = np.random.RandomState(NOISE_SEED).normal(
        size=n_snapshots * N_TAPS
    )
    base_noise = base_noise_flat.reshape(n_snapshots, N_TAPS).T

    metric_rows: list[dict[str, object]] = []
    masks = region_masks(x, y)
    for sigma in NOISE_SIGMAS:
        measurements = centred_pressure + sigma * base_noise
        coefficients, _, _, _ = np.linalg.lstsq(
            sampled_basis, measurements, rcond=None
        )
        reconstructed_state = mean + phi @ coefficients
        reconstruction = {
            "u": reconstructed_state[0:n_nodes].T,
            "v": reconstructed_state[n_nodes : 2 * n_nodes].T,
            "p": reconstructed_state[2 * n_nodes : 3 * n_nodes].T,
        }
        metric_rows.extend(evaluate(reconstruction, truth, times, masks, sigma))

    projection_fields = {
        "u": projection[0:n_nodes].T,
        "v": projection[n_nodes : 2 * n_nodes].T,
        "p": projection[2 * n_nodes : 3 * n_nodes].T,
    }
    projection_rows = evaluate(projection_fields, truth, times, masks, -1.0)
    for row in projection_rows:
        row["quantity"] = "pod_projection_" + str(row["quantity"])
    metric_rows.extend(projection_rows)

    RESULTS.mkdir(parents=True, exist_ok=True)
    metrics_file = RESULTS / "metrics.csv"
    with metrics_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["noise_sigma", "quantity", "variable", "region", "metric", "value"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    energy = singular_values**2
    configuration = {
        "purpose": "in-sample coefficient-identifiability diagnostic matched to ModalPINN",
        "data_file": "data/flow_cache.npz",
        "data_sha256": sha256(DATA_FILE),
        "state": "joint unscaled [u; v; p] on the ModalPINN evaluation crop",
        "snapshot_count": n_snapshots,
        "basis_snapshot_range": [0, n_snapshots - 1],
        "reconstruction_snapshot_range": [0, n_snapshots - 1],
        "crop_node_count": n_nodes,
        "rank": RANK,
        "rank_energy_fraction": float(energy[:RANK].sum() / energy.sum()),
        "tap_count": N_TAPS,
        "tap_full_mesh_indices": taps_full.tolist(),
        "tap_crop_indices": taps.tolist(),
        "tap_selection": "exact ModalPINN uniform target-angle nearest-node algorithm",
        "sampled_basis_rank": matrix_rank,
        "sampled_basis_condition_number": condition_number,
        "noise": {
            "model": "additive zero-mean Gaussian pressure noise",
            "sigmas": list(NOISE_SIGMAS),
            "seed": NOISE_SEED,
            "realizations": 1,
            "shared_standard_normal_draw_across_levels": True,
        },
        "region_node_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "metric": "space-time pooled relative L2 and ModalPINN first-harmonic diagnostics",
    }
    (RESULTS / "configuration.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )

    def lookup(sigma: float, quantity: str, variable: str, region: str, metric: str) -> float:
        return float(
            next(
                row["value"]
                for row in metric_rows
                if row["noise_sigma"] == sigma
                and row["quantity"] == quantity
                and row["variable"] == variable
                and row["region"] == region
                and row["metric"] == metric
            )
        )

    summary = {
        "clean": {
            "whole_domain": {
                variable: lookup(0.0, "field", variable, "whole-domain", "relative_L2")
                for variable in ("u", "v", "p")
            },
            "far_core_v": lookup(0.0, "field", "v", "far-core", "relative_L2"),
            "far_core_v1": {
                metric: lookup(0.0, "first_harmonic_v1", "v", "far-core", metric)
                for metric in ("relative_L2", "amplitude_ratio", "correlation", "phase_deg")
            },
        },
        "noise_far_core_v": {
            f"{sigma:.7g}": lookup(sigma, "field", "v", "far-core", "relative_L2")
            for sigma in NOISE_SIGMAS
        },
        "interpretation": (
            "Diagnostic reconstruction on the same snapshots used to build the POD basis; "
            "supports coefficient identifiability within the supplied subspace, not temporal generalisation."
        ),
    }
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print("Final GappyPOD diagnostic")
    print(f"  state matrix: {state.shape}")
    print(f"  rank {RANK} energy: {100 * configuration['rank_energy_fraction']:.6f}%")
    print(f"  taps: {N_TAPS}, sampled rank: {matrix_rank}, condition: {condition_number:.3g}")
    print("  clean whole-domain relative L2:")
    for variable, value in summary["clean"]["whole_domain"].items():
        print(f"    {variable}: {100 * value:.4f}%")
    print(f"  clean far-core v: {100 * summary['clean']['far_core_v']:.4f}%")
    print(f"  clean far-core v1: {summary['clean']['far_core_v1']}")
    print(f"  wrote {metrics_file}")
    print(f"  wrote {RESULTS / 'configuration.json'}")
    print(f"  wrote {RESULTS / 'summary.json'}")


if __name__ == "__main__":
    main()
