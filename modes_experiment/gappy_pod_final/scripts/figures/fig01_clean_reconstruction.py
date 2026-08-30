#!/usr/bin/env python3
"""Representative clean GappyPOD reconstruction of vertical velocity."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
FINAL_ROOT = HERE.parents[2]
sys.path.insert(0, str(FINAL_ROOT))

from figure_common import draw_cylinder, new_figure, save_figure, style_domain
from run_analysis import DATA_FILE, N_TAPS, RANK, uniform_tap_indices

import matplotlib.tri as mtri


OUTPUT = FINAL_ROOT / "figures" / "final" / "G01_clean_reconstruction.png"
PLOT_DATA = FINAL_ROOT / "results" / "representative_snapshot.npz"
SNAPSHOT_TIME = 410.0


def reconstruct_snapshot() -> dict[str, np.ndarray | float]:
    with np.load(DATA_FILE) as data:
        times = np.asarray(data["times"], dtype=np.float64)
        x_full = np.asarray(data["X"], dtype=np.float64)
        y_full = np.asarray(data["Y"], dtype=np.float64)
        fields_full = {
            "u": np.asarray(data["U"], dtype=np.float64),
            "v": np.asarray(data["V"], dtype=np.float64),
            "p": np.asarray(data["p"], dtype=np.float64),
        }

    crop = (x_full > -4.0) & (x_full < 8.0) & (y_full > -4.0) & (y_full < 4.0)
    crop_indices = np.flatnonzero(crop)
    x = x_full[crop]
    y = y_full[crop]
    truth = {name: values[:, crop] for name, values in fields_full.items()}
    n_nodes = x.size

    taps_full = uniform_tap_indices(x_full, y_full)
    full_to_crop = np.full(x_full.size, -1, dtype=int)
    full_to_crop[crop_indices] = np.arange(n_nodes)
    taps = full_to_crop[taps_full]

    state = np.vstack([truth[name].T for name in ("u", "v", "p")])
    mean = state.mean(axis=1, keepdims=True)
    fluctuations = state - mean
    pod_basis, _, _ = np.linalg.svd(fluctuations, full_matrices=False)
    phi = pod_basis[:, :RANK]

    pressure_rows = 2 * n_nodes + taps
    measurements = state[pressure_rows] - mean[pressure_rows]
    coefficients, _, _, _ = np.linalg.lstsq(
        phi[pressure_rows], measurements, rcond=None
    )

    snapshot_index = int(np.argmin(np.abs(times - SNAPSHOT_TIME)))
    reconstructed = (mean[:, 0] + phi @ coefficients[:, snapshot_index])
    truth_v = truth["v"][snapshot_index]
    reconstructed_v = reconstructed[n_nodes : 2 * n_nodes]
    error_v = np.abs(reconstructed_v - truth_v)

    result = {
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "time": float(times[snapshot_index]),
        "truth_v": truth_v.astype(np.float32),
        "reconstructed_v": reconstructed_v.astype(np.float32),
        "absolute_error_v": error_v.astype(np.float32),
    }
    np.savez_compressed(PLOT_DATA, **result)
    return result


def fluid_triangulation(x: np.ndarray, y: np.ndarray) -> mtri.Triangulation:
    triangulation = mtri.Triangulation(x, y)
    triangles = triangulation.triangles
    centroid_x = x[triangles].mean(axis=1)
    centroid_y = y[triangles].mean(axis=1)
    triangulation.set_mask(np.hypot(centroid_x, centroid_y) < 0.52)
    return triangulation


def main() -> None:
    if DATA_FILE.exists():
        data = reconstruct_snapshot()
        plot_data_message = f"wrote {PLOT_DATA}"
    elif PLOT_DATA.exists():
        with np.load(PLOT_DATA) as stored:
            data = {name: np.asarray(stored[name]) for name in stored.files}
        data["time"] = float(data["time"])
        print(f"source CFD cache not found; rendering frozen data from {PLOT_DATA}")
        plot_data_message = f"used {PLOT_DATA}"
    else:
        raise FileNotFoundError(
            "Set GAPPYPOD_FLOW_CACHE to flow_cache.npz or restore "
            f"the frozen plot data at {PLOT_DATA}"
        )
    x = np.asarray(data["x"])
    y = np.asarray(data["y"])
    truth_v = np.asarray(data["truth_v"])
    reconstructed_v = np.asarray(data["reconstructed_v"])
    error_v = np.asarray(data["absolute_error_v"])
    triangulation = fluid_triangulation(x, y)

    velocity_limit = float(
        max(np.max(np.abs(truth_v)), np.max(np.abs(reconstructed_v)))
    )
    error_limit = float(np.max(error_v))
    velocity_levels = np.linspace(-velocity_limit, velocity_limit, 41)
    error_levels = np.linspace(0.0, error_limit, 31)

    fig = new_figure(height=3.15)
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=(1.0, 0.075),
        left=0.065,
        right=0.985,
        bottom=0.15,
        top=0.79,
        wspace=0.10,
        hspace=0.30,
    )
    axes = [fig.add_subplot(grid[0, column]) for column in range(3)]

    velocity_maps = []
    for column, (ax, field, title) in enumerate(
        zip(
            axes[:2],
            (truth_v, reconstructed_v),
            ("CFD reference", "Gappy POD reconstruction"),
        )
    ):
        image = ax.tricontourf(
            triangulation,
            field,
            levels=velocity_levels,
            cmap="RdBu_r",
            extend="both",
        )
        velocity_maps.append(image)
        draw_cylinder(ax)
        style_domain(ax, show_ylabel=column == 0)
        ax.set_title(title, pad=5.0, fontweight="semibold")

    error_map = axes[2].tricontourf(
        triangulation,
        error_v,
        levels=error_levels,
        cmap="magma",
        extend="max",
    )
    draw_cylinder(axes[2])
    style_domain(axes[2], show_ylabel=False)
    axes[2].set_title(r"Absolute error, $|\hat{v}-v|$", pad=5.0, fontweight="semibold")

    velocity_cax = fig.add_subplot(grid[1, :2])
    velocity_bar = fig.colorbar(velocity_maps[0], cax=velocity_cax, orientation="horizontal")
    velocity_bar.set_label(r"Vertical velocity, $v/U_\infty$")
    velocity_bar.set_ticks(np.linspace(-velocity_limit, velocity_limit, 7))
    velocity_bar.ax.tick_params(length=2.5)

    error_cax = fig.add_subplot(grid[1, 2])
    error_bar = fig.colorbar(error_map, cax=error_cax, orientation="horizontal")
    error_bar.set_label(r"Absolute error in $v$, $|\hat{v}-v|/U_\infty$")
    error_bar.set_ticks(np.linspace(0.0, error_limit, 4))
    error_bar.ax.tick_params(length=2.5)

    fig.suptitle(
        "The supplied POD basis recovers the clean wake from 32 pressure taps",
        y=0.965,
        fontsize=11.0,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.875,
        f"Representative snapshot at t = {data['time']:.1f}  |  rank {RANK}  |  {N_TAPS} taps",
        ha="center",
        va="center",
        color="#4B5563",
        fontsize=8.4,
    )

    output = save_figure(fig, OUTPUT)
    print(f"wrote {output}")
    print(plot_data_message)
    print(f"velocity range: +/-{velocity_limit:.6g}")
    print(f"maximum absolute error: {error_limit:.6g}")


if __name__ == "__main__":
    main()
