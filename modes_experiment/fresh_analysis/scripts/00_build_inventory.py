#!/usr/bin/env python3
"""Build the machine-readable A00 dataset, sensor, and region inventory."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
FRESH_ROOT = HERE.parent
REPO_ROOT = FRESH_ROOT.parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent

DEFAULT_RAW = REPO_ROOT / "data" / "fixed_cylinder_atRe100"
DEFAULT_CACHE = REPO_ROOT / "data" / "flow_cache.npz"
DEFAULT_DERIVED = FRESH_ROOT / "derived"

XMIN, XMAX = -4.0, 8.0
YMIN, YMAX = -4.0, 4.0
XC, YC, RC = 0.0, 0.0, 0.5
D = 2.0 * RC


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_header_and_first_snapshot(path: Path):
    """Read the raw header and first snapshot without loading the 1.1 GB file."""
    with path.open("r") as stream:
        re, ur = (float(v) for v in stream.readline().split())
        stream.readline()
        nt, nnodes = (int(v) for v in stream.readline().split())
        stream.readline()
        first_time = float(stream.readline())
        block = np.empty((nnodes, 5), dtype=np.float64)
        for row in range(nnodes):
            block[row] = np.fromstring(stream.readline(), sep=" ", count=5)
    return re, ur, nt, nnodes, first_time, block


def region_masks(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    r = np.hypot(x - XC, y - YC)
    near_cylinder = r < 1.5 * RC
    near_wake = (r >= 1.5 * RC) & (x >= XC) & (x < XC + 3.0 * D)
    far_wake = (r >= 1.5 * RC) & (x >= XC + 3.0 * D)
    other = ~(near_cylinder | near_wake | far_wake)
    return {
        "near-cylinder": near_cylinder,
        "near-wake": near_wake,
        "far-wake": far_wake,
        "far-core": far_wake & (np.abs(y - YC) <= 2.0 * D),
        "other": other,
        "whole-domain": np.ones(len(x), dtype=bool),
    }


def pressure_taps(x: np.ndarray, y: np.ndarray, n_taps: int) -> tuple[np.ndarray, np.ndarray]:
    radius = np.hypot(x - XC, y - YC)
    wall_idx = np.where((radius - RC) ** 2 < 1e-5)[0]
    theta = np.linspace(0.0, 2.0 * np.pi, n_taps, endpoint=False)
    chosen = []
    for angle in theta:
        xt = XC + RC * np.cos(angle)
        yt = YC + RC * np.sin(angle)
        local = np.argmin((x[wall_idx] - xt) ** 2 + (y[wall_idx] - yt) ** 2)
        chosen.append(wall_idx[local])
    chosen = np.asarray(chosen, dtype=int)
    if len(np.unique(chosen)) != n_taps:
        raise RuntimeError(f"Only {len(np.unique(chosen))} distinct nodes for {n_taps} taps")
    return x[chosen], y[chosen]


def velocity_probes(x: np.ndarray, y: np.ndarray):
    target_x = np.repeat(np.asarray([-3.0, 1.0, 2.0, 3.0]) * D, 10)
    target_y = np.tile(np.linspace(YMIN, YMAX, 10), 4)
    chosen = np.asarray(
        [np.argmin((x - xt) ** 2 + (y - yt) ** 2) for xt, yt in zip(target_x, target_y)],
        dtype=int,
    )
    if len(np.unique(chosen)) != 40:
        raise RuntimeError("The 40 requested velocity probes do not map to distinct nodes")
    displacement = np.hypot(x[chosen] - target_x, y[chosen] - target_y)
    return target_x, target_y, x[chosen], y[chosen], displacement


def add_row(rows, category, item, value, unit, source, check=""):
    rows.append(
        {
            "category": category,
            "item": item,
            "value": value,
            "unit": unit,
            "source": source,
            "check": check,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DERIVED)
    parser.add_argument("--skip-hash", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = args.raw.resolve()
    cache = args.cache.resolve()
    if not raw.is_file() or not cache.is_file():
        raise FileNotFoundError(f"Expected raw={raw} and cache={cache}")

    raw_re, raw_ur, raw_nt, raw_nodes, raw_t0, first = raw_header_and_first_snapshot(raw)
    with np.load(cache) as data:
        re = float(data["Re"])
        ur = float(data["Ur"])
        times = np.asarray(data["times"], dtype=np.float64)
        x_all = np.asarray(data["X"], dtype=np.float64)
        y_all = np.asarray(data["Y"], dtype=np.float64)
        u = np.asarray(data["U"])
        v = np.asarray(data["V"])
        p = np.asarray(data["p"])

    expected_shape = (len(times), len(x_all))
    if y_all.shape != x_all.shape or u.shape != expected_shape or v.shape != expected_shape or p.shape != expected_shape:
        raise ValueError("Cache arrays do not share the expected fixed-mesh layout")
    if not all(np.isfinite(a).all() for a in (times, x_all, y_all, u, v, p)):
        raise ValueError("Dataset contains non-finite values")
    if (re, ur, len(times), len(x_all), times[0]) != (raw_re, raw_ur, raw_nt, raw_nodes, raw_t0):
        raise ValueError("Cache metadata does not match the raw dataset header")

    first_cache = np.column_stack((x_all, y_all, u[0], v[0], p[0]))
    first_snapshot_diff = np.max(np.abs(first_cache - first), axis=0)
    coordinate_max_abs_diff = float(first_snapshot_diff[:2].max())
    field_max_abs_diff = float(first_snapshot_diff[2:].max())
    # The cache is intentionally float32 while the text reader produces
    # float64. Coordinates as large as x=120 therefore incur a few 1e-6 of
    # round-off, while the physical fields agree more tightly.
    if coordinate_max_abs_diff > 5e-6 or field_max_abs_diff > 1e-6:
        raise ValueError("Cache first snapshot differs from the raw dataset beyond float32 round-off")

    crop = (x_all > XMIN) & (x_all < XMAX) & (y_all > YMIN) & (y_all < YMAX)
    x, y = x_all[crop], y_all[crop]
    regions = region_masks(x, y)

    region_code = np.zeros(len(x), dtype=np.int8)
    region_code[regions["near-cylinder"]] = 1
    region_code[regions["near-wake"]] = 2
    region_code[regions["far-wake"]] = 3

    taps = {n: pressure_taps(x, y, n) for n in (8, 16, 32)}
    probe_tx, probe_ty, probe_x, probe_y, probe_d = velocity_probes(x, y)

    raw_hash = "skipped" if args.skip_hash else sha256(raw)
    cache_hash = "skipped" if args.skip_hash else sha256(cache)
    rows = []
    raw_display = str(raw.relative_to(WORKSPACE_ROOT))
    cache_display = str(cache.relative_to(WORKSPACE_ROOT))
    add_row(rows, "dataset", "raw_path", raw_display, "workspace-relative path", "fixed_cylinder_atRe100", "exists")
    add_row(rows, "dataset", "raw_sha256", raw_hash, "sha256", "fixed_cylinder_atRe100")
    add_row(rows, "dataset", "cache_path", cache_display, "workspace-relative path", "flow_cache.npz", "exists")
    add_row(rows, "dataset", "cache_sha256", cache_hash, "sha256", "flow_cache.npz")
    add_row(rows, "dataset", "cache_raw_first_snapshot_coordinate_max_abs_diff", f"{coordinate_max_abs_diff:.3e}", "D", "raw versus float32 cache", "pass < 5e-6")
    add_row(rows, "dataset", "cache_raw_first_snapshot_field_max_abs_diff", f"{field_max_abs_diff:.3e}", "field units", "raw versus float32 cache", "pass < 1e-6")
    add_row(rows, "physics", "Re", f"{re:g}", "dimensionless", "dataset header", "matches training")
    add_row(rows, "physics", "Ur", f"{ur:g}", "dataset value", "dataset header")
    add_row(rows, "physics", "freestream_velocity", "1", "dimensionless", "training code")
    add_row(rows, "geometry", "cylinder_center", "(0, 0)", "D", "training code")
    add_row(rows, "geometry", "cylinder_radius", f"{RC:g}", "D", "training code")
    add_row(rows, "geometry", "cylinder_diameter", f"{D:g}", "D", "training code")
    add_row(rows, "time", "snapshot_count", len(times), "snapshots", "flow_cache.npz")
    add_row(rows, "time", "time_range", f"[{times[0]:g}, {times[-1]:g}]", "D/U_inf", "flow_cache.npz")
    add_row(rows, "time", "time_step", f"{np.median(np.diff(times)):.12g}", "D/U_inf", "flow_cache.npz", "uniform")
    add_row(rows, "mesh", "full_node_count", len(x_all), "nodes", "flow_cache.npz")
    add_row(rows, "mesh", "analysis_crop", "-4 < x < 8; -4 < y < 4", "D", "evaluation convention")
    add_row(rows, "mesh", "cropped_node_count", len(x), "nodes", "computed crop")
    add_row(rows, "variables", "stored_fields", "u; v; p", "dimensionless", "flow_cache.npz", "all finite")

    definitions = {
        "near-cylinder": "r < 0.75",
        "near-wake": "r >= 0.75 and 0 <= x < 3",
        "far-wake": "r >= 0.75 and x >= 3",
        "far-core": "far-wake and |y| <= 2 (nested subset)",
        "other": "complement of near-cylinder; near-wake; far-wake",
        "whole-domain": "all nodes in the strict analysis crop",
    }
    for name, definition in definitions.items():
        check = "nested in far-wake" if name == "far-core" else ""
        add_row(rows, "region", name, definition, "mask", "evaluation code", f"{regions[name].sum()} nodes; {check}".rstrip("; "))

    for n_taps, (tap_x, tap_y) in taps.items():
        add_row(rows, "sensor", f"pressure_taps_{n_taps}", len(tap_x), "distinct wall nodes", "nearest nodes to uniform angles", "pass")
    add_row(rows, "sensor", "velocity_probes", len(probe_x), "distinct mesh nodes", "4 x-sections times 10 y targets", "pass")
    add_row(rows, "sensor", "velocity_probe_sections", "-3; 1; 2; 3", "x/D", "training reader")
    add_row(rows, "sensor", "velocity_probe_max_target_offset", f"{probe_d.max():.6g}", "D", "nearest-node selection")

    csv_path = args.out_dir / "source_inventory.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("category", "item", "value", "unit", "source", "check"))
        writer.writeheader()
        writer.writerows(rows)

    geometry_path = args.out_dir / "a00_geometry.npz"
    np.savez_compressed(
        geometry_path,
        x=x.astype(np.float32),
        y=y.astype(np.float32),
        region_code=region_code,
        far_core=regions["far-core"],
        tap8_x=taps[8][0].astype(np.float32), tap8_y=taps[8][1].astype(np.float32),
        tap16_x=taps[16][0].astype(np.float32), tap16_y=taps[16][1].astype(np.float32),
        tap32_x=taps[32][0].astype(np.float32), tap32_y=taps[32][1].astype(np.float32),
        probe_target_x=probe_tx.astype(np.float32), probe_target_y=probe_ty.astype(np.float32),
        probe_x=probe_x.astype(np.float32), probe_y=probe_y.astype(np.float32),
        region_names=np.asarray(["other", "near-cylinder", "near-wake", "far-wake"]),
        region_counts=np.asarray([regions[n].sum() for n in ("other", "near-cylinder", "near-wake", "far-wake")]),
        far_core_count=np.asarray(regions["far-core"].sum()),
        whole_domain_count=np.asarray(regions["whole-domain"].sum()),
    )

    print(f"Dataset: Re={re:g}, {len(times)} snapshots, {len(x_all)} full nodes")
    print(f"Analysis crop: {len(x)} nodes")
    for name in ("near-cylinder", "near-wake", "far-wake", "far-core", "other", "whole-domain"):
        print(f"  {name:16s} {regions[name].sum():6d}")
    print(f"Cache/raw first-snapshot coordinate max abs difference: {coordinate_max_abs_diff:.3e}")
    print(f"Cache/raw first-snapshot field max abs difference: {field_max_abs_diff:.3e}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {geometry_path}")


if __name__ == "__main__":
    main()
