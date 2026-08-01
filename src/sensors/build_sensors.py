"""
WP4 - Pressure-sensor dataset generator (project plan Section 4, Work Package 4).

Identifies the real cylinder-surface mesh nodes in the Boudina et al. dataset,
picks uniformly-spaced subsets of them for a range of tap counts, and exports
the tap indices/positions/pressure time histories plus a visualisation of
where each set sits on the cylinder.

Usage: python build_sensors.py
Expects the real dataset at ../../Data/fixed_cylinder_atRe100 (relative to
this file), i.e. <repo_root>/Data/fixed_cylinder_atRe100.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)
from text_flow import read_flow  # noqa: E402

REPO_ROOT = os.path.dirname(SRC_DIR)
DATA_FILE = os.path.join(REPO_ROOT, 'Data', 'fixed_cylinder_atRe100')
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'sensor_indices')

# Same geometry as ModalPINN_VortexShedding.py
X_C, Y_C, R_C = 0., 0., 0.5
TAP_COUNTS = [4, 8, 16, 32]
SURFACE_EPS = 1e-3  # max distance from r_c (in the same units as x,y) to count as "on the cylinder"


def find_cylinder_surface_nodes(nodes_x, nodes_y):
    """Return indices of mesh nodes lying on the cylinder surface (r ~ r_c)."""
    r = np.sqrt((nodes_x - X_C) ** 2 + (nodes_y - Y_C) ** 2)
    on_surface = np.where(np.abs(r - R_C) < SURFACE_EPS)[0]
    return on_surface


def pick_uniform_taps(surface_idx, nodes_x, nodes_y, n_taps):
    """Pick n_taps surface nodes closest to n_taps uniformly-spaced angles."""
    target_angles = np.linspace(0., 2 * np.pi, n_taps, endpoint=False)
    target_x = X_C + R_C * np.cos(target_angles)
    target_y = Y_C + R_C * np.sin(target_angles)

    sx, sy = nodes_x[surface_idx], nodes_y[surface_idx]
    chosen = np.zeros(n_taps, dtype=int)
    for k in range(n_taps):
        d2 = (sx - target_x[k]) ** 2 + (sy - target_y[k]) ** 2
        chosen[k] = surface_idx[np.argmin(d2)]
    return chosen


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print('Reading real dataset (this is the ~1.17GB file, takes ~1 min)...')
    Re, Ur, times, nodes_X, nodes_Y, Us, Vs, Ps = read_flow(DATA_FILE)

    nodes_x0, nodes_y0 = nodes_X[0, :], nodes_Y[0, :]
    surface_idx = find_cylinder_surface_nodes(nodes_x0, nodes_y0)
    print(f'Found {len(surface_idx)} mesh nodes on the cylinder surface '
          f'(|r - {R_C}| < {SURFACE_EPS})')
    assert len(surface_idx) >= max(TAP_COUNTS), (
        f'Only {len(surface_idx)} surface nodes found, need at least {max(TAP_COUNTS)}. '
        f'SURFACE_EPS may need adjusting for this mesh.'
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    theta = np.linspace(0, 2 * np.pi, 200)

    for ax, n_taps in zip(axes.flat, TAP_COUNTS):
        chosen = pick_uniform_taps(surface_idx, nodes_x0, nodes_y0, n_taps)
        cx, cy = nodes_x0[chosen], nodes_y0[chosen]
        p_series = Ps[:, chosen]  # [Nt, n_taps]

        out_path = os.path.join(OUT_DIR, f'taps_{n_taps:02d}.npz')
        np.savez(out_path,
                 node_indices=chosen,
                 x=cx, y=cy,
                 times=times,
                 pressure=p_series)

        stds = p_series.std(axis=0)
        print(f'  {n_taps:2d} taps -> {out_path}')
        print(f'         pressure std per tap: min={stds.min():.4e}, max={stds.max():.4e} '
              f'({"OK, all taps show variation" if stds.min() > 0 else "WARNING: a flat tap found"})')

        ax.plot(X_C + R_C * np.cos(theta), Y_C + R_C * np.sin(theta), 'k-', lw=1)
        ax.scatter(nodes_x0[surface_idx], nodes_y0[surface_idx], s=2, c='lightgray', label='all surface nodes')
        ax.scatter(cx, cy, s=40, c='red', zorder=3, label=f'{n_taps} taps')
        ax.set_title(f'{n_taps} pressure taps')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=8)

    fig.suptitle('Uniformly-spaced pressure-tap placements on the cylinder')
    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'sensor_map.png')
    fig.savefig(fig_path, dpi=150)
    print(f'Saved sensor map figure to {fig_path}')


if __name__ == '__main__':
    main()
