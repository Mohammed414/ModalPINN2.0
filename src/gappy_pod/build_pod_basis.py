# -*- coding: utf-8 -*-
"""
Gappy POD baseline, step 1: build a joint (u,v,p) POD basis from a subset of
the real dataset's snapshots ("basis" timesteps - a contiguous early block of
the timeline, held-in for training the basis; a contiguous held-out block at
the end is used for testing gappy reconstruction, see gappy_reconstruct.py).

Plain numpy/scipy/matplotlib. No TensorFlow, no GPU - runs on the laptop.

Usage:
    python build_pod_basis.py --DataFile ../../Data/fixed_cylinder_atRe100
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from text_flow import read_flow  # noqa: E402

DEFAULT_DATA_FILE = 'Data/fixed_cylinder_atRe100'
R_MAX = 15  # save more modes than we expect to need (Everson & Sirovich checked via the energy plot)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--DataFile', default=DEFAULT_DATA_FILE)
    parser.add_argument('--RMax', type=int, default=R_MAX)
    parser.add_argument('--NTestBlock', type=int, default=50, help='Size of the held-out contiguous test block at the end of the timeline.')
    parser.add_argument('--Out', default='pod_basis.npz')
    args = parser.parse_args()

    print('Reading real dataset...')
    Re, Ur, times, nodes_X, nodes_Y, Us, Vs, Ps = read_flow(args.DataFile)
    Nt, Nnodes = Us.shape
    print('Us/Vs/Ps shape:', Us.shape, '| times shape:', times.shape)

    # Mesh must be static across time for the tap node_indices (picked once,
    # at t=times[0], by build_sensors.py) to mean the same physical location
    # at every other timestep - assert this rather than assume it.
    dx = np.max(np.abs(nodes_X - nodes_X[0:1, :]))
    dy = np.max(np.abs(nodes_Y - nodes_Y[0:1, :]))
    print('Max mesh drift across time: dx=%.3e dy=%.3e' % (dx, dy))
    assert dx < 1e-9 and dy < 1e-9, 'Mesh is not static across time - node_indices from build_sensors.py would not be valid at other timesteps.'
    nodes_x0 = nodes_X[0, :]
    nodes_y0 = nodes_Y[0, :]

    # Block split: basis built from a contiguous early chunk of the timeline,
    # tested on a held-out contiguous chunk that shares no immediate neighbor
    # with any basis snapshot. An earlier version used an even/odd interleave
    # (basis = even indices, test = odd), which let every test snapshot sit
    # just Delta_t=0.1 - ~1.6% of the ~6.06-time-unit shedding period - from
    # two bracketing basis snapshots. Since POD mode coefficients evolve
    # smoothly in time, that made "reconstruction" close to trivial temporal
    # interpolation rather than a real test of generalizing across the
    # shedding phase, and produced implausibly good numbers. The PINN gets no
    # equivalent held-out split at all (it trains on all 201 steps plus
    # physics), so POD is still being tested under a harder protocol than the
    # PINN - noted here and again in the eval step, not hidden.
    basis_idx = np.arange(0, Nt - args.NTestBlock)
    test_idx = np.arange(Nt - args.NTestBlock, Nt)
    print('Basis snapshots: %d (indices 0..%d) | held-out test block: %d (indices %d..%d)' %
          (len(basis_idx), basis_idx[-1], len(test_idx), test_idx[0], test_idx[-1]))

    # Snapshot matrix: stack [u; v; p] per timestep into one column of length
    # 3*Nnodes, assemble into X of shape [3*Nnodes, N_basis_snapshots].
    X = np.concatenate([Us[basis_idx, :].T, Vs[basis_idx, :].T, Ps[basis_idx, :].T], axis=0)
    print('Snapshot matrix X shape:', X.shape)

    X_mean = X.mean(axis=1, keepdims=True)  # [3*Nnodes, 1] - the POD "mode 0" analog
    Xc = X - X_mean

    print('Running SVD (this is the expensive step, a few seconds to a minute)...')
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    print('U shape:', U.shape, '| S shape:', S.shape)

    U_r = U[:, :args.RMax]
    energy = S ** 2
    cum_energy = np.cumsum(energy) / np.sum(energy)
    print('Cumulative energy captured by first %d modes:' % args.RMax)
    for r in range(1, min(args.RMax, 10) + 1):
        print('  r=%2d: %.4f%%' % (r, 100 * cum_energy[r - 1]))

    np.savez(args.Out,
             U_r=U_r, S=S, X_mean=X_mean[:, 0],
             nodes_X=nodes_x0, nodes_Y=nodes_y0,
             Nnodes=Nnodes, basis_timestep_indices=basis_idx, test_timestep_indices=test_idx,
             times=times)
    print('Saved POD basis to', args.Out)

    # Singular value spectrum + cumulative energy plot - a standalone useful
    # dissertation figure independent of everything else ("how many modes
    # does this flow actually need").
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    n_show = min(30, len(S))
    axes[0].semilogy(np.arange(1, n_show + 1), S[:n_show], 'o-')
    axes[0].set_xlabel('mode index')
    axes[0].set_ylabel('singular value (log scale)')
    axes[0].set_title('POD singular value spectrum')

    axes[1].plot(np.arange(1, n_show + 1), 100 * cum_energy[:n_show], 'o-')
    axes[1].axhline(99, color='gray', linestyle='--', linewidth=1, label='99%')
    axes[1].set_xlabel('number of modes retained')
    axes[1].set_ylabel('cumulative energy captured (%)')
    axes[1].set_title('Cumulative energy vs mode count')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('pod_spectrum.png', dpi=150)
    print('Saved spectrum plot to pod_spectrum.png')


if __name__ == '__main__':
    main()
