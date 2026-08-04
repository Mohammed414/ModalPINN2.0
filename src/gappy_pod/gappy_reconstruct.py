# -*- coding: utf-8 -*-
"""
Gappy POD baseline, step 2: given only the 32 wall-pressure taps (the same
ones the PINN uses), reconstruct the full (u,v,p) field on the held-out test
snapshots via the classical Everson & Sirovich (1995) gappy-POD pseudo-inverse
solve. Sweeps the number of retained POD modes r, since ModalPINN's Nmodes=3
is a fixed choice this should actually check, not assume.

Usage:
    python gappy_reconstruct.py --PodBasis pod_basis.npz \
        --Taps ../../data/sensor_indices/taps_32.npz
"""
import argparse

import numpy as np

R_SWEEP = [2, 3, 4, 6, 8, 10]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--PodBasis', default='pod_basis.npz')
    parser.add_argument('--Taps', default='../../data/sensor_indices/taps_32.npz')
    parser.add_argument('--Out', default='gappy_reconstructions.npz')
    args = parser.parse_args()

    basis = np.load(args.PodBasis)
    U_r_full = basis['U_r']          # [3*Nnodes, R_MAX]
    X_mean = basis['X_mean']         # [3*Nnodes]
    Nnodes = int(basis['Nnodes'])
    times = basis['times']           # [Nt]
    test_idx = basis['test_timestep_indices']  # indices into `times`

    taps = np.load(args.Taps)
    tap_node_indices = taps['node_indices']  # [Ntaps], indices into the raw read_flow node ordering
    tap_times = taps['times']                # [Nt] (same 201 timesteps)
    tap_pressure = taps['pressure']           # [Nt, Ntaps]
    Ntaps = len(tap_node_indices)
    print('Loaded %d taps' % Ntaps)

    # Sanity: the POD basis's `times` and the taps file's `times` must be the
    # exact same timeline, or index-matching test_idx into tap_pressure would
    # silently pick the wrong rows.
    assert times.shape == tap_times.shape and np.allclose(times, tap_times), \
        'POD basis and taps file were built from different timelines - cannot index-match.'

    # Rows of the stacked [u;v;p] vector corresponding to pressure at the tap
    # nodes: pressure occupies rows [2*Nnodes : 3*Nnodes], offset by the tap
    # node indices (Step 2 of the plan).
    tap_rows = 2 * Nnodes + tap_node_indices
    mean_taps = X_mean[tap_rows]  # [Ntaps]

    results = {}
    for r in R_SWEEP:
        U_r = U_r_full[:, :r]
        U_r_taps = U_r[tap_rows, :]  # [Ntaps, r]
        cond = np.linalg.cond(U_r_taps)
        print('r=%2d: U_r_taps shape %s, condition number %.3e%s' %
              (r, U_r_taps.shape, cond, ' (WARNING: ill-conditioned, r approaching Ntaps)' if cond > 1e4 else ''))

        u_pred = np.zeros((len(test_idx), Nnodes))
        v_pred = np.zeros((len(test_idx), Nnodes))
        p_pred = np.zeros((len(test_idx), Nnodes))

        for i, ti in enumerate(test_idx):
            p_taps_obs = tap_pressure[ti, :]  # [Ntaps], the actual measured tap values at this held-out time
            a, *_ = np.linalg.lstsq(U_r_taps, p_taps_obs - mean_taps, rcond=None)
            x_full = X_mean + U_r @ a  # [3*Nnodes]
            u_pred[i, :] = x_full[:Nnodes]
            v_pred[i, :] = x_full[Nnodes:2 * Nnodes]
            p_pred[i, :] = x_full[2 * Nnodes:3 * Nnodes]

        results['u_pred_r%d' % r] = u_pred
        results['v_pred_r%d' % r] = v_pred
        results['p_pred_r%d' % r] = p_pred

    np.savez(args.Out,
             nodes_X=basis['nodes_X'], nodes_Y=basis['nodes_Y'],
             test_timestep_indices=test_idx, times=times,
             r_sweep=np.array(R_SWEEP), **results)
    print('Saved gappy reconstructions to', args.Out)


if __name__ == '__main__':
    main()
