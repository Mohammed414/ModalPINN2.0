# -*- coding: utf-8 -*-
"""
Gappy POD baseline, step 3: regional relative-L2 error on the held-out test
snapshots, for each swept r. Deliberately mirrors
src/pressure_only/evaluate_regions.py's region definitions and metric exactly
(same near-cylinder/near-wake/far-wake/upstream-off-axis bins, same relative
L2 formula, same domain-box crop) so the PINN-vs-POD comparison is apples to
apples.

Usage:
    python evaluate_regions.py --Reconstructions gappy_reconstructions.npz \
        --DataFile ../../Data/fixed_cylinder_atRe100
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from text_flow import read_flow  # noqa: E402

# Geometry constants, matching src/pressure_only/evaluate_regions.py exactly.
X_C, Y_C, R_C = 0., 0., 0.5
LXMIN, LXMAX, LYMIN, LYMAX = -4., 8., -4., 4.
D = 2 * R_C

DEFAULT_DATA_FILE = 'Data/fixed_cylinder_atRe100'


def rel_l2(pred, true, mask):
    """Identical formula to src/pressure_only/evaluate_regions.py's rel_l2."""
    if mask.sum() == 0:
        return float('nan')
    diff = pred[:, mask] - true[:, mask]
    return np.linalg.norm(diff) / np.linalg.norm(true[:, mask])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Reconstructions', default='gappy_reconstructions.npz')
    parser.add_argument('--DataFile', default=DEFAULT_DATA_FILE)
    parser.add_argument('--Out', default='pod_regional_evaluation.txt')
    parser.add_argument('--SummaryCsv', default='pod_regional_evaluation_summary.csv')
    args = parser.parse_args()

    recon = np.load(args.Reconstructions)
    test_idx = recon['test_timestep_indices']
    r_sweep = recon['r_sweep']
    nodes_x0 = recon['nodes_X']
    nodes_y0 = recon['nodes_Y']

    print('Reading real dataset for ground truth...')
    Re, Ur, times, nodes_X, nodes_Y, Us, Vs, Ps = read_flow(args.DataFile)
    assert np.allclose(nodes_X[0, :], nodes_x0) and np.allclose(nodes_Y[0, :], nodes_y0), \
        'Node ordering mismatch between the POD reconstruction and a fresh read_flow() call.'

    # Same domain-box crop as src/pressure_only/evaluate_regions.py, applied
    # consistently so the region definitions below match exactly.
    in_box = ((nodes_x0 < LXMAX) & (nodes_x0 > LXMIN) &
              (nodes_y0 > LYMIN) & (nodes_y0 < LYMAX))
    box_idx = np.argwhere(in_box)[:, 0]
    nodes_xc, nodes_yc = nodes_x0[box_idx], nodes_y0[box_idx]

    r = np.sqrt((nodes_xc - X_C) ** 2 + (nodes_yc - Y_C) ** 2)
    region_near_cyl = r < 1.5 * R_C
    region_near_wake = (~region_near_cyl) & (nodes_xc >= X_C) & (nodes_xc < X_C + 3 * D)
    region_far_wake = (~region_near_cyl) & (~region_near_wake) & (nodes_xc >= X_C + 3 * D)
    region_other = ~(region_near_cyl | region_near_wake | region_far_wake)
    regions = {
        'near-cylinder': region_near_cyl,
        'near-wake': region_near_wake,
        'far-wake': region_far_wake,
        'other (upstream/off-axis)': region_other,
        'whole domain': np.ones_like(region_near_cyl, dtype=bool),
    }

    # Ground truth at the held-out test timesteps, cropped to the same box.
    Us_true = Us[test_idx, :][:, box_idx]
    Vs_true = Vs[test_idx, :][:, box_idx]
    Ps_true = Ps[test_idx, :][:, box_idx]

    lines = []
    lines.append('Gappy POD regional evaluation - held-out test snapshots only')
    lines.append('(%d held-out test snapshots forming a contiguous block at the end of the '
                  'record, unseen by the POD basis - a held-out split the PINN itself does not '
                  'get, see build_pod_basis.py)' % len(test_idx))
    lines.append('')

    summary_rows = []
    for r_val in r_sweep:
        u_pred = recon['u_pred_r%d' % r_val][:, box_idx]
        v_pred = recon['v_pred_r%d' % r_val][:, box_idx]
        p_pred = recon['p_pred_r%d' % r_val][:, box_idx]

        lines.append('r = %d POD modes' % r_val)
        header = "%-26s%9s%10s%10s%10s" % ('Region', 'n_nodes', 'E_u', 'E_v', 'E_p')
        lines.append(header)
        lines.append('-' * len(header))
        row_by_region = {}
        for name, mask in regions.items():
            eu = rel_l2(u_pred, Us_true, mask)
            ev = rel_l2(v_pred, Vs_true, mask)
            ep = rel_l2(p_pred, Ps_true, mask)
            lines.append("%-26s%9d%10.4f%10.4f%10.4f" % (name, mask.sum(), eu, ev, ep))
            row_by_region[name] = (eu, ev, ep)
        lines.append('')
        summary_rows.append((r_val,) + row_by_region['whole domain'] +
                             row_by_region['near-cylinder'] + row_by_region['near-wake'] +
                             row_by_region['far-wake'])

    output_text = '\n'.join(lines)
    print(output_text)
    with open(args.Out, 'w') as f:
        f.write(output_text)
    print('\nSaved to', args.Out)

    with open(args.SummaryCsv, 'w') as f:
        f.write('r,whole_Eu,whole_Ev,whole_Ep,near_cyl_Eu,near_cyl_Ev,near_cyl_Ep,'
                'near_wake_Eu,near_wake_Ev,near_wake_Ep,far_wake_Eu,far_wake_Ev,far_wake_Ep\n')
        for row in summary_rows:
            f.write(','.join(str(x) for x in row) + '\n')
    print('Saved summary CSV to', args.SummaryCsv)


if __name__ == '__main__':
    main()
