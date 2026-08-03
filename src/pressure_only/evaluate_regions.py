"""
Standalone regional-error evaluation for a trained (pressure-only or dense)
ModalPINN run.

Loads a saved model checkpoint (no retraining) and the real CFD dataset,
reconstructs u, v, p over the real mesh nodes, and reports relative L2
error split by region (near-cylinder / near-wake / far-wake / whole domain)
so we can see whether reconstruction quality degrades away from the
sensors, per the project plan's regional-error metric (Section 8.1).

Usage:
    python evaluate_regions.py --RunDir <path to the run's output folder> \
        --WidthLayer 25 --Nmodes 3
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')  # must be set before NN_functions imports pyplot
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import NN_functions as nnf  # noqa: E402
from text_flow import read_flow  # noqa: E402

# Geometry / physics constants, matching ModalPINN_VortexShedding.py
X_C, Y_C, R_C = 0., 0., 0.5
LXMIN, LXMAX, LYMIN, LYMAX = -4., 8., -4., 4.
GEOM = [LXMIN, LXMAX, LYMIN, LYMAX, X_C, Y_C, R_C]
OMEGA_0 = 1.036
D = 2 * R_C  # cylinder diameter

# Relative to the current working directory, matching the plain relative path
# ModalPINN_VortexShedding.py itself uses (filename_data = 'Data/fixed_cylinder_atRe100').
# Don't compute this from __file__ / repo structure - in the Colab notebook this
# script and Data/ both sit flat in /content, not nested under src/pressure_only/.
DEFAULT_DATA_FILE = 'Data/fixed_cylinder_atRe100'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--RunDir', required=True, help="Path to the run's output folder (contains DNN..._tanh.pickle)")
    parser.add_argument('--WidthLayer', type=int, required=True)
    parser.add_argument('--Nmodes', type=int, required=True)
    parser.add_argument('--DataFile', default=DEFAULT_DATA_FILE)
    parser.add_argument('--FreestreamBC', action='store_true', default=None,
                         help="Force freestream_target on restore. Default: auto-detect from 'FSBC' in --RunDir.")
    parser.add_argument('--FluctuationInletBC', action='store_true', default=None,
                         help="Force damp_fluctuations on restore. Default: auto-detect from 'FIBC' in --RunDir.")
    args = parser.parse_args()

    run_dir_name = os.path.basename(os.path.normpath(args.RunDir))
    use_freestream = True if args.FreestreamBC else ('FSBC' in run_dir_name)
    use_fluct_damp = True if args.FluctuationInletBC else ('FIBC' in run_dir_name)
    freestream_target_u = 1.0 if use_freestream else None
    freestream_target_v = 0.0 if use_freestream else None
    print('Restoring with freestream_target=%s, damp_fluctuations=%s (run dir: %s)' %
          (use_freestream, use_fluct_damp, run_dir_name))

    layers = [2, args.WidthLayer * args.Nmodes, args.WidthLayer * args.Nmodes, args.Nmodes]

    pickle_candidates = glob.glob(os.path.join(args.RunDir, 'DNN*_tanh.pickle'))
    assert pickle_candidates, f'No model pickle found in {args.RunDir}'
    model_file = pickle_candidates[0]
    print('Loading model:', model_file)

    w_u, b_u, w_v, b_v, w_p, b_p = nnf.restore_NN(layers, model_file, tf_as_constant=True)

    print('Reading real dataset...')
    Re, Ur, times, nodes_X, nodes_Y, Us, Vs, Ps = read_flow(args.DataFile)

    # Crop to the training domain box, same condition as read_cut_simulation_data
    nodes_x0, nodes_y0 = nodes_X[0, :], nodes_Y[0, :]
    in_box = ((nodes_x0 < LXMAX) & (nodes_x0 > LXMIN) &
              (nodes_y0 > LYMIN) & (nodes_y0 < LYMAX))
    idx = np.argwhere(in_box)[:, 0]
    nodes_x0, nodes_y0 = nodes_x0[idx], nodes_y0[idx]
    Us_c, Vs_c, Ps_c = Us[:, idx], Vs[:, idx], Ps[:, idx]

    r = np.sqrt((nodes_x0 - X_C) ** 2 + (nodes_y0 - Y_C) ** 2)
    region_near_cyl = r < 1.5 * R_C
    region_near_wake = (~region_near_cyl) & (nodes_x0 >= X_C) & (nodes_x0 < X_C + 3 * D)
    region_far_wake = (~region_near_cyl) & (~region_near_wake) & (nodes_x0 >= X_C + 3 * D)
    region_other = ~(region_near_cyl | region_near_wake | region_far_wake)

    regions = {
        'near-cylinder': region_near_cyl,
        'near-wake': region_near_wake,
        'far-wake': region_far_wake,
        'other (upstream/off-axis)': region_other,
        'whole domain': np.ones_like(region_near_cyl, dtype=bool),
    }

    x_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
    y_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
    t_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

    u_pred_tf = nnf.NN_time_uv(x_tf, y_tf, t_tf, w_u, b_u, GEOM, OMEGA_0,
                                freestream_target=freestream_target_u, damp_fluctuations=use_fluct_damp)
    v_pred_tf = nnf.NN_time_uv(x_tf, y_tf, t_tf, w_v, b_v, GEOM, OMEGA_0,
                                freestream_target=freestream_target_v, damp_fluctuations=use_fluct_damp)
    p_pred_tf = nnf.NN_time_p(x_tf, y_tf, t_tf, w_p, b_p, OMEGA_0)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    Nt, Nnode = Us_c.shape
    print(f'Reconstructing {Nt} timesteps x {Nnode} nodes (one timestep at a time, to keep memory bounded)...')
    x_col = nodes_x0.reshape(-1, 1).astype(np.float32)
    y_col = nodes_y0.reshape(-1, 1).astype(np.float32)

    u_pred = np.zeros((Nt, Nnode), dtype=np.float32)
    v_pred = np.zeros((Nt, Nnode), dtype=np.float32)
    p_pred = np.zeros((Nt, Nnode), dtype=np.float32)
    for k in range(Nt):
        t_col = np.full((Nnode, 1), times[k], dtype=np.float32)
        feed = {x_tf: x_col, y_tf: y_col, t_tf: t_col}
        u_pred[k, :] = sess.run(u_pred_tf, feed_dict=feed)[:, 0]
        v_pred[k, :] = sess.run(v_pred_tf, feed_dict=feed)[:, 0]
        p_pred[k, :] = sess.run(p_pred_tf, feed_dict=feed)[:, 0]
        if (k + 1) % 50 == 0 or k == Nt - 1:
            print(f'  {k + 1}/{Nt} timesteps done')

    def rel_l2(pred, true, mask):
        if mask.sum() == 0:
            return float('nan')
        diff = pred[:, mask] - true[:, mask]
        return np.linalg.norm(diff) / np.linalg.norm(true[:, mask])

    print()
    header = f"{'Region':<26}{'n_nodes':>9}{'E_u':>10}{'E_v':>10}{'E_p':>10}"
    print(header)
    print('-' * len(header))
    for name, mask in regions.items():
        eu = rel_l2(u_pred, Us_c, mask)
        ev = rel_l2(v_pred, Vs_c, mask)
        ep = rel_l2(p_pred, Ps_c, mask)
        print(f"{name:<26}{mask.sum():>9}{eu:>10.4f}{ev:>10.4f}{ep:>10.4f}")


if __name__ == '__main__':
    main()
