# -*- coding: utf-8 -*-
"""
Phase 0 of the Lighthill BVF plan (see bvf.md): validate the wall identity

    (1/Re) * d(omega)/dn = (1/R) * dp/dtheta      at r = R (cylinder wall)

against the trained E1 dense-baseline network, BEFORE writing any new
training code. LHS comes from autodiff through the E1 network; RHS comes
from bvf_targets.py's fit of the 32 pressure taps (Phase 1).

Expects, in the working directory:
    NN_functions.py               (this run's copy, from src/pressure_only/)
    DNN2_75_75_3_tanh.pickle       (E1 dense-baseline trained weights)
    bvf_targets_Ntap32_seed0.npz   (Phase 1 output)

Runs on CPU in a couple of seconds - no GPU, no full TF1.14 conda env
needed, since restore_NN/NN_time_uv only use generic tf.compat.v1 ops.
"""

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# NN_functions.py was written against TF1's top-level API. Colab's default
# runtime is TF2, which dropped a few of those aliases in favor of tf.math.*
# even though tf.compat.v1 graph-mode execution otherwise still works fine.
# Patched here (not in NN_functions.py, which stays identical to what
# training uses) since this is purely a local runtime-compatibility need for
# this standalone inference/autodiff check, not a change to training logic.
if not hasattr(tf, 'real'):
    tf.real = tf.math.real

import NN_functions as nnf

RE = 100.
R_C = 0.5
OMEGA_0 = 1.036
GEOM = (-4., 8., -4., 4., 0., 0., R_C)
LAYERS = [2, 75, 75, 3]
MODEL_PATH = 'DNN2_75_75_3_tanh.pickle'
TARGETS_PATH = 'bvf_targets_Ntap32_seed0.npz'


def main():
    d = np.load(TARGETS_PATH)
    theta_grid = d['theta_grid']
    t_grid = d['t_grid']
    x_wall = d['x_wall']
    y_wall = d['y_wall']
    G = d['G']  # RHS target, [Ntheta, Ntime]

    Ntheta = len(theta_grid)
    Ntime = len(t_grid)

    X = np.tile(x_wall.reshape(-1, 1), (1, Ntime)).astype(np.float32).reshape(-1, 1)
    Y = np.tile(y_wall.reshape(-1, 1), (1, Ntime)).astype(np.float32).reshape(-1, 1)
    Tt = np.tile(t_grid.reshape(1, -1), (Ntheta, 1)).astype(np.float32).reshape(-1, 1)

    w_u, b_u, w_v, b_v, w_p, b_p = nnf.restore_NN(LAYERS, MODEL_PATH, tf_as_constant=True)

    x_pl = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
    y_pl = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
    t_pl = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

    u = nnf.NN_time_uv(x_pl, y_pl, t_pl, w_u, b_u, GEOM, OMEGA_0)
    v = nnf.NN_time_uv(x_pl, y_pl, t_pl, w_v, b_v, GEOM, OMEGA_0)

    omega = tf.gradients(v, x_pl)[0] - tf.gradients(u, y_pl)[0]
    omega_x = tf.gradients(omega, x_pl)[0]
    omega_y = tf.gradients(omega, y_pl)[0]
    # d(omega)/dn = cos(theta)*omega_x + sin(theta)*omega_y; at r=R, cos=x/R, sin=y/R
    dwdn = (x_pl * omega_x + y_pl * omega_y) / R_C
    LHS = (1.0 / RE) * dwdn

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        lhs_val = sess.run(LHS, feed_dict={x_pl: X, y_pl: Y, t_pl: Tt})

    LHS_grid = lhs_val.reshape(Ntheta, Ntime)

    resid = LHS_grid - G
    rel_l2 = np.linalg.norm(resid) / np.linalg.norm(G)
    corr = np.sum(LHS_grid * G) / (np.linalg.norm(LHS_grid) * np.linalg.norm(G))
    lhs_rms = np.sqrt(np.mean(LHS_grid ** 2))
    rhs_rms = np.sqrt(np.mean(G ** 2))

    print('=== BVF identity check (E1 network vs 32-tap-derived target) ===')
    print('Relative L2 mismatch (LHS vs RHS): %.4f' % rel_l2)
    print('Correlation (sign/phase check, want close to +1): %.4f' % corr)
    print('LHS RMS: %.4e   RHS RMS: %.4e   ratio: %.4f' % (lhs_rms, rhs_rms, lhs_rms / rhs_rms))
    print('Acceptance target from bvf.md: relative L2 <= ~0.20 and correlation > 0')

    np.savez('bvf_validation_result.npz', LHS=LHS_grid, RHS=G,
             theta_grid=theta_grid, t_grid=t_grid, rel_l2=rel_l2, corr=corr)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(G.flatten(), LHS_grid.flatten(), s=8, alpha=0.5)
    lims = [min(G.min(), LHS_grid.min()), max(G.max(), LHS_grid.max())]
    axes[0].plot(lims, lims, 'k--', linewidth=1, label='y=x')
    axes[0].set_xlabel('RHS: (1/R) dp/dtheta (from taps)')
    axes[0].set_ylabel('LHS: (1/Re) domega/dn (E1 network)')
    axes[0].set_title('Identity check (rel L2 = %.3f, corr = %.3f)' % (rel_l2, corr))
    axes[0].legend()

    it0 = 0
    axes[1].plot(theta_grid, G[:, it0], 'o-', label='RHS (taps)')
    axes[1].plot(theta_grid, LHS_grid[:, it0], 'x-', label='LHS (E1 net)')
    axes[1].set_xlabel('theta')
    axes[1].set_title('Snapshot at t=%.2f' % t_grid[it0])
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('bvf_validation_plot.png')
    print('Saved plot to bvf_validation_plot.png')


if __name__ == '__main__':
    main()
