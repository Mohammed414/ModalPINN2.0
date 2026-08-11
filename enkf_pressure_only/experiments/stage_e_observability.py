"""
STAGE E: empirical ensemble-space observability/detectability diagnostic.

NOT a formal proof of nonlinear observability -- an empirical check of how
strongly the CURRENT ensemble's state directions are visible in the 32
wall-pressure taps, and where (spatially) each tap's correction actually
reaches. Uses a fresh ensemble (same construction as Stage D) run forward
with NO assimilation at all, over a short window, collecting the pressure
anomaly matrix Y_k at each instant.

Y_stack = [R^{-1/2} Y_1; ...; R^{-1/2} Y_L]  (stacked rows, one block per instant)
SVD: Y_stack = U Sigma V^T -> singular values show how many independent
ensemble directions are strongly vs. weakly visible in wall pressure.

Also: representer fields P_xy = X Y^T for two representative taps (front
stagnation-ish and a wake-facing shoulder), reshaped onto the (u,v) grid,
showing where in the domain each tap's pressure anomaly actually
correlates with velocity anomalies (i.e. where an analysis correction
driven by that tap alone would act).
"""
import json
import os
import numpy as np

import estimator
from estimator.ns_solver import CylinderFlowSolver
from estimator.data_interface import TapObservations
from estimator.state_vector import StateVectorizer
import experiments.stage_d_enkf as sd

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, '..', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

Q_ENSEMBLE = 16
L_WINDOWS = 30  # ~3 time units at dt_assim=0.1, roughly half a shedding period


def main():
    spin = np.load(os.path.join(HERE, 'spinup_snapshots.npz'))
    c = json.loads(str(spin['solver_config']))
    obs = TapObservations(n_taps=32)
    sigma_p = 0.3 * np.std(obs.tap_p)  # same calibration as Stage D

    rng = np.random.default_rng(0)
    members, ic_times = sd.build_ensemble(spin, 310.0, Q_ENSEMBLE, 0.4, rng, c)
    vec = StateVectorizer(members[0])

    dt_assim = obs.tap_times[1] - obs.tap_times[0]
    substeps = int(round(dt_assim / c['dt']))

    Y_blocks = []
    Xf_first = Yf_first = None
    for k in range(L_WINDOWS):
        Xf = np.stack([vec.flatten(m) for m in members], axis=1)
        Yf = np.stack([m.sample_pressure(obs.tap_x, obs.tap_y) for m in members], axis=1)
        ybar = Yf.mean(axis=1)
        Y_k = (Yf - ybar[:, None]) / np.sqrt(Q_ENSEMBLE - 1)
        Y_blocks.append(Y_k / sigma_p)
        if k == 0:
            Xf_first = Xf - Xf.mean(axis=1, keepdims=True)
            Yf_first = Y_k
        for m in members:
            for _ in range(substeps):
                m.step()

    Y_stack = np.concatenate(Y_blocks, axis=0)  # (L*n_taps, q)
    U, S, Vt = np.linalg.svd(Y_stack, full_matrices=False)

    print('=' * 70)
    print('STAGE E: empirical observability singular values (whitened, %d windows)' % L_WINDOWS)
    print('=' * 70)
    print(np.array2string(S, precision=3, suppress_small=True))
    print('condition number (S[0]/S[-1]): %.2f' % (S[0] / S[-1]))
    print('=' * 70)

    # representer fields for two taps: index 0 (theta~?) and one near the
    # shoulder (theta close to 90deg, typically strongest fundamental content)
    theta = obs.theta()
    tap_front = int(np.argmin(np.abs(theta - np.pi)))       # near base, theta~180deg
    tap_shoulder = int(np.argmin(np.abs(theta - np.pi / 2)))  # theta~90deg, shoulder

    Pxy = Xf_first @ Yf_first.T  # (n_state, n_taps)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    axs[0].semilogy(S, 'o-')
    axs[0].set_xlabel('mode index'); axs[0].set_ylabel('singular value (whitened)')
    axs[0].set_title('Stage E: empirical observability spectrum')
    axs[0].grid(True, alpha=0.3)

    dx = (c['Lxmax'] - c['Lxmin']) / c['Nx']
    dy = (c['Lymax'] - c['Lymin']) / c['Ny']
    x_centers = c['Lxmin'] + (np.arange(c['Nx']) + 0.5) * dx
    y_centers = c['Lymin'] + (np.arange(c['Ny']) + 0.5) * dy

    for ax, tap_idx, label in [(axs[1], tap_shoulder, 'shoulder (~90deg)'),
                                (axs[2], tap_front, 'base (~180deg)')]:
        col = Pxy[:, tap_idx]
        u_field = np.zeros(vec.active_u.shape)
        u_field[vec.active_u] = col[:vec.n_u]
        u_center = 0.5 * (u_field[:, :-1] + u_field[:, 1:])
        vmax = np.percentile(np.abs(u_center), 99)
        im = ax.pcolormesh(x_centers, y_centers, u_center, shading='auto',
                            cmap='PuOr', vmin=-vmax, vmax=vmax)
        ax.scatter([obs.tap_x[tap_idx]], [obs.tap_y[tap_idx]], color='k', marker='x', s=60, zorder=5)
        ax.set_aspect('equal')
        ax.set_title('representer P_xy[u], tap %d (%s)' % (tap_idx, label))
        plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'stage_e_observability.png'), dpi=130)
    print('Figure written to figures/stage_e_observability.png')

    np.savez_compressed(os.path.join(HERE, 'stage_e_observability.npz'),
                         singular_values=S, ic_times=ic_times)


if __name__ == '__main__':
    main()
