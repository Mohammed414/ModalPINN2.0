"""Robustness checks on the winner (trust ansatz around the image street).

1. Tap noise: add Gaussian noise (1%, 5% of tap-pressure std) to the RAW
   tap series, re-derive ALL anchors (omega, harmonics, street fit, phase),
   re-run the winner. Tests the whole pipeline end-to-end, not just the
   final stage.
2. Fewer taps: 16-tap subset (every 2nd tap by angle), full re-derivation.
3. Seed restarts: 3 random network seeds, same anchors - does the wake
   revival depend on initialization luck?
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import modal_pinn as mp  # noqa: E402
import analytic_street as ast  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402


def derive_anchors(ptap, tap_x, tap_y, times):
    """Re-derive anchors from a (possibly noisy/subset) tap set. Returns dict."""
    t = times - times[0]
    theta = np.arctan2(tap_y, tap_x)
    order = np.argsort(theta)
    th_s = theta[order]
    p_s = ptap[:, order]
    dth = np.diff(np.concatenate([th_s, [th_s[0] + 2 * np.pi]]))
    w = 0.5 * (dth + np.roll(dth, 1))
    a = common.R_C
    CD = -(p_s * np.cos(th_s)[None, :] * w[None, :]).sum(1) * a / (0.5 * common.D)
    CL = -(p_s * np.sin(th_s)[None, :] * w[None, :]).sum(1) * a / (0.5 * common.D)

    def resid(prm):
        A, ph, w0, c = prm
        return A * np.sin(w0 * t + ph) + c - CL
    z = CL - CL.mean()
    crossings = np.where(np.diff(np.sign(z)) != 0)[0]
    w_init = np.pi / np.mean(np.diff(t[crossings]))
    fit = least_squares(resid, [0.5 * (CL.max() - CL.min()), 0.0, w_init,
                                CL.mean()], method='lm')
    omega = abs(fit.x[2])

    def harm(sig, n=3):
        cols = [np.ones_like(t)]
        for k in range(1, n + 1):
            cols += [np.cos(k * omega * t), np.sin(k * omega * t)]
        A = np.stack(cols, 1)
        cf, *_ = np.linalg.lstsq(A, sig, rcond=None)
        return [cf[0]] + [0.5 * (cf[2 * k - 1] - 1j * cf[2 * k])
                          for k in range(1, n + 1)]
    CD_h = harm(CD)
    tap_modes = common.fit_modes(t, p_s, omega, nmodes=3)
    return dict(omega=omega, CD0=CD_h[0].real, th_s=th_s, order=order,
                tap_modes=tap_modes)


def fit_street_from(anch_d):
    omega = anch_d['omega']
    CD_target = anch_d['CD0'] / 0.75
    lo, hi = 0.5, 6.0
    for _ in range(60):
        G = 0.5 * (lo + hi)
        Uc, a = ast.uc_of_gamma(G, omega)
        cd = ast.karman_drag_CD(G, Uc, a, ast.HA_RATIO * a)
        if cd < CD_target:
            lo = G
        else:
            hi = G
    G = 0.5 * (lo + hi)
    Uc, a = ast.uc_of_gamma(G, omega)
    p1_meas = anch_d['tap_modes'][1]
    th = anch_d['th_s']
    tap_pts = np.stack([common.R_C * np.cos(th), common.R_C * np.sin(th)], 1)
    best = None
    for xf in (0.6, 0.8, 1.0, 1.2):
        for r0 in (0.2, 0.3, 0.4):
            st = ast.Street(G, Uc, x_f=xf, r0=r0, omega=omega)
            sm = st.modes(tap_pts, nk=1, nt=16)
            p1s = sm['p'][1]
            corr = np.abs(np.vdot(p1s, p1_meas)) / (
                np.linalg.norm(p1s) * np.linalg.norm(p1_meas))
            phi = np.angle(np.vdot(p1s, p1_meas))
            if best is None or corr > best[0]:
                best = (corr, xf, r0, phi)
    corr, xf, r0, phi = best
    return dict(Gamma=G, Uc=Uc, xf=xf, r0=r0, phase=phi, omega=omega,
                corr=corr)


class AnchorTrainer(mp.Trainer):
    """Trainer whose tap targets come from a supplied anchor dict."""

    def __init__(self, model, anch_d, tap_pts, **kw):
        # minimal re-init that bypasses the cache file for tap targets
        fake = dict(omega0_hat=anch_d['omega'])
        super().__init__(model, fake, **kw)
        self.omega = anch_d['omega']
        self.tap_xy = torch.tensor(tap_pts)
        self.tap_targets = [torch.tensor(anch_d['tap_modes'][k])
                            for k in range(mp.NK + 1)]


def run_one(tag, ptap, tap_x, tap_y, times, seed=0, adam=600, lbfgs=1200):
    anch_d = derive_anchors(ptap, tap_x, tap_y, times)
    sf = fit_street_from(anch_d)
    print(f'[{tag}] omega={anch_d["omega"]:.4f} G={sf["Gamma"]:.2f} '
          f'xf={sf["xf"]} r0={sf["r0"]} corr={sf["corr"]:.3f}', flush=True)
    st = mp.TorchStreet(sf['Gamma'], sf['Uc'], sf['xf'], sf['r0'],
                        sf['phase'], sf['omega'])
    model = mp.TrustModalPINN(st, width=40, depth=2, seed=seed)
    th = anch_d['th_s']
    tap_pts = np.stack([common.R_C * np.cos(th), common.R_C * np.sin(th)], 1)
    tr = AnchorTrainer(model, anch_d, tap_pts, flags={}, n_int=3000)
    parts = tr.train(adam_iters=adam, lbfgs_iters=lbfgs, log_every=400)

    # evaluate
    x, y, times_t, U, V, P = common.load_truth_fields()
    xs = torch.tensor(x.astype(np.float64))
    ys = torch.tensor(y.astype(np.float64))
    um, vm, pm = [], [], []
    with torch.no_grad():
        for i in range(0, len(x), 20000):
            u1, v1, p1 = model.modes(xs[i:i + 20000], ys[i:i + 20000])
            um.append(np.stack([q.numpy() for q in u1]))
            vm.append(np.stack([q.numpy() for q in v1]))
            pm.append(np.stack([q.numpy() for q in p1]))
    um = np.concatenate(um, 1); vm = np.concatenate(vm, 1)
    pm = np.concatenate(pm, 1)
    tt = times_t - times_t[0]
    om = anch_d['omega']

    def recon(md):
        F = np.tile(md[0].real[None, :], (len(tt), 1))
        for k in range(1, mp.NK + 1):
            F = F + 2 * (md[k][None, :] * np.exp(1j * k * om * tt[:, None])).real
        return F.astype(np.float32)

    tbl = common.regional_table(recon(um), recon(vm), recon(pm),
                                x, y, U, V, P)
    common.print_regional_table(tbl, f'--- {tag} ---')
    return {k: dict(n=v[0], E_u=v[1], E_v=v[2], E_p=v[3])
            for k, v in tbl.items()}


def main():
    taps = common.load_taps()
    times = taps['times']
    rng = np.random.default_rng(42)
    p_std = taps['pressure'].std()
    out = {}

    which = sys.argv[1] if len(sys.argv) > 1 else 'all'

    if which in ('all', 'noise'):
        for lvl in (0.01, 0.05):
            noisy = taps['pressure'] + rng.normal(
                0, lvl * p_std, taps['pressure'].shape)
            out[f'noise_{lvl}'] = run_one(f'noise_{lvl}', noisy,
                                          taps['x'], taps['y'], times)
    if which in ('all', 'taps16'):
        theta = np.arctan2(taps['y'], taps['x'])
        order = np.argsort(theta)
        sub = order[::2]
        out['taps16'] = run_one('taps16', taps['pressure'][:, sub],
                                taps['x'][sub], taps['y'][sub], times)
    if which in ('all', 'seeds'):
        for seed in (1, 2):
            out[f'seed_{seed}'] = run_one(f'seed_{seed}', taps['pressure'],
                                          taps['x'], taps['y'], times,
                                          seed=seed)
    with open(os.path.join(common.R9, 'results',
                           f'robustness_{which}.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('saved robustness_%s.json' % which)


if __name__ == '__main__':
    main()
