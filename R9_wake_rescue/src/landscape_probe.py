"""Loss-landscape probe: evaluate each arm's objective at prescribed fields.

Fits the SAME network architecture to three prescribed mode-field targets
(supervised, converged hard), then evaluates every candidate objective at
each fitted network:
    dead   - the trained baseline arm's own final state (the dead minimum)
    street - the analytic street k>=1 + baseline's k=0 (a live-wake state
             reachable from taps+physics alone)
    truth  - network fitted to the true modal fields (diagnostic ceiling)

If L(truth) < L(dead) under a candidate objective but not under the
baseline objective, that candidate genuinely re-orders the landscape - the
structural-blindness fix the project is looking for. Truth is used here
DIAGNOSTICALLY only (landscape measurement, no method trains on it).
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import modal_pinn as mp  # noqa: E402
import analytic_street as ast  # noqa: E402


def fit_to_targets(x, y, tgt, width=40, iters=3000, lr=2e-3, seed=3,
                   note=''):
    """Supervised fit of a ModalPINN net to prescribed mode fields."""
    model = mp.ModalPINN(width=width, depth=2, seed=seed)
    xt = torch.tensor(x)
    yt = torch.tensor(y)
    T = {q: [torch.tensor(tgt[q][k]) for k in range(mp.NK + 1)]
         for q in 'uvp'}
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for it in range(iters):
        opt.zero_grad()
        u, v, p = model.modes(xt, yt)
        L = torch.tensor(0.0)
        for q, pred in (('u', u), ('v', v), ('p', p)):
            for k in range(mp.NK + 1):
                L = L + torch.mean(torch.abs(pred[k] - T[q][k]) ** 2)
        L.backward()
        opt.step()
        if it % 500 == 0:
            print(f'[{note} {it:5d}] fit {float(L):.3e}', flush=True)
    print(f'[{note} done] fit {float(L):.3e}', flush=True)
    return model


def main():
    anchors = np.load(os.path.join(common.CACHE, 'tap_anchors.npz'))
    res_dir = os.path.join(common.R9, 'results')

    # --- probe points: same interior sampler as training
    pts = mp.sample_interior(4000, seed=11)
    px, py = pts[:, 0], pts[:, 1]

    # --- target 1: dead = trained baseline checkpoint
    dead = mp.ModalPINN(width=40, depth=2, seed=0)
    dead.load_state_dict(torch.load(os.path.join(res_dir,
                                                 'arm_baseline_w40_s0.pt')))

    # --- target 2: street k>=1 on top of dead's k=0
    d = np.load(os.path.join(common.CACHE, 'street_fit.npz'))
    st = ast.Street(float(d['Gamma']), float(d['Uc']), x_f=float(d['xf']),
                    r0=float(d['r0']), phase=float(d['phase']),
                    omega=float(d['omega']))
    sm = st.modes(pts, nk=mp.NK, nt=24)
    with torch.no_grad():
        du, dv, dp = dead.modes(torch.tensor(px), torch.tensor(py))
    tgt_street = {}
    for q, dq in (('u', du), ('v', dv), ('p', dp)):
        tgt_street[q] = [dq[0].numpy()] + [sm[q][k] for k in range(1, mp.NK + 1)]
    street_net = fit_to_targets(px, py, tgt_street, note='street')

    # --- target 3: truth modal fields (diagnostic)
    xm, ym, tmodes = common.load_truth_modes()
    from scipy.interpolate import griddata
    tpts = np.stack([xm, ym], 1)
    tgt_truth = {}
    for q in 'uvp':
        tgt_truth[q] = []
        for k in range(mp.NK + 1):
            f = tmodes[q][k]
            if np.iscomplexobj(f):
                zr = griddata(tpts, f.real, (px, py), method='linear')
                zi = griddata(tpts, f.imag, (px, py), method='linear')
                z = np.nan_to_num(zr + 1j * zi)
            else:
                z = np.nan_to_num(
                    griddata(tpts, f.astype(np.float64), (px, py),
                             method='linear'))
            tgt_truth[q].append(z)
    truth_net = fit_to_targets(px, py, tgt_truth, note='truth')

    # --- evaluate every objective at each state
    arms = {
        'baseline': {},
        'rel':      {'rel_residual': True},
        'rzif':     {'rzif': True},
        'cv':       {'lift_momentum': True},
        'combo':    {'rel_residual': True, 'rzif': True,
                     'lift_momentum': True},
    }
    states = {'dead': dead, 'street': street_net, 'truth': truth_net}
    out = {}
    for aname, flags in arms.items():
        out[aname] = {}
        for sname, net in states.items():
            tr = mp.Trainer(net, anchors, flags=dict(flags), n_int=4000)
            with torch.enable_grad():
                total, parts = tr.loss()
            out[aname][sname] = parts
        b = out[aname]
        print(f"{aname:>9}: dead {b['dead']['total']:.4e}  "
              f"street {b['street']['total']:.4e}  "
              f"truth {b['truth']['total']:.4e}  "
              f"ratio dead/truth {b['dead']['total']/b['truth']['total']:.3g}",
              flush=True)

    with open(os.path.join(res_dir, 'landscape_probe.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('saved landscape_probe.json')


if __name__ == '__main__':
    main()
