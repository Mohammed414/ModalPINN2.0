"""Run one controlled arm of the testbed and evaluate it.

Usage: python run_arm.py --arm <name> [--width 40] [--nint 3000]
                         [--adam 800] [--lbfgs 2000]
where <name> is any key of ARMS below (baseline, rel, rzif, cv, sym,
combo, combo_init, init, trust, trust_rel).

Evaluation (truth used HERE ONLY): regional E_u/E_v/E_p in the standard
protocol + mode-1 |v| amplitude vs x on the centerline band.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import modal_pinn as mp  # noqa: E402

ARMS = {
    'baseline':  {},
    'rel':       {'rel_residual': True},
    'rzif':      {'rzif': True, 'w_rzif': 1.0},
    'cv':        {'lift_momentum': True, 'w_cv': 1.0},
    'sym':       {'hard_sym': True},
    'combo':     {'rel_residual': True, 'rzif': True, 'lift_momentum': True,
                  'hard_sym': True},
    'combo_init': {'rel_residual': True, 'rzif': True, 'lift_momentum': True,
                   'hard_sym': True, 'street_init': True},
    'init':      {'street_init': True},
    'trust':     {'trust': True},
    'trust_rel': {'trust': True, 'rel_residual': True},
    'trust_cf':  {'trust_cf': True},
}


class TorchCFStreet:
    """Adapter: closed-form street -> TrustModalPINN's street interface.

    Implements exactly the formulas the TF1 port transcribes
    (closed_form_street.CFStreet with the saved calibration applied), so a
    testbed run with this adapter validates the port's math end to end.
    k=0 modes are zeros (the trust ansatz's k=0 is a free network anyway);
    k>=1 pressure is the linearized-Bernoulli anchor -(1-Uc)*u_k*scale_p.
    """

    def __init__(self, nk=3):
        import torch
        from closed_form_street import CFStreet
        d = np.load(os.path.join(common.CACHE, 'street_fit.npz'))
        c = np.load(os.path.join(common.CACHE, 'cf_street_calibration.npz'))
        assert not bool(c['conj']), 'calibration chose conj - port assumes not'
        self.cf = CFStreet(float(d['Gamma']), float(d['Uc']), float(d['xf']),
                           float(d['r0']),
                           float(d['phase']) + float(c['extra_phase']),
                           float(d['omega']), nk=nk)
        self.amp = float(c['amp_scale'])
        # k=1 Bernoulli amplitude calibration. NOTE: the trust_cf run recorded
        # in results/ used 1.44 (an early hand calibration); R9/src/
        # street_prior.py derives 1.239 with the same |p1|-matching recipe
        # against the numeric street. The ~16% difference sits inside the
        # pressure trust region (rho=0.6) and only affects the p-mode anchor,
        # not u/v; kept aligned with production here so a rerun of this arm
        # tests exactly what the notebook ships.
        self.scale_p = 1.239
        self.Uc = float(d['Uc'])
        self.nk = nk
        self._torch = torch

    def modes(self, x, y):
        t = self._torch
        us, vs = self.cf.modes_uv(x, y, m=t)
        z = t.zeros_like(x) + 0j
        u_modes = [z] + [u * self.amp for u in us]
        v_modes = [z] + [v * self.amp for v in vs]
        p_modes = [z] + [-(1.0 - self.Uc) * self.scale_p * u
                         for u in u_modes[1:]]
        return u_modes, v_modes, p_modes


def make_trust_cf_model(width, seed):
    st = TorchCFStreet()
    return mp.TrustModalPINN(st, width=width, depth=2, seed=seed)


def make_trust_model(width, seed):
    d = np.load(os.path.join(common.CACHE, 'street_fit.npz'))
    st = mp.TorchStreet(float(d['Gamma']), float(d['Uc']), float(d['xf']),
                        float(d['r0']), float(d['phase']), float(d['omega']))
    return mp.TrustModalPINN(st, width=width, depth=2, seed=seed)


def pretrain_street(model, n=4000, iters=600, lr=2e-3, seed=7):
    """Initialize k>=1 modes on the analytic street's modal fields.

    Legitimate: the street is fitted from tap scalars + classical physics
    only. k=0 is left untouched (the street has no boundary layer; the
    physics loss trains the mean). Pretrains ONLY where the street is
    meaningful (x > xf - 1, the formation/wake zone) plus a zero target
    upstream, mirroring the known structure.
    """
    import analytic_street as ast
    d = np.load(os.path.join(common.CACHE, 'street_fit.npz'))
    st = ast.Street(float(d['Gamma']), float(d['Uc']), x_f=float(d['xf']),
                    r0=float(d['r0']), phase=float(d['phase']),
                    omega=float(d['omega']))
    pts = mp.sample_interior(n, seed=seed)
    sm = st.modes(pts, nk=mp.NK, nt=24)
    x = torch.tensor(pts[:, 0])
    y = torch.tensor(pts[:, 1])
    tgt = {q: [torch.tensor(sm[q][k]) for k in range(mp.NK + 1)]
           for q in 'uvp'}
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for it in range(iters):
        opt.zero_grad()
        u, v, p = model.modes(x, y)
        L = torch.tensor(0.0)
        for k in range(1, mp.NK + 1):
            L = (L + torch.mean(torch.abs(u[k] - tgt['u'][k]) ** 2)
                 + torch.mean(torch.abs(v[k] - tgt['v'][k]) ** 2)
                 + torch.mean(torch.abs(p[k] - tgt['p'][k]) ** 2))
        L.backward()
        opt.step()
        if it % 200 == 0:
            print(f'[pretrain {it:4d}] street-match {float(L):.3e}',
                  flush=True)
    print(f'[pretrain done] street-match {float(L):.3e}', flush=True)


def evaluate(model, tag, outdir):
    x, y, times, U, V, P = common.load_truth_fields()
    t = torch.tensor
    with torch.no_grad():
        xs, ys = t(x.astype(np.float64)), t(y.astype(np.float64))
        # chunked forward
        um, vm, pm = [], [], []
        for i in range(0, len(x), 20000):
            u1, v1, p1 = model.modes(xs[i:i + 20000], ys[i:i + 20000])
            um.append(np.stack([q.numpy() for q in u1]))
            vm.append(np.stack([q.numpy() for q in v1]))
            pm.append(np.stack([q.numpy() for q in p1]))
        um = np.concatenate(um, axis=1)  # (NK+1, N) complex
        vm = np.concatenate(vm, axis=1)
        pm = np.concatenate(pm, axis=1)

    tt = times - times[0]
    def recon(md):
        F = np.tile(md[0].real[None, :], (len(tt), 1))
        for k in range(1, mp.NK + 1):
            F = F + 2 * (md[k][None, :]
                         * np.exp(1j * k * float(anchors['omega0_hat']) * tt[:, None])).real
        return F.astype(np.float32)

    anchors = np.load(os.path.join(common.CACHE, 'tap_anchors.npz'))
    Up, Vp, Pp = recon(um), recon(vm), recon(pm)
    tbl = common.regional_table(Up, Vp, Pp, x, y, U, V, P)
    common.print_regional_table(tbl, f'--- {tag} ---')

    # mode-1 |v| amplitude profile vs x (band |y|<0.8), pred vs truth
    xm, ym, tmodes = common.load_truth_modes()
    band = np.abs(y) < 0.8
    xbins = np.arange(0.5, 8.0, 0.25)
    prof_pred, prof_true = [], []
    v1t = np.abs(tmodes['v'][1])
    v1p = np.abs(vm[1])
    for xb in xbins:
        m = band & (np.abs(x - xb) < 0.125)
        prof_pred.append(v1p[m].mean() if m.any() else np.nan)
        prof_true.append(v1t[m].mean() if m.any() else np.nan)

    res = dict(tag=tag,
               table={k: dict(n=v[0], E_u=v[1], E_v=v[2], E_p=v[3])
                      for k, v in tbl.items()},
               xbins=list(map(float, xbins)),
               v1_pred=[float(v) for v in prof_pred],
               v1_true=[float(v) for v in prof_true])
    with open(os.path.join(outdir, f'arm_{tag}.json'), 'w') as f:
        json.dump(res, f, indent=1)
    np.savez(os.path.join(outdir, f'arm_{tag}_modes.npz'),
             x=x, y=y, u=um, v=vm, p=pm)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=list(ARMS))
    ap.add_argument('--width', type=int, default=40)
    ap.add_argument('--nint', type=int, default=3000)
    ap.add_argument('--adam', type=int, default=800)
    ap.add_argument('--lbfgs', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eps', type=float, default=0.01)
    ap.add_argument('--tag', default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    flags = dict(ARMS[args.arm])
    tag = args.tag or f'{args.arm}_w{args.width}_s{args.seed}'
    outdir = os.path.join(common.R9, 'results')

    anchors = np.load(os.path.join(common.CACHE, 'tap_anchors.npz'))
    if flags.pop('trust_cf', False):
        model = make_trust_cf_model(args.width, args.seed)
        flags.pop('street_init', None)
    elif flags.pop('trust', False):
        model = make_trust_model(args.width, args.seed)
        flags.pop('street_init', None)
    else:
        model = mp.ModalPINN(width=args.width, depth=2,
                             hard_sym=flags.pop('hard_sym', False),
                             seed=args.seed)
        if flags.pop('street_init', False):
            pretrain_street(model)
    tr = mp.Trainer(model, anchors, flags=flags, n_int=args.nint,
                    eps_rel=args.eps)
    t0 = time.time()
    parts = tr.train(adam_iters=args.adam, lbfgs_iters=args.lbfgs)
    print(f'train time {time.time() - t0:.0f}s')
    res = evaluate(model, tag, outdir)
    torch.save(model.state_dict(), os.path.join(outdir, f'arm_{tag}.pt'))
    with open(os.path.join(outdir, f'arm_{tag}_loss.json'), 'w') as f:
        json.dump(dict(final=parts, history=tr.history[-50:]), f, indent=1)


if __name__ == '__main__':
    main()
