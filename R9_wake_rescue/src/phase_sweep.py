"""Select the street's global phase by the LEGITIMATE training objective.

For each candidate phase phi (12 values over [-pi, pi)): build the trust
ansatz around the street rotated by phi, train briefly (Adam only), record
the final taps+physics objective. The selected phase is the argmin - no
truth anywhere. Diagnostic comparison to the truth-optimal rotation happens
only in the report.

Rationale: the k=1 field near the cylinder is pinned by the 32 taps (phase
included); the NS residual in the bridge region (cylinder -> wake) is the
judge of whether the street's wake oscillation is temporally consistent
with that anchor. A wrong global phase forces an unphysical seam that the
short training cannot erase.
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


def main():
    anchors = np.load(os.path.join(common.CACHE, 'tap_anchors.npz'))
    d = np.load(os.path.join(common.CACHE, 'street_fit.npz'))
    phis = np.linspace(-np.pi, np.pi, 12, endpoint=False)
    out = []
    for phi in phis:
        torch.manual_seed(0)
        st = mp.TorchStreet(float(d['Gamma']), float(d['Uc']),
                            float(d['xf']), float(d['r0']),
                            float(phi), float(d['omega']))
        model = mp.TrustModalPINN(st, width=24, depth=2, seed=0)
        tr = mp.Trainer(model, anchors, flags={}, n_int=1500)
        t0 = time.time()
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        for it in range(400):
            opt.zero_grad()
            total, parts = tr.loss()
            total.backward()
            opt.step()
        total, parts = tr.loss()
        out.append(dict(phi=float(phi), **{k: float(v)
                                           for k, v in parts.items()}))
        print(f'phi={phi:+.3f}: total={parts["total"]:.4e} '
              f'phys={parts["phys"]:.3e} tap={parts["tap"]:.3e} '
              f'bc={parts["bc"]:.3e}  ({time.time()-t0:.0f}s)', flush=True)

    best = min(out, key=lambda r: r['total'])
    print(f'SELECTED phase: {best["phi"]:+.3f} (total {best["total"]:.4e})')
    with open(os.path.join(common.R9, 'results', 'phase_sweep.json'),
              'w') as f:
        json.dump(dict(sweep=out, selected=best), f, indent=1)


if __name__ == '__main__':
    main()
