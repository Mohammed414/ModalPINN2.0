"""Standalone evaluation of the fitted analytic street (no training at all).

The street's own k=0 field is freestream + induced mean of the vortex rows -
it has no boundary layer or near-cylinder recirculation, so expect it to be
poor near the cylinder and best in the wake. That's fine: its job is the
WAKE (exactly where the PINN dies), and to initialize the PINN's k>=1 modes.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import analytic_street as ast  # noqa: E402


def main():
    d = np.load(os.path.join(common.CACHE, 'street_fit.npz'))
    st = ast.Street(float(d['Gamma']), float(d['Uc']), x_f=float(d['xf']),
                    r0=float(d['r0']), phase=float(d['phase']),
                    omega=float(d['omega']))
    x, y, times, U, V, P = common.load_truth_fields()
    pts = np.stack([x, y], 1)
    sm = st.modes(pts, nk=3, nt=24)

    tt = times - times[0]
    om = float(d['omega'])

    def recon(md):
        F = np.tile(md[0][None, :].astype(np.float64), (len(tt), 1))
        for k in range(1, 4):
            F = F + 2 * (md[k][None, :] * np.exp(1j * k * om * tt[:, None])).real
        return F.astype(np.float32)

    Up, Vp, Pp = recon(sm['u']), recon(sm['v']), recon(sm['p'])
    tbl = common.regional_table(Up, Vp, Pp, x, y, U, V, P)
    common.print_regional_table(tbl, '--- analytic street (no training) ---')

    res = {k: dict(n=v[0], E_u=v[1], E_v=v[2], E_p=v[3])
           for k, v in tbl.items()}
    with open(os.path.join(common.R9, 'results', 'street_standalone.json'),
              'w') as f:
        json.dump(res, f, indent=1)
    np.savez(os.path.join(common.R9, 'results', 'street_modes.npz'),
             x=x, y=y,
             **{f'{q}{k}': sm[q][k] for q in 'uvp' for k in range(4)})


if __name__ == '__main__':
    main()
