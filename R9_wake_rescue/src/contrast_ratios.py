"""Phase-0 gate: measure each candidate loss term's dead-vs-true contrast.

WHY THE STANDARD LOSS IS BLIND (the hypothesis this script quantifies):
every term of the k>=1 harmonic momentum residual
    i k w u_k + (conv terms) + grad p_k - nu lap u_k
is at least linear in the oscillating-mode amplitude, so residual^2 -> 0 as
the modes -> 0. Worse, (steady base flow, zero oscillation) is an EXACT NS
solution at Re=100, so "kill the wake, relax the mean toward the steady base
flow" is a genuine spurious minimum of physics; only the 32 taps oppose it,
and they live on the cylinder surface. The observed failure (live near
cylinder, dead 1-2 D downstream) is exactly that minimum, deformed locally
to fit the taps.

DIAGNOSTIC USE OF TRUTH: we build a one-parameter family q_s of fields from
the truth modal fields with the k>=1 modes progressively damped downstream
(envelope exp(-s*(x-x0)) for x > x0 = 1), mimicking the observed collapse,
and evaluate every candidate loss on each member. NOTHING here trains on
truth; this is measurement of proposed instruments.

Candidates:
  A. standard pointwise harmonic-balance NS residual (the blind baseline)
  B. amplitude-NORMALIZED (relative) k>=1 residual, eps-swept
  C. lift-anchored oscillating-momentum-flux stations (taps give L1_hat;
     NS forces the k=1 y-momentum budget of every CV [inlet..x_s] to match)
  D. linearized (about own mean) k=1 residual, amplitude-normalized -
     the trainable form of the RZIF/marginal-stability idea

Outputs: results/contrast_ratios.csv + printed table.
"""
import os
import sys

import numpy as np
from scipy.interpolate import griddata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

DX = 0.05
X0_DAMP = 1.0
NK = 3  # modes 0..3


# --------------------------------------------------------------------------
def build_grid_modes():
    """Interpolate truth modal fields onto a regular grid. Cached."""
    f = os.path.join(common.CACHE, 'grid_modes.npz')
    if os.path.exists(f):
        d = np.load(f)
        gx, gy = d['gx'], d['gy']
        modes = {q: [d[f'{q}{k}'] for k in range(NK + 1)] for q in 'uvp'}
        return gx, gy, modes
    x, y, tm = common.load_truth_modes()
    gx = np.arange(common.LXMIN, common.LXMAX + 1e-9, DX)
    gy = np.arange(common.LYMIN, common.LYMAX + 1e-9, DX)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.stack([x, y], 1)
    modes = {}
    for q in 'uvp':
        modes[q] = []
        for k in range(NK + 1):
            fk = tm[q][k]
            if np.iscomplexobj(fk):
                zr = griddata(pts, fk.real, (GX, GY), method='linear')
                zi = griddata(pts, fk.imag, (GX, GY), method='linear')
                z = zr + 1j * zi
            else:
                z = griddata(pts, fk.astype(np.float64), (GX, GY),
                             method='linear')
            modes[q].append(z)
    save = {'gx': gx, 'gy': gy}
    for q in 'uvp':
        for k in range(NK + 1):
            save[f'{q}{k}'] = modes[q][k]
    np.savez(f, **save)
    return gx, gy, modes


def two_sided(modes_q):
    """{k: C_k} for k=-NK..NK from one-sided [c0, c1..cNK]."""
    out = {0: modes_q[0].astype(complex)}
    for k in range(1, NK + 1):
        out[k] = modes_q[k]
        out[-k] = np.conj(modes_q[k])
    return out


class Diff:
    def __init__(self, gx, gy):
        self.gx, self.gy = gx, gy

    def dx(self, F):
        return np.gradient(F, self.gx, axis=1)

    def dy(self, F):
        return np.gradient(F, self.gy, axis=0)

    def lap(self, F):
        return self.dx(self.dx(F)) + self.dy(self.dy(F))


def hb_residuals(U, V, P, diff, omega, nu):
    """Two-sided harmonic-balance NS residuals for k=0..NK.

    U,V,P: dicts k->grid field (two-sided). Returns rx[k], ry[k], dv[k].
    """
    dxU = {k: diff.dx(U[k]) for k in U}
    dyU = {k: diff.dy(U[k]) for k in U}
    dxV = {k: diff.dx(V[k]) for k in V}
    dyV = {k: diff.dy(V[k]) for k in V}
    rx, ry, dv = {}, {}, {}
    for k in range(0, NK + 1):
        conv_x = np.zeros_like(U[0], dtype=complex)
        conv_y = np.zeros_like(U[0], dtype=complex)
        for m in range(-NK, NK + 1):
            n = k - m
            if abs(n) > NK:
                continue
            conv_x += U[m] * dxU[n] + V[m] * dyU[n]
            conv_y += U[m] * dxV[n] + V[m] * dyV[n]
        rx[k] = (1j * k * omega * U[k] + conv_x + diff.dx(P[k])
                 - nu * diff.lap(U[k]))
        ry[k] = (1j * k * omega * V[k] + conv_y + nu * 0
                 + diff.dy(P[k]) - nu * diff.lap(V[k]))
        dv[k] = dxU[k] + dyV[k]
    return rx, ry, dv


def lin_residual_k1(U, V, P, diff, omega, nu):
    """k=1 residual LINEARIZED about the field's own mean (RZIF form)."""
    rx = (1j * omega * U[1] + U[0] * diff.dx(U[1]) + V[0] * diff.dy(U[1])
          + U[1] * diff.dx(U[0]) + V[1] * diff.dy(U[0])
          + diff.dx(P[1]) - nu * diff.lap(U[1]))
    ry = (1j * omega * V[1] + U[0] * diff.dx(V[1]) + V[0] * diff.dy(V[1])
          + U[1] * diff.dx(V[0]) + V[1] * diff.dy(V[0])
          + diff.dy(P[1]) - nu * diff.lap(V[1]))
    return rx, ry


# --------------------------------------------------------------------------
def main():
    gx, gy, modes = build_grid_modes()
    GX, GY = np.meshgrid(gx, gy)
    diff = Diff(gx, gy)
    nu = 1.0 / common.RE

    anchors = np.load(os.path.join(common.CACHE, 'tap_anchors.npz'))
    omega = float(anchors['omega0_hat'])
    L1_meas = anchors['CL_harm'][1] * (0.5 * common.D)  # lift force k=1 (one-sided)

    r = np.sqrt(GX ** 2 + GY ** 2)
    fluid = r > (common.R_C + 2 * DX)
    interior = (fluid & (GX > common.LXMIN + 3 * DX) & (GX < common.LXMAX - 3 * DX)
                & (GY > common.LYMIN + 3 * DX) & (GY < common.LYMAX - 3 * DX)
                & (r > common.R_C + 4 * DX))
    wake = interior & (GX > 0.5) & (np.abs(GY) < 2.0)

    def nanmask(F):
        return np.where(np.isfinite(F), F, 0.0)

    base = {q: [nanmask(modes[q][k]) for k in range(NK + 1)] for q in 'uvp'}

    svals = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, np.inf]
    eps_list = [0.001, 0.01, 0.05, 0.1]
    stations = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

    rows = []
    for s in svals:
        if np.isinf(s):
            env = (GX <= X0_DAMP).astype(float)
        else:
            env = np.where(GX <= X0_DAMP, 1.0,
                           np.exp(-s * np.clip(GX - X0_DAMP, 0, None)))
        fam = {}
        for q in 'uvp':
            fam[q] = [base[q][0]] + [base[q][k] * env for k in range(1, NK + 1)]
        U, V, P = (two_sided(fam['u']), two_sided(fam['v']),
                   two_sided(fam['p']))

        rx, ry, dv = hb_residuals(U, V, P, diff, omega, nu)

        # A: standard loss (sum over modes, x2 weight for k>=1 = +/-k pair)
        def wsum(d):
            tot = np.zeros_like(d[0], dtype=float)
            for k in range(0, NK + 1):
                w = 1.0 if k == 0 else 2.0
                tot += w * np.abs(d[k]) ** 2
            return tot
        std_field = wsum(rx) + wsum(ry) + wsum(dv)
        A_dom = float(np.mean(std_field[interior]))
        A_wake = float(np.mean(std_field[wake]))
        # A restricted to k>=1 momentum only (isolates the blindness)
        osc_field = sum(2 * (np.abs(rx[k]) ** 2 + np.abs(ry[k]) ** 2)
                        for k in range(1, NK + 1))
        A1_wake = float(np.mean(osc_field[wake]))

        # B: relative residual on k>=1 (normalize by local mode amplitude)
        B = {}
        for eps in eps_list:
            tot = np.zeros_like(std_field)
            for k in range(1, NK + 1):
                amp2 = np.abs(U[k]) ** 2 + np.abs(V[k]) ** 2
                tot += 2 * (np.abs(rx[k]) ** 2 + np.abs(ry[k]) ** 2) / (
                    (k * omega) ** 2 * (amp2 + eps ** 2))
            B[eps] = float(np.mean(tot[wake]))

        # C: lift-anchored k=1 y-momentum budget per CV [inlet .. x_s]
        C_res = []
        dxg = DX
        dxV1 = diff.dx(V[1]); dyV1 = diff.dy(V[1])
        for xs in stations:
            cv = fluid & (GX <= xs)
            vol = 1j * omega * np.sum(V[1][cv]) * dxg * dxg
            i_s = np.argmin(np.abs(gx - xs))
            # outlet plane x = xs (n=+x): rho v u + 0 - nu dv/dx
            conv_out = np.zeros(len(gy), dtype=complex)
            for m in range(-NK, NK + 1):
                n = 1 - m
                if abs(n) > NK:
                    continue
                conv_out += V[m][:, i_s] * U[n][:, i_s]
            out_flux = np.sum((conv_out - nu * dxV1[:, i_s])
                              * fluid[:, i_s]) * dxg
            # inlet plane x = LXMIN (n=-x)
            conv_in = np.zeros(len(gy), dtype=complex)
            for m in range(-NK, NK + 1):
                n = 1 - m
                if abs(n) > NK:
                    continue
                conv_in += V[m][:, 0] * U[n][:, 0]
            in_flux = -np.sum((conv_in - nu * dxV1[:, 0])) * dxg
            # lateral planes y = +/-Ly (n = +/-y): rho v v + p - nu dv/dy
            lat = np.zeros((), dtype=complex)
            for j, sgn in ((len(gy) - 1, +1), (0, -1)):
                conv_l = np.zeros(len(gx), dtype=complex)
                for m in range(-NK, NK + 1):
                    n = 1 - m
                    if abs(n) > NK:
                        continue
                    conv_l += V[m][j, :] * V[n][j, :]
                mask_x = gx <= xs
                lat += sgn * np.sum((conv_l + P[1][j, :]
                                     - nu * dyV1[j, :])[mask_x]) * dxg
            R = vol + out_flux + in_flux + lat + L1_meas
            C_res.append(abs(R))
        C = float(np.mean(np.array(C_res) ** 2)) / abs(L1_meas) ** 2

        # D: linearized k=1 residual normalized by mode amplitude
        lrx, lry = lin_residual_k1(U, V, P, diff, omega, nu)
        amp2 = np.abs(U[1]) ** 2 + np.abs(V[1]) ** 2
        Dfield = (np.abs(lrx) ** 2 + np.abs(lry) ** 2) / (
            omega ** 2 * (amp2 + 0.01 ** 2))
        D_wake = float(np.mean(Dfield[wake]))

        rows.append((s, A_dom, A_wake, A1_wake, B[0.001], B[0.01], B[0.05],
                     B[0.1], C, D_wake, C_res))

    # ---------------- report ----------------
    hdr = (f"{'s':>6} {'A_dom':>11} {'A_wake':>11} {'A_k1_wake':>11} "
           f"{'B_eps.001':>11} {'B_eps.01':>11} {'B_eps.05':>11} "
           f"{'B_eps.1':>11} {'C_liftMom':>11} {'D_linRZIF':>11}")
    print(hdr)
    print('-' * len(hdr))
    for row in rows:
        s = row[0]
        lbl = 'dead' if np.isinf(s) else f'{s:.2f}'
        print(f"{lbl:>6} " + ' '.join(f'{v:>11.4g}' for v in row[1:10]))
    print()
    r0 = rows[0]
    rD = rows[-1]
    names = ['A_dom', 'A_wake', 'A_k1_wake', 'B_eps.001', 'B_eps.01',
             'B_eps.05', 'B_eps.1', 'C_liftMom', 'D_linRZIF']
    print('CONTRAST RATIO  loss(dead)/loss(true):')
    for i, nm in enumerate(names, start=1):
        print(f'  {nm:>10}: {rD[i] / max(r0[i], 1e-300):>12.4g}')
    print()
    print('Station-wise |C residual|/|L1| for true vs dead:')
    for xs, rt, rd in zip(stations, rows[0][10], rows[-1][10]):
        print(f'  x_s={xs:.1f}:  true {rt/abs(L1_meas):.4f}   dead {rd/abs(L1_meas):.4f}')

    import csv
    outf = os.path.join(common.R9, 'results', 'contrast_ratios.csv')
    with open(outf, 'w', newline='') as f:
        wcsv = csv.writer(f)
        wcsv.writerow(['s'] + names)
        for row in rows:
            wcsv.writerow([row[0]] + list(row[1:10]))
    print('saved', outf)


if __name__ == '__main__':
    main()
