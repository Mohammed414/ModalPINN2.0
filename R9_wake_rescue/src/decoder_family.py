"""Autoencoder/decoder variant: learned nonlinear prior over the ANALYTIC
street family (user-requested; ban-compliant - trained on parametric
physics fields, never on CFD or solver data).

Stage 1 (train decoder): sample street parameters
    Gamma ~ U(1.5, 3.5), xf ~ U(0.6, 1.4), r0 ~ U(0.15, 0.45),
    phase ~ U(-pi, pi), with Uc = Uc(Gamma) from classical kinematics,
generate each member's modal fields (k=0..3) on a fixed point cloud, and
train decoder D: z in R^4 -> mode fields to reproduce them (supervised on
the analytic family only). The latent is the physical parameter vector
(normalized), so this is a physics-parameterized decoder - the
"autoencoder" with the encoder replaced by direct latent optimization,
standard practice when observations are this sparse.

Stage 2 (invert from taps): freeze D, optimize z (+ gauge constant for p)
so the decoded pressure modes at the 32 tap locations match the measured
tap harmonics; a small NS-residual penalty on the decoded field
regularizes. Report the standard regional table.

This tests the user's autoencoder idea in its legitimate form and doubles
as a robustness check on the single-fit street (does optimizing within the
family from taps land near the drag-anchored member?).
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import analytic_street as ast  # noqa: E402
import modal_pinn as mp  # noqa: E402

torch.set_default_dtype(torch.float64)
NK = 3


class Decoder(torch.nn.Module):
    """z (4) + (x,y) -> mode fields for u,v,p (1 + 2*NK reals each)."""

    def __init__(self, width=128, depth=4, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        layers = []
        din = 6  # z(4) + x + y
        for _ in range(depth):
            lin = torch.nn.Linear(din, width)
            torch.nn.init.xavier_normal_(lin.weight, generator=g)
            torch.nn.init.zeros_(lin.bias)
            layers += [lin, torch.nn.Tanh()]
            din = width
        out = torch.nn.Linear(din, 3 * (1 + 2 * NK))
        torch.nn.init.xavier_normal_(out.weight, generator=g)
        torch.nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = torch.nn.Sequential(*layers)

    def forward(self, z, x, y):
        n = len(x)
        inp = torch.cat([z.expand(n, -1), x[:, None], y[:, None]], dim=1)
        return self.net(inp)  # (N, 3*(1+2NK))

    def modes(self, z, x, y):
        raw = self.forward(z, x, y)
        out = []
        for f in range(3):
            base = f * (1 + 2 * NK)
            q = [raw[:, base] + 0j]
            for k in range(1, NK + 1):
                q.append(raw[:, base + 2 * k - 1] + 1j * raw[:, base + 2 * k])
            out.append(q)
        return out


PRIOR = dict(G=(1.5, 3.5), xf=(0.6, 1.4), r0=(0.15, 0.45),
             ph=(-np.pi, np.pi))


def z_to_params(z):
    """z in [-1,1]^4 -> physical street parameters."""
    def m(lo, hi, v):
        return lo + (hi - lo) * (v + 1) / 2
    G = m(*PRIOR['G'], np.tanh(z[0]))
    xf = m(*PRIOR['xf'], np.tanh(z[1]))
    r0 = m(*PRIOR['r0'], np.tanh(z[2]))
    ph = m(*PRIOR['ph'], np.tanh(z[3]))
    return G, xf, r0, ph


def gen_member(rng, pts, omega):
    z = rng.uniform(-1.5, 1.5, 4)  # tanh-space samples
    G, xf, r0, ph = z_to_params(z)
    Uc, _ = ast.uc_of_gamma(G, omega)
    st = ast.Street(G, Uc, x_f=xf, r0=r0, phase=ph, omega=omega)
    sm = st.modes(pts, nk=NK, nt=16)
    tgt = np.empty((len(pts), 3 * (1 + 2 * NK)))
    for f, q in enumerate('uvp'):
        base = f * (1 + 2 * NK)
        tgt[:, base] = sm[q][0].real
        for k in range(1, NK + 1):
            tgt[:, base + 2 * k - 1] = sm[q][k].real
            tgt[:, base + 2 * k] = sm[q][k].imag
    return z, tgt


def train_decoder(n_members=160, n_pts=1200, epochs=1500, lr=1e-3):
    anch = np.load(os.path.join(common.CACHE, 'tap_anchors.npz'))
    omega = float(anch['omega0_hat'])
    rng = np.random.default_rng(0)
    pts = mp.sample_interior(n_pts, seed=21)
    taps = common.load_taps()
    tx = np.load(os.path.join(common.CACHE, 'tap_anchors.npz'))
    order = tx['tap_order']
    tap_pts = np.stack([taps['x'][order], taps['y'][order]], 1)
    all_pts = np.concatenate([pts, tap_pts])  # include tap locations

    print(f'generating {n_members} family members...', flush=True)
    Z, T = [], []
    t0 = time.time()
    for i in range(n_members):
        z, tgt = gen_member(rng, all_pts, omega)
        Z.append(z)
        T.append(tgt)
        if (i + 1) % 40 == 0:
            print(f'  {i+1}/{n_members} ({time.time()-t0:.0f}s)', flush=True)
    Z = torch.tensor(np.array(Z))
    T = torch.tensor(np.array(T))          # (M, N, C)
    X = torch.tensor(all_pts[:, 0])
    Y = torch.tensor(all_pts[:, 1])

    dec = Decoder(width=128, depth=4, seed=0)
    opt = torch.optim.Adam(dec.parameters(), lr=lr)
    M = len(Z)
    for ep in range(epochs):
        idx = np.random.default_rng(ep).integers(0, M, 8)
        opt.zero_grad()
        L = torch.tensor(0.0)
        for i in idx:
            pred = dec(Z[i], X, Y)
            L = L + torch.mean((pred - T[i]) ** 2)
        L = L / len(idx)
        L.backward()
        opt.step()
        if ep % 200 == 0:
            print(f'[decoder {ep:5d}] mse {float(L):.4e}', flush=True)
    torch.save(dec.state_dict(), os.path.join(common.CACHE, 'decoder.pt'))
    print('decoder saved')
    return dec


def invert_from_taps(dec, n_restarts=8, iters=600, w_phys=0.02):
    anch = np.load(os.path.join(common.CACHE, 'tap_anchors.npz'))
    taps = common.load_taps()
    order = anch['tap_order']
    tap_x = torch.tensor(taps['x'][order])
    tap_y = torch.tensor(taps['y'][order])
    tgt = [torch.tensor(anch[f'tap_p{k}']) for k in range(NK + 1)]

    pts = mp.sample_interior(600, seed=33)
    px = torch.tensor(pts[:, 0], requires_grad=True)
    py = torch.tensor(pts[:, 1], requires_grad=True)
    omega = float(anch['omega0_hat'])

    best = None
    rng = np.random.default_rng(5)
    for r in range(n_restarts):
        z = torch.tensor(rng.uniform(-1.0, 1.0, 4), requires_grad=True)
        gauge = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([z, gauge], lr=5e-2)
        for it in range(iters):
            opt.zero_grad()
            um, vm, pm = dec.modes(z, tap_x, tap_y)
            L_tap = torch.mean(torch.abs(pm[0].real + gauge - tgt[0]) ** 2)
            for k in range(1, NK + 1):
                L_tap = L_tap + 2 * torch.mean(torch.abs(pm[k] - tgt[k]) ** 2)
            L = L_tap
            if w_phys > 0 and it % 5 == 0:
                # cheap physics regularizer: k=1 divergence on probe points
                um2, vm2, pm2 = dec.modes(z, px, py)
                gux = torch.autograd.grad(um2[1].real.sum(), px,
                                          create_graph=True)[0]
                gvy = torch.autograd.grad(vm2[1].real.sum(), py,
                                          create_graph=True)[0]
                giux = torch.autograd.grad(um2[1].imag.sum(), px,
                                           create_graph=True)[0]
                givy = torch.autograd.grad(vm2[1].imag.sum(), py,
                                           create_graph=True)[0]
                L = L + w_phys * torch.mean((gux + gvy) ** 2
                                            + (giux + givy) ** 2)
            L.backward()
            opt.step()
        with torch.no_grad():
            um, vm, pm = dec.modes(z, tap_x, tap_y)
            L_tap = float(torch.mean(torch.abs(pm[0].real + gauge - tgt[0]) ** 2)
                          + sum(2 * torch.mean(torch.abs(pm[k] - tgt[k]) ** 2)
                                for k in range(1, NK + 1)))
        G, xf, r0, ph = z_to_params(z.detach().numpy())
        print(f'restart {r}: tapfit {L_tap:.4e}  G={G:.2f} xf={xf:.2f} '
              f'r0={r0:.2f} ph={ph:+.2f}', flush=True)
        if best is None or L_tap < best[0]:
            best = (L_tap, z.detach().clone(), float(gauge))
    return best


def evaluate(dec, z, tag='decoder_inv'):
    x, y, times, U, V, P = common.load_truth_fields()
    anch = np.load(os.path.join(common.CACHE, 'tap_anchors.npz'))
    omega = float(anch['omega0_hat'])
    xs = torch.tensor(x.astype(np.float64))
    ys = torch.tensor(y.astype(np.float64))
    with torch.no_grad():
        um, vm, pm = dec.modes(z, xs, ys)
    um = [q.numpy() for q in um]
    vm = [q.numpy() for q in vm]
    pm = [q.numpy() for q in pm]
    tt = times - times[0]

    def recon(md):
        F = np.tile(md[0].real[None, :], (len(tt), 1))
        for k in range(1, NK + 1):
            F = F + 2 * (md[k][None, :]
                         * np.exp(1j * k * omega * tt[:, None])).real
        return F.astype(np.float32)

    tbl = common.regional_table(recon(um), recon(vm), recon(pm),
                                x, y, U, V, P)
    common.print_regional_table(tbl, f'--- {tag} ---')
    res = {k: dict(n=v[0], E_u=v[1], E_v=v[2], E_p=v[3])
           for k, v in tbl.items()}
    with open(os.path.join(common.R9, 'results', f'{tag}.json'), 'w') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    dec_path = os.path.join(common.CACHE, 'decoder.pt')
    if os.path.exists(dec_path):
        dec = Decoder(width=128, depth=4, seed=0)
        dec.load_state_dict(torch.load(dec_path))
        print('loaded cached decoder')
    else:
        dec = train_decoder()
    L_tap, z, gauge = invert_from_taps(dec)
    G, xf, r0, ph = z_to_params(z.numpy())
    print(f'BEST: tapfit {L_tap:.4e}  G={G:.2f} xf={xf:.2f} r0={r0:.2f} '
          f'ph={ph:+.2f}')
    np.savez(os.path.join(common.CACHE, 'decoder_inversion.npz'),
             z=z.numpy(), gauge=gauge, G=G, xf=xf, r0=r0, phase=ph)
    evaluate(dec, z)
