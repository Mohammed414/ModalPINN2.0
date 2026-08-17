"""Local PyTorch reimplementation of the ModalPINN ansatz + candidate losses.

Ansatz (faithful to Raynaud et al. 2022 / the repo's TF1 code):
    q(x,y,t) = q_0(x,y) + sum_{k=1}^{NK} 2 Re[ q_k(x,y) e^{i k w0 t} ],
one spatial MLP per field (u,v,p), outputs = [q0, Re q1, Im q1, ..., Re qNK, Im qNK].

Physics loss: harmonic-balance NS residual with the full two-sided
convection convolution, autodiff derivatives - mathematically equivalent to
the time-collocation residual that did all the training work in R1-R8
(exact for a truncated series; no quadrature in t needed).

LEGITIMATE INPUTS ONLY: tap harmonic coefficients (from taps alone),
measured omega0_hat and L1 (from taps alone), NS equations, BCs, and known
structural physics of the limit cycle. No truth fields anywhere here.

Candidate terms (flags):
  rel_residual   - B: k>=1 residual normalized by local mode amplitude
  rzif           - D: mean-linearized k=1 residual, amplitude-normalized
  lift_momentum  - C: k=1 y-momentum CV budgets anchored to measured L1
  hard_sym       - shift-reflect symmetry hard-wired into the ansatz
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

torch.set_default_dtype(torch.float64)
NK = 3
NU = 1.0 / common.RE


class FieldNet(torch.nn.Module):
    """MLP: (x,y) -> [q0, Re q1, Im q1, ..., Re qNK, Im qNK]."""

    def __init__(self, width=75, depth=2, nk=NK, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        layers = []
        din = 2
        for _ in range(depth):
            lin = torch.nn.Linear(din, width)
            torch.nn.init.xavier_normal_(lin.weight, generator=g)
            torch.nn.init.zeros_(lin.bias)
            layers += [lin, torch.nn.Tanh()]
            din = width
        out = torch.nn.Linear(din, 1 + 2 * nk)
        torch.nn.init.xavier_normal_(out.weight, generator=g)
        torch.nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = torch.nn.Sequential(*layers)

    def forward(self, xy):
        return self.net(xy)


class ModalPINN(torch.nn.Module):
    """Three field nets + optional hard shift-reflect symmetry.

    Shift-reflect symmetry of the Karman limit cycle:
        u(x,-y,t+T/2) = u(x,y,t)  -> u_k(x,-y) = (-1)^k u_k(x,y)
        v(x,-y,t+T/2) = -v(x,y,t) -> v_k(x,-y) = (-1)^{k+1} v_k(x,y)
        p(x,-y,t+T/2) = p(x,y,t)  -> p_k(x,-y) = (-1)^k p_k(x,y)
    Hard-wired by (anti)symmetrizing raw outputs w.r.t. y -> -y.
    """

    def __init__(self, width=75, depth=2, hard_sym=False, seed=0):
        super().__init__()
        self.u_net = FieldNet(width, depth, seed=seed)
        self.v_net = FieldNet(width, depth, seed=seed + 1)
        self.p_net = FieldNet(width, depth, seed=seed + 2)
        self.hard_sym = hard_sym

    @staticmethod
    def _split(raw):
        """raw (N, 1+2NK) -> list of (N,) complex mode fields [q0..qNK]."""
        q = [raw[:, 0] + 0j]
        for k in range(1, NK + 1):
            q.append(raw[:, 2 * k - 1] + 1j * raw[:, 2 * k])
        return q

    def modes(self, x, y):
        xy = torch.stack([x, y], dim=1)
        if not self.hard_sym:
            u = self._split(self.u_net(xy))
            v = self._split(self.v_net(xy))
            p = self._split(self.p_net(xy))
            return u, v, p
        xym = torch.stack([x, -y], dim=1)
        ru, rum = self.u_net(xy), self.u_net(xym)
        rv, rvm = self.v_net(xy), self.v_net(xym)
        rp, rpm = self.p_net(xy), self.p_net(xym)
        u, v, p = [], [], []
        for k in range(NK + 1):
            sgn = (-1.0) ** k
            sl = slice(0, 1) if k == 0 else slice(2 * k - 1, 2 * k + 1)
            uu = 0.5 * (ru[:, sl] + sgn * rum[:, sl])
            vv = 0.5 * (rv[:, sl] - sgn * rvm[:, sl])
            pp = 0.5 * (rp[:, sl] + sgn * rpm[:, sl])
            if k == 0:
                u.append(uu[:, 0] + 0j); v.append(vv[:, 0] + 0j); p.append(pp[:, 0] + 0j)
            else:
                u.append(uu[:, 0] + 1j * uu[:, 1])
                v.append(vv[:, 0] + 1j * vv[:, 1])
                p.append(pp[:, 0] + 1j * pp[:, 1])
        return u, v, p


# --------------------------------------------------------------------------
# derivatives: one backward per (channel, order) via grad-of-sum
# --------------------------------------------------------------------------

def _grads(f, x, y, second=True):
    """f: (N,) complex built from x,y with requires_grad. Returns dict."""
    out = {}
    fr, fi = f.real, f.imag
    gr = torch.autograd.grad(fr.sum(), (x, y), create_graph=True)
    gi = torch.autograd.grad(fi.sum(), (x, y), create_graph=True)
    out['x'] = gr[0] + 1j * gi[0]
    out['y'] = gr[1] + 1j * gi[1]
    if second:
        gxx = torch.autograd.grad(gr[0].sum(), x, create_graph=True)[0] \
            + 1j * torch.autograd.grad(gi[0].sum(), x, create_graph=True)[0]
        gyy = torch.autograd.grad(gr[1].sum(), y, create_graph=True)[0] \
            + 1j * torch.autograd.grad(gi[1].sum(), y, create_graph=True)[0]
        out['lap'] = gxx + gyy
    return out


def hb_residuals(model, x, y, omega):
    """Two-sided HB residuals rx[k], ry[k], div[k], k=0..NK, and mode dicts."""
    if hasattr(model, 'modes_and_derivs'):
        U, V, P, dU, dV, dP = model.modes_and_derivs(x, y)
        for k in range(1, NK + 1):
            U[-k] = torch.conj(U[k])
            V[-k] = torch.conj(V[k])
            P[-k] = torch.conj(P[k])
        _finish = True
    else:
        _finish = False
    if not _finish:
        u1, v1, p1 = model.modes(x, y)
        U = {0: u1[0]}; V = {0: v1[0]}; P = {0: p1[0]}
        for k in range(1, NK + 1):
            U[k], U[-k] = u1[k], torch.conj(u1[k])
            V[k], V[-k] = v1[k], torch.conj(v1[k])
            P[k], P[-k] = p1[k], torch.conj(p1[k])

        dU, dV, dP = {}, {}, {}
        for k in range(0, NK + 1):
            dU[k] = _grads(U[k], x, y, second=True)
            dV[k] = _grads(V[k], x, y, second=True)
            dP[k] = _grads(P[k], x, y, second=False)
    for k in range(1, NK + 1):
        dU[-k] = {a: torch.conj(dU[k][a]) for a in dU[k]}
        dV[-k] = {a: torch.conj(dV[k][a]) for a in dV[k]}

    rx, ry, dv = {}, {}, {}
    for k in range(0, NK + 1):
        cx = torch.zeros_like(U[0])
        cy = torch.zeros_like(U[0])
        for m in range(-NK, NK + 1):
            n = k - m
            if abs(n) > NK:
                continue
            cx = cx + U[m] * dU[n]['x'] + V[m] * dU[n]['y']
            cy = cy + U[m] * dV[n]['x'] + V[m] * dV[n]['y']
        rx[k] = 1j * k * omega * U[k] + cx + dP[k]['x'] - NU * dU[k]['lap']
        ry[k] = 1j * k * omega * V[k] + cy + dP[k]['y'] - NU * dV[k]['lap']
        dv[k] = dU[k]['x'] + dV[k]['y']
    return rx, ry, dv, (U, V, P), (dU, dV, dP)


class TorchStreet:
    """Differentiable (w.r.t. x,y) analytic street modes, matching
    analytic_street.Street exactly (Lamb-Oseen rows + ramp + Bernoulli).

    Constants only - no trainable parameters. Used inside the trust ansatz,
    so autograd can differentiate the physics residual through it.
    """

    def __init__(self, Gamma, Uc, xf, r0, phase, omega,
                 nwin=6, nt=12, ramp=0.75, nk=NK):
        self.G, self.Uc, self.xf, self.r0 = Gamma, Uc, xf, r0
        self.phase, self.omega, self.ramp = phase, omega, ramp
        self.a = 2 * np.pi * Uc / omega
        self.h = 0.281 * self.a
        self.nwin, self.nt, self.nk = nwin, nt, nk
        T = 2 * np.pi / omega
        self.ts = np.arange(nt) * T / nt
        # DFT row for modes 0..nk: c_k = mean_n F_n e^{-i k w t_n}
        n = np.arange(nt)
        self.dft = torch.tensor(
            np.exp(-2j * np.pi * np.outer(np.arange(nk + 1), n) / nt) / nt)

    def _uv_at(self, x, y, t):
        ks = np.arange(-self.nwin, self.nwin + 1)
        shift = (self.Uc * t + self.phase / self.omega * self.Uc) % self.a
        xu = self.xf + self.a * ks + shift
        xl = self.xf + self.a * (ks + 0.5) + shift
        # vectorized over all vortices: (N, nv)
        xv = torch.tensor(np.concatenate([xu, xl]))
        yv = torch.tensor(np.concatenate([np.full(len(xu), +self.h / 2),
                                          np.full(len(xl), -self.h / 2)]))
        g = torch.tensor(np.concatenate([np.full(len(xu), -self.G),
                                         np.full(len(xl), +self.G)]))
        rc2 = torch.tensor(self.r0 ** 2 + 4 * NU * np.clip(
            np.concatenate([xu, xl]) - self.xf, 0, None) / self.Uc)
        dx = x[:, None] - xv[None, :]
        dy = y[:, None] - yv[None, :]
        r2 = dx * dx + dy * dy + 1e-12
        fac = (1 - torch.exp(-r2 / rc2[None, :])) / (2 * np.pi * r2)
        u = -(g[None, :] * dy * fac).sum(1)
        v = (g[None, :] * dx * fac).sum(1)
        env = 0.5 * (1 + torch.tanh((x - self.xf) / self.ramp))
        return 1.0 + u * env, v * env

    def modes(self, x, y):
        """Lists [q0..qNK] (complex tensors) for u, v, p."""
        Us, Vs, Ps = [], [], []
        for t in self.ts:
            u, v = self._uv_at(x, y, t)
            Us.append(u)
            Vs.append(v)
            Ps.append(-0.5 * ((u - self.Uc) ** 2 + v ** 2))
        out = []
        for stack in (Us, Vs, Ps):
            F = torch.stack(stack, 0) + 0j          # (nt, N)
            C = self.dft @ F                        # (nk+1, N)
            out.append([C[0].real + 0j] + [C[k] for k in range(1, self.nk + 1)])
        return out  # [u_modes, v_modes, p_modes]


class TrustModalPINN(torch.nn.Module):
    """Street-anchored trust-region ansatz.

    k=0: free network (street has no boundary layer - mean is the net's job).
    k>=1: q_k = S_k + (rho*|S_k| + cap) * (tanh(a) + i tanh(b))
    The dead wake (q_k = 0) is OUTSIDE the search space wherever
    |S_k| > cap/(1-rho): amplitude floor from tap-fitted classical physics.
    """

    def __init__(self, street, width=75, depth=2, rho=0.6, cap=0.06,
                 seed=0):
        super().__init__()
        self.street = street
        self.rho, self.cap = rho, cap
        self.u_net = FieldNet(width, depth, seed=seed)
        self.v_net = FieldNet(width, depth, seed=seed + 1)
        self.p_net = FieldNet(width, depth, seed=seed + 2)

    def modes(self, x, y):
        xy = torch.stack([x, y], dim=1)
        sm_u, sm_v, sm_p = self.street.modes(x, y)
        out = []
        for net, sm in ((self.u_net, sm_u), (self.v_net, sm_v),
                        (self.p_net, sm_p)):
            raw = net(xy)
            q = [raw[:, 0] + 0j]
            for k in range(1, NK + 1):
                A = self.rho * (torch.abs(sm[k]) + 1e-12) + self.cap
                c = torch.tanh(raw[:, 2 * k - 1]) + 1j * torch.tanh(raw[:, 2 * k])
                q.append(sm[k] + A * c)
            out.append(q)
        return tuple(out)

    # ---- cached street values + FD derivatives at fixed point sets ----
    def _street_cache(self, x, y):
        key = (x.data_ptr(), len(x))
        if not hasattr(self, '_scache'):
            self._scache = {}
        if key in self._scache:
            return self._scache[key]
        h = 1e-4
        with torch.no_grad():
            evals = {}
            for tag, (xx, yy) in dict(
                    c=(x, y), xp=(x + h, y), xm=(x - h, y),
                    yp=(x, y + h), ym=(x, y - h)).items():
                evals[tag] = self.street.modes(xx, yy)
        cache = []
        for f in range(3):  # u, v, p
            per_mode = []
            for k in range(NK + 1):
                Sc = evals['c'][f][k]
                Sx = (evals['xp'][f][k] - evals['xm'][f][k]) / (2 * h)
                Sy = (evals['yp'][f][k] - evals['ym'][f][k]) / (2 * h)
                Slap = ((evals['xp'][f][k] - 2 * Sc + evals['xm'][f][k])
                        + (evals['yp'][f][k] - 2 * Sc + evals['ym'][f][k])) / h ** 2
                # amplitude envelope A = rho*sqrt(|S|^2+eps)+cap and derivs
                def mag(z):
                    return torch.sqrt(z.real ** 2 + z.imag ** 2 + 1e-12)
                Ac = self.rho * mag(Sc) + self.cap
                Axp = self.rho * mag(evals['xp'][f][k]) + self.cap
                Axm = self.rho * mag(evals['xm'][f][k]) + self.cap
                Ayp = self.rho * mag(evals['yp'][f][k]) + self.cap
                Aym = self.rho * mag(evals['ym'][f][k]) + self.cap
                Ax = (Axp - Axm) / (2 * h)
                Ay = (Ayp - Aym) / (2 * h)
                Alap = ((Axp - 2 * Ac + Axm) + (Ayp - 2 * Ac + Aym)) / h ** 2
                per_mode.append(dict(S=Sc.detach(), Sx=Sx.detach(),
                                     Sy=Sy.detach(), Slap=Slap.detach(),
                                     A=Ac.detach(), Ax=Ax.detach(),
                                     Ay=Ay.detach(), Alap=Alap.detach()))
            cache.append(per_mode)
        self._scache[key] = cache
        return cache

    def modes_and_derivs(self, x, y):
        """(U, V, P, dU, dV, dP) with street handled analytically-cached.

        q_k = S_k + A_k c_k (k>=1), q_0 = net output (street-free mean).
        c_k = tanh(a_k) + i tanh(b_k); network derivatives via autograd,
        street/envelope derivatives from the FD cache (constants).
        """
        cache = self._street_cache(x, y)
        xy = torch.stack([x, y], dim=1)
        U, V, P, dU, dV, dP = {}, {}, {}, {}, {}, {}
        for f, (net, Qd, dQd) in enumerate(((self.u_net, U, dU),
                                            (self.v_net, V, dV),
                                            (self.p_net, P, dP))):
            raw = net(xy)
            for k in range(NK + 1):
                if k == 0:
                    q = raw[:, 0]
                    gx, gy = torch.autograd.grad(q.sum(), (x, y),
                                                 create_graph=True)
                    gxx = torch.autograd.grad(gx.sum(), x, create_graph=True)[0]
                    gyy = torch.autograd.grad(gy.sum(), y, create_graph=True)[0]
                    Qd[0] = q + 0j
                    dQd[0] = {'x': gx + 0j, 'y': gy + 0j,
                              'lap': gxx + gyy + 0j}
                    continue
                st = cache[f][k]
                a = torch.tanh(raw[:, 2 * k - 1])
                b = torch.tanh(raw[:, 2 * k])
                ax, ay = torch.autograd.grad(a.sum(), (x, y),
                                             create_graph=True)
                axx = torch.autograd.grad(ax.sum(), x, create_graph=True)[0]
                ayy = torch.autograd.grad(ay.sum(), y, create_graph=True)[0]
                bx, by = torch.autograd.grad(b.sum(), (x, y),
                                             create_graph=True)
                bxx = torch.autograd.grad(bx.sum(), x, create_graph=True)[0]
                byy = torch.autograd.grad(by.sum(), y, create_graph=True)[0]
                c = a + 1j * b
                cx = ax + 1j * bx
                cy = ay + 1j * by
                clap = (axx + ayy) + 1j * (bxx + byy)
                A, Ax, Ay, Alap = st['A'], st['Ax'], st['Ay'], st['Alap']
                Qd[k] = st['S'] + A * c
                dQd[k] = {
                    'x': st['Sx'] + Ax * c + A * cx,
                    'y': st['Sy'] + Ay * c + A * cy,
                    'lap': (st['Slap'] + Alap * c + 2 * (Ax * cx + Ay * cy)
                            + A * clap),
                }
        return U, V, P, dU, dV, dP

    def saturation_stats(self, x, y):
        """Fraction of |tanh| > 0.95 (trust region binding) per mode."""
        xy = torch.stack([x, y], dim=1)
        stats = {}
        for name, net in (('u', self.u_net), ('v', self.v_net),
                          ('p', self.p_net)):
            raw = net(xy)
            for k in range(1, NK + 1):
                z = torch.tanh(raw[:, 2 * k - 1:2 * k + 1])
                stats[f'{name}{k}'] = float((z.abs() > 0.95).double().mean())
        return stats


def rzif_residual(U, V, P, dU, dV, dP, omega):
    """k=1 residual linearized about the model's own mean (RZIF form)."""
    rx = (1j * omega * U[1] + U[0] * dU[1]['x'] + V[0] * dU[1]['y']
          + U[1] * dU[0]['x'] + V[1] * dU[0]['y'] + dP[1]['x']
          - NU * dU[1]['lap'])
    ry = (1j * omega * V[1] + U[0] * dV[1]['x'] + V[0] * dV[1]['y']
          + U[1] * dV[0]['x'] + V[1] * dV[0]['y'] + dP[1]['y']
          - NU * dV[1]['lap'])
    return rx, ry


# --------------------------------------------------------------------------
# point sets (fixed seeds -> identical across arms)
# --------------------------------------------------------------------------

def sample_interior(n, seed=1):
    rng = np.random.default_rng(seed)
    pts = []
    while sum(len(p) for p in pts) < n:
        cand = rng.uniform([common.LXMIN, common.LYMIN],
                           [common.LXMAX, common.LYMAX], size=(2 * n, 2))
        r = np.hypot(cand[:, 0], cand[:, 1])
        pts.append(cand[r > common.R_C * 1.01])
    return np.concatenate(pts)[:n]


def cylinder_points(n=128):
    th = np.linspace(-np.pi, np.pi, n, endpoint=False)
    return np.stack([common.R_C * np.cos(th), common.R_C * np.sin(th)], 1)


def farfield_points(n=200, seed=2):
    rng = np.random.default_rng(seed)
    n4 = n // 4
    xs = rng.uniform(common.LXMIN, common.LXMAX, 2 * n4)
    ys = rng.uniform(common.LYMIN, common.LYMAX, 2 * n4)
    pts = np.concatenate([
        np.stack([np.full(2 * n4, common.LXMIN), ys], 1),        # inlet
        np.stack([xs, np.full(2 * n4, common.LYMIN)], 1)[:n4],   # bottom
        np.stack([xs, np.full(2 * n4, common.LYMAX)], 1)[n4:],   # top
    ])
    return pts


# --------------------------------------------------------------------------
# loss assembly
# --------------------------------------------------------------------------

class Trainer:
    def __init__(self, model, anchors, flags=None, n_int=4000,
                 eps_rel=0.01, device='cpu'):
        self.model = model
        self.flags = flags or {}
        self.dev = device
        self.omega = float(anchors['omega0_hat'])
        # tap targets (legitimate: from taps alone)
        # tap_anchors stores tap_p* sorted by angle (theta_sorted order)
        tx = np.load(os.path.join(common.CACHE, 'tap_anchors.npz'),
                     allow_pickle=False)
        taps = common.load_taps()
        order = tx['tap_order']
        self.tap_xy = torch.tensor(
            np.stack([taps['x'][order], taps['y'][order]], 1), device=device)
        self.tap_targets = [torch.tensor(tx[f'tap_p{k}'], device=device)
                            for k in range(NK + 1)]
        self.L1_meas = complex(tx['CL_harm'][1]) * 0.5 * common.D
        self.eps_rel = eps_rel

        ip = sample_interior(n_int)
        self.xi = torch.tensor(ip[:, 0], requires_grad=True, device=device)
        self.yi = torch.tensor(ip[:, 1], requires_grad=True, device=device)
        cp = cylinder_points()
        self.xc = torch.tensor(cp[:, 0], device=device)
        self.yc = torch.tensor(cp[:, 1], device=device)
        fp = farfield_points()
        self.xf = torch.tensor(fp[:, 0], device=device)
        self.yf = torch.tensor(fp[:, 1], device=device)

        if self.flags.get('lift_momentum'):
            self._build_cv_quadrature()

        self.history = []

    # ---------------- CV quadrature for the lift-momentum term ----------
    def _build_cv_quadrature(self, stations=(2.0, 4.0, 6.0), h=0.12):
        self.stations = stations
        gy = np.arange(common.LYMIN + h / 2, common.LYMAX, h)
        self.cv = {}
        for xs in stations:
            gx = np.arange(common.LXMIN + h / 2, xs, h)
            GX, GY = np.meshgrid(gx, gy)
            m = np.hypot(GX, GY) > common.R_C
            vx = torch.tensor(GX[m], requires_grad=True)
            vy = torch.tensor(GY[m], requires_grad=True)
            # planes
            out_x = torch.tensor(np.full(len(gy), xs), requires_grad=True)
            out_y = torch.tensor(gy.copy(), requires_grad=True)
            in_x = torch.tensor(np.full(len(gy), common.LXMIN),
                                requires_grad=True)
            in_y = torch.tensor(gy.copy(), requires_grad=True)
            lat_x = torch.tensor(np.concatenate([gx, gx]), requires_grad=True)
            lat_y = torch.tensor(np.concatenate(
                [np.full(len(gx), common.LYMAX), np.full(len(gx), common.LYMIN)]),
                requires_grad=True)
            lat_sgn = torch.tensor(np.concatenate(
                [np.ones(len(gx)), -np.ones(len(gx))]))
            self.cv[xs] = dict(vx=vx, vy=vy, h=h, out=(out_x, out_y),
                               inp=(in_x, in_y), lat=(lat_x, lat_y, lat_sgn))

    def _cv_budget_residual(self, xs):
        """k=1 y-momentum budget over CV [LXMIN..xs]; anchored to L1_meas.

        d/dt int v dV + oint (v u.n) dS = -oint p n_y dS + nu oint dv/dn dS - L_y
        where L_y is the k=1 force ON the cylinder (= measured lift harmonic).
        Rearranged into a single complex residual R(xs); |R|^2 / |L1|^2 is the loss.
        """
        cv = self.cv[xs]
        h = cv['h']
        model, omega = self.model, self.omega

        # volume: i w int v1 dV
        u1, v1, p1 = model.modes(cv['vx'], cv['vy'])
        vol = 1j * omega * torch.sum(v1[1]) * h * h

        def plane_flux(px, py, normal):
            """k=1 y-momentum flux integrand through a plane with unit
            normal (nx, 0) or (0, ny): (v u nx + v v ny) + p ny - nu dv/dn."""
            u, v, p = model.modes(px, py)
            U = {0: u[0]}; V = {0: v[0]}
            for k in range(1, NK + 1):
                U[k], U[-k] = u[k], torch.conj(u[k])
                V[k], V[-k] = v[k], torch.conj(v[k])
            gv = _grads(V[1], px, py, second=False)
            conv = torch.zeros_like(V[0])
            for m in range(-NK, NK + 1):
                n = 1 - m
                if abs(n) > NK:
                    continue
                conv = conv + V[m] * (U[n] if normal[0] != 0 else V[n])
            if normal[0] != 0:   # x-normal: flux = (v u) nx - nu dv/dx nx
                fl = (conv - NU * gv['x']) * normal[0]
            else:                # y-normal: (v v + p) ny - nu dv/dy ny
                fl = (conv + p[1] - NU * gv['y']) * normal[1]
            return fl

        outf = torch.sum(plane_flux(*self.cv[xs]['out'], (1, 0))) * h
        inf_ = torch.sum(plane_flux(*self.cv[xs]['inp'], (-1, 0))) * h
        lx, ly, sgn = self.cv[xs]['lat']
        u, v, p = model.modes(lx, ly)
        U = {0: u[0]}; V = {0: v[0]}
        for k in range(1, NK + 1):
            U[k], U[-k] = u[k], torch.conj(u[k])
            V[k], V[-k] = v[k], torch.conj(v[k])
        gv = _grads(V[1], lx, ly, second=False)
        conv = torch.zeros_like(V[0])
        for m in range(-NK, NK + 1):
            n = 1 - m
            if abs(n) > NK:
                continue
            conv = conv + V[m] * V[n]
        lat = torch.sum((conv + p[1] - NU * gv['y']) * sgn) * self.cv[xs]['h']

        L1 = torch.tensor(self.L1_meas)
        return vol + outf + inf_ + lat + L1

    # ---------------- the loss ----------------
    def loss(self):
        model, omega = self.model, self.omega
        f = self.flags
        rx, ry, dv, (U, V, P), (dU, dV, dP) = hb_residuals(
            model, self.xi, self.yi, omega)

        # standard physics loss (always on: k=0 full + k>=1 either std or rel)
        L_phys = (torch.mean(torch.abs(rx[0]) ** 2 + torch.abs(ry[0]) ** 2)
                  + torch.mean(torch.abs(dv[0]) ** 2))
        for k in range(1, NK + 1):
            div_term = 2 * torch.mean(torch.abs(dv[k]) ** 2)
            mom = 2 * (torch.abs(rx[k]) ** 2 + torch.abs(ry[k]) ** 2)
            if f.get('rel_residual'):
                amp2 = torch.abs(U[k]) ** 2 + torch.abs(V[k]) ** 2
                mom = mom / ((k * omega) ** 2 * (amp2 + self.eps_rel ** 2))
                div_term = 2 * torch.mean(
                    torch.abs(dv[k]) ** 2 / (omega ** 2 * (amp2 + self.eps_rel ** 2)))
            L_phys = L_phys + torch.mean(mom) + div_term

        if f.get('rzif'):
            lrx, lry = rzif_residual(U, V, P, dU, dV, dP, omega)
            amp2 = torch.abs(U[1]) ** 2 + torch.abs(V[1]) ** 2
            L_rzif = torch.mean((torch.abs(lrx) ** 2 + torch.abs(lry) ** 2)
                                / (omega ** 2 * (amp2 + self.eps_rel ** 2)))
        else:
            L_rzif = torch.tensor(0.0)

        # tap loss: match tap harmonic coefficients
        tu, tv, tp = model.modes(self.tap_xy[:, 0], self.tap_xy[:, 1])
        L_tap = torch.mean(torch.abs(tp[0].real - self.tap_targets[0]) ** 2)
        for k in range(1, NK + 1):
            L_tap = L_tap + 2 * torch.mean(
                torch.abs(tp[k] - self.tap_targets[k]) ** 2)

        # BCs: no-slip all modes on cylinder; freestream k=0 + zero k>=1 far
        cu, cv_, cp_ = model.modes(self.xc, self.yc)
        L_bc = sum(torch.mean(torch.abs(cu[k]) ** 2 + torch.abs(cv_[k]) ** 2)
                   * (1.0 if k == 0 else 2.0) for k in range(NK + 1))
        fu, fv, fp_ = model.modes(self.xf, self.yf)
        L_bc = L_bc + torch.mean(torch.abs(fu[0] - 1.0) ** 2
                                 + torch.abs(fv[0]) ** 2)
        L_bc = L_bc + sum(2 * torch.mean(torch.abs(fu[k]) ** 2
                                         + torch.abs(fv[k]) ** 2)
                          for k in range(1, NK + 1))

        if f.get('lift_momentum'):
            L_cv = sum(torch.abs(self._cv_budget_residual(xs)) ** 2
                       for xs in self.stations) / abs(self.L1_meas) ** 2 \
                / len(self.stations)
        else:
            L_cv = torch.tensor(0.0)

        w = dict(phys=1.0, tap=f.get('w_tap', 100.0),
                 bc=f.get('w_bc', 10.0), rzif=f.get('w_rzif', 1.0),
                 cv=f.get('w_cv', 1.0))
        total = (w['phys'] * L_phys + w['tap'] * L_tap + w['bc'] * L_bc
                 + w['rzif'] * L_rzif + w['cv'] * L_cv)
        parts = dict(total=float(total), phys=float(L_phys),
                     tap=float(L_tap), bc=float(L_bc),
                     rzif=float(L_rzif), cv=float(L_cv))
        return total, parts

    # ---------------- optimization ----------------
    def train(self, adam_iters=500, lbfgs_iters=1500, log_every=100,
              lr=2e-3, seed_note=''):
        model = self.model
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for it in range(adam_iters):
            opt.zero_grad()
            total, parts = self.loss()
            total.backward()
            opt.step()
            if it % log_every == 0:
                self.history.append(('adam', it, parts))
                print(f'[adam {it:5d}] ' + ' '.join(
                    f'{k}={v:.3e}' for k, v in parts.items()), flush=True)
        lb = torch.optim.LBFGS(model.parameters(), max_iter=lbfgs_iters,
                               history_size=50, tolerance_grad=1e-12,
                               tolerance_change=1e-14,
                               line_search_fn='strong_wolfe')
        self._it = 0

        def closure():
            lb.zero_grad()
            total, parts = self.loss()
            total.backward()
            if self._it % log_every == 0:
                self.history.append(('lbfgs', self._it, parts))
                print(f'[lbfgs {self._it:5d}] ' + ' '.join(
                    f'{k}={v:.3e}' for k, v in parts.items()), flush=True)
            self._it += 1
            return total
        lb.step(closure)
        total, parts = self.loss()
        print('[final] ' + ' '.join(f'{k}={v:.3e}' for k, v in parts.items()),
              flush=True)
        return parts
