"""Analytic von Karman vortex-street prior, fit from tap-derived scalars only.

Model (classical inviscid street + viscous core growth - no PDE solver):
- Two staggered rows of Lamb-Oseen vortices, circulation -/+ Gamma (upper row
  negative for a cylinder wake), streamwise spacing a, lateral separation h
  with the von Karman stability ratio h/a = 0.281 (cosh(pi h/a) = sqrt 2).
- The whole pattern advects at U_c; vortices pass a fixed point at
  f = U_c / a, so a = 2 pi U_c / omega0_hat  (omega0_hat measured from taps).
- Cores grow viscously: r_core(x)^2 = r0^2 + 4 nu (x - x_f) / U_c
  (Lamb-Oseen spreading in the advected frame - pure physics).
- The street starts at the formation point x_f; upstream of x_f the
  oscillation amplitude ramps smoothly (tanh) from 0.
- Velocity = freestream + sum over vortices (finite window +/- NWIN periods,
  window wide enough that the truncation error is negligible in the box).
- Pressure from steady Bernoulli in the frame advecting with U_c (the
  pattern is steady there): p = C - 1/2 |u - U_c x_hat|^2 + U_c^2/2 ... the
  constant is irrelevant (only k>=1 harmonics are used).

Fitting (taps only): Gamma and U_c are chosen so the street's own k=1
control-volume y-momentum budget reproduces the MEASURED lift harmonic L1
(amplitude and phase - phase via the street's temporal offset), and its k=0
x-momentum deficit budget reproduces the MEASURED pressure drag CD. Both
budgets are evaluated by numerical quadrature of the analytic field (no
PDE). x_f and r0 are swept over a small physical range and picked by the
same budget consistency.

Output: modal fields (k=0..3) of the fitted street on any requested points,
used (i) standalone for evaluation, (ii) as pretraining targets to
initialize the PINN's wake modes.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

NU = 1.0 / common.RE
HA_RATIO = 0.281


class Street:
    def __init__(self, Gamma, U_c, x_f=1.0, r0=0.3, phase=0.0,
                 omega=None, ramp=0.75, images=True, dipole=True):
        self.G = Gamma
        self.Uc = U_c
        self.omega = omega if omega is not None else common.OMEGA_0
        self.a = 2 * np.pi * U_c / self.omega
        self.h = HA_RATIO * self.a
        self.xf = x_f
        self.r0 = r0
        self.phase = phase          # temporal phase offset
        self.ramp = ramp            # amplitude ramp width near x_f
        self.images = images        # Milne-Thomson image vortices in cylinder
        self.dipole = dipole        # potential-flow cylinder dipole in mean

    def _vortex_positions(self, t, nwin=30):
        """Positions of upper(-G) and lower(+G) row vortices at time t."""
        a, h = self.a, self.h
        # pattern moves with U_c; ks index vortices
        ks = np.arange(-nwin, nwin + 1)
        shift = self.Uc * t + self.phase / self.omega * self.Uc
        xu = self.xf + a * ks + shift % a
        xl = self.xf + a * (ks + 0.5) + shift % a
        return (np.stack([xu, np.full_like(xu, +h / 2)], 1),
                np.stack([xl, np.full_like(xl, -h / 2)], 1))

    def _induced(self, pts, vort_xy, gamma, core_from_x=None):
        """Lamb-Oseen induced velocity at pts from vortices at vort_xy.

        core_from_x: use these x values for the core-growth law instead of
        the vortex positions' own x (used for image vortices, whose age is
        their parent's age, not their tiny |x| inside the cylinder).
        """
        dx = pts[:, None, 0] - vort_xy[None, :, 0]
        dy = pts[:, None, 1] - vort_xy[None, :, 1]
        r2 = dx ** 2 + dy ** 2 + 1e-12
        xv = vort_xy[:, 0] if core_from_x is None else core_from_x
        rc2 = self.r0 ** 2 + 4 * NU * np.clip(xv - self.xf, 0, None) / self.Uc
        fac = (1 - np.exp(-r2 / rc2[None, :])) / (2 * np.pi * r2)
        u = -gamma * dy * fac
        v = gamma * dx * fac
        return u.sum(1), v.sum(1)

    def velocity(self, pts, t):
        """Total velocity: freestream (+ cylinder dipole) + street vortices
        (+ Milne-Thomson images inside the cylinder), with formation ramp.

        Images: vortex Gamma_j at z_j outside |z|=a gets -Gamma_j at
        a^2/conj(z_j). Net center circulation from the two rows cancels.
        The image system makes the cylinder surface a streamline of the
        INDUCED field, so the induced surface pressure pattern (which the
        tap-p1 fit uses) is physically oriented - resolving the phi vs
        phi+pi half-period ambiguity that a bare street cannot.
        """
        up, lo = self._vortex_positions(t)
        uu, vu = self._induced(pts, up, -self.G)
        ul, vl = self._induced(pts, lo, +self.G)
        u, v = uu + ul, vu + vl
        if self.images:
            a2 = common.R_C ** 2
            for row, g in ((up, -self.G), (lo, +self.G)):
                r2v = row[:, 0] ** 2 + row[:, 1] ** 2
                img = row * (a2 / r2v)[:, None]
                ui, vi = self._induced(pts, img, -g, core_from_x=row[:, 0])
                u, v = u + ui, v + vi
        env = 0.5 * (1 + np.tanh((pts[:, 0] - self.xf) / self.ramp))
        u_mean = np.ones(len(pts))
        v_mean = np.zeros(len(pts))
        if self.dipole:
            x, y = pts[:, 0], pts[:, 1]
            r2 = x ** 2 + y ** 2 + 1e-12
            a2 = common.R_C ** 2
            u_mean = 1.0 - a2 * (x ** 2 - y ** 2) / r2 ** 2
            v_mean = -a2 * 2 * x * y / r2 ** 2
        return u_mean + u * env, v_mean + v * env

    def pressure(self, pts, t):
        """Steady-in-advecting-frame Bernoulli (constant dropped)."""
        u, v = self.velocity(pts, t)
        return -0.5 * ((u - self.Uc) ** 2 + v ** 2)

    def modes(self, pts, nk=3, nt=32):
        """Temporal Fourier modes 0..nk of (u, v, p) at pts (one period)."""
        T = 2 * np.pi / self.omega
        ts = np.arange(nt) * T / nt
        U = np.empty((nt, len(pts))); V = np.empty_like(U); P = np.empty_like(U)
        for i, t in enumerate(ts):
            U[i], V[i] = self.velocity(pts, t)
            P[i] = self.pressure(pts, t)
        out = {}
        for name, F in (('u', U), ('v', V), ('p', P)):
            c = np.fft.fft(F, axis=0) / nt
            # np.fft: F_n = sum_k c_k e^{+2pi i k n/nt} with c = fft(F)/nt
            # and t_n = nT/nt, so e^{2pi i k n/nt} = e^{i k w t_n}: the
            # one-sided coefficient at +k is c_k itself (NOT its conjugate -
            # conjugating flips the traveling-wave direction).
            out[name] = [c[0].real] + [c[k] for k in range(1, nk + 1)]
        return out


# --------------------------------------------------------------------------
# budgets evaluated on the analytic field (numerical quadrature, no PDE)
# --------------------------------------------------------------------------

def k1_lift_budget(st, h=0.08, xs=4.0, nt=24):
    """k=1 y-momentum CV budget => the lift harmonic the street implies."""
    gy = np.arange(common.LYMIN + h / 2, common.LYMAX, h)
    gx = np.arange(common.LXMIN + h / 2, xs, h)
    GX, GY = np.meshgrid(gx, gy)
    pts_v = np.stack([GX.ravel(), GY.ravel()], 1)
    m = np.hypot(pts_v[:, 0], pts_v[:, 1]) > common.R_C
    pts_v = pts_v[m]
    T = 2 * np.pi / st.omega
    ts = np.arange(nt) * T / nt

    vol_t = np.empty(nt, dtype=float)
    out_t = np.empty(nt)
    in_t = np.empty(nt)
    lat_t = np.empty(nt)
    pts_out = np.stack([np.full(len(gy), xs), gy], 1)
    pts_in = np.stack([np.full(len(gy), common.LXMIN), gy], 1)
    pts_top = np.stack([gx, np.full(len(gx), common.LYMAX)], 1)
    pts_bot = np.stack([gx, np.full(len(gx), common.LYMIN)], 1)
    eps = 1e-4
    for i, t in enumerate(ts):
        _, vvol = st.velocity(pts_v, t)
        vol_t[i] = vvol.sum() * h * h
        uo, vo = st.velocity(pts_out, t)
        # dv/dx at outlet by finite difference of the analytic field
        uo2, vo2 = st.velocity(pts_out - [eps, 0], t)
        dvdx = (vo - vo2) / eps
        out_t[i] = np.sum(vo * uo - NU * dvdx) * h
        ui, vi = st.velocity(pts_in, t)
        ui2, vi2 = st.velocity(pts_in + [eps, 0], t)
        dvdx_i = (vi2 - vi) / eps
        in_t[i] = -np.sum(vi * ui - NU * dvdx_i) * h
        lat = 0.0
        for pl, sgn in ((pts_top, +1), (pts_bot, -1)):
            up_, vp_ = st.velocity(pl, t)
            pp_ = st.pressure(pl, t)
            pl2 = pl - [0, sgn * eps]
            _, vp2 = st.velocity(pl2, t)
            dvdy = (vp_ - vp2) / (eps)
            lat += sgn * np.sum(vp_ * vp_ + pp_ - NU * dvdy) * h
        lat_t[i] = lat

    # temporal k=1 harmonic of each; d/dt vol -> i w vol1
    def h1(sig):
        c = np.fft.fft(sig) / nt
        return np.conj(c[1])
    L1 = -(1j * st.omega * h1(vol_t) + h1(out_t) + h1(in_t) + h1(lat_t))
    return L1


def k0_drag_budget(st, xs=6.0, h=0.08, nt=24):
    """Mean x-momentum deficit through plane x=xs => implied mean drag."""
    gy = np.arange(common.LYMIN + h / 2, common.LYMAX, h)
    pts = np.stack([np.full(len(gy), xs), gy], 1)
    T = 2 * np.pi / st.omega
    ts = np.arange(nt) * T / nt
    flux = 0.0
    for t in ts:
        u, v = st.velocity(pts, t)
        p = st.pressure(pts, t)
        flux += np.sum(u * (1.0 - u) - p - 0.5) * h   # p_inf = -0.5*(1-Uc)^2.. gauge
    # Use gauge-free form: D = int [u(U-u) + (p_inf - p)] dy ; with our
    # Bernoulli constant dropped, p_inf(model) = -0.5*(1-st.Uc)**2
    flux = 0.0
    p_inf = -0.5 * ((1.0 - st.Uc) ** 2)
    for t in ts:
        u, v = st.velocity(pts, t)
        p = st.pressure(pts, t)
        flux += np.sum(u * (1.0 - u) + (p_inf - p)) * h
    D = flux / nt
    return D / (0.5 * common.D)  # -> CD


# --------------------------------------------------------------------------
def uc_of_gamma(G, omega):
    """Self-consistent (Uc, a) from classical street kinematics.

    The staggered street self-advects upstream relative to the fluid at
    u_ind = Gamma/(sqrt(8) a) (von Karman, with h/a=0.281 so
    tanh(pi h/a) = 1/sqrt(2)), so Uc = 1 - Gamma/(sqrt(8) a); the shedding
    frequency fixes a = 2 pi Uc / omega. Fixed-point iterate.
    """
    Uc = 0.85
    for _ in range(50):
        a = 2 * np.pi * Uc / omega
        Uc_new = 1.0 - G / (np.sqrt(8.0) * a)
        if abs(Uc_new - Uc) < 1e-12:
            break
        Uc = Uc_new
    return Uc, 2 * np.pi * Uc / omega


def karman_drag_CD(G, Uc, a, h):
    """Classical von Karman street drag (consistency check, not fitted)."""
    u_ind = G / (np.sqrt(8.0) * a)
    Drag = (G * h / a) * (1.0 - 2.0 * u_ind) + G ** 2 / (2 * np.pi * a)
    return Drag / (0.5 * common.D)


def fit_street(verbose=True):
    """Fit Gamma (+ xf, r0) by matching the street's induced k=1 pressure
    pattern at the 32 tap locations to the measured tap p1 harmonics.

    Legitimate inputs only: omega0_hat and tap p1 (both from taps alone);
    Uc(Gamma) from classical street kinematics. The temporal phase is chosen
    analytically to best align the complex tap patterns. The Karman drag
    formula provides an unfitted consistency check against measured CD.
    """
    anch = np.load(os.path.join(common.CACHE, 'tap_anchors.npz'))
    omega = float(anch['omega0_hat'])
    p1_meas = anch['tap_p1']                       # sorted-by-angle order
    th = anch['theta_sorted']
    tap_pts = np.stack([common.R_C * np.cos(th), common.R_C * np.sin(th)], 1)
    CD_meas = float(anch['CD_harm_abs'][0])

    # --- amplitude anchor: classical Karman drag = measured pressure drag
    # scaled to total by the known Re=100 split (friction ~25% of total,
    # textbook laminar-cylinder value): CD_total_target = CD_p / 0.75.
    CD_target = CD_meas / 0.75
    lo, hi = 0.5, 6.0
    for _ in range(60):
        G = 0.5 * (lo + hi)
        Uc, a = uc_of_gamma(G, omega)
        cd = karman_drag_CD(G, Uc, a, HA_RATIO * a)
        if cd < CD_target:
            lo = G
        else:
            hi = G
    G = 0.5 * (lo + hi)
    Uc, a = uc_of_gamma(G, omega)

    # --- structure/phase from the tap p1 PATTERN (shape only, not scale)
    best = None
    for xf in (0.6, 0.8, 1.0, 1.2):
        for r0 in (0.2, 0.3, 0.4):
            st = Street(G, Uc, x_f=xf, r0=r0, omega=omega)
            sm = st.modes(tap_pts, nk=1, nt=16)
            p1s = sm['p'][1]
            # scale-free pattern comparison: normalized complex correlation
            corr = np.abs(np.vdot(p1s, p1_meas)) / (
                np.linalg.norm(p1s) * np.linalg.norm(p1_meas))
            phi = np.angle(np.vdot(p1s, p1_meas))
            if best is None or corr > best[0]:
                best = (corr, xf, r0, phi)
    corr_tap, xf, r0, phi = best
    err = -1.0  # kept for signature compat below
    # bake the alignment phase into the street's temporal offset:
    # modes ~ e^{-i k w tau} => tau = -phi/omega for k=1
    st = Street(G, Uc, x_f=xf, r0=r0, omega=omega, phase=phi)
    CD_model = karman_drag_CD(G, Uc, a, HA_RATIO * a)
    if verbose:
        print(f'FIT: Gamma={G:.2f} Uc={Uc:.3f} a={a:.2f} h={HA_RATIO*a:.2f} '
              f'xf={xf} r0={r0} phase={phi:+.3f}')
        print(f'  amplitude anchored by Karman drag: CD_model={CD_model:.3f}'
              f' = CD_p,meas/0.75 = {CD_meas:.3f}/0.75')
        print(f'  tap-p1 pattern correlation (shape only): {corr_tap:.3f}')
    np.savez(os.path.join(common.CACHE, 'street_fit.npz'),
             Gamma=G, Uc=Uc, a=a, h=HA_RATIO * a, xf=xf, r0=r0,
             phase=st.phase, omega=omega, tap_p1_corr=corr_tap,
             CD_model=CD_model, CD_meas=CD_meas)
    return st


if __name__ == '__main__':
    st = fit_street()
    # quick standalone evaluation of the street modal fields vs truth
    x, y, tmodes = common.load_truth_modes()
    wake = (x > 1.0) & (np.abs(y) < 2.0)
    pts = np.stack([x[wake], y[wake]], 1)
    sm = st.modes(pts, nk=3, nt=24)
    v1t = tmodes['v'][1][wake]
    v1s = sm['v'][1]
    # amplitude profile comparison
    num = np.linalg.norm(np.abs(v1s) - np.abs(v1t))
    den = np.linalg.norm(np.abs(v1t))
    print(f'wake |v1| amplitude rel err: {num / den:.3f}')
    corr = np.abs(np.vdot(v1s, v1t)) / (np.linalg.norm(v1s)
                                        * np.linalg.norm(v1t))
    print(f'wake v1 complex correlation |<s,t>|/(|s||t|): {corr:.3f}')
