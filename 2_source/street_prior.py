"""R9: derive the closed-form vortex-street prior from the 32 wall taps.

Standalone, numpy-only (no TF). Run BEFORE training:

    python street_prior.py --DataFile Data/fixed_cylinder_atRe100 --NTaps 32

Writes street_prior_Ntap<N>.npz with the closed-form street parameters that
ModalPINN_VortexShedding.py --TrustStreet consumes.

EVERY number here derives from the tap pressures + classical physics:
- omega0: nonlinear sinusoid fit to the tap-integrated lift series
- Gamma:  von Karman drag relation == tap-measured pressure drag / 0.75
          (0.75 = pressure share of total drag at Re~100, textbook value)
- Uc, a:  self-consistent street kinematics (Uc = 1 - Gamma/(sqrt8 a),
          a = 2 pi Uc / omega0)
- xf, r0, phase: matching the IMAGE-SYSTEM street's induced k=1 surface
          pressure pattern to the measured tap k=1 harmonics (the
          Milne-Thomson images make the surface pattern orientation-aware)
- closed-form calibration: the TF-portable single-harmonic expansion is
          aligned (phase offset + amplitude scale) against the numeric
          Lamb-Oseen street ON WAKE PROBE POINTS - a street-to-street
          calibration, no reference data involved.

The reference CFD file is read ONLY to extract the tap pressures - the
exact signals the training script itself trains on.

Method developed and validated in R9_wake_rescue/ (see REPORT.md there).
"""
import argparse
import os

import numpy as np
from scipy.optimize import least_squares

from text_flow import read_flow

# geometry, matching ModalPINN_VortexShedding.py
X_C, Y_C, R_C = 0.0, 0.0, 0.5
LXMIN, LXMAX, LYMIN, LYMAX = -4.0, 8.0, -4.0, 4.0
GEOM = [LXMIN, LXMAX, LYMIN, LYMAX, X_C, Y_C, R_C]
D = 2 * R_C
RE = 100.0
NU = 1.0 / RE
HA_RATIO = 0.281


# ===========================================================================
# numeric street (Lamb-Oseen rows + Milne-Thomson images + dipole)
# - reference implementation for the fit; identical math to
#   R9_wake_rescue/src/analytic_street.py
# ===========================================================================
class Street:
    def __init__(self, Gamma, U_c, x_f=1.0, r0=0.3, phase=0.0, omega=1.036,
                 ramp=0.75):
        self.G, self.Uc, self.omega = Gamma, U_c, omega
        self.a = 2 * np.pi * U_c / omega
        self.h = HA_RATIO * self.a
        self.xf, self.r0, self.phase, self.ramp = x_f, r0, phase, ramp

    def _vortex_positions(self, t, nwin=30):
        ks = np.arange(-nwin, nwin + 1)
        shift = (self.Uc * t + self.phase / self.omega * self.Uc) % self.a
        xu = self.xf + self.a * ks + shift
        xl = self.xf + self.a * (ks + 0.5) + shift
        return (np.stack([xu, np.full_like(xu, +self.h / 2)], 1),
                np.stack([xl, np.full_like(xl, -self.h / 2)], 1))

    def _induced(self, pts, vort_xy, gamma, core_from_x=None):
        dx = pts[:, None, 0] - vort_xy[None, :, 0]
        dy = pts[:, None, 1] - vort_xy[None, :, 1]
        r2 = dx ** 2 + dy ** 2 + 1e-12
        xv = vort_xy[:, 0] if core_from_x is None else core_from_x
        rc2 = self.r0 ** 2 + 4 * NU * np.clip(xv - self.xf, 0, None) / self.Uc
        fac = (1 - np.exp(-r2 / rc2[None, :])) / (2 * np.pi * r2)
        return (-gamma * dy * fac).sum(1), (gamma * dx * fac).sum(1)

    def velocity(self, pts, t):
        up, lo = self._vortex_positions(t)
        uu, vu = self._induced(pts, up, -self.G)
        ul, vl = self._induced(pts, lo, +self.G)
        u, v = uu + ul, vu + vl
        a2 = R_C ** 2
        for row, g in ((up, -self.G), (lo, +self.G)):
            r2v = row[:, 0] ** 2 + row[:, 1] ** 2
            img = row * (a2 / r2v)[:, None]
            ui, vi = self._induced(pts, img, -g, core_from_x=row[:, 0])
            u, v = u + ui, v + vi
        env = 0.5 * (1 + np.tanh((pts[:, 0] - self.xf) / self.ramp))
        x, y = pts[:, 0], pts[:, 1]
        r2 = x ** 2 + y ** 2 + 1e-12
        u_mean = 1.0 - a2 * (x ** 2 - y ** 2) / r2 ** 2
        v_mean = -a2 * 2 * x * y / r2 ** 2
        return u_mean + u * env, v_mean + v * env

    def pressure(self, pts, t):
        u, v = self.velocity(pts, t)
        return -0.5 * ((u - self.Uc) ** 2 + v ** 2)

    def modes(self, pts, nk=3, nt=16):
        T = 2 * np.pi / self.omega
        ts = np.arange(nt) * T / nt
        U = np.empty((nt, len(pts))); V = np.empty_like(U); P = np.empty_like(U)
        for i, t in enumerate(ts):
            U[i], V[i] = self.velocity(pts, t)
            P[i] = self.pressure(pts, t)
        out = {}
        for name, F in (('u', U), ('v', V), ('p', P)):
            c = np.fft.fft(F, axis=0) / nt
            out[name] = [c[0].real] + [c[k] for k in range(1, nk + 1)]
        return out


def uc_of_gamma(G, omega):
    Uc = 0.85
    for _ in range(50):
        a = 2 * np.pi * Uc / omega
        Uc_new = 1.0 - G / (np.sqrt(8.0) * a)
        if abs(Uc_new - Uc) < 1e-12:
            break
        Uc = Uc_new
    return Uc, 2 * np.pi * Uc / omega


def karman_drag_CD(G, Uc, a, h):
    u_ind = G / (np.sqrt(8.0) * a)
    return ((G * h / a) * (1.0 - 2.0 * u_ind) + G ** 2 / (2 * np.pi * a)) \
        / (0.5 * D)


# ===========================================================================
# closed-form street (TF-portable) - identical math to
# R9_wake_rescue/src/closed_form_street.py, numpy backend
# ===========================================================================
def cf_modes_uv(x, y, prm, nk=3):
    """One-sided modes k=1..nk of the closed-form street. Returns us, vs."""
    G, Uc, xf, r0, phase, omega, ramp, delta = (
        prm['Gamma'], prm['Uc'], prm['xf'], prm['r0'], prm['phase'],
        prm['omega'], prm.get('ramp', 0.75), prm.get('delta', 0.35))
    a = 2 * np.pi * Uc / omega
    h = HA_RATIO * a
    env = 0.5 * (1 + np.tanh((x - xf) / ramp))
    rc2 = r0 ** 2 + 4 * NU * np.clip(x - xf, 0, None) / Uc
    us, vs = [], []
    for k in range(1, nk + 1):
        att = np.exp(-(np.pi * k) ** 2 * rc2 / a ** 2)
        tot_u = np.zeros_like(x, dtype=complex)
        tot_v = np.zeros_like(x, dtype=complex)
        for y_row, sgn_row, x0 in ((+h / 2, -1.0, xf), (-h / 2, +1.0, xf + a / 2)):
            yp = y - y_row
            sabs = np.sqrt(yp ** 2 + delta ** 2) - delta
            sgn = -np.tanh(yp / delta)
            Dk = np.exp(-2 * np.pi * k * sabs / a)
            ph = -2 * np.pi * k * (x - x0) / a - k * phase
            Ek = np.cos(ph) + 1j * np.sin(ph)
            base = sgn_row * G / (2 * a) * Ek * Dk * att
            tot_u = tot_u + sgn * base
            tot_v = tot_v + 1j * base
        us.append(tot_u * env)
        vs.append(tot_v * env)
    return us, vs


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--DataFile', default='Data/fixed_cylinder_atRe100')
    ap.add_argument('--NTaps', type=int, default=32)
    ap.add_argument('--Out', default=None)
    args = ap.parse_args()
    out_path = args.Out or f'street_prior_Ntap{args.NTaps}.npz'

    # ---- 1. tap pressures - same selection logic as Load_train_data_desync
    # cut_simu_cylinder_only (transcribed, not imported: that module imports
    # tensorflow, which this numpy-only script must not depend on).
    Re_, Ur_, times, nodes_X, nodes_Y, Us, Vs, Ps = read_flow(args.DataFile)
    eps = 1e-5
    r_all = np.sqrt((nodes_X[0, :] - X_C) ** 2 + (nodes_Y[0, :] - Y_C) ** 2)
    idx_cyl = np.argwhere((r_all - R_C) ** 2 < eps)[:, 0]
    xc_all, yc_all = nodes_X[0, idx_cyl], nodes_Y[0, idx_cyl]
    s_lin = np.linspace(0., 1., args.NTaps, endpoint=False)
    x_t = X_C + R_C * np.cos(2 * np.pi * s_lin)
    y_t = Y_C + R_C * np.sin(2 * np.pi * s_lin)
    pick = np.array([np.argmin((xc_all - x_t[k]) ** 2 + (yc_all - y_t[k]) ** 2)
                     for k in range(args.NTaps)])
    print('Cylinder taps requested: %d, distinct mesh nodes found: %d'
          % (args.NTaps, len(np.unique(pick))))
    x_cyl, y_cyl = xc_all[pick], yc_all[pick]
    p_cyl = Ps[:, idx_cyl[pick]]             # (Nt, NTaps)
    t = np.asarray(times) - times[0]
    print(f'taps: {p_cyl.shape}, t in [0, {t[-1]:.1f}]')

    theta = np.arctan2(y_cyl - Y_C, x_cyl - X_C)
    order = np.argsort(theta)
    th_s, p_s = theta[order], p_cyl[:, order]
    dth = np.diff(np.concatenate([th_s, [th_s[0] + 2 * np.pi]]))
    w = 0.5 * (dth + np.roll(dth, 1))
    CD = -(p_s * np.cos(th_s)[None, :] * w[None, :]).sum(1) * R_C / (0.5 * D)
    CL = -(p_s * np.sin(th_s)[None, :] * w[None, :]).sum(1) * R_C / (0.5 * D)

    # ---- 2. omega0 from a nonlinear sinusoid fit to CL
    z = CL - CL.mean()
    crossings = np.where(np.diff(np.sign(z)) != 0)[0]
    w_init = np.pi / np.mean(np.diff(t[crossings]))
    fit = least_squares(
        lambda prm: prm[0] * np.sin(prm[2] * t + prm[1]) + prm[3] - CL,
        [0.5 * (CL.max() - CL.min()), 0.0, w_init, CL.mean()], method='lm')
    omega = abs(float(fit.x[2]))
    CD0 = float(CD.mean())
    print(f'omega0_hat = {omega:.5f}  CD_pressure = {CD0:.4f}')

    # ---- 3. per-tap k=1 harmonics
    cols = [np.ones_like(t), np.cos(omega * t), np.sin(omega * t)]
    A = np.stack(cols, 1)
    cf_, *_ = np.linalg.lstsq(A, p_s, rcond=None)
    p1_meas = 0.5 * (cf_[1] - 1j * cf_[2])

    # ---- 4. Gamma from the Karman drag relation (bisection)
    CD_target = CD0 / 0.75
    lo, hi = 0.5, 6.0
    for _ in range(60):
        G = 0.5 * (lo + hi)
        Uc, a = uc_of_gamma(G, omega)
        if karman_drag_CD(G, Uc, a, HA_RATIO * a) < CD_target:
            lo = G
        else:
            hi = G
    G = 0.5 * (lo + hi)
    Uc, a = uc_of_gamma(G, omega)
    print(f'Gamma = {G:.3f}  Uc = {Uc:.3f}  a = {a:.3f}')

    # ---- 5. xf, r0, phase from the tap k=1 pattern (image street)
    tap_pts = np.stack([R_C * np.cos(th_s), R_C * np.sin(th_s)], 1)
    best = None
    for xf in (0.6, 0.8, 1.0, 1.2):
        for r0 in (0.2, 0.3, 0.4):
            st = Street(G, Uc, x_f=xf, r0=r0, omega=omega)
            sm = st.modes(tap_pts, nk=1, nt=16)
            p1s = sm['p'][1]
            corr = np.abs(np.vdot(p1s, p1_meas)) / (
                np.linalg.norm(p1s) * np.linalg.norm(p1_meas))
            phi = np.angle(np.vdot(p1s, p1_meas))
            if best is None or corr > best[0]:
                best = (corr, xf, r0, phi)
    corr_tap, xf, r0, phi = best
    print(f'xf = {xf}  r0 = {r0}  phase = {phi:+.3f}  tap-p1 corr = {corr_tap:.3f}')

    # ---- 6. calibrate the closed form against the numeric street
    num = Street(G, Uc, x_f=xf, r0=r0, phase=phi, omega=omega)
    rng = np.random.default_rng(3)
    pts = rng.uniform([1.0, -2.0], [8.0, 2.0], size=(1500, 2))
    sm = num.modes(pts, nk=3, nt=16)
    prm = dict(Gamma=G, Uc=Uc, xf=xf, r0=r0, omega=omega, phase=phi)
    best = None
    for extra in np.linspace(-np.pi, np.pi, 48, endpoint=False):
        prm['phase'] = phi + extra
        us, vs = cf_modes_uv(pts[:, 0], pts[:, 1], prm, nk=1)
        inner = np.vdot(vs[0], sm['v'][1])
        corr = abs(inner) / (np.linalg.norm(vs[0])
                             * np.linalg.norm(sm['v'][1]) + 1e-30)
        score = corr - abs(np.angle(inner)) * 0.05
        if best is None or score > best[0]:
            best = (score, corr, extra)
    _, corr_cf, extra = best
    prm['phase'] = phi + extra
    us, vs = cf_modes_uv(pts[:, 0], pts[:, 1], prm, nk=1)
    amp_scale = float(np.linalg.norm(sm['v'][1]) / np.linalg.norm(vs[0]))
    # pressure anchor: p_k ~ -(1-Uc) u_k, amplitude-matched at k=1
    p1_approx = -(1.0 - Uc) * us[0] * amp_scale
    scale_p = float(np.linalg.norm(sm['p'][1]) / np.linalg.norm(p1_approx))
    print(f'closed-form calibration: corr vs numeric = {corr_cf:.3f}, '
          f'amp_scale = {amp_scale:.3f}, scale_p = {scale_p:.3f}')
    assert corr_cf > 0.95, 'closed-form street failed to match numeric street'

    np.savez(out_path,
             Gamma=G, Uc=Uc, xf=xf, r0=r0, omega=omega,
             phase=prm['phase'], amp_scale=amp_scale, scale_p=scale_p,
             ramp=0.75, delta=0.35,
             CD_pressure=CD0, tap_p1_corr=corr_tap,
             cf_corr_vs_numeric=corr_cf)
    print('saved', out_path)


if __name__ == '__main__':
    main()
