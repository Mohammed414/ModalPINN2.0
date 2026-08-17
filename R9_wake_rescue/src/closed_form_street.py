"""Closed-form harmonic expansion of the Karman street — the TF1-portable prior.

A periodic row of point vortices (circulation gamma, spacing a, at
z_j = z0 + j a + Uc t) induces w = u - iv = (-i gamma/(2a)) cot(pi (z-z0-Uc t)/a).
Expanding the cotangent in e^{+-2 pi i zeta / a} and collecting e^{i k w t}
terms (omega = 2 pi Uc / a) gives, for the k-th one-sided temporal mode
(convention q = q0 + sum_k 2 Re[q_k e^{i k omega t}]):

    u_k(x,y) = -sgn(y') * (gamma/(2a)) * E_k(x) * D_k(y')
    v_k(x,y) =        i * (gamma/(2a)) * E_k(x) * D_k(y')

with y' = y - y_row, E_k = e^{-i 2 pi k (x - x0)/a}, D_k = e^{-2 pi k |y'| / a}.
(u_k flips sign across the row - it is a shear mode; v_k is continuous.)

Finite (Lamb-Oseen) cores of radius rc attenuate the k-th harmonic of the
row by the Gaussian factor e^{-(pi k rc / a)^2}; viscous growth
rc^2(x) = r0^2 + 4 nu (x - xf)/Uc makes this x-dependent. The |y'| kink is
smoothed over the core scale (softabs), matching the physical smoothing.

Sign/phase conventions are validated numerically against the (already
taps-validated) numeric Lamb-Oseen street in analytic_street.py - see
validate() below. All parameters come from cache/street_fit.npz (taps-only).

This file is the REFERENCE implementation (numpy + torch); the TF1 port in
R9/src/NN_functions.py::street_modes_uv transcribes exactly these formulas.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

NU = 1.0 / common.RE


class CFStreet:
    """Closed-form street modes. Backend-agnostic: pass np or torch as `m`."""

    def __init__(self, Gamma, Uc, xf, r0, phase, omega, ramp=0.75,
                 delta=0.35, nk=3):
        self.G, self.Uc, self.xf, self.r0 = Gamma, Uc, xf, r0
        self.phase, self.omega, self.ramp = phase, omega, ramp
        self.a = 2 * np.pi * Uc / omega
        self.h = 0.281 * self.a
        self.delta = delta   # smoothing of the |y'| kink (core scale)
        self.nk = nk

    def _row_uv_k(self, m, x, y, k, y_row, gamma, x0):
        a = self.a
        yp = y - y_row
        sabs = m.sqrt(yp * yp + self.delta ** 2) - self.delta
        sgn = -m.tanh(yp / self.delta)
        # x-dependent core attenuation
        rc2 = self.r0 ** 2 + 4 * NU * m.clip(x - self.xf, 0, None) / self.Uc \
            if m is np else self.r0 ** 2 + 4 * NU * m.clamp(x - self.xf, min=0) / self.Uc
        att = m.exp(-(np.pi * k) ** 2 * rc2 / a ** 2)
        Dk = m.exp(-2 * np.pi * k * sabs / a)
        ph = -2 * np.pi * k * (x - x0) / a - k * self.phase
        Ek = m.cos(ph) + 1j * m.sin(ph)
        base = (self.G if gamma > 0 else -self.G) / (2 * a) * Ek * Dk * att
        # sign validated against the numeric Lamb-Oseen street (u_k was pi
        # out of phase with the first convention tried; v_k fixes the global
        # phase, u_k then needs +sgn, not -sgn)
        u_k = sgn * base
        v_k = 1j * base
        return u_k, v_k

    def modes_uv(self, x, y, m=np):
        """Returns (u_modes, v_modes): lists [k=1..nk] of complex arrays.

        k=0 intentionally omitted (free network in the trust ansatz).
        """
        env = 0.5 * (1 + m.tanh((x - self.xf) / self.ramp))
        us, vs = [], []
        for k in range(1, self.nk + 1):
            # upper row: gamma = -G at +h/2, x0 = xf
            uu, vu = self._row_uv_k(m, x, y, k, +self.h / 2, -1, self.xf)
            # lower row: gamma = +G at -h/2, offset a/2
            ul, vl = self._row_uv_k(m, x, y, k, -self.h / 2, +1,
                                    self.xf + self.a / 2)
            us.append((uu + ul) * env)
            vs.append((vu + vl) * env)
        return us, vs


def calibrate_and_validate():
    """Fix global sign/phase against the NUMERIC street (taps-only), then
    report correlation vs truth (eval-only diagnostic)."""
    import analytic_street as ast
    d = np.load(os.path.join(common.CACHE, 'street_fit.npz'))
    num = ast.Street(float(d['Gamma']), float(d['Uc']), x_f=float(d['xf']),
                     r0=float(d['r0']), phase=float(d['phase']),
                     omega=float(d['omega']))
    # probe grid in the wake
    rng = np.random.default_rng(3)
    pts = rng.uniform([1.0, -2.0], [8.0, 2.0], size=(1500, 2))
    sm = num.modes(pts, nk=3, nt=16)

    best = None
    for conj_flag in (False, True):
        for extra_phase in np.linspace(-np.pi, np.pi, 24, endpoint=False):
            cf = CFStreet(float(d['Gamma']), float(d['Uc']), float(d['xf']),
                          float(d['r0']), float(d['phase']) + extra_phase,
                          float(d['omega']))
            us, vs = cf.modes_uv(pts[:, 0], pts[:, 1], m=np)
            v1 = np.conj(vs[0]) if conj_flag else vs[0]
            inner = np.vdot(v1, sm['v'][1])
            corr = abs(inner) / (np.linalg.norm(v1)
                                 * np.linalg.norm(sm['v'][1]) + 1e-30)
            # also demand the phase itself aligns (not just |corr|)
            score = corr - abs(np.angle(inner)) * 0.05
            if best is None or score > best[0]:
                best = (score, corr, np.angle(inner), conj_flag, extra_phase)
    score, corr, ang, conj_flag, extra_phase = best
    print(f'calibration vs numeric street: corr={corr:.3f} residual '
          f'angle={ang:+.3f} conj={conj_flag} extra_phase={extra_phase:+.3f}')

    # amplitude ratio correction (closed form vs numeric, wake average)
    cf = CFStreet(float(d['Gamma']), float(d['Uc']), float(d['xf']),
                  float(d['r0']), float(d['phase']) + extra_phase,
                  float(d['omega']))
    us, vs = cf.modes_uv(pts[:, 0], pts[:, 1], m=np)
    v1 = np.conj(vs[0]) if conj_flag else vs[0]
    amp_ratio = np.linalg.norm(sm['v'][1]) / np.linalg.norm(v1)
    print(f'amplitude ratio numeric/closed-form: {amp_ratio:.3f}')

    np.savez(os.path.join(common.CACHE, 'cf_street_calibration.npz'),
             conj=conj_flag, extra_phase=extra_phase, amp_scale=amp_ratio,
             corr_vs_numeric=corr)

    # ---- diagnostic vs truth (eval only) ----
    x, y, tm = common.load_truth_modes()
    wake = (x > 1.0) & (np.abs(y) < 2.0)
    us, vs = cf.modes_uv(x[wake], y[wake], m=np)
    v1 = (np.conj(vs[0]) if conj_flag else vs[0]) * amp_ratio
    u1 = (np.conj(us[0]) if conj_flag else us[0]) * amp_ratio
    for nm, s, t in (('v1', v1, tm['v'][1][wake]),
                     ('u1', u1, tm['u'][1][wake])):
        inner = np.vdot(s, t)
        corr = abs(inner) / (np.linalg.norm(s) * np.linalg.norm(t))
        rel = np.linalg.norm(s - t) / np.linalg.norm(t)
        print(f'  vs TRUTH {nm}: corr={corr:.3f} phase_err={np.angle(inner):+.2f} '
              f'rel_err={rel:.3f}')
    return conj_flag, extra_phase, amp_ratio


if __name__ == '__main__':
    calibrate_and_validate()
