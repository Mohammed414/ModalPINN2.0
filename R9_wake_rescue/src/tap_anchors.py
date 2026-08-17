"""Everything derivable from the 32 tap signals ALONE (fully legitimate).

Outputs cache/tap_anchors.npz with:
- omega0_hat: shedding fundamental from a nonlinear sinusoid fit to the
  lift series (not an FFT bin - the 20-t.u. record gives poor bin resolution)
- CL(t), CD(t): pressure-only force coefficients by trapezoid integration of
  p n ds around the (sorted-by-angle) taps
- harmonic amplitudes/phases of CL, CD at k=1..3
- mean, k=1..3 harmonic coefficients of every tap's pressure signal
- mean surface-pressure distribution vs angle

Note pressure-only forces omit the viscous stress contribution; at Re=100
the pressure part dominates CL fluctuations (~95%+), fine for anchoring.
"""
import os
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def main():
    taps = common.load_taps()
    x, y, times, ptap = taps['x'], taps['y'], taps['times'], taps['pressure']
    t = times - times[0]

    theta = np.arctan2(y - common.Y_C, x - common.X_C)   # (-pi, pi]
    order = np.argsort(theta)
    th_s = theta[order]
    p_s = ptap[:, order]                                  # (Nt, 32)

    # closed-polygon trapezoid weights on the circle (angle spacing)
    dth = np.diff(np.concatenate([th_s, [th_s[0] + 2 * np.pi]]))
    w = 0.5 * (dth + np.roll(dth, 1))                     # per-tap angular weight
    a = common.R_C

    # Force per unit span from pressure: F = -oint p n ds, n = (cos th, sin th)
    # ds = a dth. Coefficients: C = F / (0.5 rho U^2 D), rho=U=1, D=2a.
    CD = -(p_s * np.cos(th_s)[None, :] * w[None, :]).sum(1) * a / (0.5 * common.D)
    CL = -(p_s * np.sin(th_s)[None, :] * w[None, :]).sum(1) * a / (0.5 * common.D)

    # --- nonlinear frequency fit on CL (fundamental dominates lift) ---
    def resid(prm):
        A, ph, w0, c = prm
        return A * np.sin(w0 * t + ph) + c - CL

    A0 = 0.5 * (CL.max() - CL.min())
    # crude initial freq from zero crossings of demeaned CL
    z = CL - CL.mean()
    crossings = np.where(np.diff(np.sign(z)) != 0)[0]
    w_init = np.pi / np.mean(np.diff(t[crossings])) if len(crossings) > 3 else 1.0
    fit = least_squares(resid, [A0, 0.0, w_init, CL.mean()],
                        method='lm', max_nfev=20000)
    A_CL, ph_CL, omega0_hat, c_CL = fit.x
    if A_CL < 0:
        A_CL, ph_CL = -A_CL, ph_CL + np.pi

    # --- harmonics of CL, CD at k=1..3 of omega0_hat (lstsq) ---
    def harm(sig, w0, n=3):
        cols = [np.ones_like(t)]
        for k in range(1, n + 1):
            cols += [np.cos(k * w0 * t), np.sin(k * w0 * t)]
        Amat = np.stack(cols, 1)
        cf, *_ = np.linalg.lstsq(Amat, sig, rcond=None)
        out = [cf[0]] + [0.5 * (cf[2 * k - 1] - 1j * cf[2 * k]) for k in range(1, n + 1)]
        return out

    CL_h = harm(CL, omega0_hat)
    CD_h = harm(CD, omega0_hat)

    # --- per-tap harmonic decomposition ---
    tap_modes = common.fit_modes(t, p_s, omega0_hat, nmodes=3)

    out = os.path.join(common.CACHE, 'tap_anchors.npz')
    np.savez(out,
             theta_sorted=th_s, tap_order=order, weights=w,
             times=t, CL=CL, CD=CD,
             omega0_hat=omega0_hat,
             CL_harm_abs=np.abs(CL_h), CL_harm_arg=np.angle(CL_h),
             CD_harm_abs=np.abs(CD_h), CD_harm_arg=np.angle(CD_h),
             CL_harm=np.array(CL_h, dtype=complex),
             CD_harm=np.array(CD_h, dtype=complex),
             tap_p0=tap_modes[0], tap_p1=tap_modes[1],
             tap_p2=tap_modes[2], tap_p3=tap_modes[3])

    print(f'omega0_hat = {omega0_hat:.5f}  (project value {common.OMEGA_0})')
    print(f'St_hat     = {omega0_hat / (2 * np.pi):.5f}')
    print(f'CL: mean {CL_h[0]:+.4f}  |k1| {abs(CL_h[1]):.4f}  |k2| {abs(CL_h[2]):.4f}  |k3| {abs(CL_h[3]):.4f}')
    print(f'CD: mean {CD_h[0]:+.4f}  |k1| {abs(CD_h[1]):.4f}  |k2| {abs(CD_h[2]):.4f}  |k3| {abs(CD_h[3]):.4f}')
    print(f'tap |p1| max {np.abs(tap_modes[1]).max():.4f} at theta '
          f'{th_s[np.argmax(np.abs(tap_modes[1]))]*180/np.pi:.1f} deg')
    print('saved', out)


if __name__ == '__main__':
    main()
