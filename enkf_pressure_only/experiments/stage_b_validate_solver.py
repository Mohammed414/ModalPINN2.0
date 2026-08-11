"""
STAGE B: validate the independent NS solver produces the expected Re=100
periodic Karman street, with a shedding frequency in the right ballpark of
omega_0=1.036, BEFORE any EnKF work proceeds.

Diagnostics (per the spec's Experiment-0 checklist item 7): frequency
estimated independently from (a) the body force history (lift-like Fy,
computed from the IBM reaction force) and (b) the solver's own predicted
wall-pressure signal at the 32 tap locations (h(x) applied to the free-
running solver) -- both via a nonlinear single-frequency sinusoid fit
(precise regardless of record length, given the very clean single-peak
spectrum found during interactive development), not a raw FFT bin (which
at this record length is too coarse to distinguish nearby resolutions).

Also checked: no NaN/blowup, divergence-free to machine precision in the
solver interior, amplitude saturates to a constant-amplitude limit cycle
(not still growing/decaying) before the frequency fit window.

Prints PASS/FAIL and writes a JSON report + figures.
"""
import json
import os
import time
import numpy as np
from scipy.optimize import curve_fit

import estimator  # installs the leakage guard
from estimator.ns_solver import CylinderFlowSolver
from estimator.data_interface import TapObservations

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, '..', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

SEED = 0
np.random.seed(SEED)

CONFIG = dict(
    Nx=120, Ny=80, Lxmin=-4., Lxmax=8., Lymin=-4., Lymax=4.,
    x_c=0., y_c=0., r_c=0.5, Re=100., dt=0.005,
    T_total=400.0,       # total simulated time
    T_transient=250.0,   # discard before this when fitting frequency
    omega_0_reference=1.036,
    seed=SEED,
)


def fit_single_frequency(t, f, w0_guess):
    def model(t, A, w, phase, off):
        return off + A * np.cos(w * t + phase)
    p0 = [(f.max() - f.min()) / 2, w0_guess, 0.0, f.mean()]
    popt, pcov = curve_fit(model, t, f, p0=p0, maxfev=30000)
    return popt  # A, w, phase, off


def main():
    c = CONFIG
    solver = CylinderFlowSolver(Nx=c['Nx'], Ny=c['Ny'],
                                 Lxmin=c['Lxmin'], Lxmax=c['Lxmax'],
                                 Lymin=c['Lymin'], Lymax=c['Lymax'],
                                 x_c=c['x_c'], y_c=c['y_c'], r_c=c['r_c'],
                                 Re=c['Re'], dt=c['dt'])
    obs = TapObservations(n_taps=32)

    n_steps = int(round(c['T_total'] / c['dt']))
    record_every = int(round(0.05 / c['dt']))  # ~20Hz recording

    ts, Fys, Fxs, tap_p_hist = [], [], [], []
    t0 = time.time()
    for n in range(n_steps):
        solver.step()
        if n % record_every == 0:
            Fx, Fy = solver.force_on_body()
            ts.append(solver.t)
            Fxs.append(Fx)
            Fys.append(Fy)
            tap_p_hist.append(solver.sample_pressure(obs.tap_x, obs.tap_y).copy())
    wall_time = time.time() - t0

    ts = np.array(ts); Fys = np.array(Fys); Fxs = np.array(Fxs)
    tap_p_hist = np.array(tap_p_hist)  # (Nrec, 32)

    checks = {}
    checks['no_nan'] = bool(np.isfinite(solver.u).all() and np.isfinite(solver.v).all()
                             and np.isfinite(solver.p).all())

    div = ((solver.u[:, 1:] - solver.u[:, :-1]) / solver.dx
           + (solver.v[1:, :] - solver.v[:-1, :]) / solver.dy)
    max_div_interior = float(np.abs(div[3:-3, 3:-3]).max())
    checks['divergence_free_interior'] = max_div_interior < 1e-8
    checks['max_div_interior'] = max_div_interior

    mask_transient = ts > c['T_transient']
    tt = ts[mask_transient]
    amp_first = Fys[mask_transient][:len(tt) // 2].max() - Fys[mask_transient][:len(tt) // 2].min()
    amp_second = Fys[mask_transient][len(tt) // 2:].max() - Fys[mask_transient][len(tt) // 2:].min()
    checks['amplitude_saturated'] = bool(abs(amp_first - amp_second) / amp_second < 0.15)
    checks['amp_first_half'] = float(amp_first)
    checks['amp_second_half'] = float(amp_second)

    A_lift, w_lift, phase_lift, off_lift = fit_single_frequency(tt, Fys[mask_transient], 1.15)
    A_tap0, w_tap0, phase_tap0, off_tap0 = fit_single_frequency(
        tt, tap_p_hist[mask_transient, 0], 1.15)

    ratio_lift = abs(w_lift) / c['omega_0_reference']
    ratio_tap = abs(w_tap0) / c['omega_0_reference']
    checks['omega_lift'] = float(abs(w_lift))
    checks['omega_tap0'] = float(abs(w_tap0))
    checks['ratio_lift_to_omega0'] = float(ratio_lift)
    checks['ratio_tap_to_omega0'] = float(ratio_tap)
    # generous tolerance: qualitatively-right physics for a hand-written,
    # deliberately-coarser-than-truth solver -- documented in DESIGN.md
    # Stage B risk #1 (IBM no-slip is approximate). A tight match is not
    # expected or required at this resolution.
    checks['frequency_in_range'] = bool(0.7 < ratio_lift < 1.5 and 0.7 < ratio_tap < 1.5)

    overall_pass = all([checks['no_nan'], checks['divergence_free_interior'],
                         checks['amplitude_saturated'], checks['frequency_in_range']])

    report = dict(config=c, checks=checks, wall_time_seconds=wall_time,
                   overall_pass=overall_pass)
    report_path = os.path.join(HERE, 'stage_b_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    np.savez(os.path.join(HERE, 'stage_b_lift_history.npz'),
              t=ts, Fy=Fys, Fx=Fxs, tap_p=tap_p_hist, tap_x=obs.tap_x, tap_y=obs.tap_y)

    print('=' * 70)
    print('STAGE B VALIDATION REPORT')
    print('=' * 70)
    for k, v in checks.items():
        print('  %-28s %s' % (k, v))
    print('  wall_time_seconds           %.1f' % wall_time)
    print('-' * 70)
    print('OVERALL: %s' % ('PASS' if overall_pass else 'FAIL'))
    print('=' * 70)
    print('Report written to %s' % report_path)

    # ---- figures ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].plot(ts, Fys)
    axs[0].axvline(c['T_transient'], color='gray', ls=':', label='transient cutoff')
    axs[0].set_xlabel('t'); axs[0].set_ylabel('Fy (lift-like)')
    axs[0].set_title('Stage B: lift history (dx=%.3f)' % solver.dx)
    axs[0].legend()

    freqs = np.fft.rfftfreq(len(tt), d=(tt[1] - tt[0]))
    spec = np.abs(np.fft.rfft(Fys[mask_transient] - Fys[mask_transient].mean()))
    axs[1].plot(freqs, spec)
    axs[1].axvline(c['omega_0_reference'] / (2 * np.pi), color='r', ls='--', label='omega_0/2pi (truth)')
    axs[1].axvline(abs(w_lift) / (2 * np.pi), color='g', ls='--', label='fitted solver freq')
    axs[1].set_xlim(0, 1.0)
    axs[1].set_xlabel('freq'); axs[1].legend()
    axs[1].set_title('lift spectrum')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'stage_b_lift_and_spectrum.png'), dpi=130)

    u_c, v_c = solver.velocity_at_centers()
    speed = np.sqrt(u_c ** 2 + v_c ** 2)
    fig2, axs2 = plt.subplots(1, 2, figsize=(14, 5))
    im0 = axs2[0].pcolormesh(solver.Xp, solver.Yp, speed, shading='auto', cmap='viridis')
    axs2[0].set_title('speed at t=%.1f' % solver.t); axs2[0].set_aspect('equal')
    plt.colorbar(im0, ax=axs2[0])
    im1 = axs2[1].pcolormesh(solver.Xp, solver.Yp, solver.p, shading='auto', cmap='RdBu_r')
    axs2[1].set_title('pressure'); axs2[1].set_aspect('equal')
    plt.colorbar(im1, ax=axs2[1])
    fig2.tight_layout()
    fig2.savefig(os.path.join(FIG_DIR, 'stage_b_final_snapshot.png'), dpi=130)

    print('Figures written to %s' % FIG_DIR)
    return report


if __name__ == '__main__':
    main()
