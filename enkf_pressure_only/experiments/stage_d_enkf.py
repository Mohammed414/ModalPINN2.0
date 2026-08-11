"""
STAGE D: minimal pressure-only EnKF, 32 taps, q=16.

Ensemble initial condition: q members, each a snapshot of the SAME
observer solver's own saturated limit cycle (spinup_snapshots.npz),
jittered in time by a small random offset around the same base phase
Stage C's free-run used (310.0). Every member is therefore individually a
physically valid divergence-free/no-slip/BC-satisfying state (it's a real
point on the solver's own trajectory); the jitter gives a modest, honest
ensemble spread; nothing here is derived from the reference CFD.

Runs the assimilation for the same 201 instants as the tap dataset,
recording every diagnostic the spec asks for at every step.

variant=None runs the correct pressure sequence (the actual Experiment 2).
variant='shuffled' / 'scrambled_sensors' implement the negative controls C/D.
"""
import argparse
import json
import os
import numpy as np

import estimator
from estimator.ns_solver import CylinderFlowSolver
from estimator.data_interface import TapObservations
from estimator.enkf import EnKFRun

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 0

Q_ENSEMBLE = 16
BASE_IC_TIME = 310.0
JITTER_HALF_RANGE = 0.4  # time units, ~7.5% of the ~5.37 shedding period
INFLATION = 1.0


def build_ensemble(spin, base_time, q, jitter_half_range, rng, c):
    times_avail = spin['times']
    offsets = rng.uniform(-jitter_half_range, jitter_half_range, size=q)
    members = []
    chosen_times = []
    for j in range(q):
        idx = int(np.argmin(np.abs(times_avail - (base_time + offsets[j]))))
        solver = CylinderFlowSolver(Nx=c['Nx'], Ny=c['Ny'],
                                     Lxmin=c['Lxmin'], Lxmax=c['Lxmax'],
                                     Lymin=c['Lymin'], Lymax=c['Lymax'],
                                     x_c=c['x_c'], y_c=c['y_c'], r_c=c['r_c'],
                                     Re=c['Re'], dt=c['dt'])
        solver.u = spin['u'][idx].copy()
        solver.v = spin['v'][idx].copy()
        solver.p = spin['p'][idx].copy()  # dynamically-consistent pressure at this instant
        solver.t = 0.0
        members.append(solver)
        chosen_times.append(times_avail[idx])
    return members, np.array(chosen_times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['nominal', 'shuffled', 'scrambled_sensors'], default='nominal')
    ap.add_argument('--q', type=int, default=Q_ENSEMBLE)
    ap.add_argument('--inflation', type=float, default=INFLATION)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    spin = np.load(os.path.join(HERE, 'spinup_snapshots.npz'))
    c = json.loads(str(spin['solver_config']))
    obs = TapObservations(n_taps=32)

    rng = np.random.default_rng(args.seed)
    members, ic_times = build_ensemble(spin, BASE_IC_TIME, args.q, JITTER_HALF_RANGE, rng, c)
    print('Ensemble ICs (solver-internal times used): %s' % np.round(ic_times, 3))

    tap_p_series = obs.tap_p.copy()
    if args.variant == 'shuffled':
        perm = rng.permutation(len(obs.tap_times))
        tap_p_series = tap_p_series[perm]
        print('NEGATIVE CONTROL: pressure time sequence shuffled (perm seed=%d)' % args.seed)
    elif args.variant == 'scrambled_sensors':
        sensor_perm = rng.permutation(obs.n_taps)
        tap_p_series = tap_p_series[:, sensor_perm]
        print('NEGATIVE CONTROL: sensor identity scrambled (perm seed=%d)' % args.seed)

    # sigma_p calibrated via observation-space diagnostics only (NIS, ensemble
    # spread stability), NEVER against withheld field error -- see
    # docs/DESIGN.md Stage D notes. A naive 1% floor caused the ensemble
    # pressure spread to collapse ~10x within 8 cycles (NIS ~30,000, i.e. R
    # negligible vs. genuine model-truth mismatch, so the near-singular
    # (Y Y^T + R) in the ensemble's unobserved directions was dominated by
    # R^{-1} and produced wild corrections there). Sweeping sigma_p as a
    # fraction of the tap pressure std: frac=0.3 gives NIS ~38-48 (close to
    # the ideal ~32 for a 32-dim observation), stable (non-collapsing)
    # ensemble spread, and a genuinely-decreasing-but-bounded innovation.
    # frac=0.01 collapses; frac=1.0 over-regularizes (NIS ~3-4, too
    # conservative, barely uses the observations at all).
    sigma_p = 0.3 * np.std(obs.tap_p)
    print('sigma_p (observation-space calibrated, 0.3*std(tap_p)) = %.6f' % sigma_p)

    dt_assim = obs.tap_times[1] - obs.tap_times[0]
    substeps = int(round(dt_assim / c['dt']))
    n_assim = len(obs.tap_times)

    enkf = EnKFRun(members, obs, obs_noise_std=sigma_p, inflation=args.inflation, seed=args.seed)

    diags = []
    u_mean_hist = np.empty((n_assim,) + members[0].u.shape, dtype=np.float32)
    v_mean_hist = np.empty((n_assim,) + members[0].v.shape, dtype=np.float32)
    Fy_mean_hist = np.empty(n_assim)

    for k in range(n_assim):
        d = enkf.assimilate_step(tap_p_series[k])
        diags.append(d)
        u_mean_hist[k] = np.mean([m.u for m in members], axis=0)
        v_mean_hist[k] = np.mean([m.v for m in members], axis=0)
        if k % 40 == 0 or k == n_assim - 1:
            print('k=%3d t=%.2f  |innov|=%.4f  |K.innov|=%.4f  spread_p=%.4f  NIS=%.3f'
                  % (k, obs.tap_times[k] - obs.tap_times[0], d['innovation_norm'],
                     d['kalman_correction_norm'], d['ensemble_spread_pressure'], d['NIS']))
        if k < n_assim - 1:
            enkf.forecast_step(substeps)
            # lift-like force, averaged over members, evaluated right after the
            # forecast step that lands on instant k+1 (same convention as
            # stage_c_free_run.py; off by one solver substep, negligible)
            Fy_mean_hist[k + 1] = np.mean([m.force_on_body()[1] for m in members])
    Fy_mean_hist[0] = Fy_mean_hist[1]

    u_all_final = np.stack([m.u for m in members])
    v_all_final = np.stack([m.v for m in members])

    out_path = args.out or os.path.join(HERE, 'stage_d_enkf_%s.npz' % args.variant)
    np.savez_compressed(
        out_path,
        exp_times=obs.tap_times - obs.tap_times[0],
        tap_times_true=obs.tap_times,
        u_mean_hist=u_mean_hist, v_mean_hist=v_mean_hist, Fy_mean_hist=Fy_mean_hist,
        u_all_final=u_all_final, v_all_final=v_all_final,
        tap_p_measured=obs.tap_p, tap_p_series_used=tap_p_series,
        tap_x=obs.tap_x, tap_y=obs.tap_y,
        innovation_norm=np.array([d['innovation_norm'] for d in diags]),
        kalman_correction_norm=np.array([d['kalman_correction_norm'] for d in diags]),
        pressure_rmse=np.array([d['pressure_rmse'] for d in diags]),
        ensemble_spread_state=np.array([d['ensemble_spread_state'] for d in diags]),
        ensemble_spread_pressure=np.array([d['ensemble_spread_pressure'] for d in diags]),
        NIS=np.array([d['NIS'] for d in diags]),
        ybar_f_hist=np.stack([d['ybar_f'] for d in diags]),
        y_measured_hist=np.stack([d['y_measured'] for d in diags]),
        gauges_hist=np.stack([d['gauges'] for d in diags]),
        ic_times=ic_times,
        config=json.dumps(c), args=json.dumps(vars(args)),
    )
    print('Wrote %s' % out_path)


if __name__ == '__main__':
    main()
