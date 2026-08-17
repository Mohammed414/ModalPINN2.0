"""
STAGE D2 (audit repair): the repaired pressure-only EnKF.

Does NOT overwrite stage_d_enkf.py or any stage_d_*.npz -- this is a new
script writing new files (stage_d2_enkf_*.npz).

Four repairs relative to Stage D (each switchable, so before/after is
measurable rather than asserted):

  1. per-tap fixed bias vector b, estimated over a forecast-only window of
     one measured shedding period (61 cycles), then held fixed
     (--bias-mode per_tap | global | continuous | none)
  2. sigma_p calibrated from the UNSTEADY tap RMS (0.0472) rather than
     std(tap_p) (0.338, which is 98% static theta-variation of mean Cp).
     Reported via the gain fraction s^2/(s^2+sigma_p^2).
     (--sigma-mode unsteady | stage_d, or --sigma-p explicit)
  3. multi-direction ensemble: divergence-free streamfunction perturbations
     on top of a reduced phase jitter (--pert-amp, --jitter)
  4. gamma (time-dilation frequency parameter) augmented into the state
     (--augment-gamma / --no-augment-gamma, --gamma-spread)

Also fixes a rank defect found in Stage D's ensemble: BASE_IC_TIME=310.0
is the FIRST snapshot in spinup_snapshots.npz, so every negative jitter
clamped to index 0 and only 8 of 16 members were distinct (6 were
byte-identical). The base time is moved to the middle of the snapshot
window (316.0) so the jitter is two-sided.

No truth data is read anywhere in this script.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import estimator  # noqa: F401  (installs the leakage guard)
from estimator.ns_solver import CylinderFlowSolver
from estimator.data_interface import TapObservations
from estimator.enkf2 import EnKF2Run, decompose_innovation
from estimator.state_vector import StateVectorizer
from estimator import ensemble_init as ei

SEED = 0
Q_ENSEMBLE = 16
BASE_IC_TIME = 316.0        # centre of spinup_snapshots.npz (310..322)
JITTER_HALF_RANGE = 0.15    # reduced from Stage D's 0.4; see --jitter
PERT_AMPLITUDE = 0.04       # velocity-perturbation RMS in the wake core
PERT_LENGTH = 0.8           # correlation length of psi, in diameters
WAKE_BIAS = (2.5, 3.0, 1.6)  # (x0, sigma_x, sigma_y) of the energy envelope
BIAS_WINDOW = 61            # cycles = 6.1 t.u. ~ one measured shedding period
GAMMA_SPREAD = 0.07         # initial std of gamma about 1.0
SPIN_IN_STEPS = 10          # solver steps after perturbing, to relax p


def build_ensemble(spin, base_time, q, jitter_half_range, rng, c,
                   pert_amp, pert_length, wake_bias, spin_in, gammas=None):
    """q members: phase-jittered snapshots of the observer's own limit cycle,
    each plus an independent divergence-free streamfunction perturbation."""
    times_avail = spin['times']
    offsets = rng.uniform(-jitter_half_range, jitter_half_range, size=q)
    members, chosen_times = [], []
    div_before, div_after = [], []
    for j in range(q):
        idx = int(np.argmin(np.abs(times_avail - (base_time + offsets[j]))))
        dt_j = c['dt'] if gammas is None else gammas[j] * c['dt']
        solver = CylinderFlowSolver(Nx=c['Nx'], Ny=c['Ny'],
                                    Lxmin=c['Lxmin'], Lxmax=c['Lxmax'],
                                    Lymin=c['Lymin'], Lymax=c['Lymax'],
                                    x_c=c['x_c'], y_c=c['y_c'], r_c=c['r_c'],
                                    Re=c['Re'], dt=dt_j)
        solver.u = spin['u'][idx].copy()
        solver.v = spin['v'][idx].copy()
        solver.p = spin['p'][idx].copy()
        solver.t = 0.0
        div_before.append(ei.max_div_interior(solver))
        if pert_amp > 0:
            du, dv = ei.streamfunction_perturbation(
                solver, np.random.default_rng(10_000 + j),
                length_scale=pert_length, amplitude=pert_amp, wake_bias=wake_bias)
            solver.u = solver.u + du
            solver.v = solver.v + dv
        div_after.append(ei.max_div_interior(solver))
        for _ in range(spin_in):
            solver.step()
        members.append(solver)
        chosen_times.append(times_avail[idx])
    return members, np.array(chosen_times), np.array(div_before), np.array(div_after)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['nominal', 'shuffled', 'scrambled_sensors'],
                    default='nominal')
    ap.add_argument('--q', type=int, default=Q_ENSEMBLE)
    ap.add_argument('--inflation', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--bias-mode', default='per_tap',
                    choices=['per_tap', 'global', 'continuous', 'none'])
    ap.add_argument('--bias-window', type=int, default=BIAS_WINDOW)
    ap.add_argument('--sigma-mode', default='unsteady', choices=['unsteady', 'stage_d'])
    ap.add_argument('--sigma-p', type=float, default=None,
                    help='explicit sigma_p, overrides --sigma-mode')
    ap.add_argument('--jitter', type=float, default=JITTER_HALF_RANGE)
    ap.add_argument('--pert-amp', type=float, default=PERT_AMPLITUDE)
    ap.add_argument('--additive-amp', type=float, default=0.0,
                    help='additive model-error inflation amplitude per cycle '
                         '(divergence-free streamfunction perturbation)')
    ap.add_argument('--augment-gamma', dest='augment_gamma', action='store_true', default=True)
    ap.add_argument('--no-augment-gamma', dest='augment_gamma', action='store_false')
    ap.add_argument('--gamma-spread', type=float, default=GAMMA_SPREAD)
    ap.add_argument('--gamma-spread-floor', type=float, default=0.02)
    ap.add_argument('--gamma-clip', type=float, nargs=2, default=(0.7, 1.3))
    ap.add_argument('--base-time', type=float, default=BASE_IC_TIME)
    ap.add_argument('--tag', default=None, help='extra filename tag')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    t_start = time.time()
    spin = np.load(os.path.join(HERE, 'spinup_snapshots.npz'))
    c = json.loads(str(spin['solver_config']))
    obs = TapObservations(n_taps=32)
    rng = np.random.default_rng(args.seed)

    # ---- gamma initialisation: centred on 1.0 (the solver's OWN rate), NOT
    # on the truth-consistent 0.885. The filter has to find that from
    # pressure alone; seeding it there would be circular.
    if args.augment_gamma:
        gammas0 = 1.0 + args.gamma_spread * rng.standard_normal(args.q)
        gammas0 = np.clip(gammas0, *args.gamma_clip)
    else:
        gammas0 = None

    members, ic_times, div_before, div_after = build_ensemble(
        spin, args.base_time, args.q, args.jitter, rng, c,
        args.pert_amp, PERT_LENGTH, WAKE_BIAS, SPIN_IN_STEPS, gammas0)

    print('Ensemble ICs: %s (%d distinct of %d)'
          % (np.round(ic_times, 3), np.unique(np.round(ic_times, 6)).size, args.q))
    print('max|div| interior: pre-perturbation %.3e -> post-perturbation %.3e'
          % (div_before.max(), div_after.max()))
    if gammas0 is not None:
        print('gamma_0: mean %.4f std %.4f range [%.4f, %.4f]'
              % (gammas0.mean(), gammas0.std(ddof=1), gammas0.min(), gammas0.max()))

    _vec0 = StateVectorizer(members[0])
    spec0 = ei.anomaly_spectrum(np.stack([_vec0.flatten(m) for m in members], axis=1))
    print('initial ensemble n_eff = %.3f  (energy frac %s)'
          % (spec0['n_eff'], np.round(spec0['energy_fraction'][:5], 4)))

    # ---- observation series + negative controls -------------------------
    tap_p_series = obs.tap_p.copy()
    if args.variant == 'shuffled':
        perm = rng.permutation(len(obs.tap_times))
        tap_p_series = tap_p_series[perm]
        print('NEGATIVE CONTROL: pressure time order shuffled (seed=%d)' % args.seed)
    elif args.variant == 'scrambled_sensors':
        sensor_perm = rng.permutation(obs.n_taps)
        tap_p_series = tap_p_series[:, sensor_perm]
        print('NEGATIVE CONTROL: sensor identity scrambled (seed=%d)' % args.seed)

    # ---- sigma_p ---------------------------------------------------------
    unsteady_rms = float(np.sqrt(np.mean((obs.tap_p - obs.tap_p.mean(axis=0)) ** 2)))
    if args.sigma_p is not None:
        sigma_p = args.sigma_p
        sigma_src = 'explicit'
    elif args.sigma_mode == 'stage_d':
        sigma_p = 0.3 * float(np.std(obs.tap_p))
        sigma_src = '0.3*std(tap_p) [Stage D]'
    else:
        sigma_p = unsteady_rms
        sigma_src = 'RMS of unsteady tap signal'
    print('sigma_p = %.6f  (%s);  std(tap_p)=%.6f  unsteady_rms=%.6f'
          % (sigma_p, sigma_src, np.std(obs.tap_p), unsteady_rms))

    dt_assim = obs.tap_times[1] - obs.tap_times[0]
    substeps = int(round(dt_assim / c['dt']))
    n_assim = len(obs.tap_times)

    enkf = EnKF2Run(members, obs, obs_noise_std=sigma_p, inflation=args.inflation,
                    seed=args.seed, bias_mode=args.bias_mode,
                    augment_gamma=args.augment_gamma, gammas=gammas0,
                    dt_nom=c['dt'], gamma_clip=tuple(args.gamma_clip),
                    gamma_spread_floor=args.gamma_spread_floor,
                    additive_amp=args.additive_amp,
                    additive_length=PERT_LENGTH, additive_wake_bias=WAKE_BIAS)

    # ---- Phase 1: forecast-only bias window ------------------------------
    n_w = args.bias_window if args.bias_mode == 'per_tap' else 0
    bias_diag = {}
    if n_w > 0:
        bias_diag = enkf.estimate_bias(tap_p_series[:n_w], substeps, verbose=True)
        print('per-tap bias b: mean %.5f, std over taps %.5f, range [%.4f, %.4f]'
              % (bias_diag['bias_mean'], bias_diag['bias_std_over_taps'],
                 bias_diag['bias'].min(), bias_diag['bias'].max()))
        enkf.forecast_step(substeps)   # land on cycle n_w

    # ---- Phase 2: assimilation -------------------------------------------
    diags = []
    k0 = n_w
    u_mean_hist = np.empty((n_assim - k0,) + members[0].u.shape, dtype=np.float32)
    v_mean_hist = np.empty((n_assim - k0,) + members[0].v.shape, dtype=np.float32)
    Fy_mean_hist = np.full(n_assim - k0, np.nan)
    n_eff_hist = np.empty(n_assim - k0)

    for k in range(k0, n_assim):
        d = enkf.assimilate_step(tap_p_series[k])
        diags.append(d)
        i = k - k0
        u_mean_hist[i] = np.mean([m.u for m in members], axis=0)
        v_mean_hist[i] = np.mean([m.v for m in members], axis=0)
        if i % 10 == 0 or k == n_assim - 1:
            n_eff_hist[i] = enkf.anomaly_spectrum()['n_eff']
        else:
            n_eff_hist[i] = np.nan
        if i % 20 == 0 or k == n_assim - 1:
            g = d.get('gamma_mean', float('nan'))
            gs = d.get('gamma_std', float('nan'))
            print('k=%3d t=%.2f |innov|=%.4f gain=%.4f spread_p=%.5f NIS=%.2f '
                  'gamma=%.4f+-%.4f'
                  % (k, obs.tap_times[k] - obs.tap_times[0], d['innovation_norm'],
                     d['gain_fraction'], d['ensemble_spread_pressure'], d['NIS'], g, gs))
        if k < n_assim - 1:
            enkf.add_model_error()   # no-op when --additive-amp 0
            enkf.forecast_step(substeps)
            Fy_mean_hist[i + 1] = np.mean([m.force_on_body()[1] for m in members])
    if len(Fy_mean_hist) > 1:
        Fy_mean_hist[0] = Fy_mean_hist[1]

    innov_hist = np.stack([d['innovation'] for d in diags])
    decomp = decompose_innovation(innov_hist)
    print('\ninnovation decomposition over the assimilation window:')
    print('  total MS %.6f | static %.1f%% | time-varying %.1f%% | global const %.1f%%'
          % (decomp['total_ms'], 100 * decomp['static_fraction'],
             100 * decomp['varying_fraction'], 100 * decomp['global_constant_fraction']))

    gf = np.array([d['gain_fraction'] for d in diags])
    print('gain fraction: first %.4f median %.4f last %.4f  (Stage D was ~2.4e-4)'
          % (gf[0], np.median(gf), gf[-1]))
    if args.augment_gamma:
        gm = np.array([d['gamma_mean'] for d in diags])
        print('gamma: start %.4f -> end %.4f (min %.4f max %.4f); clip hits %d'
              % (gm[0], gm[-1], gm.min(), gm.max(), enkf.clip_hits))

    def col(key, default=np.nan):
        return np.array([d.get(key, default) for d in diags], dtype=float)

    tag = ('_' + args.tag) if args.tag else ''
    out_path = args.out or os.path.join(HERE, 'stage_d2_enkf_%s%s.npz' % (args.variant, tag))
    if os.path.exists(out_path):
        raise SystemExit('refusing to overwrite existing %s' % out_path)

    np.savez_compressed(
        out_path,
        cycle_index=np.arange(k0, n_assim),
        exp_times=obs.tap_times[k0:] - obs.tap_times[0],
        tap_times_true=obs.tap_times[k0:],
        u_mean_hist=u_mean_hist, v_mean_hist=v_mean_hist, Fy_mean_hist=Fy_mean_hist,
        u_all_final=np.stack([m.u for m in members]),
        v_all_final=np.stack([m.v for m in members]),
        tap_p_measured=obs.tap_p, tap_p_series_used=tap_p_series,
        tap_x=obs.tap_x, tap_y=obs.tap_y,
        innovation_hist=innov_hist,
        innovation_norm=col('innovation_norm'),
        kalman_correction_norm=col('kalman_correction_norm'),
        state_correction_norm=col('state_correction_norm'),
        pressure_rmse=col('pressure_rmse'),
        ensemble_spread_state=col('ensemble_spread_state'),
        ensemble_spread_pressure=col('ensemble_spread_pressure'),
        gain_fraction=gf,
        NIS=col('NIS'),
        n_eff_hist=n_eff_hist,
        gamma_mean=col('gamma_mean'), gamma_std=col('gamma_std'),
        gamma_update=col('gamma_update'),
        gamma_K_row_norm=col('gamma_K_row_norm'),
        gamma_obs_cov_norm=col('gamma_obs_cov_norm'),
        gamma_n_clipped=col('gamma_n_clipped'),
        gammas_hist=np.stack([d['gammas'] for d in diags]) if args.augment_gamma
        else np.zeros(0),
        gammas_init=gammas0 if gammas0 is not None else np.zeros(0),
        ybar_f_hist=np.stack([d['ybar_f'] for d in diags]),
        y_measured_hist=np.stack([d['y_measured'] for d in diags]),
        bias_vector=bias_diag.get('bias', np.zeros(0)),
        bias_resid_before=bias_diag.get('resid_rms_before', np.nan),
        bias_resid_after_pertap=bias_diag.get('resid_rms_after_pertap', np.nan),
        bias_resid_after_global=bias_diag.get('resid_rms_after_global', np.nan),
        innov_static_fraction=decomp['static_fraction'],
        innov_varying_fraction=decomp['varying_fraction'],
        innov_global_fraction=decomp['global_constant_fraction'],
        innov_static_per_tap=decomp['static_per_tap'],
        ensemble_neff_initial=spec0['n_eff'],
        ensemble_sv_initial=spec0['singular_values'],
        div_before=div_before, div_after=div_after,
        ic_times=ic_times, sigma_p=sigma_p, unsteady_rms=unsteady_rms,
        clip_hits=enkf.clip_hits, substeps=substeps, bias_window=n_w,
        config=json.dumps(c), args=json.dumps(vars(args)),
        wall_time_s=time.time() - t_start,
    )
    print('Wrote %s  (%.1f s)' % (out_path, time.time() - t_start))


if __name__ == '__main__':
    main()
