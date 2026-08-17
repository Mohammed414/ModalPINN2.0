"""
STAGE F2: withheld full-field evaluation of the REPAIRED (Stage D2) filter.

Separate script from stage_f_evaluate.py; writes stage_f2_evaluation.npz and
does not modify any Stage F output.  Truth is read only here, only through
allow_truth_access(), and only AFTER every observer run has finished and
been written to disk.

Scores, at the assimilation instants each run actually covers:
  free_run   -- Stage C, no assimilation
  enkf_d1    -- Stage D (original filter, gain fraction ~2e-4)
  enkf_d2    -- Stage D2 nominal (repaired)
  d2_shuffled / d2_scrambled -- Stage D2 negative controls

Note on the comparison window: Stage D2 spends its first 61 cycles in a
forecast-only bias-estimation window and reports diagnostics only from cycle
61 onward.  All runs are therefore compared on the COMMON window
[61, 200] using each run's own cycle indexing, so the D1-vs-D2 difference is
not an artefact of different time spans.
"""
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import estimator  # noqa: F401
from estimator._leakage_guard import allow_truth_access

from stage_f_evaluate import (grid_from_config, relative_l2_error_series,
                              phase_series, wrap)

EXP_DIR = os.path.join(ROOT, 'experiments')
FIG_DIR = os.path.join(ROOT, 'figures')
K0 = 61          # first cycle Stage D2 assimilates
NT = 201


def main():
    with allow_truth_access():
        truth = np.load(os.path.join(ROOT, 'data', 'reference_truth_full.npz'))
        ref_x, ref_y = truth['ref_x'], truth['ref_y']
        ref_cu, ref_cv = truth['ref_cu'], truth['ref_cv']

    runs = {}
    free = np.load(os.path.join(EXP_DIR, 'stage_c_free_run_control.npz'))
    runs['free_run'] = dict(u=free['u_hist'][K0:], v=free['v_hist'][K0:],
                            Fy=free['Fy_hist'][K0:],
                            config=json.loads(str(free['solver_config'])))

    d1 = np.load(os.path.join(EXP_DIR, 'stage_d_enkf_nominal.npz'))
    runs['enkf_d1'] = dict(u=d1['u_mean_hist'][K0:], v=d1['v_mean_hist'][K0:],
                           Fy=d1['Fy_mean_hist'][K0:],
                           config=json.loads(str(d1['config'])))

    for tag, fname in [('enkf_d2', 'stage_d2_enkf_nominal.npz'),
                       ('d2_shuffled', 'stage_d2_enkf_shuffled.npz'),
                       ('d2_scrambled', 'stage_d2_enkf_scrambled_sensors.npz')]:
        z = np.load(os.path.join(EXP_DIR, fname))
        assert int(z['cycle_index'][0]) == K0, 'unexpected D2 start cycle'
        runs[tag] = dict(u=z['u_mean_hist'], v=z['v_mean_hist'], Fy=z['Fy_mean_hist'],
                         config=json.loads(str(z['config'])))

    tap_p_measured = free['tap_p_measured']
    x_c, y_c, r_c = 0., 0., 0.5

    obs_taps = np.load(os.path.join(ROOT, 'data', 'tap_observations.npz'))
    tx, ty = obs_taps['tap_x_32'], obs_taps['tap_y_32']
    theta = np.arctan2(ty, tx)
    order = np.argsort(theta)
    ny_sorted = np.sin(theta[order])
    dtheta = 2 * np.pi / len(theta)
    Cl_proxy_true = -np.sum(tap_p_measured[:, order] * ny_sorted[None, :],
                            axis=1) * dtheta
    Cl_win = Cl_proxy_true[K0:]

    ref_cu_w, ref_cv_w = ref_cu[K0:], ref_cv[K0:]

    results = {}
    for name, run in runs.items():
        c = run['config']
        xc_, yc_ = grid_from_config(c)
        Eu, Ev = relative_l2_error_series(run['u'], run['v'], xc_, yc_,
                                          ref_cu_w, ref_cv_w, ref_x, ref_y,
                                          r_c, x_c, y_c)
        dphi = wrap(phase_series(run['Fy']) - phase_series(Cl_win))
        results[name] = dict(Eu=Eu, Ev=Ev, dphi=dphi)
        print('%-13s: E_u=%.4f  E_v=%.4f  mean|dphi|=%.4f rad  final|dphi|=%.4f'
              % (name, Eu.mean(), Ev.mean(), np.abs(dphi).mean(), abs(dphi[-1])))

    out = os.path.join(EXP_DIR, 'stage_f2_evaluation.npz')
    if os.path.exists(out):
        raise SystemExit('refusing to overwrite %s' % out)
    np.savez_compressed(out, window_start=K0, **{
        '%s_%s' % (n, k): v for n, r in results.items() for k, v in r.items()})
    print('Wrote %s' % out)

    print('\n' + '=' * 66)
    print('STAGE F2 VERDICT  (common window cycles %d-%d)' % (K0, NT - 1))
    print('=' * 66)
    base_u = results['free_run']['Eu'].mean()
    base_v = results['free_run']['Ev'].mean()
    for n in ['enkf_d1', 'enkf_d2', 'd2_shuffled', 'd2_scrambled']:
        du = 100 * (results[n]['Eu'].mean() - base_u) / base_u
        dv = 100 * (results[n]['Ev'].mean() - base_v) / base_v
        print('%-13s  E_u %+6.2f%%   E_v %+6.2f%%   vs free run' % (n, du, dv))
    print('=' * 66)


if __name__ == '__main__':
    main()
