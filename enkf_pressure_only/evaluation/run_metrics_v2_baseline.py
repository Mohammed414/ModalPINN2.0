"""
Apply metrics_v2 to the three EXISTING (pre-repair) runs, so that the
repaired EnKF runs produced in the next phase have a before-repair baseline
to be compared against.

Runs evaluated:
    stage_c_free_run_control.npz   (u_hist, v_hist)         -- no assimilation
    stage_d_enkf_nominal.npz       (u_mean_hist, v_mean_hist) -- EnKF
    stage_d_enkf_shuffled.npz      (u_mean_hist, v_mean_hist) -- negative control

Every metric is reported on the full 20-time-unit window AND on halves and
thirds, because the original Stage F conclusion was invalidated by exactly
this sensitivity: its ranking of the three runs reversed depending on where
the window was cut.

Writes experiments/metrics_v2_baseline.json (summary, machine-readable) and
experiments/metrics_v2_baseline.npz (the per-x profiles). Neither overwrites
an existing file.
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import estimator  # installs the leakage guard
import metrics_v2 as M

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(HERE, '..', 'experiments')

RUNS = [
    ('free_run', 'stage_c_free_run_control.npz', 'u_hist', 'v_hist', 'solver_config'),
    ('enkf', 'stage_d_enkf_nominal.npz', 'u_mean_hist', 'v_mean_hist', 'config'),
    ('shuffled', 'stage_d_enkf_shuffled.npz', 'u_mean_hist', 'v_mean_hist', 'config'),
]


def summarize(res):
    """JSON-safe summary: scalars only, profiles go to the npz."""
    out = dict(label=res['label'], n_t=res['n_t'], t0=res['t0'], t1=res['t1'],
               omega_est_full=res['omega_est'], windows={})
    for wname, w in res['windows'].items():
        d = dict(t0=w['t0'], t1=w['t1'], omega_est=w['omega_est'],
                 T_window=w['T_window'], leakage_att_k1=w['leakage_att_k1'],
                 leakage_att_k2=w['leakage_att_k2'])
        for comp in ('modal_v', 'modal_u', 'modal_v_ownfreq'):
            m = w[comp]
            dd = dict(omega=m['omega'], mean_rel_k0=m[0]['mean_rel'])
            for k in (1, 2):
                for key in ('amp_rel', 'cplx_rel_aligned', 'cplx_rel_raw', 'psi_opt',
                            'prof_rel_err', 'peak_est', 'peak_true', 'peak_x_est',
                            'peak_x_true', 'persistence_est', 'persistence_true',
                            'at_x7_est', 'at_x7_true', 'amp_ratio_mean',
                            'amp_ratio_deficit', 'phase_err_mean', 'phase_err_std'):
                    dd['k%d_%s' % (k, key)] = float(m[k][key])
            d[comp] = dd
        for comp in ('aligned_v', 'aligned_u'):
            a = w[comp]
            d[comp] = None if a is None else dict(
                E_aligned=a['E_aligned'], tau_opt=a['tau_opt'],
                phase_lag=a['phase_lag'], E_unaligned=a['E_unaligned'])
        out['windows'][wname] = d
    return out


def main():
    truth = M.load_truth()
    mt = M.modal_metrics(truth.times, truth.V, truth, 'v')[1]
    print('truth |v1|: peak %.4f at x=%.2f, x=7 value %.4f, persistence %.4f'
          % (mt['peak_true'], mt['peak_x_true'], mt['at_x7_true'], mt['persistence_true']))

    summary = dict(truth=dict(v1_peak=mt['peak_true'], v1_peak_x=mt['peak_x_true'],
                              v1_at_x7=mt['at_x7_true'],
                              v1_persistence=mt['persistence_true'],
                              omega_0=truth.omega_0,
                              n_fluid_points=int(truth.fluid.sum()),
                              r_exclude=M.R_EXCLUDE),
                   runs={})
    arrays = dict(gx=truth.gx,
                  truth_v1_prof_max=mt['prof_true_max'],
                  truth_v1_prof_rms=mt['prof_true_rms'])

    for label, fname, uk, vk, ck in RUNS:
        d = np.load(os.path.join(EXP_DIR, fname))
        cfg = json.loads(str(d[ck]))
        t_abs = d['tap_times_true'].astype(float)
        print('\n=== %s (%s) ===' % (label, fname))
        res = M.evaluate_run(truth, d[uk], d[vk], t_abs, cfg, label=label)
        summary['runs'][label] = summarize(res)

        full = res['windows']['full']
        m1 = full['modal_v'][1]
        arrays['%s_v1_prof_max' % label] = m1['prof_est_max']
        arrays['%s_v1_prof_rms' % label] = m1['prof_est_rms']
        arrays['%s_v1_amp_ratio_x' % label] = m1['amp_ratio_x']
        arrays['%s_v1_phase_err_x' % label] = m1['phase_err_x']
        m1o = full['modal_v_ownfreq'][1]
        arrays['%s_v1_prof_max_ownfreq' % label] = m1o['prof_est_max']
        arrays['%s_Et_aligned' % label] = full['aligned_v']['E_t']
        arrays['%s_t_eval' % label] = full['aligned_v']['t_eval']
        arrays['%s_tau_scan' % label] = full['aligned_v']['curve']
        arrays['%s_tau_grid' % label] = full['aligned_v']['taus']

        a = full['aligned_v']
        print('  omega_est          = %.5f  (omega_0 = %.4f, %+.2f%%)'
              % (full['omega_est'], truth.omega_0,
                 100 * (full['omega_est'] / truth.omega_0 - 1)))
        print('  aligned E_v        = %.4f   (unaligned %.4f)   tau* = %+.4f  (%+.3f rad)'
              % (a['E_aligned'], a['E_unaligned'], a['tau_opt'], a['phase_lag']))
        print('  modal k=1 @omega_0 : amp_rel %.4f  cplx_aligned %.4f  psi %+.3f rad'
              % (m1['amp_rel'], m1['cplx_rel_aligned'], m1['psi_opt']))
        print('  |v1| profile       : peak %.4f at x=%.2f (truth %.4f at x=%.2f)'
              % (m1['peak_est'], m1['peak_x_est'], m1['peak_true'], m1['peak_x_true']))
        print('  persistence x=7    : %.4f  (truth %.4f)'
              % (m1['persistence_est'], m1['persistence_true']))
        print('  modal k=1 @own w   : amp_rel %.4f  peak %.4f  persistence %.4f'
              % (m1o['amp_rel'], m1o['peak_est'], m1o['persistence_est']))

    # ---- window-sensitivity table --------------------------------------
    print('\n' + '=' * 96)
    print('WINDOW SENSITIVITY -- does the ranking survive re-cutting the window?')
    print('=' * 96)
    wnames = ['full', 'half_1', 'half_2', 'third_1', 'third_2', 'third_3']
    for metric, path in [('E_v unaligned (OLD)', ('aligned_v', 'E_unaligned')),
                         ('E_v phase-aligned', ('aligned_v', 'E_aligned')),
                         ('modal k1 amp_rel', ('modal_v', 'k1_amp_rel')),
                         ('|v1|(x) profile err', ('modal_v', 'k1_prof_rel_err')),
                         ('k1 amp deficit @w0', ('modal_v', 'k1_amp_ratio_deficit')),
                         ('k1 amp deficit @own', ('modal_v_ownfreq', 'k1_amp_ratio_deficit'))]:
        print('\n%-22s %s' % (metric, ''.join('%12s' % w for w in wnames)))
        best = {}
        for label in summary['runs']:
            row = []
            for w in wnames:
                blk = summary['runs'][label]['windows'][w][path[0]]
                v = None if blk is None else blk[path[1]]
                row.append(v)
                if v is not None:
                    best.setdefault(w, []).append((v, label))
            print('  %-20s %s' % (label, ''.join(
                '%12s' % ('%.4f' % v if v is not None else '--') for v in row)))
        print('  %-20s %s' % ('WINNER', ''.join(
            '%12s' % (min(best[w])[1] if w in best else '--') for w in wnames)))
    print('=' * 96)

    def fresh(p):
        if not os.path.exists(p):
            return p
        base, ext = os.path.splitext(p)
        n = 2
        while os.path.exists('%s_v%d%s' % (base, n, ext)):
            n += 1
        return '%s_v%d%s' % (base, n, ext)

    jpath = fresh(os.path.join(EXP_DIR, 'metrics_v2_baseline.json'))
    npath = fresh(os.path.join(EXP_DIR, 'metrics_v2_baseline.npz'))
    with open(jpath, 'w') as f:
        json.dump(summary, f, indent=1)
    np.savez_compressed(npath, **arrays)
    print('Wrote %s' % jpath)
    print('Wrote %s' % npath)


if __name__ == '__main__':
    main()
