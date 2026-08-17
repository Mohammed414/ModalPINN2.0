"""
STAGE F2 (metrics_v2): the wake-amplitude evaluation of every run.

Scores five runs with evaluation/metrics_v2.py -- the metric family that was
validated (evaluation/validate_metrics_v2.py) to be un-gameable by deleting
the wake, unlike Stage F's E_v:

    free_run      stage_c_free_run_control.npz        no assimilation
    enkf_d1       stage_d_enkf_nominal.npz            ORIGINAL filter
    enkf_d2       stage_d2_enkf_nominal.npz           REPAIRED filter
    d2_shuffled   stage_d2_enkf_shuffled.npz          negative control
    d2_scrambled  stage_d2_enkf_scrambled_sensors.npz negative control

COMMON WINDOW. Stage D2 reports only cycles 61..200 (its first 61 cycles are
the forecast-only bias-estimation window).  Every run is therefore cut to
ABSOLUTE cycles 61..200 -- t = 406.1 .. 420.0, 140 samples, 13.9 time units.
free_run and enkf_d1 are additionally scored on their own full 0..200 window
as a supplementary row, clearly labelled, so the cost of the shorter window
is visible rather than hidden.

WINDOW LENGTH IS A FIRST-CLASS RESULT HERE, NOT A FOOTNOTE. The original
Stage F conclusion died because its ranking reversed at t ~ 10.5 inside a
20-t.u. window.  13.9 t.u. is 2.29 periods of omega_0 = 1.036.  Thirds are
therefore 0.76 of a period -- shorter than one oscillation -- so a
harmonic fit on a third is not identifiable and the thirds are reported
as DIAGNOSTIC ONLY.  Halves (1.15 periods) are the shortest sub-window on
which a k=1 fit is meaningful, and even there the phase-aligned metric has
only ~10 evaluation instants left after the +-T/2 inset.  The beat period
between a run shedding at omega_est and the truth's omega_0 is
2*pi/|omega_est - omega_0|; for the free run (omega_est ~ 1.17) that is
~47 time units, i.e. THREE TIMES the available window.  No run here covers
a full beat.  That is stated in the output, not worked around.

No number in this file is tuned; truth is read only through
allow_truth_access(), only inside metrics_v2, and only after every observer
run was written to disk.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import estimator  # noqa: F401  (installs the leakage guard)
import metrics_v2 as M

EXP_DIR = os.path.join(ROOT, 'experiments')
K0 = 61

RUNS = [
    ('free_run', 'stage_c_free_run_control.npz', 'u_hist', 'v_hist',
     'solver_config', True),
    ('enkf_d1', 'stage_d_enkf_nominal.npz', 'u_mean_hist', 'v_mean_hist',
     'config', True),
    ('enkf_d2', 'stage_d2_enkf_nominal.npz', 'u_mean_hist', 'v_mean_hist',
     'config', False),
    ('d2_shuffled', 'stage_d2_enkf_shuffled.npz', 'u_mean_hist', 'v_mean_hist',
     'config', False),
    ('d2_scrambled', 'stage_d2_enkf_scrambled_sensors.npz', 'u_mean_hist',
     'v_mean_hist', 'config', False),
]

SCALARS = ('amp_rel', 'cplx_rel_aligned', 'cplx_rel_raw', 'psi_opt',
           'prof_rel_err', 'peak_est', 'peak_true', 'peak_x_est', 'peak_x_true',
           'persistence_est', 'persistence_true', 'at_x7_est', 'at_x7_true',
           'amp_ratio_mean', 'amp_ratio_deficit', 'phase_err_mean',
           'phase_err_std')


def summarize(res):
    out = dict(label=res['label'], n_t=res['n_t'], t0=res['t0'], t1=res['t1'],
               omega_est_full=res['omega_est'], windows={})
    for wname, w in res['windows'].items():
        d = dict(t0=w['t0'], t1=w['t1'], omega_est=w['omega_est'],
                 T_window=w['T_window'], leakage_att_k1=w['leakage_att_k1'],
                 leakage_att_k2=w['leakage_att_k2'],
                 periods_of_omega0=w['T_window'] / (2 * np.pi / 1.036),
                 beat_period=float(2 * np.pi / abs(w['omega_est'] - 1.036))
                 if abs(w['omega_est'] - 1.036) > 1e-9 else float('inf'))
        for comp in ('modal_v', 'modal_u', 'modal_v_ownfreq'):
            m = w[comp]
            dd = dict(omega=m['omega'], mean_rel_k0=m[0]['mean_rel'])
            for k in (1, 2):
                for key in SCALARS:
                    dd['k%d_%s' % (k, key)] = float(m[k][key])
            d[comp] = dd
        for comp in ('aligned_v', 'aligned_u'):
            a = w[comp]
            d[comp] = None if a is None else dict(
                E_aligned=float(a['E_aligned']), tau_opt=float(a['tau_opt']),
                phase_lag=float(a['phase_lag']),
                E_unaligned=float(a['E_unaligned']),
                n_eval=int(len(a['t_eval'])))
        out['windows'][wname] = d
    return out


def main():
    truth = M.load_truth()
    T0 = 2 * np.pi / truth.omega_0
    mt_v = M.modal_metrics(truth.times, truth.V, truth, 'v')
    mt_u = M.modal_metrics(truth.times, truth.U, truth, 'u')
    print('truth |v1|: peak %.4f at x=%.2f ; x=7 %.4f ; persistence %.4f'
          % (mt_v[1]['peak_true'], mt_v[1]['peak_x_true'],
             mt_v[1]['at_x7_true'], mt_v[1]['persistence_true']))
    print('truth |v2|: peak %.4f at x=%.2f ; x=7 %.4f ; persistence %.4f'
          % (mt_v[2]['peak_true'], mt_v[2]['peak_x_true'],
             mt_v[2]['at_x7_true'], mt_v[2]['persistence_true']))

    summary = dict(
        window=dict(cycle_start=K0, cycle_end=200, T_window=13.9,
                    periods_of_omega0=13.9 / T0, omega_0=truth.omega_0,
                    period_omega0=T0),
        truth=dict(
            v1_peak=mt_v[1]['peak_true'], v1_peak_x=mt_v[1]['peak_x_true'],
            v1_at_x7=mt_v[1]['at_x7_true'],
            v1_persistence=mt_v[1]['persistence_true'],
            v2_peak=mt_v[2]['peak_true'], v2_peak_x=mt_v[2]['peak_x_true'],
            v2_persistence=mt_v[2]['persistence_true'],
            u1_peak=mt_u[1]['peak_true'], u1_persistence=mt_u[1]['persistence_true'],
            omega_0=truth.omega_0, n_fluid=int(truth.fluid.sum()),
            r_exclude=M.R_EXCLUDE),
        runs={}, supplementary={})
    arrays = dict(gx=truth.gx,
                  truth_v1_prof_max=mt_v[1]['prof_true_max'],
                  truth_v1_prof_rms=mt_v[1]['prof_true_rms'],
                  truth_v2_prof_max=mt_v[2]['prof_true_max'],
                  truth_u1_prof_max=mt_u[1]['prof_true_max'])

    for label, fname, uk, vk, ck, needs_cut in RUNS:
        d = np.load(os.path.join(EXP_DIR, fname), allow_pickle=True)
        cfg = json.loads(str(d[ck]))
        t_abs = d['tap_times_true'].astype(float)
        u, v = d[uk], d[vk]
        if needs_cut:
            u, v, t_abs = u[K0:], v[K0:], t_abs[K0:]
        else:
            assert int(d['cycle_index'][0]) == K0, 'unexpected D2 start cycle'
        assert len(t_abs) == 140, (label, len(t_abs))
        assert abs(t_abs[0] - 406.1) < 1e-9, (label, t_abs[0])

        print('\n=== %s  (%s)  t=%.1f..%.1f, %d samples ==='
              % (label, fname, t_abs[0], t_abs[-1], len(t_abs)))
        res = M.evaluate_run(truth, u, v, t_abs, cfg, label=label)
        summary['runs'][label] = summarize(res)

        full = res['windows']['full']
        for comp, mk in (('modal_v', 'v'), ('modal_u', 'u')):
            for k in (1, 2):
                m = full[comp][k]
                arrays['%s_%s%d_prof_max' % (label, mk, k)] = m['prof_est_max']
                arrays['%s_%s%d_amp_ratio_x' % (label, mk, k)] = m['amp_ratio_x']
                arrays['%s_%s%d_phase_err_x' % (label, mk, k)] = m['phase_err_x']
        m1o = full['modal_v_ownfreq'][1]
        arrays['%s_v1_prof_max_ownfreq' % label] = m1o['prof_est_max']
        arrays['%s_v1_amp_ratio_x_ownfreq' % label] = m1o['amp_ratio_x']
        a = full['aligned_v']
        arrays['%s_Et_aligned' % label] = a['E_t']
        arrays['%s_t_eval' % label] = a['t_eval']
        arrays['%s_tau_scan' % label] = a['curve']
        arrays['%s_tau_grid' % label] = a['taus']

        # halves, for the per-x sub-window stability of the KEY curve
        for wname in ('half_1', 'half_2'):
            arrays['%s_v1_amp_ratio_x_%s' % (label, wname)] = \
                res['windows'][wname]['modal_v'][1]['amp_ratio_x']
            arrays['%s_v1_prof_max_%s' % (label, wname)] = \
                res['windows'][wname]['modal_v'][1]['prof_est_max']

        m1 = full['modal_v'][1]
        print('  omega_est %.5f (%+.2f%% vs omega_0); beat period %.1f t.u. '
              'vs window %.1f' %
              (full['omega_est'], 100 * (full['omega_est'] / truth.omega_0 - 1),
               2 * np.pi / abs(full['omega_est'] - truth.omega_0), full['T_window']))
        print('  aligned E_v %.4f (unaligned %.4f)  tau* %+.4f t.u. (%+.3f rad), '
              'n_eval=%d' % (a['E_aligned'], a['E_unaligned'], a['tau_opt'],
                             a['phase_lag'], len(a['t_eval'])))
        print('  k=1 @omega_0: amp_rel %.4f  amp_ratio_mean %.4f  peak %.4f at '
              'x=%.2f  x7 %.4f  persistence %.4f'
              % (m1['amp_rel'], m1['amp_ratio_mean'], m1['peak_est'],
                 m1['peak_x_est'], m1['at_x7_est'], m1['persistence_est']))
        print('  k=1 @own w : amp_rel %.4f  amp_ratio_mean %.4f  peak %.4f'
              % (m1o['amp_rel'], m1o['amp_ratio_mean'], m1o['peak_est']))

    # ---- supplementary: free_run and enkf_d1 on their own full 20 t.u. -----
    for label, fname, uk, vk, ck, _ in RUNS[:2]:
        d = np.load(os.path.join(EXP_DIR, fname), allow_pickle=True)
        cfg = json.loads(str(d[ck]))
        res = M.evaluate_run(truth, d[uk], d[vk], d['tap_times_true'].astype(float),
                             cfg, label=label + '_full20', windows=False)
        summary['supplementary'][label + '_full20'] = summarize(res)

    # ---- per-x recovery relative to the free-run baseline ------------------
    # For each x, the k=1 amplitude-ratio ERROR |1 - |c(x)||; a run "recovers"
    # the oscillating mode at x if its error is below the free run's there.
    gx = truth.gx
    base = np.abs(1.0 - arrays['free_run_v1_amp_ratio_x'])
    recov = {}
    for label in ('enkf_d1', 'enkf_d2', 'd2_shuffled', 'd2_scrambled'):
        e = np.abs(1.0 - arrays['%s_v1_amp_ratio_x' % label])
        better = np.isfinite(e) & np.isfinite(base) & (e < base) & (gx >= 1.0)
        # also require it to hold in BOTH halves, so a downstream extent is
        # only claimed where it is not an artefact of where the window was cut
        eh1 = np.abs(1.0 - arrays['%s_v1_amp_ratio_x_half_1' % label])
        eh2 = np.abs(1.0 - arrays['%s_v1_amp_ratio_x_half_2' % label])
        bh1 = np.abs(1.0 - arrays['free_run_v1_amp_ratio_x_half_1'])
        bh2 = np.abs(1.0 - arrays['free_run_v1_amp_ratio_x_half_2'])
        stable = better & (eh1 < bh1) & (eh2 < bh2)
        xs_b = gx[better]; xs_s = gx[stable]
        recov[label] = dict(
            n_x_better=int(better.sum()),
            x_max_better=float(xs_b.max()) if better.any() else float('nan'),
            x_contig_max=float(_contig_max(gx, better)),
            n_x_stable=int(stable.sum()),
            x_max_stable=float(xs_s.max()) if stable.any() else float('nan'),
            x_contig_max_stable=float(_contig_max(gx, stable)),
            mean_amp_ratio_x_ge1=float(np.nanmean(
                arrays['%s_v1_amp_ratio_x' % label][gx >= 1.0])),
            mean_abs_err_x_ge1=float(np.nanmean(e[gx >= 1.0])))
        arrays['%s_v1_better_mask' % label] = better
        arrays['%s_v1_stable_mask' % label] = stable
    recov['free_run'] = dict(
        mean_amp_ratio_x_ge1=float(np.nanmean(
            arrays['free_run_v1_amp_ratio_x'][gx >= 1.0])),
        mean_abs_err_x_ge1=float(np.nanmean(base[gx >= 1.0])))
    summary['downstream_recovery_v1'] = recov

    # ---- window-sensitivity ranking table ----------------------------------
    wnames = ['full', 'half_1', 'half_2', 'third_1', 'third_2', 'third_3']
    metrics = [('E_v unaligned (OLD)', ('aligned_v', 'E_unaligned')),
               ('E_v phase-aligned', ('aligned_v', 'E_aligned')),
               ('|tau*| (t.u.)', ('aligned_v', 'tau_opt')),
               ('k1 amp_rel @w0', ('modal_v', 'k1_amp_rel')),
               ('k1 amp deficit @w0', ('modal_v', 'k1_amp_ratio_deficit')),
               ('k1 amp deficit @own', ('modal_v_ownfreq', 'k1_amp_ratio_deficit')),
               ('k1 |v1| profile err', ('modal_v', 'k1_prof_rel_err')),
               ('k2 amp deficit @w0', ('modal_v', 'k2_amp_ratio_deficit'))]
    print('\n' + '=' * 104)
    print('WINDOW SENSITIVITY  (thirds = 0.76 period: DIAGNOSTIC ONLY, a k=1 '
          'fit is not identifiable there)')
    print('=' * 104)
    stability = {}
    for name, path in metrics:
        print('\n%-24s %s' % (name, ''.join('%12s' % w for w in wnames)))
        best = {}
        for label in summary['runs']:
            row = []
            for w in wnames:
                blk = summary['runs'][label]['windows'][w][path[0]]
                v = None if blk is None else blk[path[1]]
                if v is not None and path[1] == 'tau_opt':
                    v = abs(v)
                row.append(v)
                if v is not None:
                    best.setdefault(w, []).append((v, label))
            print('  %-22s %s' % (label, ''.join(
                '%12s' % ('%.4f' % v if v is not None else '--') for v in row)))
        winners = {w: min(best[w])[1] for w in wnames if w in best}
        print('  %-22s %s' % ('WINNER', ''.join(
            '%12s' % winners.get(w, '--') for w in wnames)))
        meaningful = [w for w in ('full', 'half_1', 'half_2') if w in winners]
        stability[name] = dict(
            winners=winners,
            stable_over_halves=len(set(winners[w] for w in meaningful)) == 1,
            winner_full=winners.get('full'))
    summary['window_stability'] = stability
    print('=' * 104)

    jpath = os.path.join(EXP_DIR, 'stage_f2_metrics_v2.json')
    npath = os.path.join(EXP_DIR, 'stage_f2_metrics_v2.npz')
    for p in (jpath, npath):
        if os.path.exists(p):
            raise SystemExit('refusing to overwrite %s' % p)
    with open(jpath, 'w') as f:
        json.dump(summary, f, indent=1)
    np.savez_compressed(npath, **arrays)
    print('Wrote %s' % jpath)
    print('Wrote %s' % npath)


def _contig_max(gx, mask):
    """Largest x reached by the contiguous run of True that starts at or
    before x = 2 (the near-wake). Zero if no such run exists. This is the
    honest 'recovered out to x' number: an isolated island of improvement at
    x = 6 does not mean the wake was reconstructed out to 6."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return 0.0
    start = None
    for i in idx:
        if gx[i] <= 2.0:
            start = i
            break
    if start is None:
        return 0.0
    j = start
    while j + 1 < len(gx) and mask[j + 1]:
        j += 1
    return float(gx[j])


if __name__ == '__main__':
    main()
