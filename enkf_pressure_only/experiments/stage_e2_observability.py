"""
STAGE E2 (audit repair): the ensemble-space observability diagnostic, run
with the REPAIRED multi-direction ensemble and reported ALONGSIDE its null.

WHY STAGE E HAD TO BE REDONE
----------------------------
Stage E built 16 members by time-jittering ONE solver trajectory, stacked the
whitened wall-pressure anomalies, took the SVD and reported "only ~6 of 16
directions are visible in wall pressure".  The audit showed that conclusion
does not follow:

  (a) the ensemble was rank-deficient BY CONSTRUCTION (n_eff = 1.01: every
      member was the same field at a different phase, so there was only ever
      one real direction to see), and

  (b) everything past index ~6 sat at the double-precision floor
      (3.7e-15 relative, floor 3.6e-15) -- those singular values are round-off
      in the SVD, not physics.  A synthetic PERFECTLY OBSERVABLE system whose
      observation operator is the identity, sampled through the same ensemble
      construction, decays just as steeply.

So the Stage E spectrum measured the ENSEMBLE, not the SENSORS.

WHAT THIS SCRIPT MEASURES INSTEAD
---------------------------------
Three spectra on the same axes, plus a floor:

  1. PRESSURE (the thing under test).  Whitened wall-pressure anomaly matrix
     of the repaired ensemble (estimator/ensemble_init.py: divergence-free
     streamfunction perturbations + reduced phase jitter + gamma spread),
     stacked over L forecast instants and SVD'd.  Units: sigma_p.  A singular
     value of 1 means that ensemble direction produces exactly one noise
     standard deviation of pressure signal -- i.e. the detectability
     threshold is a MEANINGFUL ABSOLUTE LEVEL on this axis, not a relative
     one.

  2. STATE (what the construction guarantees).  The same ensemble's STATE
     anomaly spectrum.  This is the ceiling: a direction the ensemble does
     not span cannot be observed by any sensor, so the pressure spectrum can
     only ever be read relative to this one.

  3. SYNTHETIC NULL (perfect observability).  A system whose observation IS
     the identity on a rich multi-harmonic state, sampled with the same
     ensemble construction (q members, L instants, independent random
     multi-harmonic content per member).  This is the audit's control,
     rebuilt: it shows the decay that the sampling scheme produces even when
     nothing whatsoever is hidden from the sensors.

  4. RANDOM-PROJECTION NULL.  The SAME ensemble observed through 32 random
     linear functionals of the state instead of the 32 physical taps,
     normalised to the same total energy.  This isolates "is wall pressure
     worse than a generic 32-dimensional view of this flow?" from "does any
     32-row observation of a 16-member ensemble decay?".

  5. PER-DIRECTION VISIBILITY (the ordering-free answer).  Spectra are only
     loosely comparable because the i-th singular direction of the pressure
     map is not the i-th singular direction of the state map.  So for each
     STATE direction v_j (right singular vectors of the state anomaly
     matrix, i.e. the directions the ensemble actually spans, ordered by
     state energy) we compute

         vis_j = || Y v_j || / || X v_j ||_normalised

     the whitened pressure signal produced per unit of state anomaly along
     that direction.  vis_j > 1 means: perturbing the flow along direction j
     by its ensemble-typical amplitude moves the taps by more than the
     measurement noise.  That is the number the observability question
     actually asks for, and it does not depend on SVD ordering.

Everything is computed from the q x q Gram matrices G_x = sum_l X_l^T X_l and
G_y = sum_l Y_l^T Y_l, which is algebraically identical to the SVD of the
stacked matrices but never forms the (L*n_state, q) stack.

No truth data is read anywhere in this script.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import estimator  # noqa: F401  (installs the leakage guard)
from estimator.data_interface import TapObservations
from estimator.state_vector import StateVectorizer
from estimator import ensemble_init as ei

import stage_d2_enkf_repaired as sd2

Q = 16
L_WINDOWS = 60          # 6.0 t.u. at dt_assim = 0.1 ~ one measured shedding period
SIGMA_P = 0.0472        # Stage D2 calibration: RMS of the unsteady tap signal
SEED = 0


def spectrum_from_gram(G):
    """Singular values of a stacked anomaly matrix from its Gram matrix,
    plus the participation ratio. Eigenvalues clipped at 0 (round-off can
    make the smallest slightly negative)."""
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1]
    w = np.clip(w[order], 0.0, None)
    V = V[:, order]
    s = np.sqrt(w)
    tot = w.sum()
    n_eff = float(tot ** 2 / np.sum(w ** 2)) if tot > 0 else 0.0
    return s, V, n_eff


def numerical_floor(s, n_rows):
    """Level below which a singular value of a stacked matrix is round-off.

    A backward-stable SVD perturbs singular values by O(eps * ||A||_2), and
    forming the Gram matrix (as we do) squares the condition number, so the
    floor on s is sqrt(eps) * s[0] rather than eps * s[0]. Both are
    reported; the Gram floor is the operative one here.
    """
    eps = np.finfo(float).eps
    return dict(svd_floor=float(eps * s[0] * np.sqrt(n_rows)),
                gram_floor=float(np.sqrt(eps) * s[0]),
                eps=float(eps))


def synthetic_null(q, L, n_state=400, n_harm=6, omega=1.036, dt=0.1, seed=1):
    """A PERFECTLY OBSERVABLE system sampled the same way.

    Each member carries independent random amplitudes and phases on n_harm
    harmonics over n_state spatial basis functions; the observation operator
    is the IDENTITY (every state DOF measured, no noise).  Anomalies are
    collected at the same L instants and stacked exactly as for the real
    ensemble.  Any decay in this spectrum is produced by the ensemble
    sampling and the temporal stacking alone -- nothing is hidden.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((q, n_harm, n_state))
    P = rng.uniform(0, 2 * np.pi, (q, n_harm, n_state))
    G = np.zeros((q, q))
    for l in range(L):
        t = l * dt
        ph = np.arange(1, n_harm + 1)[None, :, None] * omega * t + P
        X = np.einsum('mhn,mhn->mn', A, np.cos(ph))      # (q, n_state)
        Xa = (X - X.mean(axis=0, keepdims=True)) / np.sqrt(q - 1)
        G += Xa @ Xa.T
    return G


def main():
    t_start = time.time()
    spin = np.load(os.path.join(HERE, 'spinup_snapshots.npz'))
    c = json.loads(str(spin['solver_config']))
    obs = TapObservations(n_taps=32)
    rng = np.random.default_rng(SEED)

    # ---- repaired ensemble, EXACTLY the Stage D2 construction --------------
    gammas0 = np.clip(1.0 + sd2.GAMMA_SPREAD * rng.standard_normal(Q), 0.7, 1.3)
    members, ic_times, div_b, div_a = sd2.build_ensemble(
        spin, sd2.BASE_IC_TIME, Q, sd2.JITTER_HALF_RANGE, rng, c,
        sd2.PERT_AMPLITUDE, sd2.PERT_LENGTH, sd2.WAKE_BIAS,
        sd2.SPIN_IN_STEPS, gammas0)
    vec = StateVectorizer(members[0])
    print('repaired ensemble: %d distinct ICs, max|div| %.2e -> %.2e, n_state=%d'
          % (np.unique(np.round(ic_times, 6)).size, div_b.max(), div_a.max(),
             vec.n_state))

    # ---- the SAME construction with phase jitter ONLY (Stage E's ensemble),
    # rebuilt here so the two spectra come from the same code path -----------
    rng_j = np.random.default_rng(SEED)
    members_j, ic_times_j, _, _ = sd2.build_ensemble(
        spin, 310.0, Q, 0.4, rng_j, c, 0.0, sd2.PERT_LENGTH, sd2.WAKE_BIAS,
        sd2.SPIN_IN_STEPS, None)
    print('phase-jitter-only ensemble (Stage E recipe): %d distinct ICs'
          % np.unique(np.round(ic_times_j, 6)).size)

    dt_assim = float(obs.tap_times[1] - obs.tap_times[0])
    substeps = int(round(dt_assim / c['dt']))

    # 32 random linear functionals of the state, fixed across the run
    rng_r = np.random.default_rng(12345)
    Hrand = rng_r.standard_normal((32, vec.n_state)) / np.sqrt(vec.n_state)

    def collect(mem, with_rand):
        """Accumulate Gram matrices over L instants of free forecast."""
        Gx = np.zeros((Q, Q)); Gy = np.zeros((Q, Q)); Gr = np.zeros((Q, Q))
        Xs = None
        state_rms = []
        for k in range(L_WINDOWS):
            X = np.stack([vec.flatten(m) for m in mem], axis=1)
            Y = np.stack([m.sample_pressure(obs.tap_x, obs.tap_y) for m in mem], axis=1)
            Xa = (X - X.mean(axis=1, keepdims=True)) / np.sqrt(Q - 1)
            Ya = (Y - Y.mean(axis=1, keepdims=True)) / np.sqrt(Q - 1) / SIGMA_P
            Gx += Xa.T @ Xa
            Gy += Ya.T @ Ya
            if with_rand:
                Ra = Hrand @ Xa
                Gr += Ra.T @ Ra
            state_rms.append(np.sqrt(np.mean(Xa ** 2)))
            if k == 0:
                Xs = Xa.copy()
            for m in mem:
                for _ in range(substeps):
                    m.step()
        return Gx, Gy, Gr, Xs, np.array(state_rms)

    Gx, Gy, Gr, Xs0, srms = collect(members, True)
    print('repaired forward pass done (%.0f s)' % (time.time() - t_start))
    Gxj, Gyj, _, _, _ = collect(members_j, False)
    print('phase-jitter forward pass done (%.0f s)' % (time.time() - t_start))

    s_p, Vp, neff_p = spectrum_from_gram(Gy)            # pressure, repaired
    s_x, Vx, neff_x = spectrum_from_gram(Gx)            # state, repaired
    s_r, _, neff_r = spectrum_from_gram(Gr)             # random-projection null
    s_pj, _, neff_pj = spectrum_from_gram(Gyj)          # pressure, jitter-only
    s_xj, _, neff_xj = spectrum_from_gram(Gxj)          # state, jitter-only
    Gsyn = synthetic_null(Q, L_WINDOWS, omega=float(obs.omega_0), dt=dt_assim)
    s_syn, _, neff_syn = spectrum_from_gram(Gsyn)       # perfectly observable

    # ---- per-state-direction visibility (ordering-free) -------------------
    # For each state direction v_j: pressure signal per unit state anomaly,
    # normalised so that the ensemble-typical state amplitude along direction
    # j maps to the pressure it actually produces, in units of sigma_p.
    vis = np.full(Q, np.nan)
    for j in range(Q):
        v = Vx[:, j]
        xs = float(v @ Gx @ v)
        ys = float(v @ Gy @ v)
        if xs > 0:
            vis[j] = np.sqrt(ys)          # already in sigma_p units; v is unit-norm
    # the state energy carried by each direction, for context
    state_energy = np.array([float(Vx[:, j] @ Gx @ Vx[:, j]) for j in range(Q)])

    floor_p = numerical_floor(s_p, L_WINDOWS * 32)
    floor_x = numerical_floor(s_x, L_WINDOWS * vec.n_state)
    floor_syn = numerical_floor(s_syn, L_WINDOWS * 400)

    stage_e = np.load(os.path.join(HERE, 'stage_e_observability.npz'))
    s_e_orig = stage_e['singular_values']

    def n_above(s, thr):
        return int(np.sum(s > thr))

    rel_p = s_p / s_p[0]
    rel_syn = s_syn / s_syn[0]
    rel_e = s_e_orig / s_e_orig[0]
    rel_x = s_x / s_x[0]

    report = dict(
        L_windows=L_WINDOWS, q=Q, sigma_p=SIGMA_P, n_state=int(vec.n_state),
        n_eff=dict(pressure_repaired=neff_p, state_repaired=neff_x,
                   randproj=neff_r, pressure_jitter_only=neff_pj,
                   state_jitter_only=neff_xj, synthetic_null=neff_syn),
        # the operative count: directions producing above-noise pressure signal
        n_directions_pressure_above_1sigma=n_above(s_p, 1.0),
        n_directions_visibility_above_1=int(np.sum(vis > 1.0)),
        n_directions_pressure_above_gramfloor=n_above(s_p, floor_p['gram_floor']),
        n_directions_state_above_gramfloor=n_above(s_x, floor_x['gram_floor']),
        n_directions_synthetic_above_gramfloor=n_above(s_syn, floor_syn['gram_floor']),
        n_directions_stageE_above_relfloor=int(np.sum(rel_e > 1e-8)),
        floors=dict(pressure=floor_p, state=floor_x, synthetic=floor_syn),
    )
    print('=' * 72)
    print('STAGE E2 OBSERVABILITY  (L=%d instants, q=%d, sigma_p=%.4f)'
          % (L_WINDOWS, Q, SIGMA_P))
    print('=' * 72)
    print('pressure  s (sigma_p units): %s'
          % np.array2string(s_p, precision=3, suppress_small=False))
    print('state     s (relative)     : %s' % np.array2string(rel_x, precision=3))
    print('synth null (relative)      : %s' % np.array2string(rel_syn, precision=3))
    print('Stage E orig (relative)    : %s' % np.array2string(rel_e, precision=3))
    print('visibility per state dir   : %s' % np.array2string(vis, precision=3))
    print('-' * 72)
    print(json.dumps({k: v for k, v in report.items() if k != 'floors'}, indent=1))
    print('floors: %s' % json.dumps(report['floors'], indent=1))

    out = os.path.join(HERE, 'stage_e2_observability.npz')
    if os.path.exists(out):
        raise SystemExit('refusing to overwrite %s' % out)
    np.savez_compressed(
        out,
        s_pressure=s_p, s_state=s_x, s_randproj=s_r, s_synthetic=s_syn,
        s_pressure_jitter_only=s_pj, s_state_jitter_only=s_xj,
        s_stage_e_original=s_e_orig,
        visibility=vis, state_energy=state_energy,
        Gx=Gx, Gy=Gy, Gr=Gr, G_synthetic=Gsyn,
        ic_times=ic_times, ic_times_jitter=ic_times_j, gammas0=gammas0,
        state_rms=srms, sigma_p=SIGMA_P, L_windows=L_WINDOWS, q=Q,
        report=json.dumps(report))
    print('Wrote %s  (%.0f s)' % (out, time.time() - t_start))


if __name__ == '__main__':
    main()
