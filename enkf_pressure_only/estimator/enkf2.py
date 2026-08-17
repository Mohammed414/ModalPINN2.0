"""
STAGE D2 (audit repair): pressure-only EnKF with the four Phase-2 fixes.

The Stage D filter (estimator/enkf.py) ran with a Kalman gain fraction of
~2e-4: the analysis was numerically indistinguishable from a free run, so
"pressure-only reconstruction failed" was not actually a statement about
observability, it was a statement about a mis-specified filter.  Four
changes, each independently switchable so the old behaviour stays reachable:

1.  PER-TAP BIAS REMOVAL (``bias_mode``).
    enkf.py removed ONE scalar per member, c = mean_s(p_meas - p_pred),
    because the observer's pressure Poisson solve is only defined up to a
    constant.  Phase 1 showed that a global constant accounts for only ~2%
    of innovation variance after the wall-probe sensor fix, while ~78-83%
    is a THETA-DEPENDENT static bias that no single scalar can absorb --
    it comes from the immersed-boundary momentum leak, which over-predicts
    the surface pressure amplitude by ~20% in a way that varies around the
    cylinder.  So we estimate a per-tap offset vector b (length n_taps)
    over a forecast-only spin-up window and hold it FIXED thereafter:

        b = mean_over_window[ p_measured(t) - p_predicted_ensmean(t) ]
        innovation(t) = p_measured(t) - ( p_predicted(t) + b )

    b is estimated from tap data and the model's own output only -- no
    truth field is touched.  It is held fixed on purpose: re-estimating it
    every cycle would drive the innovation to (almost) zero by
    construction, absorbing exactly the signal the filter is supposed to
    assimilate.  ``bias_mode='continuous'`` implements that wrong variant
    so the claim can be measured rather than asserted.

    Modes: 'per_tap' (default), 'global' (Stage D behaviour), 'continuous'
    (diagnostic straw man), 'none'.

2.  sigma_p FROM THE UNSTEADY SIGNAL (set by the caller).
    Stage D used sigma_p = 0.3*std(tap_p) = 0.102.  std(tap_p) is
    dominated by the STATIC theta-variation of mean Cp (0.335 of the 0.338
    total), which after fix 1 is not part of the innovation at all.  The
    quantity the filter must actually weigh its ensemble against is the
    unsteady tap RMS, 0.0472.  The primary diagnostic reported here is the
    GAIN FRACTION

        g = s^2 / (s^2 + sigma_p^2),   s = ensemble tap-pressure spread

    i.e. the scalar-analogue weight the filter places on the measurement.

3.  MULTI-DIRECTION ENSEMBLE (built by the caller, see ensemble_init.py).

4.  FREQUENCY AUGMENTATION (``augment_gamma``).
    Member j steps with dt' = gamma_j * dt_nom while the observer clock
    advances dt_nom, which Phase 1 validated as a near-exact scalar handle
    on shedding frequency (omega_eff = gamma * omega_s to 0.044% over
    gamma in [0.85, 1.05]).  gamma is appended to the state vector so the
    same K(innovation) machinery corrects it.

    Note on "is gamma swamped by 18844 velocity entries": no -- K is
    computed row-wise, so gamma's update depends only on
    cov(gamma, tap pressure) and not on the state dimension.  What CAN go
    wrong is (a) that covariance being pure sampling noise, and (b) the
    ensemble gamma-spread collapsing after a few updates so the parameter
    freezes.  (b) is guarded by ``gamma_spread_floor`` (relaxation-to-prior
    inflation applied to the gamma anomalies only).  Both are reported.
"""
import numpy as np

from .state_vector import StateVectorizer


def gauge_correct(p_predicted, p_measured):
    """Stage D's global scalar gauge: c = mean_s[p_meas - p_pred].
    Kept for backward compatibility / the 'global' bias mode."""
    c = np.mean(p_measured - p_predicted)
    return p_predicted + c, c


class EnKF2Run:
    """Perturbed-observation EnKF with per-tap bias, unsteady-calibrated R,
    and optional gamma (frequency) state augmentation.

    members       : list of q CylinderFlowSolver instances
    taps          : estimator.data_interface.TapObservations
    obs_noise_std : sigma_p, R = sigma_p^2 I
    bias_mode     : 'per_tap' | 'global' | 'continuous' | 'none'
    augment_gamma : bool; if True, gammas must be supplied
    gammas        : (q,) initial per-member gamma
    dt_nom        : nominal solver dt (member j runs with gamma_j * dt_nom)
    gamma_clip    : (lo, hi) hard clip applied after every analysis
    gamma_spread_floor : if the analysis gamma ensemble std falls below
                    this, re-inflate the gamma anomalies back up to it
                    (prevents parameter-estimation collapse). None = off.
    """

    def __init__(self, members, taps, obs_noise_std, inflation=1.0, seed=0,
                 bias_mode='per_tap', augment_gamma=False, gammas=None,
                 dt_nom=None, gamma_clip=(0.7, 1.3), gamma_spread_floor=0.02,
                 additive_amp=0.0, additive_length=0.8,
                 additive_wake_bias=(2.5, 3.0, 1.6)):
        self.members = members
        self.q = len(members)
        self.taps = taps
        self.sigma_p = float(obs_noise_std)
        self.alpha = inflation
        self.rng = np.random.default_rng(seed)
        self.vec = StateVectorizer(members[0])
        self.n_taps = taps.n_taps
        if bias_mode not in ('per_tap', 'global', 'continuous', 'none'):
            raise ValueError('bad bias_mode %r' % bias_mode)
        self.bias_mode = bias_mode

        # per-tap bias vector, set by estimate_bias(); None until then
        self.bias = None
        self.bias_window_n = 0

        self.augment_gamma = augment_gamma
        self.gamma_clip = gamma_clip
        self.gamma_spread_floor = gamma_spread_floor
        if augment_gamma:
            if gammas is None or dt_nom is None:
                raise ValueError('augment_gamma requires gammas and dt_nom')
            self.gammas = np.asarray(gammas, dtype=float).copy()
            if self.gammas.size != self.q:
                raise ValueError('need one gamma per member')
            self.dt_nom = float(dt_nom)
            self._apply_gammas()
        else:
            self.gammas = None
            self.dt_nom = float(dt_nom) if dt_nom is not None else float(members[0].dt)
        self.clip_hits = 0

        # additive model-error inflation (see add_model_error())
        self.additive_amp = float(additive_amp)
        self.additive_length = float(additive_length)
        self.additive_wake_bias = additive_wake_bias
        self._pert_counter = 0

    # ------------------------------------------------------------------
    def add_model_error(self):
        """ADDITIVE model-error inflation: give every member an independent,
        freshly-drawn divergence-free streamfunction perturbation after the
        analysis.

        Why this and not multiplicative inflation.  All members live on the
        SAME attractor -- the observer solver's own limit cycle -- so the
        forecast dynamics are contracting transverse to it: an anomaly the
        analysis leaves behind decays over the next few cycles and nothing
        regenerates it.  Multiplicative inflation can only rescale anomalies
        that still exist, so once the spread has collapsed by a factor ~20
        (measured: tap spread 0.036 -> 0.002 in ~20 cycles at alpha=1.0, and
        alpha=1.10 only moves the median gain fraction 0.0039 -> 0.0047) it
        has almost nothing left to amplify.  Additive perturbations inject
        new directions unconditionally.

        It is also the physically honest representation of what we know: the
        forward model has a CHARACTERISED error (Phase 1 -- the binary-mask
        IBM leaks momentum, over-predicting surface pressure amplitude by
        ~20%, Cd 1.55-1.61 vs 1.32-1.36 in the literature).  A filter that
        assumes a perfect model when the model is known to be biased is
        precisely how the gain fraction ends up at 1e-4.

        Divergence-freeness is preserved exactly (the perturbation is a
        discrete curl -- see ensemble_init.streamfunction_perturbation).
        """
        if self.additive_amp <= 0:
            return 0.0
        from .ensemble_init import streamfunction_perturbation
        max_div = 0.0
        for m in self.members:
            self._pert_counter += 1
            du, dv = streamfunction_perturbation(
                m, np.random.default_rng(self.rng.integers(0, 2 ** 31)),
                length_scale=self.additive_length, amplitude=self.additive_amp,
                wake_bias=self.additive_wake_bias)
            m.u = m.u + du
            m.v = m.v + dv
        return max_div

    # ------------------------------------------------------------------
    def _apply_gammas(self):
        """Member j's solver steps with dt' = gamma_j * dt_nom."""
        for j, m in enumerate(self.members):
            m.dt = self.gammas[j] * self.dt_nom

    def _predict_pressure(self, solver):
        return solver.sample_pressure(self.taps.tap_x, self.taps.tap_y)

    def predicted_ensemble_pressure(self):
        """(n_taps, q) raw predicted tap pressure, no bias correction."""
        return np.stack([self._predict_pressure(m) for m in self.members], axis=1)

    # ------------------------------------------------------------------
    def _debias(self, Yf_raw, p_measured):
        """Apply the configured bias/gauge correction to every member's
        prediction. Returns (Yf, offsets) where offsets is (n_taps, q) --
        the quantity added to each member's raw prediction."""
        q = self.q
        if self.bias_mode == 'none':
            return Yf_raw.copy(), np.zeros_like(Yf_raw)

        if self.bias_mode == 'global':
            off = np.empty_like(Yf_raw)
            for j in range(q):
                off[:, j] = np.mean(p_measured - Yf_raw[:, j])
            return Yf_raw + off, off

        if self.bias_mode == 'continuous':
            # WRONG-BY-DESIGN straw man: re-estimate the full per-tap offset
            # from THIS cycle's own measurement. The ensemble-mean innovation
            # is then identically zero, so the filter has nothing to
            # assimilate -- this variant exists to measure that, not to use.
            b_now = p_measured - Yf_raw.mean(axis=1)
            off = np.tile(b_now[:, None], (1, q))
            return Yf_raw + off, off

        # 'per_tap': fixed vector estimated once from the spin-up window
        if self.bias is None:
            raise RuntimeError("bias_mode='per_tap' requires estimate_bias() first")
        off = np.tile(self.bias[:, None], (1, self.q))
        return Yf_raw + off, off

    def estimate_bias(self, p_measured_window, substeps, verbose=False):
        """FORECAST-ONLY spin-up: step the ensemble through the window
        without any analysis update, accumulating

            b = mean_t [ p_measured(t) - ensmean_j p_predicted_j(t) ]

        p_measured_window : (n_window, n_taps)
        substeps          : solver substeps per assimilation cycle

        Returns a diagnostics dict. Uses tap data + model output only.
        """
        n_w = len(p_measured_window)
        resid = np.empty((n_w, self.n_taps))
        for k in range(n_w):
            Yf_raw = self.predicted_ensemble_pressure()
            resid[k] = p_measured_window[k] - Yf_raw.mean(axis=1)
            if k < n_w - 1:
                self.forecast_step(substeps)
        b = resid.mean(axis=0)
        self.bias = b
        self.bias_window_n = n_w
        diag = dict(
            bias=b.copy(),
            bias_mean=float(b.mean()),
            bias_std_over_taps=float(b.std()),
            resid_rms_before=float(np.sqrt(np.mean(resid ** 2))),
            resid_rms_after_pertap=float(np.sqrt(np.mean((resid - b) ** 2))),
            resid_rms_after_global=float(np.sqrt(np.mean((resid - resid.mean()) ** 2))),
            window_n=n_w,
        )
        if verbose:
            print('bias window n=%d: |resid| %.5f -> %.5f (per-tap) / %.5f (global scalar)'
                  % (n_w, diag['resid_rms_before'], diag['resid_rms_after_pertap'],
                     diag['resid_rms_after_global']))
        return diag

    # ------------------------------------------------------------------
    def assimilate_step(self, p_measured):
        q, n_taps = self.q, self.n_taps

        Xf_v = np.stack([self.vec.flatten(m) for m in self.members], axis=1)
        if self.augment_gamma:
            Xf = np.vstack([Xf_v, self.gammas[None, :]])
        else:
            Xf = Xf_v
        n_v = Xf_v.shape[0]

        Yf_raw = self.predicted_ensemble_pressure()
        Yf, offsets = self._debias(Yf_raw, p_measured)

        xbar_f = Xf.mean(axis=1)
        ybar_f = Yf.mean(axis=1)

        Xa = (Xf - xbar_f[:, None]) * self.alpha
        Ya = (Yf - ybar_f[:, None]) * self.alpha
        Xs = Xa / np.sqrt(q - 1)
        Ys = Ya / np.sqrt(q - 1)

        R = (self.sigma_p ** 2) * np.eye(n_taps)
        S = Ys @ Ys.T + R                      # (n_taps, n_taps) only
        K = Xs @ Ys.T @ np.linalg.solve(S, np.eye(n_taps))

        innovation_mean = p_measured - ybar_f
        xbar_a = xbar_f + K @ innovation_mean

        eps = self.rng.normal(0.0, self.sigma_p, size=(n_taps, q))
        eps -= eps.mean(axis=1, keepdims=True)
        Xf_analysis = Xf + K @ (p_measured[:, None] + eps - Yf)

        # ---- scalar gain fraction: the primary diagnostic ----------------
        s2 = float(np.mean(np.sum(Ys ** 2, axis=1)))   # mean per-tap ens variance
        gain_fraction = s2 / (s2 + self.sigma_p ** 2)

        # ---- innovation decomposition (static vs time-varying) -----------
        # recorded per cycle; the split is done afterwards over the run

        # ---- write the analysis back -------------------------------------
        for j in range(q):
            self.vec.unflatten_into(self.members[j], Xf_analysis[:n_v, j])

        gamma_diag = {}
        if self.augment_gamma:
            g_new = Xf_analysis[n_v, :].copy()
            g_pre_clip = g_new.copy()
            lo, hi = self.gamma_clip
            n_clip = int(np.sum((g_new < lo) | (g_new > hi)))
            self.clip_hits += n_clip
            g_new = np.clip(g_new, lo, hi)

            # relaxation-to-prior-spread inflation on gamma only: parameter
            # ensembles collapse fast, after which the parameter can never be
            # revised again. Re-inflate anomalies about the analysis MEAN
            # (mean is untouched, so this adds no information).
            g_mean = g_new.mean()
            g_std = g_new.std(ddof=1) if q > 1 else 0.0
            inflated = False
            if self.gamma_spread_floor is not None and 0 < g_std < self.gamma_spread_floor:
                g_new = g_mean + (g_new - g_mean) * (self.gamma_spread_floor / g_std)
                g_new = np.clip(g_new, lo, hi)
                inflated = True

            self.gammas = g_new
            self._apply_gammas()
            gamma_diag = dict(
                gamma_mean=float(g_new.mean()),
                gamma_std=float(g_new.std(ddof=1)) if q > 1 else 0.0,
                gamma_mean_forecast=float(xbar_f[n_v]),
                gamma_std_forecast=float(Xf[n_v].std(ddof=1)) if q > 1 else 0.0,
                gamma_update=float(g_new.mean() - xbar_f[n_v]),
                gamma_K_row_norm=float(np.linalg.norm(K[n_v])),
                gamma_obs_cov_norm=float(np.linalg.norm(Xs[n_v] @ Ys.T)),
                gamma_n_clipped=n_clip,
                gamma_pre_clip_min=float(g_pre_clip.min()),
                gamma_pre_clip_max=float(g_pre_clip.max()),
                gamma_inflated=bool(inflated),
                gammas=g_new.copy(),
            )

        diag = dict(
            ybar_f=ybar_f, y_measured=p_measured.copy(),
            y_pred_raw_mean=Yf_raw.mean(axis=1),
            innovation=innovation_mean,
            innovation_norm=float(np.linalg.norm(innovation_mean)),
            kalman_correction_norm=float(np.linalg.norm(K @ innovation_mean)),
            state_correction_norm=float(np.linalg.norm((K @ innovation_mean)[:n_v])),
            pressure_rmse=float(np.sqrt(np.mean(innovation_mean ** 2))),
            ensemble_spread_state=float(np.sqrt(np.mean(Xs[:n_v] ** 2)) * np.sqrt(q - 1)),
            ensemble_spread_pressure=float(np.sqrt(np.mean(Ys ** 2)) * np.sqrt(q - 1)),
            gain_fraction=float(gain_fraction),
            NIS=float(innovation_mean @ np.linalg.solve(S, innovation_mean)),
            bias_offset_mean=float(offsets.mean()),
        )
        diag.update(gamma_diag)
        return diag

    def forecast_step(self, substeps):
        for m in self.members:
            for _ in range(substeps):
                m.step()

    # ------------------------------------------------------------------
    def anomaly_spectrum(self):
        """Effective number of ensemble directions, current state."""
        from .ensemble_init import anomaly_spectrum
        X = np.stack([self.vec.flatten(m) for m in self.members], axis=1)
        return anomaly_spectrum(X)


def decompose_innovation(innov_hist):
    """Split an (n_cycles, n_taps) innovation history into its
    time-independent (static, per-tap mean) and time-varying parts.

    Returns a dict with the variance fractions. The Phase-1 finding was
    that ~85-89% of Stage D's innovation variance was static, i.e. the
    filter was mostly being fed a constant model bias rather than a
    signal it could dynamically track.
    """
    static = innov_hist.mean(axis=0)
    varying = innov_hist - static[None, :]
    v_tot = float(np.mean(innov_hist ** 2))
    v_static = float(np.mean(static ** 2))
    v_vary = float(np.mean(varying ** 2))
    glob = float(innov_hist.mean())
    return dict(
        total_ms=v_tot,
        static_ms=v_static,
        varying_ms=v_vary,
        static_fraction=v_static / v_tot if v_tot > 0 else 0.0,
        varying_fraction=v_vary / v_tot if v_tot > 0 else 0.0,
        global_constant_ms=glob ** 2,
        global_constant_fraction=(glob ** 2) / v_tot if v_tot > 0 else 0.0,
        static_per_tap=static,
    )
