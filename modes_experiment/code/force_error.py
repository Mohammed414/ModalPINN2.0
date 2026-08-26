"""
force_error.py — lift/drag error of a ModalPINN reconstruction against the DNS.

Numpy only. Rebuilds the modal fields from a saved DNN*_tanh.pickle and integrates
the surface tractions on the cylinder to C_L(t) and C_D(t), then does the same on
the DNS field and reports the error per harmonic.

WHY NO PRIOR / BC ARGUMENTS ARE NEEDED
The cylinder wall ring sits at r = 0.5, i.e. x in [-0.55, 0.55] including the
friction offset rings. Every optional wrap in the trainer is inactive there:
  * v1 radial trust  : x-gate is 0 for x <= xstart - xwidth = 2.70   -> exactly off
  * freestream / damp: ramp 0.5(1 - tanh(3(x+2))) is <= 1.7e-04      -> negligible
So the wall field is the plain complex network output, masked by the paper's
no-slip prior dictionary f_BC = tanh(5(r - 0.5)) for u and v, unmasked for p.
This makes the same script valid for prior-on and prior-off arms alike, and it
sidesteps the hardcoded-inlet bug in evaluate_v1_smoke.py / evaluate_physics_uniform.py.

CONVENTIONS (read from NN_functions.py, not assumed)
  * weights/biases are complex64; the output layer width IS the mode count
  * q(x,y,t) = Re( sum_{k=0}^{N-1} q_k(x,y) e^{i k w0 t} )   -- no factor of 2
  * hidden activation tanh, applied to complex arguments

FORCE DEFINITIONS (following analysis/evidence/drag_split.json)
  C_Dp = 2 * closed-ring integral of (-p n_x) a dtheta        no differentiation
  C_Df = 2 * closed-ring integral of tau_t t_x a dtheta,
         tau_t = (1/Re) du_t/dn,  du_t/dn from one-sided offset rings
  (du_n/ds vanishes at a no-slip wall, so it is dropped)

Usage
  python force_error.py --RunDir <dir with DNN*.pickle> --DataFile <DNS file> \
      --Out out.json [--Ntheta 628] [--Offset 0.02] [--Re 100]
"""
import argparse, glob, json, os, pickle, sys
import numpy as np

OMEGA0 = 1.0357     # DNS shedding frequency, verified in the project records
A_CYL  = 0.5        # cylinder radius, D = 1


# --------------------------------------------------------------- network
def load_weights(run_dir):
    p = sorted(glob.glob(os.path.join(run_dir, "DNN*_tanh.pickle")))
    assert p, "no DNN*_tanh.pickle under %s" % run_dir
    with open(p[0], "rb") as f:
        d = pickle.load(f)
    assert len(d) == 6, "expected [w_u,b_u,w_v,b_v,w_p,b_p], got %d" % len(d)
    return p[0], d


def forward(x, y, W, B):
    """Complex MLP, tanh hidden, linear out. Returns [N, Nmode] complex."""
    W = [np.asarray(w) for w in W]
    B = [np.asarray(b).ravel() for b in B]
    H = np.stack([x.astype(np.complex128), y.astype(np.complex128)], axis=1)
    for l in range(len(W) - 1):
        H = np.tanh(H @ W[l] + B[l])
    return H @ W[-1] + B[-1]


def f_bc(x, y, gamma=5.0, r_c=A_CYL):
    return np.tanh(gamma * (np.sqrt(x**2 + y**2) - r_c))


def field_modes(x, y, W, B, mask):
    q = forward(x, y, W, B)
    if mask:
        q = q * f_bc(x, y)[:, None]
    return q


def in_time(q, t):
    """Re( sum_k q_k e^{i k w0 t} ) — the trainer's NN_time_* convention."""
    out = np.zeros(q.shape[0])
    for k in range(q.shape[1]):
        out = out + np.real(q[:, k] * np.exp(1j * k * OMEGA0 * t))
    return out


# --------------------------------------------------------------- DNS
def read_flow(path, xlim=(-4., 8.), ylim=(-4., 4.)):
    with open(path) as f:
        Re, Ur = [float(v) for v in f.readline().split()]
        f.readline()
        nt, nn = [int(v) for v in f.readline().split()]
        times = np.zeros(nt)
        X = np.empty((nt, nn), np.float32); Y = np.empty((nt, nn), np.float32)
        U = np.empty((nt, nn), np.float32); V = np.empty((nt, nn), np.float32)
        P = np.empty((nt, nn), np.float32)
        f.readline()                      # blank line after the "Nt N_nodes" line
        for it in range(nt):
            times[it] = float(f.readline())
            blk = np.fromstring(" ".join(f.readline() for _ in range(nn)),
                                sep=" ").reshape(nn, -1)
            X[it], Y[it], U[it], V[it], P[it] = (blk[:, 0], blk[:, 1],
                                                 blk[:, 2], blk[:, 3], blk[:, 4])
    keep = ((X[0] > xlim[0]) & (X[0] < xlim[1]) &
            (Y[0] > ylim[0]) & (Y[0] < ylim[1]))
    return Re, Ur, times, X[:, keep], Y[:, keep], U[:, keep], V[:, keep], P[:, keep]


# --------------------------------------------------------------- forces
def ring(ntheta, r):
    th = np.linspace(0., 2. * np.pi, ntheta, endpoint=False)
    return th, r * np.cos(th), r * np.sin(th)


def coefficients(th, p_w, ut_off, offset, Re, a=A_CYL):
    """
    C_D, C_L split into pressure and friction parts.
    p_w    : pressure on the wall ring, [Ntheta]
    ut_off : tangential velocity on the ring at r = a + offset, [Ntheta]
             (u_t = 0 on the wall itself, enforced by f_BC)
    """
    dth = 2. * np.pi / len(th)
    nx, ny = np.cos(th), np.sin(th)
    tx, ty = -np.sin(th), np.cos(th)
    CDp = 2. * np.sum(-p_w * nx) * a * dth
    CLp = 2. * np.sum(-p_w * ny) * a * dth
    tau = (ut_off - 0.0) / offset / Re          # du_t/dn one-sided, u_t(wall)=0
    CDf = 2. * np.sum(tau * tx) * a * dth
    CLf = 2. * np.sum(tau * ty) * a * dth
    return CDp, CLp, CDf, CLf


def harmonic_fit(sig, t, kmax=3):
    cols = [np.ones_like(t)]
    for k in range(1, kmax + 1):
        cols += [np.cos(k * OMEGA0 * t), np.sin(k * OMEGA0 * t)]
    A = np.stack(cols, axis=1)
    c, *_ = np.linalg.lstsq(A, sig, rcond=None)
    out = {"mean": float(c[0])}
    for k in range(1, kmax + 1):
        a, b = c[2 * k - 1], c[2 * k]
        out["k%d" % k] = {"amplitude": float(np.hypot(a, b)),
                          "phase_rad": float(np.arctan2(-b, a))}
    out["fit_residual_rms"] = float(np.sqrt(np.mean((sig - A @ c) ** 2)))
    return out


def wrap(d):
    return float((d + np.pi) % (2. * np.pi) - np.pi)


# --------------------------------------------------------------- DNS on the ring
def dns_on_ring(X, Y, U, V, P, th, r_wall, r_off, k_neigh=6):
    """
    Inverse-distance interpolation of the DNS field onto two rings.
    The DNS mesh is unstructured, so nearest-neighbour weighting is used rather
    than a structured lookup. k_neigh neighbours, weights 1/d^2.
    Returns p_wall[nt, nth] and ut_off[nt, nth].
    """
    xw, yw = r_wall * np.cos(th), r_wall * np.sin(th)
    xo, yo = r_off * np.cos(th), r_off * np.sin(th)
    tx, ty = -np.sin(th), np.cos(th)

    xs, ys = X[0], Y[0]                      # mesh is fixed in time
    def weights(xt, yt):
        W = np.zeros((len(xt), k_neigh)); I = np.zeros((len(xt), k_neigh), int)
        for i in range(len(xt)):
            d2 = (xs - xt[i]) ** 2 + (ys - yt[i]) ** 2
            idx = np.argpartition(d2, k_neigh)[:k_neigh]
            w = 1.0 / np.maximum(d2[idx], 1e-12)
            I[i] = idx; W[i] = w / w.sum()
        return I, W

    Iw, Ww = weights(xw, yw)
    Io, Wo = weights(xo, yo)
    nt, nth = X.shape[0], len(th)
    p_wall = np.zeros((nt, nth)); ut_off = np.zeros((nt, nth))
    for it in range(nt):
        p_wall[it] = np.sum(P[it][Iw] * Ww, axis=1)
        uo = np.sum(U[it][Io] * Wo, axis=1)
        vo = np.sum(V[it][Io] * Wo, axis=1)
        ut_off[it] = uo * tx + vo * ty
    return p_wall, ut_off


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--RunDir", required=True)
    ap.add_argument("--DataFile", required=True)
    ap.add_argument("--Out", default="force_error.json")
    ap.add_argument("--Ntheta", type=int, default=628)
    ap.add_argument("--Offset", type=float, default=0.02,
                    help="radial offset for the du_t/dn difference")
    ap.add_argument("--Re", type=float, default=100.0)
    ap.add_argument("--Label", default="")
    args = ap.parse_args()

    pick, wb = load_weights(args.RunDir)
    w_u, b_u, w_v, b_v, w_p, b_p = wb
    nmode = np.asarray(w_u[-1]).shape[1]
    print("checkpoint : %s" % os.path.basename(pick))
    print("modes      : %d  (k = 0..%d)" % (nmode, nmode - 1))

    Re, Ur, times, X, Y, U, V, P = read_flow(args.DataFile)
    print("DNS        : Re=%g  %d snapshots  %d nodes in the cropped domain"
          % (Re, len(times), X.shape[1]))

    th, xw, yw = ring(args.Ntheta, A_CYL)
    _,  xo, yo = ring(args.Ntheta, A_CYL + args.Offset)
    tx, ty = -np.sin(th), np.cos(th)

    # network modes on both rings, once
    qp_w = field_modes(xw, yw, w_p, b_p, mask=False)          # pressure, unmasked
    qu_o = field_modes(xo, yo, w_u, b_u, mask=True)
    qv_o = field_modes(xo, yo, w_v, b_v, mask=True)

    # sanity: the no-slip mask must annihilate u,v exactly on the wall
    qu_w = field_modes(xw, yw, w_u, b_u, mask=True)
    assert np.max(np.abs(qu_w)) < 1e-12, ("f_BC did not vanish on the wall: %.3e"
                                          % np.max(np.abs(qu_w)))

    p_dns, ut_dns = dns_on_ring(X, Y, U, V, P, th, A_CYL, A_CYL + args.Offset)

    rows = []
    for it, t in enumerate(times):
        pw  = in_time(qp_w, t)
        uo  = in_time(qu_o, t)
        vo  = in_time(qv_o, t)
        uto = uo * tx + vo * ty
        n = coefficients(th, pw, uto, args.Offset, args.Re)
        d = coefficients(th, p_dns[it], ut_dns[it], args.Offset, args.Re)
        rows.append((t, n, d))

    t   = np.array([r[0] for r in rows])
    def series(i, which):
        return np.array([r[1 if which == "nn" else 2][i] for r in rows])
    out = {"checkpoint": os.path.basename(pick), "run_dir": args.RunDir,
           "label": args.Label, "nmodes": int(nmode), "Re": args.Re,
           "ntheta": args.Ntheta, "wall_offset": args.Offset,
           "n_snapshots": int(len(t)), "omega0": OMEGA0,
           "method": ("pressure: 2*ring_integral(-p n) a dtheta ; "
                      "friction: tau_t=(1/Re) u_t(a+offset)/offset, "
                      "u_t(wall)=0 by f_BC ; "
                      "DNS interpolated by inverse-distance, 6 neighbours"),
           "gates_inactive_at_wall": ("v1 trust x-gate is 0 for x<=2.70 and the "
                                      "ring spans |x|<=0.55 ; the inlet ramp is "
                                      "<=1.7e-04 there. Neither affects the wall.")}

    for nm, ip, ifr in (("CD", 0, 2), ("CL", 1, 3)):
        nn = series(ip, "nn") + series(ifr, "nn")
        dn = series(ip, "dns") + series(ifr, "dns")
        hn, hd = harmonic_fit(nn, t), harmonic_fit(dn, t)
        kk = "k2" if nm == "CD" else "k1"       # the report's channel assignment
        amp_n, amp_d = hn[kk]["amplitude"], hd[kk]["amplitude"]
        out[nm] = {
            "dns":     {"mean": hd["mean"], "osc_amplitude_%s" % kk: amp_d,
                        "phase_rad_%s" % kk: hd[kk]["phase_rad"],
                        "rms": float(np.sqrt(np.mean(dn ** 2)))},
            "network": {"mean": hn["mean"], "osc_amplitude_%s" % kk: amp_n,
                        "phase_rad_%s" % kk: hn[kk]["phase_rad"],
                        "rms": float(np.sqrt(np.mean(nn ** 2)))},
            "error": {
              "mean_abs":        float(hn["mean"] - hd["mean"]),
              "mean_rel":        float((hn["mean"] - hd["mean"]) / hd["mean"])
                                 if hd["mean"] else None,
              "amplitude_rel":   float((amp_n - amp_d) / amp_d) if amp_d else None,
              "phase_err_rad":   wrap(hn[kk]["phase_rad"] - hd[kk]["phase_rad"]),
              "phase_err_deg":   float(np.degrees(
                                   wrap(hn[kk]["phase_rad"] - hd[kk]["phase_rad"]))),
              "phase_err_frac_period": float(abs(wrap(
                                   hn[kk]["phase_rad"] - hd[kk]["phase_rad"]))
                                   / (2. * np.pi)),
              "timeseries_rel_L2": float(np.linalg.norm(nn - dn)
                                         / np.linalg.norm(dn)),
              "correlation":     float(np.corrcoef(nn, dn)[0, 1]),
            },
            "channel": kk,
            "split": {"dns_pressure_mean": float(np.mean(series(ip, "dns"))),
                      "dns_friction_mean": float(np.mean(series(ifr, "dns"))),
                      "nn_pressure_mean":  float(np.mean(series(ip, "nn"))),
                      "nn_friction_mean":  float(np.mean(series(ifr, "nn")))},
        }

    with open(args.Out, "w") as f:
        json.dump(out, f, indent=2)

    print()
    for nm in ("CD", "CL"):
        e = out[nm]["error"]; k = out[nm]["channel"]
        print("%s  (oscillation in %s)" % (nm, k))
        print("   DNS      mean %+.5f   amp %.6f   phase %+.4f rad"
              % (out[nm]["dns"]["mean"], out[nm]["dns"]["osc_amplitude_%s" % k],
                 out[nm]["dns"]["phase_rad_%s" % k]))
        print("   network  mean %+.5f   amp %.6f   phase %+.4f rad"
              % (out[nm]["network"]["mean"],
                 out[nm]["network"]["osc_amplitude_%s" % k],
                 out[nm]["network"]["phase_rad_%s" % k]))
        print("   error    mean %+.1f%%   amp %+.1f%%   phase %+.2f deg "
              "(%.4f of a period)"
              % (100 * (e["mean_rel"] or 0), 100 * (e["amplitude_rel"] or 0),
                 e["phase_err_deg"], e["phase_err_frac_period"]))
        print("   time series: rel L2 %.4f   correlation %.4f"
              % (e["timeseries_rel_L2"], e["correlation"]))
        print()
    print("wrote", args.Out)


if __name__ == "__main__":
    main()
