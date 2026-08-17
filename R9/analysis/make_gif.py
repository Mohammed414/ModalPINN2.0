"""Animated GIF of one shedding period: reference CFD vs the R9 network.

Frames are generated from the trained modal fields analytically -
q(x,y,t) = Re[sum_k q_k e^{i k w0 t}] - so the network side is a genuine
continuous-in-time evaluation, not interpolated snapshots.

The CFD side is the reference dataset resampled onto the same phases by
its own modal fit (k=0..3), so both panels show exactly one period of the
same harmonic content and the comparison is like-for-like.

Velocity only: this module's pressure field is not validated near the
cylinder (see eval_runs.py docstring).
"""
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.interpolate import griddata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_runs as er                                    # noqa: E402
sys.path.insert(0, os.path.join(er.REPO, 'R9_wake_rescue', 'src'))
import common                                             # noqa: E402

FIG = os.path.join(er.R9_ROOT, 'figures')
NFRAMES = 40
OMEGA = er.OMEGA_0


def build(field='v', nframes=NFRAMES, out=None):
    """field: 'v' or 'u' (velocity component) or 'w' (spanwise vorticity).

    Vorticity is computed by finite differences ON THE INTERPOLATION GRID
    after reconstruction, identically for both panels, so it is a fair
    comparison; it is not the analytic curl of the modal fields.
    """
    x, y, times, U, V, P = common.load_truth_fields()
    xm, ym, tmodes = common.load_truth_modes()
    assert np.allclose(xm, x)

    d = glob.glob(os.path.join(er.REPO, 'runs/R9_extracted/*/'))[0]
    run = er.load_run(d, trust=True)
    u_net, v_net, p_net = er.eval_run(run, x, y)

    def to_one_sided(comp):
        """truth modes are two-sided (c0 + sum 2Re[ck e^{ikwt}]); convert to
        the network's one-sided convention so both reconstruct identically."""
        tm = tmodes[comp]
        return [tm[0].astype(complex)] + [2.0 * tm[k] for k in (1, 2, 3)]

    vort = field == 'w'
    if vort:
        net_pair = (u_net, v_net)
        truth_pair = (to_one_sided('u'), to_one_sided('v'))
    else:
        net_modes = {'u': u_net, 'v': v_net}[field]
        truth_modes = to_one_sided(field)

    # interpolation grid
    gx = np.linspace(-3.0, 8.0, 420)
    gy = np.linspace(-2.4, 2.4, 184)
    GX, GY = np.meshgrid(gx, gy)
    body = np.hypot(GX, GY) < 0.5
    pts = np.stack([x, y], 1)

    T = 2 * np.pi / OMEGA
    phases = np.arange(nframes) * T / nframes

    def at_phase(modes, t):
        f = np.zeros(len(x))
        for k, qk in enumerate(modes):
            f += (qk * np.exp(1j * k * OMEGA * t)).real
        return f

    def grid(f):
        return np.where(body, np.nan, griddata(pts, f, (GX, GY),
                                               method='linear'))

    def frames_for(modes):
        return [grid(at_phase(modes, t)) for t in phases]

    def frames_vort(pair):
        um, vm = pair
        dx, dy = gx[1] - gx[0], gy[1] - gy[0]
        out = []
        for t in phases:
            Ug, Vg = grid(at_phase(um, t)), grid(at_phase(vm, t))
            dvdx = np.gradient(Vg, dx, axis=1)
            dudy = np.gradient(Ug, dy, axis=0)
            out.append(dvdx - dudy)
        return out

    if vort:
        truth_frames = frames_vort(truth_pair)
        net_frames = frames_vort(net_pair)
        vmax, vmin, cmap = 2.2, -2.2, 'RdBu_r'
    else:
        truth_frames = frames_for(truth_modes)
        net_frames = frames_for(net_modes)
        vmax = 0.55 if field == 'v' else 1.35
        vmin = -vmax if field == 'v' else -0.35
        cmap = 'RdBu_r'

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.5), sharex=True,
                             sharey=True)
    ims = []
    for ax, lab, col in ((axes[0], 'reference CFD', '#1a1a1a'),
                         (axes[1], 'R9 network (32 pressure taps + physics)',
                          '#1b6f4a')):
        im = ax.pcolormesh(GX, GY, truth_frames[0], cmap=cmap, vmin=vmin,
                           vmax=vmax, shading='auto')
        ax.add_patch(plt.Circle((0, 0), 0.5, fc='0.88', ec='0.35', lw=0.8,
                                zorder=4))
        ax.set_aspect('equal')
        ax.set_ylabel('$y/D$')
        ax.text(0.011, 0.94, lab, transform=ax.transAxes, va='top', ha='left',
                fontsize=7.5, color=col, fontweight='bold', zorder=5,
                bbox=dict(fc='white', ec='none', alpha=0.78, pad=1.6))
        ims.append(im)
    axes[1].set_xlabel('$x/D$')
    title = fig.suptitle('', fontsize=8, y=0.965)
    cb = fig.colorbar(ims[0], ax=axes, shrink=0.8, pad=0.02, aspect=26)
    cb.set_label({'u': '$u / U_\\infty$', 'v': '$v / U_\\infty$',
                  'w': '$\\omega_z D / U_\\infty$'}[field])

    def update(i):
        ims[0].set_array(truth_frames[i].ravel())
        ims[1].set_array(net_frames[i].ravel())
        title.set_text(f'One shedding period at Re=100   '
                       f'($t/T$ = {i / nframes:.2f})')
        return ims + [title]

    out = out or os.path.join(FIG, f'anim_{field}_truth_vs_R9.gif')
    anim = FuncAnimation(fig, update, frames=nframes, blit=False)
    anim.save(out, writer=PillowWriter(fps=12), dpi=100)
    plt.close(fig)
    _quantize(out)
    return out


def _quantize(path, colors=96):
    """Re-encode with a shared adaptive palette - a smooth diverging field
    burns most of a 256-colour GIF palette per frame, so quantizing once and
    reusing the palette cuts the file several-fold with no visible change."""
    from PIL import Image
    im = Image.open(path)
    frames = []
    for i in range(im.n_frames):
        im.seek(i)
        frames.append(im.convert('RGB'))
    pal = frames[0].quantize(colors=colors, method=Image.MEDIANCUT)
    out = [f.quantize(palette=pal, dither=Image.NONE) for f in frames]
    out[0].save(path, save_all=True, append_images=out[1:],
                duration=im.info.get('duration', 83), loop=0, optimize=True)


if __name__ == '__main__':
    f = sys.argv[1] if len(sys.argv) > 1 else 'v'
    print(build(field=f))
