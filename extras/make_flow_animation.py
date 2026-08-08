"""
Animate the full CFD flow field (data/fixed_cylinder_atRe100) as vorticity
contours over the cylinder wake, matching the region ModalPINN is actually
trained/evaluated on (see Lxmin/Lxmax/Lymin/Lymax/x_c/y_c/r_c in
src/pressure_only/ModalPINN_VortexShedding.py).

The raw file only gives scattered node values (x, y, U, V, p) per timestep,
no mesh connectivity, so we re-triangulate with Delaunay and:
  - drop triangles that fall inside the solid cylinder (Qhull has no
    knowledge of the hole and will happily bridge across it), and
  - drop slivers with an abnormally long edge (same bridging problem at
    the mesh's outer refinement boundary).

Vorticity (dv/dx - du/dy) is computed once per triangle as the exact
gradient of the piecewise-linear interpolant over that triangle (standard
P1 finite-element gradient, closed form, fully vectorised over all
triangles/timesteps -- no scipy/FEM library needed).

Usage:
    python3 make_flow_animation.py            # full animation -> flow_animation.gif
    python3 make_flow_animation.py --test      # single PNG frame, for tuning
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.animation as animation

from parse_flow import load_flow

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_FLOW_FILE = os.path.join(HERE, '..', 'data', 'fixed_cylinder_atRe100')
CACHE_FILE = os.path.join(HERE, 'flow_cache.npz')

# Region of interest -- identical to geom in src/pressure_only/ModalPINN_VortexShedding.py
Lxmin, Lxmax = -4., 8.
Lymin, Lymax = -4., 4.
x_c, y_c, r_c = 0., 0., 0.5


def build_triangulation(X, Y):
    triang = mtri.Triangulation(X, Y)
    tris = triang.triangles  # (Ntri, 3) vertex indices

    x0, y0 = X[tris[:, 0]], Y[tris[:, 0]]
    x1, y1 = X[tris[:, 1]], Y[tris[:, 1]]
    x2, y2 = X[tris[:, 2]], Y[tris[:, 2]]

    # edge lengths, to mask slivers bridging the cylinder hole / domain edges
    e01 = np.hypot(x1 - x0, y1 - y0)
    e12 = np.hypot(x2 - x1, y2 - y1)
    e20 = np.hypot(x0 - x2, y0 - y2)
    max_edge = np.maximum(np.maximum(e01, e12), e20)
    edge_thresh = 6. * np.median(max_edge)

    cx, cy = (x0 + x1 + x2) / 3., (y0 + y1 + y2) / 3.
    r_centroid = np.hypot(cx - x_c, cy - y_c)

    mask = (max_edge > edge_thresh) | (r_centroid < r_c)
    triang.set_mask(mask)

    print('Triangulation: %d triangles total, %d masked (hole/slivers), %d kept'
          % (len(tris), mask.sum(), (~mask).sum()))

    return triang, tris, mask


def triangle_gradient(tris, X, Y, f):
    """P1 (piecewise-linear) gradient of f, constant per triangle. f: (..., Nnodes)."""
    x0, y0 = X[tris[:, 0]], Y[tris[:, 0]]
    x1, y1 = X[tris[:, 1]], Y[tris[:, 1]]
    x2, y2 = X[tris[:, 2]], Y[tris[:, 2]]

    m00, m01 = x1 - x0, y1 - y0
    m10, m11 = x2 - x0, y2 - y0
    det = m00 * m11 - m01 * m10  # = 2 * signed triangle area

    f0, f1, f2 = f[..., tris[:, 0]], f[..., tris[:, 1]], f[..., tris[:, 2]]
    d1, d2 = f1 - f0, f2 - f0

    dfdx = (m11 * d1 - m01 * d2) / det
    dfdy = (-m10 * d1 + m00 * d2) / det
    return dfdx, dfdy


def compute_vorticity(tris, X, Y, U, V):
    """U, V: (Nt, Nnodes) -> vorticity: (Nt, Ntri), constant per triangle."""
    _, dudy = triangle_gradient(tris, X, Y, U)
    dvdx, _ = triangle_gradient(tris, X, Y, V)
    return dvdx - dudy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', action='store_true', help='render a single frame to PNG instead of the full animation')
    ap.add_argument('--field', choices=['vorticity', 'speed', 'pressure'], default='vorticity')
    ap.add_argument('--stride', type=int, default=1, help='use every Nth timestep (speeds up / shortens the animation)')
    ap.add_argument('--fps', type=int, default=15)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    Re, Ur, times, X, Y, U, V, p = load_flow(RAW_FLOW_FILE, cache=CACHE_FILE)

    crop = (X > Lxmin) & (X < Lxmax) & (Y > Lymin) & (Y < Lymax)
    X, Y = X[crop], Y[crop]
    U, V, p = U[:, crop], V[:, crop], p[:, crop]
    print('Cropped to region of interest: %d nodes' % len(X))

    triang, tris, mask = build_triangulation(X, Y)
    keep = ~mask

    # facecolors passed to tripcolor must cover ALL triangles (masked ones
    # included) -- matplotlib drops the masked entries internally via
    # `facecolors[~tri.mask]`. Compute on the full triangle array; masked
    # (hole/sliver) triangles' bogus values are simply never drawn.
    if args.field == 'vorticity':
        field_all = compute_vorticity(tris, X, Y, U, V)  # (Nt, Ntri)
        cmap, label = 'RdBu_r', 'Vorticity  $\\omega_z = \\partial v/\\partial x - \\partial u/\\partial y$'
        vmax = np.percentile(np.abs(field_all[:, keep]), 99)
        vmin = -vmax
    elif args.field == 'speed':
        speed = np.sqrt(U ** 2 + V ** 2)  # per-node
        field_all = speed[:, tris].mean(axis=2)
        cmap, label = 'viridis', 'Speed  $\\sqrt{u^2+v^2}$'
        vmin, vmax = 0., np.percentile(field_all[:, keep], 99)
    else:
        field_all = p[:, tris].mean(axis=2)
        cmap, label = 'coolwarm', 'Pressure  $p$'
        vmax = np.percentile(np.abs(field_all[:, keep]), 99)
        vmin = -vmax

    frame_idx = np.arange(0, len(times), args.stride)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(Lxmin, Lxmax)
    ax.set_ylim(Lymin, Lymax)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # tripcolor's PolyCollection only stores colors for the unmasked
    # (kept) triangles internally, so all later .set_array() calls must
    # be pre-filtered with `keep` too -- passing the full-length array
    # silently misaligns colors to the wrong triangles.
    tpc = ax.tripcolor(triang, facecolors=field_all[frame_idx[0]], cmap=cmap,
                        vmin=vmin, vmax=vmax, shading='flat')
    cyl_patch = plt.Circle((x_c, y_c), r_c, color='0.15', zorder=5)
    ax.add_patch(cyl_patch)
    cbar = fig.colorbar(tpc, ax=ax, label=label)
    title = ax.set_title('Cylinder wake, Re=%d  |  t = %.2f' % (Re, times[frame_idx[0]]))
    fig.tight_layout()

    if args.test:
        out = args.out or os.path.join(HERE, 'test_frame.png')
        fig.savefig(out, dpi=150)
        print('Wrote test frame to %s' % out)
        return

    def update(i):
        idx = frame_idx[i]
        tpc.set_array(field_all[idx][keep])
        title.set_text('Cylinder wake, Re=%d  |  t = %.2f' % (Re, times[idx]))
        return tpc, title

    anim = animation.FuncAnimation(fig, update, frames=len(frame_idx), blit=False)

    out = args.out or os.path.join(HERE, 'flow_animation_%s.gif' % args.field)
    print('Rendering %d frames to %s ...' % (len(frame_idx), out))
    anim.save(out, writer=animation.PillowWriter(fps=args.fps))
    print('Done: %s' % out)


if __name__ == '__main__':
    main()
