"""Shared visual language for every fresh-analysis dissertation figure.

Figure-specific scripts should contain layout and annotations only.  Colours,
typography, dimensions, axis treatment, and export settings live
here so later figures cannot silently drift in appearance.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "modalpinn-fresh-analysis-mpl")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "modalpinn-fresh-analysis-cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


MM_PER_INCH = 25.4
SINGLE_COLUMN_IN = 85.0 / MM_PER_INCH
FULL_WIDTH_IN = 178.0 / MM_PER_INCH

COLORS = {
    "reference": "#111111",
    "pressure_only": "#6B7280",
    "sparse_probes": "#0072B2",
    "dense": "#009E73",
    "prior": "#D55E00",
    "prior_network": "#CC79A7",
    "accent": "#E69F00",
    "tap_8": "#7A2E00",
    "tap_16": "#D55E00",
    "tap_32": "#E9A27F",
    "grid": "#D1D5DB",
    "muted": "#6B7280",
}

REGION_COLORS = {
    "other": "#D9D9D9",
    "near-cylinder": "#CC79A7",
    "near-wake": "#56B4E9",
    "far-wake": "#009E73",
    "far-core": "#E69F00",
}

METHOD_LABELS = {
    "arm01": "Pressure only (32 taps)",
    "arm1_baseline": "Pressure-only + physics",
    "pressure_only_physics": "Pressure-only + physics",
    "arm04": "Pressure taps + velocity probes",
    "pressure_and_velocity_probes_physics": "Pressure + velocity probes + physics",
    "arm05": "Dense observations",
    "dense_observations": "Dense observations (ceiling)",
    "arm15": "Kármán prior (32 taps)",
    "arm15_v1_radial_trust": "Pressure-only + physics + Kármán prior",
    "prior_only": "Kármán prior only",
    "dns": "CFD reference",
}


def apply_style() -> None:
    """Apply the fixed dissertation style."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "stixsans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.5,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "axes.axisbelow": True,
            "savefig.dpi": 300,
            # Preserve the declared physical width. Tight bounding boxes make
            # nominal 178 mm figures silently export narrower than specified.
            "savefig.bbox": None,
        }
    )


def new_figure(*, width: str = "full", height: float = 3.6, **kwargs):
    """Create a consistently sized figure."""
    apply_style()
    width_in = FULL_WIDTH_IN if width == "full" else SINGLE_COLUMN_IN
    return plt.figure(figsize=(width_in, height), facecolor="white", **kwargs)


def domain_axes(ax, *, xlim=(-4.0, 8.0), ylim=(-4.0, 4.0)) -> None:
    """Apply the common cylinder-domain axes."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x/D$")
    ax.set_ylabel(r"$y/D$")
    ax.set_xticks([-4, -2, 0, 2, 4, 6, 8])
    ax.set_yticks([-4, -2, 0, 2, 4])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#4B5563")
        spine.set_linewidth(0.7)


def draw_cylinder(ax, *, radius: float = 0.5, zorder: int = 10) -> Circle:
    """Draw the solid cylinder consistently."""
    patch = Circle(
        (0.0, 0.0), radius, facecolor="white", edgecolor=COLORS["reference"],
        linewidth=1.0, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def region_patch(ax, kind, *, color, alpha=0.40, zorder=1, **kwargs):
    """Draw one evaluation region as filled geometry rather than node scatter.

    The regions are defined by inequalities, so they are drawn as exact patches;
    plotting the mesh nodes instead makes the figure a mesh-density map and
    leaves the region boundaries invisible where the mesh is coarse.
    """
    from matplotlib.patches import Rectangle

    if kind == "band":                       # x in [x0, x1), full height
        x0, x1, y0, y1 = kwargs.pop("extent")
        patch = Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=color,
                          alpha=alpha, edgecolor="none", zorder=zorder, **kwargs)
    elif kind == "disk":                     # r < r_outer
        patch = Circle((0.0, 0.0), kwargs.pop("radius"), facecolor=color,
                       alpha=alpha, edgecolor="none", zorder=zorder, **kwargs)
    else:
        raise ValueError(f"unknown region kind {kind!r}")
    ax.add_patch(patch)
    return patch


def save_figure(fig, output_base: Path) -> Path:
    """Save a 300-dpi PNG at the declared physical figure size."""
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png = output_base.with_suffix(".png")
    fig.savefig(png, dpi=300, bbox_inches=None)
    plt.close(fig)
    return png


def check_text_overlaps(fig):
    """Return visible text boxes that collide or are clipped, for the
    render-then-verify pass.

    Two independent failure modes are checked: (1) two text boxes overlapping
    each other or a spine, via matplotlib's own window-extent bookkeeping, and
    (2) a text box extending past the figure canvas - checked by actually
    rendering to a raster and looking for non-background ink against the
    canvas edge, rather than by trusting get_window_extent() near the
    boundary. That bbox can disagree with the rendered pixels for text placed
    with a horizontal-alignment offset (e.g. ha="right"/"left" with a
    negative x), which produced false positives here.
    """
    import io

    import matplotlib as mpl
    import numpy as np
    from PIL import Image

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
             if t.get_text().strip() and t.get_visible()]
    ticks = {ax: set(ax.get_xticklabels() + ax.get_yticklabels()) for ax in fig.axes}
    tick_texts = {id(t) for tick_set in ticks.values() for t in tick_set}
    bad = [(a.get_text(), b.get_text())
           for i, (a, ba) in enumerate(texts) for b, bb in texts[i + 1:]
           if id(a) not in tick_texts and id(b) not in tick_texts and ba.overlaps(bb)]
    for t, bt in texts:
        for ax in fig.axes:
            if id(t) in tick_texts:
                continue
            for s in ax.spines.values():
                if s.get_visible() and bt.overlaps(s.get_window_extent(r)):
                    bad.append((t.get_text(), f"spine:{ax.get_ylabel()[:14]}"))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi)
    buf.seek(0)
    arr = np.asarray(Image.open(buf).convert("L"))
    edge_has_ink = (arr[0].min() < 250 or arr[-1].min() < 250
                    or arr[:, 0].min() < 250 or arr[:, -1].min() < 250)
    if edge_has_ink:
        bad.append(("(figure edge)", "CLIPPED: ink touches the outer canvas edge"))
    return bad
