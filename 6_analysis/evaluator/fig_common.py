"""Shared setup for the ModalPINN arm-study figures.

Every figure script imports from here so that colours, regime definitions and
the data load are identical across the set. Run any figure script directly:

    cd figures && python fig1_collapse.py

Each script writes <name>.png and <name>.pdf at 300 dpi into this directory.
"""
import os
import sys

# Make this directory importable even under an isolated interpreter
# (PYTHONSAFEPATH=1 drops the script directory from sys.path).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# Canonical copy lives at the experiment root; this file used to hold a
# byte-identical duplicate of it.
CSV = os.path.join(HERE, os.pardir, os.pardir, "4_runs", "arms_master_results.csv")

# ----------------------------------------------------------------------------
# Regime definitions. These are the rules the figures' category labels encode.
#   collapsed            oscillating mode is essentially absent
#   amplitude, no phase  mode has magnitude but relative L2 > 1, i.e. the
#                        reconstruction is worse than predicting zero
#   recovered            mode present with correct phase structure
# ----------------------------------------------------------------------------
COLLAPSE_AMP = 0.15   # far-core amplitude ratio below this is "collapsed"
ZERO_BASELINE = 1.0   # relative L2 at or above this is worse than zero

C_COLLAPSED = "#B0B7BE"   # neutral grey  - no wake recovered
C_FALSE = "#D1495B"       # alarm red     - amplitude without phase
C_RECOVERED = "#1F6FB2"   # focal blue    - genuine recovery
C_PROBE = "#0B3C5D"       # dark blue     - the information-rich reference
C_REF = "#4A4A4A"         # reference lines and annotations

REGIME_COLOR = {
    "collapsed": C_COLLAPSED,
    "amplitude, no phase": C_FALSE,
    "recovered": C_RECOVERED,
}


def regime(row):
    """Classify one arm. Order matters: the false-recovery test comes first."""
    if row.relL2_far_core >= ZERO_BASELINE:
        return "amplitude, no phase"
    if row.amp_far_core < COLLAPSE_AMP:
        return "collapsed"
    return "recovered"


def load():
    """Load the master table with the regime column attached."""
    df = pd.read_csv(CSV)
    df["regime"] = [regime(r) for r in df.itertuples()]
    df["color"] = df.regime.map(REGIME_COLOR)
    return df.sort_values("arm").reset_index(drop=True)


def style(sizes=(9, 8, 7)):
    """Font-size ladder mapped to role: base / annotation / ticks."""
    base, ann, tick = sizes
    mpl.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "font.size": base, "axes.titlesize": base, "axes.labelsize": base,
        "legend.fontsize": ann, "xtick.labelsize": tick, "ytick.labelsize": tick,
        "axes.titleweight": "regular", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": False, "axes.grid": False,
        "font.family": "sans-serif",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def panel_letter(ax, letter, dx=-0.08, dy=1.04):
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=mpl.rcParams["font.size"] + 2, fontweight="bold",
            va="bottom", ha="right")


def save(fig, stem):
    """Write PNG + PDF at 300 dpi and report the paths."""
    for ext in ("png", "pdf"):
        p = os.path.join(HERE, f"{stem}.{ext}")
        fig.savefig(p)
    return os.path.join(HERE, f"{stem}.png")


def check_overlaps(fig):
    """Geometric text-collision check. Returns a list of offending pairs."""
    r = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
             if t.get_text().strip() and t.get_visible()]
    ticks = {ax: set(ax.get_xticklabels() + ax.get_yticklabels()) for ax in fig.axes}
    bad = [(a.get_text(), b.get_text())
           for i, (a, ba) in enumerate(texts) for b, bb in texts[i + 1:]
           if ba.overlaps(bb)]
    for t, bt in texts:
        for ax in fig.axes:
            for s in ax.spines.values():
                if not s.get_visible():
                    continue
                if bt.overlaps(s.get_window_extent(r)) and t not in ticks[ax]:
                    bad.append((t.get_text(), f"spine:{ax.get_ylabel()[:12]}"))
    return bad
