"""F04a - pressure-tap count: near-body sensitivity versus wake recovery."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from figure_common import COLORS, REGION_COLORS, check_text_overlaps, new_figure, save_figure  # noqa: E402


METRICS = json.loads((ROOT / "data" / "analysis" / "a02_tap_count_metrics.json").read_text())
METHODS = [
    ("pressure_only_physics_8_taps", 8),
    ("pressure_only_physics_16_taps", 16),
    ("pressure_only_physics_32_taps", 32),
]
REGIONS = ["near-cylinder", "near-wake", "far-core"]
REGION_LABELS = {
    "near-cylinder": "Near cylinder\n$r<0.75$",
    "near-wake": "Near wake\n$0\u2264 x<3$",
    "far-core": "Far core\n$x\u22653$, $|y|\u22642$",
}
REGION_LINE_COLORS = {
    "near-cylinder": REGION_COLORS["near-cylinder"],
    "near-wake": REGION_COLORS["near-wake"],
    "far-core": REGION_COLORS["far-core"],
}
MARKERS = {8: "o", 16: "s", 32: "D"}


def v1(region, metric):
    return [
        float(METRICS["models"][method]["v1_mode_metrics"][region][metric])
        for method, _ in METHODS
    ]


def field(quantity, region):
    return [
        float(METRICS["models"][method]["field_metrics"][region][quantity])
        for method, _ in METHODS
    ]


fig = new_figure(width="full", height=4.65)
gs = fig.add_gridspec(
    1, 2, width_ratios=[1.1, 0.95], wspace=0.39,
    left=0.095, right=0.975, bottom=0.30, top=0.72,
)

# Panel a: full-field near-cylinder errors. This shows the quantities for which
# denser pressure data can help, rather than implying that the harmonic trend
# represents every reconstructed variable.
ax = fig.add_subplot(gs[0, 0])
x = np.arange(len(METHODS), dtype=float)
for quantity, label, color, marker in [
    ("u", r"$u$", COLORS["sparse_probes"], "o"),
    ("v", r"$v$", COLORS["prior"], "s"),
    ("p", r"$p$", COLORS["prior_network"], "D"),
]:
    ys = field(quantity, "near-cylinder")
    ax.plot(
        x, ys, color=color, linewidth=1.8,
        marker=marker, markeredgecolor="white", markeredgewidth=0.65,
        markersize=5.8,
        zorder=4,
    )
    ax.text(2.05, ys[-1], label, color=color, va="center", ha="left",
            fontsize=8.0, fontweight="bold", clip_on=False)
ax.set_xticks(x)
ax.set_xticklabels([str(ntaps) for _, ntaps in METHODS])
ax.set_xlabel("number of cylinder pressure taps")
ax.set_ylabel(r"near-cylinder relative $L^2$ error")
ax.set_ylim(0.0, 0.17)
ax.set_xlim(-0.18, 2.33)
ax.set_title("More taps improve selected local fields", loc="left", pad=8.0, fontsize=9.0)
ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, zorder=0)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.tick_params(axis="x", length=0, pad=3)
ax.tick_params(axis="y", length=2.5, width=0.6)

# Panel b: first-harmonic error by region. This is the wake quantity targeted
# by the reconstruction and makes the local/wake contrast explicit.
ax2 = fig.add_subplot(gs[0, 1])
for region in REGIONS:
    ax2.plot(
        x, v1(region, "rel_L2"), color=REGION_LINE_COLORS[region],
        linewidth=1.8, marker="o", markersize=5.5,
        markeredgecolor="white", markeredgewidth=0.65,
        label=REGION_LABELS[region].replace("\n", " "), zorder=4,
    )
ax2.axhline(1.0, color=COLORS["reference"], linewidth=0.8,
            linestyle=(0, (4, 2.5)), zorder=2)
ax2.set_xticks(x)
ax2.set_xticklabels([str(ntaps) for _, ntaps in METHODS])
ax2.set_xlabel("number of cylinder pressure taps")
ax2.set_ylabel(r"$v_1$ relative $L^2$ error")
ax2.set_ylim(0.0, 1.17)
ax2.set_xlim(-0.18, 2.18)
ax2.set_title("The wake harmonic remains near\nthe zero-prediction baseline", loc="left", pad=8.0, fontsize=9.0)
ax2.grid(axis="y", color=COLORS["grid"], linewidth=0.55, zorder=0)
ax2.set_axisbelow(True)
for spine in ("top", "right"):
    ax2.spines[spine].set_visible(False)
ax2.tick_params(axis="x", length=0, pad=3)
ax2.tick_params(axis="y", length=2.5, width=0.6)
ax2.legend(loc="lower left", bbox_to_anchor=(0.0, 0.01), frameon=False,
           fontsize=7.0, handlelength=1.8, handletextpad=0.55,
           labelspacing=0.55)

fig.suptitle("More pressure taps improve local fields, not the unobserved wake",
             y=0.96, fontsize=10.6, color=COLORS["reference"])
fig.text(
    0.095, 0.028,
    "Pressure-only + physics; 201 common snapshots. Left: near-cylinder field errors for $u$, $v$, and $p$.\n"
    "Right: regional first-harmonic $v_1$ error; lower is better and the dashed line marks zero prediction (1.0).\n"
    "More wall-pressure taps improve selected local fields, but do not supply the downstream wake information.",
    ha="left", va="bottom", fontsize=6.9, color=COLORS["muted"], linespacing=1.45,
)

bad = check_text_overlaps(fig)
if bad:
    print("TEXT ISSUES:", bad)
out = save_figure(fig, ROOT / "figures" / "F04a_tap_count")
print(out)
