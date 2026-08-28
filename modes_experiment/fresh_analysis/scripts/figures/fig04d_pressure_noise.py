"""F04d - pressure-noise robustness: noise degrades the learned part only.

Left panel is the reason this figure exists: the far-wake curve sits on the
prior-only level at every noise amplitude, so far-field agreement is a property
of the prior blend, not evidence that the reconstruction tolerates noise. Right
panel isolates what the network itself contributes, which is where noise
actually costs something.

Training length is deliberately kept off the axes (it belongs to the caveat, not
to the variable being studied) and reported in the caption instead.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from figure_common import COLORS, REGION_COLORS, check_text_overlaps, new_figure, save_figure  # noqa: E402

METRICS = json.loads((ROOT / "derived" / "a06_pressure_noise_metrics.json").read_text())
PRIOR = json.loads((ROOT / "derived" / "a04_prior_only_metrics.json").read_text())

# noise level -> the trained model at that level, named by what it is
LEVELS = [
    ("prior_noise_00pct", 0.0, "clean taps"),
    ("prior_noise_01pct", 1.0, "1%"),
    ("prior_noise_05pct", 5.0, "5%"),
    ("prior_noise_10pct", 10.0, "10%"),
]
REGIONS = [
    ("near-cylinder", "near cylinder ($r<0.75$)"),
    ("near-wake", "near wake ($0\\leq x<3$)"),
    ("far-core", "far wake core ($x\\geq3$)"),
]


def v1(key: str, region: str) -> float:
    return float(METRICS["models"][key]["v1_mode_metrics"][region]["rel_L2"])


def prior_v1(region: str) -> float:
    return float(PRIOR["v1_mode_metrics"][region]["rel_L2"])


def steps(key: str) -> int:
    return int(METRICS["models"][key]["effort"]["lbfgs_evals"])


fig = new_figure(width="full", height=5.0)
gs = fig.add_gridspec(1, 2, wspace=0.36, left=0.115, right=0.965,
                      bottom=0.335, top=0.695)
x = np.arange(len(LEVELS), dtype=float)
ticks = [label for _key, _pct, label in LEVELS]

# ---- left: absolute error, against each region's prior-only level ----------
ax = fig.add_subplot(gs[0, 0])
for region, label in REGIONS:
    color = REGION_COLORS[region]
    ax.plot(x, [v1(key, region) for key, _pct, _lab in LEVELS], color=color,
            linewidth=1.9, marker="o", markersize=5.4, markeredgecolor="white",
            markeredgewidth=0.65, zorder=4, label=label)
    level = prior_v1(region)
    if level <= 1.0:
        ax.axhline(level, color=color, linewidth=0.9, linestyle=(0, (3, 2.5)), zorder=2)
ax.text(2.97, prior_v1("far-core") - 0.11, "prior formula alone",
        color=REGION_COLORS["far-core"], fontsize=6.5, ha="right", va="top")
ax.text(2.97, prior_v1("near-wake") - 0.045, "prior formula alone",
        color=REGION_COLORS["near-wake"], fontsize=6.5, ha="right", va="top")
ax.set_xticks(x)
ax.set_xticklabels(ticks, fontsize=7.4)
ax.set_xlabel("pressure-tap noise level")
ax.set_ylabel(r"$v_1$ relative $L^2$ error" + "\n" + r"$\it{(0=exact,\ 1=zero\ prediction)}$", fontsize=8.2)
ax.set_ylim(0.0, 1.0)
ax.set_xlim(-0.13, 3.13)
ax.set_title("Far wake never leaves the prior level", loc="left", pad=7.0, fontsize=8.6)
ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, zorder=0)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.tick_params(axis="x", length=0, pad=3)
ax.tick_params(axis="y", length=2.5, width=0.6)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=6.8,
          handlelength=1.8, handletextpad=0.55, labelspacing=0.5)

# ---- right: what the network contributes over the prior --------------------
ax2 = fig.add_subplot(gs[0, 1])
for region, label in REGIONS:
    gains = [prior_v1(region) - v1(key, region) for key, _pct, _lab in LEVELS]
    ax2.plot(x, gains, color=REGION_COLORS[region], linewidth=1.9, marker="o",
             markersize=5.4, markeredgecolor="white", markeredgewidth=0.65, zorder=4)
    ax2.text(3.08, gains[-1], label.split(" (")[0], color=REGION_COLORS[region],
             fontsize=6.7, va="center", ha="left", clip_on=False)
ax2.axhline(0.0, color=COLORS["reference"], linewidth=0.9, zorder=3)
ax2.annotate("zero: model only\nreproduces the prior", xy=(1.5, 0.0), xytext=(1.15, 0.34),
             fontsize=6.4, color=COLORS["muted"], ha="left", va="bottom", linespacing=1.35,
             arrowprops=dict(arrowstyle="-|>", color=COLORS["muted"], linewidth=0.7, shrinkB=3))
ax2.set_xticks(x)
ax2.set_xticklabels(ticks, fontsize=7.4)
ax2.set_xlabel("pressure-tap noise level")
ax2.set_ylabel(r"prior $v_1$ error $-$ model $v_1$ error")
ax2.set_ylim(-0.1, 1.12)
ax2.set_xlim(-0.13, 4.45)
ax2.set_title("Noise erodes what the network adds", loc="left", pad=7.0, fontsize=8.6)
ax2.grid(axis="y", color=COLORS["grid"], linewidth=0.55, zorder=0)
ax2.set_axisbelow(True)
for spine in ("top", "right"):
    ax2.spines[spine].set_visible(False)
ax2.tick_params(axis="x", length=0, pad=3)
ax2.tick_params(axis="y", length=2.5, width=0.6)

fig.suptitle("Tap noise degrades the learned near-body field, not the prior-set far wake",
             y=0.955, fontsize=10.4, color=COLORS["reference"])
fig.text(
    0.115, 0.025,
    "All four models: 32 wall pressure taps, physics residual, and the same analytical Karman-street prior\n"
    "(identical prior file, so the prior itself never saw the noise); only the tap noise differs. Dashed lines\n"
    "are that prior evaluated with no network; its near-cylinder level, 1.37, is above the left panel.\n"
    "Downstream the prior sets the field by construction, so the far-wake curve staying put is not robustness.\n"
    "One training run per noise level, and training length was not equalised (%s optimiser steps for clean,\n"
    "1%%, 5%%, 10%% respectively), so the near-body trend is a direction, not a dose-response curve."
    % ", ".join(format(steps(key), ",") for key, _pct, _lab in LEVELS),
    ha="left", va="bottom", fontsize=6.8, color=COLORS["muted"], linespacing=1.45,
)

bad = check_text_overlaps(fig)
if bad:
    print("TEXT ISSUES:", bad)
out = save_figure(fig, ROOT / "figures" / "draft" / "F04d_pressure_noise")
print(out)
