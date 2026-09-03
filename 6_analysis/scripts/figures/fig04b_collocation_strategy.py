"""F04b - collocation trade-off: local gain versus wake fidelity."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from figure_common import COLORS, REGION_COLORS, check_text_overlaps, new_figure, save_figure  # noqa: E402


METRICS = json.loads((ROOT / "derived" / "a03_collocation_metrics.json").read_text())
METHODS = [
    ("uniform_collocation", "Uniform interior points", COLORS["pressure_only"], "o"),
    ("wake_biased_random_collocation", "Wake-biased random points", COLORS["prior"], "s"),
    ("wake_biased_grid_collocation", "Wake-biased grid points", COLORS["sparse_probes"], "D"),
]
REGIONS = ["near-cylinder", "near-wake", "far-core"]
REGION_LABELS = {
    "near-cylinder": "Near cylinder\n$r<0.75$",
    "near-wake": "Near wake\n$0\u2264 x<3$",
    "far-core": "Far core\n$x\u22653$, $|y|\u22642$",
}


def v1(method, region, metric):
    return float(METRICS["models"][method]["v1_mode_metrics"][region][metric])


fig = new_figure(width="full", height=4.65)
gs = fig.add_gridspec(
    1, 2, width_ratios=[1.1, 0.95], wspace=0.39,
    left=0.095, right=0.975, bottom=0.30, top=0.72,
)
x_regions = np.arange(len(REGIONS), dtype=float)

# Panel a: the regional error redistribution is the primary result.
ax = fig.add_subplot(gs[0, 0])
ax.axhline(1.0, color=COLORS["reference"], linewidth=0.8,
           linestyle=(0, (4, 2.5)), zorder=2)
for method, label, color, marker in METHODS:
    ax.plot(
        x_regions, [v1(method, region, "rel_L2") for region in REGIONS],
        color=color, linewidth=1.8, marker=marker, markersize=5.7,
        markeredgecolor="white", markeredgewidth=0.65, label=label, zorder=4,
    )
ax.set_xticks(x_regions)
ax.set_xticklabels([REGION_LABELS[r] for r in REGIONS], fontsize=7.2)
ax.set_ylabel(r"$v_1$ relative $L^2$ error")
ax.set_ylim(0.0, 1.28)
ax.set_xlim(-0.18, 2.18)
ax.set_title("Wake-biased points shift the error downstream", loc="left", pad=8.0, fontsize=9.0)
ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, zorder=0)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.tick_params(axis="x", length=0, pad=3)
ax.tick_params(axis="y", length=2.5, width=0.6)
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.01), frameon=False,
          fontsize=6.8, handlelength=1.8, handletextpad=0.55,
          labelspacing=0.55)

# Panel b: far-core amplitude and correlation share the same ideal value. The
# amplitude grows with wake bias, but correlation remains close to zero, so the
# additional oscillation is not the correct spatial/phase structure.
ax2 = fig.add_subplot(gs[0, 1])
x_methods = np.arange(len(METHODS), dtype=float)
for metric, label, color, linestyle, marker in [
    ("amp_ratio", "amplitude ratio", COLORS["prior"], "-", "o"),
    ("corr", "complex correlation", COLORS["sparse_probes"], "--", "s"),
]:
    ax2.plot(
        x_methods, [v1(method, "far-core", metric) for method, *_ in METHODS],
        color=color, linestyle=linestyle, linewidth=1.8, marker=marker,
        markersize=5.5, markeredgecolor="white", markeredgewidth=0.65,
        label=label, zorder=4,
    )
ax2.axhline(1.0, color=COLORS["reference"], linewidth=0.8,
            linestyle=(0, (4, 2.5)), zorder=2)
ax2.set_xticks(x_methods)
ax2.set_xticklabels(["Uniform", "Random\nwake bias", "Grid\nwake bias"], fontsize=7.2)
ax2.set_ylabel("far-core $v_1$ diagnostic")
ax2.set_ylim(0.0, 1.12)
ax2.set_xlim(-0.18, 2.18)
ax2.set_title("Amplitude grows, but agreement\nwith CFD remains low", loc="left", pad=8.0, fontsize=9.0)
ax2.grid(axis="y", color=COLORS["grid"], linewidth=0.55, zorder=0)
ax2.set_axisbelow(True)
for spine in ("top", "right"):
    ax2.spines[spine].set_visible(False)
ax2.tick_params(axis="x", length=0, pad=3)
ax2.tick_params(axis="y", length=2.5, width=0.6)
ax2.legend(loc="upper left", bbox_to_anchor=(0.0, 0.76), frameon=False,
           fontsize=7.0, handlelength=1.8, handletextpad=0.55)

fig.suptitle("Wake-biased collocation helps locally, but does not recover the wake",
             y=0.96, fontsize=10.6, color=COLORS["reference"])
fig.text(
    0.095, 0.028,
    "Pressure-only + physics; 32 cylinder pressure taps; 201 common snapshots.\n"
    "Lower relative $L^2$ is better; dashed line = zero prediction (1.0).\n"
    "Wake-biased sampling reduces near-cylinder $v_1$ error, but the far-core mode stays structurally\n"
    "incorrect: amplitude rises ~16x while correlation falls ~40%.\n"
    "Effort is not controlled (5,503 L-BFGS evaluations uniform vs 43,676 and 37,713 wake-biased).\n"
    "Evaluation count is not an accuracy proxy, so the between-arm gains and losses are descriptive;\n"
    "the common result is that no endpoint recovers the downstream travelling wake.",
    ha="left", va="bottom", fontsize=6.9, color=COLORS["muted"], linespacing=1.45,
)

bad = check_text_overlaps(fig)
if bad:
    print("TEXT ISSUES:", bad)
out = save_figure(fig, ROOT / "figures" / "final" / "F04b_collocation_strategy")
print(out)
