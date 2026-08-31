"""F04c - prior plus collocation: wake-biased sampling does not help the prior.

Top row shows what was changed: the actual collocation point sets, regenerated
from the training code (``gen_int_random_points`` with the run's own seed), so
the reader sees the intervention rather than a description of it. Bottom row
shows what it did.

Every panel carries the prior-only baseline, because without it the far-field
numbers read as network performance when they are in fact set by the analytical
prior through the v1 radial trust blend.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from figure_common import COLORS, REGION_COLORS, check_text_overlaps, new_figure, save_figure  # noqa: E402

METRICS = json.loads((ROOT / "data" / "analysis" / "a05_prior_collocation_metrics.json").read_text())
PRIOR = json.loads((ROOT / "data" / "analysis" / "a04_prior_only_metrics.json").read_text())
VENDOR = ROOT / "code" / "vendor"

# The two strategies, named by what they are rather than by run label.
STRATEGIES = [
    ("prior_uniform_collocation", "uniform sampling", COLORS["prior_network"],
     VENDOR / "load_train_data__15_karman_prior_fluct_off.py", "uniform"),
    ("prior_wake_biased_grid", "wake-biased grid", COLORS["accent"],
     VENDOR / "load_train_data__arm_10_prior_wake_biased_grid.py", "wake_biased_grid"),
]
REGIONS = ["near-cylinder", "near-wake", "far-core", "far-wake"]
REGION_TICKS = {
    "near-cylinder": "near cylinder\n$r<0.75$",
    "near-wake": "near wake\n$0\\leq x<3$",
    "far-core": "far wake core\n$x\\geq3$, $|y|\\leq2$",
    "far-wake": "far wake\n$x\\geq3$",
}
GEOM = [-4.0, 8.0, -4.0, 4.0, 0.0, 0.0, 0.5]
NINT = 50000
SEED = 0
NSHOW = 7000


def v1(key: str, region: str) -> float:
    return float(METRICS["models"][key]["v1_mode_metrics"][region]["rel_L2"])


def prior_v1(region: str) -> float:
    return float(PRIOR["v1_mode_metrics"][region]["rel_L2"])


def collocation_points(loader_source: Path, method: str):
    """Regenerate a run's interior collocation set from its own training code.

    The checkpoint-local ``Load_train_data_desync.py`` imports repo helpers
    (text_flow, reactions_process) that pull in scipy, none of which the sampler
    itself needs, so the function is extracted and executed on its own rather
    than importing the module. The source text is that run's own file.
    """
    source = loader_source.read_text()
    start = source.index("def gen_int_random_points")
    end = source.index("\ndef ", start + 1)
    namespace = {"np": np, "plt": plt}
    exec(compile(source[start:end], "gen_int_random_points", "exec"), namespace)
    np.random.seed(SEED)
    x, y = namespace["gen_int_random_points"](NINT, GEOM, method=method)
    if x.size != NINT:
        raise AssertionError("expected %d points, got %d" % (NINT, x.size))
    rng = np.random.default_rng(SEED)
    pick = rng.choice(x.size, size=min(NSHOW, x.size), replace=False)
    return x[pick], y[pick]


fig = new_figure(width="full", height=7.3)
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.22], hspace=0.62, wspace=0.30,
                      left=0.115, right=0.975, bottom=0.215, top=0.845)

# ---- top row: the intervention itself -------------------------------------
for column, (key, label, color, run_dir, method) in enumerate(STRATEGIES):
    ax = fig.add_subplot(gs[0, column])
    px, py = collocation_points(run_dir, method)
    ax.scatter(px, py, s=0.45, color=color, alpha=0.55, linewidths=0, zorder=3)
    ax.add_patch(plt.Circle((0.0, 0.0), 0.5, facecolor="white",
                            edgecolor=COLORS["reference"], linewidth=0.9, zorder=5))
    ax.axvline(3.0, color=COLORS["reference"], linewidth=0.8,
               linestyle=(0, (4, 3)), zorder=4)
    ax.text(3.15, -3.75, "$x=3$: start of far wake", fontsize=6.2,
            color=COLORS["reference"], ha="left", va="bottom")
    ax.set_xlim(-4.0, 8.0)
    ax.set_ylim(-4.0, 4.0)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x/D$ (downstream distance, diameters)")
    if column == 0:
        ax.set_ylabel(r"$y/D$ (cross-stream distance, diameters)")
    ax.set_title("%s: collocation points" % label, loc="left", pad=6.0, fontsize=8.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)

# ---- bottom row: what it produced ----------------------------------------
x = np.arange(len(REGIONS), dtype=float)
width = 0.34

ax = fig.add_subplot(gs[1, 0])
for offset, (key, label, color, _dir, _method) in zip((-width / 2, width / 2), STRATEGIES):
    ax.bar(x + offset, [v1(key, region) for region in REGIONS], width, color=color,
           edgecolor="white", linewidth=0.6, zorder=3, label=label)
for index, region in enumerate(REGIONS):
    level = prior_v1(region)
    ax.plot([index - 0.42, index + 0.42], [level, level], color=COLORS["reference"],
            linewidth=1.15, linestyle=(0, (3, 2)), zorder=5,
            label="prior formula on its own" if index == 0 else None)
ax.axhline(1.0, color=COLORS["muted"], linewidth=0.8, linestyle=(0, (1, 2.2)), zorder=2)
ax.text(3.40, 1.015, "predicting no oscillation at all", color=COLORS["muted"],
        fontsize=6.3, ha="right", va="bottom")
ax.text(0.0, prior_v1("near-cylinder") + 0.03, "1.37", color=COLORS["reference"],
        fontsize=6.4, ha="center", va="bottom")
ax.set_xticks(x)
ax.set_xticklabels([REGION_TICKS[region] for region in REGIONS], fontsize=7.0,
                   linespacing=1.35)
ax.set_ylabel(r"$v_1$ relative $L^2$ error" + "\n" + r"$\it{(0=exact,\ 1=zero\ prediction)}$", fontsize=8.2)
ax.set_ylim(0.0, 1.45)
ax.set_xlim(-0.62, 3.42)
ax.set_title("Worse in every region", loc="left", pad=7.0, fontsize=8.6)
ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, zorder=0)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.tick_params(axis="x", length=0, pad=3)
ax.tick_params(axis="y", length=2.5, width=0.6)
ax.legend(loc="upper center", bbox_to_anchor=(0.58, 1.005), frameon=False,
          fontsize=6.9, handlelength=1.7, handletextpad=0.5, labelspacing=0.42)

ax2 = fig.add_subplot(gs[1, 1])
for offset, (key, label, color, _dir, _method) in zip((-width / 2, width / 2), STRATEGIES):
    gains = [prior_v1(region) - v1(key, region) for region in REGIONS]
    ax2.bar(x + offset, gains, width, color=color, edgecolor="white",
            linewidth=0.6, zorder=3)
ax2.axhline(0.0, color=COLORS["reference"], linewidth=0.9, zorder=4)
ax2.annotate("below zero: the trained field is\nworse than the prior formula alone",
             xy=(2.5, -0.16), xytext=(0.72, 0.62), fontsize=6.7,
             color=COLORS["muted"], linespacing=1.4,
             arrowprops=dict(arrowstyle="-|>", color=COLORS["muted"], linewidth=0.75,
                             shrinkB=3))
ax2.set_xticks(x)
ax2.set_xticklabels([REGION_TICKS[region] for region in REGIONS], fontsize=7.0,
                    linespacing=1.35)
ax2.set_ylabel(r"prior $v_1$ error $-$ model $v_1$ error")
ax2.set_ylim(-0.36, 1.10)
ax2.set_xlim(-0.62, 3.42)
ax2.set_title("The network only helps near the body",
              loc="left", pad=7.0, fontsize=8.6)
ax2.grid(axis="y", color=COLORS["grid"], linewidth=0.55, zorder=0)
ax2.set_axisbelow(True)
for spine in ("top", "right"):
    ax2.spines[spine].set_visible(False)
ax2.tick_params(axis="x", length=0, pad=3)
ax2.tick_params(axis="y", length=2.5, width=0.6)

fig.suptitle("Concentrating collocation points in the wake does not improve the prior-assisted reconstruction",
             y=0.965, fontsize=10.2, color=COLORS["reference"])
fig.text(
    0.105, 0.022,
    "Both models use 32 wall pressure taps, the physics residual, and the analytical Karman-street prior;\n"
    "the recorded input-setting difference is where the 50,000 interior physics points are placed. Top:\n"
    "those point sets, regenerated from each run's own training code with its seed, 7,000 of 50,000 shown.\n"
    "Bottom: errors over 201 common snapshots. In the far core the radial trust constrains $v_1$ to the prior\n"
    "plus a bounded correction; the dashed prior-only level is therefore the necessary attribution baseline.\n"
    "Training length was not equalised (34,643 vs 26,129 evaluations), so differences are descriptive, not causal.",
    ha="left", va="bottom", fontsize=6.8, color=COLORS["muted"], linespacing=1.45,
)

bad = check_text_overlaps(fig)
if bad:
    print("TEXT ISSUES:", bad)
out = save_figure(fig, ROOT / "figures" / "F04c_prior_collocation")
print(out)
