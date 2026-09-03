"""Build all_results.xlsx: every number in this folder, in one filterable book.

Run:  python3 code/make_workbook.py
"""
from __future__ import annotations

import csv
import json
import math
import pathlib
import subprocess
import sys

import numpy as np
import xlsxwriter

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ANA = DATA / "analysis"
GAPPY = DATA / "gappy"
OUT = ROOT / "all_results.xlsx"

INK = "#1F2933"
MUTED = "#7B8794"
RULE = "#D8DEE4"
BAND = "#F4F6F8"
GOOD = "#2E7D5B"
WARN = "#B5651D"
BADC = "#A33A3A"


def rows(p: pathlib.Path) -> list[dict]:
    with p.open() as f:
        return list(csv.DictReader(f))


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else str(x)
    except (TypeError, ValueError):
        return x


book = xlsxwriter.Workbook(OUT, {"nan_inf_to_errors": True})

F = {
    "title":  book.add_format({"bold": True, "font_size": 15, "font_color": INK}),
    "sub":    book.add_format({"font_size": 10, "font_color": MUTED, "text_wrap": True, "valign": "top"}),
    "h":      book.add_format({"bold": True, "font_color": "white", "bg_color": "#3E4C59",
                               "border": 1, "border_color": "#3E4C59", "align": "left",
                               "valign": "vcenter", "text_wrap": True}),
    "t":      book.add_format({"font_color": INK, "border": 1, "border_color": RULE}),
    "tb":     book.add_format({"font_color": INK, "border": 1, "border_color": RULE, "bg_color": BAND}),
    "n":      book.add_format({"num_format": "0.000000", "border": 1, "border_color": RULE}),
    "n3":     book.add_format({"num_format": "0.000", "border": 1, "border_color": RULE}),
    "pct":    book.add_format({"num_format": "0.0", "border": 1, "border_color": RULE}),
    "int":    book.add_format({"num_format": "#,##0", "border": 1, "border_color": RULE}),
    "sec":    book.add_format({"bold": True, "font_size": 11, "font_color": INK, "bottom": 2,
                               "border_color": "#3E4C59"}),
    "good":   book.add_format({"font_color": GOOD, "bold": True}),
    "bad":    book.add_format({"font_color": BADC, "bold": True}),
    "warn":   book.add_format({"font_color": WARN, "bold": True}),
    "mono":   book.add_format({"font_name": "Menlo", "font_size": 9, "font_color": INK,
                               "text_wrap": True, "valign": "top"}),
    "wrap":   book.add_format({"text_wrap": True, "valign": "top", "border": 1,
                               "border_color": RULE}),
}


def table(ws, data, headers, start=0, widths=None, fmts=None, name=None):
    """Write a header row + rows, with autofilter and frozen header."""
    for c, h in enumerate(headers):
        ws.write(start, c, h, F["h"])
    for r, row in enumerate(data, start=start + 1):
        for c, v in enumerate(row):
            f = (fmts or {}).get(c)
            if f is None:
                f = F["n"] if isinstance(v, float) else F["t"]
            ws.write(r, c, v, f)
    if data:
        ws.autofilter(start, 0, start + len(data), len(headers) - 1)
    ws.freeze_panes(start + 1, 0)
    for c, w in enumerate(widths or []):
        ws.set_column(c, c, w)
    ws.set_row(start, 30)
    return start + len(data)


# ===========================================================================
# 1. Contents
# ===========================================================================
ws = book.add_worksheet("Contents")
ws.hide_gridlines(2)
ws.set_column(0, 0, 26)
ws.set_column(1, 1, 92)
ws.write(0, 0, "ModalPINN wake reconstruction — all results", F["title"])
ws.write(1, 0, "Every number behind the dissertation, and where each one comes from. "
               "Each sheet has filters on its header row.", F["sub"])
ws.set_row(1, 30)

CONTENTS = [
    ("Headline", "The handful of numbers the report actually turns on, with the file each is read from."),
    ("Master", "All 945 accepted ModalPINN rows. One row = one metric, one region, one arm. Filter here."),
    ("Far core v", "The cross-method comparison, laid out arm x region. This is the report's pivot table."),
    ("Prior attribution", "What the Karman prior supplies vs what the network adds, by region."),
    ("Tap count", "8 / 16 / 32 wall pressure taps."),
    ("Collocation", "Uniform vs wake-biased interior sampling, with the training-effort caveat."),
    ("Noise", "Prior-assisted reconstruction at 0 / 1 / 5 / 10 % pressure noise."),
    ("Gappy POD", "The linear baseline: 210 metric rows plus the report-facing values."),
    ("Wavelengths", "A07 centreline phase gradients. The mechanism evidence."),
    ("Termination census", "How all 17 training runs died, from the L-BFGS logs."),
    ("All results", "Every metric from both methods in one table, with the configuration of each arm. The best sheet to filter."),
    ("Phase correction", "The superseded pre-correction values, and what each became. Reference only."),
    ("Figures", "The 20 figures, what each shows, and the script that draws it."),
    ("Verification", "Output of code/verify.py: what was checked and what did not check out."),
]
r = 3
ws.write(r, 0, "Sheet", F["h"]); ws.write(r, 1, "What is in it", F["h"])
for i, (a, b) in enumerate(CONTENTS):
    fmt = F["tb"] if i % 2 else F["t"]
    ws.write_url(r + 1 + i, 0, f"internal:'{a}'!A1", fmt, a)
    ws.write(r + 1 + i, 1, b, fmt)


# ===========================================================================
# 2. Master
# ===========================================================================
MASTER = rows(DATA / "results_master.csv")
ws = book.add_worksheet("Master")
ws.write(0, 0, "results_master.csv — every accepted ModalPINN metric", F["title"])
ws.write(1, 0, f"{len(MASTER)} rows. value_type 'raw_metric' is measured; the rest are derived from those. "
               "Use the filter arrows on row 4.", F["sub"])
HEAD = ["analysis", "arm", "section", "metric", "region", "value", "unit",
        "reference", "value_type", "source file", "status", "notes"]
data = [[r_["analysis_id"], r_["arm_id"], r_["section"], r_["metric"], r_["region"],
         num(r_["value"]), r_["unit"], r_["reference"], r_["value_type"],
         pathlib.Path(r_["source"]).name, r_["status"], r_["notes"]] for r_ in MASTER]
end = table(ws, data, HEAD, start=3,
            widths=[9, 34, 20, 40, 14, 14, 14, 22, 18, 38, 10, 46],
            fmts={5: F["n"]})
ws.conditional_format(4, 5, end, 5,
                      {"type": "3_color_scale", "min_color": "#2E7D5B",
                       "mid_color": "#F0E2B6", "max_color": "#A33A3A"})
ws.set_row(3, 30)


# ===========================================================================
# 3. Headline
# ===========================================================================
def master_get(aid, arm, metric, region):
    for r_ in MASTER:
        if (r_["analysis_id"] == aid and r_["arm_id"] == arm
                and r_["metric"] == metric and r_["region"] == region):
            return float(r_["value"])
    return None


gvals = rows(GAPPY / "gappy_chapter4_values.csv")
gappy_clean = next(float(r_["far_core_v_relative_L2"]) for r_ in gvals
                   if r_["figure_id"] == "G02" and "Gappy" in r_["method"])
a07 = json.loads((ANA / "a07_centreline_wavelength.json").read_text())

ws = book.add_worksheet("Headline")
ws.hide_gridlines(2)
ws.write(0, 0, "The numbers the argument rests on", F["title"])
ws.write(1, 0, "Each is read from the file named in the last column. Nothing here is retyped by hand.",
         F["sub"])

HEADLINE = [
    ("Wall pressure alone cannot recover the wake", "", "", ""),
    ("  far-core v relative L2, pressure only",
     master_get("A01", "pressure_only_physics", "field.v.rel_L2", "far-core"),
     "worse than predicting zero", "results_master.csv (A01)"),
    ("  far-core v1 correlation, pressure only",
     master_get("A01", "pressure_only_physics", "v1_mode.v.corr", "far-core"),
     "no phase agreement", "results_master.csv (A01)"),
    ("  near-wake v relative L2, pressure only",
     master_get("A01", "pressure_only_physics", "field.v.rel_L2", "near-wake"),
     "also worse than zero, close to the body", "results_master.csv (A01)"),
    ("", "", "", ""),
    ("The information is present in the same 32 taps", "", "", ""),
    ("  far-core v relative L2, Gappy POD, 32 taps", gappy_clean,
     "linear method, same measurements", "gappy_metrics.csv"),
    ("  far-core v relative L2, dense observations",
     master_get("A01", "dense_observations", "field.v.rel_L2", "far-core"),
     "the network's dense-observation reference", "results_master.csv (A01)"),
    ("  far-core v relative L2, velocity probes",
     master_get("A01", "pressure_and_velocity_probes_physics", "field.v.rel_L2", "far-core"),
     "swap taps for 40 probes, change nothing else", "results_master.csv (A01)"),
    ("", "", "", ""),
    ("A structural prior recovers it, and we can say who did what", "", "", ""),
    ("  far-core v1 relative L2, prior alone",
     master_get("A04", "karman_prior_only", "v1_mode.v.rel_L2", "far-core"),
     "closed form, no network", "results_master.csv (A04)"),
    ("  far-core v1 relative L2, prior + network",
     master_get("A04", "pressure_only_physics_karman_prior", "v1_mode.v.rel_L2", "far-core"),
     "the network adds nothing in the far field", "results_master.csv (A04)"),
    ("  near-cylinder v1 relative L2, prior alone",
     master_get("A04", "karman_prior_only", "v1_mode.v.rel_L2", "near-cylinder"), "",
     "results_master.csv (A04)"),
    ("  near-cylinder v1 relative L2, prior + network",
     master_get("A04", "pressure_only_physics_karman_prior", "v1_mode.v.rel_L2", "near-cylinder"),
     "the network earns its place where the sensors are", "results_master.csv (A04)"),
    ("", "", "", ""),
    ("The mechanism: amplitude without wavelength", "", "", ""),
    ("  centreline wavelength / D, DNS", a07["dns_reference"]["wavelength_D"],
     f"fit R2 = {a07['dns_reference']['fit_r2']:.3f}", "a07_centreline_wavelength.json"),
    ("  centreline wavelength / D, prior-assisted",
     a07["arms"]["pressure_only_physics_karman_prior"]["wavelength_D"],
     f"fit R2 = {a07['arms']['pressure_only_physics_karman_prior']['fit_r2']:.3f}",
     "a07_centreline_wavelength.json"),
    ("  centreline wavelength / D, wake-biased grid",
     a07["arms"]["wake_biased_grid_collocation"]["wavelength_D"],
     "phase barely advances: a standing disturbance",
     "a07_centreline_wavelength.json"),
    ("  centreline wavelength / D, pressure only",
     a07["arms"]["pressure_only_physics"]["wavelength_D"],
     f"fit R2 = {a07['arms']['pressure_only_physics']['fit_r2']:.3f} — NOT a wavelength, "
     f"there is no coherent phase to fit", "a07_centreline_wavelength.json"),
]
ws.set_column(0, 0, 48); ws.set_column(1, 1, 13)
ws.set_column(2, 2, 58); ws.set_column(3, 3, 34)
r = 3
for c, h in enumerate(["quantity", "value", "what it means", "read from"]):
    ws.write(r, c, h, F["h"])
for i, (a, b, c, d) in enumerate(HEADLINE, start=r + 1):
    if b == "" and c == "" and a:
        ws.write(i, 0, a, F["sec"])
        for cc in range(1, 4):
            ws.write_blank(i, cc, None, F["sec"])
    elif not a:
        continue
    else:
        ws.write(i, 0, a, F["t"])
        ws.write(i, 1, b, F["n"] if isinstance(b, float) else F["t"])
        ws.write(i, 2, c, F["wrap"])
        ws.write(i, 3, d, F["t"])
ws.set_row(3, 30)


# ===========================================================================
# 4. Far core v — the pivot everyone reads
# ===========================================================================
REGIONS = ["near-cylinder", "near-wake", "far-wake", "far-core", "other", "whole-domain"]


def pivot_sheet(title, subtitle, metric, arms, extra=()):
    ws = book.add_worksheet(title)
    ws.write(0, 0, title, F["title"])
    ws.write(1, 0, subtitle, F["sub"])
    ws.set_row(1, 30)
    head = ["arm"] + REGIONS + [e[0] for e in extra]
    data = []
    for aid, arm, label in arms:
        row = [label]
        for reg in REGIONS:
            row.append(master_get(aid, arm, metric, reg))
        for _, fn in extra:
            row.append(fn(aid, arm))
        data.append(row)
    ws.set_column(0, 0, 44)
    for c in range(1, len(head)):
        ws.set_column(c, c, 15)
    end = table(ws, data, head, start=3, fmts={i: F["n"] for i in range(1, len(REGIONS) + 1)})
    ws.conditional_format(4, 1, end, len(REGIONS),
                          {"type": "3_color_scale", "min_color": "#2E7D5B",
                           "mid_color": "#F0E2B6", "max_color": "#A33A3A"})
    return ws


pivot_sheet(
    "Far core v",
    "Transverse-velocity relative L2 by region. 1.0 means no better than predicting zero. "
    "Green is good, red is at or beyond the zero-prediction floor.",
    "field.v.rel_L2",
    [("A01", "pressure_only_physics", "pressure only + physics (32 taps)"),
     ("A01", "pressure_and_velocity_probes_physics", "pressure + 40 velocity probes"),
     ("A01", "dense_observations", "dense observations (ceiling, non-converged)"),
     ("A04", "karman_prior_only", "Karman prior alone (closed form)"),
     ("A04", "pressure_only_physics_karman_prior", "pressure + physics + Karman prior"),
     ("A02", "pressure_only_physics_8_taps", "8 taps"),
     ("A02", "pressure_only_physics_16_taps", "16 taps"),
     ("A03", "wake_biased_random_collocation", "wake-biased random collocation"),
     ("A03", "wake_biased_grid_collocation", "wake-biased grid collocation")])

pivot_sheet(
    "Prior attribution",
    "First-harmonic v1 relative L2. The prior supplies the far field; the network adds near the body.",
    "v1_mode.v.rel_L2",
    [("A04", "karman_prior_only", "Karman prior alone"),
     ("A04", "pressure_only_physics", "network alone (pressure only)"),
     ("A04", "pressure_only_physics_karman_prior", "prior + network")])

pivot_sheet(
    "Tap count",
    "Does adding wall-pressure taps help? Transverse-velocity relative L2 by region.",
    "field.v.rel_L2",
    [("A02", "pressure_only_physics_8_taps", "8 taps"),
     ("A02", "pressure_only_physics_16_taps", "16 taps"),
     ("A02", "pressure_only_physics_32_taps", "32 taps")])

effort = {r_["arm_id"]: float(r_["value"]) for r_ in MASTER
          if r_["metric"] == "training_effort.lbfgs_evals"}
pivot_sheet(
    "Collocation",
    "Uniform vs wake-biased interior sampling. The arms are NOT effort-matched: the "
    "L-BFGS evaluation counts are in the last column. Evaluation count is not an "
    "accuracy proxy, so the comparison is descriptive rather than causal.",
    "v1_mode.v.rel_L2",
    [("A03", "uniform_collocation", "uniform sampling"),
     ("A03", "wake_biased_random_collocation", "wake-biased random"),
     ("A03", "wake_biased_grid_collocation", "wake-biased grid")],
    extra=[("L-BFGS evals", lambda aid, arm: effort.get(arm))])

pivot_sheet(
    "Noise",
    "Prior-assisted reconstruction under pressure noise. First-harmonic v1 relative L2. "
    "One seed per level: read direction, not magnitude.",
    "v1_mode.v.rel_L2",
    [("A06", "prior_physics_noise_00pct", "clean"),
     ("A06", "prior_physics_noise_01pct", "1 % noise"),
     ("A06", "prior_physics_noise_05pct", "5 % noise"),
     ("A06", "prior_physics_noise_10pct", "10 % noise")])


# ===========================================================================
# 5. Gappy POD
# ===========================================================================
ws = book.add_worksheet("Gappy POD")
ws.write(0, 0, "Gappy POD — the linear baseline", F["title"])
gcfg = json.loads((GAPPY / "gappy_configuration.json").read_text())
gsum = json.loads((GAPPY / "gappy_summary.json").read_text())
ws.write(1, 0, "Same 32 uniformly spaced wall-pressure taps as the ModalPINN arms, with a rank-6 POD "
               "basis supplying the spatial structure. " + gsum.get("interpretation", ""), F["sub"])
ws.set_row(1, 44)

ws.write(3, 0, "Report-facing values", F["sec"])
gv = [[r_["figure_id"], r_["method"], num(r_["noise_percent"]) if r_["noise_percent"] else "clean",
       float(r_["far_core_v_relative_L2"]), r_["source"]] for r_ in gvals]
table(ws, gv, ["figure", "method", "noise %", "far-core v rel L2", "source"], start=4,
      widths=[9, 38, 10, 20, 52], fmts={3: F["n"]})

start = 4 + len(gv) + 3
ws.write(start - 1, 0, "All metrics", F["sec"])
gm = [[num(r_["noise_sigma"]), r_["quantity"], r_["variable"], r_["region"],
       r_["metric"], num(r_["value"])] for r_ in rows(GAPPY / "gappy_metrics.csv")]
table(ws, gm, ["noise sigma", "quantity", "variable", "region", "metric", "value"],
      start=start, fmts={5: F["n"], 0: F["n"]})


# ===========================================================================
# 6. Wavelengths
# ===========================================================================
ws = book.add_worksheet("Wavelengths")
ws.write(0, 0, "Centreline wavelength — the mechanism evidence", F["title"])
w = a07["fit_window"]
ws.write(1, 0, f"Phase of the first harmonic along the wake centreline, fitted over "
               f"{w['x_min']} <= x/D <= {w['x_max']}, |y| <= {w['y_halfwidth']}. A travelling "
               f"vortex street advances phase steadily; a standing disturbance does not, so its "
               f"fitted wavelength diverges. Read the R2 column first: a poor fit means there is "
               f"no wavelength to report, not a long one.", F["sub"])
ws.set_row(1, 58)
LABEL = {
    "dns_reference": "DNS reference",
    "pressure_and_velocity_probes": "pressure + velocity probes",
    "pressure_only_physics_karman_prior": "pressure + physics + Karman prior",
    "wake_biased_grid_collocation": "wake-biased grid collocation",
    "wake_biased_random_collocation": "wake-biased random collocation",
    "pressure_only_physics": "pressure only + physics",
}
entries = {"dns_reference": a07["dns_reference"], **a07["arms"]}
wdata = []
for k, e in entries.items():
    wdata.append([LABEL.get(k, k), e["wavelength_D"], e["phase_gradient_rad_per_D"],
                  e["fit_r2"], e["n_bins"],
                  "usable" if e["fit_r2"] >= 0.9 else "NO coherent phase — do not quote as a wavelength"])
wdata.sort(key=lambda r_: r_[1])
table(ws, wdata, ["arm", "wavelength / D", "phase gradient (rad/D)", "fit R2", "bins", "reading"],
      start=3, widths=[40, 16, 22, 11, 8, 52],
      fmts={1: F["n3"], 2: F["n"], 3: F["n3"], 4: F["int"], 5: F["wrap"]})


# ===========================================================================
# 7. Termination census
# ===========================================================================
TR = np.load(DATA / "census_traces.npz")
names = sorted(k[len("trace__"):] for k in TR.files if k.startswith("trace__"))


def exit_blow(A):
    g = A[:, 2]
    last = len(A) - 1
    base = (np.median(g[max(0, last - 600):last - 100]) if last > 150
            else np.median(g[:max(1, last - 10)]))
    return float((np.median(g[last - 50:last + 1]) / base) if last > 60 else g[last] / base)


ws = book.add_worksheet("Termination census")
ws.write(0, 0, "How all 17 training runs stopped", F["title"])
ws.write(1, 0, "Every run terminates on SciPy's relative-reduction test after a failed line search. "
               "At ftol = 1e-12 that test sits below float32 loss resolution, so it fires at the "
               "first completely failed line search. Failures are far likelier during a gradient "
               "spike, so runs that stopped early stopped mid-instability.", F["sub"])
ws.set_row(1, 44)
cdata = []
for n in names:
    A = TR[f"trace__{n}"]
    cdata.append([n, int(A[-1, 0]), len(A), float(A[-1, 1]), float(A[-1, 2]),
                  exit_blow(A), int(TR[f"nwarn__{n}"]),
                  "line-search warning" if int(TR[f"nwarn__{n}"]) else "hit evaluation cap"])
cdata.sort(key=lambda r_: -r_[1])
end = table(ws, cdata,
            ["run", "stopped at iteration", "logged iterates", "final loss",
             "final |proj g|", "exit gradient elevation", "line-search warnings", "how it ended"],
            start=3, widths=[36, 18, 15, 15, 15, 20, 18, 24],
            fmts={1: F["int"], 2: F["int"], 3: F["n"], 4: F["n"], 5: F["n3"], 6: F["int"]})
ws.conditional_format(4, 5, end, 5, {"type": "data_bar", "bar_color": "#B5651D"})


# ===========================================================================
# 8. Figures
# ===========================================================================
FIGURE_NOTES = [
    ("F_setup", "The physical problem: cylinder, freestream, domain, Re and St", "code/figures/fig_setup.py"),
    ("F_dataset", "The reference flow and the evidence for the shedding frequency", "code/figures/fig_dataset.py"),
    ("F00_evaluation_regions", "Where every regional metric is computed", "code/figures/fig00_evaluation_regions.py"),
    ("F00a_probe_locations", "Velocity-probe positions", "code/figures/fig00a_probe_locations.py"),
    ("F00b_tap_layout", "The nested 8 / 16 / 32 wall-pressure tap sets", "code/figures/fig00b_tap_layout.py"),
    ("F01_prior_attribution_fields", "CFD, prior and both reconstructions on one snapshot", "code/figures/fig01_prior_attribution_fields.py"),
    ("F02_prior_attribution", "What the prior supplies vs what the network adds, by region", "code/figures/fig02_prior_attribution.py"),
    ("F02b_upstream_artefact", "The upstream ratio is a near-zero denominator, not a large error", "code/figures/fig02b_upstream_artefact.py"),
    ("F03_information_comparison", "Taps vs velocity probes vs dense observation", "code/figures/fig03_information_comparison.py"),
    ("F04a_tap_count", "8 / 16 / 32 taps: no recovery of the wake", "code/figures/fig04a_tap_count.py"),
    ("F04b_collocation_strategy", "Wake-biased collocation moves error, it does not remove it", "code/figures/fig04b_collocation_strategy.py"),
    ("F04c_prior_collocation", "Wake-biased sampling costs the learned near-wake contribution", "code/figures/fig04c_prior_collocation.py"),
    ("F04d_pressure_noise", "Far-field invariance under noise belongs to the prior", "code/figures/fig04d_pressure_noise.py"),
    ("F05_information_structure", "The cross-method synthesis", "code/figures/fig05_information_structure.py"),
    ("F06_noise_tradeoff", "Accuracy against noise robustness, both methods", "code/figures/fig06_noise_tradeoff.py"),
    ("F_arm1_rerun_breakdown", "The re-run is bit-identical for 99 iterations, then diverges", "code/figures/fig07_arm1_rerun_breakdown.py"),
    ("F_termination_anatomy", "All 17 runs die the same way", "code/figures/fig08_termination_anatomy.py"),
    ("G01_clean_reconstruction", "Gappy POD reconstruction on one snapshot", "code/figures/gappy/fig01_clean_reconstruction.py"),
    ("G02_clean_method_comparison", "Gappy POD against the ModalPINN arms", "code/figures/gappy/fig02_clean_method_comparison.py"),
    ("G03_noise_sensitivity", "Gappy POD under matched pressure noise", "code/figures/gappy/fig03_noise_sensitivity.py"),
]
ws = book.add_worksheet("Figures")
ws.write(0, 0, "Figures, and the script that draws each", F["title"])
ws.write(1, 0, "Every script here runs from this folder alone: python3 <script>. "
               "Output goes to figures/.", F["sub"])
present = {p.stem for p in (ROOT / "figures").glob("*.png")}
fdata = [[n, d, s, "present" if n in present else "MISSING"] for n, d, s in FIGURE_NOTES]
table(ws, fdata, ["figure", "what it shows", "generator", "file"], start=3,
      widths=[32, 62, 46, 11], fmts={1: F["wrap"]})


# ===========================================================================
# 8b. Verified results — the enriched table, with configuration columns
# ===========================================================================
VR = DATA / "verified_results.csv"
if VR.exists():
    vr = rows(VR)
    ws = book.add_worksheet("All results")
    ws.write(0, 0, "verified_results.csv — every metric, with its configuration", F["title"])
    ws.write(1, 0, f"{len(vr)} rows: all {sum(1 for r_ in vr if r_.get('analysis_id','').startswith('A'))} "
                   f"ModalPINN metrics and all "
                   f"{sum(1 for r_ in vr if not r_.get('analysis_id','').startswith('A'))} Gappy POD metrics "
                   "in one table. Unlike the Master sheet this carries the configuration of each arm "
                   "(taps, collocation, prior, noise), so you can filter by setup rather than by arm name.",
             F["sub"])
    ws.set_row(1, 44)
    head = list(vr[0].keys())
    data = [[num(r_[h]) for h in head] for r_ in vr]
    widths = [14 if h not in ("notes", "source", "method") else 40 for h in head]
    end = table(ws, data, head, start=3, widths=widths,
                fmts={head.index("value"): F["n"]} if "value" in head else None)
    if "value" in head:
        c = head.index("value")
        ws.conditional_format(4, c, end, c,
                              {"type": "3_color_scale", "min_color": "#2E7D5B",
                               "mid_color": "#F0E2B6", "max_color": "#A33A3A"})

# ===========================================================================
# 8c. The time-origin correction, before and after
# ===========================================================================
PC = DATA / "v1_phase_correction_audit.csv"
if PC.exists():
    pc = rows(PC)
    ws = book.add_worksheet("Phase correction")
    ws.write(0, 0, "The 2026-08-29 time-origin correction", F["title"])
    ws.write(1, 0, "The CFD first harmonic was fitted in shifted time (tau = t - 400) but the network "
                   "defines its mode in absolute time, so the two were compared on different clocks. "
                   "Rotating the fitted coefficient by exp(-i*omega*t0) fixes it. Amplitude ratios and "
                   "correlations are untouched; every phase below moves by exactly -16.6290 degrees. "
                   "These are the superseded values — the report and every other sheet use the corrected ones.",
             F["sub"])
    ws.set_row(1, 58)
    head = list(pc[0].keys())
    data = [[num(r_[h]) for h in head] for r_ in pc]
    widths = [14 if h not in ("correction_reason", "verification", "method") else 46 for h in head]
    numcols = {head.index(h): F["n"] for h in
               ("legacy_window_value", "corrected_absolute_value", "absolute_change") if h in head}
    table(ws, data, head, start=3, widths=widths, fmts=numcols)

# ===========================================================================
# 9. Verification
# ===========================================================================
ws = book.add_worksheet("Verification")
ws.write(0, 0, "Verification — code/verify.py", F["title"])
ws.write(1, 0, "Re-run at any time with: python3 code/verify.py. "
               "It checks that every value in results_master.csv matches the analysis file it "
               "came from, that every derived quantity recomputes from the raw ones, and that "
               "the GappyPOD bundle agrees with itself and with the ModalPINN values it quotes.",
         F["sub"])
ws.set_row(1, 44)
ws.set_column(0, 0, 12)
ws.set_column(1, 1, 130)
proc = subprocess.run([sys.executable, str(ROOT / "code" / "verify.py")],
                      capture_output=True, text=True)
lines = [ln for ln in proc.stdout.splitlines() if ln.strip() and not ln.startswith("=")]
ws.write(3, 0, "result", F["h"]); ws.write(3, 1, "check", F["h"])
for i, ln in enumerate(lines, start=4):
    if ln.startswith("PASS"):
        ws.write(i, 0, "PASS", F["good"]); ws.write(i, 1, ln[6:], F["mono"])
    elif ln.startswith("FAIL"):
        ws.write(i, 0, "FAIL", F["bad"]); ws.write(i, 1, ln[6:], F["mono"])
    else:
        ws.write(i, 0, "", F["t"]); ws.write(i, 1, ln, F["mono"])

book.close()
print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
print(f"  {len(book.worksheets())} sheets, {len(MASTER)} master rows")
