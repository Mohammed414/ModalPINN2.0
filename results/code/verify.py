"""Check every number in this folder against the file it came from.

Nothing here recomputes a reconstruction: re-deriving the metrics themselves
needs the TensorFlow checkpoints, which are not part of the bundle.  What this
does check is that the chain of files is internally consistent --

    metrics JSON  ->  per-analysis summary CSV  ->  results_master.csv

-- that the derived quantities (percentage changes, gains, contributions) are
reproducible from the raw ones, that the GappyPOD bundle agrees with itself and
with the ModalPINN values it quotes, and that the standalone A07 and
termination-census numbers are internally consistent.

Run:  python3 code/verify.py
"""
from __future__ import annotations

import csv
import json
import math
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ANA = DATA / "analysis"
GAPPY = DATA / "gappy"

TOL = 1e-9              # values are copied, not recomputed, so this is exact
REL_TOL = 1e-6          # for quantities we recompute in floating point

passes: list[str] = []
failures: list[str] = []


def ok(msg: str) -> None:
    passes.append(msg)


def bad(msg: str) -> None:
    failures.append(msg)


def close(a: float, b: float, rel: float = REL_TOL) -> bool:
    if a == b:
        return True
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


def load_json(p: pathlib.Path) -> dict:
    return json.loads(p.read_text())


def rows(p: pathlib.Path) -> list[dict]:
    with p.open() as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# 1. metrics JSON  ->  per-analysis summary CSV
# ---------------------------------------------------------------------------
# Each summary CSV is a flattening of its metrics JSON.  Walk the JSON and
# confirm every summary row reproduces the value it was flattened from.

JSON_FOR = {
    "a01_information_comparison_summary.csv": ("a01_information_comparison_metrics.json", "models"),
    "a02_tap_count_summary.csv":              ("a02_tap_count_metrics.json", "models"),
    "a03_collocation_summary.csv":            ("a03_collocation_metrics.json", "models"),
    "a05_prior_collocation_summary.csv":      ("a05_prior_collocation_metrics.json", "models"),
    "a06_pressure_noise_summary.csv":         ("a06_pressure_noise_metrics.json", "models"),
}
GROUP_KEY = {
    "field": "field_metrics",
    "v1_mode": "v1_mode_metrics",
    "v1_mode_window_convention": "v1_mode_metrics_window_convention",
}


def json_value(entry: dict, group: str, region: str, quantity: str, metric: str):
    block = entry.get(GROUP_KEY.get(group, group))
    if block is None:
        return None
    reg = block.get(region)
    if reg is None:
        return None
    if group == "field":
        # field_metrics[region][quantity] is the rel_L2 value itself
        return reg.get(quantity) if metric == "rel_L2" else None
    return reg.get(metric)


for csv_name, (json_name, container) in JSON_FOR.items():
    cpath, jpath = ANA / csv_name, ANA / json_name
    if not (cpath.exists() and jpath.exists()):
        bad(f"[1] missing {csv_name} or {json_name}")
        continue
    doc = load_json(jpath)
    models = doc[container]
    checked = mismatched = unresolved = 0
    for r in rows(cpath):
        entry = models.get(r["method"])
        if entry is None:
            unresolved += 1
            continue
        v = json_value(entry, r["metric_group"], r["region"], r["quantity"], r["metric"])
        if v is None:
            unresolved += 1
            continue
        checked += 1
        if not close(float(r["value"]), float(v), TOL):
            mismatched += 1
            bad(f"[1] {csv_name}: {r['method']}/{r['region']}/{r['metric_group']}."
                f"{r['quantity']}.{r['metric']} CSV={r['value']} JSON={v}")
    if unresolved:
        bad(f"[1] {csv_name}: {unresolved} rows could not be located in {json_name}")
    if not mismatched and not unresolved:
        ok(f"[1] {csv_name}: {checked} values match {json_name}")

# A04 keeps the two trained models and the prior in separate files.
a04_doc = load_json(ANA / "a04_prior_attribution_metrics.json")
a04_prior = load_json(ANA / "a04_prior_only_metrics.json")
A04_SOURCES = {
    "prior_only": a04_prior,
    "arm1_baseline": a04_doc["models"]["arm1_baseline"],
    "arm15_v1_radial_trust": a04_doc["models"]["arm15_v1_radial_trust"],
}
checked = mismatched = unresolved = 0
for r in rows(ANA / "a04_prior_attribution_summary.csv"):
    entry = A04_SOURCES.get(r["method"])
    if entry is None:
        unresolved += 1
        continue
    v = json_value(entry, r["metric_group"], r["region"], r["quantity"], r["metric"])
    if v is None:
        unresolved += 1
        continue
    checked += 1
    if not close(float(r["value"]), float(v), TOL):
        mismatched += 1
        bad(f"[1] a04 summary: {r['method']}/{r['region']}/{r['metric_group']}."
            f"{r['quantity']}.{r['metric']} CSV={r['value']} JSON={v}")
if unresolved:
    bad(f"[1] a04_prior_attribution_summary.csv: {unresolved} rows unresolved")
if not mismatched and not unresolved:
    ok(f"[1] a04_prior_attribution_summary.csv: {checked} values match their JSON")


# ---------------------------------------------------------------------------
# 2. summary CSVs  ->  results_master.csv (the raw_metric rows)
# ---------------------------------------------------------------------------
MASTER = rows(DATA / "results_master.csv")

# arm_id in the master table vs the `method` label used inside each summary CSV
ARM_ALIAS = {
    ("A04", "karman_prior_only"): "prior_only",
    ("A04", "pressure_only_physics"): "arm1_baseline",
    ("A04", "pressure_only_physics_karman_prior"): "arm15_v1_radial_trust",
    ("A05", "prior_physics_uniform_collocation"): "prior_uniform_collocation",
    ("A05", "prior_physics_wake_biased_grid"): "prior_wake_biased_grid",
    ("A06", "prior_physics_noise_00pct"): "prior_noise_00pct",
    ("A06", "prior_physics_noise_01pct"): "prior_noise_01pct",
    ("A06", "prior_physics_noise_05pct"): "prior_noise_05pct",
    ("A06", "prior_physics_noise_10pct"): "prior_noise_10pct",
}
SUMMARY_FOR = {
    "A01": "a01_information_comparison_summary.csv",
    "A02": "a02_tap_count_summary.csv",
    "A03": "a03_collocation_summary.csv",
    "A04": "a04_prior_attribution_summary.csv",
    "A05": "a05_prior_collocation_summary.csv",
    "A06": "a06_pressure_noise_summary.csv",
}

lookup: dict[tuple, float] = {}
for aid, name in SUMMARY_FOR.items():
    for r in rows(ANA / name):
        key = (aid, r["method"], r["region"],
               f"{r['metric_group']}.{r['quantity']}.{r['metric']}")
        lookup[key] = float(r["value"])

checked = mismatched = unresolved = 0
for r in MASTER:
    if r["value_type"] != "raw_metric":
        continue
    aid = r["analysis_id"]
    method = ARM_ALIAS.get((aid, r["arm_id"]), r["arm_id"])
    key = (aid, method, r["region"], r["metric"])
    if key not in lookup:
        unresolved += 1
        bad(f"[2] no summary row for {aid}/{r['arm_id']}/{r['region']}/{r['metric']}")
        continue
    checked += 1
    if not close(float(r["value"]), lookup[key], TOL):
        mismatched += 1
        bad(f"[2] {aid}/{r['arm_id']}/{r['region']}/{r['metric']}: "
            f"master={r['value']} summary={lookup[key]}")
if not mismatched and not unresolved:
    ok(f"[2] results_master.csv: all {checked} raw_metric rows match their summary CSV")


# ---------------------------------------------------------------------------
# 3. derived rows recompute from the raw ones
# ---------------------------------------------------------------------------
def raw(aid: str, arm: str, region: str, metric: str):
    method = ARM_ALIAS.get((aid, arm), arm)
    return lookup.get((aid, method, region, metric))


# --- A04 network_gain_over_karman_prior: reductions against the prior alone
n_ok = n_bad = 0
for r in MASTER:
    if r["analysis_id"] != "A04" or r["value_type"] != "derived_change":
        continue
    region, metric, got = r["region"], r["metric"], float(r["value"])
    base_arm, hyb = "karman_prior_only", "pressure_only_physics_karman_prior"
    if metric.endswith(".rel_L2_reduction_pct"):
        stem = metric.rsplit(".", 1)[0] + ".rel_L2"
        b, h = raw("A04", base_arm, region, stem), raw("A04", hyb, region, stem)
        want = 100.0 * (b - h) / b
    elif metric.endswith(".correlation_gain_pp"):
        b = raw("A04", base_arm, region, "v1_mode.v.corr")
        h = raw("A04", hyb, region, "v1_mode.v.corr")
        want = 100.0 * (h - b)
    elif metric.endswith(".amplitude_abs_error_reduction_pct"):
        b = abs(raw("A04", base_arm, region, "v1_mode.v.amp_ratio") - 1.0)
        h = abs(raw("A04", hyb, region, "v1_mode.v.amp_ratio") - 1.0)
        want = 100.0 * (b - h) / b
    elif metric.endswith(".phase_abs_error_reduction_pct"):
        b = abs(raw("A04", base_arm, region, "v1_mode.v.phase_deg"))
        h = abs(raw("A04", hyb, region, "v1_mode.v.phase_deg"))
        want = 100.0 * (b - h) / b
    else:
        continue
    if close(got, want, 1e-6):
        n_ok += 1
    else:
        n_bad += 1
        bad(f"[3] A04 {metric} @ {region}: stored {got:.6f}, recomputed {want:.6f}")
if n_bad == 0 and n_ok:
    ok(f"[3] A04: all {n_ok} derived_change rows recompute from the raw metrics")

# --- A05/A06 learned_contribution: prior error minus the arm error in the
#     same region. Positive means the network improved on the analytical
#     prior. Re-derive all 36 stored rows rather than trusting their labels.
learned = [r for r in MASTER
           if r["value_type"] == "derived_metric"
           and r["metric"].endswith("learned_contribution")]
n_ok = n_bad = 0
for r in learned:
    arm_error = raw(r["analysis_id"], r["arm_id"], r["region"], "v1_mode.v.rel_L2")
    prior_error = raw("A04", "karman_prior_only", r["region"], "v1_mode.v.rel_L2")
    want = prior_error - arm_error
    got = float(r["value"])
    if close(got, want, 1e-12):
        n_ok += 1
    else:
        n_bad += 1
        bad(f"[3] {r['analysis_id']}/{r['arm_id']} learned contribution @ "
            f"{r['region']}: stored {got:.12g}, recomputed {want:.12g}")
if n_bad == 0 and n_ok == 36:
    ok("[3] A05/A06: all 36 learned-contribution rows recompute exactly")
elif n_ok + n_bad != 36:
    bad(f"[3] expected 36 learned-contribution rows, found {n_ok + n_bad}")

# --- cross-analysis consistency: arm 15 is evaluated in A04, A05 and A06.
#     Those three independent evaluations must agree exactly.
n_ok = n_bad = 0
SHARED = [("A05", "prior_physics_uniform_collocation"),
          ("A06", "prior_physics_noise_00pct")]
for region in ["near-cylinder", "near-wake", "far-wake", "far-core", "other", "whole-domain"]:
    for metric in ["v1_mode.v.rel_L2", "v1_mode.v.amp_ratio", "v1_mode.v.corr",
                   "v1_mode.v.phase_deg", "field.u.rel_L2", "field.v.rel_L2", "field.p.rel_L2"]:
        ref = raw("A04", "pressure_only_physics_karman_prior", region, metric)
        if ref is None:
            continue
        for aid, arm in SHARED:
            got = raw(aid, arm, region, metric)
            if got is None:
                continue
            if close(got, ref, TOL):
                n_ok += 1
            else:
                n_bad += 1
                bad(f"[3] arm 15 disagrees between A04 and {aid} at {region}/{metric}: "
                    f"{ref} vs {got}")
if n_bad == 0 and n_ok:
    ok(f"[3] arm 15 is evaluated in A04, A05 and A06: all {n_ok} shared values agree exactly")

# --- A03 confound audit: the L-BFGS evaluation counts behind the effort caveat
effort = {r["arm_id"]: float(r["value"]) for r in MASTER
          if r["value_type"] == "confound_audit" and r["metric"] == "training_effort.lbfgs_evals"}
if effort:
    u = effort.get("uniform_collocation")
    ratios = {k: v / u for k, v in effort.items() if k != "uniform_collocation"}
    ok(f"[3] A03 effort audit: uniform {u:,.0f} evals; "
       + ", ".join(f"{k.replace('_collocation','')} {v:,.0f} ({v/u:.1f}x)"
                   for k, v in effort.items() if k != "uniform_collocation"))
    if min(ratios.values()) < 1.0:
        bad("[3] A03: an arm the caveat calls better-trained is not actually better-trained")


# ---------------------------------------------------------------------------
# 4. the GappyPOD bundle agrees with itself
# ---------------------------------------------------------------------------
g_metrics = rows(GAPPY / "gappy_metrics.csv")
g_values = rows(GAPPY / "gappy_chapter4_values.csv")
g_summary = load_json(GAPPY / "gappy_summary.json")

# far-core v relative L2 at each noise level, straight from metrics.csv
far_core = {}
for r in g_metrics:
    if r["quantity"] == "field" and r["variable"] == "v" and \
       r["region"] == "far-core" and r["metric"] == "relative_L2":
        far_core[float(r["noise_sigma"])] = float(r["value"])

n_ok = n_bad = 0
for sigma, v in g_summary["noise_far_core_v"].items():
    s = float(sigma)
    if s not in far_core:
        bad(f"[4] gappy summary lists sigma={s} with no matching metrics.csv row")
        n_bad += 1
    elif close(float(v), far_core[s], TOL):
        n_ok += 1
    else:
        n_bad += 1
        bad(f"[4] gappy sigma={s}: summary {v} vs metrics.csv {far_core[s]}")
if not n_bad:
    ok(f"[4] gappy_summary.json: all {n_ok} far-core noise values match gappy_metrics.csv")

if close(float(g_summary["clean"]["far_core_v"]), far_core.get(0.0, float("nan")), TOL):
    ok("[4] gappy_summary.json clean far-core v matches the sigma=0 metrics row")
else:
    bad("[4] gappy_summary.json clean far-core v does not match the sigma=0 metrics row")

# the report-facing values file
NOISE_SIGMA = {0: 0.0, 1: 0.00047265, 5: 0.0023633, 10: 0.0047265}
n_ok = n_bad = 0
for r in g_values:
    if "Gappy POD" not in r["method"]:
        continue
    val = float(r["far_core_v_relative_L2"])
    pct = r["noise_percent"].strip()
    sigma = NOISE_SIGMA[int(pct)] if pct else 0.0
    if sigma in far_core and close(val, far_core[sigma], TOL):
        n_ok += 1
    else:
        n_bad += 1
        bad(f"[4] chapter4_values row ({r['figure_id']}, noise {pct or '0'}%): "
            f"{val} not found in gappy_metrics.csv at sigma={sigma}")
if not n_bad:
    ok(f"[4] gappy_chapter4_values.csv: all {n_ok} GappyPOD rows trace to gappy_metrics.csv")


# ---------------------------------------------------------------------------
# 5. the GappyPOD values file quotes ModalPINN correctly
# ---------------------------------------------------------------------------
MASTER_FAR_CORE = {
    (r["analysis_id"], r["arm_id"]): float(r["value"])
    for r in MASTER
    if r["metric"] == "field.v.rel_L2" and r["region"] == "far-core"
    and r["value_type"] == "raw_metric"
}
CLAIMED = {
    "Pressure only + physics": ("A01", "pressure_only_physics"),
    "Dense observations": ("A01", "dense_observations"),
    "Pressure + physics + Karman prior": ("A04", "pressure_only_physics_karman_prior"),
}
n_ok = n_bad = 0
for r in g_values:
    key = CLAIMED.get(r["method"])
    if key is None:
        continue
    got = float(r["far_core_v_relative_L2"])
    want = MASTER_FAR_CORE.get(key)
    if want is None:
        bad(f"[5] chapter4_values quotes '{r['method']}' with no matching master row")
        n_bad += 1
    elif close(got, want, TOL):
        n_ok += 1
    else:
        n_bad += 1
        bad(f"[5] chapter4_values '{r['method']}': {got} but results_master has {want}")
if not n_bad and n_ok:
    ok(f"[5] gappy_chapter4_values.csv: all {n_ok} ModalPINN cross-references match results_master.csv")


# ---------------------------------------------------------------------------
# 6. A07 centreline wavelengths are internally consistent
# ---------------------------------------------------------------------------
a07 = load_json(ANA / "a07_centreline_wavelength.json")
entries = dict(a07["arms"])
entries["dns_reference"] = a07["dns_reference"]
n_ok = n_bad = 0
for name, e in entries.items():
    want = 2.0 * math.pi / abs(e["phase_gradient_rad_per_D"])
    if close(e["wavelength_D"], want, 1e-9):
        n_ok += 1
    else:
        n_bad += 1
        bad(f"[6] A07 {name}: wavelength {e['wavelength_D']} != 2*pi/|{e['phase_gradient_rad_per_D']}|")
if not n_bad:
    ok(f"[6] A07: all {n_ok} wavelengths equal 2*pi/|dphi/dx| from the same file")

weak = {k: v["fit_r2"] for k, v in entries.items() if v["fit_r2"] < 0.9}
if weak:
    ok("[6] A07 low-quality fits (a wavelength here is not a wavelength): "
       + ", ".join(f"{k} R2={v:.3f}" for k, v in weak.items()))


# ---------------------------------------------------------------------------
# 7. the termination census reproduces from the stored traces
# ---------------------------------------------------------------------------
TR = np.load(DATA / "census_traces.npz")
names = sorted(k[len("trace__"):] for k in TR.files if k.startswith("trace__"))

if len(names) == 17:
    ok(f"[7] census: {len(names)} runs stored")
else:
    bad(f"[7] census: expected 17 runs, found {len(names)}")

warned = [n for n in names if int(TR[f"nwarn__{n}"]) > 0]
if len(warned) == 16 and len(names) == 17:
    quiet = [n for n in names if n not in warned]
    ok(f"[7] census: 16/17 runs print the line-search warning; "
       f"the exception is {quiet[0]}")
else:
    bad(f"[7] census: {len(warned)}/{len(names)} runs warn, expected 16/17")


def exit_blow(A: np.ndarray) -> float:
    g = A[:, 2]
    last = len(A) - 1
    base = (np.median(g[max(0, last - 600):last - 100]) if last > 150
            else np.median(g[:max(1, last - 10)]))
    return float((np.median(g[last - 50:last + 1]) / base) if last > 60 else g[last] / base)


last_it = np.array([TR[f"trace__{n}"][-1, 0] for n in names], dtype=float)
blow = np.array([exit_blow(TR[f"trace__{n}"]) for n in names])

# Spearman rho without scipy: Pearson on ranks
def rank(a):
    order = a.argsort()
    r = np.empty(len(a), dtype=float)
    r[order] = np.arange(len(a), dtype=float)
    # average ties
    for v in np.unique(a):
        m = a == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r

rx, ry = rank(last_it), rank(blow)
rho = float(np.corrcoef(rx, ry)[0, 1])
ok(f"[7] census: Spearman rho(stopping iteration, exit gradient elevation) = {rho:.3f} over n={len(names)}")

early = blow[last_it < 12000]
late = blow[last_it > 20000]
ok(f"[7] census: runs stopping before 12,000 its exit at median {np.median(early):.1f}x "
   f"(range {early.min():.1f}-{early.max():.1f}x); past 20,000 its, median "
   f"{np.median(late):.1f}x ({late.min():.1f}-{late.max():.1f}x)")


# ---------------------------------------------------------------------------
# 8. the master table is well formed
# ---------------------------------------------------------------------------
statuses = {r["status"] for r in MASTER}
if statuses == {"accepted"}:
    ok(f"[8] results_master.csv: all {len(MASTER)} rows are 'accepted'")
else:
    bad(f"[8] results_master.csv carries non-accepted rows: {sorted(statuses)}")

nonfinite = [r for r in MASTER if not math.isfinite(float(r["value"]))]
if nonfinite:
    bad(f"[8] results_master.csv has {len(nonfinite)} non-finite values")
else:
    ok("[8] results_master.csv: every value is finite")

dupes: dict[tuple, int] = {}
for r in MASTER:
    k = (r["analysis_id"], r["arm_id"], r["metric"], r["region"], r["value_type"])
    dupes[k] = dupes.get(k, 0) + 1
rep = {k: v for k, v in dupes.items() if v > 1}
if rep:
    bad(f"[8] results_master.csv has {len(rep)} duplicated (analysis, arm, metric, region) keys")
else:
    ok("[8] results_master.csv: no duplicated analysis/arm/metric/region keys")

missing = [r["source"] for r in MASTER
           if not (ROOT.parents[1] / r["source"]).exists()
           and not (ANA / pathlib.Path(r["source"]).name).exists()]
if missing:
    bad(f"[8] {len(set(missing))} distinct source paths in results_master.csv do not resolve")
else:
    ok("[8] results_master.csv: every cited source file is present in data/analysis/")


# ---------------------------------------------------------------------------
print("=" * 78)
for line in passes:
    print("PASS  " + line)
if failures:
    print()
    for line in failures:
        print("FAIL  " + line)
print("=" * 78)
print(f"{len(passes)} checks passed, {len(failures)} failed")
raise SystemExit(1 if failures else 0)
