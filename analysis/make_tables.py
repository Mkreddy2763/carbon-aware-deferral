"""Single source of every number in the paper.

Reads results/raw/*.parquet (one row per cell) and data/forecasts/*_quality.json,
pairs each policy run with the carbon-agnostic (P0) and oracle (P5 / P6-oracle)
runs on the *same* start-week offset, aggregates across offsets (mean and 95% t
CI), runs paired Wilcoxon signed-rank tests, and writes
  <out>/summary.csv, <out>/*.tex tables, and paper/macros.tex.
Macros are looked up in LaTeX as \\res{exp}{region}{variant}{metric}; anything
not produced here renders as a bold ?? in the PDF.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

BASE_KEY = ["exp", "region", "slack_h", "dur_noise", "defer_min_h", "p_gpu", "pue", "util_target"]
VAR_KEY = BASE_KEY + ["policy", "model", "method", "alpha", "k"]
METRICS = ["saving_pct", "capture", "regret_pct", "harm_rate", "harm_rate_of_deferrable", "shift_rate",
           "harm_rate_voluntary", "harm_rate_forced", "forced_frac_of_shifted",
           "slowdown_mean", "slowdown_p95", "slowdown_mean_deferrable", "wait_h_mean_deferrable",
           "deadline_miss_rate", "utilisation", "emissions_kg", "harm_excess_pct_of_gain",
           "decision_ms_median", "decision_ms_p99", "relocated_frac",
           # v2 diagnostics (absent in v1 cells -> NaN)
           "harm_share_forecast", "harm_share_replan", "harm_share_contention", "harmed_frac_forecast",
           "got_wanted_slot_frac", "decisions_per_job", "defer_decisions_per_job", "moves_per_job",
           "defer_decisions_harmed", "defer_decisions_unharmed", "wait_min_mean_nondeferrable",
           "wait_h_mean_forced"]


def load(raws):
    raws = [raws] if isinstance(raws, str) else raws
    files = [f for raw in raws for f in glob.glob(os.path.join(raw, "*.parquet"))]
    files = [f for f in files if os.path.exists(f + ".done")]
    if not files:
        raise SystemExit(f"no finished cells in {raws}")
    # a cell marked done but unreadable (e.g. Windows 260-char path limit) must not vanish silently
    n_done = sum(f.endswith(".parquet.done") for raw in raws if os.path.isdir(raw) for f in os.listdir(raw))
    if n_done != len(files):
        raise SystemExit(f"{n_done} cells are marked done but only {len(files)} are readable. On Windows "
                         f"this is the 260-character path limit: move the project to a short path or "
                         f"enable long paths (see README_REPRODUCE.md).")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def pair(df):
    df = df.copy()
    if "util_target" not in df:
        df["util_target"] = 0.65
    for c in ("dur_noise", "defer_min_h", "p_gpu", "pue", "alpha", "k", "slack_h", "util_target"):
        df[c] = df[c].astype(float)
    b = df[df.policy == "P0"][BASE_KEY + ["offset", "emissions_kg"]].rename(columns={"emissions_kg": "E_P0"})
    o = df[df.policy == "P5"][BASE_KEY + ["offset", "emissions_kg"]].rename(columns={"emissions_kg": "E_P5"})
    os_ = df[df.policy == "P6-oracle"][BASE_KEY + ["offset", "emissions_kg"]].rename(
        columns={"emissions_kg": "E_P6o"})
    df = df.merge(b, on=BASE_KEY + ["offset"], how="left").merge(o, on=BASE_KEY + ["offset"], how="left")
    df = df.merge(os_, on=BASE_KEY + ["offset"], how="left")
    spatial = df.policy.str.startswith("P6")
    oracle = np.where(spatial & df.E_P6o.notna(), df.E_P6o, df.E_P5)
    df["E_oracle"] = oracle
    df["saving_pct"] = 100 * (df.E_P0 - df.emissions_kg) / df.E_P0
    df["capture"] = (df.E_P0 - df.emissions_kg) / (df.E_P0 - df.E_oracle)
    df["regret_pct"] = 100 * (df.emissions_kg - df.E_oracle) / df.E_P0
    gain = df.shift_gain_kg + df.harm_excess_kg  # gross gain of beneficial shifts
    df["harm_excess_pct_of_gain"] = 100 * df.harm_excess_kg / gain.where(gain > 0)
    df["relocated_frac"] = df.n_relocated / df.n_deferrable.where(df.n_deferrable > 0)
    return df


def ci95(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, 0
    m = x.mean()
    if n < 2:
        return m, np.nan, n
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return m, h, n


def aggregate(df):
    rows = []
    for key, g in df.groupby(VAR_KEY, dropna=False):
        r = dict(zip(VAR_KEY, key))
        r["n_offsets"] = g.offset.nunique()
        for m in METRICS:
            if m in g:
                mean, half, n = ci95(g[m])
                r[m] = mean
                r[m + "_ci"] = half
        rows.append(r)
    return pd.DataFrame(rows)


def wilcoxon(df, a, b, metric, key_extra=()):
    """Paired signed-rank test of policy a vs b on identical offsets, per config."""
    out = []
    ka = [k for k in VAR_KEY if k not in ("policy", "model", "method", "alpha")] + list(key_extra)
    A = df[df.policy == a]
    B = df[df.policy == b]
    kb = [k for k in ka if k not in key_extra]
    m = A.merge(B, on=kb + ["offset"], suffixes=("_a", "_b"))
    for key, g in m.groupby([k if k in kb else k + "_a" for k in ka], dropna=False):
        x, y = g[metric + "_a"].to_numpy(float), g[metric + "_b"].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < 2 or np.allclose(x, y):
            p = np.nan
        else:
            p = stats.wilcoxon(x, y, zero_method="zsplit").pvalue
        out.append({**dict(zip(ka, key if isinstance(key, tuple) else (key,))), "a": a, "b": b,
                    "metric": metric, "n": len(x), "median_diff": float(np.median(x - y)) if len(x) else np.nan,
                    "wins_a": int((x < y).sum()), "p": p})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- LaTeX output
def fmt(v, nd=1):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    return f"{v:.{nd}f}"


class Macros:
    def __init__(self):
        self.d = {}

    def put(self, exp, region, variant, metric, value, nd=1):
        s = fmt(value, nd) if not isinstance(value, str) else value
        if s is not None:
            self.d[(exp, region, variant, metric)] = s

    def write(self, path):
        lines = ["% generated by analysis/make_tables.py -- do not edit",
                 "\\makeatletter",
                 "\\providecommand{\\res}[4]{\\ifcsname r@#1@#2@#3@#4\\endcsname"
                 "\\csname r@#1@#2@#3@#4\\endcsname\\else\\textbf{??}\\fi}"]
        for (e, r, v, m), s in sorted(self.d.items()):
            lines.append(f"\\expandafter\\def\\csname r@{e}@{r}@{v}@{m}\\endcsname{{{s}}}")
        lines.append("\\makeatother")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")


def variant_name(r):
    if r["policy"] in ("P0", "P1", "P5", "P6-oracle"):
        return r["policy"]
    v = r["policy"]
    if r["policy"].startswith("P4") and r["policy"] != "P4-l0.25" and r["method"] not in ("-", "aci"):
        v += "-" + r["method"]
    if r["alpha"] not in (0.0, 0.1) and r["policy"] in ("P4", "P6", "P4-noreplan", "P7", "P7-noreplan", "P6-cdec"):
        v += f"-a{r['alpha']:g}"
    if r["k"] not in (0.0, 1.0):
        v += f"-k{r['k']:g}"
    if r["model"] not in ("-", "main"):
        v += "-" + r["model"]
    return v


def cond_name(r):
    """Encode non-default experimental knobs into the macro's region slot."""
    s = r["region"] + f"-S{r['slack_h']:g}"
    for k, d in (("dur_noise", 0.0), ("defer_min_h", 1.0), ("p_gpu", 300.0), ("pue", 1.2),
                 ("util_target", 0.65)):
        if float(r[k]) != d:
            s += f"-{k.replace('_', '')}{r[k]:g}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", default=["results/raw"])
    ap.add_argument("--out", default="analysis/out")
    ap.add_argument("--forecasts", default="data/forecasts")
    ap.add_argument("--macros", default=None)
    ap.add_argument("--tex-tables", default=None, help="write LaTeX tables here (e.g. paper/tables)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pair(load(a.raw))
    df.to_csv(os.path.join(a.out, "cells.csv"), index=False)
    agg = aggregate(df)
    agg["variant"] = agg.apply(variant_name, axis=1)
    agg["cond"] = agg.apply(cond_name, axis=1)
    agg.to_csv(os.path.join(a.out, "summary.csv"), index=False)

    tests = []
    for a_, b_ in (("P8", "P3"), ("P9", "P2"), ("P9-r", "P3"), ("P8", "P4"),
                   ("P4", "P3"), ("P4", "P2"), ("P3", "P2"), ("P6", "P6-point"), ("P6", "P4"),
                   ("P4-l0.5", "P3"), ("P4", "P4-noreplan"), ("P7", "P3"), ("P7", "P4"), ("P7", "P2"),
                   ("P6-cdec", "P6-point"), ("P7", "P7-noreplan")):
        for metric in ("emissions_kg", "harm_rate", "harm_rate_voluntary"):
            if (df.policy == a_).any() and (df.policy == b_).any():
                t = wilcoxon(df, a_, b_, metric)
                if len(t):
                    tests.append(t)
    if tests:
        pd.concat(tests).to_csv(os.path.join(a.out, "wilcoxon.csv"), index=False)

    M = Macros()
    for _, r in agg.iterrows():
        for m, nd in (("saving_pct", 1), ("capture", 2), ("regret_pct", 1), ("harm_rate", 1),
                      ("harm_rate_voluntary", 1), ("harm_rate_forced", 1), ("forced_frac_of_shifted", 1),
                      ("shift_rate", 1), ("slowdown_mean", 2), ("slowdown_p95", 2),
                      ("wait_h_mean_deferrable", 1), ("deadline_miss_rate", 2), ("utilisation", 1),
                      ("harm_excess_pct_of_gain", 1), ("decision_ms_median", 2),
                      ("decision_ms_p99", 2), ("relocated_frac", 1),
                      ("harm_share_contention", 0), ("harm_share_forecast", 0), ("harm_share_replan", 0),
                      ("got_wanted_slot_frac", 1), ("defer_decisions_per_job", 2), ("moves_per_job", 2),
                      ("wait_min_mean_nondeferrable", 1), ("wait_h_mean_forced", 1)):
            v = r.get(m)
            if v is None or not np.isfinite(v):
                continue
            scale = 100 if m in ("harm_rate", "shift_rate", "deadline_miss_rate", "utilisation",
                                 "relocated_frac", "harm_rate_voluntary", "harm_rate_forced",
                                 "forced_frac_of_shifted", "harm_share_contention", "harm_share_forecast",
                                 "harm_share_replan", "got_wanted_slot_frac") else 1
            M.put(r["exp"], r["cond"], r["variant"], m.replace("_", ""), v * scale, nd)
            ci = r.get(m + "_ci")
            if ci is not None and np.isfinite(ci):
                M.put(r["exp"], r["cond"], r["variant"], m.replace("_", "") + "ci", ci * scale, nd)
        M.put(r["exp"], r["cond"], r["variant"], "noffsets", str(int(r["n_offsets"])))
    if tests:
        for _, t in pd.concat(tests).iterrows():
            if np.isfinite(t["p"]):
                cond = f"{t['region']}-S{t['slack_h']:g}" + (
                    f"-utiltarget{t['util_target']:g}" if float(t.get("util_target", 0.65)) != 0.65 else "")
                M.put(t["exp"], cond, f"{t['a']}vs{t['b']}{'' if t['k'] in (0, 1) else '-k%g' % t['k']}",
                      f"p{t['metric'].replace('_', '')}", t["p"], 3)
    if tests:
        # cross-region summaries of paired tests (v2): regions significant at 5% and week-level wins
        T = pd.concat(tests)
        T = T[T.exp == "V2U"]
        for (u, a_, b_, met), g in T.groupby(["util_target", "a", "b", "metric"]):
            tag = f"u{u:g}"
            M.put("V2W", tag, f"{a_}vs{b_}", f"{met.replace('_', '')}sig", str(int((g.p < 0.05).sum())))
            M.put("V2W", tag, f"{a_}vs{b_}", f"{met.replace('_', '')}wins", str(int(g.wins_a.sum())))
            M.put("V2W", tag, f"{a_}vs{b_}", f"{met.replace('_', '')}n", str(int(g.n.sum())))
            M.put("V2W", tag, f"{a_}vs{b_}", "nregions", str(len(g)))
    # forecast quality
    for qf in sorted(glob.glob(os.path.join(a.forecasts, "*_quality.json"))):
        q = json.load(open(qf))
        reg = q["region"]
        for model, d in q["models"].items():
            M.put("FQ", reg, model, "mae", d["mae"], 1)
            M.put("FQ", reg, model, "bias", d["bias"], 1)
            for hb in ("h1-6", "h7-24", "h25-48"):
                i0, i1 = {"h1-6": (0, 6), "h7-24": (6, 24), "h25-48": (24, 48)}[hb]
                M.put("FQ", reg, model, "mae" + hb.replace("-", "to"),
                      float(np.nanmean(d["mae_by_h"][i0:i1])), 1)
            for meth in ("split", "aci"):
                for al in ("0.1", "0.2"):
                    k = f"{meth}_{al}"
                    if k in d:
                        tag = meth + al.replace("0.", "a")
                        M.put("FQ", reg, model, "cov" + tag, 100 * d[k]["coverage"], 1)
                        M.put("FQ", reg, model, "ucov" + tag, 100 * d[k]["upper_coverage"], 1)
                        M.put("FQ", reg, model, "width" + tag, d[k]["width"], 0)
                        for hb, v in d[k]["by_h"].items():
                            M.put("FQ", reg, model, "cov" + tag + hb.replace("-", "to"), 100 * v["coverage"], 1)
                            M.put("FQ", reg, model, "width" + tag + hb.replace("-", "to"), v["width"], 0)
    # data and run facts (so the setup section is also generated, not typed)
    tm = os.path.join(os.path.dirname(a.forecasts), "trace", "meta.json")
    if os.path.exists(tm):
        t = json.load(open(tm))
        M.put("DATA", "trace", "all", "ntasks", f"{t['jobs']:,}".replace(",", "{,}"))
        M.put("DATA", "trace", "all", "ntasksraw", f"{t['tasks_total']:,}".replace(",", "{,}"))
        M.put("DATA", "trace", "all", "spandays", t["span_days"], 0)
        M.put("DATA", "trace", "all", "fracjobsgeoneh", 100 * t["frac_jobs_ge_1h"], 1)
        M.put("DATA", "trace", "all", "fracgpuhgeoneh", 100 * t["frac_gpu_hours_ge_1h"], 1)
        M.put("DATA", "trace", "all", "utilcov", 100 * t["util_coverage"], 0)
        M.put("DATA", "trace", "all", "utilmean", t["util_mean"], 2)
    for qf in sorted(glob.glob(os.path.join(a.forecasts, "*_quality.json"))):
        q = json.load(open(qf))
        M.put("DATA", q["region"], "all", "teststart", q["test_start"][:10])
        M.put("DATA", q["region"], "all", "calibstart", q["calib_start"][:10])
        M.put("DATA", q["region"], "all", "end", q["end"][:10])
    for exp, g in df.groupby("exp"):
        M.put("RUN", exp, "all", "ncells", f"{len(g):,}".replace(",", "{,}"))
        M.put("RUN", exp, "all", "corehours", float(g.cell_seconds.sum() / 3600), 1)
        if "n_jobs" in g:
            M.put("RUN", exp, "all", "jobsperweekmin", f"{int(g.n_jobs.min()):,}".replace(",", "{,}"))
            M.put("RUN", exp, "all", "jobsperweekmax", f"{int(g.n_jobs.max()):,}".replace(",", "{,}"))
    M.put("RUN", "ALL", "all", "ncells", f"{len(df):,}".replace(",", "{,}"))
    M.put("RUN", "ALL", "all", "corehours", float(df.cell_seconds.sum() / 3600), 1)
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import v2_extras
    v2_extras.run(df, agg, M, a.out)
    if a.tex_tables:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import latex_tables
        latex_tables.write_all(agg, a.forecasts, a.tex_tables)
        latex_tables.write_v2(agg, a.out, a.tex_tables)
    macros = a.macros or os.path.join(a.out, "macros.tex")
    M.write(macros)
    print(f"[make_tables] {len(df)} cells -> {len(agg)} variants; {len(M.d)} macros -> {macros}")


if __name__ == "__main__":
    main()
