"""LaTeX tables for the paper. Called from make_tables.py (single source of numbers).
Missing entries render as a bold ?? so absent results cannot pass as numbers."""
import glob
import json
import os

import numpy as np

REG = ["GBop", "GB", "CISO", "ERCO", "PJM", "MISO"]
REG_TEX = {"GBop": "GB-op$^\\dagger$", "GB": "GB-25", "CISO": "CISO", "ERCO": "ERCO", "PJM": "PJM", "MISO": "MISO"}
Q = "\\textbf{??}"


def f(v, nd=1, scale=1.0):
    if v is None or not np.isfinite(v):
        return Q
    return f"{v * scale:.{nd}f}"


def forecast_table(fc_dir, path):
    rows = []
    order = [("GBop", "F2"), ("GBop", "F1"), ("GB", "F1"), ("GB", "F2s"), ("CISO", "F1"), ("ERCO", "F1"),
             ("PJM", "F1"), ("MISO", "F1")]
    qs = {json.load(open(p))["region"]: json.load(open(p)) for p in glob.glob(os.path.join(fc_dir, "*_quality.json"))}
    names = {"F1": "LGBM", "F2": "NESO", "F2s": "NESO-st", "F0": "naive"}
    for reg, m in order:
        q = qs.get(reg, {}).get("models", {})
        d = q.get(m)
        if d is None:
            continue
        mh = np.asarray(d["mae_by_h"], float)
        f0 = q.get("F0", {}).get("mae", np.nan)
        rows.append(" & ".join([
            REG_TEX[reg], names[m], f(np.nanmean(mh[0:6])), f(np.nanmean(mh[6:24])), f(np.nanmean(mh[24:48])),
            f(f0), f(d["split_0.1"]["coverage"], 0, 100), f(d["aci_0.1"]["coverage"], 0, 100),
            f(d["aci_0.1"]["width"], 0)]) + " \\\\")
    body = "\n".join(rows)
    tex = f"""\\begin{{table}}[t]
\\caption{{Test-period forecast quality. MAE (gCO$_2$/kWh) by horizon band; empirical coverage (\\%) of
nominal 90\\% intervals from split conformal and ACI; mean ACI interval width. Naive = seasonal-naive MAE.
$^\\dagger$GB with archived horizon-resolved operator (NESO) forecasts; NESO-st = the API's stored forecast.}}
\\label{{tab:forecast}}
\\centering\\scriptsize\\setlength{{\\tabcolsep}}{{2.4pt}}
\\resizebox{{\\columnwidth}}{{!}}{{%
\\begin{{tabular}}{{llrrrrrrr}}
\\toprule
 & & \\multicolumn{{3}}{{c}}{{MAE by horizon (h)}} & & \\multicolumn{{2}}{{c}}{{Cov.\\ @90\\%}} & \\\\
\\cmidrule(lr){{3-5}}\\cmidrule(lr){{7-8}}
Region & Model & 1--6 & 7--24 & 25--48 & Naive & Split & ACI & Width \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}}}
\\end{{table}}
"""
    with open(path, "w") as fh:
        fh.write(tex)


def _get(agg, **kw):
    m = np.ones(len(agg), bool)
    for k, v in kw.items():
        m &= (agg[k] == v).to_numpy()
    s = agg[m]
    return s.iloc[0] if len(s) else None


def main_table(agg, path):
    pols = [("P1", "P1"), ("P2", "P2"), ("P3", "P3"), ("P4", "P4"), ("P4-l0.5", "P4-$\\lambda$"), ("P5", "P5")]
    d = agg[(agg.exp == "E1") & (agg.util_target == 0.65)]
    rows = []
    for reg in REG:
        cells = []
        for p, _ in pols:
            r = _get(d, region=reg, policy=p)
            if r is None:
                cells.append(Q)
            else:
                cells.append(f"{f(r['saving_pct'])}/{f(r['harm_rate'], 0, 100)}")
        rows.append(REG_TEX[reg] + " & " + " & ".join(cells) + " \\\\")
    means = []
    for p, _ in pols:
        s = d[d.policy == p]
        means.append(f"{f(s.saving_pct.mean())}/{f(s.harm_rate.mean(), 0, 100)}" if len(s) == len(REG) else Q)
    rows.append("\\midrule\nMean & " + " & ".join(means) + " \\\\")
    head = " & ".join(n for _, n in pols)
    tex = f"""\\begin{{table}}[t]
\\caption{{E1, main result ($S{{=}}12$\\,h, real forecast errors, 10 weekly offsets): emission saving vs.\\ P0 (\\%) /
harm rate (\\% of shifted jobs that emit more than if run at submission). 95\\% CIs in the text and artifact.}}
\\label{{tab:main}}
\\centering\\scriptsize\\setlength{{\\tabcolsep}}{{2.5pt}}
\\begin{{tabular}}{{lrrrrrr}}
\\toprule
Region & {head} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    with open(path, "w") as fh:
        fh.write(tex)


def spatial_table(agg, path):
    pols = [("P3", "P3 point, temporal"), ("P4", "P4 conformal, temporal"), ("P5", "CG, temporal"),
            ("P6-point", "P6 point, spatio-temp."), ("P6", "P6 conformal, spatio-temp."),
            ("P6-oracle", "CG, spatio-temp.")]
    d = agg[(agg.exp == "E5")]
    rows = []
    for p, name in pols:
        cells = [name]
        for S in (6, 12):
            r = _get(d, policy=p, slack_h=float(S))
            if r is None:
                cells += [Q, Q, Q]
            else:
                cells += [f"{f(r['saving_pct'])}$\\pm${f(r['saving_pct_ci'])}", f(r["harm_rate"], 0, 100),
                          f(r.get("relocated_frac", np.nan), 0, 100) if p.startswith("P6") else "--"]
        rows.append(" & ".join(cells) + " \\\\")
    tex = f"""\\begin{{table}}[t]
\\caption{{E5, spatio-temporal shifting over CISO, ERCO and PJM (jobs homed uniformly at random; 10-minute
migration delay). Saving vs.\\ P0 (\\%, mean$\\pm$95\\% CI), harm rate (\\%) and share of deferrable jobs relocated (\\%).}}
\\label{{tab:spatial}}
\\centering\\scriptsize\\setlength{{\\tabcolsep}}{{2.5pt}}
\\begin{{tabular}}{{lrrrrrr}}
\\toprule
 & \\multicolumn{{3}}{{c}}{{$S=6$\\,h}} & \\multicolumn{{3}}{{c}}{{$S=12$\\,h}} \\\\
\\cmidrule(lr){{2-4}}\\cmidrule(lr){{5-7}}
Policy & Saving & Harm & Reloc. & Saving & Harm & Reloc. \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    with open(path, "w") as fh:
        fh.write(tex)


def write_all(agg, fc_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    forecast_table(fc_dir, os.path.join(out_dir, "forecast.tex"))
    main_table(agg, os.path.join(out_dir, "main.tex"))
    spatial_table(agg, os.path.join(out_dir, "spatial.tex"))


# ------------------------------------------------------------------ v2 tables
V2_POLS = [("P3", "P3"), ("P4", "P4"), ("P4-l0.5", "P4-$\\lambda$"), ("P7", "P7"), ("P8", "P8"),
           ("P9-r", "P9-r"), ("P5", "CG")]
EOL = " \\\\"


def main_table_v2(agg, path, util=0.7):
    d = agg[(agg.exp == "V2U") & (agg.util_target == util)]
    rows = []
    for reg in REG:
        cells = []
        for p, _ in V2_POLS:
            r = _get(d, region=reg, policy=p)
            cells.append(Q if r is None else f"{f(r['saving_pct'])}/{f(r['harm_rate'], 0, 100)}")
        rows.append(REG_TEX[reg] + " & " + " & ".join(cells) + EOL)
    means = []
    for p, _ in V2_POLS:
        s = d[d.policy == p]
        means.append(f"{f(s.saving_pct.mean())}/{f(s.harm_rate.mean(), 0, 100)}" if len(s) == len(REG) else Q)
    rows.append("\\midrule")
    rows.append("Mean & " + " & ".join(means) + EOL)
    head = " & ".join(n for _, n in V2_POLS)
    lines = [
        "\\begin{table}[t]",
        "\\caption{Emission saving vs.\\ FCFS (\\%) / harm rate (\\% of shifted jobs emitting more than at "
        "submission) at 70\\% utilisation, $S{=}12$\\,h, mean of 10 weekly offsets. GB-op uses NESO's own "
        "forecasts.}",
        "\\label{tab:main}",
        "\\centering\\scriptsize\\setlength{\\tabcolsep}{2.2pt}",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Region & " + head + EOL,
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def util_table(us, path):
    pols = [("P5", "Clairvoyant greedy (CG)"), ("P2", "Point P2"), ("P3", "Point+replan P3"), ("P4", "Conformal UB P4"),
            ("P7", "Decision-conf.\\ P7"), ("P8", "Headroom P8"), ("P9-r", "Jitter P9-r")]
    rows = []
    for p, name in pols:
        cells = [name]
        for u in (0.5, 0.7, 0.9):
            r = us[(us.policy == p) & (us.util == u)]
            if r.empty:
                cells += [Q, Q]
            else:
                r = r.iloc[0]
                cells += [f(r["harm_rate"], 0, 100), f(r["saving_pct"])]
        rows.append(" & ".join(cells) + EOL)
    extra = []
    for key, name in (("slowdown_p95", "P3 p95 slowdown"),
                      ("wait_min_mean_nondeferrable", "P3 non-defer.\\ wait (min)")):
        cells = [name]
        for u in (0.5, 0.7, 0.9):
            r = us[(us.policy == "P3") & (us.util == u)]
            cells += [f(r[key].iloc[0] if len(r) else np.nan), ""]
        extra.append(" & ".join(cells) + EOL)
    lines = [
        "\\begin{table}[t]",
        "\\caption{Utilisation sweep (mean of six regions, $S{=}12$\\,h): harm rate (\\%) and saving (\\%). "
        "Even clairvoyant greedy's harm rises with load; all of it is contention (its voluntary harm is 0).}",
        "\\label{tab:util}",
        "\\centering\\scriptsize\\setlength{\\tabcolsep}{2.6pt}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        " & \\multicolumn{2}{c}{50\\%} & \\multicolumn{2}{c}{70\\%} & \\multicolumn{2}{c}{90\\%}" + EOL,
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
        "Policy & Harm & Sav. & Harm & Sav. & Harm & Sav." + EOL,
        "\\midrule",
        *rows,
        "\\midrule",
        *extra,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_v2(agg, out_tables, out_dir):
    import pandas as pd
    p = os.path.join(out_tables, "v2_util_sweep.csv")
    if not os.path.exists(p):
        return
    main_table_v2(agg, os.path.join(out_dir, "main_v2.tex"))
    util_table(pd.read_csv(p), os.path.join(out_dir, "util.tex"))


