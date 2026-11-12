"""v2 aggregates for the paper (called from make_tables.py; single source of numbers).

* util sweep: per (util, policy) means over the six regions of the per-region means
* pooled harm decomposition: sum of forecast / re-plan / contention excess (kg) over all
  regions and offsets, as shares of total harmful excess
* re-planning mechanism: Spearman correlation between defer-decisions per job and harm
  rate across (region, util, policy) for policies with and without re-planning, plus
  paired re-plan minus no-re-plan differences
* frontier test at 70 % utilisation: is P8 / P9 dominated by any point of the
  uncertainty-aware dial (P3, P4, P4-lambda, P7 and P7-noreplan at all alpha)?
Writes <out>/v2_summary.json and adds macros (region slot 'ALL' for cross-region means).
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

UTILS = (0.5, 0.7, 0.9)
MEAN_METRICS = ("saving_pct", "harm_rate", "harm_rate_voluntary", "harm_rate_forced",
                "forced_frac_of_shifted", "capture", "slowdown_p95", "slowdown_mean",
                "wait_min_mean_nondeferrable", "wait_h_mean_forced", "shift_rate",
                "defer_decisions_per_job", "decisions_per_job", "moves_per_job",
                "got_wanted_slot_frac", "harmed_frac_forecast", "deadline_miss_rate")
PCT = {"harm_rate", "harm_rate_voluntary", "harm_rate_forced", "forced_frac_of_shifted", "shift_rate",
       "got_wanted_slot_frac", "harmed_frac_forecast", "deadline_miss_rate"}
DECOMP_POLICIES = ("P2", "P3", "P4", "P4-l0.5", "P5", "P7", "P7-noreplan", "P8", "P8-h0.5",
                   "P8-noreplan", "P9", "P9-r")
REPLAN_PAIRS = (("P3", "P2"), ("P7", "P7-noreplan"), ("P8", "P8-noreplan"), ("P9-r", "P9"))


def util_sweep(agg, M):
    d = agg[agg.exp == "V2U"]
    rows = []
    for (u, p), g in d.groupby(["util_target", "policy"]):
        r = {"util": u, "policy": p, "n_regions": len(g)}
        for m in MEAN_METRICS:
            if m in g:
                v = float(g[m].mean())
                r[m] = v
                M.put("V2U", f"ALL-S12-utiltarget{u:g}", p, m.replace("_", ""), v * (100 if m in PCT else 1),
                      2 if m in ("capture", "slowdown_p95", "slowdown_mean", "defer_decisions_per_job",
                                 "decisions_per_job", "moves_per_job") else 1)
        rows.append(r)
    return pd.DataFrame(rows)


def pooled_decomposition(cells, M):
    d = cells[(cells.exp == "V2U") & cells.policy.isin(DECOMP_POLICIES)]
    out = []
    for keys, g in d.groupby(["util_target", "policy"]):
        u, p = keys
        fc, rp, ct = (g.harm_excess_forecast_kg.sum(), g.harm_excess_replan_kg.sum(),
                      g.harm_excess_contention_kg.sum())
        tot = fc + rp + ct
        r = {"util": u, "policy": p, "forecast_kg": fc, "replan_kg": rp, "contention_kg": ct,
             "total_kg": tot, "share_forecast": fc / tot if tot > 0 else np.nan,
             "share_replan": rp / tot if tot > 0 else np.nan,
             "share_contention": ct / tot if tot > 0 else np.nan}
        out.append(r)
        for k in ("share_forecast", "share_replan", "share_contention"):
            if np.isfinite(r[k]):
                M.put("V2U", f"POOL-S12-utiltarget{u:g}", p, k.replace("_", ""), 100 * r[k], 0)
        M.put("V2U", f"POOL-S12-utiltarget{u:g}", p, "harmexcesst", tot / 1e3, 1)
    # per-region shares for the primary region
    for (reg, u, p), g in d.groupby(["region", "util_target", "policy"]):
        fc, rp, ct = (g.harm_excess_forecast_kg.sum(), g.harm_excess_replan_kg.sum(),
                      g.harm_excess_contention_kg.sum())
        tot = fc + rp + ct
        if tot > 0:
            for k, v in (("shareforecast", fc), ("sharereplan", rp), ("sharecontention", ct)):
                M.put("V2U", f"{reg}-S12-utiltarget{u:g}", p, k, 100 * v / tot, 0)
    return pd.DataFrame(out)


def replanning(agg, M):
    d = agg[(agg.exp == "V2U") & agg.policy.isin([p for pair in REPLAN_PAIRS for p in pair])]
    res = {}
    for metric in ("harm_rate", "harm_rate_voluntary"):
        x, y = d.defer_decisions_per_job.to_numpy(float), d[metric].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        rho, p = stats.spearmanr(x[ok], y[ok])
        res[f"spearman_{metric}"] = (float(rho), float(p), int(ok.sum()))
        M.put("V2R", "ALL", "decisions", f"rho{metric.replace('_', '')}", float(rho), 2)
        M.put("V2R", "ALL", "decisions", f"p{metric.replace('_', '')}", float(p), 3)
        M.put("V2R", "ALL", "decisions", "n", str(int(ok.sum())))
    pairs = []
    for a, b in REPLAN_PAIRS:
        A = d[d.policy == a].set_index(["region", "util_target"])
        B = d[d.policy == b].set_index(["region", "util_target"])
        j = A.join(B, lsuffix="_a", rsuffix="_b", how="inner")
        if j.empty:
            continue
        dd = j.defer_decisions_per_job_a - j.defer_decisions_per_job_b
        dh = 100 * (j.harm_rate_a - j.harm_rate_b)
        dv = 100 * (j.harm_rate_voluntary_a - j.harm_rate_voluntary_b)
        ds = j.saving_pct_a - j.saving_pct_b
        r = {"replan": a, "noreplan": b, "n": len(j), "d_decisions": float(dd.mean()),
             "d_harm_pp": float(dh.mean()), "d_harm_vol_pp": float(dv.mean()), "d_saving_pp": float(ds.mean()),
             "harm_up_frac": float((dh > 0).mean()), "saving_up_frac": float((ds > 0).mean())}
        r["harm_drop_pp"] = -r["d_harm_pp"]
        r["saving_loss_pp"] = -r["d_saving_pp"]
        pairs.append(r)
        M.put("V2R", "ALL", f"{a}vs{b}", "harmdroppp", r["harm_drop_pp"], 1)
        M.put("V2R", "ALL", f"{a}vs{b}", "savinglosspp", r["saving_loss_pp"], 1)
        for k in ("d_decisions", "d_harm_pp", "d_harm_vol_pp", "d_saving_pp", "harm_up_frac", "saving_up_frac"):
            M.put("V2R", "ALL", f"{a}vs{b}", k.replace("_", ""), r[k] * (100 if k.endswith("frac") else 1),
                  2 if k == "d_decisions" else 1)
    res["pairs"] = pairs
    return res


def frontier(agg, M, util=0.7):
    base = agg[(agg.util_target == util) & (agg.slack_h == 12)]
    dial = base[(base.exp.isin(["V2DL", "V2DA"])) & ~base.policy.isin(["P0", "P5"]) |
                ((base.exp == "V2U") & base.policy.isin(["P3", "P4", "P2"]))]
    out = []
    for cand in ("P8", "P8-h0.5", "P8-noreplan", "P9", "P9-r"):
        for reg, g in base[(base.exp == "V2U") & (base.policy == cand)].groupby("region"):
            r0 = g.iloc[0]
            f = dial[dial.region == reg]
            dominated_by = f[(f.harm_rate <= r0.harm_rate) & (f.saving_pct >= r0.saving_pct)]
            p3 = base[(base.exp == "V2U") & (base.policy == "P3") & (base.region == reg)]
            p4 = base[(base.exp == "V2U") & (base.policy == "P4") & (base.region == reg)]
            target = bool(len(p3) and len(p4) and r0.harm_rate <= p4.harm_rate.iloc[0]
                          and r0.saving_pct >= p3.saving_pct.iloc[0])
            out.append({"policy": cand, "region": reg, "saving": r0.saving_pct, "harm": r0.harm_rate,
                        "non_dominated": dominated_by.empty, "hits_target": target,
                        "p3_saving": float(p3.saving_pct.iloc[0]) if len(p3) else np.nan,
                        "p4_harm": float(p4.harm_rate.iloc[0]) if len(p4) else np.nan})
    df = pd.DataFrame(out)
    for cand, g in df.groupby("policy"):
        M.put("V2F", "ALL", cand, "nondominated", str(int(g.non_dominated.sum())))
        M.put("V2F", "ALL", cand, "hitstarget", str(int(g.hits_target.sum())))
        M.put("V2F", "ALL", cand, "nregions", str(len(g)))
    # the same target test at every utilisation
    tgt = []
    for u in UTILS:
        b = agg[(agg.exp == "V2U") & (agg.util_target == u)]
        for cand in ("P8", "P8-h0.5", "P9", "P9-r"):
            n = 0
            for reg, g in b[b.policy == cand].groupby("region"):
                p3 = b[(b.policy == "P3") & (b.region == reg)]
                p4 = b[(b.policy == "P4") & (b.region == reg)]
                if len(p3) and len(p4) and g.harm_rate.iloc[0] <= p4.harm_rate.iloc[0] \
                        and g.saving_pct.iloc[0] >= p3.saving_pct.iloc[0]:
                    n += 1
            tgt.append({"util": u, "policy": cand, "regions_hitting_target": n})
            M.put("V2F", f"u{u:g}", cand, "hitstarget", str(n))
    return df, pd.DataFrame(tgt)


def guarantee(cells, M):
    """What the decision-level conformal rule actually guarantees.

    Calibration bounds, over *all* decisions where running now was possible,
    P(defer and realised saving < 0) <= alpha (marginal). The harm rate instead
    conditions on deferring, P(harm | defer), which conformal prediction does not
    bound. Denominator of the marginal rate: deferrable jobs that could start at
    submission = n_deferrable - n_forced (a forced job is exactly a deferrable job
    whose submission slot was full)."""
    d = cells[(cells.exp == "V2DA") & cells.policy.isin(["P7", "P7-noreplan"])].copy()
    if d.empty:
        return pd.DataFrame()
    n_forced = d.n_shifted * d.forced_frac_of_shifted
    n_vol = d.n_shifted - n_forced
    d["marginal"] = d.harm_rate_voluntary * n_vol / (d.n_deferrable - n_forced)
    d["conditional"] = d.harm_rate_voluntary
    g = d.groupby(["policy", "alpha", "region"], as_index=False)[["marginal", "conditional"]].mean()
    for (p, a), s in g.groupby(["policy", "alpha"]):
        M.put("V2G", f"a{a:g}", p, "marginalmax", 100 * s.marginal.max(), 1)
        M.put("V2G", f"a{a:g}", p, "conditionalmax", 100 * s.conditional.max(), 1)
        M.put("V2G", f"a{a:g}", p, "marginalbelow", str(int((s.marginal <= a).sum())))
        M.put("V2G", f"a{a:g}", p, "conditionalbelow", str(int((s.conditional <= a).sum())))
        M.put("V2G", f"a{a:g}", p, "conditionalabove", str(int((s.conditional > a).sum())))
        for _, r in s.iterrows():
            M.put("V2G", f"a{a:g}-{r.region}", p, "marginal", 100 * r.marginal, 1)
            M.put("V2G", f"a{a:g}-{r.region}", p, "conditional", 100 * r.conditional, 1)
    return g


def v1_means(agg, M):
    """Cross-region means from the v1 grid (65 % utilisation) used by the guidance section."""
    d = agg[agg.exp.isin(["E1", "E2"]) & (agg.util_target == 0.65) & agg.policy.isin(["P3", "P5", "P7", "P2"])]
    d = d[d.k.isin([0.0, 1.0])]
    for (s, p), g in d.groupby(["slack_h", "policy"]):
        M.put("V1S", f"ALL-S{s:g}", p, "savingpct", float(g.saving_pct.mean()), 1)
        M.put("V1S", f"ALL-S{s:g}", p, "harmrate", 100 * float(g.harm_rate.mean()), 0)
    d = agg[agg.exp.isin(["E1", "E3"]) & (agg.slack_h == 12) & agg.policy.isin(["P2", "P3", "P4", "P7"])].copy()
    d["kk"] = d.k.replace(0.0, 1.0)
    for (k, p), g in d.groupby(["kk", "policy"]):
        M.put("V1K", f"ALL-k{k:g}", p, "capture", float(g.capture.mean()), 2)
        M.put("V1K", f"ALL-k{k:g}", p, "harmrate", 100 * float(g.harm_rate.mean()), 0)


def relative(agg, M, util=0.7):
    """Relative harm reduction and saving difference of each policy vs P3 (for prose)."""
    d = agg[(agg.exp == "V2U") & (agg.util_target == util)]
    for scope, f in (("ALL", d), ("GBop", d[d.region == "GBop"])):
        m = f.groupby("policy")[["harm_rate", "saving_pct"]].mean()
        if "P3" not in m.index:
            continue
        for p in m.index:
            rel = 100 * (1 - m.loc[p, "harm_rate"] / m.loc["P3", "harm_rate"])
            M.put("V2X", f"{scope}-u{util:g}", p, "harmcutvsPthree", float(rel), 0)
            M.put("V2X", f"{scope}-u{util:g}", p, "savingdiffvsPthree",
                  float(m.loc[p, "saving_pct"] - m.loc["P3", "saving_pct"]), 1)
            M.put("V2X", f"{scope}-u{util:g}", p, "savinglossvsPthree",
                  float(m.loc["P3", "saving_pct"] - m.loc[p, "saving_pct"]), 1)


def run(cells, agg, M, out_dir):
    v1_means(agg, M)
    relative(agg, M)
    if not (cells.exp == "V2U").any():
        return
    us = util_sweep(agg, M)
    dec = pooled_decomposition(cells, M)
    rep = replanning(agg, M)
    fr, tgt = frontier(agg, M)
    gu = guarantee(cells, M)
    gu.to_csv(os.path.join(out_dir, "v2_guarantee.csv"), index=False)
    us.to_csv(os.path.join(out_dir, "v2_util_sweep.csv"), index=False)
    dec.to_csv(os.path.join(out_dir, "v2_decomposition.csv"), index=False)
    fr.to_csv(os.path.join(out_dir, "v2_frontier.csv"), index=False)
    tgt.to_csv(os.path.join(out_dir, "v2_target.csv"), index=False)
    with open(os.path.join(out_dir, "v2_summary.json"), "w") as f:
        json.dump({"replanning": rep}, f, indent=1, default=float)
