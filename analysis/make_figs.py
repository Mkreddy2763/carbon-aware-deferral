"""Paper figures (vector PDF, 3.5-inch IEEE column) from analysis/out/*.csv and
results/jobs/. Run after make_tables.py.

fig1_week.pdf     one example week: actual CI, day-ahead forecast + ACI band, and the
                  GPU power profile under P0 / P3 / P4 (where deferred work lands)
fig2_pareto.pdf   saving vs harm rate per policy (E1), one marker per region
fig3_k.pdf        oracle capture vs forecast-error multiplier k (E1 + E3)
fig4_slack.pdf    saving vs slack per region, P3 vs P4 (E1 + E2)
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# reference categorical palette (light), fixed order; markers are the secondary encoding
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
POL = {"P1": ("P1 threshold", C[3], "v"), "P2": ("P2 point", C[1], "s"), "P3": ("P3 point+replan", C[0], "o"),
       "P4": ("P4 conformal (ours)", C[2], "D"), "P4-l0.5": ("P4-$\\lambda$0.5 (ours)", C[6], "^"),
       "P5": ("P5 oracle", INK2, "*")}
REG = ["GBop", "GB", "CISO", "ERCO", "PJM", "MISO"]
REG_LABEL = {"GBop": "GB-op", "GB": "GB-25"}
W = 3.5

plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
    "pdf.fonttype": 42, "font.family": "serif", "legend.frameon": False,
})


def fig_week(jobs_dir, npz_dir, out, offset, region="GBop", model="F2"):
    files = {p: os.path.join(jobs_dir, f"FIG__{region}__{p}__S12__{m}__{meth}__k{k}__o{offset}.parquet")
             for p, m, meth, k in (("P0", "-", "-0", "0"), ("P3", "main", "-0", "1"),
                                   ("P4", "main", "aci0.1", "1"))}
    if not all(os.path.exists(f) for f in files.values()):
        print("fig1: job dumps missing", files)
        return
    raw = glob.glob(os.path.join(jobs_dir, "..", "raw", f"FIG__{region}__P0__*o{offset}.parquet"))
    sim0 = pd.Timestamp(pd.read_parquet(raw[0]).sim0_utc.iloc[0])
    z = np.load(os.path.join(npz_dir, f"{region}.npz"))
    y = z["y"]
    ts0 = pd.Timestamp(int(z["ts0_hours"]) * 3600, unit="s", tz="UTC")
    p0 = int((sim0 - ts0) / pd.Timedelta(hours=1))
    hrs = np.arange(24, 24 + 7 * 24)          # skip the warm-up day
    pos = p0 + hrs
    issue = z["issue"]
    h = 24
    row = pos - h - issue[0]
    fc = z[model][row, h - 1]
    lo = fc + z[f"{model}_aci_0.1_qlo"][row, h - 1]
    hi = fc + z[f"{model}_aci_0.1_qhi"][row, h - 1]
    t = sim0 + pd.to_timedelta(hrs, unit="h")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, 2.6), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    a1.fill_between(t, lo, hi, color=C[0], alpha=0.15, lw=0, label="90% ACI band (24 h ahead)")
    a1.plot(t, fc, color=C[0], lw=1.0, label="operator forecast (24 h ahead)")
    a1.plot(t, y[pos], color=INK, lw=1.2, label="actual")
    a1.set_ylabel("gCO$_2$/kWh")
    a1.legend(loc="upper left", ncol=3, fontsize=5.5, bbox_to_anchor=(0, 1.22))
    for (p, f), col, ls in zip(files.items(), (INK2, C[0], C[2]), ("-", "-", "-")):
        j = pd.read_parquet(f)
        prof = np.zeros(len(hrs) * 12 + 12 * 24 * 10)
        for s, d, kw in zip(j.start.to_numpy(), j.dur.to_numpy(), j.kw.to_numpy()):
            prof[s:s + d] += kw
        mw = prof[: (24 + 7 * 24) * 12].reshape(-1, 12).mean(axis=1)[24:] / 1e3
        a2.plot(t, mw, color=col, lw=0.9 if p != "P0" else 0.8, label=POL.get(p, (p,))[0] if p != "P0" else "P0 FCFS")
    a2.set_ylabel("GPU power (MW)")
    a2.legend(loc="upper left", ncol=3, fontsize=5.5)
    a2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%a\n%d %b"))
    fig.tight_layout(h_pad=0.3)
    fig.savefig(out)
    plt.close(fig)


def fig_pareto(agg, out):
    d = agg[(agg.exp == "E1") & (agg.util_target == 0.65)]
    fig, ax = plt.subplots(figsize=(W, 2.3))
    for p, (lab, col, mk) in POL.items():
        s = d[d.policy == p]
        if s.empty:
            continue
        ax.scatter(100 * s.harm_rate, s.saving_pct, color=col, marker=mk, s=16, label=lab,
                   edgecolor="white", linewidth=0.4, zorder=3)
        m = s[["harm_rate", "saving_pct"]].mean()
        ax.scatter(100 * m.harm_rate, m.saving_pct, color=col, marker=mk, s=60, edgecolor=INK,
                   linewidth=0.8, zorder=4)
    ax.set_xlabel("harm rate (% of shifted jobs emitting more than at submission)")
    ax.set_ylabel("emission saving vs P0 (%)")
    ax.legend(loc="lower right", ncol=2, fontsize=5.5)
    ax.text(0.01, 0.98, "small: per region; large: mean over regions", transform=ax.transAxes,
            va="top", fontsize=5.5, color=INK2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_k(agg, out):
    d = agg[(agg.exp.isin(["E1", "E3"])) & (agg.slack_h == 12) & (agg.util_target == 0.65)].copy()
    d.loc[d.exp == "E1", "k"] = d.loc[d.exp == "E1", "k"].replace(0.0, 1.0)
    fig, axs = plt.subplots(1, 2, figsize=(W, 1.9))
    for p in ("P2", "P3", "P4", "P4-l0.5"):
        lab, col, mk = POL[p]
        s = d[(d.policy == p) & (d.k > 0)].groupby("k")[["capture", "harm_rate"]].mean().sort_index()
        axs[0].plot(s.index, s.capture, color=col, marker=mk, ms=4, label=lab)
        axs[1].plot(s.index, 100 * s.harm_rate, color=col, marker=mk, ms=4)
    axs[0].set_xlabel("error multiplier $k$")
    axs[1].set_xlabel("error multiplier $k$")
    axs[0].set_ylabel("oracle capture ratio")
    axs[1].set_ylabel("harm rate (%)")
    axs[0].axhline(0, color=INK2, lw=0.6)
    fig.legend(loc="upper center", ncol=4, fontsize=5.3, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out)
    plt.close(fig)


def fig_slack(agg, out):
    d = agg[(agg.exp.isin(["E1", "E2"])) & (agg.util_target == 0.65) & (agg.k.isin([0.0, 1.0]))]
    fig, axs = plt.subplots(1, 2, figsize=(W, 1.9), sharey=True)
    for ax, p in zip(axs, ("P3", "P4")):
        for i, r in enumerate(REG):
            s = d[(d.policy == p) & (d.region == r)].sort_values("slack_h")
            if s.empty:
                continue
            ax.errorbar(s.slack_h, s.saving_pct, yerr=s.saving_pct_ci, color=C[i], marker="o", ms=3,
                        lw=1.0, capsize=1.5, elinewidth=0.6, label=REG_LABEL.get(r, r))
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 6, 12, 24])
        ax.set_xticklabels(["2", "6", "12", "24"])
        ax.set_xlabel("slack $S$ (h)")
        ax.set_title(POL[p][0], fontsize=7)
    axs[0].set_ylabel("saving vs P0 (%)")
    axs[1].legend(fontsize=5.3, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="analysis/out")
    ap.add_argument("--jobs", default="results/jobs")
    ap.add_argument("--forecasts", default="data/forecasts")
    ap.add_argument("--out", default="paper/figs")
    ap.add_argument("--week-offset", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    agg = pd.read_csv(os.path.join(a.tables, "summary.csv"))
    fig_week(a.jobs, a.forecasts, os.path.join(a.out, "fig1_week.pdf"), a.week_offset)
    fig_pareto(agg, os.path.join(a.out, "fig2_pareto.pdf"))
    fig_k(agg, os.path.join(a.out, "fig3_k.pdf"))
    fig_slack(agg, os.path.join(a.out, "fig4_slack.pdf"))
    print("figures ->", a.out)


if __name__ == "__main__":
    main()
