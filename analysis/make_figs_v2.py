"""v2 figures (vector PDF, 3.5-inch column) from analysis/out/*.csv written by make_tables.py.

fig_util.pdf     harm rate and emission saving vs cluster utilisation (mean over regions)
fig_decomp.pdf   where harmful excess comes from: forecast / re-plan / contention shares
fig_dial.pdf     savings vs harm dial at 70 % utilisation (GB-op and mean over regions)
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# reference categorical palette (light), fixed slot order; markers are secondary encoding
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
W = 3.5
POL = {  # label, colour, marker
    "P5": ("Clairvoyant greedy (CG)", INK, "*"),
    "P3": ("Point+replan (P3)", C[0], "o"),
    "P4": ("Conformal UB (P4)", C[1], "s"),
    "P7": ("Decision-conformal (P7)", C[2], "D"),
    "P8": ("Headroom (P8)", C[6], "^"),
    "P9-r": ("Jitter (P9-r)", C[4], "v"),
}
plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 5.8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.3,
    "pdf.fonttype": 42, "font.family": "serif", "legend.frameon": False,
})


def fig_util(us, out):
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.6))
    for p, (lab, col, mk) in POL.items():
        s = us[us.policy == p].sort_values("util")
        if s.empty:
            continue
        x = 100 * s.util
        axs[0].plot(x, 100 * s.harm_rate, color=col, marker=mk, ms=4, label=lab)
        axs[1].plot(x, s.saving_pct, color=col, marker=mk, ms=4)
    axs[0].set_ylabel("harm rate (% of shifted jobs)")
    axs[1].set_ylabel("emission saving vs P0 (%)")
    for a in axs:
        a.set_xlabel("baseline utilisation (%)")
        a.set_xticks([50, 70, 90])
    fig.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.27, 1, 1))
    fig.savefig(out)
    if os.environ.get("FIG_PNG"):
        fig.savefig(os.path.join(os.environ["FIG_PNG"], os.path.basename(out).replace(".pdf", ".png")), dpi=400)
    plt.close(fig)


def fig_decomp(dec, out):
    # exact attribution needs a single decision per job: use the no-re-planning variants
    pols = ["P5", "P2", "P7-noreplan", "P8-noreplan", "P9"]
    names = {"P5": "Clairv. greedy", "P2": "Point", "P7-noreplan": "Dec.-conf.", "P8-noreplan": "Headroom",
             "P9": "Jitter"}
    utils = [0.5, 0.7, 0.9]
    # single-decision policies have no re-planning term (it is identically 0)
    parts = [("share_forecast", "forecast error", C[1]),
             ("share_contention", "contention (missed wanted slot)", C[0])]
    fig, ax = plt.subplots(figsize=(W, 2.2))
    x, labels, xt = 0, [], []
    for p in pols:
        for u in utils:
            r = dec[(dec.policy == p) & (dec.util == u)]
            if r.empty or not np.isfinite(r.share_contention.iloc[0]):
                x += 1
                continue
            bottom_pos, bottom_neg = 0.0, 0.0
            for k, lab, col in parts:
                v = 100 * float(r[k].iloc[0])
                if v >= 0:
                    ax.bar(x, v, bottom=bottom_pos, color=col, width=0.8, edgecolor="white", linewidth=0.6,
                           label=lab if (p == pols[1] and u == utils[0]) else None)
                    bottom_pos += v
                else:
                    ax.bar(x, v, bottom=bottom_neg, color=col, width=0.8, edgecolor="white", linewidth=0.6,
                           label=lab if (p == pols[1] and u == utils[0]) else None)
                    bottom_neg += v
            xt.append(x)
            labels.append(f"{int(100 * u)}")
            x += 1
        x += 0.6
    ax.set_xticks(xt)
    ax.set_xticklabels(labels, fontsize=5.5)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_ylabel("share of harmful excess (%)")
    # policy names under each group of three bars
    pos = 0
    for p in pols:
        ax.text(pos + 1, -0.2, names[p], transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=5.8, color=INK)
        pos += 3.6
    ax.set_xlabel("")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    fig.savefig(out)
    if os.environ.get("FIG_PNG"):
        fig.savefig(os.path.join(os.environ["FIG_PNG"], os.path.basename(out).replace(".pdf", ".png")), dpi=400)
    plt.close(fig)


def fig_dial(agg, out, util=0.7):
    d = agg[(agg.util_target == util) & (agg.slack_h == 12)]

    def pts(frame):
        return frame.groupby(["exp", "policy", "alpha"], as_index=False)[["saving_pct", "harm_rate"]].mean()

    fig, axs = plt.subplots(1, 2, figsize=(W, 2.7), sharey=False)
    for ax, reg, title in ((axs[0], "GBop", "GB-op (NESO forecasts)"), (axs[1], None, "mean of six regions")):
        f = d if reg is None else d[d.region == reg]
        if reg is None:
            f = f.groupby(["exp", "policy", "alpha"], as_index=False)[["saving_pct", "harm_rate"]].mean()
        else:
            f = pts(f)
        lam = f[(f.exp == "V2DL") & f.policy.str.startswith("P4-l")].copy()
        lam["lam"] = lam.policy.str.replace("P4-l", "").astype(float)
        lam = lam.sort_values("lam")
        ax.plot(100 * lam.harm_rate, lam.saving_pct, color=C[1], marker="s", ms=3, lw=1.0,
                label="hourly conformal, $\\lambda$ sweep")
        for pol, col, lab in (("P7", C[2], "decision conformal, $\\alpha$ sweep"),
                              ("P7-noreplan", C[5], "same, no re-planning")):
            a = f[(f.exp == "V2DA") & (f.policy == pol)].sort_values("alpha")
            ax.plot(100 * a.harm_rate, a.saving_pct, color=col, marker="D", ms=3, lw=1.0, label=lab)
        for p, (lab, col, mk) in POL.items():
            if p in ("P4", "P7"):
                continue
            s = f[(f.exp == "V2U") & (f.policy == p)]
            if len(s):
                ax.scatter(100 * s.harm_rate, s.saving_pct, color=col, marker=mk, s=40 if p == "P5" else 28, zorder=4,
                           edgecolor="white", linewidth=0.5, label=lab)
        ax.set_title(title, fontsize=6.5)
        ax.set_xlabel("harm rate (%)")
    axs[0].set_ylabel("saving vs P0 (%)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0), fontsize=5.5)
    fig.tight_layout(rect=(0, 0.3, 1, 1))
    fig.savefig(out)
    if os.environ.get("FIG_PNG"):
        fig.savefig(os.path.join(os.environ["FIG_PNG"], os.path.basename(out).replace(".pdf", ".png")), dpi=400)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="analysis/out")
    ap.add_argument("--out", default="paper/figs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    agg = pd.read_csv(os.path.join(a.tables, "summary.csv"))
    us = pd.read_csv(os.path.join(a.tables, "v2_util_sweep.csv"))
    dec = pd.read_csv(os.path.join(a.tables, "v2_decomposition.csv"))
    fig_util(us, os.path.join(a.out, "fig_util.pdf"))
    fig_decomp(dec, os.path.join(a.out, "fig_decomp.pdf"))
    fig_dial(agg, os.path.join(a.out, "fig_dial.pdf"))
    print("v2 figures ->", a.out)


if __name__ == "__main__":
    main()
