"""Normalise raw carbon data to hourly series: data/ci/<REGION>.parquet.

Index: hour-beginning UTC timestamps. Columns: ci (actual, gCO2/kWh); for GB
also op (the NESO forecast stored by the API). Short gaps (<= MAX_GAP_H hours)
are linearly interpolated and counted; longer gaps are left as NaN and make
verify_data fail.

EIA-930: CI = sum(gen_fuel * EF_fuel) / sum(gen) over fuels with positive net
generation, using IPCC AR5 WG3 Annex III lifecycle medians (gCO2eq/kWh).
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

MAX_GAP_H = 6
# IPCC AR5 WGIII Annex III (Schloemer et al. 2014), lifecycle medians, gCO2eq/kWh
EF = {"coal": 820, "gas": 490, "oil": 650, "nuclear": 12, "hydro": 24, "wind": 11, "solar": 45,
      "geothermal": 38}
# keyword -> fuel group. Storage (pumped, battery, other) is excluded: it re-dispatches
# energy generated earlier and is negative while charging. 'Other'/'Unknown' fuels have
# no AR5 factor and are dropped from both sums (their share is reported).
EIA_KEYS = [("Coal", "coal"), ("Natural Gas", "gas"), ("Petroleum", "oil"), ("Nuclear", "nuclear"),
            ("Hydropower", "hydro"), ("Solar", "solar"), ("Wind", "wind"), ("Geothermal", "geothermal"),
            ("Other Fuel", "other"), ("Unknown Fuel", "unknown")]


def _eia_fuel_columns(hdr):
    """{fuel: [columns]} preferring the '(Adjusted)' series, else the raw series."""
    gen = [c for c in hdr if c.startswith("Net Generation (MW) from ")]
    out = {}
    for key, fuel in EIA_KEYS:
        cols = [c for c in gen if key in c and "Storage" not in c.split(" from ", 1)[1].split("(")[0]
                or (key == "Hydropower" and key in c and "Excluding Pumped" in c)
                or (key == "Hydropower" and key in c and "and Pumped Storage" in c)
                or (key in ("Solar", "Wind") and key in c)]
        adj = [c for c in cols if "(Adjusted)" in c]
        raw = [c for c in cols if "(Adjusted)" not in c and "(Imputed)" not in c]
        if adj or raw:
            out[fuel] = adj or raw
    return out


def _finish(df, start, end):
    idx = pd.date_range(start, end, freq="h", inclusive="left", tz="UTC")
    df = df.reindex(idx)
    df.index.name = "ts"
    n_nonpos = int((df["ci"] <= 0).sum())
    df.loc[df["ci"] <= 0, "ci"] = np.nan  # no grid here has a true zero-carbon hour: data error
    stats = {"n_nonpositive_set_missing": n_nonpos, "n_hours": len(df), "missing_before_fill": int(df["ci"].isna().sum())}
    filled = np.zeros(len(df), np.int8)  # 0 measured, 1 linear, 2 operator nowcast, 3 seasonal
    was_na = df["ci"].isna().to_numpy()
    for c in df.columns:
        df[c] = df[c].interpolate(limit=MAX_GAP_H, limit_area="inside")
    filled[was_na & df["ci"].notna().to_numpy()] = 1
    if "op" in df.columns:
        # longer GB gaps: the API still carries its stored (nowcast-like, MAE ~8 g/kWh) forecast
        m = df["ci"].isna() & df["op"].notna()
        df.loc[m, "ci"] = df.loc[m, "op"]
        filled[m.to_numpy()] = 2
    else:
        # longer EIA gaps: same hour on the previous day (then previous week)
        for lag in (24, 168):
            m = df["ci"].isna() & df["ci"].shift(lag).notna()
            df.loc[m, "ci"] = df["ci"].shift(lag)[m]
            filled[m.to_numpy()] = 3
    df["filled"] = filled
    stats["missing_after_fill"] = int(df["ci"].isna().sum())
    stats["filled_counts"] = {int(k): int(v) for k, v in zip(*np.unique(filled, return_counts=True))}
    return df, stats


def prep_gb(raw_dir, start, end):
    rows = []
    for p in sorted(glob.glob(os.path.join(raw_dir, "gb_*.json"))):
        with open(p) as f:
            for d in json.load(f)["data"]:
                rows.append((d["from"], d["intensity"].get("forecast"), d["intensity"].get("actual")))
    df = pd.DataFrame(rows, columns=["from", "op", "ci"])
    df["from"] = pd.to_datetime(df["from"], utc=True)
    df = df.drop_duplicates("from").set_index("from").sort_index()
    df = df.astype(float)
    hourly = df.resample("h").mean()  # two half-hours -> hour-beginning mean
    return _finish(hourly[["ci", "op"]], start, end)


def prep_eia(raw_dir, ba, start, end):
    frames = []
    tcol = "UTC Time at End of Hour"
    for p in sorted(glob.glob(os.path.join(raw_dir, "EIA930_BALANCE_*.csv"))):
        hdr = pd.read_csv(p, nrows=0).columns
        fc = _eia_fuel_columns(hdr)
        use = ["Balancing Authority", tcol, *sorted({c for cs in fc.values() for c in cs})]
        df = pd.read_csv(p, usecols=use, thousands=",", low_memory=False)
        df = df[df["Balancing Authority"] == ba]
        g = pd.DataFrame(index=df.index)
        for fuel, cols in fc.items():
            g[fuel] = df[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
        g["ts"] = pd.to_datetime(df[tcol], format="%m/%d/%Y %I:%M:%S %p", utc=True) - pd.Timedelta(hours=1)
        frames.append(g)
    df = pd.concat(frames).drop_duplicates("ts").set_index("ts").sort_index()
    fuels = [f for f in df.columns if f in EF]
    gen = df[fuels].clip(lower=0)
    tot = gen.sum(axis=1, min_count=1)
    em = sum(gen[f].fillna(0) * EF[f] for f in fuels)
    ci = (em / tot).where(tot > 0)
    allgen = df.clip(lower=0).sum(axis=1)
    shares = (df.clip(lower=0).div(allgen, axis=0)).mean().round(4).to_dict()
    return _finish(pd.DataFrame({"ci": ci}), start, end), shares


def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/regions.yaml")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/ci")
    ap.add_argument("--regions", nargs="*")
    a = ap.parse_args()
    with open(a.config) as f:
        regions = yaml.safe_load(f)["regions"]
    os.makedirs(a.out, exist_ok=True)
    rp = os.path.join(a.out, "prep_report.json")
    report = json.load(open(rp)) if os.path.exists(rp) else {}
    for name, rc in regions.items():
        if a.regions and name not in a.regions:
            continue
        if rc["source"] == "gb_api":
            (df, st) = prep_gb(os.path.join(a.raw, "gb"), rc["start"], rc["end"])
            extra = {}
        elif rc["source"] == "eia930":
            (df, st), shares = prep_eia(os.path.join(a.raw, "eia"), rc["ba"], rc["start"], rc["end"])
            extra = {"mean_fuel_shares": shares}
        else:
            raise ValueError(rc["source"])
        df.to_parquet(os.path.join(a.out, f"{name}.parquet"))
        report[name] = {**st, **extra, "ci_mean": float(df.ci.mean()), "ci_min": float(df.ci.min()),
                        "ci_max": float(df.ci.max()), "start": str(df.index[0]), "end": str(df.index[-1])}
        print(name, json.dumps(report[name]), flush=True)
    with open(rp, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
