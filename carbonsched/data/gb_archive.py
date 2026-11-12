"""Horizon-resolved NESO operator forecasts from the nmpowell/carbon-intensity-
forecast-tracking archive (MIT; scrapes api.carbonintensity.org.uk fw48h every
30 min and stores each capture). The API itself keeps only one forecast per slot,
so this archive is the only source of real multi-horizon operator forecast errors.

For issue hour t (hour-beginning, UTC) we use the latest capture with
capture_utc <= t; the forecast for hour t+h is the mean of that capture's two
half-hour windows in that hour. Captures older than MAX_STALE_H are ignored.

Output: long table data/ci/<name>_opfc.parquet with columns issue_ts, h, fc.
"""
import argparse
import glob
import os
import sqlite3

import numpy as np
import pandas as pd

H = 48
MAX_STALE_H = 3


def load(archive_dir):
    frames = []
    for p in sorted(glob.glob(os.path.join(archive_dir, "national_*.sqlite"))):
        with sqlite3.connect(p) as c:
            frames.append(pd.read_sql("select window_utc, capture_utc, forecast from national_intensity "
                                      "where forecast is not null", c))
    df = pd.concat(frames).drop_duplicates(["window_utc", "capture_utc"])
    return df


def build(archive_dir, start, end):
    df = load(archive_dir)
    df["hour"] = df.window_utc // 3600 * 3600
    hourly = df.groupby(["capture_utc", "hour"], as_index=False).forecast.mean()
    caps = np.sort(hourly.capture_utc.unique())
    issue = pd.date_range(start, end, freq="h", inclusive="left", tz="UTC")
    it = issue.as_unit("s").asi8
    k = np.searchsorted(caps, it, side="right") - 1
    cap_for = np.where(k >= 0, caps[np.maximum(k, 0)], -1)
    stale = (it - cap_for) > MAX_STALE_H * 3600
    cap_for = np.where(stale | (k < 0), -1, cap_for)
    req = pd.DataFrame({"issue": np.repeat(it, H), "capture_utc": np.repeat(cap_for, H),
                        "h": np.tile(np.arange(1, H + 1), len(it))})
    req["hour"] = req.issue + 3600 * req.h
    out = req.merge(hourly, on=["capture_utc", "hour"], how="left")
    out = out.rename(columns={"forecast": "fc"})
    out["issue_ts"] = pd.to_datetime(out.issue, unit="s", utc=True)
    stats = {"issues": len(it), "stale_issues": int(stale.sum()),
             "missing_frac": float(out.fc.isna().mean()),
             "captures": int(len(caps))}
    return out[["issue_ts", "h", "fc"]], stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="data/raw/gb_archive")
    ap.add_argument("--name", default="GBop")
    ap.add_argument("--start", default="2023-03-15")
    ap.add_argument("--end", default="2024-02-01")
    ap.add_argument("--out", default="data/ci")
    a = ap.parse_args()
    out, st = build(a.archive, a.start, a.end)
    os.makedirs(a.out, exist_ok=True)
    out.to_parquet(os.path.join(a.out, f"{a.name}_opfc.parquet"), index=False)
    print(st)


if __name__ == "__main__":
    main()
