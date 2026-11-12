"""Verify prepared data and write cluster/MANIFEST.json. Fails loudly.

Checks per region: full hourly coverage of [start, end); no duplicate timestamps;
CI in (0, 1000] gCO2/kWh; *no* missing hours in the test period (the only period
the scheduler sees) plus a lag week; <= 3% missing in calibration (those residuals
are dropped); documented filling (see prep.py;
at most 1% of evaluation hours may be filled beyond linear interpolation); training-period gaps below a documented
threshold (<= 1% of hours, longest gap <= 96 h; LightGBM skips rows with NaN
features). GB: stored operator forecast aligned with actual (corr > 0.9).
GBop: archived operator forecasts present for >= 99% of test issue hours.
Trace: row counts and plausible durations/GPU counts.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

MAX_TRAIN_GAP_FRAC = 0.01
MAX_TRAIN_GAP_H = 96
MAX_EVAL_FILLED_FRAC = 0.01
MAX_CALIB_MISSING_FRAC = 0.03


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def longest_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def verify(config, data_dir):
    from carbonsched.forecast.prepare import periods
    with open(config) as f:
        regions = yaml.safe_load(f)["regions"]
    errors, report = [], {}
    for name, rc in regions.items():
        p = os.path.join(data_dir, "ci", f"{name}.parquet")
        if not os.path.exists(p):
            errors.append(f"{name}: missing {p}")
            continue
        df = pd.read_parquet(p)
        ts = df.index
        exp = pd.date_range(rc["start"], rc["end"], freq="h", inclusive="left", tz="UTC")
        r = {"rows": len(df), "start": str(ts[0]), "end": str(ts[-1])}
        if not ts.equals(exp):
            errors.append(f"{name}: hourly index does not cover [{rc['start']}, {rc['end']})")
        if ts.duplicated().any():
            errors.append(f"{name}: duplicate timestamps")
        ci = df.ci
        bad = ((ci <= 0) | (ci > 1000)) & ci.notna()
        if bad.any():
            errors.append(f"{name}: {int(bad.sum())} CI values outside (0, 1000]")
        calib_start, test_start, end = periods(ts, rc.get("test_months", 3), rc.get("calib_months", 2))
        ev = ts >= test_start - pd.Timedelta(days=8)  # + a week of lags for features / threshold policy
        n_ev = int(ci[ev].isna().sum())
        if n_ev:
            errors.append(f"{name}: {n_ev} missing hours in test period (+lag week)")
        cal = (ts >= calib_start) & (ts < test_start)
        cal_miss = float(ci[cal].isna().mean())
        r["calib_missing_frac"] = cal_miss  # residuals with missing targets are dropped
        if cal_miss > MAX_CALIB_MISSING_FRAC:
            errors.append(f"{name}: {cal_miss:.2%} of calibration hours missing")
        ev = ev | cal | (ts >= calib_start)
        if "filled" in df:
            f_ev = df.filled[ev]
            r["eval_filled"] = {int(k): int(v) for k, v in f_ev.value_counts().items() if k}
            nonlin = float((f_ev >= 2).mean())
            if nonlin > MAX_EVAL_FILLED_FRAC:
                errors.append(f"{name}: {nonlin:.3%} of evaluation hours filled beyond interpolation")
        tr = ~ev
        frac = float(ci[tr].isna().mean())
        lg = longest_run(ci[tr].isna().to_numpy())
        r.update(train_missing_frac=frac, train_longest_gap_h=lg, eval_missing=n_ev,
                 calib_start=str(calib_start), test_start=str(test_start),
                 ci_min=float(ci.min()), ci_mean=float(ci.mean()), ci_max=float(ci.max()))
        if frac > MAX_TRAIN_GAP_FRAC or lg > MAX_TRAIN_GAP_H:
            errors.append(f"{name}: training gaps {frac:.4f} frac / {lg} h exceed threshold")
        if "op" in df and rc["source"] == "gb_api":
            ok = df.ci.notna() & df.op.notna()
            corr = float(np.corrcoef(df.ci[ok], df.op[ok])[0, 1])
            r["op_actual_corr"] = corr
            if corr < 0.9:
                errors.append(f"{name}: stored forecast/actual misaligned (corr {corr:.3f})")
        if rc.get("op_archive"):
            lf = pd.read_parquet(os.path.join(data_dir, "ci", rc["op_archive"]))
            t = lf[lf.issue_ts >= test_start]
            miss = float(t.fc.isna().mean())
            r["archive_test_missing_frac_before_fill"] = miss
            if t.issue_ts.nunique() < 0.99 * (end - test_start) / pd.Timedelta(hours=1):
                errors.append(f"{name}: archive covers too few test issue hours")
        report[name] = r
    tp = os.path.join(data_dir, "trace", "jobs.parquet")
    if os.path.exists(tp):
        j = pd.read_parquet(tp)
        report["trace"] = {"rows": len(j), "span_days": float(j.submit_s.max() / 86400),
                           "dur_h_max": float(j.dur_s.max() / 3600), "gpus_max": float(j.gpus.max())}
        if len(j) < 100_000 or (j.dur_s <= 0).any() or (j.gpus <= 0).any():
            errors.append("trace: implausible content")
    else:
        errors.append("trace: missing data/trace/jobs.parquet")
    return errors, report


def manifest(data_dir, out, report):
    files = []
    pats = ["raw/gb/*.json", "raw/gb_archive/*.sqlite", "raw/eia/*.csv", "raw/alibaba/*",
            "ci/*.parquet", "trace/*", "forecasts/*"]
    for pat in pats:
        for p in sorted(glob.glob(os.path.join(data_dir, pat))):
            st = os.stat(p)
            files.append({"path": os.path.relpath(p, data_dir).replace("\\", "/"), "bytes": st.st_size,
                          "sha256": sha256(p),
                          "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()})
    m = {"created_utc": datetime.now(timezone.utc).isoformat(), "host": os.environ.get("COMPUTERNAME")
         or os.uname().nodename, "sources": {
            "gb_api": "https://api.carbonintensity.org.uk/intensity/{from}/{to} (7-day chunks)",
            "gb_archive": "https://github.com/nmpowell/carbon-intensity-forecast-tracking (MIT), "
                          "data/db/<YYYY>/national_<YYYY-MM>.sqlite",
            "eia930": "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_<Y>_<H>.csv",
            "alibaba": "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/<table>.tar.gz"},
         "verification": report, "files": files}
    with open(out, "w") as f:
        json.dump(m, f, indent=1)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/regions.yaml")
    ap.add_argument("--data", default="data")
    ap.add_argument("--manifest", default="cluster/MANIFEST.json")
    a = ap.parse_args()
    errors, report = verify(a.config, a.data)
    print(json.dumps(report, indent=1))
    if errors:
        print("VERIFY FAILED:\n  " + "\n  ".join(errors), file=sys.stderr)
        sys.exit(1)
    manifest(a.data, a.manifest, report)
    print(f"VERIFY OK; manifest -> {a.manifest}")


if __name__ == "__main__":
    main()
