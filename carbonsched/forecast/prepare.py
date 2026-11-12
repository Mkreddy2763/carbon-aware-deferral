"""Fit forecasters once per region and cache forecasts + conformal bounds.

Output: data/forecasts/<REGION>.npz and data/forecasts/<REGION>_quality.json.
Simulation cells load the npz read-only; nothing is refit per cell.

Usage: python -m carbonsched.forecast.prepare --region GB [--config configs/regions.yaml]
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import yaml

from . import conformal as cf
from . import models as fm

ALPHAS = (0.1, 0.2)
H = fm.H


def periods(ts, test_months, calib_months):
    end = ts[-1] + pd.Timedelta(hours=1)
    test_start = (end - pd.DateOffset(months=test_months)).normalize()
    calib_start = (test_start - pd.DateOffset(months=calib_months)).normalize()
    return calib_start, test_start, end


def prepare(region, cfg, data_dir="data", seed=0, n_jobs=-1, gamma=0.005):
    t0 = time.time()
    df = pd.read_parquet(os.path.join(data_dir, "ci", f"{region}.parquet"))
    ts = df.index
    y = df["ci"].to_numpy(dtype=float)
    tz = cfg["tz"]
    calib_start, test_start, end = periods(ts, cfg.get("test_months", 3), cfg.get("calib_months", 2))
    pos = np.arange(len(ts))
    i_cal = pos[(ts >= calib_start) & (ts < test_start)]
    i_test = pos[ts >= test_start]
    i_train = pos[(ts < calib_start) & (pos >= 168)]
    issue = np.concatenate([i_cal, i_test])
    is_test = np.r_[np.zeros(len(i_cal), bool), np.ones(len(i_test), bool)]
    Y = fm.target_matrix(y, issue)
    # calibration residuals must not peek into the test period
    Y_cal = Y.copy()
    Y_cal[~is_test] = np.where(issue[~is_test, None] + np.arange(1, H + 1) < i_test[0],
                               Y[~is_test], np.nan)

    out = {"y": y.astype(np.float64), "ts0_hours": np.int64(ts[0].value // 3_600_000_000_000),
           "issue": issue.astype(np.int64), "is_test": is_test}
    quality = {"region": region, "calib_start": str(calib_start), "test_start": str(test_start),
               "end": str(end), "n_train_issue": int(len(i_train)), "n_cal": int(len(i_cal)),
               "n_test": int(len(i_test)), "models": {}}
    F = {"F0": fm.seasonal_naive(y, issue)}
    model = fm.fit_lgbm(y, ts, tz, i_train, seed=seed, n_jobs=n_jobs)
    F["F1"] = fm.predict_lgbm(model, y, ts, tz, issue)
    if "op" in df.columns:
        # the API's single stored forecast per slot (lead time unknown, nowcast-like)
        F["F2s"] = fm.operator(df["op"].to_numpy(dtype=float), issue)
    if cfg.get("op_archive"):
        # horizon-resolved operator forecasts (archived fw48h captures)
        lf = pd.read_parquet(os.path.join(data_dir, "ci", cfg["op_archive"]))
        mat = lf.pivot(index="issue_ts", columns="h", values="fc").reindex(ts[issue])
        mat = mat.reindex(columns=range(1, H + 1)).ffill(axis=1)  # h=48 is never captured: carry h=47
        F["F2"] = mat.to_numpy(dtype=float)
        quality["F2_missing_frac_test"] = float(np.isnan(F["F2"][is_test]).mean())

    for name, Fm in F.items():
        out[f"{name}"] = Fm.astype(np.float32)
        E = Y - Fm
        E_cal = np.where(is_test[:, None], np.nan, Y_cal - Fm)
        E_test = E[is_test]
        mq = {"mae_by_h": np.nanmean(np.abs(E_test), axis=0).round(3).tolist(),
              "mae": float(np.nanmean(np.abs(E_test))),
              "bias": float(np.nanmean(E_test)),
              "mape": float(np.nanmean(np.abs(E_test) / np.maximum(Y[is_test], 1.0)))}
        for a in ALPHAS:
            qlo_s, qhi_s = cf.split_quantiles(E_cal[~is_test], a)
            qlo_a, qhi_a, alo, ahi = cf.aci_bounds(E_cal[~is_test], E_test, a, gamma=gamma)
            full = lambda test_part, split_part: np.vstack(
                [np.broadcast_to(split_part, (int((~is_test).sum()), H)), test_part]).astype(np.float32)
            out[f"{name}_split_{a}_qlo"] = full(np.broadcast_to(qlo_s, E_test.shape), qlo_s)
            out[f"{name}_split_{a}_qhi"] = full(np.broadcast_to(qhi_s, E_test.shape), qhi_s)
            out[f"{name}_aci_{a}_qlo"] = full(qlo_a, qlo_s)
            out[f"{name}_aci_{a}_qhi"] = full(qhi_a, qhi_s)
            for meth, lo, hi in (("split", qlo_s, qhi_s), ("aci", qlo_a, qhi_a)):
                c = cf.coverage(E_test, lo, hi)
                byh = {}
                for lab, sl in (("h1-6", slice(0, 6)), ("h7-24", slice(6, 24)), ("h25-48", slice(24, 48))):
                    lo_b = np.broadcast_to(lo, E_test.shape)[:, sl]
                    hi_b = np.broadcast_to(hi, E_test.shape)[:, sl]
                    byh[lab] = cf.coverage(E_test[:, sl], lo_b, hi_b)
                mq[f"{meth}_{a}"] = {**c, "by_h": byh}
        quality["models"][name] = mq
    os.makedirs(os.path.join(data_dir, "forecasts"), exist_ok=True)
    path = os.path.join(data_dir, "forecasts", f"{region}.npz")
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path)
    quality["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(data_dir, "forecasts", f"{region}_quality.json"), "w") as f:
        json.dump(quality, f, indent=1)
    return quality


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    ap.add_argument("--config", default="configs/regions.yaml")
    ap.add_argument("--data", default="data")
    ap.add_argument("--n-jobs", type=int, default=-1)
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)["regions"][a.region]
    q = prepare(a.region, cfg, a.data, n_jobs=a.n_jobs)
    print(json.dumps({m: {k: v for k, v in d.items() if k in ("mae", "split_0.1", "aci_0.1")}
                      for m, d in q["models"].items()}, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
