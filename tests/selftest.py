"""Self-test on synthetic data (no downloads needed).

1. Conformal: forecasts with iid noise -> split conformal coverage ~ 1 - alpha;
   ACI tracks the target under a variance shift where split undercovers.
2. Simulator invariants for every policy: no start before submission, capacity
   never exceeded (recomputed from the final schedule), deadlines met when
   durations are exact and capacity is ample, oracle emissions <= every other
   policy with unlimited capacity (per-job optimum is then the global optimum).
3. A toy sweep through carbonsched.run and analysis/make_tables.py.
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from carbonsched.forecast import conformal as cf  # noqa: E402
from carbonsched.forecast.prepare import prepare  # noqa: E402
from carbonsched.metrics.metrics import summarise  # noqa: E402
from carbonsched.policies import registry  # noqa: E402
from carbonsched.sim import workload as wlm  # noqa: E402
from carbonsched.sim.engine import Engine  # noqa: E402
from carbonsched.sim.env import RegionEnv  # noqa: E402

WORK = os.path.join(ROOT, ".selftest")
OK = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    OK.append(msg)
    print("  ok:", msg, flush=True)


def test_conformal():
    rng = np.random.default_rng(0)
    H = 48
    for a in (0.1, 0.2):
        E_cal = rng.normal(0, 10, (1500, H))
        E_test = rng.normal(0, 10, (2000, H))
        lo, hi = cf.split_quantiles(E_cal, a)
        c = cf.coverage(E_test, lo, hi)["coverage"]
        check(abs(c - (1 - a)) < 0.02, f"split conformal coverage {c:.3f} ~ {1 - a} on iid data")
        # variance doubles in test: split undercovers, ACI recovers
        E_shift = rng.normal(0, 20, (3000, H))
        cs = cf.coverage(E_shift, lo, hi)["coverage"]
        qlo, qhi, *_ = cf.aci_bounds(E_cal, E_shift, a, gamma=0.01)
        ca = cf.coverage(E_shift[1000:], qlo[1000:], qhi[1000:])["coverage"]
        check(cs < 1 - a - 0.1 and abs(ca - (1 - a)) < 0.05,
              f"under shift split={cs:.3f} undercovers, ACI={ca:.3f} ~ {1 - a}")


def synth_region(path_dir, name, seed, n_days=400, tz="UTC"):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-02", periods=n_days * 24, freq="h", tz="UTC")
    h = np.arange(len(ts))
    base = 250 + 80 * np.sin(2 * np.pi * (h % 24) / 24 + seed) + 30 * np.sin(2 * np.pi * h / (24 * 7))
    ar = np.zeros(len(ts))
    for i in range(1, len(ts)):
        ar[i] = 0.95 * ar[i - 1] + rng.normal(0, 8)
    y = np.clip(base + ar, 20, None)
    os.makedirs(os.path.join(path_dir, "ci"), exist_ok=True)
    pd.DataFrame({"ci": y}, index=pd.DatetimeIndex(ts, name="ts")).to_parquet(
        os.path.join(path_dir, "ci", f"{name}.parquet"))
    return {"source": "synthetic", "tz": tz, "test_months": 3, "calib_months": 2, "main_forecast": "F1"}


def synth_trace(path_dir, n=6000, seed=0):
    rng = np.random.default_rng(seed)
    days = 42
    submit = np.sort(rng.uniform(0, days * 86400, n))
    dur = np.exp(rng.normal(np.log(1800), 1.5, n)).clip(60, 5 * 86400)
    gpus = rng.choice([0.5, 1, 2, 4, 8], n, p=[0.3, 0.4, 0.15, 0.1, 0.05])
    util = rng.uniform(0.2, 1.0, n)
    os.makedirs(os.path.join(path_dir, "trace"), exist_ok=True)
    pd.DataFrame({"submit_s": submit, "dur_s": dur, "gpus": gpus, "util": util}).to_parquet(
        os.path.join(path_dir, "trace", "jobs.parquet"))
    with open(os.path.join(path_dir, "trace", "meta.json"), "w") as f:
        json.dump({"monday0_s": 0.0}, f)


def schedule_usage(wl, start, region, R, n_slots):
    use = np.zeros((R, n_slots))
    for j in range(len(wl)):
        use[region[j], start[j]:start[j] + wl.dur[j]] += wl.gpus[j]
    return use


def test_sim(data_dir, regions):
    tr = pd.read_parquet(os.path.join(data_dir, "trace", "jobs.parquet"))
    for cap_mode in ("ample", "tight", "infinite"):
        wl, info = wlm.build(tr, 0.0, window_days=7, warmup_days=1, trace_day0=7, slack_h=12,
                             sample_rate=1.0, n_regions=2, seed=0)
        cap = wlm.capacity_for(wl, 0.65, 7) / 2
        if cap_mode == "ample":
            cap *= 3
        elif cap_mode == "infinite":
            cap = 1e9
        totals = {}
        for pname in ["P0", "P1", "P2", "P3", "P4", "P4-l0.5", "P5", "P7", "P8", "P9", "P9-r", "P6", "P6-point", "P6-oracle", "P6-cdec"]:
            envs = []
            for name in ("SYN1", "SYN2"):
                e = RegionEnv(os.path.join(data_dir, "forecasts", f"{name}.npz"), model="F1",
                              method="aci", alpha=0.1)
                e.p0 = int(e.issue[e.is_test][0]) + 24
                if registry.get(pname)["rule"] == "cdec":
                    e.setup_decision_conformal(0.1)
                envs.append(e)
            pol = registry.get(pname)
            eng = Engine(envs, wl, cap, pol)
            start, region = eng.run()
            diag = dict(start=start, region=region, des_start=eng.des_start, des_region=eng.des_region,
                        last_start=eng.last_start, last_region=eng.last_region, n_dec=eng.n_dec,
                        n_defer=eng.n_defer, n_move=eng.n_move)
            m, E, E0 = summarise(envs, wl, start, region, cap, eng.decision_s, eng.now_ok, diag=diag)
            parts = m["harm_excess_forecast_kg"] + m["harm_excess_replan_kg"] + m["harm_excess_contention_kg"]
            check(abs(parts - m["harm_excess_kg"]) <= 1e-6 * max(1.0, abs(m["harm_excess_kg"])),
                  f"[{cap_mode}] {pname}: harm decomposition is exact ({parts:.3f} = {m['harm_excess_kg']:.3f})")
            if pol["rule"] == "oracle":
                check(m["harm_excess_forecast_kg"] <= 1e-9, f"[{cap_mode}] {pname}: oracle forecast part <= 0")
            if cap_mode == "infinite" and not pol["replan"] and pol["rule"] != "jitter":
                check(not m["got_wanted_slot_frac"] < 1.0, f"[{cap_mode}] {pname}: every job gets its wanted slot")
            totals[pname] = m["emissions_kg"]
            check((start >= wl.submit).all(), f"[{cap_mode}] {pname}: no job starts before submission")
            if cap_mode != "infinite":
                use = schedule_usage(wl, start, region, 2, int((start + wl.dur).max()) + 1)
                check(use.max() <= cap + 1e-6, f"[{cap_mode}] {pname}: capacity never exceeded "
                                               f"(peak {use.max():.1f} <= {cap:.1f})")
            if pname == "P5":
                check(m["harm_rate_voluntary"] == 0.0, f"[{cap_mode}] oracle never harms a job it could start now")
            if not pol["spatial"]:
                check((region == wl.home).all(), f"[{cap_mode}] {pname}: temporal policy keeps jobs home")
            if cap_mode in ("ample", "infinite") and pname != "P0":
                check(m["deadline_miss_rate"] == 0.0, f"[{cap_mode}] {pname}: no deadline misses "
                                                      f"with exact durations and ample capacity")
        if cap_mode == "infinite":
            temporal = ["P0", "P1", "P2", "P3", "P4", "P4-l0.5", "P7", "P8", "P9", "P9-r"]
            check(all(totals["P5"] <= totals[p] + 1e-6 for p in temporal),
                  f"oracle P5 <= all temporal policies with unlimited capacity {totals}")
            check(all(totals["P6-oracle"] <= totals[p] + 1e-6 for p in totals),
                  "spatial oracle <= every policy with unlimited capacity")


def test_sweep(data_dir):
    cfg_path = os.path.join(WORK, "experiments.yaml")
    import yaml
    cfg = {
        "regions_file": os.path.join(WORK, "regions.yaml"),
        "workload": {"window_days": 7, "warmup_days": 1, "trace_day0": 7, "dur_cap_days": 7,
                     "sample_rate": 1.0, "seed": 0, "migration_min": 10},
        "groups": {"SYN2R": {"members": ["SYN1", "SYN2"], "tz": "UTC"}},
        "experiments": {"TOY": {"grid": {"region": ["SYN1", "SYN2R"],
                                         "policy": ["P0", "P2", "P3", "P4", "P5", "P7"],
                                         "slack_h": [6, 12], "offset": [0, 1]}}},
    }
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    out = os.path.join(WORK, "results", "raw")
    cmd = [sys.executable, "-m", "carbonsched.run", "--exp", "TOY", "--config", cfg_path,
           "--data", data_dir, "--out", out, "--n-jobs", "4"]
    subprocess.run(cmd, check=True, cwd=ROOT)
    n = len([f for f in os.listdir(out) if f.endswith(".done")])
    check(n == 48, f"toy sweep wrote {n} cells")
    subprocess.run(cmd, check=True, cwd=ROOT)  # resubmission must skip everything
    check(len([f for f in os.listdir(out) if f.endswith(".done")]) == 48, "resubmission skips done cells")
    subprocess.run([sys.executable, "analysis/make_tables.py", "--raw", out,
                    "--out", os.path.join(WORK, "tables")], check=True, cwd=ROOT)
    check(os.path.exists(os.path.join(WORK, "tables", "summary.csv")), "make_tables produced summary.csv")


def main():
    shutil.rmtree(WORK, ignore_errors=True)
    data_dir = os.path.join(WORK, "data")
    print("[selftest] conformal")
    test_conformal()
    print("[selftest] synthetic data + forecasts")
    regions = {n: synth_region(data_dir, n, s) for n, s in (("SYN1", 1), ("SYN2", 2))}
    import yaml
    with open(os.path.join(WORK, "regions.yaml"), "w") as f:
        yaml.safe_dump({"regions": regions}, f)
    for n, rc in regions.items():
        q = prepare(n, rc, data_dir, n_jobs=4)
        c = q["models"]["F1"]["aci_0.1"]["coverage"]
        check(0.85 < c < 0.95, f"{n}: ACI coverage on synthetic AR data {c:.3f} ~ 0.9")
    synth_trace(data_dir)
    print("[selftest] simulator invariants")
    test_sim(data_dir, regions)
    print("[selftest] toy sweep")
    test_sweep(data_dir)
    print(f"[selftest] PASSED ({len(OK)} checks)")


if __name__ == "__main__":
    main()
