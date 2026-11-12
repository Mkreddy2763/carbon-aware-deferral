"""Expand experiment grids into cells and run them.

A cell = (experiment, region-or-group, policy, slack, forecast model, conformal
method, alpha, error multiplier k, start-week offset, ablation knobs). Each cell
writes results/raw/<cell_id>.parquet atomically (tmp + rename) plus a .done
marker; completed cells are skipped, so a preempted job loses at most the cells
it was running.

  python -m carbonsched.run --exp E1 --list            # count cells / groups
  python -m carbonsched.run --exp E1 --group 3 --n-jobs 32   # one array task
  python -m carbonsched.run --exp E1 --n-jobs 20       # everything, locally
"""
import argparse
import itertools
import json
import os
import time
import traceback

import numpy as np
import pandas as pd
import yaml

from .metrics.metrics import summarise
from .policies import registry
from .sim import workload as wlm
from .sim.engine import Engine
from .sim.env import SPH, RegionEnv

_TRACE = {}

DEFAULTS = dict(slack_h=12, model="main", method="aci", alpha=0.1, k=1.0, dur_noise=0.0,
                defer_min_h=1.0, p_gpu=300.0, pue=1.2, util_target=0.65, save_jobs=False)


def load_cfg(path="configs/experiments.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def _trace(data_dir):
    if data_dir not in _TRACE:
        tr = pd.read_parquet(os.path.join(data_dir, "trace", "jobs.parquet"))
        with open(os.path.join(data_dir, "trace", "meta.json")) as f:
            meta = json.load(f)
        _TRACE[data_dir] = (tr, meta)
    return _TRACE[data_dir]


def expand(exp_name, cfg):
    e = cfg["experiments"][exp_name]
    base = {**DEFAULTS, **cfg.get("defaults", {}), **e.get("fixed", {})}
    grid = e["grid"]
    keys = list(grid)
    cells = []
    for vals in itertools.product(*[grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]):
        c = {**base, **dict(zip(keys, vals)), "exp": exp_name}
        pol = registry.get(c["policy"])
        # forecast knobs are irrelevant to forecast-free policies: collapse duplicates
        if pol["rule"] in ("fcfs", "oracle", "threshold"):
            c.update(model="-", method="-", alpha=0.0, k=0.0)
        if pol["rule"] == "cdec":
            c["method"] = "dec"
        if pol["rule"] in ("greedy", "oracle", "fcfs", "threshold", "lam"):
            c.update(method="-" if pol["rule"] != "lam" else c["method"])
            if pol["rule"] != "lam":
                c["alpha"] = 0.0
        c["cell_id"] = cell_id(c)
        cells.append(c)
    uniq = {c["cell_id"]: c for c in cells}
    return list(uniq.values())


def cell_id(c):
    parts = [c["exp"], c["region"], c["policy"], f"S{c['slack_h']:g}", str(c["model"]),
             f"{c['method']}{c['alpha']:g}", f"k{c['k']:g}", f"o{c['offset']}"]
    for key in ("dur_noise", "defer_min_h", "p_gpu", "pue", "util_target"):
        if c[key] != DEFAULTS[key]:
            parts.append(f"{key}{c[key]:g}")
    return "__".join(parts)


def groups(cells, group_by):
    g = {}
    for c in cells:
        g.setdefault(tuple(c[k] for k in group_by), []).append(c)
    return [g[k] for k in sorted(g, key=str)]


def run_cell(c, cfg, data_dir="data", out_dir="results/raw"):
    path = os.path.join(out_dir, c["cell_id"] + ".parquet")
    if os.path.exists(path + ".done"):
        return "skip"
    t0 = time.time()
    regions_cfg = cfg["regions"]
    grp = cfg.get("groups", {}).get(c["region"])
    names = grp["members"] if grp else [c["region"]]
    align_tz = grp["tz"] if grp else regions_cfg[c["region"]]["tz"]
    wcfg = cfg["workload"]
    tr, meta = _trace(data_dir)
    energy = wlm.EnergyModel(p_gpu_w=c["p_gpu"], pue=c["pue"])
    day0 = wcfg["trace_day0"]
    if day0 == "rotate":  # offset o replays trace week 1 + (o mod n) against carbon week o
        day0 = 7 * (1 + int(c["offset"]) % int(wcfg.get("trace_weeks", 7)))
    wl, info = wlm.build(tr, meta["monday0_s"], window_days=wcfg["window_days"],
                         warmup_days=wcfg["warmup_days"], trace_day0=day0,
                         slack_h=c["slack_h"], defer_min_h=c["defer_min_h"],
                         dur_cap_days=wcfg["dur_cap_days"], sample_rate=wcfg["sample_rate"],
                         dur_noise_sigma=c["dur_noise"], n_regions=len(names), energy=energy,
                         seed=wcfg.get("seed", 0))
    # capacity is fixed by the default energy-free demand, identical for all policies
    cap_total = wlm.capacity_for(wl, c["util_target"], wcfg["window_days"])
    cap = cap_total / len(names) if len(names) > 1 else cap_total
    cap = max(cap, float(wl.gpus.max()))
    envs = []
    test_starts = []
    for name in names:
        rc = regions_cfg[name]
        model = rc["main_forecast"] if c["model"] in ("main", "-") else c["model"]
        method = c["method"] if c["method"] not in ("-", "dec") else "split"  # hourly bounds unused by these rules
        # P7 (cdec) never reads hourly bounds: load the cached 0.1 set; its own alpha drives
        # setup_decision_conformal below
        alpha = 0.1 if (not c["alpha"] or c["method"] == "dec") else c["alpha"]
        k = c["k"] if c["k"] else 1.0
        e = RegionEnv(os.path.join(data_dir, "forecasts", f"{name}.npz"), model=model,
                      method=method, alpha=alpha, k_err=k)
        if registry.get(c["policy"])["rule"] == "cdec":
            e.setup_decision_conformal(alpha=c["alpha"] or 0.1)
        test_starts.append(int(e.issue[e.is_test][0]) + e.ts0_hours)
        envs.append(e)
    t_test = pd.Timestamp(max(test_starts) * 3600, unit="s", tz="UTC")
    mon = wlm.region_monday(None, align_tz, t_test)
    sim0 = mon + pd.Timedelta(weeks=int(c["offset"])) - pd.Timedelta(days=wcfg["warmup_days"])
    for e in envs:
        e.p0 = e.hour_pos(sim0)
    pol = registry.get(c["policy"])
    eng = Engine(envs, wl, cap, pol, migration_slots=int(round(wcfg.get("migration_min", 10) / 5)))
    t_sim = time.time()
    start, region = eng.run()
    t_sim = time.time() - t_sim
    diag = dict(start=start, region=region, des_start=eng.des_start, des_region=eng.des_region,
                last_start=eng.last_start, last_region=eng.last_region, n_dec=eng.n_dec,
                n_defer=eng.n_defer, n_move=eng.n_move)
    m, E, E0 = summarise(envs, wl, start, region, cap, eng.decision_s, eng.now_ok, diag=diag)
    row = {**{k: v for k, v in c.items() if k != "save_jobs"}, **m,
           "members": ",".join(names), "sim0_utc": str(sim0), "sim_seconds": t_sim,
           "cell_seconds": time.time() - t0, "trace_day0": day0, "cap": cap,
           **{f"wl_{k}": v for k, v in info.items()}}
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame([row])
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    if c.get("save_jobs"):
        jobs = pd.DataFrame({"submit": wl.submit, "start": start, "dur": wl.dur, "gpus": wl.gpus,
                             "kw": wl.kw, "deferrable": wl.deferrable, "home": wl.home,
                             "region": region, "E": E, "E0": E0, "counted": wl.counted,
                             "now_ok": eng.now_ok, "deadline": wl.deadline,
                             "des_start": eng.des_start, "last_start": eng.last_start,
                             "n_dec": eng.n_dec, "n_defer": eng.n_defer, "n_move": eng.n_move})
        jp = os.path.join(out_dir, "..", "jobs", c["cell_id"] + ".parquet")
        os.makedirs(os.path.dirname(jp), exist_ok=True)
        jobs.to_parquet(jp + ".tmp", index=False)
        os.replace(jp + ".tmp", jp)
    open(path + ".done", "w").close()
    return "ok"


def _safe_run(c, cfg, data_dir, out_dir):
    try:
        return c["cell_id"], run_cell(c, cfg, data_dir, out_dir)
    except Exception:  # noqa: BLE001 - report, keep other cells going
        return c["cell_id"], "FAIL " + traceback.format_exc(limit=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, nargs="+")
    ap.add_argument("--config", default="configs/experiments.yaml")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/raw")
    ap.add_argument("--group", type=int, default=None, help="array task index")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--max-offsets", type=int, default=None,
                    help="quick mode: keep only start-week offsets < N (default: all 10)")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    with open(cfg.get("regions_file", "configs/regions.yaml")) as f:
        cfg["regions"] = yaml.safe_load(f)["regions"]
    cells = [c for e in a.exp for c in expand(e, cfg)]
    if a.max_offsets is not None:
        cells = [c for c in cells if int(c["offset"]) < a.max_offsets]
    gb = cfg.get("group_by", ["exp", "region", "policy", "slack_h"])
    gs = groups(cells, gb)
    if a.list:
        done = sum(os.path.exists(os.path.join(a.out, c["cell_id"] + ".parquet.done")) for c in cells)
        print(json.dumps({"cells": len(cells), "groups": len(gs), "done": done}))
        return
    todo = gs[a.group] if a.group is not None else cells
    todo = [c for c in todo if not os.path.exists(os.path.join(a.out, c["cell_id"] + ".parquet.done"))]
    print(f"[run] {len(todo)} cells to run", flush=True)
    if a.n_jobs == 1:
        res = [_safe_run(c, cfg, a.data, a.out) for c in todo]
    else:
        from joblib import Parallel, delayed
        # big cells first for better packing
        res = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(_safe_run)(c, cfg, a.data, a.out) for c in todo)
    fails = [r for r in res if r[1].startswith("FAIL")]
    for cid, msg in fails:
        print(cid, msg, flush=True)
    print(f"[run] ok={sum(r[1] == 'ok' for r in res)} fail={len(fails)}", flush=True)
    if fails:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
