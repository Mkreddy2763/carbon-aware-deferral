"""Normalise the Alibaba PAI GPU trace (2020) into data/trace/jobs.parquet.

Conventions (see notes/trace_conventions.md):
* Schedulable unit = a GPU *task* (job_name, task_name). Most PAI jobs have one GPU
  task (the workers); CPU-only tasks (parameter servers, evaluators with
  plan_gpu 0/NaN) draw no GPU power and are dropped.
* plan_gpu is a percentage of one GPU per instance (100 = 1 GPU, 50 = half).
  GPUs held = inst_num * plan_gpu / 100.
* Keep status == 'Terminated' (successful) with finite start < end.
* submit_s = task start_time (the trace's own queueing delay is replaced by the
  simulator's); dur_s = end_time - start_time.
* util = mean over the task's instances of gpu_wrk_util / plan_gpu, clipped to
  [0, 1]; NaN if no sensor row (the energy model then uses 0.7).
* Trace timestamps are shifted by an unknown constant but read as Unix seconds in
  UTC+8 they keep the real time of day and day of week; monday0_s is the first
  Monday 00:00 (UTC+8) of the trace.
"""
import argparse
import json
import os
import tarfile

import numpy as np
import pandas as pd

TZ_OFFSET_S = 8 * 3600


def _read(raw_dir, name):
    with open(os.path.join(raw_dir, name + ".header")) as f:
        cols = f.read().strip().split(",")
    with tarfile.open(os.path.join(raw_dir, name + ".tar.gz")) as tf:
        member = [m for m in tf.getmembers() if m.name.endswith(".csv")][0]
        return pd.read_csv(tf.extractfile(member), header=None, names=cols, low_memory=False)


def normalise(raw_dir, out_dir):
    task = _read(raw_dir, "pai_task_table")
    stats = {"tasks_total": len(task)}
    task = task[task.status == "Terminated"]
    stats["tasks_terminated"] = len(task)
    task = task[np.isfinite(task.start_time) & np.isfinite(task.end_time) & (task.end_time > task.start_time)]
    task = task[task.plan_gpu.fillna(0) > 0]
    stats["gpu_tasks_valid"] = len(task)
    task = task.assign(gpus=task.inst_num * task.plan_gpu / 100.0,
                       dur_s=task.end_time - task.start_time)
    util = None
    if os.path.exists(os.path.join(raw_dir, "pai_sensor_table.tar.gz")):
        sen = _read(raw_dir, "pai_sensor_table")[["job_name", "task_name", "gpu_wrk_util"]]
        util = sen.groupby(["job_name", "task_name"], as_index=False).gpu_wrk_util.mean()
        task = task.merge(util, on=["job_name", "task_name"], how="left")
        task["util"] = (task.gpu_wrk_util / task.plan_gpu).clip(0, 1)
    else:
        task["util"] = np.nan
    stats["util_coverage"] = float(task.util.notna().mean())
    t0 = task.start_time.min()
    local = pd.to_datetime(task.start_time + TZ_OFFSET_S, unit="s")
    first = pd.Timestamp(local.min()).normalize()
    monday = first + pd.Timedelta(days=(7 - first.dayofweek) % 7)
    monday0_s = (monday - pd.Timestamp(0)).total_seconds() - TZ_OFFSET_S - t0
    jobs = pd.DataFrame({"submit_s": task.start_time - t0, "dur_s": task.dur_s,
                         "gpus": task.gpus, "util": task.util,
                         "gpu_type": task.gpu_type.astype(str)}).sort_values("submit_s")
    jobs = jobs.reset_index(drop=True)
    # weekly profile check: submissions per local weekday
    dow = pd.to_datetime(jobs.submit_s + t0 + TZ_OFFSET_S, unit="s").dt.dayofweek
    stats.update({
        "jobs": len(jobs), "t0_raw": float(t0), "span_days": float(jobs.submit_s.max() / 86400),
        "monday0_s": float(monday0_s), "first_local_day": str(first.date()),
        "submissions_by_weekday": dow.value_counts().sort_index().tolist(),
        "gpus_quantiles": jobs.gpus.quantile([0.5, 0.9, 0.99, 1]).round(3).tolist(),
        "dur_h_quantiles": (jobs.dur_s.quantile([0.5, 0.9, 0.99, 1]) / 3600).round(3).tolist(),
        "frac_jobs_ge_1h": float((jobs.dur_s >= 3600).mean()),
        "frac_gpu_hours_ge_1h": float((jobs.gpus * jobs.dur_s)[jobs.dur_s >= 3600].sum() /
                                      (jobs.gpus * jobs.dur_s).sum()),
        "util_mean": float(jobs.util.mean()),
    })
    os.makedirs(out_dir, exist_ok=True)
    jobs.to_parquet(os.path.join(out_dir, "jobs.parquet"), index=False)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(stats, f, indent=1)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/alibaba")
    ap.add_argument("--out", default="data/trace")
    a = ap.parse_args()
    print(json.dumps(normalise(a.raw, a.out), indent=1))


if __name__ == "__main__":
    main()
