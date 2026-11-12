# Alibaba PAI GPU trace 2020: conventions adopted

Source: `alibaba/clusterdata/cluster-trace-gpu-v2020`, tables `pai_task_table`, `pai_sensor_table`
(`pai_job_table` downloaded for provenance). CSVs have no header row; column names come from the
`<table>.header` files in the repo.

* **Unit of scheduling = GPU task** (`job_name`, `task_name`). CPU-only tasks (`plan_gpu` 0/NaN, e.g.
  parameter servers) draw no GPU power and are dropped.
* **`plan_gpu` is a percentage of one GPU per instance** (100 = 1 GPU, 50 = half a GPU; the README's
  convention, confirmed by the value distribution). GPUs held = `inst_num * plan_gpu / 100`. Getting this wrong
  would rescale every emission number by 100x.
* Keep `status == Terminated` (successful) with finite `start_time < end_time`.
  Result: 1,261,050 tasks → 885,073 terminated → **732,691 valid GPU tasks**.
* `submit_s = start_time` (the trace's own queueing is replaced by the simulator's). `dur_s = end - start`.
  Median duration 0.17 h, p90 3.5 h, p99 19 h, max 174 h. Durations are capped at 7 days (affects < 0.1% of
  GPU-hours).
* Tasks ≥ 1 h are 25.6% of tasks but **93.6% of GPU-hours**, so the deferrable set carries almost all energy.
* **Utilisation** = mean over instances of `gpu_wrk_util / plan_gpu`, clipped to [0, 1]. The sensor table is one
  lifetime-averaged row per instance. Coverage is 80.2% of tasks and the mean is 0.19, which is low but
  consistent with the known under-utilisation in PAI. Tasks with no sensor row use 0.7 (spec default).
* **Timestamps** are shifted by an unknown constant, but read as Unix seconds in UTC+8 they keep the real time of
  day and day of week. Submissions per weekday (Mon..Sun): 115k, 107k, 110k, 119k, 107k, 86k, 89k. The weekend dip
  is where it should be, so we align the trace's local hour-of-week with each region's local hour-of-week.
* Trace weeks 1–7 (counted from the first Monday) are stable at ~520–600k GPU-hours per week. Week 0 is partial
  and week 8 starts tapering. Offset *o* replays trace week `1 + (o mod 7)` against carbon test week *o*.
