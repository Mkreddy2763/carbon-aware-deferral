# Data source reconnaissance (probed 2026-09-24 from this workstation)

## 1. Alibaba PAI GPU trace v2020 (cluster-trace-gpu-v2020)

- README: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020
- Download script: https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-gpu-v2020/data/download_data.sh (plain `curl -O` of the OSS URLs below)
- Backup mirror (GitHub LFS, often over quota): https://github.com/qzweng/clusterdata-cluster-trace-gpu-v2020-data
- Trace: >6,500 GPUs on ~1,800 machines, July-August 2020 (~2 months). Paper: Weng et al., "MLaaS in the Wild", NSDI'22.

### Files (HEAD from this machine: all HTTP 200, Last-Modified 2023-03-20)
Base: `https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/<name>.tar.gz`

| file | Content-Length (bytes) | ~MiB |
|---|---|---|
| pai_job_table.tar.gz | 62,065,432 | 59 |
| pai_task_table.tar.gz | 35,514,117 | 34 |
| pai_instance_table.tar.gz | 694,839,139 | 663 |
| pai_sensor_table.tar.gz | 406,119,947 | 387 |
| pai_group_tag_table.tar.gz | 55,064,781 | 53 |
| pai_machine_spec.tar.gz | 30,449 | 0.03 |
| pai_machine_metric.tar.gz | 206,596,175 | 197 |

### Format
- Each `.tar.gz` contains a single `<name>.csv` (verified by Range-fetching the first 64 KB: tar member `pai_task_table.csv` / `pai_job_table.csv`, data rows begin immediately).
- **The CSVs have NO header row.** Headers live in separate files:
  `https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-gpu-v2020/data/<name>.header`
  (the repo's `analysis/utils.py` does `pd.read_csv(file, header=None)` then assigns columns from `.header`).
- SHA-256 checksums for tar.gz, header and csv files are in the README.
- Missing values are empty fields (e.g. end_time empty for Running/Failed jobs; plan_gpu empty for CPU-only tasks such as `ps`).

### Columns (from .header files)
- pai_job_table: `job_name,inst_id,user,status,start_time,end_time`
- pai_task_table: `job_name,task_name,inst_num,status,start_time,end_time,plan_cpu,plan_mem,plan_gpu,gpu_type`
- pai_instance_table: `job_name,task_name,inst_name,worker_name,inst_id,status,start_time,end_time,machine`
- pai_sensor_table: `job_name,task_name,worker_name,inst_id,machine,gpu_name,cpu_usage,gpu_wrk_util,avg_mem,max_mem,avg_gpu_wrk_mem,max_gpu_wrk_mem,read,write,read_count,write_count`
- pai_group_tag_table: `inst_id,user,gpu_type_spec,group,workload`
- pai_machine_spec: `machine,gpu_type,cap_cpu,cap_mem,cap_gpu`
- pai_machine_metric: `worker_name,machine,start_time,end_time,machine_cpu_iowait,machine_cpu_kernel,machine_cpu_usr,machine_gpu,machine_load_1,machine_net_receive,machine_num_worker,machine_cpu`

### Semantics / units (README)
- `plan_gpu`: GPUs requested **in percent, per instance**: 100 = 1 full GPU, 50 = half a GPU (sharing). The task value is inherited by each instance, so job GPUs = sum over tasks of `inst_num * plan_gpu / 100`. utils.py uses `plan_gpu/100`.
- `plan_cpu`: vCPU cores in percent (600 = 6 cores), per instance. `plan_mem`: GB. `gpu_type`: V100, V100M32, P100, T4, MISC (older K40m/K80/M60), etc.
- `gpu_wrk_util` (sensor): GPU used, in percent on the same scale as plan_gpu (50 = 50% of one GPU). `cpu_usage`: same scale as plan_cpu. `avg_mem/max_mem`, `avg_gpu_wrk_mem/max_gpu_wrk_mem`: GB. `read/write`: network bytes.
- Sensor metrics: **one row per instance (keyed by `worker_name`)**, averaged over the instance lifetime (max_* are maxima). Not a time series.
- `status`: 'Running', 'Terminated', 'Failed', 'Waiting'; only 'Terminated' is successful.
- `start_time/end_time`: **seconds**, shifted by an undisclosed constant (NOT seconds since trace start; observed values ~1e6-7e6). Interpreted as Unix time in Asia/Shanghai (UTC+8) they preserve the original time-of-day and day-of-week; dates/months/years are fake. utils.py: `pd.Timestamp(t, unit='s', tz='Asia/Shanghai')`. start_time == 0 is treated as missing.
- Job start_time = submission time; (earliest task start_time - job start_time) = queueing delay.
- `inst_id` should be treated as job_id (1:1 with job_name).

### Joins
- job <-> task / instance / sensor: `job_name` (or `inst_id`).
- instance <-> sensor / machine_metric: `worker_name`.
- instance / sensor <-> machine_spec: `machine`.
- group_tag <-> job: `inst_id`.

## 2. EIA-930 hourly balancing-authority data

### Six-month bulk CSVs (no key; HEAD verified HTTP 200)
Pattern: `https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_<TYPE>_<YYYY>_<Jan_Jun|Jul_Dec>.csv`, TYPE in {BALANCE, INTERCHANGE, SUBREGION}.

| file | Content-Length (bytes) |
|---|---|
| EIA930_BALANCE_2023_Jan_Jun.csv | 41,981,693 |
| EIA930_BALANCE_2023_Jul_Dec.csv | 42,574,041 |
| EIA930_BALANCE_2024_Jan_Jun.csv | 41,742,812 |
| EIA930_BALANCE_2024_Jul_Dec.csv | 47,928,993 |
| EIA930_SUBREGION_2024_Jan_Jun.csv | 26,103,351 |
| EIA930_INTERCHANGE_2024_Jan_Jun.csv | 101,399,174 |

Last-Modified 2026-09-23 (files are regenerated as data get revised).
All-in-one bulk: https://api.eia.gov/bulk/EBA.zip -> 301 -> https://www.eia.gov/opendata/bulk/EBA.zip (691,484,412 bytes; not downloaded).

### BALANCE file header (44 columns)
1-16: `Balancing Authority, Data Date, Hour Number, Local Time at End of Hour, UTC Time at End of Hour, Demand Forecast (MW), Demand (MW), Net Generation (MW), Total Interchange (MW), Sum(Valid DIBAs) (MW), Demand (MW) (Imputed), Net Generation (MW) (Imputed), Total Interchange (MW) (Imputed), Demand (MW) (Adjusted), Net Generation (MW) (Adjusted), Total Interchange (MW) (Adjusted)`

17-43: three blocks (raw, then `(Imputed)`, then `(Adjusted)`) of
`Net Generation (MW) from Coal | from Natural Gas | from Nuclear | from All Petroleum Products | from Hydropower and Pumped Storage | from Solar | from Wind | from Other Fuel Sources | from Unknown Fuel Sources`

44: `Region` (e.g. CAL, MIDW).

- No separate battery or pumped-storage columns in this file: pumped storage is lumped into "Hydropower and Pumped Storage"; storage otherwise lands in Other. Use the `(Adjusted)` columns for cleaned data.
- Datetimes: `MM/DD/YYYY h:mm:ss AM/PM`, both local and UTC, **hour-ending**. Example row:
  `CISO,01/01/2024,1,01/01/2024 1:00:00 AM,01/01/2024 9:00:00 AM,21350,22365,17478,...`
- CISO rows confirmed in the file; ERCO, PJM, MISO are the standard respondent codes in the same files (file sorted alphabetically by BA).
- Parse numbers defensively (`thousands=','`) in case of formatted values.

### EIA API v2
- Requires a key: without `api_key` -> HTTP 403 `API_KEY_MISSING` ("register at https://www.eia.gov/opendata/register.php").
- `api_key=DEMO_KEY` works (rate-limited).
- Route: `https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/?api_key=KEY&frequency=hourly&data[0]=value&facets[respondent][]=CISO&start=2024-06-01T00&end=2024-06-01T23`
  Returns fueltype COL, NG, NUC, OIL, OTH, SUN, WAT, WND; units "megawatthours"; `period` is UTC for `frequency=hourly` (`local-hourly` also exists).

## 3. Electricity Maps historical data
- `https://www.electricitymaps.com/data-portal` and `https://portal.electricitymaps.com/datasets` redirect to `https://app.electricitymaps.com/datasets`, which returns a Cloudflare 403 challenge to curl.
- In a real browser, `https://app.electricitymaps.com/datasets?zone=DE` redirects an anonymous user to the map; the sidebar shows Map / API / Forecast / API reference / Data coverage / Methodology and "Sign in" - no Datasets / download entry without login.
- `data.electricitymaps.com` does **not resolve** (DNS failure) from this machine; the old direct pattern `https://data.electricitymaps.com/<date>/<ZONE>_<YEAR>_hourly.csv` is dead.
- https://www.electricitymaps.com/data now points to "request trial / contact sales" (signup https://app.electricitymaps.com/auth/signup).
- Conclusion: **no public no-login CSV route found**; requires an account (whether a free tier still includes CSV downloads after signup is unverified).

## 4. GB Carbon Intensity API (https://api.carbonintensity.org.uk)
Docs: https://carbon-intensity.github.io/api-definitions/ (send `Accept: application/json`).

- National: `/intensity/{from}/{to}`, `/intensity/{from}/fw24h`, `/intensity/{from}/fw48h`, `/intensity/{from}/pt24h`, `/intensity/date/{date}`, `/intensity/stats/{from}/{to}[/{block}]`, `/intensity/factors`, `/generation/{from}/{to}`.
- Regional: `/regional/intensity/{from}/{to}`, `/regional/intensity/{from}/fw24h|fw48h|pt24h`, `/regional/intensity/{from}/{to}/regionid/{1..17}` (also `/postcode/{pc}`). `/regional/regionid/{id}/intensity/...` is wrong (400 "Missing Authentication Token").
- Max range: docs say 14 days for `/intensity/{from}/{to}` and regional, 30 days for stats. Empirically a 19-day request returned all 882 slots and >31 days is rejected ("greater than 31 days"). Safe choice: 14-day chunks.
- Half-hourly, ISO8601 UTC; results start with the slot ending at `from` (i.e. from - 30 min).
- National rows: `{from, to, intensity:{forecast, actual, index}}` (gCO2/kWh). Regional rows: `intensity:{forecast, index}` + `generationmix` (fuel %) - **no `actual` in regional**.
- History: national available back to 2017-09 (tested 2017-09-26); regional back to 2018-05 (tested 2018-05-10).
- Rate limits: **not documented**.

### fw48h issue-time experiment
- `/intensity/2024-06-01T00:00Z/fw48h`: 97 slots, 2024-05-31T23:30Z .. 2024-06-02T23:30Z
- `/intensity/2024-05-31T12:00Z/fw48h`: 97 slots, 2024-05-31T11:30Z .. 2024-06-02T11:30Z
- 73 overlapping slots, **0 differences in `forecast`**. E.g. 2024-06-01T06:00Z forecast 65 / actual 53 in both; 2024-06-01T18:00Z 68/68; 2024-06-02T06:00Z 79/72. `pt24h` returns the same values too.
- => For historical periods the API stores **one forecast value per half-hour slot**, independent of issue time; `fw48h` on a past date is just a range query. It cannot reconstruct true forecast vintages (e.g. the 48-h-ahead forecast as issued); treat historical `forecast` as a single (probably latest) forecast.
