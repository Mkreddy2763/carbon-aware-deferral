# GB carbon-intensity forecast archive: nmpowell/carbon-intensity-forecast-tracking

Checked on 2026-09-24 via the GitHub API and raw.githubusercontent.com. About 0.6 MB was downloaded.

## Verdict for our study

**This repo does not cover 2025-08-01 → 2025-12-31.** Scraping stopped on 2024-02-03. GitHub then
auto-disabled the cron. Scraping resumed on 2026-07-17 (v2.0.0). No commits exist between
2024-01-25 and 2026-07-17. `analysis.sqlite.stats_history` confirms a gap from 2024-02-02 to 2026-07-17,
with zero 2025 dates. None of the side branches holds 2025 data (latest: `sqlite-conversion-1`
2025-01-25 dev commit, `claude/sqlite-schema-design-qx061` 2026-02-22 schema only).
We need another source for Aug–Dec 2025 forecast vintages, or we must move the GB window to
2023-03→2024-02 or 2026-07→present.

## 1. Licence, README, layout

- **Licence:** MIT License, "Copyright (c) 2023 Nick Powell" (`LICENSE`, 1068 B).
  The data are scraped from the NESO Carbon Intensity API, which is CC BY 4.0 upstream. Cite both.
- **README:** tracks NESO's carbon-intensity forecast against the eventual value. The API keeps no
  forecast history, so GitHub Actions scrapes all five endpoints twice an hour.
  Each half-hour window is forecast about 96 times over the preceding 48 h. Blog post: nickmp.com/posts/carbon-intensity-forecast-tracking.
- **Storage since v2.0.0 (2026-07-17, ADR-001):** SQLite files only. There are no JSON/CSV data files.
  The 19 GB legacy JSON/CSV tree (Mar 2023–Feb 2024) was migrated into SQLite. After that the
  **git history was rewritten to a fresh root**, so the old raw files are gone from the repo.
- **What is archived:** national 48 h forward forecasts (`national_fw48h`) plus the past 24 h
  (`national_pt24h`, which gives actuals), regional forecasts with the generation mix (DNO regions),
  and the national generation mix.
- **Layout** (`data/db/`, partitioned by *window* time):
  - `<YYYY>/national_<YYYY-MM>.sqlite`: full fidelity, 0.4–4.2 MB/month
  - `<YYYY>/regional_<YYYY-MM>{a,b}.sqlite`: half-month change-log, 4–37 MB each
  - `<YYYY>/generation_<YYYY>.sqlite`: change-log
  - `analysis.sqlite` (daily stats), `reference.sqlite` (CI index bands, region names, NGESO 2017–23 history, 2.3 MB)
  - `inbox/snap_<YYYY-MM-DDTHHMMZ>.sqlite`: about 131 KB per scrape, folded in by the daily job
- Files present: 2023-03…2023-12 and 2024-01, 2024-02 (national about 1.9–4.2 MB/month, regional about 45 MB/month),
  then 2026-07, 08, 09. There is **no 2025 directory**.

### Example: `data/db/2026/national_2026-09.sqlite` (471 KB)

Timestamps are unix seconds UTC.

```
national_intensity(window_utc INT, capture_utc INT, forecast INT, actual INT, PK(window_utc,capture_utc))
  (1788220800, 1788067800, 136, None)   # window 2026-09-01 00:00Z, captured 2026-08-30 05:30Z
  (1788220800, 1788076800, 137, None)
  (1788220800, 1788089400, 137, None)
  22782 rows; 1148 windows; 179 distinct captures
captures(capture_utc, endpoint, window_first_utc, window_last_utc, observed_utc, source)
  (1788067800,'national_fw48h',1788067800,1788238800,1788068869,'live')
regional_intensity(window_utc, region_id, capture_utc, forecast, biomass..wind)   # empty in national files
generation_mix(window_utc, capture_utc, biomass..wind)                            # empty in national files
capture_gaps(capture_utc, endpoint, window_utc, region_id)
```

`capture_utc` is the issue time and `window_utc - capture_utc` is the lead time, so these files fit forecast-vintage work directly.

## 2. Coverage and completeness

- First data 2023-03-14. There are small holes (2023-03-30→04-03), then continuous data to 2024-02-02.
  The big gap runs from 2024-02-02 to 2026-07-17. Data resumes 2026-07-17 and runs to now (2026-09-24).
- Actual cadence in 2026: GitHub cron throttling gives only about **6–10 national_fw48h captures per day**
  (Sep 2026), not 48. The README says about 90% capture is "normal". This is sparse, but each capture
  covers 48 h ahead, so every window still has about 15–20 vintages.
- Aug–Dec 2025: **0 files and 0 bytes. Not covered.**

## 3. Pre-processed / summary data

- `https://raw.githubusercontent.com/nmpowell/carbon-intensity-forecast-tracking/main/data/db/analysis.sqlite` (80 KB):
  `stats_history` (daily absolute and % error mean/sem/CI, 390 days), `error_probabilities`, and
  `all_data_error_summary`. It has no per-horizon breakdown and no 2025 rows.
- The national monthly SQLite files are already the compact "all forecasts" form (about 4 MB/month).
  No release assets or parquet files exist. Releases v1.0.0, v1.0.1, v2.0.0 and v2.0.1 are tags only.

## 4. Download strategy (for periods that are covered)

URL pattern for raw files (no API quota):

```
https://raw.githubusercontent.com/nmpowell/carbon-intensity-forecast-tracking/main/data/db/<YYYY>/national_<YYYY-MM>.sqlite
```

For example, 2023-08 through 2023-12 is about 19 MB in total. List the files with one API call:
`https://api.github.com/repos/nmpowell/carbon-intensity-forecast-tracking/git/trees/main:data/db?recursive=1`.
Alternatively, run `git clone --filter=blob:none --sparse <repo>` and then `git sparse-checkout set data/db/2023`. A whole clone is about 0.5 GB or more.

To query the data:
`SELECT window_utc, capture_utc, forecast, actual FROM national_intensity`
Take the actual from the latest capture that has a non-null `actual`, or use NESO final values.

## Alternatives for Aug–Dec 2025 (not verified here)

Look for other forecast-vintage archives, such as NESO data-portal "historic day-ahead/2-day-ahead
carbon intensity forecast" datasets or Elexon/BMRS-based reconstructions. The other option is to
evaluate the GB forecast on 2026-07→2026-09 from this repo.
