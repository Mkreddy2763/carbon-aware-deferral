#!/usr/bin/env bash
# Horizon-resolved NESO forecasts (nmpowell/carbon-intensity-forecast-tracking, MIT).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw/gb_archive
for ym in 2023-03 2023-04 2023-05 2023-06 2023-07 2023-08 2023-09 2023-10 2023-11 2023-12 2024-01 2024-02; do
  f=data/raw/gb_archive/national_$ym.sqlite
  [ -s "$f" ] && continue
  curl -sSfL --retry 5 -o "$f.tmp" \
    "https://raw.githubusercontent.com/nmpowell/carbon-intensity-forecast-tracking/main/data/db/${ym%-*}/national_$ym.sqlite"
  mv "$f.tmp" "$f"
done
ls -la data/raw/gb_archive
