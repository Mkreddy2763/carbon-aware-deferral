"""Fetch GB national carbon intensity (NESO operator forecast + actual) from
api.carbonintensity.org.uk. Resumable: one cached JSON per chunk; existing
chunks are skipped. Polite: fixed delay between requests, retry with backoff.

Usage: python -m carbonsched.data.fetch_gb --out data/raw/gb --start 2023-01-01 --end 2026-01-01
"""
import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone

import requests

API = "https://api.carbonintensity.org.uk/intensity/{frm}/{to}"
CHUNK_DAYS = 7  # the API caps a single range request; 7 days is well inside the cap


def _fmt(d):
    return d.strftime("%Y-%m-%dT%H:%MZ")


def fetch(out_dir, start, end, delay=1.0, max_retries=6, session=None):
    os.makedirs(out_dir, exist_ok=True)
    s = session or requests.Session()
    s.headers["Accept"] = "application/json"
    cur = start
    n_new = 0
    while cur < end:
        nxt = min(cur + timedelta(days=CHUNK_DAYS), end)
        path = os.path.join(out_dir, f"gb_{cur:%Y%m%d}_{nxt:%Y%m%d}.json")
        if not os.path.exists(path):
            url = API.format(frm=_fmt(cur), to=_fmt(nxt))
            for attempt in range(max_retries):
                try:
                    r = s.get(url, timeout=60)
                    if r.status_code == 200:
                        data = r.json()
                        if "data" not in data:
                            raise ValueError(f"unexpected payload: {str(data)[:200]}")
                        tmp = path + ".tmp"
                        with open(tmp, "w") as f:
                            json.dump({"url": url, "fetched_utc": datetime.now(timezone.utc).isoformat(),
                                       "data": data["data"]}, f)
                        os.replace(tmp, path)
                        n_new += 1
                        break
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                except Exception as e:  # noqa: BLE001 - retry any transport/API error
                    wait = min(300, 5 * 2 ** attempt)
                    print(f"[gb] {url} attempt {attempt + 1} failed: {e}; sleeping {wait}s", flush=True)
                    time.sleep(wait)
            else:
                raise SystemExit(f"[gb] giving up on {url}")
            time.sleep(delay)
        cur = nxt
    print(f"[gb] done; {n_new} new chunks in {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/gb")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2026-01-01")
    ap.add_argument("--delay", type=float, default=1.0)
    a = ap.parse_args()
    st = datetime.fromisoformat(a.start).replace(tzinfo=timezone.utc)
    en = datetime.fromisoformat(a.end).replace(tzinfo=timezone.utc)
    fetch(a.out, st, en, a.delay)


if __name__ == "__main__":
    main()
