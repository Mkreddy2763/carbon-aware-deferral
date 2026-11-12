"""Resumable bulk downloads (Alibaba PAI GPU trace 2020, EIA-930 six-month files).

A file is complete when its size equals the server's Content-Length; partial
files are resumed with an HTTP Range request.

  python -m carbonsched.data.download --what alibaba --out data/raw/alibaba
  python -m carbonsched.data.download --what eia --out data/raw/eia
"""
import argparse
import os
import time

import requests

ALI = "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/{}.tar.gz"
ALI_HDR = ("https://raw.githubusercontent.com/alibaba/clusterdata/master/"
           "cluster-trace-gpu-v2020/data/{}.header")
ALI_TABLES = ["pai_job_table", "pai_task_table", "pai_sensor_table"]
EIA = "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_{}_{}.csv"
IDENT = {"Accept-Encoding": "identity"}  # sizes must match the raw Content-Length
EIA_PERIODS = [(y, h) for y in (2023, 2024) for h in ("Jan_Jun", "Jul_Dec")]


def get(url, path, retries=6, chunk=1 << 20):
    for attempt in range(retries):
        try:
            head = requests.head(url, timeout=60, allow_redirects=True, headers=IDENT)
            total = int(head.headers.get("Content-Length", -1))
            have = os.path.getsize(path) if os.path.exists(path) else 0
            if total > 0 and have == total:
                print(f"[dl] complete: {path}", flush=True)
                return path
            hdr = {**IDENT, **({"Range": f"bytes={have}-"} if have and total > 0 else {})}
            with requests.get(url, stream=True, timeout=120, headers=hdr) as r:
                r.raise_for_status()
                mode = "ab" if "Range" in hdr and r.status_code == 206 else "wb"
                with open(path, mode) as f:
                    for b in r.iter_content(chunk):
                        f.write(b)
            if total > 0 and os.path.getsize(path) != total:
                raise IOError(f"size mismatch {os.path.getsize(path)} != {total}")
            print(f"[dl] ok: {path} ({os.path.getsize(path)} bytes)", flush=True)
            return path
        except Exception as e:  # noqa: BLE001
            wait = min(300, 10 * 2 ** attempt)
            print(f"[dl] {url}: {e}; retry in {wait}s", flush=True)
            time.sleep(wait)
    raise SystemExit(f"[dl] failed: {url}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["alibaba", "eia"], required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.what == "alibaba":
        for t in ALI_TABLES:
            get(ALI_HDR.format(t), os.path.join(a.out, t + ".header"))
            get(ALI.format(t), os.path.join(a.out, t + ".tar.gz"))
    else:
        for y, h in EIA_PERIODS:
            get(EIA.format(y, h), os.path.join(a.out, f"EIA930_BALANCE_{y}_{h}.csv"))


if __name__ == "__main__":
    main()
