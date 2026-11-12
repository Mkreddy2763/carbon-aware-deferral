"""Workstation-side helper: which array groups still have unfinished cells?

The login node only runs sbatch/squeue/ls, so the group bookkeeping happens here.
  ssh dgx2 'ls ~/carbonsched/results/raw' > done.txt
  python cluster/remote.py --done done.txt E1 E2 ...    -> prints an --array spec (e.g. 0-12,15,40-44)
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from carbonsched.run import expand, groups, load_cfg  # noqa: E402


def compress(ix):
    out, i = [], 0
    while i < len(ix):
        j = i
        while j + 1 < len(ix) and ix[j + 1] == ix[j] + 1:
            j += 1
        out.append(str(ix[i]) if i == j else f"{ix[i]}-{ix[j]}")
        i = j + 1
    return ",".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="+")
    ap.add_argument("--done", help="file listing results/raw on the cluster")
    ap.add_argument("--config", default="configs/experiments.yaml")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    cfg["regions"] = yaml.safe_load(open(cfg.get("regions_file", "configs/regions.yaml")))["regions"]
    done = set()
    if a.done:
        done = {l.strip()[:-len(".parquet.done")] for l in open(a.done) if l.strip().endswith(".parquet.done")}
    cells = [c for e in a.exps for c in expand(e, cfg)]
    gs = groups(cells, cfg.get("group_by", ["exp", "region", "policy", "slack_h"]))
    todo = [i for i, g in enumerate(gs) if any(c["cell_id"] not in done for c in g)]
    n_cells = sum(c["cell_id"] not in done for c in cells)
    print(f"# {len(cells)} cells in {len(gs)} groups; {n_cells} cells in {len(todo)} groups to run", file=sys.stderr)
    print(compress(todo))


if __name__ == "__main__":
    main()
