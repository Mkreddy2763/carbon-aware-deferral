# When Carbon-Aware Deferral Backfires

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24170183.svg)](https://doi.org/10.5281/zenodo.24170183)

Code, configurations and analysis for the PCDS 2026 paper

> **When Carbon-Aware Deferral Backfires: Contention, Not Forecast Error, Drives Harm in GPU Workload Shifting**
> Bilakanti Mukesh Reddy, Manda Siddharth, Akhil Varun Varma Kalidindi, Rishi Kamishetty,
> Lakshmi Pavan Rangisetti, Veeraiah Talagondapati.
> Dept. of Computer Science and Engineering, École Centrale School of Engineering, Mahindra University.

Carbon-aware schedulers defer GPU jobs to cleaner hours. This repository replays the Alibaba PAI GPU
trace (2020) against six electricity grids and measures what happens to each deferred job. A deferred
job is *harmed* if it emits more than it would have by running on arrival. Every harmed job's excess is
split exactly into forecast error, re-planning and contention. Even a clairvoyant greedy scheduler harms
13%, 29% and 41% of the jobs it shifts at 50%, 70% and 90% utilisation, and most harm comes from
contention among deferred jobs rather than from forecast error.

## What is here

| Path | Contents |
|---|---|
| `carbonsched/` | Python package: data download and preparation, forecasting (LightGBM, split-conformal, ACI), cluster simulator, scheduling policies P0–P9, metrics and harm decomposition |
| `configs/` | Regions and the experiment grids (`experiments.yaml` v1, `experiments_v2.yaml` v2) |
| `analysis/` | `make_tables.py` (source of every number in the paper) and `make_figs_v2.py` |
| `tests/` | Self-test of simulator invariants, conformal calibration and the exact harm decomposition |
| `paper/` | LaTeX sources; numbers and tables are generated into `paper/numbers.tex` and `paper/tables/` |
| `cluster/` | Optional Slurm scripts, and `MANIFEST.json` with checksums of the paper's inputs |
| `data/sample/` | Small samples for the self-test |
| `reproduce.py` | One-command driver for every stage |

## Quick start

```
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.lock  # Python >= 3.11   (Python 3.10: requirements-py310.txt)
python reproduce.py check
```

Rebuild the paper's tables and figures from its own simulation results (about 2 minutes): download
`carbonsched_results.zip` from the [Releases](../../releases) page, unzip it in this folder, then run

```
python reproduce.py tables figs paper
```

Re-run everything with the paper's exact inputs (the full sweep takes about 6–10 hours on 8–20 cores):
download `carbonsched_data_prepared.zip` from the Releases page, then run

```
python reproduce.py all --prepared-data carbonsched_data_prepared.zip
```

Details of every stage, the run times, determinism across platforms and the Windows path-length note
are in [README_REPRODUCE.md](README_REPRODUCE.md).

## Data

The release assets hold the paper's exact inputs and outputs:

- `carbonsched_data_prepared.zip`: normalised carbon-intensity series, the prepared job trace, fitted
  forecast caches and the archived GB operator forecasts.
- `carbonsched_results.zip`: the raw results of all 9,900 simulation cells (`results/raw`, `results/raw_v2`).

The inputs come from third-party sources, which keep their own terms of use: the Alibaba PAI GPU cluster
trace (`alibaba/clusterdata`), the GB Carbon Intensity API (NESO), the archive of its 48-hour forecasts
(`nmpowell/carbon-intensity-forecast-tracking`) and the U.S. EIA-930 hourly grid monitor.
`python reproduce.py data` downloads them from the original sources instead.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`). The code is archived on Zenodo:
[doi:10.5281/zenodo.24170183](https://doi.org/10.5281/zenodo.24170183).

## Licence

The code in this repository is released under the MIT licence (see `LICENSE`). Third-party data keeps
its original terms.
