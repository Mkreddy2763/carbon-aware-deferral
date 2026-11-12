# Reproducing "When Carbon-Aware Deferral Backfires" on an ordinary machine

Everything (data download, forecasting, all ~9,900 simulations, tables, figures, paper) runs from one
script, `reproduce.py`, on Windows, macOS or Linux. No HPC, Slurm or GPU is needed. The `cluster/` folder
is only for an optional Slurm run and can be ignored.

## 1. Install (Python 3.10–3.13)

```
python -m venv .venv
.venv\Scripts\activate          (Windows)      source .venv/bin/activate   (macOS/Linux)
pip install -r requirements.lock          # Python >= 3.11
pip install -r requirements-py310.txt     # Python 3.10 instead
python reproduce.py check
```
Optional: a LaTeX install (TeX Live or MiKTeX) for `paper/main.pdf`, and Windows + MS Word for the
Word version.

## 2. Run

| goal | command | time on 8–20 cores |
|---|---|---|
| everything, from the original sources | `python reproduce.py all` | data ~15 min (about 500 MB of downloads), forecasts ~5 min, **sweep ~6–10 h** |
| everything, with the paper's exact inputs | `python reproduce.py all --prepared-data carbonsched_data_prepared.zip` | sweep ~6–10 h |
| quick pipeline check (2 of 10 weeks, core experiments) | `python reproduce.py all --quick --prepared-data carbonsched_data_prepared.zip` | ~1 h |
| tables/figures/paper from the paper's own results | unzip `carbonsched_results.zip` here, then `python reproduce.py tables figs paper` | ~2 min |

- `--jobs N` sets the number of parallel simulation workers. The default is all CPUs but two.
- The sweep needs about 0.3–0.5 GB of RAM per worker.
- **Every stage resumes.** Each simulation cell writes `results/raw*/<cell>.parquet` plus a `.done` marker,
  so an interrupted run continues where it stopped. Just re-run the same command.
- Individual stages: `check data selftest forecasts sweep tables figs paper word`.

## 3. What each stage does

1. **data**: downloads and normalises the inputs, then verifies them.
   - Sources: the Alibaba PAI GPU trace 2020; the GB Carbon Intensity API (2021–2025); the archived NESO
     48 h forecasts (`nmpowell/carbon-intensity-forecast-tracking`); and EIA-930 (CISO, ERCO, PJM, MISO).
   - Verification (`carbonsched/data/verify.py`) checks for gaps, duplicates and implausible CI.
   - It then compares the download with the paper's snapshot (`cluster/MANIFEST.json`). Grid operators
     revise "actual" CI after the fact, so a fresh pull can differ slightly. `--prepared-data` gives the
     exact, checksummed inputs.
2. **selftest**: 205 synthetic checks on the simulator, conformal calibration and the exact harm
   decomposition.
3. **forecasts**: fits LightGBM per region and calibrates split-conformal and ACI bounds. Output is cached
   in `data/forecasts/`.
4. **sweep**:
   - v1 grid (`configs/experiments.yaml`, 6,100 cells) → `results/raw/`
   - v2 congestion round (`configs/experiments_v2.yaml`, 3,780 cells) → `results/raw_v2/`
5. **tables**: `analysis/make_tables.py` is the single source of every number in the paper. It writes
   `paper/numbers.tex`, `paper/tables/*.tex`, `analysis/out/*.csv`.
6. **figs**: `analysis/make_figs_v2.py` → `paper/figs/*.pdf`.
7. **paper**: `paper/main.pdf`. A number that was not produced shows as a bold **??**, so a partial run
   (e.g. `--quick`) is easy to spot.
8. **word**: the PCDS Word-template version (`paper/word/`). This stage needs MS Word, plus
   `paper/word/template.docx` converted from the conference's `Proceeding-Template.doc`.

## 4. Determinism

Workload sampling, offsets and the jittered policy are all seeded.
- **Same environment:** re-running a cell on the same inputs reproduces every scheduling decision
  exactly.
- **Across environments:** we re-simulated a sample of the paper's cells on Windows / Python 3.13 / numpy
  2.3, against the paper's Linux / Python 3.10 / numpy 2.2 run. Every decision, shifted-job count and harm
  rate was identical. Emission totals agreed to a relative 3×10⁻¹⁶, the floating-point summation order
  between numpy versions, so the rounded numbers in the paper are unaffected.
- **Forecasts:** LightGBM fitting can differ in the last digits across library versions or thread counts.
  That is why `--prepared-data` ships the exact fitted forecast caches the paper used.

**Windows path length.** Result file names are long (up to ~110 characters below the project folder).
Either keep the project at a short path such as `C:\cs\carbonsched`, or enable Windows long paths. Both
`reproduce.py check` and `make_tables.py` stop with a clear message rather than silently skipping files.

## 5. Layout

```
reproduce.py            one-command driver (this file's companion)
carbonsched/            package: data/ forecast/ sim/ policies/ metrics/ run.py
configs/                regions.yaml, experiments.yaml (v1), experiments_v2.yaml (v2)
analysis/               make_tables.py (+ v2_extras.py, latex_tables.py), make_figs_v2.py
tests/selftest.py       invariants
paper/                  LaTeX sources (sections/, refs.bib), word/ converter
cluster/                optional Slurm scripts + MANIFEST.json (checksums of the paper's inputs)
```
