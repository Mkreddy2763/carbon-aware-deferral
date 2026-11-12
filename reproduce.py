"""One-command local reproduction of the CarbonSched study (no HPC / Slurm needed).

Stages (run in this order by `all`):
  check      Python version + required packages
  data       download raw data, normalise, verify          (skipped with --prepared-data)
  selftest   synthetic invariants (simulator, conformal, harm decomposition)
  forecasts  fit forecasters + conformal bounds per region  (skipped with --prepared-data)
  sweep      all simulation cells (v1 grid + v2 congestion round), resumable
  tables     analysis/make_tables.py -> paper/numbers.tex + paper/tables/*.tex
  figs       analysis/make_figs_v2.py -> paper/figs/*.pdf
  paper      pdflatex + bibtex -> paper/main.pdf            (needs a LaTeX install)
  word       Word version in the PCDS template              (needs Windows + MS Word)

Examples
  python reproduce.py all                          # full study, downloads everything
  python reproduce.py all --prepared-data data.zip # use the shipped, checksummed inputs
  python reproduce.py all --quick                  # 2 of 10 weekly offsets: pipeline check
  python reproduce.py sweep --jobs 8               # resume the sweep on 8 cores
Every stage is idempotent: finished downloads and simulation cells are skipped on re-run.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
REGIONS = ["GB", "GBop", "CISO", "ERCO", "PJM", "MISO"]
V1_EXPS = ["E1", "E2", "E3", "E4", "E5", "E6a", "E6b", "E6c", "E6d", "E6e", "E6f", "FIG"]
V2_EXPS = ["V2U", "V2DL", "V2DA"]
QUICK_V1 = ["E1", "E2", "E3", "E5", "FIG"]
ARCHIVE_MONTHS = ["2023-03", "2023-04", "2023-05", "2023-06", "2023-07", "2023-08", "2023-09",
                  "2023-10", "2023-11", "2023-12", "2024-01", "2024-02"]
ARCHIVE_URL = ("https://raw.githubusercontent.com/nmpowell/carbon-intensity-forecast-tracking/main/"
               "data/db/{y}/national_{ym}.sqlite")


def sh(args, **kw):
    print("  $", " ".join(str(a) for a in args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True, **kw)


def py(*args):
    sh([PY, *args])


def stage(name):
    print(f"\n=== {name} [{time.strftime('%H:%M:%S')}] ===", flush=True)


# ------------------------------------------------------------------ stages
def check(_):
    stage("check")
    if sys.version_info < (3, 10):
        raise SystemExit(f"Python >= 3.10 required (found {sys.version.split()[0]})")
    missing = []
    for mod in ("numpy", "pandas", "pyarrow", "lightgbm", "sklearn", "scipy", "matplotlib",
                "requests", "joblib", "yaml"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        req = "requirements.lock" if sys.version_info >= (3, 11) else "requirements-py310.txt"
        raise SystemExit(f"missing packages {missing}: run  {PY} -m pip install -r {req}")
    print(f"  python {sys.version.split()[0]}, all packages present; {os.cpu_count()} CPUs")
    if sys.platform == "win32" and not long_paths_ok():
        raise SystemExit(
            f"  This folder's path is {len(ROOT)} characters long. Simulation result files add up to\n"
            f"  {LONGEST_REL} more, which exceeds Windows' 260-character path limit. Either move the\n"
            f"  project to a short path (e.g. C:\\cs\\carbonsched) or enable Windows long paths\n"
            f"  (Group Policy 'Enable Win32 long paths', or registry LongPathsEnabled=1).")


LONGEST_REL = 110  # results/raw_v2/<longest cell id>.parquet.done.tmp


def long_paths_ok():
    """True if the deepest result path fits, or Windows long paths are enabled (read-only check)."""
    if len(ROOT) + LONGEST_REL < 255:
        return True
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem")
        return winreg.QueryValueEx(k, "LongPathsEnabled")[0] == 1
    except OSError:
        return False


def fetch_archive():
    import requests
    out = os.path.join(ROOT, "data", "raw", "gb_archive")
    os.makedirs(out, exist_ok=True)
    for ym in ARCHIVE_MONTHS:
        p = os.path.join(out, f"national_{ym}.sqlite")
        if os.path.exists(p) and os.path.getsize(p) > 0:
            continue
        r = requests.get(ARCHIVE_URL.format(y=ym[:4], ym=ym), timeout=300)
        r.raise_for_status()
        with open(p + ".tmp", "wb") as f:
            f.write(r.content)
        os.replace(p + ".tmp", p)
        print(f"  archive {ym}: {len(r.content)} bytes")


def data(a):
    stage("data")
    if a.prepared_data:
        unpack_prepared(a.prepared_data)
        return
    py("-m", "carbonsched.data.download", "--what", "alibaba", "--out", "data/raw/alibaba")
    py("-m", "carbonsched.data.fetch_gb", "--out", "data/raw/gb", "--start", "2021-01-01",
       "--end", "2026-01-01", "--delay", "1.0")
    fetch_archive()
    py("-m", "carbonsched.data.download", "--what", "eia", "--out", "data/raw/eia")
    py("-m", "carbonsched.data.prep")
    py("-m", "carbonsched.data.gb_archive")
    py("-m", "carbonsched.data.trace")
    # write the manifest of *this* download next to the shipped one, then compare
    py("-m", "carbonsched.data.verify", "--manifest", "data/MANIFEST.local.json")
    compare_manifest()


def unpack_prepared(path):
    print(f"  unpacking prepared inputs from {path}")
    with zipfile.ZipFile(path) as z:
        z.extractall(ROOT)
    import hashlib
    import json
    m = json.load(open(os.path.join(ROOT, "cluster", "MANIFEST.json")))
    bad = n = 0
    for f in m["files"]:
        p = os.path.join(ROOT, "data", f["path"])
        if not os.path.exists(p):
            continue
        n += 1
        if hashlib.sha256(open(p, "rb").read()).hexdigest() != f["sha256"]:
            print("  CHECKSUM MISMATCH", f["path"])
            bad += 1
    print(f"  {n} prepared files checked against cluster/MANIFEST.json, {bad} mismatches")
    if bad:
        raise SystemExit("prepared data does not match the manifest")


def compare_manifest():
    """Report (not fail) differences between freshly downloaded data and the paper's inputs.
    Grid operators revise 'actual' CI for a while after the fact, so a fresh pull can differ
    slightly from the snapshot used in the paper (fetched 2026-09-24)."""
    import json
    ship = {f["path"]: f["sha256"] for f in json.load(open(os.path.join(ROOT, "cluster", "MANIFEST.json")))["files"]}
    mine = {f["path"]: f["sha256"] for f in json.load(open(os.path.join(ROOT, "data", "MANIFEST.local.json")))["files"]}
    same = sum(1 for k, v in mine.items() if ship.get(k) == v)
    diff = sorted(k for k, v in mine.items() if k in ship and ship[k] != v)
    print(f"  {same}/{len(mine)} files byte-identical to the paper's snapshot")
    if diff:
        print("  differing (expected for revised operator data):", ", ".join(diff[:10]),
              "..." if len(diff) > 10 else "")


def selftest(_):
    stage("selftest")
    py("tests/selftest.py")


def forecasts(a):
    stage("forecasts")
    if a.prepared_data:
        print("  using the shipped forecast caches (data/forecasts/*.npz)")
        return
    for r in REGIONS:
        py("-m", "carbonsched.forecast.prepare", "--region", r)


def sweep(a):
    stage("sweep")
    jobs = str(a.jobs)
    extra = ["--max-offsets", "2"] if a.quick else []
    v1 = QUICK_V1 if a.quick else V1_EXPS
    py("-m", "carbonsched.run", "--exp", *v1, "--n-jobs", jobs, *extra)
    py("-m", "carbonsched.run", "--exp", *V2_EXPS, "--config", "configs/experiments_v2.yaml",
       "--out", "results/raw_v2", "--n-jobs", jobs, *extra)


def tables(_):
    stage("tables")
    py("analysis/make_tables.py", "--raw", "results/raw", "results/raw_v2", "--out", "analysis/out",
       "--macros", "paper/numbers.tex", "--tex-tables", "paper/tables")


def figs(_):
    stage("figs")
    py("analysis/make_figs_v2.py", "--tables", "analysis/out", "--out", "paper/figs")


def paper(a):
    stage("paper")
    if not shutil.which("pdflatex"):
        print("  pdflatex not found: skipping (install TeX Live / MiKTeX to build paper/main.pdf)")
        return
    d = os.path.join(ROOT, "paper")
    for cmd in (["pdflatex", "-interaction=nonstopmode", "main.tex"], ["bibtex", "main"],
                ["pdflatex", "-interaction=nonstopmode", "main.tex"],
                ["pdflatex", "-interaction=nonstopmode", "main.tex"]):
        subprocess.run(cmd, cwd=d, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pdf = os.path.join(d, "main.pdf")
    print(f"  wrote {pdf}" if os.path.exists(pdf) else "  LaTeX build failed: see paper/main.log")
    if a.quick:
        print("  note: --quick runs a reduced grid, so some numbers render as ?? by design")


def word(_):
    stage("word")
    if sys.platform != "win32" or not shutil.which("powershell"):
        print("  the Word build drives MS Word through COM: Windows + MS Word only. Skipping.")
        return
    w = os.path.join(ROOT, "paper", "word")
    if not os.path.exists(os.path.join(w, "template.docx")):
        print("  paper/word/template.docx missing (convert the PCDS Proceeding-Template.doc first). Skipping.")
        return
    env = dict(os.environ, FIG_PNG=os.path.join(w, "figs"))
    os.makedirs(env["FIG_PNG"], exist_ok=True)
    sh([PY, "analysis/make_figs_v2.py", "--tables", "analysis/out", "--out", "paper/figs"], env=env)
    py("paper/word/tex2blocks.py")
    py("paper/word/build_docx.py")
    docx = os.path.join(w, "CarbonSched_PCDS2026.docx")
    sh(["powershell", "-ExecutionPolicy", "Bypass", "-File", os.path.join(w, "export_pdf.ps1"),
        "-In", docx, "-Pdf", docx.replace(".docx", ".pdf")])


STAGES = {"check": check, "data": data, "selftest": selftest, "forecasts": forecasts, "sweep": sweep,
          "tables": tables, "figs": figs, "paper": paper, "word": word}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="+", choices=[*STAGES, "all"])
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="parallel simulation workers (default: all CPUs but two)")
    ap.add_argument("--quick", action="store_true",
                    help="2 of 10 weekly offsets and the core experiments only (pipeline check, ~1 h)")
    ap.add_argument("--prepared-data", metavar="ZIP",
                    help="carbonsched_data_prepared.zip: skip downloads and forecast fitting")
    a = ap.parse_args()
    if a.prepared_data:
        a.prepared_data = os.path.abspath(a.prepared_data)
    order = list(STAGES) if "all" in a.stages else a.stages
    t0 = time.time()
    for s in order:
        STAGES[s](a)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
