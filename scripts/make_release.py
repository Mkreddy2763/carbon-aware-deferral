"""Build the release zips in dist/release/.

  carbonsched_code.zip            all committed code, configs, paper sources, notes, MANIFEST
  carbonsched_data_prepared.zip   the paper's exact inputs (data/ci, trace, forecasts, gb_archive)
  carbonsched_results.zip         the paper's simulation results (results/raw, results/raw_v2)
"""
import os
import subprocess
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "dist", "release")
os.makedirs(OUT, exist_ok=True)


def zip_files(name, files, prefix="carbonsched/"):
    path = os.path.join(OUT, name)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in files:
            z.write(os.path.join(ROOT, f), prefix + f.replace("\\", "/"))
    print(f"{name}: {len(files)} files, {os.path.getsize(path) / 1e6:.1f} MB")


def walk(*dirs, skip=()):
    out = []
    for d in dirs:
        for dp, _, fs in os.walk(os.path.join(ROOT, d)):
            for f in fs:
                rel = os.path.relpath(os.path.join(dp, f), ROOT)
                if not any(s in rel for s in skip):
                    out.append(rel)
    return sorted(out)


# code: exactly what git tracks (no data, results, submission files or local docs)
code = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
code = [f for f in code if not f.startswith(("submission/", "results/", "dist/"))
        and f not in ("paper/word/Proceeding-Template.doc",)]
zip_files("carbonsched_code.zip", code)
zip_files("carbonsched_data_prepared.zip",
          walk("data/ci", "data/trace", "data/forecasts", "data/raw/gb_archive") + ["cluster/MANIFEST.json"],
          prefix="")
zip_files("carbonsched_results.zip", walk("results/raw", "results/raw_v2"), prefix="")
