#!/usr/bin/env bash
# Environment + data setup as a Slurm job (the login node only runs sbatch).
#   sbatch -p cpu_student --export=ALL,BUNDLE=../carbonsched_data.tar.gz cluster/jobs/setup.sbatch
# pip installs from PyPI if the compute node has egress, otherwise from the shipped
# wheelhouse (cluster/wheels/, built on the workstation for the cluster's python).
#SBATCH --job-name=cs-setup
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/setup_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/cluster.env
[ -n "${CS_MODULES:-}" ] && module load $CS_MODULES
PY=${CS_PYTHON:-python3}
echo "[$(date -u +%FT%TZ)] $(hostname) python=$($PY --version 2>&1)"
$PY -c 'import sys; assert sys.version_info >= (3, 10), sys.version'

if [ -n "${BUNDLE:-}" ]; then
  echo "== unpack $BUNDLE"
  tar -xzf "$BUNDLE"
fi

echo "== venv $CS_VENV"
[ -x "$CS_VENV/bin/python" ] || $PY -m venv "$CS_VENV"
source "$CS_VENV/bin/activate"
REQ=requirements.lock
[ -f "cluster/requirements.cluster.txt" ] && REQ=cluster/requirements.cluster.txt
if [ -d cluster/wheels ] && ls cluster/wheels/*.whl >/dev/null 2>&1; then
  pip install --no-index --find-links cluster/wheels -r "$REQ"
else
  pip install --upgrade pip
  pip install -r "$REQ"
fi
python -c "import numpy, pandas, pyarrow, lightgbm, scipy, yaml, joblib; print('imports ok', numpy.__version__, pandas.__version__, lightgbm.__version__)"

echo "== checksums vs cluster/MANIFEST.json"
python - <<'EOF'
import hashlib, json, os, sys
m = json.load(open("cluster/MANIFEST.json")); bad = n = 0
for f in m["files"]:
    p = os.path.join("data", f["path"])
    if not os.path.exists(p):
        continue
    n += 1
    if hashlib.sha256(open(p, "rb").read()).hexdigest() != f["sha256"]:
        print("CHECKSUM MISMATCH", p); bad += 1
print(f"checked {n} files, {bad} mismatches"); sys.exit(1 if bad else 0)
EOF
echo "== data verification"
python - <<'EOF'
from carbonsched.data.verify import verify
errors, _ = verify("configs/regions.yaml", "data")
assert not errors, errors
print("data verified")
EOF
echo "== self-test (synthetic, ~1 min)"
python tests/selftest.py | tail -2
du -sh "$CS_VENV" data 2>/dev/null
echo "[$(date -u +%FT%TZ)] setup done"
