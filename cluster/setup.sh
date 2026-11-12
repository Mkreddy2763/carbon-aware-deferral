#!/usr/bin/env bash
# NOTE: login-node variant for sites that allow it. On dgx2 (Slurm-only rule) use
#       cluster/jobs/setup.sbatch instead (see README_REPRODUCE.md).
# B1: run ONCE on the LOGIN node, inside tmux. All network access happens here;
# compute jobs only read local files.
#   bash cluster/setup.sh                          # fetch everything from the original sources
#   bash cluster/setup.sh --bundle data.tar.gz     # use workstation-prepared data (checksummed)
#   bash cluster/setup.sh --no-venv ...            # reuse an existing venv
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/cluster.env
PY=${PYTHON:-python3}
BUNDLE=""; VENV=1
while [ $# -gt 0 ]; do
  case "$1" in
    --bundle) BUNDLE="$2"; shift 2 ;;
    --no-venv) VENV=0; shift ;;
    *) echo "unknown arg $1"; exit 1 ;;
  esac
done

$PY - <<'EOF' || { echo "need Python >= 3.11 (module load python/3.11?)"; exit 1; }
import sys; assert sys.version_info >= (3, 11), sys.version
EOF

if [ "$VENV" = 1 ]; then
  echo "== venv ($CS_VENV)"
  [ -x "$CS_VENV/bin/python" ] || $PY -m venv "$CS_VENV"
  source "$CS_VENV/bin/activate"
  pip install --upgrade pip >/dev/null
  pip install -r requirements.lock
else
  source "$CS_VENV/bin/activate"
fi

if [ -n "$BUNDLE" ]; then
  echo "== unpack prepared data bundle $BUNDLE"
  tar -xzf "$BUNDLE"
  python - <<'EOF'
import hashlib, json, os, sys
m = json.load(open("cluster/MANIFEST.json"))
bad = 0; n = 0
for f in m["files"]:
    p = os.path.join("data", f["path"])
    if not os.path.exists(p):
        continue                      # raw files not shipped in the bundle
    n += 1
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if h != f["sha256"]:
        print("CHECKSUM MISMATCH", p); bad += 1
print(f"checked {n} files against MANIFEST, {bad} mismatches")
sys.exit(1 if bad else 0)
EOF
else
  echo "== fetch (resumable; re-run this script if interrupted)"
  python -m carbonsched.data.download --what alibaba --out data/raw/alibaba
  python -m carbonsched.data.fetch_gb --out data/raw/gb --start 2021-01-01 --end 2026-01-01 --delay 1.0
  bash cluster/fetch_gb_archive.sh
  python -m carbonsched.data.download --what eia --out data/raw/eia
  echo "== normalise"
  python -m carbonsched.data.prep
  python -m carbonsched.data.gb_archive
  python -m carbonsched.data.trace
  echo "== verify + manifest"
  bash cluster/verify_data.sh
fi
python - <<'EOF'
# the prepared data must pass the same checks on the cluster
from carbonsched.data.verify import verify
errors, _ = verify("configs/regions.yaml", "data")
assert not errors, errors
print("data verified")
EOF
mkdir -p notes && { echo "## $(hostname) $(date -u +%FT%TZ)"; du -sh "$CS_VENV" data results 2>/dev/null; } | tee -a notes/storage.md
echo "setup done. Forecast caches: shipped in the bundle (or: bash cluster/submit.sh prep)."
echo "next: bash cluster/submit.sh smoke"
