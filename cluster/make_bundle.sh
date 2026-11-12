#!/usr/bin/env bash
# Workstation: build what gets shipped to the cluster.
#   dist/carbonsched_code.tar.gz  - code, configs, notes, paper, tiny sample (no data, no results)
#   dist/carbonsched_data.tar.gz  - OPTIONAL prepared data (data/ci, data/trace, data/forecasts,
#                                   data/raw/gb_archive + MANIFEST) so the cluster can skip the
#                                   downloads; setup.sh --bundle verifies it against MANIFEST.json
# Usage: bash cluster/make_bundle.sh [--with-data]
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p dist
tar -czf dist/carbonsched_code.tar.gz --transform 's,^\./,carbonsched/,'   --exclude=./data/raw --exclude=./data/ci --exclude=./data/trace --exclude=./data/forecasts   --exclude=./results --exclude=./analysis/out --exclude=./dist --exclude=./.selftest --exclude=./logs --exclude='__pycache__'   --exclude=./.venv --exclude='*.pyc' --exclude=./paper/*.aux --exclude=./paper/*.log .
if [ "${1:-}" = "--with-data" ]; then
  tar -czf dist/carbonsched_data.tar.gz data/ci data/trace data/forecasts data/raw/gb_archive cluster/MANIFEST.json
fi
ls -la dist
cat <<EOF
Ship (from the workstation):
  scp dist/carbonsched_code.tar.gz dist/carbonsched_data.tar.gz <user>@<login>:\$SCRATCH/
On the login node:
  cd \$SCRATCH && tar -xzf carbonsched_code.tar.gz && cd carbonsched
  vi cluster/cluster.env            # partition (CPU!), account, qos, maxtime
  bash cluster/probe.sh             # -> notes/cluster_env.md
  tmux new -s cs 'bash cluster/setup.sh --bundle ../carbonsched_data.tar.gz 2>&1 | tee notes/setup.log'
  bash cluster/submit.sh smoke      # then: bash cluster/submit.sh E1 E2 E3 E4 E5 E6a E6b E6c E6d E6e E6f FIG
EOF
