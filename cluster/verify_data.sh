#!/usr/bin/env bash
# Fails loudly on gaps, duplicates, implausible CI or misaligned GB forecast/actual,
# then pins provenance in cluster/MANIFEST.json (checksums, rows, ranges, times).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f cluster/cluster.env ] && source cluster/cluster.env
[ -n "${CS_VENV:-}" ] && [ -x "$CS_VENV/bin/python" ] && source "$CS_VENV/bin/activate"
python -m carbonsched.data.verify --manifest cluster/MANIFEST.json
