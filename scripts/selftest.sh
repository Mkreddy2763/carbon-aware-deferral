#!/usr/bin/env bash
# Workstation self-test on synthetic data: conformal coverage, simulator invariants,
# one toy sweep through carbonsched.run and analysis/make_tables.py.
set -euo pipefail
cd "$(dirname "$0")/.."
python tests/selftest.py
