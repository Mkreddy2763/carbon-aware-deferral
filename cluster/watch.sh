#!/usr/bin/env bash
# Progress: queue state, finished cells per experiment, failures. Shell only (no python on login).
cd "$(dirname "$0")/.."
squeue -u "$USER" -o "%.18i %.12P %.14j %.8T %.10M %.10l %R" 2>/dev/null | head -30
echo "done cells per experiment:"
ls results/raw 2>/dev/null | grep '\.done$' | cut -d_ -f1 | sort | uniq -c
grep -l "FAIL\|Traceback\|CANCELLED\|PREEMPTED\|TIME LIMIT\|oom" logs/*.out logs/*.err 2>/dev/null | tail -5
