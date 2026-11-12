#!/usr/bin/env bash
# Submit a sweep as a Slurm array on the CPU partition. Only sbatch runs on the login
# node; the group list is computed on the workstation (cluster/remote.py) and passed in.
#   bash cluster/submit.sh --array 0-186 E1 E2 E3 ...
#   bash cluster/submit.sh --all E1 E2 ...       # one CS_CPUS-core job running every remaining cell
#   bash cluster/submit.sh --prep                 # forecast refit, one task per region (optional)
# Tasks whose cells are all done exit in seconds, so over-submitting is harmless.
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/cluster.env
[ -z "$CS_PARTITION" ] && { echo "set CS_PARTITION in cluster/cluster.env"; exit 1; }
case "$CS_PARTITION" in *gpu*) echo "refusing: $CS_PARTITION is a GPU partition; this paper is CPU-only"; exit 1 ;; esac
mkdir -p logs
common=(-p "$CS_PARTITION" --time="$CS_MAXTIME" --cpus-per-task="$CS_CPUS" --mem="$CS_MEM")
[ -n "$CS_ACCOUNT" ] && common+=(-A "$CS_ACCOUNT")
[ -n "$CS_QOS" ] && common+=(--qos "$CS_QOS")
if [ "${1:-}" = "--all" ]; then
  # one job, all cores, all remaining cells of the listed experiments (priority = listed order)
  shift
  sbatch "${common[@]}" --job-name=carbonsched-all --export=ALL,MODE=all,EXPS="$*" cluster/run_cell.sbatch
  exit 0
fi
if [ "${1:-}" = "--prep" ]; then
  sbatch "${common[@]}" --array=0-5 --export=ALL,MODE=prep cluster/run_cell.sbatch
  exit 0
fi
[ "${1:-}" = "--array" ] || { echo "usage: submit.sh --array <spec> EXP..."; exit 1; }
ARR="$2"; shift 2
EXPS="$*"
sbatch "${common[@]}" --array="${ARR}%${CS_MAX_ARRAY_PAR}" --export=ALL,MODE=sim,EXPS="$EXPS" cluster/run_cell.sbatch
