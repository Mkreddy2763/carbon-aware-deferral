#!/usr/bin/env bash
# One array task = one (exp, region, policy, slack) group of cells; joblib fans the
# group's offsets / k / alpha values across the task's cores. Every cell writes its
# parquet atomically + a .done marker, so a preempted or timed-out task is simply
# resubmitted and only redoes the cells that were in flight.
#SBATCH --job-name=carbonsched
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
# partition/account/qos/time/cpus/mem are passed by cluster/submit.sh from cluster.env
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/cluster.env
[ -n "${CS_MODULES:-}" ] && module load $CS_MODULES
source "$CS_VENV/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # one core per joblib worker
echo "[$(date -u +%FT%TZ)] host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID:-} cpus=${SLURM_CPUS_PER_TASK} exps=${EXPS}"
if [ "${MODE:-sim}" = "all" ]; then
  python -m carbonsched.run --exp $EXPS --n-jobs "$SLURM_CPUS_PER_TASK" --config "${CFG:-configs/experiments.yaml}" --out "${OUT:-results/raw}"
elif [ "${MODE:-sim}" = "prep" ]; then
  # forecasts: fitted once per region, cached read-only for all cells
  REGIONS=(GB GBop CISO ERCO PJM MISO)
  python -m carbonsched.forecast.prepare --region "${REGIONS[$SLURM_ARRAY_TASK_ID]}" --n-jobs "$SLURM_CPUS_PER_TASK"
else
  python -m carbonsched.run --exp $EXPS --group "$SLURM_ARRAY_TASK_ID" --n-jobs "$SLURM_CPUS_PER_TASK"
fi
echo "[$(date -u +%FT%TZ)] done"
