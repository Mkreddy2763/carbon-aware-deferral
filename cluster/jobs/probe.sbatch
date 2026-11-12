#!/usr/bin/env bash
# Node-side half of the B0 probe, run as a Slurm job (nothing heavy on the login node).
#SBATCH --job-name=cs-probe
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/probe_%j.out
echo "## compute node: $(hostname)  $(date -u +%FT%TZ)"
echo "nproc=$(nproc)  mem: $(free -g | awk '/Mem:/{print $2" GB total, "$7" GB avail"}')"
lscpu | grep -E "Model name|Socket|Core\(s\)|Thread\(s\)" | sed 's/  */ /g'
echo "## python"
for p in python3 python3.13 python3.12 python3.11 python3.10 python3.9; do command -v $p >/dev/null && echo "$p -> $($p --version 2>&1) at $(command -v $p)"; done
(module -t avail python 2>&1 || true) | grep -i python | head -20
(module -t avail anaconda miniconda conda 2>&1 || true) | grep -i conda | head -5
echo "## filesystems"
df -h "$HOME" /scratch* /tmp 2>/dev/null
(quota -s 2>/dev/null || true) | tail -4
echo "SCRATCH=${SCRATCH:-unset} TMPDIR=${TMPDIR:-unset}"
echo "## egress from the compute node"
for u in https://pypi.org/simple/numpy/ https://files.pythonhosted.org https://api.carbonintensity.org.uk/intensity; do
  printf "%-50s " "$u"; curl -sS -o /dev/null -m 15 -w "%{http_code}\n" "$u" 2>&1 | tail -1
done
env | grep -i _proxy || echo "no *_proxy variables"
