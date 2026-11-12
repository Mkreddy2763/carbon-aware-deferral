#!/usr/bin/env bash
# B0 probe. On the login node this only runs Slurm client queries (sinfo/scontrol/
# sacctmgr) and submits cluster/jobs/probe.sbatch; all node-side checks (cores,
# RAM, python, filesystems, egress) run inside that Slurm job. Writes notes/cluster_env.md.
set -u
cd "$(dirname "$0")/.."
source cluster/cluster.env
mkdir -p notes logs
OUT=notes/cluster_env.md
{
echo "# Cluster environment (probe $(date -u +%FT%TZ), login $(hostname), user $USER)"
echo '## Partitions (sinfo -o "%P %l %c %m %G %a %D")'
echo '```'; sinfo -o "%P %l %c %m %G %a %D" 2>&1; echo '```'
for p in $CS_PARTITION; do
  echo "## scontrol show partition $p"; echo '```'
  scontrol show partition "$p" 2>&1 | tr ' ' '\n' | grep -Ei \
    'MaxTime|DefaultTime|MaxNodes|MaxCPUsPerNode|DefMemPerCPU|MaxMemPerCPU|MaxMemPerNode|PreemptMode|QoS|AllowAccounts|AllowQos|TotalCPUs|TotalNodes|Nodes='
  echo '```'
done
echo "## Slurm config"; echo '```'
scontrol show config 2>&1 | grep -Ei 'MaxArraySize|MaxJobCount|PreemptType|PreemptMode|SchedulerType'
echo '```'
echo "## Associations and QOS"; echo '```'
sacctmgr -n -P show assoc user="$USER" format=account,partition,qos,maxjobs,maxsubmit,grptres 2>&1 | head -20
sacctmgr -n -P show qos format=name,maxwall,maxjobspu,maxsubmitpu,maxtrespu,preempt 2>&1 | head -20
echo '```'
echo "## Nodes in $CS_PARTITION"; echo '```'
sinfo -N -p "$CS_PARTITION" -o "%N %c %m %G %T" 2>&1 | head -20
echo '```'
echo "## Compute-node checks (Slurm job)"; echo '```'
jid=$(sbatch --parsable -p "$CS_PARTITION" ${CS_ACCOUNT:+-A $CS_ACCOUNT} ${CS_QOS:+--qos $CS_QOS} \
      --wait cluster/jobs/probe.sbatch 2>&1 | tail -1)
cat logs/probe_"$jid".out 2>/dev/null || echo "probe job output missing (jid=$jid)"
echo '```'
} > "$OUT" 2>&1
cat "$OUT"
