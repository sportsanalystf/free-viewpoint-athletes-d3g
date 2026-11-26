#!/bin/bash
# Build the replica in $CVP_ROOT (default /scratch/$USER/cv_project_v2) with three chained jobs:
#   setup_v2 (CPU: clone code, new conda env, read-only copies of data/models)
#   -> rebuild (GPU: compile the rasterizer into the new env)
#   -> figs    (GPU: re-make every stage figure and held-out score, to check the replica reproduces our results)
# Run from the repo root inside the replica:
#   cd /scratch/$USER/cv_project_v2/cv-project && bash cluster/setup_replica.sh
# Afterwards:  source cluster/config.sh && python3 cluster/verify_replica.py "$D3G"
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/config.sh
[ "$(realpath -m "$CVP_ROOT")" != "$(realpath -m "$CVP_OLD")" ] || { echo "CVP_ROOT points at the original folder; refusing"; exit 1; }
mkdir -p "$LOGS"
V100=$(sinfo -h -p gpu -o "%N %G" | grep -i v100 | awk '{print $1}' | paste -sd, -)
S=$(sbatch --parsable cluster/setup_replica.slurm)
B=$(sbatch --parsable -J rebuild -t 00:40:00 ${V100:+--exclude=$V100} --dependency=afterok:$S cluster/jobs.slurm rebuild)
F=$(sbatch --parsable -J figs -t 01:00:00 ${V100:+--exclude=$V100} --dependency=afterok:$B cluster/jobs.slurm figs)
echo "submitted: setup $S -> rebuild $B -> figs $F   (root $CVP_ROOT, env $CVP_ENV)"
echo "watch:  squeue -u \$USER      logs:  ls $LOGS/"
echo "then:   source cluster/config.sh && python3 cluster/verify_replica.py \"\$D3G\""
