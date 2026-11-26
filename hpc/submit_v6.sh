#!/bin/bash
# Render the v6 cinematic replay for all 6 clips: one job per clip, never on V100s (unsupported GPU).
#   cd /scratch/$USER/cv_project_v2/cv-project && bash hpc/submit_v6.sh
set -e
cd "$(dirname "$0")/.."
source hpc/config.sh; mkdir -p "$LOGS"
V100=$(sinfo -h -p gpu -o "%N %G" | grep -i v100 | awk '{print $1}' | paste -sd, -)
for s in basketball boxes football juggle softball tennis; do
  if squeue -h -u $USER -n cine_$s | grep -q .; then echo "skip $s: a cine_$s job is already queued/running"; continue; fi
  exp=pretrained; [ $s = basketball ] && exp=n27_allcam
  j=$(sbatch --parsable -J cine_$s -t 00:40:00 ${V100:+--exclude=$V100} hpc/jobs.slurm cine $s $exp)
  echo "submitted cine_$s ($exp): $j"
done
