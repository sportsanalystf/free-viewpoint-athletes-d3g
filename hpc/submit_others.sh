#!/bin/bash
# Cinematic replays for the other 5 clips, using the authors' trained 27-camera models (saves ~15 GPU-hours).
#   cd /scratch/$USER/cv_project_v2/cv-project && bash hpc/submit_others.sh
set -e
cd "$(dirname "$0")/.."
source hpc/config.sh; mkdir -p "$LOGS"
F=$(sbatch --parsable hpc/fetch_pretrained.slurm)
ids=""
for s in boxes football juggle softball tennis; do
  j=$(sbatch --parsable -J cine_$s -t 00:40:00 --dependency=afterok:$F hpc/jobs.slurm cine $s pretrained); ids="$ids $s:$j"
done
echo "submitted: fetch $F |$ids"
echo "watch: squeue -u \$USER   |   tail -3 $LOGS/fetch_$F.log"
