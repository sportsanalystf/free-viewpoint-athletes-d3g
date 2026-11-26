#!/bin/bash
# Round 1: everything needed to SEE results, chained so each step waits for the one it needs.
#   cd /scratch/$USER/cv_project_v2/cv-project && bash hpc/submit_round1.sh
set -e
cd "$(dirname "$0")/.."
source hpc/config.sh; mkdir -p "$LOGS"
R=$(sbatch --parsable -J rebuild -t 00:30:00 hpc/jobs.slurm rebuild)
S=$(sbatch --parsable -J sweep   -t 03:30:00 --dependency=afterok:$R hpc/jobs.slurm sweep basketball)
F1=$(sbatch --parsable -J full27a -t 04:00:00 --dependency=afterok:$R hpc/jobs.slurm full basketball 27 allcam)
F2=$(sbatch --parsable -J full04a -t 04:00:00 --dependency=afterok:$R hpc/jobs.slurm full basketball 4 allcam)
F3=$(sbatch --parsable -J full04r -t 04:00:00 --dependency=afterok:$R hpc/jobs.slurm full basketball 4 random)
V=$(sbatch --parsable -J videos  -t 01:00:00 --dependency=afterok:$F1:$F2:$F3 hpc/jobs.slurm videos basketball n27_allcam,n04_allcam,n04_random)
echo "submitted: rebuild $R | sweep $S | full 27-allcam $F1, 4-allcam $F2, 4-random $F3 | videos $V"
echo "watch:  squeue -u \$USER        logs:  ls $LOGS/"
