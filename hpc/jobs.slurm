#!/bin/bash
#SBATCH -A cs6501_fall26
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -o /scratch/%u/cv_project_v2/logs/%x_%j.log
# GPU job. Roles and arguments: see hpc/run_role.sh.
# Submit from the repository root, e.g.   cd /scratch/$USER/cv_project_v2/cv-project && sbatch hpc/jobs.slurm figs
# Paths come from hpc/config.sh (default root /scratch/$USER/cv_project_v2).
source "${SLURM_SUBMIT_DIR}/hpc/run_role.sh" "$@"
