#!/bin/bash
#SBATCH -A cs6501_fall26
#SBATCH -p standard
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -o /scratch/%u/cv_project_v2/logs/%x_%j.log
# CPU job (align, sheets, ablate, tracks, analysis). Roles and arguments: see hpc/run_role.sh.
# Submit from the repository root, e.g.   sbatch -J align -t 01:00:00 hpc/cpu.slurm align
source "${SLURM_SUBMIT_DIR}/hpc/run_role.sh" "$@"
