#!/bin/bash
# Control run for the prior ablation: train full D3G on basketball a second time with exactly the same settings
# (default loss weights, same seed), saved as experiment n27_allcam_rep. Its difference from n27_allcam is the
# run-to-run noise that the prior-ablation gaps are compared against. Then score its 3D track error.
#   cd /scratch/$USER/cv_project_v2/cv-project && git pull && bash hpc/submit_control.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/config.sh; mkdir -p "$LOGS"
V100=$(sinfo -h -p gpu -o "%N %G" | grep -i v100 | awk '{print $1}' | paste -sd, -); X=${V100:+--exclude=$V100}
T=$(sbatch --parsable $X -J control_rep -t 05:30:00 hpc/jobs.slurm full basketball 27 allcam "rigid=4" n27_allcam_rep)
echo "control_rep   $T"
K=$(sbatch --parsable -J tracks_rep -t 01:00:00 --dependency=afterok:$T hpc/cpu.slurm tracks n27_allcam,n27_allcam_rep basketball)
echo "tracks_rep    $K   (after $T)"
echo "When both finish, send back: output/n27_allcam_rep/basketball/per_frame.csv and results/tracks/tracks_n27_allcam+n27_allcam_rep.csv"
echo "(both under \$D3G = $D3G)"
