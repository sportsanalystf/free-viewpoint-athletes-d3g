#!/bin/bash
# Pack everything small and useful from the cluster into one archive to bring back to the local repo:
# code, logs, results (videos, figures, CSVs), run configs, the D3G patch, and the exact environment.
# Leaves out the dataset, model weights (*.npz), temporary render folders and the raw Panoptic videos.
#   bash hpc/snapshot_from_hpc.sh                                   (the replica, $CVP_ROOT)
#   CVP_ROOT=/scratch/$USER/cv_project CVP_ENV=d3g bash hpc/snapshot_from_hpc.sh   (the original setup)
#   -> /scratch/$USER/<root name>_snapshot_<date>.tar.gz
set -uo pipefail
source "$(cd "$(dirname "$0")" && pwd)/config.sh"; cd "$WD"
OUT=/scratch/$USER/$(basename "$WD")_snapshot_$(date +%Y%m%d_%H%M).tar.gz
[[ "$CVP_ENV" == */* ]] && ENVSEL="-p $CVP_ENV" || ENVSEL="-n $CVP_ENV"
META=_snapshot_meta; rm -rf $META; mkdir -p $META
module load miniforge >/dev/null 2>&1 || true
conda env export $ENVSEL --no-builds > $META/environment_d3g.yml 2>&1
conda run $ENVSEL pip freeze > $META/pip_freeze_d3g.txt 2>&1
( cd Dynamic3DGaussians && git remote -v; git rev-parse HEAD; git status --short; git diff ) > $META/d3g_git.txt 2>&1
( cd "$RAST" && git remote -v; git rev-parse HEAD; git status --short ) > $META/rasterizer_git.txt 2>&1
ls -la Dynamic3DGaussians/output/*/*/ > $META/model_outputs_listing.txt 2>&1
ls -la Dynamic3DGaussians/data/ > $META/data_listing.txt 2>&1
du -sh "$WD"/* > $META/disk_use.txt 2>&1
ls -la ~/exp > $META/home_exp_listing.txt 2>&1
tar -czf "$OUT" --ignore-failed-read \
  --exclude='*.npz' --exclude='*.zip' --exclude='*/_cine_*' --exclude='*/_orbit_*' --exclude='*/_cmp' \
  --exclude='*/test_renders' --exclude='__pycache__' \
  $META logs ./*.log ./*.txt ./*.png \
  Dynamic3DGaussians/results Dynamic3DGaussians/helpers.py \
  $(ls Dynamic3DGaussians/output/*/*/per_frame.csv Dynamic3DGaussians/output/*/*/config.json 2>/dev/null) \
  $( [ -d "$HOME/exp" ] && echo "-C $HOME exp" ) 2>&1 | grep -v "Removing leading" || true
ls -lh "$OUT"
echo "Download this file from OOD (Files > /scratch/$USER) and put it in the local repo's hpc_pull/ folder."
