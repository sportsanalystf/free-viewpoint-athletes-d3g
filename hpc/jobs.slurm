#!/bin/bash
#SBATCH -A cs6501_fall26
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -o /scratch/%u/cv_project_v2/logs/%x_%j.log
# One script, several roles (first argument):
#   rebuild            rebuild the rasterizer for every GPU type on the cluster (A100 ... B200)
#   sweep SEQ          frame-0 sweep: cameras x init x iterations x seeds  (~2 h)
#   full SEQ N INIT    full 150-frame training with N cameras             (~1.5-2.5 h)
#   videos SEQ EXPS    side-by-side + orbit videos for comma-separated experiment names
#   cine SEQ EXP [ARGS] smooth cinematic replay (slow-mo + bullet time at ball apex, auto framing), 1280x720
#   figs               intermediate-output figures for every pipeline step (results/stage_figs.zip)
#
# Submit from the repository root, e.g.   cd /scratch/$USER/cv_project_v2/cv-project && sbatch hpc/jobs.slurm figs
# Paths come from hpc/config.sh (default root /scratch/$USER/cv_project_v2).
set -u
REPO=${REPO:-$SLURM_SUBMIT_DIR}
[ -f "$REPO/hpc/jobs.slurm" ] || { echo "FAIL: submit from the repo root (cd $CVP_ROOT/cv-project), or set REPO=..."; exit 1; }
source $REPO/hpc/env.sh
cp $REPO/src/*.py $REPO/scripts/*.py $D3G/      # the scripts run inside the Dynamic3DGaussians checkout
cd $D3G
case "$1" in
  rebuild)
    export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0"
    cd $RAST && rm -rf build && pip install -q --no-build-isolation . \
      && cd $WD && python -c "import diff_gaussian_rasterization; print('PASS rasterizer rebuilt for', '$TORCH_CUDA_ARCH_LIST')" ;;
  sweep)
    python -X faulthandler sweep_frame0.py --seq $2 --cams 2,3,4,6,8,12,16,27 --iters 500,1000,3000,10000 --seeds 0,1 \
      && python plot_sweep.py --seq $2 ;;
  full)
    python -X faulthandler train_sparse.py --seq $2 --cams $3 --init $4 --exp n$(printf %02d $3)_$4 ;;
  videos)
    python -X faulthandler render_videos.py --seq $2 --exps $3 ${4:-} ;;
  cine)
    python -X faulthandler render_cinematic.py --seq $2 --exp $3 ${4:-} ;;
  figs)
    python -X faulthandler stage_figs.py ;;
esac
