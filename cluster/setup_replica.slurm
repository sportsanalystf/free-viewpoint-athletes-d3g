#!/bin/bash
#SBATCH -A cs6501_fall26
#SBATCH -p standard
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 03:00:00
#SBATCH -J setup_v2
#SBATCH -o /scratch/%u/cv_project_v2/logs/%x_%j.log
# Build a fresh, self-contained copy of the project under $CVP_ROOT (cluster/config.sh), CPU-only part:
#   1. clone Dynamic3DGaussians + the rasterizer at the exact commits we used, apply our k-NN patch
#   2. create a new conda env ($CVP_ENV) with the same packages as the original "d3g" env
#   3. copy data and models READ-ONLY from the original folder ($CVP_OLD); nothing there is changed
# Submitted by cluster/setup_replica.sh, which then chains the GPU rasterizer build and a check run.
set -euo pipefail
REPO=${REPO:-$SLURM_SUBMIT_DIR}; source "$REPO/cluster/config.sh"
[ "$(realpath -m "$CVP_ROOT")" != "$(realpath -m "$CVP_OLD")" ] || { echo "FAIL: CVP_ROOT points at the original folder"; exit 1; }
step() { echo; echo "== $*  ($(date +%T))"; }
mkdir -p "$WD" "$LOGS"

step "1. upstream code"
[ -d "$D3G/.git" ] || git clone -q https://github.com/JonathonLuiten/Dynamic3DGaussians "$D3G"
git -C "$D3G" checkout -q "$D3G_COMMIT"
grep -q cKDTree "$D3G/helpers.py" || git -C "$D3G" apply "$REPO/cluster/patches/helpers_knn.diff" || python3 "$REPO/cluster/patch_d3g_knn.py" "$D3G"
grep -q cKDTree "$D3G/helpers.py" && echo "PASS Dynamic3DGaussians at ${D3G_COMMIT:0:7}, k-NN patched"
[ -d "$RAST/.git" ] || git clone -q --recursive https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth "$RAST"
git -C "$RAST" checkout -q "$RAST_COMMIT" && git -C "$RAST" submodule update -q --init --recursive
echo "PASS rasterizer source at ${RAST_COMMIT:0:7} (built later on a GPU node)"

step "2. conda env $CVP_ENV"
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
[ -d "$CVP_ENV" ] || conda create -y -q -p "$CVP_ENV" python=3.10
conda activate "$CVP_ENV"
pip install -q torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
grep -viE '^(torch|torchvision|triton|nvidia-|cuda-|diff[-_]gaussian)' "$REPO/cluster/env/pip_freeze_d3g.txt" \
  | sed 's/^packaging @.*/packaging/' > "$WD/requirements_frozen.txt"
pip install -q -r "$WD/requirements_frozen.txt" || {
  echo "bulk install failed, installing one by one"
  while read -r p; do [ -n "$p" ] && { pip install -q "$p" || echo "WARN could not install $p"; }; done < "$WD/requirements_frozen.txt"; }
python -c "import torch, numpy, scipy, cv2, open3d, PIL, matplotlib; print('PASS env: torch', torch.__version__, 'cuda', torch.version.cuda)"

step "3. data and models (read-only copies from $CVP_OLD)"
O=$CVP_OLD/Dynamic3DGaussians
mkdir -p "$D3G/data" "$D3G/output/n27_allcam/basketball" "$D3G/results" "$WD/pretrained"
for s in $SEQS; do rsync -a "$O/data/$s/" "$D3G/data/$s/"; done
rsync -a "$O/output/n27_allcam/basketball/"{params.npz,config.json,per_frame.csv} "$D3G/output/n27_allcam/basketball/"
for s in $SEQS; do mkdir -p "$WD/pretrained/$s"; rsync -aL "$O/output/pretrained/$s/params.npz" "$WD/pretrained/$s/"; done
ln -sfn "$WD/pretrained" "$D3G/output/pretrained"
rsync -a --exclude='_cine_*' --exclude='_orbit_*' --exclude='_cmp' "$O/results/" "$D3G/results/"
[ -d "$CVP_OLD/161029_sports1" ] && rsync -a --exclude='hdVideos' "$CVP_OLD/161029_sports1/" "$WD/161029_sports1/"   # skeletons + calibration
[ -d "$CVP_OLD/panoptic-toolbox" ] && rsync -a "$CVP_OLD/panoptic-toolbox/" "$WD/panoptic-toolbox/"

step "4. check"
ok=1
for s in $SEQS; do
  if [ -f "$D3G/data/$s/train_meta.json" ] && [ -f "$D3G/output/pretrained/$s/params.npz" ]; then echo "PASS $s data + released model"; else echo "FAIL $s"; ok=0; fi
done
[ -f "$D3G/output/n27_allcam/basketball/params.npz" ] && echo "PASS our basketball model" || { echo "FAIL our basketball model"; ok=0; }
du -sh "$WD"/* 2>/dev/null
[ $ok = 1 ] && echo "SETUP OK" || { echo "SETUP INCOMPLETE"; exit 1; }
