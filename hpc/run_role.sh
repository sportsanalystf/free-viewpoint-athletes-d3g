# Body of every job (sourced by hpc/jobs.slurm on GPU nodes and hpc/cpu.slurm on CPU nodes). First argument = role:
#   rebuild                       compile the rasterizer for every GPU type on the cluster (GPU)
#   full SEQ N INIT [W EXP]       train the 4D model on SEQ with N cameras; optional loss-weight override W
#                                 (e.g. rigid=0) saved as experiment EXP (prior ablation)             (GPU, ~3 h)
#   cine SEQ EXP [ARGS...]        cinematic replay; ARGS go to render_cinematic.py (--variant, --interp) (GPU)
#   videos SEQ EXPS [ARGS]        held-out side-by-side + orbit videos                                 (GPU)
#   figs [MODELS]                 stage figures + held-out scores; MODELS = mixed | ours | released     (GPU)
#   interp EXP                    in-between frame experiment                                           (GPU)
#   align                         align Panoptic skeletons to the six clips                             (CPU)
#   sheets                        contact sheets for labelling key moments                              (CPU)
#   ablate EXP                    director ablation (all variants, no rendering)                        (CPU)
#   tracks EXPS [SEQS]            3D track error against the skeletons                                  (CPU)
#   analysis EXP                  error vs speed + worst-frame gallery                                   (CPU)
#   sweep SEQ                     (mid-term) frame-0 camera sweep                                        (GPU)
set -u
REPO=${REPO:-$SLURM_SUBMIT_DIR}
[ -f "$REPO/hpc/run_role.sh" ] || { echo "FAIL: submit from the repo root (cd \$CVP_ROOT/cv-project), or set REPO=..."; exit 1; }
source "$REPO/hpc/env.sh"
cp "$REPO"/src/*.py "$REPO"/scripts/*.py "$D3G"/      # the scripts run inside the Dynamic3DGaussians checkout
cd "$D3G"
echo "role $*  repo $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"
role=$1; shift
case "$role" in
  rebuild)
    export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0"
    cd "$RAST" && rm -rf build && pip install -q --no-build-isolation . \
      && cd "$WD" && python -c "import diff_gaussian_rasterization; print('PASS rasterizer rebuilt for', '$TORCH_CUDA_ARCH_LIST')" ;;
  full)
    if [ $# -ge 5 ]; then python -X faulthandler train_sparse.py --seq $1 --cams $2 --init $3 --weights "$4" --exp $5
    else python -X faulthandler train_sparse.py --seq $1 --cams $2 --init $3 --exp n$(printf %02d $2)_$3; fi ;;
  cine)    python -X faulthandler render_cinematic.py --seq $1 --exp $2 "${@:3}" ;;
  videos)  python -X faulthandler render_videos.py --seq $1 --exps $2 "${@:3}" ;;
  figs)    python -X faulthandler stage_figs.py --models ${1:-mixed} ;;
  interp)  python -X faulthandler eval_interp.py --exp ${1:-n27_allcam} ;;
  align)   python -X faulthandler align_skeletons.py --raw "$WD/161029_sports1" ;;
  sheets)  python -X faulthandler contact_sheets.py ;;
  ablate)  python -X faulthandler ablate_director.py --exp ${1:-n27_allcam} ;;
  tracks)  python -X faulthandler eval_tracks.py --exps ${1:-n27_allcam} ${2:+--seqs $2} ;;
  analysis) python -X faulthandler analysis.py --exp ${1:-n27_allcam} --old "$CVP_OLD/Dynamic3DGaussians" ;;
  sweep)
    python -X faulthandler sweep_frame0.py --seq $1 --cams 2,3,4,6,8,12,16,27 --iters 500,1000,3000,10000 --seeds 0,1 \
      && python plot_sweep.py --seq $1 ;;
  *) echo "unknown role $role"; exit 2 ;;
esac
