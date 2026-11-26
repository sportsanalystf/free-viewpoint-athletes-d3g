#!/bin/bash
# Every final-project experiment as one chain of Slurm jobs. Run from the repo root on the cluster:
#   cd /scratch/$USER/cv_project_v2/cv-project && git pull && bash cluster/submit_final.sh
# Stage 1 (now):   align skeletons (CPU); train the 4D model on the 5 sports we have not trained yet (GPU, ~3 h,
#                  in parallel); prior ablation on basketball: no_rigid, no_rot, no_iso (GPU, ~3 h each)
# Stage 2 (after): contact sheets; director ablation; 3D track error; in-between frame experiment; stage figures and
#                  held-out scores with our own models; analysis; final replays for all six sports; comparison videos
# Set SKIP_PRIORS=1 to leave out the prior ablation.
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/config.sh; mkdir -p "$LOGS"
V100=$(sinfo -h -p gpu -o "%N %G" | grep -i v100 | awk '{print $1}' | paste -sd, -); X=${V100:+--exclude=$V100}
gpu() { sbatch --parsable $X "$@"; }
cpu() { sbatch --parsable "$@"; }
EXP=n27_allcam

A=$(cpu -J align -t 01:30:00 cluster/cpu.slurm align); echo "align        $A"
TR=""
for s in boxes football juggle softball tennis; do
  if [ -f "$D3G/output/$EXP/$s/params.npz" ]; then echo "train_$s      skipped (model exists)"; continue; fi
  j=$(gpu -J train_$s -t 05:30:00 cluster/jobs.slurm full $s 27 allcam); TR="$TR:$j"; echo "train_$s     $j"
done
PR=""
if [ "${SKIP_PRIORS:-0}" != 1 ]; then
  for w in rigid rot iso; do
    if [ -f "$D3G/output/no_$w/basketball/params.npz" ]; then echo "prior_no_$w  skipped (exists)"; continue; fi
    j=$(gpu -J prior_no_$w -t 05:30:00 cluster/jobs.slurm full basketball 27 allcam "$w=0" no_$w); PR="$PR:$j"; echo "prior_no_$w  $j"
  done
fi
D="afterok:$A$TR"                                         # needs skeletons + all six of our models
echo "sheets       $(cpu -J sheets -t 00:30:00 --dependency=afterok:$A cluster/cpu.slurm sheets)"
echo "ablate       $(cpu -J ablate -t 03:00:00 --dependency=$D cluster/cpu.slurm ablate $EXP)"
echo "tracks       $(cpu -J tracks -t 02:00:00 --dependency=$D cluster/cpu.slurm tracks $EXP)"
echo "analysis     $(cpu -J analysis -t 01:00:00 --dependency=$D cluster/cpu.slurm analysis $EXP)"
echo "interp       $(gpu -J interp -t 03:00:00 --dependency=$D cluster/jobs.slurm interp $EXP)"
echo "figs_ours    $(gpu -J figs_ours -t 01:30:00 --dependency=$D cluster/jobs.slurm figs ours)"
for s in basketball boxes football juggle softball tennis; do
  echo "cine_$s $(gpu -J cine_$s -t 01:00:00 --dependency=$D cluster/jobs.slurm cine $s $EXP --variant full)"
done
# comparison videos for the ablation figure / user study (basketball and tennis)
for s in basketball tennis; do
  for v in v6 no_facing smooth_none no_clearance; do
    echo "cmp_${s}_$v $(gpu -J cmp_${s}_$v -t 01:00:00 --dependency=$D cluster/jobs.slurm cine $s $EXP --variant $v)"
  done
  echo "cmp_${s}_repeat $(gpu -J cmp_${s}_repeat -t 01:00:00 --dependency=$D cluster/jobs.slurm cine $s $EXP --variant full --interp nearest)"
done
if [ -n "$PR" ]; then
  echo "tracks_prior $(cpu -J tracks_prior -t 01:00:00 --dependency=afterok:$A$PR cluster/cpu.slurm tracks $EXP,no_rigid,no_rot,no_iso basketball)"
fi
echo; echo "watch: squeue -u \$USER     logs: ls -lt $LOGS | head"
echo "when the queue is empty: bash cluster/snapshot_from_hpc.sh  (then unpack into outputs/)"
