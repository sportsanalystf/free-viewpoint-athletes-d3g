#!/bin/bash
#SBATCH -A cs6501_fall26
#SBATCH -p standard
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 03:00:00
#SBATCH -J fetch
#SBATCH -o /scratch/%u/cv_project_v2/logs/%x_%j.log
# Download the authors' trained 27-camera models for all 6 PanopticSports clips (10.8 GB) and link them
# as output/pretrained/<seq>/params.npz (same format as our own runs).
set -eu
source "${SLURM_SUBMIT_DIR}/hpc/config.sh"   # submit from the repo root
mkdir -p $WD/pretrained_dl && cd $WD/pretrained_dl
wget -c -q https://omnomnom.vision.rwth-aachen.de/data/Dynamic3DGaussians/output.zip
python3 - <<'PY'
import zipfile
z = zipfile.ZipFile('output.zip')
infos = z.infolist()
print(f"zip has {len(infos)} entries, {sum(i.file_size for i in infos)/1e9:.1f} GB uncompressed")
want = [i for i in infos if i.filename.endswith('params.npz')]
for i in want:
    print(f"  {i.filename}  {i.file_size/1e9:.2f} GB")
    z.extract(i, '.')
print(f"extracted {len(want)} params.npz files")
PY
P=$(dirname "$(dirname "$(find . -path '*basketball/params.npz' | head -1)")")
[ -n "$P" ] || { echo "FAIL could not find params.npz in the zip"; exit 1; }
mkdir -p $D3G/output && ln -sfn "$(cd "$P" && pwd)" $D3G/output/pretrained
for s in basketball boxes football juggle softball tennis; do
  [ -f $D3G/output/pretrained/$s/params.npz ] && echo "PASS pretrained $s" || echo "FAIL missing $s"
done
