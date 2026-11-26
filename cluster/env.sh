# Sourced by every GPU job: paths, modules and the conda env, in the order that worked on UVA HPC.
_HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_HPC_DIR/config.sh"
module load gcc/11.4.0 miniforge cuda/12.8.0
export CC=$(which gcc) CXX=$(which g++)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null; conda activate "$CVP_ENV"
mkdir -p "$LOGS"
echo "node $(hostname)  gpu $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -1)  python $(which python)  root $CVP_ROOT"
