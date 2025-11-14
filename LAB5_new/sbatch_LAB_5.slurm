#!/usr/bin/env bash
#SBATCH -J Lab-5
#SBATCH -p h200q
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=15:00:00
#SBATCH -o %x-%j.out

set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# 推薦將 tmp 指到家目錄
export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

# 載入 conda
if command -v module &>/dev/null; then
  module purge || true
  module load anaconda || true
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate all-test

cd "$SLURM_SUBMIT_DIR"

papermill Lab5.ipynb Lab5_v3_darknet53_out.ipynb -k conda-all-test