#!/bin/bash
#SBATCH -J conditional_groups
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --qos=inferno
#SBATCH --output=Report-conditional-groups-%j.out

set -euo pipefail

# Submit from the repository root, or export RUNPATH to its cluster location.
RUNPATH="${RUNPATH:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$RUNPATH"
export PYTHONPATH="$RUNPATH:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dscp

# One CPU worker per sequence, up to the allocated CPUs and sequence count.
# One shared artifact per dataset/base predictor. CLI options override the config.
# For example: sbatch <this-file> --config <grouping.yaml> --dry-run
exec python -u -m conditional_coverage.prepare_groups \
  --config "${CC_GROUP_CONFIG:-configs/conditional_coverage_configs/air_lstm.yaml}" \
  "$@" \
  --cpu-workers "${SLURM_CPUS_PER_TASK:-1}"
