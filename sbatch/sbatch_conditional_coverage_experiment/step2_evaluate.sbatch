#!/bin/bash
#SBATCH -J conditional_evaluation
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --qos=inferno
#SBATCH --output=Report-conditional-evaluation-%j.out

set -euo pipefail

# Submit from the repository root, or export RUNPATH to its cluster location.
RUNPATH="${RUNPATH:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$RUNPATH"
export PYTHONPATH="$RUNPATH:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MPLBACKEND=Agg

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dscp

# Edit groups_dir and run_dir in this YAML before submitting. No model training.
# CLI options (including --config and --dry-run) override the defaults below.
exec python -u -m conditional_coverage.evaluate \
  --config "${CC_EVALUATION_CONFIG:-configs/conditional_coverage_configs/evaluate_air_lstm.yaml}" \
  "$@"
