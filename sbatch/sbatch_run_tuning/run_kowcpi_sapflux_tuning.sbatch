#!/bin/bash
#SBATCH -J kowcpi_sapflux_tuning
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t720
#SBATCH -qinferno
#SBATCH -o Report-kowcpi-sapflux-tuning-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

set -euo pipefail

# Tasks 0/1/2 tune KOWCPI on LR/LSTM/Chronos forecasts.
TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
case "$TASK_ID" in
  0|1|2) ;;
  *) echo "Expected SLURM_ARRAY_TASK_ID=0, 1, or 2; got '${TASK_ID:-unset or empty}'." >&2; exit 2 ;;
esac

RUNPATH="${RUNPATH:-/storage/project/r-yxie77-0/jlee3541/projects/dscp}"
cd "$RUNPATH"
export PYTHONPATH="$RUNPATH:$RUNPATH/sbatch:${PYTHONPATH:-}"
# Parallelize sequences without nesting BLAS/OpenMP threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dscp

python -u -m sbatch.sbatch_run_tuning.run_kowcpi_tuning_job sapflux \
  --task-id "$TASK_ID" \
  --output-root "${KOWCPI_TUNING_OUTPUT_ROOT:-$RUNPATH/results/tuning/kowcpi}" \
  --seed "${KOWCPI_SEED:-2026}" \
  --num-cores "${SLURM_CPUS_PER_TASK:-12}" \
  --top-k "${KOWCPI_TOP_K:-3}" \
  "$@"
