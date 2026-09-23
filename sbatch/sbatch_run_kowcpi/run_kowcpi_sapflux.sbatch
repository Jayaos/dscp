#!/bin/bash
#SBATCH -J kowcpi_sapflux
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-2
#SBATCH --cpus-per-task=24
#SBATCH --mem=384G
#SBATCH -t30
#SBATCH -qinferno
#SBATCH -o Report-kowcpi-sapflux-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

set -euo pipefail

# Tasks 0/1/2: KOWCPI on LR/LSTM/Chronos forecasts.
RUNPATH="${RUNPATH:-/storage/project/r-yxie77-0/jlee3541/projects/dscp}"
cd "$RUNPATH"

export PYTHONPATH="$RUNPATH:$RUNPATH/sbatch:${PYTHONPATH:-}"
# Parallelize independent sequences; avoid nested BLAS/OpenMP threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg

module load anaconda3
conda activate dscp

python -u -m sbatch_run_kowcpi.run_kowcpi_job sapflux \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --output-root "${KOWCPI_OUTPUT_ROOT:-$RUNPATH/results/kowcpi}" \
  --seed "${KOWCPI_SEED:-2026}" \
  --num-cores "${SLURM_CPUS_PER_TASK}"
