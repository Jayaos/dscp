#!/bin/bash
#SBATCH -J spci_air
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --t10
#SBATCH -qinferno
#SBATCH -o Report-spci-air-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

set -euo pipefail

# Tasks 0/1/2: SPCI on LR/LSTM/Chronos forecasts.
RUNPATH="${RUNPATH:-/storage/project/r-yxie77-0/jlee3541/projects/dscp}"
cd "$RUNPATH"

export PYTHONPATH="$RUNPATH:$RUNPATH/sbatch:${PYTHONPATH:-}"
# Bound the forest's n_jobs=-1 to the allocation; avoid nested BLAS threads.
export LOKY_MAX_CPU_COUNT="${SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg

module load anaconda3
conda activate spci

python -u -m sbatch_run_spci.run_spci_job air \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --output-root "${SPCI_OUTPUT_ROOT:-$RUNPATH/results/spci}" \
  --seed "${SPCI_SEED:-2026}"
