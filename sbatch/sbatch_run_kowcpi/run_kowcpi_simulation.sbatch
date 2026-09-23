#!/bin/bash
#SBATCH -J kowcpi_simulation
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH -t10
#SBATCH -qinferno
#SBATCH -o Report-kowcpi-simulation-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

set -euo pipefail

# One synthetic sequence with saved LightGBM point forecasts.
RUNPATH="${RUNPATH:-/storage/project/r-yxie77-0/jlee3541/projects/dscp}"
cd "$RUNPATH"

export PYTHONPATH="$RUNPATH:$RUNPATH/sbatch:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg

module load anaconda3
conda activate dscp

python -u -m sbatch_run_kowcpi.run_kowcpi \
  configs/kowcpi_configs/kowcpi_simulation_config.yaml \
  --num-cores "${SLURM_CPUS_PER_TASK:-1}"
