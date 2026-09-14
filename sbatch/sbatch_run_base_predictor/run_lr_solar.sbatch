#!/bin/bash
#SBATCH -J lr_solar
#SBATCH --account=gts-yxie77-paid
#SBATCH --mem=16G
#SBATCH -t120
#SBATCH -qinferno
#SBATCH -o Report-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

set -euo pipefail

RUNPATH=/storage/project/r-yxie77-0/jlee3541/projects/dscp
cd "$RUNPATH"

export PYTHONPATH="$RUNPATH:$RUNPATH/sbatch:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

module load anaconda3
conda activate dscp

python -m sbatch_run_base_predictor.run_regression nsdb-60m \
  --data-dir "$RUNPATH/data/nsdb_2018-2020" \
  --save-dir "$RUNPATH/data/solar_prediction/lr" \
  --train-ratio 0.33 \
  --past-window 100 \
  --prediction-step 1
