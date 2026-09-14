#!/bin/bash
#SBATCH -J lr_air
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
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

python -m sbatch_run_base_predictor.run_regression air-10 \
  --data-dir "$RUNPATH/data/bejing_air_quality" \
  --save-dir "$RUNPATH/data/air-10_prediction/lr" \
  --train-ratio 0.33 \
  --past-window 100 \
  --prediction-step 1
