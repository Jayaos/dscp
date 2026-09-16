#!/bin/bash
#SBATCH -J rescp_air
#SBATCH --account=gts-yxie77-paid
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%A_%a.out

# Submit from the repository root, or set DSCP_RUNPATH explicitly.
# Array tasks 0, 1 and 2 run LR, LSTM and Chronos, respectively.
set -euo pipefail
RUNPATH="${DSCP_RUNPATH:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$RUNPATH"
export PYTHONPATH="$RUNPATH:$RUNPATH/sbatch:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
module load anaconda3
conda activate "${RESCP_ENV:-rescp}"

BASE_PREDICTORS=(lr lstm chronos)
BASE_PREDICTOR="${BASE_PREDICTORS[${SLURM_ARRAY_TASK_ID:-0}]}"
python -m sbatch.sbatch_run_rescp.run_rescp \
  --dataset air \
  --base-predictor "$BASE_PREDICTOR" \
  --seed "${RESCP_SEED:-2026}" \
  --num-cores "${SLURM_CPUS_PER_TASK:-1}" \
  "$@"
