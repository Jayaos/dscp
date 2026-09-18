#!/bin/bash
#SBATCH -J rescp_sapflux
#SBATCH --account=gts-yxie77-paid
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t120
#SBATCH -qinferno
#SBATCH -o Report-rescp-sapflux-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu


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
  --dataset sapflux \
  --base-predictor "$BASE_PREDICTOR" \
  --seed "${RESCP_SEED:-2026}" \
  --num-cores "${SLURM_CPUS_PER_TASK:-1}" \
  "$@"
