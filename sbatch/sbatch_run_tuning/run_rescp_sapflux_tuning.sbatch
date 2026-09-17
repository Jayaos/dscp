#!/bin/bash
#SBATCH -J rescp_sapflux_tuning
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-2
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH -t720
#SBATCH --qos=inferno
#SBATCH --output=Report-rescp-sapflux-tuning-%A_%a.out

set -euo pipefail

# Array tasks 0/1/2 tune LR/LSTM/Chronos forecasts, respectively.
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
case "$TASK_ID" in
  0|1|2) ;;
  *) echo "Expected SLURM_ARRAY_TASK_ID=0, 1, or 2; got '$TASK_ID'." >&2; exit 2 ;;
esac
BASE_PREDICTORS=(lr lstm chronos)
BASE_PREDICTOR="${BASE_PREDICTORS[$TASK_ID]}"

# Submit from the repository root, or set DSCP_RUNPATH explicitly.
RUNPATH="${DSCP_RUNPATH:-${RUNPATH:-${SLURM_SUBMIT_DIR:-$PWD}}}"
cd "$RUNPATH"
export PYTHONPATH="$RUNPATH:$RUNPATH/sbatch:${PYTHONPATH:-}"
# Parallelize independent sequences; keep each worker's numerical threads at one.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg

if type module >/dev/null 2>&1; then
  module load anaconda3
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${RESCP_ENV:-rescp}"

GRID_CONFIG="${RESCP_GRID_CONFIG:-configs/rescp_configs/rescp_sapflux_tuning_config.yaml}"
OUTPUT_ROOT="${RESCP_TUNING_OUTPUT_ROOT:-$RUNPATH/results/tuning/rescp_sapflux}"
OUTPUT_DIR="$OUTPUT_ROOT/job_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}/$BASE_PREDICTOR"

# The tuning YAML controls the inner validation ratio and number of series.
# Each submission/predictor gets separate trial artifacts and best_config.yaml.
python -u -m sbatch.sbatch_run_tuning.run_rescp_tuning \
  --base-config "configs/rescp_configs/rescp_${BASE_PREDICTOR}_sapflux_config.yaml" \
  --grid-config "$GRID_CONFIG" \
  --save-dir "$OUTPUT_DIR" \
  --sequence-index 0 \
  --top-k "${RESCP_TOP_K:-3}" \
  --seed "${RESCP_SEED:-2026}" \
  --num-cores "${SLURM_CPUS_PER_TASK:-4}" \
  "$@"
