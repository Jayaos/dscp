#!/bin/bash
#SBATCH -J distmatch_air_tuning
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t720
#SBATCH -qinferno
#SBATCH -o Report-distmatch-air-tuning-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

set -euo pipefail

# Submit from the repository root, or set DSCP_RUNPATH explicitly.
RUNPATH="${DSCP_RUNPATH:-${RUNPATH:-${SLURM_SUBMIT_DIR:-$PWD}}}"
cd "$RUNPATH"
export PYTHONPATH="$RUNPATH:$RUNPATH/sbatch:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg

if type module >/dev/null 2>&1; then
  module load "${DISTMATCH_CONDA_MODULE:-anaconda3}"
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${DISTMATCH_ENV:-distmatch}"

# Tune air validation data; array tasks 0/1/2 use LR/LSTM/Chronos forecasts.
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
case "$TASK_ID" in
  0|1|2) ;;
  *) echo "Expected SLURM_ARRAY_TASK_ID=0, 1, or 2; got '$TASK_ID'." >&2; exit 2 ;;
esac
BASE_PREDICTORS=(lr lstm chronos)
BASE_PREDICTOR="${BASE_PREDICTORS[$TASK_ID]}"

BASE_CONFIG="configs/distmatch_configs/distmatch_${BASE_PREDICTOR}_air_config.yaml"
GRID_CONFIG="${DISTMATCH_GRID_CONFIG:-configs/distmatch_configs/distmatch_air_tuning_config.yaml}"
OUTPUT_ROOT="${DISTMATCH_TUNING_OUTPUT_ROOT:-$RUNPATH/results/tuning/distmatch_air}"
EXTRA_ARGS=()
if [[ -n "${DISTMATCH_SEED:-}" ]]; then
  EXTRA_ARGS+=(--seed "$DISTMATCH_SEED")
fi

# Validate worker allocation/artifact availability before starting the grid.
python -u -m sbatch.sbatch_run_distmatch.run_distmatch \
  "$BASE_CONFIG" --dry-run "${EXTRA_ARGS[@]}"

# Keep output directories distinct across array tasks and tuning submissions.
python -u -m sbatch.sbatch_run_tuning.run_distmatch_tuning \
  --base-config "$BASE_CONFIG" \
  --grid-config "$GRID_CONFIG" \
  --save-dir "$OUTPUT_ROOT/job_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}/$BASE_PREDICTOR" \
  --top-k "${DISTMATCH_TOP_K:-3}" \
  "${EXTRA_ARGS[@]}"
