#!/bin/bash
#SBATCH -J distmatch_air
#SBATCH --account=gts-yxie77-paid
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t10
#SBATCH --output=%x_%A_%a.out

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

# Array tasks 0/1/2 use LR/LSTM/Chronos forecasts, respectively.
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
case "$TASK_ID" in
  0|1|2) ;;
  *) echo "Expected SLURM_ARRAY_TASK_ID=0, 1, or 2; got '$TASK_ID'." >&2; exit 2 ;;
esac
BASE_PREDICTORS=(lr lstm chronos)
BASE_PREDICTOR="${BASE_PREDICTORS[$TASK_ID]}"

CONFIG_PATH="configs/distmatch_configs/distmatch_${BASE_PREDICTOR}_air_config.yaml"
EXTRA_ARGS=()
if [[ -n "${DISTMATCH_SEED:-}" ]]; then
  EXTRA_ARGS+=(--seed "$DISTMATCH_SEED")
fi

# Worker/thread counts come from YAML. The CLI rejects an insufficient CPU allocation.
python -u -m sbatch.sbatch_run_distmatch.run_distmatch \
  "$CONFIG_PATH" "${EXTRA_ARGS[@]}" "$@"

