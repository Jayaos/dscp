#!/bin/bash
#SBATCH -J aci_air
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --qos=inferno
#SBATCH -o Report-aci-air-%A_%a.out

set -euo pipefail

# Submit from the repository root, or set DSCP_RUNPATH / RUNPATH explicitly.
RUNPATH="${DSCP_RUNPATH:-${RUNPATH:-${SLURM_SUBMIT_DIR:-$PWD}}}"
cd "$RUNPATH"
export PYTHONPATH="$RUNPATH:$RUNPATH/sbatch:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg

# Tasks 0/1/2 evaluate ACI on the saved LR/LSTM/Chronos forecasts.
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
case "$TASK_ID" in
  0|1|2) ;;
  *) echo "Expected SLURM_ARRAY_TASK_ID=0, 1, or 2; got '$TASK_ID'." >&2; exit 2 ;;
esac
BASE_PREDICTORS=(lr lstm chronos)
BASE_PREDICTOR="${BASE_PREDICTORS[$TASK_ID]}"
# Optional first argument is an exported tuning config; use a single array task.
CONFIG_PATH="${1:-configs/aci_configs/aci_${BASE_PREDICTOR}_air_config.yaml}"
if (( $# > 0 )); then shift; fi

if type module >/dev/null 2>&1; then
  module load "${ACI_CONDA_MODULE:-anaconda3}"
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ACI_CONDA_ENV:-dscp}"

python -u -m sbatch.sbatch_run_aci.run_aci "$CONFIG_PATH" "$@"
