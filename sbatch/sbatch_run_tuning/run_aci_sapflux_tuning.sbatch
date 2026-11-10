#!/bin/bash
#SBATCH --job-name=aci_sapflux_tuning
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --qos=inferno
#SBATCH --output=Report-aci-sapflux-tuning-%A_%a.out

set -euo pipefail
RUNPATH="${DSCP_RUNPATH:-${RUNPATH:-${SLURM_SUBMIT_DIR:-$PWD}}}"
cd "$RUNPATH"
export PYTHONPATH="$RUNPATH:$RUNPATH/sbatch:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
case "$TASK_ID" in
  0|1|2) ;;
  *) echo "Expected array index 0, 1, or 2." >&2; exit 2 ;;
esac
BASE_PREDICTORS=(lr lstm chronos)
BASE_PREDICTOR="${BASE_PREDICTORS[$TASK_ID]}"
if type module >/dev/null 2>&1; then
  module load "${ACI_CONDA_MODULE:-anaconda3}"
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ACI_CONDA_ENV:-dscp}"

python -u -m sbatch.sbatch_run_tuning.run_aci_tuning \
  --base-config "configs/aci_configs/aci_${BASE_PREDICTOR}_sapflux_config.yaml" \
  --grid-config "configs/aci_configs/aci_sapflux_tuning_config.yaml" \
  --save-dir "results/tuning/aci_${BASE_PREDICTOR}_sapflux" \
  --sequence-index 0 --top-k 3 --seed 2026 "$@"
