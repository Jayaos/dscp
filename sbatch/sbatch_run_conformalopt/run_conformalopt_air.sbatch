#!/bin/bash
#SBATCH -J conformalopt_air
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-5
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --qos=inferno
#SBATCH -o Report-conformalopt-air-%A_%a.out

set -euo pipefail

# Submit from the repository root, or set DSCP_RUNPATH / RUNPATH explicitly.
RUNPATH="${DSCP_RUNPATH:-${RUNPATH:-${SLURM_SUBMIT_DIR:-$PWD}}}"
cd "$RUNPATH"
export PYTHONPATH="$RUNPATH:$RUNPATH/sbatch:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg

# Tasks 0/1/2: LQT with LR/LSTM/Chronos; 3/4/5: SQT+AR with the same predictors.
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
case "$TASK_ID" in
  0|1|2|3|4|5) ;;
  *) echo "Expected SLURM_ARRAY_TASK_ID=0 through 5; got '$TASK_ID'." >&2; exit 2 ;;
esac
METHODS=(lqt sqt_ar)
BASE_PREDICTORS=(lr lstm chronos)
METHOD="${METHODS[$((TASK_ID / 3))]}"
BASE_PREDICTOR="${BASE_PREDICTORS[$((TASK_ID % 3))]}"
CONFIG_PATH="configs/conformalopt_configs/${METHOD}_${BASE_PREDICTOR}_air_config.yaml"

if type module >/dev/null 2>&1; then
  module load "${CONFORMALOPT_CONDA_MODULE:-anaconda3}"
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONFORMALOPT_CONDA_ENV:-dscp}"

python -u -m sbatch.sbatch_run_conformalopt.run_conformalopt "$CONFIG_PATH"
