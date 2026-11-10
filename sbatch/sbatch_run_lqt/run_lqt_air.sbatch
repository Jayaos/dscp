#!/bin/bash
#SBATCH -J lqt_air
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --qos=inferno
#SBATCH -o Report-lqt-air-%A_%a.out

set -euo pipefail

# Submit from the repository root, or set DSCP_RUNPATH / RUNPATH explicitly.
RUNPATH="${DSCP_RUNPATH:-${RUNPATH:-${SLURM_SUBMIT_DIR:-$PWD}}}"
cd "$RUNPATH"
export PYTHONPATH="$RUNPATH:$RUNPATH/sbatch:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg

# Tasks 0/1/2 evaluate LQT with LR/LSTM/Chronos forecasts.
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
case "$TASK_ID" in
  0|1|2) ;;
  *) echo "Expected SLURM_ARRAY_TASK_ID=0, 1, or 2; got '$TASK_ID'." >&2; exit 2 ;;
esac
BASE_PREDICTORS=(lr lstm chronos)
BASE_PREDICTOR="${BASE_PREDICTORS[$TASK_ID]}"
CONFIG_PATH="configs/lqt_configs/lqt_${BASE_PREDICTOR}_air_config.yaml"

if type module >/dev/null 2>&1; then
  module load "${LQT_CONDA_MODULE:-anaconda3}"
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${LQT_CONDA_ENV:-dscp}"

# Use one sequence worker per allocated CPU, capped by the number of sequences.
python -u -m sbatch.sbatch_run_lqt.run_lqt "$CONFIG_PATH" --num-cores "${SLURM_CPUS_PER_TASK:-1}"
