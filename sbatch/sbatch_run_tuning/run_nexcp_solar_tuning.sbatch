#!/bin/bash
#SBATCH -J nexcp_solar_tuning
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH -t60
#SBATCH -qinferno
#SBATCH -o Report-nexcp-solar-tuning-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

set -euo pipefail

RUNPATH="${RUNPATH:-/storage/project/r-yxie77-0/jlee3541/projects/dscp}"
cd "$RUNPATH"
export PYTHONPATH="$RUNPATH:$RUNPATH/sbatch:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# Tasks 0/1/2 tune NexCP on LR/LSTM/Chronos Solar forecasts, respectively.
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
case "$TASK_ID" in
  0|1|2) ;;
  *) echo "Expected SLURM_ARRAY_TASK_ID=0, 1, or 2; got '$TASK_ID'." >&2; exit 2 ;;
esac
BASE_PREDICTORS=(lr lstm chronos)
BASE_PREDICTOR="${BASE_PREDICTORS[$TASK_ID]}"

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dscp

python -u -m sbatch.sbatch_run_tuning.run_nexcp_tuning \
  --base-config "configs/nexcp_configs/nexcp_${BASE_PREDICTOR}_solar_config.yaml" \
  --grid-config configs/nexcp_configs/nexcp_solar_tuning_config.yaml \
  --save-dir "results/tuning/nexcp_solar/${BASE_PREDICTOR}" \
  --sequence-index 0 \
  --top-k 3 \
  --seed 2026
