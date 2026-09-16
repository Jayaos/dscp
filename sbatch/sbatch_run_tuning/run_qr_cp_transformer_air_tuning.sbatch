#!/bin/bash
#SBATCH -J transformer_qr_cp_air_tuning
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=gts-yxie77-paid
#SBATCH --gres=gpu:2 -C V100
#SBATCH --mem=16GB
#SBATCH -t20
#SBATCH -qinferno
#SBATCH -o Report-%x-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

set -euo pipefail

# Defaults to two GPUs and two trial workers per array task.
# Single GPU: sbatch --gres=gpu:1 --cpus-per-task=2 --mem=8G <this-script> --num-gpus 1

# Array tasks 0/1/2 select the LR/LSTM/Chronos base configurations.
TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
case "$TASK_ID" in
  0|1|2) ;;
  *) echo "Expected SLURM_ARRAY_TASK_ID=0, 1, or 2; got '${TASK_ID:-unset}'. Submit with sbatch." >&2; exit 2 ;;
esac
BASE_PREDICTORS=(lr lstm chronos)
BASE_PREDICTOR="${BASE_PREDICTORS[$TASK_ID]}"
BASE_CONFIG="configs/qr_cp_configs/qr_transformer_${BASE_PREDICTOR}_air_config.yaml"

RUNPATH="${RUNPATH:-/storage/project/r-yxie77-0/jlee3541/projects/dscp}"
cd "$RUNPATH"
export PYTHONPATH="$RUNPATH:$RUNPATH/sbatch:${PYTHONPATH:-}"
module load anaconda3
conda activate dscp

python -u sbatch/sbatch_run_tuning/run_qr_cp_tuning.py \
  --base-config "$BASE_CONFIG" \
  --grid-config configs/qr_cp_configs/qr_transformer_air_tuning_config.yaml \
  --save-dir "results/tuning/qr_transformer_{base_predictor}_air" \
  --sequence-index 0 \
  --top-k 3 \
  --seed 2026 \
  --num-gpus 2 \
  "$@"
