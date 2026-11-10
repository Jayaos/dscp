#!/bin/bash
#SBATCH -J rnn_lcp_epanechnikov_solar_tuning
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=gts-yxie77-paid
#SBATCH --gres=gpu:1 -C V100
#SBATCH --mem=32G
#SBATCH -t480
#SBATCH -qinferno
#SBATCH -o Report-%x-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

set -euo pipefail

# Defaults to one GPU and sequential tuning trials per array task.
# Two GPUs: sbatch --gres=gpu:2 <this-script> --num-gpus 2
# Kernel and bandwidth candidates come from the grid YAML.

# Array tasks 0/1/2 select the LR/LSTM/Chronos base and grid configurations.
TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
case "$TASK_ID" in
  0|1|2) ;;
  *) echo "Expected SLURM_ARRAY_TASK_ID=0, 1, or 2; got '${TASK_ID:-unset}'. Submit with sbatch." >&2; exit 2 ;;
esac
BASE_PREDICTORS=(lr lstm chronos)
BASE_PREDICTOR="${BASE_PREDICTORS[$TASK_ID]}"
BASE_CONFIG="configs/lcp_configs/lcp_rnn_${BASE_PREDICTOR}_solar_config.yaml"
GRID_CONFIG="configs/lcp_epanechnikov_kernel_configs/lcp_rnn_epanechnikov_${BASE_PREDICTOR}_solar_tuning_config.yaml"

RUNPATH="${RUNPATH:-/storage/project/r-yxie77-0/jlee3541/projects/dscp}"
cd "$RUNPATH"
export PYTHONPATH="$RUNPATH:$RUNPATH/sbatch:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg

for CONFIG_PATH in "$BASE_CONFIG" "$GRID_CONFIG"; do
  if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Configuration does not exist: $CONFIG_PATH" >&2
    exit 2
  fi
done

module load anaconda3
conda activate dscp

python -u sbatch/sbatch_run_tuning/run_local_cp_tuning.py \
  --base-config "$BASE_CONFIG" \
  --grid-config "$GRID_CONFIG" \
  --save-dir "results/tuning/lcp_rnn_epanechnikov_{base_predictor}_solar" \
  --sequence-index 0 \
  --top-k 3 \
  --seed 2026 \
  --num-gpus 1 \
  "$@"

