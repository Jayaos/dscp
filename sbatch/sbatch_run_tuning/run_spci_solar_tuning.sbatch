#!/bin/bash
#SBATCH -J spci_solar_tuning
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t720
#SBATCH -qinferno
#SBATCH -o Report-spci-solar-tuning-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

set -euo pipefail

# Array tasks 0/1/2 tune SPCI on LR/LSTM/Chronos forecasts.
TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
case "$TASK_ID" in
  0|1|2) ;;
  *) echo "Expected SLURM_ARRAY_TASK_ID=0, 1, or 2; got '${TASK_ID:-unset}'. Submit with sbatch." >&2; exit 2 ;;
esac
BASE_PREDICTORS=(lr lstm chronos)
BASE_PREDICTOR="${BASE_PREDICTORS[$TASK_ID]}"

RUNPATH="${RUNPATH:-/storage/project/r-yxie77-0/jlee3541/projects/dscp}"
cd "$RUNPATH"
export PYTHONPATH="$RUNPATH:$RUNPATH/sbatch:${PYTHONPATH:-}"
# Trials run sequentially; bound each forest to the allocated CPUs.
export LOKY_MAX_CPU_COUNT="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate spci

python -u -m sbatch.sbatch_run_tuning.run_spci_tuning \
  --base-config "configs/spci_configs/spci_${BASE_PREDICTOR}_solar_config.yaml" \
  --grid-config configs/spci_configs/spci_solar_tuning_config.yaml \
  --save-dir "results/tuning/spci_{base_predictor}_solar" \
  --sequence-index 0 \
  --top-k 3 \
  --seed 2026 \
  "$@"
