#!/bin/bash
#SBATCH -J nexcp_air_tuning
#SBATCH --account=gts-yxie77-paid
#SBATCH --mem=4GB
#SBATCH -t30
#SBATCH -qinferno
#SBATCH -o Report-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu


RUNPATH=/storage/project/r-yxie77-0/jlee3541/projects/dscp
cd "$RUNPATH"
export PYTHONPATH="$RUNPATH:$PYTHONPATH"
module load anaconda3
conda activate dscp


python sbatch_run_tuning/run_nexcp_tuning.py \
  --base-config configs/nexcp_configs/nexcp_chronos_air_config.yaml \
  --grid-config configs/nexcp_configs/nexcp_air_tuning_config.yaml \
  --save-dir results/tuning/nexcp_air \
  --sequence-index 0 \
  --top-k 3 \
  --seed 2026