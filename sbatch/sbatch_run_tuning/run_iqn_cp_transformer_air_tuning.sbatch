#!/bin/bash
#SBATCH -J transformer_iqn_cp_air_tuning
#SBATCH --account=gts-yxie77-paid
#SBATCH --gres=gpu:1 -C V100
#SBATCH --mem=4GB
#SBATCH -t1080
#SBATCH -qinferno
#SBATCH -o Report-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu


RUNPATH=/storage/project/r-yxie77-0/jlee3541/projects/dscp
cd "$RUNPATH"
export PYTHONPATH="$RUNPATH:$RUNPATH/sbatch:$PYTHONPATH"
module load anaconda3
conda activate dscp


python sbatch/sbatch_run_tuning/run_iqn_cp_tuning.py \
  --base-config configs/iqn_cp_configs/iqn_transformer_air_config.yaml \
  --grid-config configs/iqn_cp_configs/iqn_transformer_air_tuning_config.yaml \
  --save-dir results/tuning/iqn_transformer_chronos_air_partially_monotonic \
  --sequence-index 0 \
  --top-k 3 \
  --seed 2026
