#!/bin/bash
#SBATCH -J kowcpi_air_tuning
#SBATCH --account=gts-yxie77-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH -t120
#SBATCH -qinferno
#SBATCH -o Report-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu


RUNPATH=/storage/project/r-yxie77-0/jlee3541/projects/dscp
cd "$RUNPATH"
export PYTHONPATH="$RUNPATH:$PYTHONPATH"
module load anaconda3
conda activate dscp


python sbatch_run_tuning/run_kowcpi_tuning.py \
  --base-config configs/kowcpi_configs/kowcpi_chronos_air_config.yaml \
  --grid-config configs/kowcpi_configs/kowcpi_air_tuning_config.yaml \
  --save-dir results/tuning/kowcpi_tuning_air \
  --top-k 3 \
  --seed 2026 \
  --num-cores "${SLURM_CPUS_PER_TASK}" 


