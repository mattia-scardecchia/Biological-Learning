#!/bin/bash

#SBATCH --job-name=bio
#SBATCH --output=slurm_logs/output/output_%x_%j.txt
#SBATCH --error=slurm_logs/error/error_%x_%j.txt

#SBATCH --time=23:00:00
#SBATCH --partition=defq
#SBATCH --nodelist=cnode01
#SBATCH --qos=normal

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G


mkdir -p slurm_logs/output slurm_logs/error
module load miniconda3

# Grid search
# conda run -p /home/3144860/.conda/envs/bio python scripts/grid_search.py name=grid seed=4

# Train
conda run -p /home/3144860/.conda/envs/bio python scripts/train.py --multirun data.p=0.3,0.35,0.4 data.P=10,20,30 num_epochs=20 name=baseline seed=1

# MLP
# conda run -p /home/3144860/.conda/envs/bio python scripts/train_mlp.py