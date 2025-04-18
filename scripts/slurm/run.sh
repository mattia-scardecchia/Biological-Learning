#!/bin/bash

#SBATCH --job-name=bio
#SBATCH --output=slurm_logs/output/output_%x_%j.txt
#SBATCH --error=slurm_logs/error/error_%x_%j.txt

#SBATCH --time=23:00:00
#SBATCH --partition=compute
#SBATCH --nodelist=cnode06
#SBATCH --qos=normal

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G


mkdir -p slurm_logs/output slurm_logs/error
module load miniconda3

Train
conda run -p /home/3144860/.conda/envs/bio python scripts/train.py name=mnist data.P=1000 data.P_eval=20
conda run -p /home/3144860/.conda/envs/bio python scripts/train.py name=mnist data.P=6000 data.P_eval=20

# MLP
# conda run -p /home/3144860/.conda/envs/bio python scripts/train_mlp.py seed=$SEED