#!/bin/bash

#SBATCH --job-name=bio
#SBATCH --output=slurm_logs/output/output_%x_%j.txt
#SBATCH --error=slurm_logs/error/error_%x_%j.txt

#SBATCH --time=23:00:00
#SBATCH --partition=defq
#SBATCH --nodelist=cnode04
#SBATCH --qos=normal

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G


mkdir -p slurm_logs/output slurm_logs/error
module load miniconda3
conda run -p /home/3144860/.conda/envs/bio python scripts/train.py seed=4
