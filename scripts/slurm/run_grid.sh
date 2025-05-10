#!/bin/bash

#SBATCH --job-name=bio
#SBATCH --output=slurm_logs/output/output_%x_%j.txt
#SBATCH --error=slurm_logs/error/error_%x_%j.txt

#SBATCH --time=23:00:00
#SBATCH --partition=compute
#SBATCH --nodelist=cnode08
#SBATCH --qos=normal

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G


mkdir -p slurm_logs/output slurm_logs/error
module load miniconda3
SEED=${seed:-11}
echo "Running run.sh with seed=$SEED"

# Grid search
conda run -p /home/3144860/.conda/envs/bio python scripts/grid_search.py name=full-mnist seed=$SEED
