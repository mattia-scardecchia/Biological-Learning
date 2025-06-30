#!/bin/bash

#SBATCH --job-name=local-structure
#SBATCH --output=slurm_logs/output/output_%x_%j.txt
#SBATCH --error=slurm_logs/error/error_%x_%j.txt

#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu
#SBATCH --qos=normal

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1


mkdir -p slurm_logs/output slurm_logs/error
module load miniconda3

conda run -p /home/3144860/.conda/envs/bio python scripts/train.py -cn baseline_1layer_largeP --multirun \
    name=local-structure \
    J_D=0.0,0.5 \
    lambda_input_skip=2.0,5.0 \
    lambda_wback=0.45,0.9 \
    save_model_and_data=true \
    num_epochs=30