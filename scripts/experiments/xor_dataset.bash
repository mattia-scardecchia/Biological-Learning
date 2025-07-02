#!/bin/bash

#SBATCH --job-name=xor
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

# ours
# conda run -p /home/3144860/.conda/envs/bio python scripts/train.py -cn xor_dataset_1layer --multirun \
#     name=xor/ours \
#     bias_std=0.0,3.0 \
#     num_epochs=10 \
#     H=100 \
#     seed=0,1,2
conda run -p /home/3144860/.conda/envs/bio python scripts/grid_search.py -cn xor_dataset_1layer \
    name=xor/ours \
    '+bias_std_values=[0.0,3.0]' \
    num_epochs=10 \
    H=100 \
    '+seed_values=[0,1,2]'

# perceptron
conda run -p /home/3144860/.conda/envs/bio python scripts/mlp_train.py -cn xor_mlp --multirun \
    name=xor/perceptron \
    'model.hidden_dims=[]' \
    trainer.max_epochs=10 \
    seed=0,1,2

# mlp
conda run -p /home/3144860/.conda/envs/bio python scripts/mlp_train.py -cn xor_mlp --multirun \
    name=xor/mlp \
    'model.hidden_dims=[100]' \
    trainer.max_epochs=10 \
    model.activation=beta_tanh,square_tanh \
    model.use_bias=true,false \
    model.binarize=true,false \
    seed=0,1,2