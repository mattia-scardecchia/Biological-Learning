#!/bin/bash

#SBATCH --job-name=sym-rules
#SBATCH --output=slurm_logs/output/output_%x_%j.txt
#SBATCH --error=slurm_logs/error/error_%x_%j.txt

#SBATCH --time=3-00:00:00
#SBATCH --partition=long_gpu
#SBATCH --qos=normal

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1


mkdir -p slurm_logs/output slurm_logs/error
module load miniconda3

# perceptron rule
conda run -p /home/3144860/.conda/envs/bio python scripts/grid_search.py -cn baseline_1layer_largeP name=symmetric_rules/perceptron \
    '+J_D_values=[0.0,0.5]' \
    '+lambda_input_skip_values=[2.0,5.0]' \
    '+double_dynamics_values=[true,false]' \
    '+symmetric_J_init_values=[true,false]' \
    '+seed_values=[0,1,2]' \
    num_epochs=30

# symmetric rules
conda run -p /home/3144860/.conda/envs/bio python scripts/grid_search.py -cn baseline_1layer_largeP name=symmetric_rules/sym1 \
    '+J_D_values=[0.0,0.5]' \
    '+lambda_input_skip_values=[2.0,5.0]' \
    '+double_dynamics_values=[true,false]' \
    symmetric_J_init=true \
    symmetric_threshold_internal_couplings=true \
    '+seed_values=[0,1,2]' \
    num_epochs=30
conda run -p /home/3144860/.conda/envs/bio python scripts/grid_search.py -cn baseline_1layer_largeP name=symmetric_rules/sym3 \
    '+J_D_values=[0.0,0.5]' \
    '+lambda_input_skip_values=[2.0,5.0]' \
    '+double_dynamics_values=[true,false]' \
    symmetric_J_init=true \
    symmetrize_internal_couplings=true \
    '+seed_values=[0,1,2]' \
    num_epochs=30
conda run -p /home/3144860/.conda/envs/bio python scripts/grid_search.py -cn baseline_1layer_largeP name=symmetric_rules/sym2 \
    '+J_D_values=[0.0,0.5]' \
    '+lambda_input_skip_values=[2.0,5.0]' \
    '+double_dynamics_values=[true,false]' \
    symmetric_J_init=true \
    symmetric_update_internal_couplings=true \
    '+seed_values=[0,1,2]' \
    num_epochs=30
