#!/bin/bash

#SBATCH --job-name=bio
#SBATCH --output=slurm_logs/output/output_%x_%j.txt
#SBATCH --error=slurm_logs/error/error_%x_%j.txt

#SBATCH --time=3-00:00:00
#SBATCH --partition=long_gpu
#SBATCH --nodelist=gnode02
#SBATCH --qos=normal

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1



mkdir -p slurm_logs/output slurm_logs/error
module load miniconda3

# Training Run
# conda run -p /home/3144860/.conda/envs/bio python scripts/multi_step_train.py --multirun name=multi-step H=100,200,400,800,1600,3200 data.P=6000,600,60,6

# Grid Search
conda run -p /home/3144860/.conda/envs/bio python scripts/grid_search.py --multirun name=tuning num_epochs=20 '+threshold_readout_values=[3.0,7.5]' '+threshold_hidden_values=[0.8,1.0,1.2]' '+weight_decay_J_values=[0.005,0.01]' '+J_D_values=[0.3,0.4,0.5,0.6,0.7]' '+lambda_wback_values=[1.0,2.0]' '+double_dynamics_values=[true,false]'

# Double Dynamics
# conda run -p /home/3144860/.conda/envs/bio python scripts/grid_search.py --multirun name=double-dynamics num_epochs=20 double_dynamics=true,false '+max_steps_values=[3,5,10]' '+J_D_values=[0.1,0.3,0.4,0.5,0.6,0.7,0.9]' '+lambda_wback_values=[0.5,1.0,2.0,3.0,4.0,5.0]'

# Local CE
# conda run -p /home/3144860/.conda/envs/bio python scripts/grid_search.py --multirun name=local-ce num_epochs=20 use_local_ce=true '+beta_ce_values=[1.0,2.5,5.0,7.5,10.0,20.0,30.0,50.0]' '+threshold_hidden_values=[0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4]' '+lr_J_values=[0.01,0.03,0.05,0.09]'

# Feature Learning
# conda run -p /home/3144860/.conda/envs/bio python scripts/grid_search.py --multirun name=feature-learning symmetric_W=buggy lambda_wback=1.0 'lr=[0.03,0.0,0.1]' data.P=6000,600

# Reservoir Computing
# conda run -p /home/3144860/.conda/envs/bio python scripts/grid_search.py --multirun name=dynamic-features symmetric_W=false lambda_wback=0.0 'lr=[0.0,0.0,0.1]' data.P=6000,600

# Random Features
# conda run -p /home/3144860/.conda/envs/bio python scripts/grid_search.py --multirun name=random-features symmetric_W=false lambda_wback=0.0 'lr=[0.0,0.0,0.1]' max_steps=2 init_mode=zeros data.P=6000,600
# conda run -p /home/3144860/.conda/envs/bio python scripts/grid_search.py --multirun name=random-features symmetric_W=false lambda_wback=0.0 'lr=[0.0,0.0,0.1]' max_steps=1 init_mode=zeros lambda_fc=1.0 fc_input=true lambda_internal=0.0 lambda_x=0.0,1000.0 data.P=6000,600