#!/bin/bash
python scripts/grid_search.py -cn baseline_1layer_largeP \
    name=tuning-JD0 \
    J_D=0.0 \
    lambda_input_skip=5.0 \
    double_dynamics=true \
    '+threshold_hidden_values=[0.0,0.3,0.6,0.9,1.2]' \
    '+lr_J_values=[0.001,0.005,0.01,0.02,0.03,0.05]' \
    lambda_wback=0.3,0.6,0.9 \
    num_epochs=20 \
    --multirun

# NOTE: remember to vary lr_J and lr_input_skip together!