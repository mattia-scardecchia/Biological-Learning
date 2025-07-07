#!/bin/bash
python scripts/grid_search.py -cn lambda_advantage name=lambda_advantage \
    J_D=0.0 \
    lambda_input_skip=2.0,5.0 \
    double_dynamics=true \
    num_epochs=30 \
    '+lambda_l_values=[0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]' \
    '+seed_values=[0,1,2]' \
    --multirun