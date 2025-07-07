#!/bin/bash
python scripts/grid_search.py -cn baseline_1layer_largeP name=symmetric_rules/perceptron-long \
    J_D=0.5 \
    lambda_input_skip=5.0 \
    double_dynamics=true \
    '+symmetric_J_init_values=[true,false]' \
    '+seed_values=[0,1]' \
    H=400 \
    num_epochs=500 \
    --multirun