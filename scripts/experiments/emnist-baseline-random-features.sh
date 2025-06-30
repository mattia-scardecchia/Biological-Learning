#!/bin/bash
python3.11 scripts/grid_search.py --config-name=baseline_1layer_largeP \
    "H=100,200,400,800,1600,3200,6400" \
    num_epochs=200 \
    name="emnist-baseline-random-features" \
    "+J_D_values=[0.5]" \
    lambda_wback=0.0 \
    max_steps_train=2 \
    lr_J=0.0 \
    lr_input_skip=0.0 \
    lambda_internal=0.0 \
    --multirun