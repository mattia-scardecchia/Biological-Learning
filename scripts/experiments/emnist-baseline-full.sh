#!/bin/bash
python3.11 scripts/grid_search.py --config-name=baseline_1layer_largeP \
    "H=100,200,400,800,1600,3200,6400" \
    num_epochs=200 \
    name="emnist-baseline-full"  \
    "+J_D_values=[0.5]" \
    --multirun