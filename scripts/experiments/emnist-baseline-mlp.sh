#!/bin/bash
python3.11 scripts/mlp_train.py "hidden_dims=[200]" "model.beta=1.0,3.0,10.0,20.0,50.0,100.0,1000.0" --multirun;
python3.11 scripts/mlp_train.py "hidden_dims=[400]" "model.beta=1.0,3.0,10.0,20.0,50.0,100.0,1000.0" --multirun;
python3.11 scripts/mlp_train.py "hidden_dims=[800]" "model.beta=1.0,3.0,10.0,20.0,50.0,100.0,1000.0" --multirun;
python3.11 scripts/mlp_train.py "hidden_dims=[1600]" "model.beta=1.0,3.0,10.0,20.0,50.0,100.0,1000.0" --multirun;
python3.11 scripts/mlp_train.py "hidden_dims=[3200]" "model.beta=1.0,3.0,10.0,20.0,50.0,100.0,1000.0" --multirun;
python3.11 scripts/mlp_train.py "hidden_dims=[6400]" "model.beta=1.0,3.0,10.0,20.0,50.0,100.0,1000.0" --multirun;