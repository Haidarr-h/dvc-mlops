#!/bin/bash
dvc run -n train_model \
    -d data/iris.csv \
    -o models/logistic_regression.pkl \
    -M metrics/accuracy.txt \
    python train.py