#!/bin/bash
# Auto-generated script to run lstm_360
# Usage: ./run_lstm_360.sh [args]

# Ensure conda is initialized
source $(conda info --base)/etc/profile.d/conda.sh
conda activate qlib

# Run training
# -u for unbuffered output (better for logs)
echo "Starting training for lstm_360..."
python -u custom_exp/rolling_train.py \
    --model lstm_360 \
    --market csi500 \
    --data_start 2015-01-01 \
    --data_end 2020-12-31 \
    --train_years 3 \
    --test_years 1 \
    "$@"
