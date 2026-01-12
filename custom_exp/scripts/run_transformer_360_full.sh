#!/bin/bash
# Auto-generated script to run transformer_360
# Usage: ./run_transformer_360_full.sh [args]

# Ensure conda is initialized
source $(conda info --base)/etc/profile.d/conda.sh
conda activate qlib

# Run training
# -u for unbuffered output (better for logs)
echo "Starting FULL training for transformer_360..."
python -u custom_exp/rolling_train.py \
    --model transformer_360 \
    --market csi500 \
    --data_start 2012-01-01 \
    --data_end 2025-12-31 \
    --train_years 6 \
    --test_years 1 \
    --output_dir /home/ethan/qlib/custom_exp/full_train_exp \
    "$@"
