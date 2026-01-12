#!/bin/bash
# Study training window length effect (3y vs 6y)
# Target Test Start: 2016-01-01 for Fold 0
# 
# 3y Window: Start 2013-01-01 (+3y = 2016)
# 6y Window: Start 2010-01-01 (+6y = 2016)

# Ensure log directory exists
mkdir -p custom_exp/log_launch

echo "Launching 4 length study tasks..."

# Task 1: GRU 360 (3y window)
echo "Launching GRU 360 (3y, Start 2013)..."
CUDA_VISIBLE_DEVICES=0 nohup /home/ethan/miniconda3/envs/qlib/bin/python custom_exp/rolling_train.py --model gru_360 --train_years 3 --data_start 2013-01-01 --gpu 0 > custom_exp/log_launch/gru_360_3y.log 2>&1 &

# Task 2: GRU 360 (6y window)
echo "Launching GRU 360 (6y, Start 2010)..."
CUDA_VISIBLE_DEVICES=1 nohup /home/ethan/miniconda3/envs/qlib/bin/python custom_exp/rolling_train.py --model gru_360 --train_years 6 --data_start 2010-01-01 --gpu 0 > custom_exp/log_launch/gru_360_6y.log 2>&1 &

# Task 3: GRU 158 (3y window) - Using gru_158_20
echo "Launching GRU 158 (3y, Start 2013)..."
CUDA_VISIBLE_DEVICES=2 nohup /home/ethan/miniconda3/envs/qlib/bin/python custom_exp/rolling_train.py --model gru_158_20 --train_years 3 --data_start 2013-01-01 --gpu 0 > custom_exp/log_launch/gru_158_20_3y.log 2>&1 &

# Task 4: GRU 158 (6y window) - Using gru_158_20
echo "Launching GRU 158 (6y, Start 2010)..."
CUDA_VISIBLE_DEVICES=3 nohup /home/ethan/miniconda3/envs/qlib/bin/python custom_exp/rolling_train.py --model gru_158_20 --train_years 6 --data_start 2010-01-01 --gpu 0 > custom_exp/log_launch/gru_158_20_6y.log 2>&1 &

echo "All tasks launched. Logs are in custom_exp/log_launch/"
