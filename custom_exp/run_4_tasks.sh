#!/bin/bash
# Launch 4 rolling training tasks for advanced models: HIST, IGMTF, TRA, SFM
# All using 6-year training window (Start 2010 -> Test 2016)

# Ensure log directory exists
mkdir -p custom_exp/log_launch

echo "Launching 4 advanced model tasks..."

# Task 1: HIST 360 (6y window)
echo "Launching HIST 360 (6y)..."
CUDA_VISIBLE_DEVICES=0 nohup /home/ethan/miniconda3/envs/qlib/bin/python custom_exp/rolling_train.py --model hist_360 --train_years 6 --data_start 2010-01-01 --gpu 0 > custom_exp/log_launch/hist_360_6y.log 2>&1 &

# Task 2: IGMTF 360 (6y window)
echo "Launching IGMTF 360 (6y)..."
CUDA_VISIBLE_DEVICES=1 nohup /home/ethan/miniconda3/envs/qlib/bin/python custom_exp/rolling_train.py --model igmtf_360 --train_years 6 --data_start 2010-01-01 --gpu 0 > custom_exp/log_launch/igmtf_360_6y.log 2>&1 &

# Task 3: TRA 360 (6y window)
echo "Launching TRA 360 (6y)..."
CUDA_VISIBLE_DEVICES=2 nohup /home/ethan/miniconda3/envs/qlib/bin/python custom_exp/rolling_train.py --model tra_360 --train_years 6 --data_start 2010-01-01 --gpu 0 > custom_exp/log_launch/tra_360_6y.log 2>&1 &

# Task 4: SFM 360 (6y window)
echo "Launching SFM 360 (6y)..."
CUDA_VISIBLE_DEVICES=3 nohup /home/ethan/miniconda3/envs/qlib/bin/python custom_exp/rolling_train.py --model sfm_360 --train_years 6 --data_start 2010-01-01 --gpu 0 > custom_exp/log_launch/sfm_360_6y.log 2>&1 &

echo "All tasks launched. Logs are in custom_exp/log_launch/"
