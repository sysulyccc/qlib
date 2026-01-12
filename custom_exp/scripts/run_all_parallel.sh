#!/bin/bash
# Run all models in parallel batches

source $(conda info --base)/etc/profile.d/conda.sh
conda activate qlib

mkdir -p logs_full_run

echo 'Launching mlp_158 on GPU 0 (Log: logs_full_run/mlp_158.log)'
./custom_exp/scripts/run_mlp_158_full.sh --gpu 0 > logs_full_run/mlp_158.log 2>&1 &
echo 'Launching mlp_360 on GPU 1 (Log: logs_full_run/mlp_360.log)'
./custom_exp/scripts/run_mlp_360_full.sh --gpu 1 > logs_full_run/mlp_360.log 2>&1 &
echo 'Launching gru_158_20 on GPU 2 (Log: logs_full_run/gru_158_20.log)'
./custom_exp/scripts/run_gru_158_20_full.sh --gpu 2 > logs_full_run/gru_158_20.log 2>&1 &
echo 'Launching gru_158_32 on GPU 3 (Log: logs_full_run/gru_158_32.log)'
./custom_exp/scripts/run_gru_158_32_full.sh --gpu 3 > logs_full_run/gru_158_32.log 2>&1 &

# Wait for batch to prevent overload
wait
echo 'Batch finished. Starting next batch...'

echo 'Launching gru_360 on GPU 0 (Log: logs_full_run/gru_360.log)'
./custom_exp/scripts/run_gru_360_full.sh --gpu 0 > logs_full_run/gru_360.log 2>&1 &
echo 'Launching transformer_158 on GPU 1 (Log: logs_full_run/transformer_158.log)'
./custom_exp/scripts/run_transformer_158_full.sh --gpu 1 > logs_full_run/transformer_158.log 2>&1 &
echo 'Launching transformer_360 on GPU 2 (Log: logs_full_run/transformer_360.log)'
./custom_exp/scripts/run_transformer_360_full.sh --gpu 2 > logs_full_run/transformer_360.log 2>&1 &
echo 'Launching hist_360 on GPU 3 (Log: logs_full_run/hist_360.log)'
./custom_exp/scripts/run_hist_360_full.sh --gpu 3 > logs_full_run/hist_360.log 2>&1 &

# Wait for batch to prevent overload
wait
echo 'Batch finished. Starting next batch...'

echo 'Launching igmtf_360 on GPU 0 (Log: logs_full_run/igmtf_360.log)'
./custom_exp/scripts/run_igmtf_360_full.sh --gpu 0 > logs_full_run/igmtf_360.log 2>&1 &
echo 'Launching sfm_360 on GPU 1 (Log: logs_full_run/sfm_360.log)'
./custom_exp/scripts/run_sfm_360_full.sh --gpu 1 > logs_full_run/sfm_360.log 2>&1 &
echo 'Launching tra_360 on GPU 2 (Log: logs_full_run/tra_360.log)'
./custom_exp/scripts/run_tra_360_full.sh --gpu 2 > logs_full_run/tra_360.log 2>&1 &

wait
echo 'All training jobs finished!'
