
import os
from pathlib import Path

# Define models to generate scripts for
models = [
    # GRU 360
    "gru_360", "gru_360_cs",
    # GRU 158 variants
    "gru_158_8_z", "gru_158_8_cs",
    "gru_158_16_z", "gru_158_16_cs",
    "gru_158_32_z", "gru_158_32_cs",
    # LSTM variants
    "lstm_158", "lstm_360", "lstm_360_cs",
    # Baselines
    "lightgbm", "xgboost", "linear", "mlp"
]

script_dir = Path(__file__).parent
script_dir.mkdir(exist_ok=True)

# Template for the run script
template = """#!/bin/bash
# Auto-generated script to run {model}
# Usage: ./run_{model}.sh [args]

# Ensure conda is initialized
source $(conda info --base)/etc/profile.d/conda.sh
conda activate qlib

# Run training
# -u for unbuffered output (better for logs)
echo "Starting training for {model}..."
python -u custom_exp/rolling_train.py \\
    --model {model} \\
    --market csi500 \\
    --data_start 2015-01-01 \\
    --data_end 2020-12-31 \\
    --train_years 3 \\
    --test_years 1 \\
    "$@"
"""

for model in models:
    filename = script_dir / f"run_{model}.sh"
    content = template.format(model=model)
    
    with open(filename, "w") as f:
        f.write(content)
    
    # Make executable
    os.chmod(filename, 0o755)
    print(f"Generated {filename}")

print("Done.")
