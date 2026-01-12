import os
from pathlib import Path

# Define all models to run (based on updated config.py)
models = [
    # MLP
    "mlp_158",
    "mlp_360",
    # GRU
    "gru_158_20",
    "gru_158_32",
    "gru_360",
    # Transformer
    "transformer_158",
    "transformer_360",
    # HIST
    "hist_360",
    # New models
    "igmtf_360",
    "sfm_360",
    "tra_360",
]

script_dir = Path(__file__).parent
script_dir.mkdir(exist_ok=True)

# Full training configuration
DATA_START = "2012-01-01"
DATA_END = "2025-12-31"
TRAIN_YEARS = 6
TEST_YEARS = 1
OUTPUT_DIR = "/home/ethan/qlib/custom_exp/full_train_exp"

# Template for the run script
template = """#!/bin/bash
# Auto-generated script to run {model}
# Usage: ./run_{model}_full.sh [args]

# Ensure conda is initialized
source $(conda info --base)/etc/profile.d/conda.sh
conda activate qlib

# Run training
# -u for unbuffered output (better for logs)
echo "Starting FULL training for {model}..."
python -u custom_exp/rolling_train.py \\
    --model {model} \\
    --market csi500 \\
    --data_start {data_start} \\
    --data_end {data_end} \\
    --train_years {train_years} \\
    --test_years {test_years} \\
    --output_dir {output_dir} \\
    "$@"
"""

# Generate individual run scripts
for model in models:
    filename = script_dir / f"run_{model}_full.sh"
    content = template.format(
        model=model,
        data_start=DATA_START,
        data_end=DATA_END,
        train_years=TRAIN_YEARS,
        test_years=TEST_YEARS,
        output_dir=OUTPUT_DIR
    )
    
    with open(filename, "w") as f:
        f.write(content)
    
    # Make executable
    os.chmod(filename, 0o755)
    print(f"Generated {filename}")

# Generate a master runner script (parallel execution)
master_script = script_dir / "run_all_parallel.sh"
with open(master_script, "w") as f:
    f.write("#!/bin/bash\n")
    f.write("# Run all models in parallel batches\n\n")
    f.write("source $(conda info --base)/etc/profile.d/conda.sh\n")
    f.write("conda activate qlib\n\n")
    f.write("mkdir -p logs_full_run\n\n")
    
    # Let's generate a smart runner that cycles through GPUs
    gpu_ids = [0, 1, 2, 3] # Detected 4 GPUs
    current_gpu_idx = 0
    
    for i, model in enumerate(models):
        gpu_id = gpu_ids[current_gpu_idx % len(gpu_ids)]
        current_gpu_idx += 1
        
        log_file = f"logs_full_run/{model}.log"
        cmd = f"./custom_exp/scripts/run_{model}_full.sh --gpu {gpu_id} > {log_file} 2>&1 &"
        f.write(f"echo 'Launching {model} on GPU {gpu_id} (Log: {log_file})'\n")
        f.write(f"{cmd}\n")
        
        # Batch control: wait every 4 jobs (one per GPU)
        if (i + 1) % 4 == 0:
            f.write("\n# Wait for batch to prevent overload\n")
            f.write("wait\n")
            f.write("echo 'Batch finished. Starting next batch...'\n\n")

    f.write("\nwait\n")
    f.write("echo 'All training jobs finished!'\n")

os.chmod(master_script, 0o755)
print(f"Generated master script: {master_script}")

print("Done.")
