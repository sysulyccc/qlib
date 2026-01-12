import os
import time
import subprocess
import sys
from collections import deque

# Add current directory to path to import config
sys.path.append(os.getcwd())
from custom_exp.config import ALL_MODELS

# Configuration
MAX_CONCURRENT = 4
GPU_IDS = [0, 1, 2, 3]
PYTHON_PATH = "/home/ethan/miniconda3/envs/qlib/bin/python"
SCRIPT_PATH = "custom_exp/rolling_train.py"
LOG_DIR = "custom_exp/log_launch"

def run_experiments():
    # Get all tasks
    tasks = list(ALL_MODELS.keys())
    tasks.sort() # Sort for deterministic order
    
    task_queue = deque(tasks)
    running_tasks = {} # gpu_id -> (process, model_name, log_file_handle)
    
    # Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)
    
    print(f"Found {len(tasks)} tasks to run.")
    print(f"Tasks: {tasks}")
    print("-" * 50)
    
    while task_queue or running_tasks:
        # Check for finished tasks
        finished_gpus = []
        for gpu_id, (proc, model_name, log_f) in running_tasks.items():
            if proc.poll() is not None: # Process finished
                finished_gpus.append(gpu_id)
                log_f.close()
                
                # Check exit code
                if proc.returncode != 0:
                    print(f"[{time.strftime('%H:%M:%S')}] Task '{model_name}' on GPU {gpu_id} FAILED with exit code {proc.returncode}. See {LOG_DIR}/{model_name}.log")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] Task '{model_name}' on GPU {gpu_id} COMPLETED successfully.")
        
        # Clean up finished tasks
        for gpu_id in finished_gpus:
            del running_tasks[gpu_id]
            
        # Schedule new tasks
        while task_queue and len(running_tasks) < MAX_CONCURRENT:
            # Find an available GPU
            # We prefer lower IDs, but any free one works
            available_gpus = [g for g in GPU_IDS if g not in running_tasks]
            if not available_gpus:
                break 
            
            gpu_id = available_gpus[0]
            model_name = task_queue.popleft()
            
            print(f"[{time.strftime('%H:%M:%S')}] Launching '{model_name}' on GPU {gpu_id}...")
            
            # Construct command
            # We use CUDA_VISIBLE_DEVICES isolation, so we always tell the script to use gpu 0
            cmd = [
                PYTHON_PATH,
                SCRIPT_PATH,
                "--model", model_name,
                "--train_years", "6",
                "--data_start", "2010-01-01",
                "--gpu", "0" 
            ]
            
            # Log file
            log_path = os.path.join(LOG_DIR, f"{model_name}.log")
            log_file = open(log_path, "w")
            
            # Environment variables
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            # Start process
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env
                )
                running_tasks[gpu_id] = (proc, model_name, log_file)
            except Exception as e:
                print(f"Error launching {model_name}: {e}")
                log_file.close()
                # If launch fails, maybe retry or skip? For now, we lose the task but don't crash loop.
                # But actually, let's just not add to running_tasks and maybe put back in queue?
                # No, simpler to just report error.
        
        time.sleep(5) # Check every 5 seconds

    print("-" * 50)
    print("All tasks finished.")

if __name__ == "__main__":
    run_experiments()
