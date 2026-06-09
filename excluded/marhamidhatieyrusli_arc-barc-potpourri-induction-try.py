# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Create the directory where the wheel files will be saved in the writable /kaggle/working/ folder
# !mkdir -p /kaggle/working/pkgs

# # Download the packages to the /kaggle/working/pkgs directory
# !pip download \
#     --dest /kaggle/working/pkgs \
#     transformers bitsandbytes accelerate arc-py torchvision unsloth trl sentence-transformers scikit-learn tqdm func_timeout umap-learn


%pip install --no-index \
  --find-links /kaggle/input/d/marhamidhatieyrusli/packages/pkgs \
  transformers bitsandbytes accelerate arc-py torchvision unsloth trl peft sentence-transformers scipy scikit-learn tqdm umap-learn

%pip install --no-index \
  --find-links /kaggle/input/packages-func-timeout \
  func-timeout


from unsloth import FastLanguageModel
import transformers.utils.import_utils as _iu
_iU = _iu 
# _iU.is_torchvision_available = lambda : False

# import sys
# sys.modules.pop("torchvision", None)
# sys.modules.pop("torchvision.io", None)
# sys.modules.pop("torchvision.transforms", None)


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import json
import multiprocessing as mp
from multiprocessing import Process, Manager
import numpy as np
import torch
import re
from typing import List, Dict, Tuple, Optional, Any, Union, TypedDict
from enum import Enum
from collections import deque, Counter

import sys
sys.path.insert(0, "/kaggle/input/python-files") 

import traceback
import gc
import logging
from functools import partial, lru_cache
from datasets import Dataset
from datetime import datetime
from tqdm import tqdm
from peft import LoraConfig

# SentenceTransformer related imports
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from scipy import ndimage

# TRL library imports
from trl import GRPOTrainer
from trl import GRPOConfig

# Import arc-py modules
# from arc import train_problems, validation_problems
# from arc.read import parse_dir
# from arc.types import ArcIOPair, ArcProblem

# PyTorch/CUDA optimization settings
torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

if '/kaggle/input/worker-ttrl-py' not in sys.path:
    sys.path.insert(0, '/kaggle/input/worker-ttrl-py')

# import the functions from worker.py
try:
    from worker_ttrl import solve_ttrl_problem
    print("Successfully imported solve_ttrl_problem from worker_ttrl.py")
except ImportError as e:
    print(f"Error importing from worker_ttrl.py: {e}")
    print("Please ensure worker_ttrl.py was written correctly and all its internal imports are satisfied by installed packages.")
    raise


# === Configuration Constants (Adjust as needed) ===
MODEL_PATH       = "/kaggle/input/barc-potpourri-induction8b/Llama-3.1-ARC-Potpourri-Induction-8B"
MAX_SEQ_LENGTH   = 4096 # Max sequence length for TTRL (was 8192 in original notebook, TTRL used 4096)
USE_4BIT         = True # Set to True for 4-bit quantization, False for FP16/BF16
# USE_FLOAT16 is implicitly handled by Unsloth based on USE_4BIT and model's native precision.

# TTRL Specific Parameters
TTRL_MAX_ITERATIONS = 3       # Max TTRL iterations per problem (reduced for Kaggle time limits)
TTRL_STEPS_PER_ITERATION = 2  # Training steps within each TTRL iteration (reduced)
TTRL_NUM_CANDIDATES = 3       # Number of solutions to generate per TTRL iteration (reduced)
TTRL_LEARNING_RATE = 2e-5     # Learning rate for TTRL fine-tuning
TTRL_CODE_EXEC_TIMEOUT = 7    # Timeout for executing generated code (seconds)

# Path to your offline SentenceTransformer model dataset on Kaggle
# Example: "/kaggle/input/sentence-all-mpnet-base-v2/all-mpnet-base-v2"
# Set to None if not using or to disable description diversity reward based on embeddings
SENTENCE_TRANSFORMER_MODEL_PATH = "/kaggle/input/all-mpnet-simcse-256d-supervised-concepts/all-mpnet-simcse-256d-supervised-concepts"


def parallelize_runs(
    gpu_quotas_bytes: List[int], 
    task_requirements_bytes_list: List[int], 
    task_args_list: List[Dict[str, Any]], # Each dict here will contain args for solve_ttrl_problem
    # n_iterations_per_worker_task: int, # This is now TTRL_MAX_ITERATIONS, passed in task_args_list
    end_time_for_all_tasks: float, # Overall end time for the submission run
    num_cpus_available: int, 
    num_gpus_available: int,
    # Removed g_model_path_const etc., as they are now part of task_args_list
    verbose: bool = False
):
    n_total_tasks = len(task_args_list)
    current_gpu_available_bytes = list(gpu_quotas_bytes) 

    t_start_parallel = time.time()

    tasks_started_flags = [False] * n_total_tasks
    tasks_finished_flags = [False] * n_total_tasks
    active_processes_list = [None] * n_total_tasks 
    process_to_gpu_assignment_map = [None] * n_total_tasks

    manager = Manager()
    solutions_shared_dict = manager.dict() 
    error_shared_queue = manager.Queue()   

    processed_tasks_count = 0
    task_scheduling_pointer = 0 

    while processed_tasks_count < n_total_tasks:
        if time.time() >= end_time_for_all_tasks:
            print("Overall time limit reached in parallelize_runs. Stopping task scheduling.")
            # Send terminate signal to running processes? (More complex)
            # For now, just stop scheduling new ones. Existing ones will run until their own end_time_worker.
            break

        # Check for errors from workers
        while not error_shared_queue.empty():
            try:
                err_msg = error_shared_queue.get_nowait()
                if verbose: print(f"Error from worker: {err_msg[:500]}...") # Print snippet
            except multiprocessing.queues.Empty: 
                break
            except Exception as q_err: # Handle other queue errors
                if verbose: print(f"Queue reading error: {q_err}")
                break
        
        # 1. Reap finished processes
        for i in range(n_total_tasks):
            if tasks_started_flags[i] and not tasks_finished_flags[i]:
                process_obj = active_processes_list[i]
                if process_obj is not None and not process_obj.is_alive():
                    process_obj.join(timeout=0.1) 
                    tasks_finished_flags[i] = True
                    assigned_gpu_id = process_to_gpu_assignment_map[i]
                    if assigned_gpu_id is not None: # Ensure GPU was assigned
                         current_gpu_available_bytes[assigned_gpu_id] += task_requirements_bytes_list[i]
                    processed_tasks_count += 1
                    if verbose:
                        task_info = task_args_list[i]
                        print(f"<<< Finished: {task_info['task_name']} (idx {i}) on GPU {assigned_gpu_id}. Reclaimed memory. Total finished: {processed_tasks_count}/{n_total_tasks}")
                    active_processes_list[i] = None 

        # 2. Schedule new tasks
        if task_scheduling_pointer < n_total_tasks:
            # Simple round-robin GPU assignment for available slots
            # More sophisticated: find GPU with most free memory that fits the task
            for gpu_id_to_try in range(num_gpus_available):
                if task_scheduling_pointer >= n_total_tasks: 
                    break

                current_task_index = task_scheduling_pointer
                
                if tasks_started_flags[current_task_index]:
                    # This should not happen if task_scheduling_pointer is managed correctly
                    # print(f"Warning: Task {current_task_index} already started, but tried to schedule again.")
                    task_scheduling_pointer +=1 
                    continue

                num_currently_active_procs = sum(1 for p in active_processes_list if p is not None and p.is_alive())
                
                task_mem_needed_bytes = task_requirements_bytes_list[current_task_index]
                
                is_enough_mem_on_gpu = current_gpu_available_bytes[gpu_id_to_try] >= task_mem_needed_bytes
                is_cpu_slot_available = num_currently_active_procs < num_cpus_available 
                
                # Individual worker end time: min of its own max duration or overall deadline
                # Max duration per worker can be set, e.g. total_time / num_tasks_per_gpu_wave
                # For now, all workers get the global end_time_for_all_tasks.
                # The worker's internal TTRL loop should also check this.
                worker_specific_end_time = end_time_for_all_tasks


                if is_enough_mem_on_gpu and is_cpu_slot_available:
                    current_gpu_available_bytes[gpu_id_to_try] -= task_mem_needed_bytes
                    
                    # task_args_list[current_task_index] already contains all args for solve_ttrl_problem
                    worker_args_dict = task_args_list[current_task_index]
                    worker_args_dict['gpu_id'] = gpu_id_to_try # Assign GPU
                    worker_args_dict['solutions_dict_shared'] = solutions_shared_dict
                    worker_args_dict['error_queue_shared'] = error_shared_queue
                    worker_args_dict['end_time_worker'] = worker_specific_end_time # Pass deadline

                    # Target the imported solve_ttrl_problem
                    p = Process(target=solve_ttrl_problem, kwargs=worker_args_dict) 
                    p.start()
                    
                    active_processes_list[current_task_index] = p
                    tasks_started_flags[current_task_index] = True
                    process_to_gpu_assignment_map[current_task_index] = gpu_id_to_try
                    
                    if verbose:
                        print(f">>> Started: {worker_args_dict['task_name']} (idx {current_task_index}) on GPU {gpu_id_to_try}. "
                              f"Mem needed: {task_mem_needed_bytes / (1024**3):.2f} GiB. "
                              f"GPU {gpu_id_to_try} free: {current_gpu_available_bytes[gpu_id_to_try] / (1024**3):.2f} GiB. "
                              f"Active Procs: {num_currently_active_procs + 1}")
                    
                    task_scheduling_pointer += 1 
                    break # Scheduled one task, re-evaluate GPUs for next task (or continue to fill current GPU)

        if processed_tasks_count >= n_total_tasks and task_scheduling_pointer >= n_total_tasks:
            all_processes_joined = True
            for i in range(n_total_tasks):
                if active_processes_list[i] is not None and active_processes_list[i].is_alive():
                    active_processes_list[i].join(timeout=1) 
                    if active_processes_list[i].is_alive():
                         all_processes_joined = False 
                         if verbose: print(f"Waiting for task {task_args_list[i]['task_name']} to finish...")
            if all_processes_joined:
                break
        
        if task_scheduling_pointer >= n_total_tasks and processed_tasks_count < n_total_tasks:
             # All tasks launched, just wait for them to finish or timeout
             pass


        time.sleep(1.0) # Polling interval

    # Final cleanup for any processes that might still be marked as active but are done or timed out
    for i in range(n_total_tasks):
        if active_processes_list[i] is not None and active_processes_list[i].is_alive():
            if verbose: print(f"Attempting to join lingering process for task {task_args_list[i]['task_name']}...")
            active_processes_list[i].join(timeout=5) # Short final timeout
            if active_processes_list[i].is_alive():
                if verbose: print(f"Process for task {task_args_list[i]['task_name']} did not terminate. Terminating.")
                active_processes_list[i].terminate() # Force terminate if still alive
                active_processes_list[i].join() # Wait for termination
                # Update counts if it was terminated now
                if not tasks_finished_flags[i]:
                    tasks_finished_flags[i] = True
                    processed_tasks_count +=1
                    # Add error to queue for this terminated task
                    error_shared_queue.put(f"Task {task_args_list[i]['task_name']} terminated due to overall timeout.")


    # Final error check
    while not error_shared_queue.empty():
        try:
            err_msg = error_shared_queue.get_nowait()
            if verbose: print(f"Final error check from worker: {err_msg[:500]}...")
        except multiprocessing.queues.Empty:
            break
        except Exception as q_err:
            if verbose: print(f"Final queue reading error: {q_err}")
            break
            
    final_solutions_dict = dict(solutions_shared_dict) 
    time_taken_parallel = time.time() - t_start_parallel

    if verbose:
        print(f"All {n_total_tasks} tasks processing loop completed in {time_taken_parallel:.2f} sec.")
        if processed_tasks_count < n_total_tasks:
             print(f"Warning: Only {processed_tasks_count}/{n_total_tasks} tasks confirmed finished.")
    
    return final_solutions_dict, time_taken_parallel


# Adjust these paths if your Kaggle input structure is different
ARC_DATA_DIR = '/kaggle/input/arc-prize-2025'
# Fallback for local testing (if files are in the same directory as the notebook)
if not os.path.exists(ARC_DATA_DIR):
    ARC_DATA_DIR = '.' 

training_solutions_path = os.path.join(ARC_DATA_DIR, 'arc-agi_training_solutions.json')
evaluation_solutions_path = os.path.join(ARC_DATA_DIR, 'arc-agi_evaluation_solutions.json')
evaluation_challenges_path = os.path.join(ARC_DATA_DIR, 'arc-agi_evaluation_challenges.json')
sample_submission_path = os.path.join(ARC_DATA_DIR, 'sample_submission.json')
training_challenges_path = os.path.join(ARC_DATA_DIR, 'arc-agi_training_challenges.json')
# This is the one we'll process for submission:
test_challenges_path = os.path.join(ARC_DATA_DIR, 'arc-agi_test_challenges.json')



def load_json_data(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return {} 

# test_challenges_data will be loaded in __main__
sample_submission_data = load_json_data(sample_submission_path) # Load if needed for format reference


import torch
import multiprocessing
from multiprocessing import Manager
import time
import json
import os
from typing import Dict, Any, List

# Ensure these imports are available from your notebook's setup
# from worker import solve_bar_problem # Assuming this is correctly imported
# from utils import parse_dir, load_json_data # Assuming these are correctly imported

# --- Global Constants (Ensure these are defined in your notebook, e.g., from Cell 6) ---
# Example:
# MODEL_PATH = "/kaggle/input/model-name/model_files"
# USE_4BIT = True
# USE_FLOAT16 = False
# MAX_SEQ_LENGTH = 8192
# test_challenges_path = "/kaggle/input/arc-agi-2024/arc-agi/data/evaluation" # Assuming this path is defined


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("FATAL: No CUDA GPUs available. This script requires GPUs.")
        exit()

    try:
        # On Linux/macOS, 'spawn' is a good default for CUDA. 'fork' can be problematic.
        # Kaggle notebooks run on Linux.
        if multiprocessing.get_start_method(allow_none=True) != 'spawn':
            multiprocessing.set_start_method('spawn', force=True)
            print("Set multiprocessing start method to 'spawn'.")
    except RuntimeError as e:
        print(f"Note: multiprocessing start method already set or cannot be forced: {e}")
    except AttributeError: # for older python versions that might not have get_start_method with allow_none
        try:
            multiprocessing.set_start_method('spawn', force=True)
            print("Set multiprocessing start method to 'spawn' (fallback).")
        except RuntimeError as e_f:
             print(f"Note: multiprocessing start method already set or cannot be forced (fallback): {e_f}")


    overall_start_time = time.time()
    
    # Kaggle GPU notebook time limit (e.g., 9 hours = 32400 seconds)
    # L4x2 typically has 9 hours.
    TOTAL_RUNTIME_SECONDS = int(os.environ.get('KAGGLE_KERNEL_RUN_TIME', 9 * 3600)) # Get from env if available
    SAFETY_BUFFER_SECONDS = 20 * 60    # 20 minutes safety buffer
    effective_end_time_overall = overall_start_time + TOTAL_RUNTIME_SECONDS - SAFETY_BUFFER_SECONDS
    
    num_logical_cpus = multiprocessing.cpu_count()
    num_physical_gpus = torch.cuda.device_count()

    print(f"Starting ARC TTRL Solver. Detected {num_logical_cpus} CPUs and {num_physical_gpus} GPUs.")
    if num_physical_gpus == 0:
        print("FATAL: No GPUs detected by PyTorch. Exiting.")
        exit()

    print(f"Loading test challenges from: {test_challenges_path}")
    all_test_challenges_data = load_json_data(test_challenges_path)
    if not all_test_challenges_data:
        # Try one more common location for ARC Prize data
        print("Primary test challenges path failed. Trying /kaggle/input/arc-prize-2024/test_tasks.json")
        test_challenges_path_alt = "/kaggle/input/arc-prize-2024/test_tasks.json"
        all_test_challenges_data = load_json_data(test_challenges_path_alt)
        if not all_test_challenges_data:
            print("FATAL: Could not load test challenges from any known path. Exiting.")
            exit()
        else:
            test_challenges_path = test_challenges_path_alt # Update path if alternative worked
            print(f"Successfully loaded test challenges from {test_challenges_path}")
        
    task_names_list = list(all_test_challenges_data.keys())
    num_total_tasks_to_solve = len(task_names_list)
    print(f"Found {num_total_tasks_to_solve} tasks in the test set.")
    if num_total_tasks_to_solve == 0:
        print("No tasks found. Creating dummy submission.json and exiting.")
        with open('submission.json', 'w') as f_out:
            json.dump({}, f_out)
        exit()


    # --- GPU Memory Configuration ---
    # Estimate memory per TTRL task. This is complex due to model loading, data, and optimizer states.
    # Llama-3.1 8B 4-bit needs ~5-6GB for inference. LoRA fine-tuning adds overhead.
    # Let's estimate higher for TTRL.
    if USE_4BIT:
        MEM_PER_TASK_BYTES = int(10 * (1024**3)) # Estimate 10 GiB for 4-bit TTRL (conservative)
        print(f"Using 4-bit quantization. Estimated memory per TTRL task: {MEM_PER_TASK_BYTES / (1024**3):.2f} GiB")
    else: # FP16/BF16
        MEM_PER_TASK_BYTES = int(22 * (1024**3)) # Estimate 22 GiB for FP16 TTRL (Llama 8B FP16 is ~16GB + overhead)
        print(f"Using FP16/BF16. Estimated memory per TTRL task: {MEM_PER_TASK_BYTES / (1024**3):.2f} GiB")

    initial_gpu_free_bytes_info = []
    for i in range(num_physical_gpus):
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(i)
            initial_gpu_free_bytes_info.append((free_mem, total_mem))
        except Exception as e:
            print(f"Could not get mem_info for GPU {i}: {e}. Assuming 0 free memory.")
            initial_gpu_free_bytes_info.append((0,0))


    gpus_available_memory_bytes = [free for free, total in initial_gpu_free_bytes_info]
    print(f"Initial GPU free memory (GiB): {[mem / (1024**3) for mem in gpus_available_memory_bytes]}")

    # L4 GPUs on Kaggle typically have 22-24GB.
    # Safety margin per GPU (e.g., for CUDA context, unsloth overheads not perfectly captured)
    SAFETY_MARGIN_PER_GPU_BYTES = int(2 * (1024**3)) 
    gpus_usable_memory_bytes = [max(0, mem - SAFETY_MARGIN_PER_GPU_BYTES) for mem in gpus_available_memory_bytes]
    print(f"Usable GPU memory after safety margin (GiB): {[mem / (1024**3) for mem in gpus_usable_memory_bytes]}")

    if not any(mem >= MEM_PER_TASK_BYTES for mem in gpus_usable_memory_bytes):
        print(f"FATAL: No GPU has enough usable memory ({MEM_PER_TASK_BYTES / (1024**3):.2f} GiB required) for a single TTRL task. Exiting.")
        # Create a dummy submission if this happens, so the run doesn't fail.
        submission_output_dict_fallback = {}
        for task_name_fallback in task_names_list:
            num_test_inputs_fallback = len(all_test_challenges_data[task_name_fallback].get('test',[])) if task_name_fallback in all_test_challenges_data else 1
            if num_test_inputs_fallback == 0: num_test_inputs_fallback = 1
            fallback_list_dummy = [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in range(num_test_inputs_fallback)]
            submission_output_dict_fallback[task_name_fallback] = fallback_list_dummy
        with open('submission.json', 'w') as f_out_dummy:
            json.dump(submission_output_dict_fallback, f_out_dummy, indent=4)
        print("Created dummy submission.json due to insufficient GPU memory.")
        exit()
        
    task_memory_requirements_bytes = [MEM_PER_TASK_BYTES] * num_total_tasks_to_solve
    
    process_args_for_tasks = []
    for task_idx in range(num_total_tasks_to_solve):
        current_task_name = task_names_list[task_idx]
        current_challenge_data = all_test_challenges_data[current_task_name]
        process_args_for_tasks.append({
            "task_name": current_task_name,
            "challenge_data": current_challenge_data,
            # Model and TTRL params common to all tasks
            "g_model_path_const": MODEL_PATH,
            "g_use_4bit_const": USE_4BIT,
            "g_max_seq_length_const": MAX_SEQ_LENGTH,
            "ttrl_max_iterations": TTRL_MAX_ITERATIONS,
            "ttrl_steps_per_iteration": TTRL_STEPS_PER_ITERATION,
            "ttrl_num_candidates": TTRL_NUM_CANDIDATES,
            "ttrl_learning_rate": TTRL_LEARNING_RATE,
            "code_exec_timeout": TTRL_CODE_EXEC_TIMEOUT,
            "sentence_transformer_path": SENTENCE_TRANSFORMER_MODEL_PATH,
            # gpu_id, solutions_dict_shared, error_queue_shared, end_time_worker added by parallelize_runs
        })
    
    print(f"Starting parallel execution of {num_total_tasks_to_solve} TTRL tasks...")
    print(f"Overall timeout set to {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(effective_end_time_overall))}")

    final_solutions, time_taken_for_parallel_run = parallelize_runs(
        gpu_quotas_bytes=gpus_usable_memory_bytes,
        task_requirements_bytes_list=task_memory_requirements_bytes,
        task_args_list=process_args_for_tasks,
        end_time_for_all_tasks=effective_end_time_overall, # Pass the overall deadline
        num_cpus_available=num_logical_cpus,
        num_gpus_available=num_physical_gpus,
        verbose=True 
    )

    print(f"\nParallel TTRL execution finished in {time_taken_for_parallel_run:.2f} seconds.")
    print(f"{len(final_solutions)} tasks have results in the solutions dictionary.")

    # --- Create Submission File ---
    submission_output_dict = {}
    for task_name_key in task_names_list: # Iterate in original order
        if task_name_key in final_solutions:
            # final_solutions[task_name_key] is already the list of {"attempt_1": ..., "attempt_2": ...}
            # as prepared by solve_ttrl_problem
            submission_output_dict[task_name_key] = final_solutions[task_name_key]
        else:
            # Fallback for tasks not found in final_solutions (e.g., worker failed catastrophically)
            print(f"Warning: Task {task_name_key} not found in final solutions. Using default fallback.")
            num_test_inputs = len(all_test_challenges_data[task_name_key].get('test',[])) if task_name_key in all_test_challenges_data else 1
            if num_test_inputs == 0: num_test_inputs = 1 # Ensure at least one fallback entry
            
            fallback_list = [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in range(num_test_inputs)]
            submission_output_dict[task_name_key] = fallback_list

    submission_file_path = 'submission.json'
    with open(submission_file_path, 'w') as f_out:
        json.dump(submission_output_dict, f_out) # No indent for smaller file size
    
    print(f"\nSubmission file '{submission_file_path}' created successfully.")
    num_tasks_in_submission = len(submission_output_dict)
    print(f"Submission contains results for {num_tasks_in_submission} tasks.")

    # Basic check for non-default solutions
    successful_solves_count = 0
    for task_name_check, solutions_list_check in submission_output_dict.items():
        if solutions_list_check: # solutions_list_check is a list of dicts
            for attempt_dict_check in solutions_list_check: # Each dict is for one test input
                # Check if 'attempt_1' is not the default [[0]]
                attempt_1_grid = attempt_dict_check.get("attempt_1", [[0]])
                # A more robust check for default:
                is_default = (isinstance(attempt_1_grid, list) and
                              len(attempt_1_grid) == 1 and
                              isinstance(attempt_1_grid[0], list) and
                              len(attempt_1_grid[0]) == 1 and
                              attempt_1_grid[0][0] == 0)
                if not is_default:
                    successful_solves_count += 1
                    break # Count this task as having at least one non-default solution for one test input
    
    print(f"Number of tasks with at least one non-default solution for attempt_1: {successful_solves_count} out of {num_tasks_in_submission}")
    
    overall_time_taken = time.time() - overall_start_time
    print(f"Total script execution time: {overall_time_taken:.2f} seconds.")
    print(f"Remaining time from Kaggle limit: {(TOTAL_RUNTIME_SECONDS - overall_time_taken)/60:.2f} minutes (approx).")

