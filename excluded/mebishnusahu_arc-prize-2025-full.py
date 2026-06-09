# Part 1: Setup and Imports

import os
import sys
import time
import json
import importlib
import multiprocessing
from multiprocessing import Pool, Process, Manager
from pathlib import Path
import numpy as np
import torch
import logging

# Configure logging for better debugging and monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Use float32 for performance; adjust dtype globally
torch.set_default_dtype(torch.float32)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Constants
MAX_RUNTIME_SECONDS = 12 * 3600  # 12 hours
SAFETY_BUFFER = 5 * 60           # 5 minutes buffer for file saving etc.
SPLIT = "test"                   # Can be 'train', 'evaluation', or 'test'

# Input path setup for Kaggle-style environment
DATA_PATH = Path("../input/arc-prize-2025")
CHALLENGES_FILE = DATA_PATH / f"arc-agi_{SPLIT}_challenges.json"

# Append utility modules to sys.path
sys.path.append("/kaggle/input/compressarc")

# Safe dynamic import of custom preprocessing
def safe_import(name: str, file_path: str):
    """
    Dynamically imports a module from the given file path.
    Logs errors if the module cannot be imported.
    """
    try:
        spec = importlib.util.spec_from_file_location(name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        logging.info(f"âœ… Successfully imported module: {name}")
        return module
    except Exception as e:
        logging.error(f"â�Œ Failed to import module {name}: {e}")
        raise

# Import the rest of the modules
try:
    preprocessing = safe_import("preprocessing", "/kaggle/input/compressarc/preprocessing.py")
    import train
    import arc_compressor
    import initializers
    import multitensor_systems
    import layers
    import solution_selection
    import visualization
    import solve_task
    logging.info("âœ… All required modules imported successfully.")
except Exception as e:
    logging.error(f"â�Œ Error during module imports: {e}")
    raise

# Multiprocessing setup (cross-platform safe)
try:
    if sys.platform == 'win32':
        multiprocessing.set_start_method('spawn', force=True)
    else:
        multiprocessing.set_start_method('fork', force=True)
    logging.info("âœ… Multiprocessing start method set to 'spawn' for cross-platform safety.")
except RuntimeError as e:
    logging.warning(f"âš ï¸� Multiprocessing start method already set: {e}")

# GPU setup and fallback mechanism
try:
    if not torch.cuda.is_available():
        logging.warning("âš ï¸� No GPU found. Falling back to CPU execution.")
        torch.set_default_device('cpu')
    else:
        torch.set_default_device('cuda')
        n_gpus = torch.cuda.device_count()
        gpu_memories = [torch.cuda.mem_get_info(i)[0] for i in range(n_gpus)]
        logging.info(f"ğŸ§  Detected {n_gpus} GPU(s) with memory: {[round(m / 1e9, 2) for m in gpu_memories]} GB")
except Exception as e:
    logging.error(f"â�Œ Error during GPU setup: {e}")
    raise


#Step 2: Task Loader & Device Scanner

import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Constants
MAX_RUNTIME_SECONDS = int(os.getenv("MAX_RUNTIME_SECONDS", 12 * 3600))  # Default: 12 hours
SAFETY_BUFFER = 5 * 60  # 5 minutes buffer for file saving etc.
SPLIT = "test"  # Can be 'train', 'evaluation', or 'test'

# Input path setup for Kaggle-style environment
DATA_PATH = Path(os.getenv("DATA_PATH", "../input/arc-prize-2025"))
CHALLENGES_FILE = DATA_PATH / f"arc-agi_{SPLIT}_challenges.json"

def load_tasks(challenge_file: Path) -> list[str]:
    """Load task names from the ARC-AGI challenge JSON."""
    try:
        logging.info(f"ğŸ”� Loading tasks from: {challenge_file}")
        with open(challenge_file, 'r') as f:
            problems = json.load(f)
        task_ids = list(problems.keys())
        logging.info(f"âœ… Loaded {len(task_ids)} tasks.")
        return task_ids
    except FileNotFoundError:
        logging.error(f"â�Œ File not found: {challenge_file}")
        raise
    except json.JSONDecodeError:
        logging.error(f"â�Œ Invalid JSON format in file: {challenge_file}")
        raise

def get_device_info() -> tuple[int, list[int]]:
    """Return number of GPUs and list of available memory in bytes for each GPU."""
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        logging.warning("âš ï¸� No GPU found. Falling back to CPU execution.")
        return 0, []
    gpu_memories = [torch.cuda.mem_get_info(i)[0] for i in range(n_gpus)]
    logging.info(f"ğŸ§  Detected {n_gpus} GPU(s) with memory: {[round(m / 1e9, 2) for m in gpu_memories]} GB")
    return n_gpus, gpu_memories

def get_end_time(runtime_limit_seconds: int = MAX_RUNTIME_SECONDS, buffer_seconds: int = SAFETY_BUFFER) -> float:
    """Calculate end time (UNIX timestamp) for solving."""
    end_time = time.time() + runtime_limit_seconds - buffer_seconds
    logging.info(f"â�° End time set to: {time.ctime(end_time)}")
    return end_time


# Part 3: Parallel Task Runner

import multiprocessing
import time
from multiprocessing import Manager, Process

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def parallelize_tasks(
    task_names: list[str],
    split: str,
    end_time: float,
    gpu_quotas: list[int],
    task_usages: list[int],
    n_iterations: int,
    verbose: bool = True
) -> tuple[dict, dict, float]:
    """
    Spawns processes for each task, assigns to GPUs based on available memory quotas.
    Returns memory used and final solutions.
    """
    logging.info("ğŸš€ Launching parallel solver...")
    n_tasks = len(task_names)
    n_gpus = len(gpu_quotas)
    n_cpus = multiprocessing.cpu_count()
    tasks_started = [False] * n_tasks
    tasks_finished = [False] * n_tasks
    processes = [None] * n_tasks
    process_gpu_ids = [None] * n_tasks

    with Manager() as manager:
        memory_dict = manager.dict()
        solutions_dict = manager.dict()
        error_queue = manager.Queue()

        while not all(tasks_finished):
            # Check for errors in child processes
            if not error_queue.empty():
                error_msg = error_queue.get()
                logging.error(f"â�Œ Error occurred in child process: {error_msg}")
                raise RuntimeError("Child process encountered an error.")

            # Check if any running tasks have finished
            for i in range(n_tasks):
                if tasks_started[i] and not tasks_finished[i]:
                    processes[i].join(timeout=0)
                    if not processes[i].is_alive():
                        tasks_finished[i] = True
                        gpu_quotas[process_gpu_ids[i]] += task_usages[i]
                        torch.cuda.empty_cache()  # Free unused GPU memory
                        if verbose:
                            logging.info(f"âœ… Task {task_names[i]} finished on GPU {process_gpu_ids[i]}.")

            # Assign new tasks to GPUs if quotas allow
            for gpu_id in range(n_gpus):
                for i in range(n_tasks):
                    if tasks_started[i]:
                        continue
                    enough_quota = gpu_quotas[gpu_id] >= task_usages[i]
                    active_tasks = sum(tasks_started) - sum(tasks_finished)
                    if enough_quota and active_tasks < n_cpus:
                        gpu_quotas[gpu_id] -= task_usages[i]
                        args = (
                            task_names[i], split, end_time, n_iterations,
                            gpu_id, memory_dict, solutions_dict, error_queue
                        )
                        proc = Process(target=solve_task.solve_task, args=args)
                        proc.start()
                        processes[i] = proc
                        process_gpu_ids[i] = gpu_id
                        tasks_started[i] = True
                        if verbose:
                            logging.info(f"â�³ Task {task_names[i]} started on GPU {gpu_id} with {task_usages[i]} quota.")

            # Avoid busy waiting
            time.sleep(1)

        # Gather final results
        memory_result = dict(memory_dict)
        solution_result = dict(solutions_dict)
        time_taken = time.time() - start_time

    logging.info(f"ğŸ�� All {n_tasks} tasks completed in {round(time_taken / 60, 2)} minutes.")
    return memory_result, solution_result, time_taken


# Part 4: Profiling Task Memory Usage

import torch
from typing import List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def get_memory_usage_per_task(task_names: List[str], split: str, end_time: float) -> Tuple[List[str], List[int], List[int]]:
    """
    Estimates GPU memory used by each task using 5 iterations.
    Returns sorted task names, their estimated memory usage, and safe GPU quotas.
    """
    logging.info("ğŸ“Š Profiling memory usage of each task...")
    n_gpus = torch.cuda.device_count()

    # Get available GPU memory in bytes
    raw_gpu_memory_quotas = [torch.cuda.mem_get_info(i)[0] for i in range(n_gpus)]

    # Rough per-task quota estimate: 4GB per task
    gpu_task_quotas = [int(mem // (4 * 1024 ** 3)) for mem in raw_gpu_memory_quotas]

    dummy_usages = [1 for _ in task_names]  # Start with 1 quota per task

    try:
        # Run each task for 5 iterations to check memory usage
        memory_dict, _, _ = parallelize_tasks(
            task_names=task_names,
            split=split,
            end_time=end_time,
            gpu_quotas=gpu_task_quotas,
            task_usages=dummy_usages,
            n_iterations=5,  # Increased iterations for better profiling
            verbose=False
        )
    except RuntimeError as e:
        logging.error(f"â�Œ Error during memory profiling: {e}")
        raise

    # Sort by memory usage descending
    sorted_tasks = sorted(memory_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_names, sorted_usages = zip(*sorted_tasks)

    # Dynamically adjust quotas based on profiling results
    gpu_task_quotas = [int(mem // max(sorted_usages)) for mem in raw_gpu_memory_quotas]

    # Leave 6GB headroom on each GPU for safety
    safe_gpu_quotas = [q - 6 * 1024**3 for q in raw_gpu_memory_quotas]

    # Free unused GPU memory
    torch.cuda.empty_cache()
    logging.info("ğŸ§¹ GPU memory cache cleared after profiling.")

    logging.info("âœ… Memory profiling complete.")
    return list(sorted_names), list(sorted_usages), safe_gpu_quotas

    print("âœ… Memory profiling complete.")
    return list(sorted_names), list(sorted_usages), safe_gpu_quotas



# Part 5: Profiling Time per Step

import torch
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def profile_time_per_step(
    task_names: List[str],
    task_memory_usages: List[int],
    gpu_quotas: List[int],
    split: str,
    end_time: float,
    test_steps: int = 50  # Increased iterations for better profiling
) -> float:
    """
    Runs a small number of steps for all tasks to determine average time per step.
    Returns estimated seconds per step.
    """
    logging.info(f"â�±ï¸� Profiling time usage with {test_steps} steps per task...")
    try:
        # Perform timed run
        _, _, time_taken = parallelize_tasks(
            task_names=task_names,
            split=split,
            end_time=end_time,
            gpu_quotas=gpu_quotas,
            task_usages=task_memory_usages,
            n_iterations=test_steps,
            verbose=False
        )
    except RuntimeError as e:
        logging.error(f"â�Œ Error during time profiling: {e}")
        raise

    # Average time per step across all tasks
    time_per_step = time_taken / test_steps

    # Free unused GPU memory
    torch.cuda.empty_cache()
    logging.info("ğŸ§¹ GPU memory cache cleared after profiling.")

    logging.info(f"âœ… Time profiling complete: {time_per_step:.2f} seconds/step.")
    return time_per_step



#Step 6: Final execution and submission
import json
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def run_final_solve(
    task_names: List[str],
    task_memory_usages: List[int],
    gpu_quotas: List[int],
    split: str,
    end_time: float,
    time_per_step: float
):
    """
    Runs the final multi-GPU solve, creates submission.json file.
    """
    logging.info("ğŸš€ Running final solve under memory and time constraints...")
    # Calculate how many iterations we can afford before time runs out
    time_left = end_time - time.time()
    if time_left <= 0:
        logging.warning("âš ï¸� No time left for final solve. Skipping execution.")
        return
    n_final_steps = int(time_left // time_per_step)
    logging.info(f"ğŸ•’ Time remaining: {int(time_left)} seconds")
    logging.info(f"ğŸ“ˆ Running {n_final_steps} steps per task...")

    try:
        _, solutions_dict, time_taken = parallelize_tasks(
            task_names=task_names,
            split=split,
            end_time=end_time,
            gpu_quotas=gpu_quotas,
            task_usages=task_memory_usages,
            n_iterations=n_final_steps,
            verbose=True
        )
    except RuntimeError as e:
        logging.error(f"â�Œ Error during final solve: {e}")
        raise

    # Free unused GPU memory
    torch.cuda.empty_cache()
    logging.info("ğŸ§¹ GPU memory cache cleared after final solve.")

    # Save the final solutions to a JSON file incrementally
    submission_path = 'submission.json'
    with open(submission_path, 'a') as f:  # Append mode
        for task_name, solution in solutions_dict.items():
            json.dump({task_name: solution}, f, indent=4)
            f.write('\n')

    logging.info(f"""
âœ… All tasks solved and submission written to '{submission_path}'
""")
    logging.info(f"ğŸ§© {len(solutions_dict)} tasks solved")
    logging.info(f"â�±ï¸� {time_taken:.2f} seconds total")

