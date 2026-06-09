import random
import numpy as np
import torch
import os

# Global seed value for reproducibility
GLOBAL_SEED = 42

# Detect if we're in fake (debug) mode or real competition mode
fake_mode = not os.getenv('KAGGLE_IS_COMPETITION_RERUN')

def set_all_seeds(seed=GLOBAL_SEED):
    """
    Set all random seeds to ensure reproducibility across Python, NumPy, and PyTorch.
    Also configures CUDA to be deterministic for consistent results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# Optionally uncomment to enforce deterministic behavior
# set_all_seeds()


import sys
import time
import json
import importlib
import multiprocessing
from multiprocessing import Pool

import numpy as np
import torch

# Add external module path (from Kaggle input directory)
sys.path.append('/kaggle/input/ab-compress-arg')

# Import preprocessing dynamically to avoid name collisions
module_path = "/kaggle/input/ab-compress-arg/preprocessing.py"
module_name = "preprocessing"
spec = importlib.util.spec_from_file_location(module_name, module_path)
preprocessing = importlib.util.module_from_spec(spec)
sys.modules[module_name] = preprocessing
spec.loader.exec_module(preprocessing)

# Load project modules
import train
import arc_compressor
import initializers
import multitensor_systems
import layers
import solution_selection
import visualization
import solve_task

# Multiprocessing and Torch configuration
multiprocessing.set_start_method('spawn', force=True)
torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


if __name__ == '__main__':
    # Timer to control the execution window
    start_time = time.time()
    end_time = start_time + 12*3600 - 1200  # 12 hours minus buffer

    n_cpus = multiprocessing.cpu_count()
    n_gpus = torch.cuda.device_count()

    # Load task metadata
    split = "evaluation" if fake_mode else "test"
    with open(f'../input/arc-prize-2025/arc-agi_{split}_challenges.json', 'r') as f:
        problems = json.load(f)
    
    task_names = list(problems.keys())
    del problems  # Free memory
    n_tasks = len(task_names)


def parallelize_runs(gpu_quotas, task_usages, n_iterations, verbose=False):
    """
    Distribute tasks across available GPUs using multiprocessing with memory awareness.
    """
    gpu_quotas = gpu_quotas[:]
    t = time.time()

    tasks_started = [False] * n_tasks
    tasks_finished = [False] * n_tasks
    processes = [None] * n_tasks
    process_gpu_ids = [None] * n_tasks

    with multiprocessing.Manager() as manager:
        memory_dict = manager.dict()
        solutions_dict = manager.dict()
        error_queue = manager.Queue()

        while not all(tasks_finished):
            if not error_queue.empty():
                raise ValueError(error_queue.get())

            # Check for finished tasks and update GPU memory quotas
            for i in range(n_tasks):
                if tasks_started[i] and not tasks_finished[i]:
                    processes[i].join(timeout=0)
                    if not processes[i].is_alive():
                        tasks_finished[i] = True
                        gpu_quotas[process_gpu_ids[i]] += task_usages[i]
                        if verbose:
                            print(task_names[i], 'finished on GPU', process_gpu_ids[i])

            # Launch new tasks if resources are available
            for gpu_id in range(n_gpus):
                for i in range(n_tasks):
                    enough_quota = gpu_quotas[gpu_id] > task_usages[i]
                    enough_cpus = sum(map(int, tasks_started)) - sum(map(int, tasks_finished)) < n_cpus
                    if not tasks_started[i] and enough_quota and enough_cpus:
                        gpu_quotas[gpu_id] -= task_usages[i]
                        args = (task_names[i], split, end_time, n_iterations, gpu_id, memory_dict, solutions_dict, error_queue)
                        p = multiprocessing.Process(target=solve_task.solve_task, args=args)
                        p.start()
                        processes[i] = p
                        tasks_started[i] = True
                        process_gpu_ids[i] = gpu_id
                        if verbose:
                            print(task_names[i], 'started on GPU', process_gpu_ids[i])

            time.sleep(1)

        # Final result dictionaries
        memory_dict = dict(memory_dict)
        solutions_dict = dict(solutions_dict)

    time_taken = time.time() - t
    if verbose:
        print('All jobs finished in', time_taken, 'seconds.')

    return memory_dict, solutions_dict, time_taken


if __name__ == '__main__':
    # Stage 1: Rough estimation of task memory usage
    gpu_memory_quotas = [torch.cuda.mem_get_info(i)[0] for i in range(n_gpus)]
    gpu_task_quotas = [int(qty // (4 * 1024**3)) for qty in gpu_memory_quotas]  # Assume each task needs ~4GB
    task_usages = [1 for _ in range(n_tasks)]

    memory_dict, _, _ = parallelize_runs(gpu_task_quotas, task_usages, n_iterations=2, verbose=False)

    # Sort tasks by estimated memory usage (descending)
    tasks = sorted(memory_dict.items(), key=lambda x: x[1], reverse=True)
    task_names, task_memory_usages = zip(*tasks)

    # Stage 2: Try test steps with more accurate usage & reduced memory
    test_steps = 5 if fake_mode else 20
    safe_gpu_memory_quotas = [q - 6 * 1024**3 for q in gpu_memory_quotas]  # Reserve 6GB for safety
    _, _, time_taken = parallelize_runs(safe_gpu_memory_quotas, task_memory_usages, test_steps, verbose=False)

    # Stage 3: Use remaining time to do more steps
    time_per_step = time_taken / test_steps
    time_left = end_time - time.time()
    n_steps = 5 if fake_mode else int(time_left // time_per_step)

    _, solutions_dict, time_taken = parallelize_runs(safe_gpu_memory_quotas, task_memory_usages, n_steps, verbose=True)

    # Save final submission
    with open('submission.json', 'w') as f:
        json.dump(solutions_dict, f, indent=4)

    print(n_tasks, 'tasks solved.')
    print(n_steps, 'steps taken.')
    print(time_taken, 'seconds taken.')


import json

# Load ground truth from training and evaluation solutions
with open('/kaggle/input/arc-prize-2025/arc-agi_training_solutions.json', 'r') as f:
    training_solution = json.load(f)

with open('/kaggle/input/arc-prize-2025/arc-agi_evaluation_solutions.json', 'r') as f:
    evaluation_solution = json.load(f)

# Load the ARC challenge JSON depending on the mode
arc_challenge_file = '/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json' if fake_mode \
                     else '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json'

with open(arc_challenge_file, 'r') as f:
    arc_data = json.load(f)

# Combine solutions (for visualization / scoring)
conpiled_solution = {}
for case_id in arc_data:
    if case_id in training_solution:
        conpiled_solution[case_id] = training_solution[case_id][0]
    elif case_id in evaluation_solution:
        conpiled_solution[case_id] = evaluation_solution[case_id][0]

