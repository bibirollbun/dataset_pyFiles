import random
import numpy as np
import torch
import os
import sys
import time
import json
import importlib
import multiprocessing
from multiprocessing import Pool
import gc
import traceback # Import traceback

# === FIX 1: Help PyTorch manage memory fragmentation ===
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# === CONTROL PANEL ===
# Set these to True/False to run only the stages you need.
#
# We've already run Stage 1 and 2, so we can set them to False!
# We just want to re-run the main solution with our new hyperparameters.

RUN_STAGE_1_MEASUREMENT = False
RUN_STAGE_2_TIMING = False
RUN_STAGE_3_SOLUTION = True
RUN_STAGE_4_VISUALIZE = True

# === FILE PATHS ===
# Make sure these point to your cached stage files
CACHE_STAGE_1_MEMORY = '/kaggle/input/arc-stage-wise-data/stage_1_memory.json'
CACHE_STAGE_2_TIMING = '/kaggle/input/arc-stage-wise-data/stage_2_timing.json'


GLOBAL_SEED = 42
fake_mode = not os.getenv('KAGGLE_IS_COMPETITION_RERUN')

def set_all_seeds(seed=GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

#set_all_seeds()

# Imports
# === This path must point to YOUR dataset ===
# Make sure this matches your dataset's path on Kaggle
DATASET_PATH = '/kaggle/input/my-arc-code' 
sys.path.append(DATASET_PATH)

# === FIX: Use importlib to avoid name collision ===
# This is critical to ensure we load *our* modules, not system ones
def load_module_from_path(module_name, file_name):
    """Helper function to load a module from a specific file."""
    module_path = os.path.join(DATASET_PATH, file_name)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ImportError(f"Could not load spec for module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

try:
    preprocessing = load_module_from_path("preprocessing", "preprocessing.py")
    train = load_module_from_path("train", "train.py")
    arc_compressor = load_module_from_path("arc_compressor", "arc_compressor.py")
    initializers = load_module_from_path("initializers", "initializers.py")
    multitensor_systems = load_module_from_path("multitensor_systems", "multitensor_systems.py")
    layers = load_module_from_path("layers", "layers.py")
    solution_selection = load_module_from_path("solution_selection", "solution_selection.py")
    visualization = load_module_from_path("visualization", "visualization.py")
    solve_task = load_module_from_path("solve_task", "solve_task.py")
except Exception as e:
    print(f"FATAL ERROR during main notebook import: {e}")
    print(traceback.format_exc())


# Getting all the task names, setting defaults and constants
multiprocessing.set_start_method('spawn', force=True)
torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Function that can spawn processes and schedule them on GPUs
def parallelize_runs(gpu_quotas, task_usages, n_iterations, main_end_time, verbose=False):
    # Ensure task_usages is a list, not a zip object
    task_usages = list(task_usages)
    
    # Make local copies
    gpu_quotas = gpu_quotas[:]
    
    t = time.time()
    tasks_started = [False for i in range(n_tasks)]
    tasks_finished = [False for i in range(n_tasks)]
    processes = [None for i in range(n_tasks)]
    process_gpu_ids = [None for i in range(n_tasks)]
    
    # === FIX: Use the main_end_time passed from __main__ ===
    # This ensures child processes respect the *real* 11-hour limit
    
    with multiprocessing.Manager() as manager:
        memory_dict = manager.dict()
        # === FIX: Use a global solutions_dict ===
        # This is declared outside the function so it persists
        # in the `finally` block even if this function fails.
        
        error_queue = manager.Queue()
        
        while not all(tasks_finished):
            # === IMPROVEMENT: Robust error checking ===
            while not error_queue.empty():
                error_msg = error_queue.get()
                print("="*50)
                print("FATAL ERROR from child process:")
                print(error_msg)
                print("="*50)
                # In a real run, you might want to raise this
                # raise ValueError(error_msg)

            for i in range(n_tasks):
                if tasks_started[i] and not tasks_finished[i]:
                    processes[i].join(timeout=0)
                    if not processes[i].is_alive():
                        tasks_finished[i] = True
                        # Only free quota if the task *had* a GPU
                        if process_gpu_ids[i] is not None and process_gpu_ids[i] < len(gpu_quotas):
                            gpu_quotas[process_gpu_ids[i]] += task_usages[i]
                        if verbose:
                            print(task_names[i], 'finished on gpu', process_gpu_ids[i],
                                  'New quota is', (gpu_quotas[process_gpu_ids[i]] if process_gpu_ids[i] is not None and process_gpu_ids[i] < len(gpu_quotas) else 'N/A'))
            
            for gpu_id in range(n_gpus):
                for i in range(n_tasks):
                    # Check if task i is already running
                    if tasks_started[i]:
                        continue
                    
                    # Ensure task_usages[i] is valid
                    if i >= len(task_usages):
                        print(f"Warning: Task index {i} out of bounds for task_usages. Skipping.")
                        tasks_started[i] = True # Mark as "started" to avoid re-checking
                        tasks_finished[i] = True # Mark as "finished"
                        continue
                        
                    enough_quota = gpu_quotas[gpu_id] >= task_usages[i]
                    enough_cpus = sum(map(int, tasks_started)) - sum(map(int, tasks_finished)) < n_cpus
                    
                    if enough_quota and enough_cpus:
                        gpu_quotas[gpu_id] -= task_usages[i]
                        
                        # === FIX: Pass DATASET_PATH and json_path to child process ===
                        # This must be defined *outside* the __main__ block to be seen here
                        args = (task_names[i], split, main_end_time, n_iterations, gpu_id, memory_dict, solutions_dict_manager, error_queue, DATASET_PATH, json_path)
                        
                        # We use the new solve_task.py
                        p = multiprocessing.Process(target=solve_task.solve_task, args=args)
                        p.start()
                        processes[i] = p
                        tasks_started[i] = True
                        process_gpu_ids[i] = gpu_id
                        
                        if verbose:
                            print(task_names[i], 'started on gpu', process_gpu_ids[i],
                                  'New quota is', gpu_quotas[process_gpu_ids[i]])
                        
                        # Break inner loop to give other GPUs a chance
                        break 
            
            time.sleep(1)
            
            # === Failsafe: Check if time is up ===
            if time.time() > main_end_time:
                print("--- WARNING: Main time limit reached in parallelize_runs. Terminating processes. ---")
                for p in processes:
                    if p and p.is_alive():
                        p.terminate()
                break # Exit the while loop
        
        # Final error check
        while not error_queue.empty():
            error_msg = error_queue.get()
            print("="*50)
            print("FATAL ERROR from child process (final check):")
            print(error_msg)
            print("="*50)

        memory_dict = dict(memory_dict)
        
        # Don't return solutions_dict, it's now managed globally
        
    time_taken = time.time() - t
    if verbose:
        print('All jobs finished in', time_taken, 'seconds.')
    # Return memory_dict, but solutions are now in the global solutions_dict_manager
    return memory_dict, time_taken


# === MAIN EXECUTION BLOCK ===
if __name__ == '__main__':
    
    # --- Global Setup (Needed for all stages) ---
    start_time = time.time()
    
    # === FIX: 11-HOUR RUNTIME ===
    # Set the absolute end time to 11 hours from now, giving a 1-hour buffer
    # for shutdown and saving the JSON.
    end_time = start_time + 11 * 3600 # 11 hours
    
    # This end_time is for short measurement tasks ONLY
    measure_end_time = start_time + 3600 # 1 hour time limit for measurement task
    
    n_cpus = multiprocessing.cpu_count()
    n_gpus = torch.cuda.device_count()

    split = "evaluation" if fake_mode else "test"
    
    json_path = f'../input/arc-prize-2025/arc-agi_{split}_challenges.json'
    if not os.path.exists(json_path):
        print(f"Warning: Path not found. Trying /kaggle/input/arc-prize-2025/...")
        json_path = f'/kaggle/input/arc-prize-2025/arc-agi_{split}_challenges.json'
        if not os.path.exists(json_path):
             print(f"Warning: Path not found. Trying dataset path {DATASET_PATH}...")
             json_path = f'{DATASET_PATH}/arc-agi_{split}_challenges.json'

    print(f"Using challenge file: {json_path}")
        
    with open(json_path, 'r') as f:
        problems = json.load(f)
    task_names = list(problems.keys())
    del problems
    n_tasks = len(task_names)
    print(f"Found {n_tasks} tasks for {split} split.")
    
    # --- Define variables that will be loaded/saved ---
    task_memory_usages = []
    time_per_step = 1.0 # default
    
    # === Failsafe: Setup a global manager for solutions ===
    # This must be done *outside* the try block
    manager = multiprocessing.Manager()
    solutions_dict_manager = manager.dict()


    try:
        # --- STAGE 1: Measuring memory usage (SERIAL) ---
        if RUN_STAGE_1_MEASUREMENT:
            print("\n--- STAGE 1: Measuring memory usage (SERIAL) ---")
            
            measured_tasks = {}
            OOM_DEFAULT_MEMORY = 24 * 1024**3 # 24 GB default for failed tasks
            measurement_iterations = 20 # Run for 20 iterations to capture peak memory

            for i in range(n_tasks):
                task_name = task_names[i]
                print(f"Measuring task {i+1}/{n_tasks}: {task_name}...")
                
                # We need a new manager for each serial task
                with multiprocessing.Manager() as task_manager:
                    memory_dict_serial = task_manager.dict()
                    solutions_dict_serial = task_manager.dict() # Dummy dict
                    error_queue_serial = task_manager.Queue()
                    
                    args = (task_name, split, measure_end_time, measurement_iterations, 0, memory_dict_serial, solutions_dict_serial, error_queue_serial, DATASET_PATH, json_path)
                    
                    p = multiprocessing.Process(target=solve_task.solve_task, args=args)
                    p.start()
                    p.join() # Wait for this task to finish
                    
                    crashed = False
                    while not error_queue_serial.empty():
                        error_msg = error_queue_serial.get()
                        print("="*50)
                        print(f"ERROR during memory measurement for {task_name}:")
                        print(error_msg)
                        print("="*50)
                        crashed = True
                    
                    if crashed:
                        print(f"Warning: Task {task_name} failed memory measurement. Assigning high default memory.")
                        measured_tasks[task_name] = OOM_DEFAULT_MEMORY
                    elif task_name not in memory_dict_serial:
                         print(f"Warning: Task {task_name} finished but did not report memory. Assigning high default memory.")
                         measured_tasks[task_name] = OOM_DEFAULT_MEMORY
                    else:
                        measured_tasks[task_name] = memory_dict_serial[task_name]
                        print(f"Task {task_name} measured: {measured_tasks[task_name] / 1024**3:.2f} GB")

                gc.collect()
                torch.cuda.empty_cache()

            tasks = sorted(measured_tasks.items(), key=lambda x: x[1], reverse=True)
            task_names, task_memory_usages = zip(*tasks)
            
            print("\nTask memory usage (sorted):")
            for i in range(len(task_names)):
                print(f"{task_names[i]}: {task_memory_usages[i] / 1024**3:.2f} GB")
            
            # --- CACHE STAGE 1 OUTPUT ---
            # Note: This saves to the /kaggle/working/ directory
            print(f"Caching Stage 1 results to {CACHE_STAGE_1_MEMORY}...")
            # Convert tuples to lists for JSON serialization
            cache_data = {'task_names': list(task_names), 'task_memory_usages': list(task_memory_usages)}
            # We must create the file in /kaggle/working/
            os.makedirs(os.path.dirname(CACHE_STAGE_1_MEMORY.replace('/kaggle/input/', '/kaggle/working/')), exist_ok=True)
            with open(CACHE_STAGE_1_MEMORY.replace('/kaggle/input/', '/kaggle/working/'), 'w') as f:
                json.dump(cache_data, f)
                
            print("--- STAGE 1: Finished ---")
            
        else:
            print(f"\n--- STAGE 1: SKIPPED. Loading from {CACHE_STAGE_1_MEMORY} ---")
            try:
                with open(CACHE_STAGE_1_MEMORY, 'r') as f:
                    cache_data = json.load(f)
                task_names = cache_data['task_names']
                task_memory_usages = cache_data['task_memory_usages']
                n_tasks = len(task_names) # Redefine n_tasks
                print(f"Successfully loaded {n_tasks} tasks from cache.")
            except Exception as e:
                print(f"FATAL: Could not load cache {CACHE_STAGE_1_MEMORY}. Set RUN_STAGE_1_MEASUREMENT = True and re-run.")
                raise e

        # --- STAGE 2: Measuring time per step ---
        if RUN_STAGE_2_TIMING:
            print("\n--- STAGE 2: Measuring time per step ---")
            
            test_steps = 20 # Use 20 for a more stable average
            
            gpu_memory_quotas = [torch.cuda.mem_get_info(i)[0] for i in range(n_gpus)]
            # === FIX 3: Increase GPU safety margin ===
            safe_gpu_memory_quotas = [memory_quota - 6 * 1024**3 for memory_quota in gpu_memory_quotas]
            
            # Use the 'measure_end_time' for this short run
            _, time_taken = parallelize_runs(safe_gpu_memory_quotas, task_memory_usages, test_steps, measure_end_time, verbose=False)
            time_per_step = (time_taken / test_steps) if test_steps > 0 else 1.0
            
            # --- CACHE STAGE 2 OUTPUT ---
            # Note: This saves to the /kaggle/working/ directory
            print(f"Caching Stage 2 results to {CACHE_STAGE_2_TIMING}...")
            os.makedirs(os.path.dirname(CACHE_STAGE_2_TIMING.replace('/kaggle/input/', '/kaggle/working/')), exist_ok=True)
            with open(CACHE_STAGE_2_TIMING.replace('/kaggle/input/', '/kaggle/working/'), 'w') as f:
                json.dump({'time_per_step': time_per_step}, f)
                
            print(f"--- STAGE 2: Finished (Time taken: {time_taken:.2f}s) ---")
            
        else:
            print(f"\n--- STAGE 2: SKIPPED. Loading from {CACHE_STAGE_2_TIMING} ---")
            try:
                with open(CACHE_STAGE_2_TIMING, 'r') as f:
                    cache_data = json.load(f)
                time_per_step = cache_data['time_per_step']
                print(f"Successfully loaded time_per_step: {time_per_step:.2f}s")
            except Exception as e:
                print(f"FATAL: Could not load cache {CACHE_STAGE_2_TIMING}. Set RUN_STAGE_2_TIMING = True and re-run.")
                raise e

        # --- STAGE 3: Running full solution ---
        if RUN_STAGE_3_SOLUTION:
            print("\n--- STAGE 3: Running full solution ---")
            
            time_left = end_time - time.time()
            
            # === FIX: Use a reasonable number of steps for fake_mode (validation) ===
            # This is our best guess from previous runs
            n_steps_fake_mode = 500
            
            n_steps = n_steps_fake_mode if fake_mode else int(time_left // time_per_step)
            
            if n_steps < 500 and not fake_mode:
                print(f"Warning: Calculated steps ({n_steps}) is very low. Setting to 500.")
                n_steps = 500
            
            print(f"Time per step: {time_per_step:.2f}s. Time left: {time_left:.2f}s. Calculated steps: {n_steps}")
            
            gpu_memory_quotas = [torch.cuda.mem_get_info(i)[0] for i in range(n_gpus)]
            safe_gpu_memory_quotas = [memory_quota - 6 * 1024**3 for memory_quota in gpu_memory_quotas]
            
            # Use the *real* end_time for the main run
            _, time_taken = parallelize_runs(safe_gpu_memory_quotas, task_memory_usages, n_steps, end_time, verbose=True)
            
            print(n_tasks, 'tasks solved.')
            print(n_steps, 'steps taken.')
            print(time_taken, 'seconds taken.')
            print("--- STAGE 3: Finished ---")
            
        else:
            print("\n--- STAGE 3: SKIPPED ---")

    except Exception as e:
        print("\n--- FATAL ERROR IN MAIN EXECUTION ---")
        print(e)
        print(traceback.format_exc())
        
    finally:
        # === FAILSAFE: ALWAYS write submission.json ===
        # This block will run even if Stage 3 is skipped, crashes, or times out.
        print("\n--- FINALIZING: Saving submission.json ---")
        try:
            # Convert the managed dict to a regular dict for saving
            final_solutions = dict(solutions_dict_manager)
            with open('submission.json', 'w') as f:
                json.dump(final_solutions, f, indent=4)
            print(f"Successfully saved submission.json with {len(final_solutions)} task entries.")
        except Exception as e:
            print(f"FATAL: Could not write submission.json: {e}")
            # As a last resort, write an empty file to prevent "Scoring Failed"
            with open('submission.json', 'w') as f:
                json.dump({}, f)
            print("Wrote an empty submission.json to prevent scoring error.")
        
        # Shut down the manager
        manager.shutdown()


    # --- STAGE 4: Visualizing results ---
    if RUN_STAGE_4_VISUALIZE:
        print("\n--- STAGE 4: Visualizing results ---")
        
        # All imports and definitions must be *inside* this block
        # to run independently.
        import json
        import matplotlib.pyplot as plt
        from matplotlib import colors
        import os
        import numpy as np

        training_solution_path = '/kaggle/input/arc-prize-2025/arc-agi_training_solutions.json'
        evaluation_solution_path = '/kaggle/input/arc-prize-2025/arc-agi_evaluation_solutions.json'
        
        if not os.path.exists(training_solution_path):
            training_solution_path = f'{DATASET_PATH}/arc-agi_training_solutions.json'
        if not os.path.exists(evaluation_solution_path):
            evaluation_solution_path = f'{DATASET_PATH}/arc-agi_evaluation_solutions.json'

        try:
            with open(training_solution_path, 'r') as f:
                training_solution = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Training solutions not found at {training_solution_path}")
            training_solution = {}

        try:
            with open(evaluation_solution_path, 'r') as f:
                evaluation_solution = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Evaluation solutions not found at {evaluation_solution_path}")
            evaluation_solution = {}
            
        conbiled_solution = {} # Renamed to avoid name clash

        if fake_mode:
            arc_challenge_file = json_path # Use path from above
        else:
            arc_challenge_file = json_path # Use path from above
            
        with open(arc_challenge_file, 'r') as f:
            arc_data = json.load(f)
            
        for case_id in arc_data:
            if case_id in training_solution:
                conbiled_solution[case_id] = training_solution[case_id] # Load list
            elif case_id in evaluation_solution:
                conbiled_solution[case_id] = evaluation_solution[case_id] # Load list
                
    
        def visualize_arc_results():
            """Visualize ARC problem solutions from submission.json"""
            
            print("\n" + "="*80)
            print("VISUALIZING ARC SOLUTION RESULTS")
            print("="*80)
            
            submission_path = 'submission.json'
            if not os.path.exists(submission_path):
                print(f"Submission file not found at {submission_path}")
                return
            
            print(f"Found submission file: {submission_path}")
            
            try:
                with open(submission_path, 'r') as f:
                    submission_data = json.load(f)
            except json.JSONDecodeError:
                print(f"FATAL: submission.json is corrupted or empty.")
                return

            
            print(f"Loaded submission with {len(submission_data)} tasks")
            
            cmap = colors.ListedColormap(
                ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
                 '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
            norm = colors.Normalize(vmin=0, vmax=9)
            
            def is_non_trivial_prediction(pred_array):
                try:
                    return np.any(np.array(pred_array, dtype=int) > 0)
                except:
                    return False
            
            def visualize_submission_result(task_id, task_data, submission_output, test_idx):
                try:
                    pred_1 = np.array(submission_output['attempt_1'], dtype=int)
                    pred_2 = np.array(submission_output['attempt_2'], dtype=int)
                except:
                    print(f"  Skipping visualization for Task {task_id} - Test #{test_idx+1} (malformed prediction)")
                    return False
                
                if not is_non_trivial_prediction(pred_1) and not is_non_trivial_prediction(pred_2):
                    print(f"  Skipping visualization for Task {task_id} - Test #{test_idx+1} (all predictions are zeros)")
                    return False
                
                fig = plt.figure(figsize=(15, 8))
                grid_spec = plt.GridSpec(2, 3, width_ratios=[1, 1, 1])
                
                if task_data['train']:
                    ax1 = fig.add_subplot(grid_spec[0, 0])
                    ax1.imshow(task_data['train'][0]['input'], cmap=cmap, norm=norm)
                    ax1.grid(True, which='both', color='lightgrey', linewidth=0.5)
                    ax1.set_title("Training Input")
                    ax1.set_xticks([]); ax1.set_yticks([])
                    
                    ax2 = fig.add_subplot(grid_spec[1, 0])
                    ax2.imshow(task_data['train'][0]['output'], cmap=cmap, norm=norm)
                    ax2.grid(True, which='both', color='lightgrey', linewidth=0.5)
                    ax2.set_title("Training Output")
                    ax2.set_xticks([]); ax2.set_yticks([])
                
                if test_idx < len(task_data['test']):
                    ax3 = fig.add_subplot(grid_spec[0, 1])
                    ax3.imshow(task_data['test'][test_idx]['input'], cmap=cmap, norm=norm)
                    ax3.grid(True, which='both', color='lightgrey', linewidth=0.5)
                    ax3.set_title(f"Test Input (Test #{test_idx+1})")
                    ax3.set_xticks([]); ax3.set_yticks([])
                
                ax5 = fig.add_subplot(grid_spec[0, 2])
                try:
                    ax5.imshow(pred_1, cmap=cmap, norm=norm)
                except:
                    ax5.text(0.5, 0.5, "Invalid Shape", ha='center', va='center')
                ax5.grid(True, which='both', color='lightgrey', linewidth=0.5)
                ax5.set_title("Model Prediction (Attempt 1)")
                ax5.set_xticks([]); ax5.set_yticks([])
                
                ax6 = fig.add_subplot(grid_spec[1, 2])
                try:
                    ax6.imshow(pred_2, cmap=cmap, norm=norm)
                except:
                    ax6.text(0.5, 0.5, "Invalid Shape", ha='center', va='center')
                ax6.grid(True, which='both', color='lightgrey', linewidth=0.5)
                ax6.set_title("Model Prediction (Attempt 2)")
                ax6.set_xticks([]); ax6.set_yticks([])
                
                ground_truth = None
                if task_id in conbiled_solution and len(conbiled_solution[task_id]) > test_idx:
                    ground_truth = conbiled_solution[task_id][test_idx]
                        
                if ground_truth is not None:
                    ground_truth_np = np.array(ground_truth, dtype=int)
                    ax4 = fig.add_subplot(grid_spec[1, 1])
                    ax4.imshow(ground_truth_np, cmap=cmap, norm=norm)
                    ax4.grid(True, which='both', color='lightgrey', linewidth=0.5)
                    ax4.set_title("Ground Truth")
                    ax4.set_xticks([]); ax4.set_yticks([])
                    
                    match_1 = False
                    match_2 = False
                    
                    if is_non_trivial_prediction(pred_1) and pred_1.shape == ground_truth_np.shape:
                        match_1 = np.array_equal(pred_1, ground_truth_np)
                    if is_non_trivial_prediction(pred_2) and pred_2.shape == ground_truth_np.shape:
                        match_2 = np.array_equal(pred_2, ground_truth_np)
                    
                    ax5.set_title(f"Prediction 1: {'✓' if match_1 else '✗'}")
                    ax6.set_title(f"Prediction 2: {'✓' if match_2 else '✗'}")
                    
                    print(f"  Results: Attempt 1: {'✓' if match_1 else '✗'}, Attempt 2: {'✓' if match_2 else '✗'}")
                    print(f"  Shape - Ground Truth: {ground_truth_np.shape}, "
                          f"Prediction 1: {pred_1.shape}, Prediction 2: {pred_2.shape}")
                    print(f"  Values - Ground Truth unique values: {np.unique(ground_truth_np)}")
                    print(f"          Prediction 1 unique values: {np.unique(pred_1) if pred_1.size > 0 else '[]'}")
                    print(f"          Prediction 2 unique values: {np.unique(pred_2) if pred_2.size > 0 else '[]'}")
                
                plt.suptitle(f"Task {task_id} - Test Example #{test_idx+1}", fontsize=16)
                plt.tight_layout()
                plt.subplots_adjust(top=0.9)
                plt.show()
                return True
            
            visualized_count = 0
            skipped_count = 0
            
            all_predictions = []
            for task_id in submission_data:
                if task_id not in arc_data:
                    continue # Skip tasks not in our current challenge set
                    
                task_data = arc_data[task_id]
                for test_idx, test_prediction in enumerate(submission_data[task_id]):
                    try:
                        pred_1 = np.array(test_prediction['attempt_1'], dtype=int)
                        pred_2 = np.array(test_prediction['attempt_2'], dtype=int)
                    except:
                        all_predictions.append((task_id, test_idx, 0, False, False)) # Malformed
                        continue
                        
                    has_non_zero_pred = is_non_trivial_prediction(pred_1) or is_non_trivial_prediction(pred_2)
                    
                    has_ground_truth = False
                    correct_count = 0
                    
                    ground_truth_np = None
                    if task_id in conbiled_solution and len(conbiled_solution[task_id]) > test_idx:
                        has_ground_truth = True
                        ground_truth_np = np.array(conbiled_solution[task_id][test_idx], dtype=int)
                        
                        if has_non_zero_pred:
                            match_1 = False
                            match_2 = False
                            if is_non_trivial_prediction(pred_1) and pred_1.shape == ground_truth_np.shape:
                                match_1 = np.array_equal(pred_1, ground_truth_np)
                            if is_non_trivial_prediction(pred_2) and pred_2.shape == ground_truth_np.shape:
                                match_2 = np.array_equal(pred_2, ground_truth_np)
                            correct_count = int(match_1) + int(match_2)
                    
                    all_predictions.append((task_id, test_idx, correct_count, has_ground_truth, has_non_zero_pred))
            
            all_predictions.sort(key=lambda x: (-int(x[3]), -x[2]))
            
            print(f"\nFound {len(all_predictions)} total predictions to visualize")
            
            max_samples = 10
            samples_to_show = all_predictions[:max_samples]
            
            print(f"Showing {len(samples_to_show)} of {len(all_predictions)} prediction samples")
            
            for task_id, test_idx, correct_count, has_ground_truth, has_non_zero_pred in samples_to_show:
                if task_id not in arc_data: continue
                task_data = arc_data[task_id]
                
                if task_id not in submission_data or test_idx >= len(submission_data[task_id]):
                    print(f"\nSkipping Task: {task_id} - Test #{test_idx+1} (Not found in submission)")
                    continue
                    
                submission_output = submission_data[task_id][test_idx]
                
                score_info = f" (Score: {correct_count}/2)" if has_ground_truth and has_non_zero_pred else " (no ground truth)" if not has_ground_truth else " (all zeros - no score)"
                print(f"\nTask: {task_id} - Test #{test_idx+1}{score_info}")
                
                if visualize_submission_result(task_id, task_data, submission_output, test_idx):
                    visualized_count += 1
                else:
                    skipped_count += 1
            
            print(f"\nVisualized {visualized_count} inference results (skipped {skipped_count} with all-zero predictions)")
            
            if fake_mode:
                total_tests = 0
                total_scored_tests = 0
                correct_attempt1 = 0
                correct_attempt2 = 0
                correct_any = 0
                zero_predictions = 0
                
                for task_id, test_idx, _, has_ground_truth, _ in all_predictions:
                    if has_ground_truth:
                        total_tests += 1
                        
                        if task_id not in conbiled_solution or len(conbiled_solution[task_id]) <= test_idx:
                            continue # Should not happen, but safeguard
                        
                        ground_truth_np = np.array(conbiled_solution[task_id][test_idx], dtype=int)
                        
                        if task_id not in submission_data or test_idx >= len(submission_data[task_id]):
                            zero_predictions += 1 # Count missing as zero
                            continue

                        try:
                            pred_1 = np.array(submission_data[task_id][test_idx]['attempt_1'], dtype=int)
                            pred_2 = np.array(submission_data[task_id][test_idx]['attempt_2'], dtype=int)
                        except:
                            zero_predictions += 1 # Count malformed as zero
                            continue
                        
                        if not is_non_trivial_prediction(pred_1) and not is_non_trivial_prediction(pred_2):
                            zero_predictions += 1
                            continue
                        
                        total_scored_tests += 1
                        
                        match_1 = False
                        match_2 = False
                        if is_non_trivial_prediction(pred_1) and pred_1.shape == ground_truth_np.shape:
                            match_1 = np.array_equal(pred_1, ground_truth_np)
                        if is_non_trivial_prediction(pred_2) and pred_2.shape == ground_truth_np.shape:
                            match_2 = np.array_equal(pred_2, ground_truth_np)

                        if match_1: correct_attempt1 += 1
                        if match_2: correct_attempt2 += 1
                        if match_1 or match_2: correct_any += 1
                
                if total_tests > 0:
                    print("\n" + "="*80)
                    print("OVERALL ACCURACY STATISTICS")
                    print("="*80)
                    print(f"Total test examples: {total_tests}")
                    print(f"Test examples with zero/malformed/missing predictions (excluded from accuracy): {zero_predictions}")
                    print(f"Test examples included in accuracy calculation: {total_scored_tests}")
                    
                    if total_scored_tests > 0:
                        print(f"Correct on attempt 1: {correct_attempt1}/{total_scored_tests} ({correct_attempt1/total_scored_tests:.2%})")
                        print(f"Correct on attempt 2: {correct_attempt2}/{total_scored_tests} ({correct_attempt2/total_scored_tests:.2%})")
                        print(f"Correct on either attempt: {correct_any}/{total_scored_tests} ({correct_any/total_scored_tests:.2%})")
                    else:
                        print("No non-zero predictions to calculate accuracy")
                        
                    print(f"Overall completion rate: {total_scored_tests/total_tests:.2%} of tests have non-zero predictions")
                    print("="*80)

        # Call after your submission.json has been created
        if fake_mode:
            visualize_arc_results()
        
        print("\n--- STAGE 4: Finished ---")
        
    print("\nNotebook finished.")



