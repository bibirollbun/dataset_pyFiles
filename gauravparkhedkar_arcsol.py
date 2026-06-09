# ==============================================================================
# SECTION 1: IMPORTS AND GLOBAL SETUP
# ==============================================================================

import random
import numpy as np
import torch
import os

GLOBAL_SEED = 42
fake_mode = not os.getenv('KAGGLE_IS_COMPETITION_RERUN')

def set_all_seeds(seed=GLOBAL_SEED):
    """设置所有可能的随机种子来确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 为了完全确定性，禁用CUDA的非确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 设置Python哈希种子
    os.environ['PYTHONHASHSEED'] = str(seed)
#set_all_seeds()

# Imports
import os
import sys
import time
import json
import importlib
import multiprocessing
from multiprocessing import Pool
import numpy as np
import torch

sys.path.append('/kaggle/input/publiccompressarc') # Using public version path
# This little block of code does "import preprocessing" but avoids a name collision
module_path = "/kaggle/input/publiccompressarc/preprocessing.py"
module_name = "preprocessing"
spec = importlib.util.spec_from_file_location(module_name, module_path)
preprocessing = importlib.util.module_from_spec(spec)
sys.modules[module_name] = preprocessing
spec.loader.exec_module(preprocessing)

import train
import arc_compressor
import initializers
import multitensor_systems
import layers
import solution_selection
import visualization
import solve_task

# Getting all the task names, setting defaults and constants
multiprocessing.set_start_method('spawn', force=True)
torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

if __name__ == '__main__':

    start_time = time.time()
    end_time = start_time + 12*3600 - 600 # 12 hours - 10 min buffer

    n_cpus = multiprocessing.cpu_count()
    n_gpus = torch.cuda.device_count()

    # Find all the puzzle names
    split = "evaluation" if fake_mode else "test"
    challenge_file_path = f'../input/arc-prize-2025/arc-agi_{split}_challenges.json'
    with open(challenge_file_path, 'r') as f:
        problems = json.load(f)
        
    task_names = list(problems.keys())
    n_tasks = len(task_names)

# ==============================================================================
# SECTION 3: PUZZLE PRE-ANALYSIS (HYBRID TRIAGE STRATEGY)
# ==============================================================================

def analyze_task_properties(task_data):
    """
    Analyzes training pairs to create a robust "solvability" score.
    Returns:
        - base_score (float): Score based on grid size and color count (like V2).
        - is_geometric (bool): True if grid size and colors are static.
        - grid_size_changes (bool): True if grid size ever changes.
        - colors_change (bool): True if new colors are introduced.
    """
    is_geometric = True
    grid_size_changes = False
    colors_change = False
    
    total_color_score = 0
    total_pixel_score = 0
    num_train_pairs = len(task_data['train'])

    if num_train_pairs == 0:
        # Fallback for tasks with no training data
        input_matrix = np.array(task_data['test'][0]['input'])
        num_unique_values = len(np.unique(input_matrix))
        total_color_score = (1 - num_unique_values / 11) * 10
        total_pixel_score = 1 - (input_matrix.shape[0] * input_matrix.shape[1] / (31*31))
    else:
        for pair in task_data['train']:
            input_grid = np.array(pair['input'])
            output_grid = np.array(pair['output'])
            
            # Check for non-geometric properties
            if input_grid.shape != output_grid.shape:
                is_geometric = False
                grid_size_changes = True

            input_colors = set(np.unique(input_grid)) - {0}
            output_colors = set(np.unique(output_grid)) - {0}
            if not output_colors.issubset(input_colors):
                is_geometric = False
                colors_change = True
            
            # Calculate base simplicity score
            num_unique_values = len(np.unique(input_grid))
            total_color_score += (1 - num_unique_values / 11) * 10
            total_pixel_score += 1 - (input_grid.shape[0] * input_grid.shape[1] / (31*31))
    
    avg_color_score = total_color_score / max(1, num_train_pairs)
    avg_pixel_score = total_pixel_score / max(1, num_train_pairs)
    
    base_score = np.sqrt(avg_color_score + avg_pixel_score + 1)
        
    return base_score, is_geometric, grid_size_changes, colors_change

if __name__ == '__main__':
    print("Analyzing task properties for Hybrid Triage...")
    
    task_properties = {}
    
    for name, data in problems.items():
        base_score, is_geo, size_change, color_change = analyze_task_properties(data)
        
        # --- This is the new Hybrid Score Logic ---
        final_score = base_score
        
        if is_geo:
            final_score *= 2.0  # Large bonus for "pure geometric" tasks
        else:
            if size_change:
                final_score *= 0.5  # Heavy penalty for size changes
            if color_change:
                final_score *= 0.7  # Medium penalty for color changes
        
        task_properties[name] = {
            'score': final_score
        }

    del problems
    print("Hybrid Triage analysis complete.")
    
# ==============================================================================
# SECTION 4: PARALLEL TASK SCHEDULER
# ==============================================================================

def parallelize_runs(gpu_quotas, task_usages, n_iteration_list, verbose=False):
    gpu_quotas = gpu_quotas[:]
    t = time.time()
    tasks_started = [False for i in range(n_tasks)]
    tasks_finished = [False for i in range(n_tasks)]
    processes = [None for i in range(n_tasks)]
    process_gpu_ids = [None for i in range(n_tasks)]
    
    with multiprocessing.Manager() as manager:
        memory_dict = manager.dict()
        solutions_dict = manager.dict()
        error_queue = manager.Queue()
        
        # --- MODIFICATION: Only run tasks with > 0 steps ---
        tasks_to_run_indices = [i for i, steps in enumerate(n_iteration_list) if steps > 0]
        num_tasks_to_run = len(tasks_to_run_indices)
        
        # Set tasks with 0 steps to 'finished' immediately
        for i in range(n_tasks):
            if i not in tasks_to_run_indices:
                tasks_finished[i] = True
        
        if num_tasks_to_run == 0:
            print("Warning: No tasks scheduled to run.")
            return dict(), dict(), 0

        # Prioritize based on the global sorted_taskid (easiest first)
        task_scheduler_order = [i for i in sorted_taskid if i in tasks_to_run_indices]
        
        print(f"Starting batch with {num_tasks_to_run} tasks.")
        
        while not all(tasks_finished): # Check *all* tasks
            if not error_queue.empty():
                raise ValueError(error_queue.get())
                
            # --- 2. Check for finished processes ---
            for i in tasks_to_run_indices: # Only check tasks that were started
                if tasks_started[i] and not tasks_finished[i]:
                    processes[i].join(timeout=0)
                    if not processes[i].is_alive():
                        tasks_finished[i] = True
                        gpu_quotas[process_gpu_ids[i]] += task_usages[i]
                        if verbose:
                            print(f"{task_names[i]} finished on gpu {process_gpu_ids[i]}. "
                                  f"New quota: {gpu_quotas[process_gpu_ids[i]]}")

            # --- 3. Schedule new processes (Greedy, checking easiest first) ---
            for gpu_id in range(n_gpus):
                for i in task_scheduler_order: # Iterate in simplicity order
                    if not tasks_started[i]: # Check if task i (global index) has started
                        enough_quota = gpu_quotas[gpu_id] > task_usages[i]
                        active_processes = sum(map(int, tasks_started)) - sum(map(int, tasks_finished))
                        enough_cpus = active_processes < n_cpus
                        
                        if enough_quota and enough_cpus:
                            gpu_quotas[gpu_id] -= task_usages[i]
                            args = (task_names[i], split, end_time, n_iteration_list[i], gpu_id, memory_dict, solutions_dict, error_queue)
                            p = multiprocessing.Process(target=solve_task.solve_task, args=args)
                            p.start()
                            processes[i] = p
                            tasks_started[i] = True
                            process_gpu_ids[i] = gpu_id
                            if verbose:
                                print(f"{task_names[i]} (Score: {hybrid_scores[i]:.2f}) started on gpu {process_gpu_ids[i]}. "
                                      f"New quota: {gpu_quotas[gpu_id]}")
            time.sleep(1)
            
        if not error_queue.empty():
            raise ValueError(error_queue.get())
            
        memory_dict = dict(memory_dict)
        solutions_dict = dict(solutions_dict)
        
    time_taken = time.time() - t
    if verbose:
        print(f'All jobs in batch finished in {time_taken:.2f} seconds.')
    return memory_dict, solutions_dict, time_taken

# ==============================================================================
# SECTION 5: CALIBRATION - MEASURE MEMORY USAGE
# ==============================================================================

if __name__ == '__main__':
    print("\n--- Calibrating Memory Usage ---")
    gpu_memory_quotas = [torch.cuda.mem_get_info(i)[0] for i in range(n_gpus)]

    gpu_task_quotas = [int(gpu_memory_quota // (4 * 1024**3)) for gpu_memory_quota in gpu_memory_quotas]
    task_usages = [1 for i in range(n_tasks)]
    
    # We must run all tasks here to get their memory, even if we skip them later
    # This needs the global 'sorted_taskid' from the original simplicity score,
    # so we'll compute a temporary one.
    temp_scores = np.array([task_properties[name]['score'] for name in task_names])
    sorted_taskid = np.argsort(-temp_scores) # Create a temporary sorted_taskid
    
    memory_dict, _, _ = parallelize_runs(gpu_task_quotas, task_usages, 2*np.ones(n_tasks, dtype=int), verbose=False)
    
    # Sort the tasks by decreasing memory usage
    tasks = sorted(memory_dict.items(), key=lambda x: x[1], reverse=True)
    task_names_sorted_by_mem, task_memory_usages_sorted = zip(*tasks)
    
    # Re-order our global lists to match this memory-sorted order
    task_names = list(task_names_sorted_by_mem)
    task_memory_usages = list(task_memory_usages_sorted)
    
    # --- IMPORTANT: Re-calculate all scores/IDs to match the NEW task_names order ---
    hybrid_scores = np.array([task_properties[name]['score'] for name in task_names])
    
    # This is the FINAL global scheduler order, from easiest to hardest
    sorted_taskid = np.argsort(-hybrid_scores) 
    
    print("Memory calibration complete. Tasks re-ordered by memory usage.")

# ==============================================================================
# SECTION 6: CALIBRATION - MEASURE TIME PER STEP
# ==============================================================================

if __name__ == '__main__':
    print("\n--- Calibrating Time Per Step ---")
    test_steps_calib_total = 500 if fake_mode else 2000
    
    # Distribute calibration steps based on the new hybrid score
    iterations_list_calib = (1.0+hybrid_scores*test_steps_calib_total/sum(hybrid_scores)).astype(int)
    
    safe_gpu_memory_quotas = [memory_quota - 6 * 1024**3 for memory_quota in gpu_memory_quotas]

    _, _, time_taken = parallelize_runs(safe_gpu_memory_quotas, task_memory_usages, iterations_list_calib, verbose=False)

    # --- CRITICAL BUG FIX: Divide by actual steps run ---
    total_steps_run_calib = sum(iterations_list_calib)
    if total_steps_run_calib == 0: total_steps_run_calib = 1 # Avoid division by zero
    
    time_per_step = time_taken / total_steps_run_calib
    if time_per_step == 0: time_per_step = 1e-3 # Avoid division by zero
    
    print(f"Time calibration complete. Ran {total_steps_run_calib} steps in {time_taken:.2f}s.")
    print(f"Average time per step: {time_per_step:.4f} seconds.")

# ==============================================================================
# SECTION 7: FINAL SOLVE (HYBRID TRIAGE STRATEGY)
# ==============================================================================

if __name__ == '__main__':
    print("\n--- Starting Final Solve ---")
    time_left = end_time - time.time()
    
    # Calculate total steps we can afford for the *final* run
    test_steps_final = 500 if fake_mode else int(time_left / time_per_step)
    # Use a 1.5x multiplier to aim to use the full buffer
    test_steps_final = int(test_steps_final * 1.5) 

    print(f"Time left: {time_left/3600:.2f} hours. Aiming for {test_steps_final} total steps.")

    # --- HYBRID TRIAGE: Focus on easiest 75% of tasks ---
    tasks_to_keep_pct = 0.75 
    num_tasks_to_solve = int(n_tasks * tasks_to_keep_pct)
    
    # Get the indices of tasks to solve (top 75% from sorted_taskid)
    task_ids_to_solve = sorted_taskid[:num_tasks_to_solve]
    task_ids_to_ignore = sorted_taskid[num_tasks_to_solve:]
    
    num_tasks_ignored = len(task_ids_to_ignore)

    print(f"Applying Triage: Focusing on the {num_tasks_to_solve} easiest tasks.")
    print(f"Ignoring the {num_tasks_ignored} hardest tasks.")

    # Initialize all iterations to 0
    iterations_list = np.zeros(n_tasks, dtype=int)
    
    # Get the hybrid scores for *only* the tasks we are solving
    scores_to_use = hybrid_scores[task_ids_to_solve]
    
    if sum(scores_to_use) > 0:
        # Distribute the total step budget among *only* the Top 75%
        distributed_steps = (1.0 + scores_to_use * test_steps_final / sum(scores_to_use)).astype(int)
        
        # Assign the steps to the correct indices in the main list
        iterations_list[task_ids_to_solve] = distributed_steps
    else:
        print("Warning: Sum of scores for high-potential tasks is zero. No steps will be run.")

    # --- Run the final solver ---
    _, solutions_dict, time_taken = parallelize_runs(safe_gpu_memory_quotas, task_memory_usages, iterations_list, verbose=True)
    
    # Format the solutions and put into submission file
    with open('submission.json', 'w') as f:
        json.dump(solutions_dict, f, indent=4)
        
    print(f"\n{n_tasks} tasks processed.")
    print(f"Ran a total of {sum(iterations_list)} steps across {num_tasks_to_solve} tasks.")
    print(f"{time_taken:.2f} seconds taken.")

# ==============================================================================
# SECTION 8: VISUALIZATION (Fixed)
# ==============================================================================

# --- SYNTAX ERROR FIX: 'import json' is now on its own line ---
import json
training_solution_path = '/kaggle/input/arc-prize-2025/arc-agi_training_solutions.json'
with open(training_solution_path, 'r') as f:
    training_solution = json.load(f)
    
evaluation_solution_path = '/kaggle/input/arc-prize-2025/arc-agi_evaluation_solutions.json'
with open(evaluation_solution_path, 'r') as f:
    evaluation_solution = json.load(f)

# This block now correctly loads data for local evaluation in fake_mode
conbiled_solution = {}
if fake_mode:
    arc_challenge_file = '/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json'
else:
    arc_challenge_file = '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json'

# Load challenge data for visualization
with open(arc_challenge_file, 'r') as f:
    arc_data = json.load(f)

# Combine solutions *only if* in fake_mode (where we have eval solutions)
if fake_mode:
    print(f"Loading ground truth solutions for {len(arc_data)} evaluation tasks.")
    for case_id in arc_data:
        if case_id in evaluation_solution:
            conbiled_solution[case_id] = evaluation_solution[case_id]
        elif case_id in training_solution:
            conbiled_solution[case_id] = training_solution[case_id] 

# --- INDENTATION ERROR FIX: This function definition is now at the top level ---
def visualize_arc_results():    
    """Visualize ARC problem solutions from submission.json"""
    import matplotlib.pyplot as plt
    from matplotlib import colors
    import json
    import os
    import numpy as np
    
    print("\n" + "="*80)
    print("VISUALIZING ARC SOLUTION RESULTS")
    print("="*80)
    
    submission_path = 'submission.json'
    if not os.path.exists(submission_path):
        print(f"Submission file not found at {submission_path}")
        return
    
    print(f"Found submission file: {submission_path}")
    
    with open(submission_path, 'r') as f:
        submission_data = json.load(f)
    
    print(f"Loaded submission with {len(submission_data)} tasks")
    
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    
    def is_non_trivial_prediction(pred_array):
        return np.any(np.array(pred_array) > 0)
    
    def visualize_submission_result(task_id, task_data, submission_output, test_idx):
        pred_1 = np.array(submission_output['attempt_1'])
        pred_2 = np.array(submission_output['attempt_2'])
        
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
        ax5.imshow(pred_1, cmap=cmap, norm=norm)
        ax5.grid(True, which='both', color='lightgrey', linewidth=0.5)
        ax5.set_title("Model Prediction (Attempt 1)")
        ax5.set_xticks([]); ax5.set_yticks([])
        
        ax6 = fig.add_subplot(grid_spec[1, 2])
        ax6.imshow(pred_2, cmap=cmap, norm=norm)
        ax6.grid(True, which='both', color='lightgrey', linewidth=0.5)
        ax6.set_title("Model Prediction (Attempt 2)")
        ax6.set_xticks([]); ax6.set_yticks([])
        
        # Check for ground truth using the fixed conbiled_solution
        if task_id in conbiled_solution and len(conbiled_solution[task_id]) > test_idx:
            ground_truth = conbiled_solution[task_id][test_idx]
            
            if ground_truth:
                ax4 = fig.add_subplot(grid_spec[1, 1])
                ax4.imshow(ground_truth, cmap=cmap, norm=norm)
                ax4.grid(True, which='both', color='lightgrey', linewidth=0.5)
                ax4.set_title("Ground Truth")
                ax4.set_xticks([]); ax4.set_yticks([])
                
                match_1 = np.array_equal(pred_1, ground_truth) if is_non_trivial_prediction(pred_1) else False
                match_2 = np.array_equal(pred_2, ground_truth) if is_non_trivial_prediction(pred_2) else False
                
                ax5.set_title(f"Prediction 1: {'✓' if match_1 else '✗'}")
                ax6.set_title(f"Prediction 2: {'✓' if match_2 else '✗'}")
                
                print(f"  Results: Attempt 1: {'✓' if match_1 else '✗'}, Attempt 2: {'✓' if match_2 else '✗'}")
                print(f"  Shape - GT: {np.array(ground_truth).shape}, P1: {pred_1.shape}, P2: {pred_2.shape}")
        
        plt.suptitle(f"Task {task_id} - Test Example #{test_idx+1}", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
        return True
    
    # Process all results from submission
    visualized_count = 0
    skipped_count = 0
    all_predictions = []
    
    for task_id in submission_data:
        if task_id in arc_data:
            task_data = arc_data[task_id]
            for test_idx, test_prediction in enumerate(submission_data[task_id]):
                pred_1 = np.array(test_prediction['attempt_1'])
                pred_2 = np.array(test_prediction['attempt_2'])
                has_non_zero_pred = is_non_trivial_prediction(pred_1) or is_non_trivial_prediction(pred_2)
                
                has_ground_truth = False
                correct_count = 0
                
                if task_id in conbiled_solution and len(conbiled_solution[task_id]) > test_idx:
                    has_ground_truth = True
                    ground_truth = conbiled_solution[task_id][test_idx]
                    
                    if has_non_zero_pred:
                        match_1 = np.array_equal(pred_1, ground_truth) if is_non_trivial_prediction(pred_1) else False
                        match_2 = np.array_equal(pred_2, ground_truth) if is_non_trivial_prediction(pred_2) else False
                        correct_count = int(match_1) + int(match_2)
                
                all_predictions.append((task_id, test_idx, correct_count, has_ground_truth, has_non_zero_pred))
    
    all_predictions.sort(key=lambda x: (-int(x[3]), -x[2]))
    
    print(f"\nFound {len(all_predictions)} total predictions to visualize")
    
    max_samples = 10 
    samples_to_show = all_predictions[:max_samples]
    
    print(f"Showing {len(samples_to_show)} of {len(all_predictions)} prediction samples (sorted by correctness)")
    
    for task_id, test_idx, correct_count, has_ground_truth, has_non_zero_pred in samples_to_show:
        task_data = arc_data[task_id]
        submission_output = submission_data[task_id][test_idx]
        
        score_info = f" (Score: {correct_count}/2)" if has_ground_truth and has_non_zero_pred else " (no ground truth)" if not has_ground_truth else " (all zeros - no score)"
        print(f"\nTask: {task_id} - Test #{test_idx+1}{score_info}")
        
        if visualize_submission_result(task_id, task_data, submission_output, test_idx):
            visualized_count += 1
        else:
            skipped_count += 1
    
    print(f"\nVisualized {visualized_count} inference results (skipped {skipped_count} with all-zero predictions)")
    
    # Calculate overall accuracy statistics if in fake/debug mode
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
                ground_truth = conbiled_solution[task_id][test_idx]
                
                if not ground_truth: continue
                    
                pred_1 = np.array(submission_data[task_id][test_idx]['attempt_1'])
                pred_2 = np.array(submission_data[task_id][test_idx]['attempt_2'])
                
                if not is_non_trivial_prediction(pred_1) and not is_non_trivial_prediction(pred_2):
                    zero_predictions += 1
                    continue
                
                total_scored_tests += 1
                
                match_1 = np.array_equal(pred_1, ground_truth) if is_non_trivial_prediction(pred_1) else False
                match_2 = np.array_equal(pred_2, ground_truth) if is_non_trivial_prediction(pred_2) else False
                
                if match_1: correct_attempt1 += 1
                if match_2: correct_attempt2 += 1
                if match_1 or match_2: correct_any += 1
        
        if total_tests > 0:
            print("\n" + "="*80)
            print("OVERALL ACCURACY STATISTICS (LOCAL EVALUATION)")
            print("="*80)
            print(f"Total test examples with ground truth: {total_tests}")
            print(f"Test examples with all-zero predictions (excluded): {zero_predictions}")
            print(f"Test examples scored: {total_scored_tests}")
            
            if total_scored_tests > 0:
                # --- TYPO FIX: Changed 'total_total_scored_tests' to 'total_scored_tests' ---
                print(f"Correct on attempt 1: {correct_attempt1}/{total_scored_tests} ({correct_attempt1/total_scored_tests:.2%})") 
                print(f"Correct on attempt 2: {correct_attempt2}/{total_scored_tests} ({correct_attempt2/total_scored_tests:.2%})")
                print(f"Correct on either attempt: {correct_any}/{total_scored_tests} ({correct_any/total_scored_tests:.2%})")
            else:
                print("No non-zero predictions to calculate accuracy")
                
            print(f"Overall completion rate: {total_scored_tests/total_tests:.2%} of tests have non-zero predictions")
            print("="*80)

# Add this line to the notebook to call the visualization function
# Call after your submission.json has been created
if fake_mode:
    visualize_arc_results()

