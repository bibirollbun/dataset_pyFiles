# ==============================================================================
# ARC Prize 2025 - Final Intelligent Solver
#
# Author: Sanyam Sanjay Sharma
# Date: August 17, 2025
#
# Description:
# This script is the definitive version of the AI solver for the ARC Prize 2025.
# It employs a sophisticated "deduce-and-verify" strategy, making it
# significantly more intelligent and efficient. It perceives high-level
# patterns, deduces likely transformations, and then verifies them with a
# powerful heuristic search engine.
#
# Core Components:
#   1. Advanced Perception: Identifies objects, their properties, and global
#      grid properties like symmetry.
#   2. Expanded Domain-Specific Language (DSL): A rich vocabulary of actions,
#      including object manipulation and advanced grid transformations.
#   3. Deductive Engine: Intelligently generates candidate solutions by
#      analyzing the differences between training pairs.
#   4. Heuristic Search Engine: A guided A* search algorithm that prioritizes
#      the most promising solutions to solve complex, multi-step puzzles.
# ==============================================================================

import json
import os
from pathlib import Path
import numpy as np
from collections import deque
import time
import heapq

# ==============================================================================
# SECTION 1: CORE PERCEPTION MODULE (The "Eyes")
# ==============================================================================

def analyze_grid_properties(grid):
    """Extracts high-level properties from a grid."""
    properties = {
        'height': grid.shape[0],
        'width': grid.shape[1],
        'unique_colors': len(np.unique(grid)),
        'is_horizontally_symmetric': np.array_equal(grid, np.fliplr(grid)),
        'is_vertically_symmetric': np.array_equal(grid, np.flipud(grid)),
    }
    return properties

def find_objects(grid):
    """Identifies all distinct, contiguous objects in a grid."""
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid, dtype=int)
    height, width = grid.shape
    if grid.size == 0: return []
    colors, counts = np.unique(grid, return_counts=True)
    background_color = colors[np.argmax(counts)]
    visited = np.zeros_like(grid, dtype=bool)
    objects = []
    
    for r in range(height):
        for c in range(width):
            if visited[r, c] or grid[r, c] == background_color: continue
            obj_color = grid[r, c]
            new_object = {'color': int(obj_color), 'pixels': [], 'min_row': r, 'max_row': r, 'min_col': c, 'max_col': c}
            q = deque([(r, c)])
            visited[r, c] = True
            pixels_in_obj = []
            while q:
                row, col = q.popleft()
                pixels_in_obj.append((row, col))
                new_object['min_row'] = min(new_object['min_row'], row)
                new_object['max_row'] = max(new_object['max_row'], row)
                new_object['min_col'] = min(new_object['min_col'], col)
                new_object['max_col'] = max(new_object['max_col'], col)
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < height and 0 <= nc < width and not visited[nr, nc] and grid[nr, nc] == obj_color:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            new_object['pixels'] = sorted(pixels_in_obj)
            obj_h = new_object['max_row'] - new_object['min_row'] + 1
            obj_w = new_object['max_col'] - new_object['min_col'] + 1
            shape = np.full((obj_h, obj_w), int(background_color), dtype=int)
            for r_pix, c_pix in new_object['pixels']:
                shape[r_pix - new_object['min_row'], c_pix - new_object['min_col']] = obj_color
            new_object['shape'] = shape
            new_object['size'] = len(new_object['pixels'])
            new_object['is_touching_border'] = any(p[0] == 0 or p[0] == height - 1 or p[1] == 0 or p[1] == width - 1 for p in new_object['pixels'])
            objects.append(new_object)
            
    objects.sort(key=lambda o: (o['min_row'], o['min_col']))
    for i, obj in enumerate(objects):
        obj['id'] = i
        
    return objects

# ==============================================================================
# SECTION 2: DOMAIN-SPECIFIC LANGUAGE (DSL) & ABSTRACT REASONING
# ==============================================================================

def select_objects(objects, selector):
    """Selects objects based on abstract properties."""
    prop = selector['property']
    val = selector['value']
    
    if prop == 'all': return objects
    if prop == 'color': return [o for o in objects if o['color'] == val]
    if prop == 'size':
        sizes = [o['size'] for o in objects]
        if not sizes: return []
        if val == 'max': target_size = max(sizes) if sizes else 0
        elif val == 'min': target_size = min(sizes) if sizes else 0
        else: target_size = val
        return [o for o in objects if o['size'] == target_size]
    if prop == 'is_touching_border': return [o for o in objects if o['is_touching_border'] == val]
    return []

def apply_program(grid, program):
    """Applies a single transformation program to a grid."""
    grid = np.array(grid)
    program_name, params = program
    
    if program_name.startswith('obj_'):
        objects = find_objects(grid)
        selected_objects = select_objects(objects, params['selector'])
        if not selected_objects: return grid
        
        new_grid = grid.copy()
        for obj in selected_objects:
            if program_name == 'obj_recolor': new_grid = recolor_object(new_grid, obj, params['new_color'])
            elif program_name == 'obj_move': new_grid = move_object(new_grid, obj, params['dr'], params['dy'])
            elif program_name == 'obj_delete': new_grid = delete_object(new_grid, obj)
            elif program_name == 'obj_copy': new_grid = copy_object(new_grid, obj, params['dr'], params['dy'])
            elif program_name == 'obj_fill_shape': new_grid = fill_shape(new_grid, obj)
        return new_grid
            
    elif program_name.startswith('grid_'):
        if program_name == 'grid_rotate': return np.rot90(grid, k=params['k'])
        if program_name == 'grid_flip': return np.flip(grid, axis=params['axis'])
            
    return grid

def recolor_object(grid, obj, new_color):
    for r, c in obj['pixels']: grid[r, c] = new_color
    return grid

def move_object(grid, obj, dr, dy):
    bg = get_background_color(grid)
    temp_grid = grid.copy()
    for r, c in obj['pixels']: temp_grid[r, c] = bg
    h, w = grid.shape
    for r, c in obj['pixels']:
        nr, nc = r + dr, c + dy
        if 0 <= nr < h and 0 <= nc < w: temp_grid[nr, nc] = obj['color']
    return temp_grid

def delete_object(grid, obj):
    bg = get_background_color(grid)
    for r, c in obj['pixels']: grid[r, c] = bg
    return grid

def copy_object(grid, obj, dr, dy):
    h, w = grid.shape
    for r, c in obj['pixels']:
        nr, nc = r + dr, c + dy
        if 0 <= nr < h and 0 <= nc < w: grid[nr, nc] = obj['color']
    return grid

def fill_shape(grid, obj):
    """Fills the bounding box of an object with its color."""
    for r in range(obj['min_row'], obj['max_row'] + 1):
        for c in range(obj['min_col'], obj['max_col'] + 1):
            grid[r, c] = obj['color']
    return grid

def get_background_color(grid):
    colors, counts = np.unique(grid, return_counts=True)
    return colors[np.argmax(counts)]

# ==============================================================================
# SECTION 3: THE PROGRAM SYNTHESIS ENGINE (The "Brain")
# ==============================================================================

def make_hashable(o):
    """Recursively converts a container to a hashable type (tuple)."""
    if isinstance(o, dict): return tuple(sorted((k, make_hashable(v)) for k, v in o.items()))
    if isinstance(o, (list, tuple)): return tuple(make_hashable(e) for e in o)
    if isinstance(o, np.ndarray): return o.tobytes()
    return o

def generate_heuristic_programs(task):
    """Generates a targeted list of candidate programs by analyzing the first training pair."""
    candidate_programs = []
    in_grid = np.array(task['train'][0]['input'])
    out_grid = np.array(task['train'][0]['output'])
    in_props = analyze_grid_properties(in_grid)
    out_props = analyze_grid_properties(out_grid)

    # Grid-level heuristics
    if np.array_equal(np.fliplr(in_grid), out_grid): candidate_programs.append(('grid_flip', {'axis': 1}))
    if np.array_equal(np.flipud(in_grid), out_grid): candidate_programs.append(('grid_flip', {'axis': 0}))
    for k in [1, 2, 3]:
        if np.array_equal(np.rot90(in_grid, k=k), out_grid): candidate_programs.append(('grid_rotate', {'k': k}))

    # Object-level heuristics
    in_objects = find_objects(in_grid)
    out_objects = find_objects(out_grid)
    selectors = [{'property': 'all'}, {'property': 'size', 'value': 'max'}, {'property': 'size', 'value': 'min'}]
    for c in np.unique(in_grid): selectors.append({'property': 'color', 'value': c})
    
    for selector in selectors:
        # Simple move/recolor guesses
        for color in np.unique(out_grid): candidate_programs.append(('obj_recolor', {'selector': selector, 'new_color': color}))
        candidate_programs.append(('obj_move', {'selector': selector, 'dr': 1, 'dy': 0}))
        candidate_programs.append(('obj_move', {'selector': selector, 'dr': 0, 'dy': 1}))
        candidate_programs.append(('obj_delete', {'selector': selector}))
        candidate_programs.append(('obj_fill_shape', {'selector': selector}))

    return list(set(make_hashable(p) for p in candidate_programs))

def search_for_program(task, candidate_programs, max_depth=3, timeout_seconds=30):
    """Performs a Heuristic Search (A*) to find a sequence of transformations."""
    start_time = time.time()
    initial_states = [np.array(p['input']) for p in task['train']]
    target_states = [np.array(p['output']) for p in task['train']]

    def heuristic(states):
        total_mismatch = 0
        for s, t in zip(states, target_states):
            if s.shape != t.shape: total_mismatch += s.size + t.size
            else: total_mismatch += np.sum(s != t)
        return total_mismatch

    queue = [(heuristic(initial_states), 0, [], initial_states)] # (cost, depth, program_seq, states)
    visited_states = {make_hashable(initial_states)}

    while queue:
        if time.time() - start_time > timeout_seconds: return None
        
        cost, depth, current_program_seq, current_states = heapq.heappop(queue)

        if cost == 0 and current_program_seq: return current_program_seq
        if depth >= max_depth: continue

        for program in candidate_programs:
            new_program_seq = current_program_seq + [program]
            try:
                next_states = [apply_program(s, program) for s in current_states]
            except Exception: continue

            next_states_hash = make_hashable(next_states)
            if next_states_hash in visited_states: continue
            visited_states.add(next_states_hash)
            
            new_cost = heuristic(next_states)
            heapq.heappush(queue, (new_cost, depth + 1, new_program_seq, next_states))
            
    return None

def solve_task(task):
    """Orchestrates the process of finding and applying a solution program."""
    candidate_programs = generate_heuristic_programs(task)
    solution_program_seq = search_for_program(task, candidate_programs)
    
    predictions = []
    for test_pair in task['test']:
        predicted_grid = np.array(test_pair['input'])
        if solution_program_seq:
            for program in solution_program_seq:
                predicted_grid = apply_program(predicted_grid, program)
        predictions.append(predicted_grid.tolist())
        
    return predictions

# ==============================================================================
# SECTION 4: KAGGLE SUBMISSION BOILERPLATE
# ==============================================================================

if __name__ == '__main__':
    start_time_total = time.time()
    data_path = Path('/kaggle/input/arc-prize-2025')
    test_challenges_file = data_path / 'arc-agi_test_challenges.json'
    submission_file = Path('/kaggle/working/submission.json')

    if not data_path.exists():
        print("Kaggle environment not found. Creating dummy files for local testing.")
        os.makedirs('/kaggle/input/arc-prize-2025', exist_ok=True)
        os.makedirs('/kaggle/working', exist_ok=True)
        dummy_task_id = "a7495283" # A task solved by "recolor the largest object"
        dummy_data = {dummy_task_id: {"train":[{"input":[[0,0,0,0],[0,1,1,0],[0,1,0,0],[0,0,0,0]],"output":[[0,0,0,0],[0,2,2,0],[0,2,0,0],[0,0,0,0]]}],"test":[{"input":[[0,3,3,0],[0,3,3,0],[0,0,3,0],[0,0,0,0]],"output":[[0,8,8,0],[0,8,8,0],[0,0,8,0],[0,0,0,0]]}]}}
        with open(test_challenges_file, 'w') as f: json.dump(dummy_data, f)

    try:
        with open(test_challenges_file, 'r') as f: tasks = json.load(f)
    except FileNotFoundError:
        tasks = {}

    submission = {}
    for i, (task_id, task_data) in enumerate(tasks.items()):
        print(f"Processing task {i+1}/{len(tasks)}: {task_id}")
        predicted_grids = solve_task(task_data)
        task_predictions = [{"attempt_1": grid, "attempt_2": grid} for grid in predicted_grids]
        submission[task_id] = task_predictions

    with open(submission_file, 'w') as f: json.dump(submission, f)
    
    end_time_total = time.time()
    print("-" * 30)
    print(f"Submission file created at: {submission_file}")
    print(f"Total tasks processed: {len(submission)}")
    print(f"Total runtime: {end_time_total - start_time_total:.2f} seconds")
    print("Script finished successfully.")


