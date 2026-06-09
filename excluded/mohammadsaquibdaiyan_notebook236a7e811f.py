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


import json
import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
import itertools
import random
from collections import defaultdict, Counter
import os

# ============================================================================
# DATA LOADING AND VISUALIZATION
# ============================================================================

def load_data(file_path):
    """
    Load and parse the ARC challenge data from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file containing ARC tasks
        
    Returns:
        dict: Dictionary containing task data
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def visualize_grid(grid, ax=None, title=None):
    """
    Visualize a grid using a color map.
    
    Args:
        grid (list): 2D grid of integers representing colors
        ax (matplotlib.axes.Axes, optional): Axes to plot on
        title (str, optional): Title for the plot
        
    Returns:
        matplotlib.axes.Axes: The axes with the visualization
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    
    # Use a consistent colormap for visualization
    colors = ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00', 
              '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']
    
    # Create a colormap
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors[:max(10, max([max(row) for row in grid]) + 1)])
    
    # Plot the grid
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=9)
    
    # Add grid lines
    ax.grid(color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(np.arange(-0.5, len(grid[0]), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(grid), 1), minor=True)
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add title if provided
    if title:
        ax.set_title(title)
    
    return ax

def visualize_task(task, task_id=None):
    """
    Visualize a task with its train examples and test input.
    
    Args:
        task (dict): Task dictionary containing train and test pairs
        task_id (str, optional): ID of the task for title
    """
    n_train = len(task['train'])
    
    # Set up the figure
    fig = plt.figure(figsize=(4 * n_train, 8))
    
    # Add a title if task_id is provided
    if task_id:
        fig.suptitle(f'Task {task_id}', fontsize=16)
    
    # Plot training examples
    for i, train_pair in enumerate(task['train']):
        # Plot input
        ax1 = plt.subplot(2, n_train, i + 1)
        visualize_grid(train_pair['input'], ax=ax1, title=f'Train Input {i+1}')
        
        # Plot output
        ax2 = plt.subplot(2, n_train, i + n_train + 1)
        visualize_grid(train_pair['output'], ax=ax2, title=f'Train Output {i+1}')
    
    # Plot test input
    ax_test = plt.subplot(1, n_train, (n_train // 2) + 1)
    if isinstance(task['test'], list):
        # Multiple test inputs
        visualize_grid(task['test'][0]['input'], ax=ax_test, title='Test Input 1')
    else:
        # Single test input
        visualize_grid(task['test']['input'], ax=ax_test, title='Test Input')
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# GRID OPERATIONS AND TRANSFORMATIONS
# ============================================================================

def grid_to_numpy(grid):
    """Convert a grid (list of lists) to a numpy array."""
    return np.array(grid, dtype=int)

def numpy_to_grid(array):
    """Convert a numpy array to a grid (list of lists)."""
    return array.tolist()

def get_grid_dimensions(grid):
    """Get the height and width of a grid."""
    return len(grid), len(grid[0])

def get_unique_colors(grid):
    """Get all unique colors in a grid."""
    return np.unique(grid_to_numpy(grid))

def rotate_grid(grid, k=1):
    """Rotate the grid by 90 degrees k times (counterclockwise)."""
    arr = grid_to_numpy(grid)
    rotated = np.rot90(arr, k=k)
    return numpy_to_grid(rotated)

def flip_grid_horizontal(grid):
    """Flip the grid horizontally."""
    arr = grid_to_numpy(grid)
    flipped = np.fliplr(arr)
    return numpy_to_grid(flipped)

def flip_grid_vertical(grid):
    """Flip the grid vertically."""
    arr = grid_to_numpy(grid)
    flipped = np.flipud(arr)
    return numpy_to_grid(flipped)

def color_remap(grid, color_map):
    """Remap colors in a grid according to a color map."""
    arr = grid_to_numpy(grid)
    result = np.zeros_like(arr)
    
    for old_color, new_color in color_map.items():
        result[arr == old_color] = new_color
    
    return numpy_to_grid(result)

def check_exact_match(grid1, grid2):
    """Check if two grids exactly match."""
    return grid_to_numpy(grid1).tolist() == grid_to_numpy(grid2).tolist()

# ============================================================================
# PATTERN RECOGNITION
# ============================================================================

def identify_objects(grid, background=0):
    """
    Identify connected components (objects) in a grid.
    
    Args:
        grid (list): 2D grid of integers
        background (int): Value representing the background
        
    Returns:
        list: List of objects, where each object is a list of (x, y, color) tuples
    """
    arr = grid_to_numpy(grid)
    visited = np.zeros_like(arr, dtype=bool)
    objects = []
    
    height, width = arr.shape
    
    for i in range(height):
        for j in range(width):
            if not visited[i, j] and arr[i, j] != background:
                # Start a new object
                color = arr[i, j]
                object_cells = []
                
                # Use BFS to find all connected cells of the same color
                queue = [(i, j)]
                visited[i, j] = True
                
                while queue:
                    y, x = queue.pop(0)
                    object_cells.append((x, y, color))
                    
                    # Check neighboring cells
                    for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < height and 0 <= nx < width and
                                not visited[ny, nx] and arr[ny, nx] == color):
                            queue.append((ny, nx))
                            visited[ny, nx] = True
                
                objects.append(object_cells)
    
    return objects

def count_objects_by_color(grid, background=0):
    """Count the number of objects of each color in the grid."""
    objects = identify_objects(grid, background)
    counts = defaultdict(int)
    
    for obj in objects:
        color = obj[0][2]  # Get color from first cell of the object
        counts[color] += 1
    
    return counts

def extract_object_properties(objects):
    """
    Extract properties from a list of objects.
    
    Returns:
        list: List of dictionaries containing object properties
    """
    object_props = []
    
    for obj in objects:
        # Extract x and y coordinates
        x_coords = [cell[0] for cell in obj]
        y_coords = [cell[1] for cell in obj]
        
        # Get color (should be the same for all cells in an object)
        color = obj[0][2]
        
        # Calculate bounding box
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Calculate dimensions
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        area = len(obj)
        
        # Create a property dictionary
        props = {
            'color': color,
            'area': area,
            'width': width,
            'height': height,
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max,
            'cells': obj,
            'center': (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))
        }
        
        object_props.append(props)
    
    return object_props

# ============================================================================
# TRANSFORMATION DETECTION
# ============================================================================

def detect_size_changes(input_grid, output_grid):
    """Detect if the grid size has changed and how."""
    in_height, in_width = get_grid_dimensions(input_grid)
    out_height, out_width = get_grid_dimensions(output_grid)
    
    size_changes = {
        'height_ratio': out_height / in_height if in_height > 0 else 0,
        'width_ratio': out_width / in_width if in_width > 0 else 0,
        'height_diff': out_height - in_height,
        'width_diff': out_width - in_width,
    }
    
    return size_changes

def detect_color_mapping(input_grid, output_grid):
    """
    Attempt to detect a color mapping from input to output.
    
    Returns:
        dict: Color mapping dictionary or None if no consistent mapping found
    """
    # Create flattened versions of the grids
    in_arr = grid_to_numpy(input_grid).flatten()
    out_arr = grid_to_numpy(output_grid).flatten()
    
    # If grids have different sizes, we can't create a direct mapping
    if len(in_arr) != len(out_arr):
        return None
    
    # Check for consistent mapping
    mapping = {}
    for i, o in zip(in_arr, out_arr):
        if i in mapping and mapping[i] != o:
            # Inconsistent mapping found
            return None
        mapping[i] = o
    
    return mapping

def detect_object_transformations(input_grid, output_grid):
    """
    Detect transformations applied to objects from input to output.
    
    Returns:
        list: List of transformation dictionaries
    """
    # Get objects in both grids
    input_objects = extract_object_properties(identify_objects(input_grid))
    output_objects = extract_object_properties(identify_objects(output_grid))
    
    transformations = []
    
    # First, try to match objects by position and analyze transforms
    for in_obj in input_objects:
        for out_obj in output_objects:
            # Check if the objects overlap or are in similar positions
            in_center = in_obj['center']
            out_center = out_obj['center']
            
            distance = np.sqrt((in_center[0] - out_center[0])**2 + 
                               (in_center[1] - out_center[1])**2)
            
            # If the centers are close, consider these as the same object transformed
            if distance < max(in_obj['width'], in_obj['height'], out_obj['width'], out_obj['height']):
                transform = {
                    'type': 'object_transform',
                    'input_object': in_obj,
                    'output_object': out_obj,
                    'color_change': in_obj['color'] != out_obj['color'],
                    'size_change': {
                        'width_ratio': out_obj['width'] / in_obj['width'] if in_obj['width'] > 0 else 0,
                        'height_ratio': out_obj['height'] / in_obj['height'] if in_obj['height'] > 0 else 0,
                        'area_ratio': out_obj['area'] / in_obj['area'] if in_obj['area'] > 0 else 0
                    },
                    'position_change': {
                        'x_diff': out_obj['x_min'] - in_obj['x_min'],
                        'y_diff': out_obj['y_min'] - in_obj['y_min']
                    }
                }
                transformations.append(transform)
    
    # Also check for simple count changes by color
    input_counts = count_objects_by_color(input_grid)
    output_counts = count_objects_by_color(output_grid)
    
    for color in set(input_counts.keys()) | set(output_counts.keys()):
        in_count = input_counts.get(color, 0)
        out_count = output_counts.get(color, 0)
        
        if in_count != out_count:
            transformations.append({
                'type': 'count_change',
                'color': color,
                'input_count': in_count,
                'output_count': out_count,
                'diff': out_count - in_count,
                'ratio': out_count / in_count if in_count > 0 else float('inf')
            })
    
    return transformations

# ============================================================================
# PATTERN RULE EXTRACTION
# ============================================================================

def extract_rules_from_examples(task):
    """
    Extract potential transformation rules from the task examples.
    
    Args:
        task (dict): Task dictionary containing train examples
        
    Returns:
        list: List of rule dictionaries
    """
    rules = []
    
    for i, train_pair in enumerate(task['train']):
        input_grid = train_pair['input']
        output_grid = train_pair['output']
        
        # Basic properties
        size_changes = detect_size_changes(input_grid, output_grid)
        color_mapping = detect_color_mapping(input_grid, output_grid)
        object_transformations = detect_object_transformations(input_grid, output_grid)
        
        # Check for simple transformations
        # 1. Rotation
        for k in range(1, 4):
            rotated = rotate_grid(input_grid, k)
            if check_exact_match(rotated, output_grid):
                rules.append({
                    'type': 'rotation',
                    'steps': k,
                    'example_idx': i
                })
        
        # 2. Flipping
        flipped_h = flip_grid_horizontal(input_grid)
        if check_exact_match(flipped_h, output_grid):
            rules.append({
                'type': 'flip_horizontal',
                'example_idx': i
            })
        
        flipped_v = flip_grid_vertical(input_grid)
        if check_exact_match(flipped_v, output_grid):
            rules.append({
                'type': 'flip_vertical',
                'example_idx': i
            })
        
        # 3. Color mapping
        if color_mapping is not None:
            rules.append({
                'type': 'color_mapping',
                'mapping': color_mapping,
                'example_idx': i
            })
        
        # 4. Size change
        if abs(1 - size_changes['height_ratio']) > 0.01 or abs(1 - size_changes['width_ratio']) > 0.01:
            rules.append({
                'type': 'resize',
                'changes': size_changes,
                'example_idx': i
            })
        
        # 5. Object transformations
        if object_transformations:
            rules.append({
                'type': 'object_transforms',
                'transformations': object_transformations,
                'example_idx': i
            })
    
    # Find consistent rules across examples
    consistent_rules = []
    rule_types = set(rule['type'] for rule in rules)
    
    for rule_type in rule_types:
        type_rules = [r for r in rules if r['type'] == rule_type]
        if len(type_rules) == len(task['train']):
            # Rule appears in all examples
            consistent_rules.append({
                'type': rule_type,
                'details': type_rules,
                'confidence': 1.0
            })
        else:
            # Rule appears in some examples
            consistent_rules.append({
                'type': rule_type,
                'details': type_rules,
                'confidence': len(type_rules) / len(task['train'])
            })
    
    return consistent_rules

# ============================================================================
# SOLUTION GENERATION
# ============================================================================

def apply_rule(grid, rule):
    """
    Apply a transformation rule to a grid.
    
    Args:
        grid (list): Input grid
        rule (dict): Rule to apply
        
    Returns:
        list: Transformed grid
    """
    if rule['type'] == 'rotation':
        return rotate_grid(grid, rule['details'][0]['steps'])
    
    elif rule['type'] == 'flip_horizontal':
        return flip_grid_horizontal(grid)
    
    elif rule['type'] == 'flip_vertical':
        return flip_grid_vertical(grid)
    
    elif rule['type'] == 'color_mapping':
        # Use the mapping from the first example
        mapping = rule['details'][0]['mapping']
        return color_remap(grid, mapping)
    
    elif rule['type'] == 'resize':
        # Get the resize ratio from the first example
        changes = rule['details'][0]['changes']
        h_ratio = changes['height_ratio']
        w_ratio = changes['width_ratio']
        
        # Simple resizing for integer ratios
        if abs(h_ratio - round(h_ratio)) < 0.01 and abs(w_ratio - round(w_ratio)) < 0.01:
            h_ratio = int(round(h_ratio))
            w_ratio = int(round(w_ratio))
            
            in_height, in_width = get_grid_dimensions(grid)
            out_height, out_width = in_height * h_ratio, in_width * w_ratio
            
            # Create a new grid with the right size
            result = [[0 for _ in range(out_width)] for _ in range(out_height)]
            
            # Fill in the result based on the resize type
            for i in range(out_height):
                for j in range(out_width):
                    # Map to the original grid
                    orig_i = i // h_ratio
                    orig_j = j // w_ratio
                    
                    # Copy the value
                    result[i][j] = grid[orig_i][orig_j]
            
            return result
    
    # If no transformation succeeded or rule type not recognized, return original grid
    return grid

def generate_solutions(task, max_solutions=2):
    """
    Generate potential solutions for a task.
    
    Args:
        task (dict): Task dictionary
        max_solutions (int): Maximum number of solutions to generate
        
    Returns:
        list: List of potential solution grids
    """
    # Extract rules from examples
    rules = extract_rules_from_examples(task)
    
    # Sort rules by confidence
    rules.sort(key=lambda r: r['confidence'], reverse=True)
    
    # Get the test input
    if isinstance(task['test'], list):
        test_inputs = [t['input'] for t in task['test']]
    else:
        test_inputs = [task['test']['input']]
    
    all_solutions = []
    
    # Generate solutions for each test input
    for test_input in test_inputs:
        solutions = []
        
        # Try applying high-confidence rules first
        for rule in rules[:min(len(rules), 10)]:  # Consider only top 10 rules
            solution = apply_rule(test_input, rule)
            solutions.append(solution)
        
        # If we haven't got enough solutions yet, try combinations of rules
        if len(solutions) < max_solutions and len(rules) > 1:
            for r1, r2 in itertools.combinations(rules[:min(len(rules), 5)], 2):
                # Apply rules in sequence
                intermediate = apply_rule(test_input, r1)
                solution = apply_rule(intermediate, r2)
                solutions.append(solution)
        
        # If we still need more solutions, add some simple heuristics
        if len(solutions) < max_solutions:
            # Try common transformations not captured by rules
            solutions.append(rotate_grid(test_input))  # 90-degree rotation
            solutions.append(flip_grid_horizontal(test_input))  # Horizontal flip
        
        # Ensure we have at least max_solutions
        if len(solutions) < max_solutions:
            # Just duplicate our best solution
            solutions = solutions + [solutions[0]] * (max_solutions - len(solutions))
        
        # Take only the requested number of solutions
        all_solutions.append(solutions[:max_solutions])
    
    return all_solutions

# ============================================================================
# ADVANCED PATTERN SOLVERS
# ============================================================================

def solve_symmetry_patterns(task):
    """
    Solve tasks that involve symmetry patterns.
    
    Args:
        task (dict): Task dictionary
        
    Returns:
        list: List of solution grids (for each test input)
    """
    solutions = []
    
    for train_pair in task['train']:
        input_grid = grid_to_numpy(train_pair['input'])
        output_grid = grid_to_numpy(train_pair['output'])
        
        # Check for horizontal symmetry completion
        h_sym = np.all(input_grid == np.fliplr(input_grid))
        v_sym = np.all(input_grid == np.flipud(input_grid))
        
        if h_sym and v_sym:
            # Task might involve completing symmetry
            solutions.append({'type': 'both_symmetry', 'confidence': 0.9})
        elif h_sym:
            solutions.append({'type': 'horizontal_symmetry', 'confidence': 0.9})
        elif v_sym:
            solutions.append({'type': 'vertical_symmetry', 'confidence': 0.9})
        
        # Check for partial symmetry completion
        h_match_ratio = np.mean(input_grid == np.fliplr(input_grid))
        v_match_ratio = np.mean(input_grid == np.flipud(input_grid))
        
        if h_match_ratio > 0.8:
            solutions.append({'type': 'partial_horizontal_symmetry', 'confidence': h_match_ratio})
        if v_match_ratio > 0.8:
            solutions.append({'type': 'partial_vertical_symmetry', 'confidence': v_match_ratio})
    
    # Sort solutions by confidence
    solutions.sort(key=lambda s: s['confidence'], reverse=True)
    
    if not solutions:
        # No symmetry patterns found
        return []
    
    # Try to apply the most confident symmetry pattern
    pattern = solutions[0]['type']
    test_inputs = task['test']['input'] if not isinstance(task['test'], list) else [t['input'] for t in task['test']]
    
    results = []
    for test_input in test_inputs:
        grid = grid_to_numpy(test_input)
        
        if pattern == 'horizontal_symmetry' or pattern == 'both_symmetry':
            # Complete horizontal symmetry
            result = grid.copy()
            for i in range(len(grid)):
                for j in range(len(grid[0]) // 2):
                    result[i, -(j+1)] = result[i, j]
        
        elif pattern == 'vertical_symmetry' or pattern == 'both_symmetry':
            # Complete vertical symmetry
            result = grid.copy()
            for i in range(len(grid) // 2):
                for j in range(len(grid[0])):
                    result[-(i+1), j] = result[i, j]
        
        elif pattern == 'partial_horizontal_symmetry':
            # Complete partial horizontal symmetry
            result = grid.copy()
            for i in range(len(grid)):
                for j in range(len(grid[0]) // 2):
                    if random.random() < 0.8:  # Probabilistic completion
                        result[i, -(j+1)] = result[i, j]
        
        elif pattern == 'partial_vertical_symmetry':
            # Complete partial vertical symmetry
            result = grid.copy()
            for i in range(len(grid) // 2):
                for j in range(len(grid[0])):
                    if random.random() < 0.8:  # Probabilistic completion
                        result[-(i+1), j] = result[i, j]
        
        results.append(numpy_to_grid(result))
    
    return results

def solve_pattern_completion(task):
    """
    Solve tasks that involve completing a pattern.
    
    Args:
        task (dict): Task dictionary
        
    Returns:
        list: List of solution grids (for each test input)
    """
    solutions = []
    
    # Get all train pairs
    train_pairs = task['train']
    
    # Check if all output grids have the same dimensions
    output_dims = [get_grid_dimensions(pair['output']) for pair in train_pairs]
    if len(set(output_dims)) == 1:
        # If all outputs have the same dimensions, try to identify a pattern
        # This is useful for tasks where the output follows a template
        
        # Count the frequency of each color at each position across all outputs
        h, w = output_dims[0]
        position_colors = defaultdict(Counter)
        
        for pair in train_pairs:
            output = pair['output']
            for i in range(h):
                for j in range(w):
                    position_colors[(i, j)][output[i][j]] += 1
        
        # Create a template with the most common color for each position
        template = [[0 for _ in range(w)] for _ in range(h)]
        for (i, j), counter in position_colors.items():
            most_common = counter.most_common(1)[0][0]  # Most common color
            template[i][j] = most_common
        
        # Add the template as a solution
        solutions.append({'type': 'template', 'grid': template, 'confidence': 0.8})
    
    # Sort solutions by confidence
    solutions.sort(key=lambda s: s['confidence'], reverse=True)
    
    if not solutions:
        # No pattern completion identified
        return []
    
    # Try to apply the most confident pattern
    pattern = solutions[0]
    
    if pattern['type'] == 'template':
        # Just return the template for each test input
        test_inputs = task['test']['input'] if not isinstance(task['test'], list) else [t['input'] for t in task['test']]
        return [pattern['grid'] for _ in test_inputs]
    
    # If no pattern works, return empty list
    return []

# ============================================================================
# MAIN SOLVER
# ============================================================================

def solve_task(task, task_id=None, visualize=False):
    """
    Attempt to solve an ARC task.
    
    Args:
        task (dict): Task dictionary
        task_id (str, optional): Task ID for visualization
        visualize (bool): Whether to visualize the task and solutions
        
    Returns:
        list: List of solutions (two attempts for each test input)
    """
    # Visualize the task if requested
    if visualize:
        visualize_task(task, task_id)
    
    # Generate solutions using different approaches
    general_solutions = generate_solutions(task, max_solutions=1)
    symmetry_solutions = solve_symmetry_patterns(task)
    pattern_solutions = solve_pattern_completion(task)
    
    # Use advanced solvers
    cellular_automaton_solutions = solve_cellular_automaton(task)
    object_arithmetic_solutions = solve_object_arithmetic(task)
    pattern_extension_solutions = solve_pattern_extension(task)
    
    # Normalize task test format
    if not isinstance(task['test'], list):
        task_test = [task['test']]
    else:
        task_test = task['test']
    
    # For each test input, generate two solution attempts
    all_solutions = []
    for i, _ in enumerate(task_test):
        solutions = []
        
        # Gather all solutions from different approaches
        solution_candidates = []
        
        # Check all solvers and add their solutions if available
        if i < len(general_solutions) and general_solutions[i] and len(general_solutions[i]) > 0:
            solution_candidates.append(('general', general_solutions[i][0]))
        
        if symmetry_solutions and i < len(symmetry_solutions):
            solution_candidates.append(('symmetry', symmetry_solutions[i]))
        
        if pattern_solutions and i < len(pattern_solutions):
            solution_candidates.append(('pattern', pattern_solutions[i]))
            
        if cellular_automaton_solutions and i < len(cellular_automaton_solutions):
            solution_candidates.append(('cellular_automaton', cellular_automaton_solutions[i]))
            
        if object_arithmetic_solutions and i < len(object_arithmetic_solutions):
            solution_candidates.append(('object_arithmetic', object_arithmetic_solutions[i]))
            
        if pattern_extension_solutions and i < len(pattern_extension_solutions):
            solution_candidates.append(('pattern_extension', pattern_extension_solutions[i]))
        
        # Prioritize solutions:
        # 1. The more specialized the solver, the better
        # 2. If multiple solutions, use different approaches for diversity
        solver_priority = {
            'pattern_extension': 5,     # Most specific pattern
            'cellular_automaton': 4,    # Complex rule-based system
            'object_arithmetic': 3,     # Mathematical operations
            'symmetry': 2,              # Geometric properties
            'pattern': 1,               # General pattern completion
            'general': 0                # General transformations
        }
        
        # Sort by priority
        solution_candidates.sort(key=lambda x: solver_priority[x[0]], reverse=True)
        
        # Take the top 2 solutions from different approaches
        used_approaches = set()
        for approach, sol in solution_candidates:
            if len(solutions) < 2 and approach not in used_approaches:
                solutions.append(sol)
                used_approaches.add(approach)
        
        # If we still need more solutions, take remaining candidates
        remaining = [sol for _, sol in solution_candidates if len(solutions) < 2]
        solutions.extend(remaining[:2 - len(solutions)])
        
        # Make sure we have exactly 2 solution attempts
        if len(solutions) == 0:
            # If no solutions, create a default solution (copy the input)
            input_grid = task_test[i]['input']
            solutions = [input_grid, input_grid]
        elif len(solutions) == 1:
            # If only one solution, duplicate it
            solutions = [solutions[0], solutions[0]]
        
        all_solutions.append({
            "attempt_1": solutions[0],
            "attempt_2": solutions[1]
        })
    
    return all_solutions

def create_submission(test_tasks, output_file='submission.json'):
    """
    Create a submission file for the competition.
    
    Args:
        test_tasks (dict): Dictionary of test tasks
        output_file (str): Output file path
    """
    submission = {}
    
    for task_id, task in tqdm(test_tasks.items(), desc="Solving tasks"):
        solutions = solve_task(task, task_id)
        submission[task_id] = solutions
    
    # Write submission to file
    with open(output_file, 'w') as f:
        json.dump(submission, f)
    
    print(f"Submission saved to {output_file}")

# ============================================================================
# ADDITIONAL ADVANCED SOLVERS
# ============================================================================

def solve_cellular_automaton(task):
    """
    Attempt to solve tasks that follow cellular automaton rules.
    
    Args:
        task (dict): Task dictionary
        
    Returns:
        list: List of solution grids (for each test input)
    """
    solutions = []
    
    # Look for evidence of CA-like behavior in train examples
    for pair in task['train']:
        input_grid = grid_to_numpy(pair['input'])
        output_grid = grid_to_numpy(pair['output'])
        
        # Check if dimensions match (cellular automata typically maintain grid size)
        if input_grid.shape != output_grid.shape:
            continue
            
        # Look for local neighborhood transformations
        # For simplicity, we'll check if each output cell depends on its input neighbors
        
        height, width = input_grid.shape
        neighborhood_patterns = defaultdict(list)
        
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                # Extract 3x3 neighborhood
                neighborhood = input_grid[i-1:i+2, j-1:j+2].flatten()
                neighborhood_key = tuple(neighborhood)
                center_output = output_grid[i, j]
                
                # Record this pattern
                neighborhood_patterns[neighborhood_key].append(center_output)
        
        # Check if transformation is consistent (same neighborhood -> same output)
        ca_consistent = True
        for neighborhood, outputs in neighborhood_patterns.items():
            if len(set(outputs)) > 1:
                ca_consistent = False
                break
        
        if ca_consistent and len(neighborhood_patterns) > 0:
            # Found potential CA pattern
            rules = {n: outputs[0] for n, outputs in neighborhood_patterns.items()}
            solutions.append({'type': 'ca', 'rules': rules, 'confidence': 0.85})
    
    if not solutions:
        return []
        
    # Apply the highest confidence CA rule
    solution = solutions[0]
    test_inputs = task['test']['input'] if not isinstance(task['test'], list) else [t['input'] for t in task['test']]
    
    results = []
    for test_input in test_inputs:
        input_arr = grid_to_numpy(test_input)
        height, width = input_arr.shape
        output_arr = np.zeros_like(input_arr)
        
        # Apply CA rules
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                neighborhood = input_arr[i-1:i+2, j-1:j+2].flatten()
                neighborhood_key = tuple(neighborhood)
                
                # If we know this pattern, apply the rule
                if neighborhood_key in solution['rules']:
                    output_arr[i, j] = solution['rules'][neighborhood_key]
                else:
                    # Otherwise keep the same value (identity rule)
                    output_arr[i, j] = input_arr[i, j]
        
        # For border cells (no complete neighborhood), keep original value
        output_arr[0, :] = input_arr[0, :]
        output_arr[-1, :] = input_arr[-1, :]
        output_arr[:, 0] = input_arr[:, 0]
        output_arr[:, -1] = input_arr[:, -1]
        
        results.append(numpy_to_grid(output_arr))
    
    return results

def solve_object_arithmetic(task):
    """
    Attempt to solve tasks involving arithmetic operations on object counts.
    
    Args:
        task (dict): Task dictionary
        
    Returns:
        list: List of solution grids (for each test input)
    """
    operations = []
    
    # Check if there's a pattern in the number of objects
    for pair in task['train']:
        input_counts = count_objects_by_color(pair['input'])
        output_counts = count_objects_by_color(pair['output'])
        
        # Check addition/subtraction
        for color in set(input_counts.keys()) | set(output_counts.keys()):
            in_count = input_counts.get(color, 0)
            out_count = output_counts.get(color, 0)
            
            # Addition
            for other_color in input_counts.keys():
                if other_color != color and in_count + input_counts[other_color] == out_count:
                    operations.append({
                        'type': 'add',
                        'color1': color,
                        'color2': other_color,
                        'confidence': 0.8
                    })
            
            # Subtraction
            for other_color in input_counts.keys():
                if other_color != color and in_count - input_counts[other_color] == out_count:
                    operations.append({
                        'type': 'subtract',
                        'color1': color,
                        'color2': other_color,
                        'confidence': 0.8
                    })
            
            # Multiplication
            for other_color in input_counts.keys():
                if other_color != color and in_count * input_counts[other_color] == out_count:
                    operations.append({
                        'type': 'multiply',
                        'color1': color,
                        'color2': other_color,
                        'confidence': 0.8
                    })
            
            # Division
            for other_color in input_counts.keys():
                if other_color != color and input_counts[other_color] != 0 and in_count / input_counts[other_color] == out_count:
                    operations.append({
                        'type': 'divide',
                        'color1': color,
                        'color2': other_color,
                        'confidence': 0.8
                    })
    
    # Count operation types and select the most common
    op_counter = Counter([op['type'] for op in operations])
    if not op_counter:
        return []
    
    most_common_op = op_counter.most_common(1)[0][0]
    relevant_ops = [op for op in operations if op['type'] == most_common_op]
    
    # Group by color pairs
    color_pairs = defaultdict(list)
    for op in relevant_ops:
        color_pairs[(op['color1'], op['color2'])].append(op)
    
    # Select the most frequent color pair
    if not color_pairs:
        return []
    
    most_common_pair = max(color_pairs.items(), key=lambda x: len(x[1]))[0]
    color1, color2 = most_common_pair
    
    # Apply the operation to test inputs
    test_inputs = task['test']['input'] if not isinstance(task['test'], list) else [t['input'] for t in task['test']]
    
    results = []
    for test_input in test_inputs:
        input_counts = count_objects_by_color(test_input)
        count1 = input_counts.get(color1, 0)
        count2 = input_counts.get(color2, 0)
        
        # Compute result based on operation
        if most_common_op == 'add':
            result_count = count1 + count2
        elif most_common_op == 'subtract':
            result_count = count1 - count2
        elif most_common_op == 'multiply':
            result_count = count1 * count2
        elif most_common_op == 'divide':
            result_count = count1 // count2 if count2 != 0 else 0
        
        # Create a grid with the right number of objects
        # For simplicity, we'll use a grid of the same size as input with result_count cells of color1
        input_arr = grid_to_numpy(test_input)
        result_arr = np.zeros_like(input_arr)
        
        # Fill in the result count
        total_cells = input_arr.size
        cells_to_fill = min(result_count, total_cells)
        
        if cells_to_fill > 0:
            flat_indices = np.random.choice(total_cells, cells_to_fill, replace=False)
            rows, cols = np.unravel_index(flat_indices, input_arr.shape)
            result_arr[rows, cols] = color1
        
        results.append(numpy_to_grid(result_arr))
    
    return results

def solve_pattern_extension(task):
    """
    Attempt to solve tasks involving extending patterns.
    
    Args:
        task (dict): Task dictionary
        
    Returns:
        list: List of solution grids (for each test input)
    """
    # Look for sequence patterns in train examples
    all_patterns = []
    
    for pair in task['train']:
        input_grid = grid_to_numpy(pair['input'])
        output_grid = grid_to_numpy(pair['output'])
        
        # Check if output extends input horizontally
        if input_grid.shape[0] == output_grid.shape[0] and input_grid.shape[1] < output_grid.shape[1]:
            # Check if the left part of output matches input
            if np.array_equal(output_grid[:, :input_grid.shape[1]], input_grid):
                # Extract the extension pattern
                extension = output_grid[:, input_grid.shape[1]:]
                all_patterns.append({
                    'type': 'horizontal_extension',
                    'pattern': extension,
                    'confidence': 0.85
                })
        
        # Check if output extends input vertically
        elif input_grid.shape[1] == output_grid.shape[1] and input_grid.shape[0] < output_grid.shape[0]:
            # Check if the top part of output matches input
            if np.array_equal(output_grid[:input_grid.shape[0], :], input_grid):
                # Extract the extension pattern
                extension = output_grid[input_grid.shape[0]:, :]
                all_patterns.append({
                    'type': 'vertical_extension',
                    'pattern': extension,
                    'confidence': 0.85
                })
    
    if not all_patterns:
        return []
    
    # Use the pattern with highest confidence
    pattern = max(all_patterns, key=lambda p: p['confidence'])
    
    # Apply pattern to test inputs
    test_inputs = task['test']['input'] if not isinstance(task['test'], list) else [t['input'] for t in task['test']]
    
    results = []
    for test_input in test_inputs:
        input_arr = grid_to_numpy(test_input)
        
        if pattern['type'] == 'horizontal_extension':
            # Extend horizontally
            extension = pattern['pattern']
            result = np.hstack((input_arr, extension))
        elif pattern['type'] == 'vertical_extension':
            # Extend vertically
            extension = pattern['pattern']
            result = np.vstack((input_arr, extension))
        else:
            # Unknown pattern type, return input
            result = input_arr
        
        results.append(numpy_to_grid(result))
    
    return results

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """Main function to process and solve ARC tasks."""
    # Load challenge data
    train_challenges = load_data('/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json')
    train_solutions = load_data('/kaggle/input/arc-prize-2025/arc-agi_training_solutions.json')
    eval_challenges = load_data('/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json')
    eval_solutions = load_data('/kaggle/input/arc-prize-2025/arc-agi_evaluation_solutions.json')
    test_challenges = load_data('/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json')
    
    # Visualize a sample task
    sample_task_id = list(train_challenges.keys())[0]
    sample_task = train_challenges[sample_task_id]
    print(f"Visualizing sample task {sample_task_id}")
    visualize_task(sample_task, sample_task_id)
    
    # Create submission for test tasks
    print("Creating submission for test tasks...")
    create_submission(test_challenges)
    
    print("Done!")

if __name__ == "__main__":
    main()




