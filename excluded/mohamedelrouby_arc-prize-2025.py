# Import common libraries for data analysis and machine learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')


import json
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

class ARCAGILoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = self._load_json()
        self.tasks = self._parse_tasks()

    def _load_json(self) -> Any:
        with open(self.file_path, 'r') as f:
            return json.load(f)

    def _parse_tasks(self) -> List[Dict[str, Any]]:
        # Handles both list of tasks and dict of tasks
        if isinstance(self.data, list):
            return self.data
        elif isinstance(self.data, dict):
            # Some files may have a dict with task ids as keys
            return list(self.data.values())
        else:
            raise ValueError('Unknown ARC-AGI JSON structure')

    def get_task(self, idx: int) -> Dict[str, Any]:
        return self.tasks[idx]

    def get_train_pairs(self, idx: int) -> List[Dict[str, np.ndarray]]:
        task = self.get_task(idx)
        train_pairs = []
        for pair in task['train']:
            train_pairs.append({
                'input': np.array(pair['input']),
                'output': np.array(pair['output'])
            })
        return train_pairs

    def get_test_inputs(self, idx: int) -> List[np.ndarray]:
        task = self.get_task(idx)
        return [np.array(pair['input']) for pair in task['test']]

def plot_grid(grid: np.ndarray, ax=None, title=None, show_values=False, dpi=100, cell_size=0.5):
    """
    Enhanced grid visualization with optional cell value annotations and better scaling.
    
    Args:
        grid: numpy array of integers representing the grid
        ax: matplotlib axis to plot on (optional)
        title: title for the plot (optional)
        show_values: whether to show numerical values in cells (optional)
        dpi: resolution of the plot (optional)
        cell_size: size of each cell in inches (optional)
    """
    # Define a color map for up to 10 colors (0-9)
    colors = ['black', 'blue', 'red', 'green', 'yellow', 'gray', 'magenta', 'cyan', 'orange', 'white']
    cmap = mcolors.ListedColormap(colors[:max(np.max(grid) + 1, len(colors))])
    norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, np.max(grid)+1.5), ncolors=np.max(grid)+2)
    
    if ax is None:
        # Calculate figure size based on grid dimensions and cell size
        figsize = (cell_size * grid.shape[1], cell_size * grid.shape[0])
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
    # Plot the grid with sharp edges
    im = ax.imshow(grid, cmap=cmap, norm=norm, interpolation='nearest', aspect='equal')
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='gray', linewidth=0.8, alpha=0.5)
    
    # Remove regular ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add cell values if requested
    if show_values:
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                val = grid[i, j]
                # Choose text color based on background brightness
                cell_color = colors[val] if val < len(colors) else 'white'
                text_color = 'white' if cell_color in ['black', 'blue', 'red', 'green', 'magenta'] else 'black'
                ax.text(j, i, str(val), ha='center', va='center', color=text_color)
    
    if title:
        ax.set_title(title)
    
    # Add thin border around the grid
    ax.set_frame_on(True)
    ax.spines['top'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(True)
    
    return ax

# Example usage (uncomment and set file path to use):
# loader = ARCAGILoader('arc-agi_training_challenges.json')
# train_pairs = loader.get_train_pairs(0)
# plot_grid(train_pairs[0]['input'], show_values=True)
# plt.show()


from scipy.ndimage import label, find_objects, center_of_mass
from collections import defaultdict

# Extract connected components and features from a grid
def extract_objects(grid: np.ndarray):
    objects = []
    color_masks = {}
    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    for color in np.unique(grid):
        if color == 0:
            continue  # skip background if desired
        mask = (grid == color)
        labeled, num_features = label(mask, structure=structure)
        slices = find_objects(labeled)
        for i, slc in enumerate(slices):
            if slc is None:
                continue
            obj_mask = (labeled[slc] == (i+1))
            area = np.sum(obj_mask)
            bbox = (slc[0].start, slc[0].stop, slc[1].start, slc[1].stop)  # (row_start, row_end, col_start, col_end)
            centroid = center_of_mass(obj_mask)
            # Convert centroid to grid coordinates
            centroid = (centroid[0] + slc[0].start, centroid[1] + slc[1].start)
            shape = obj_mask.shape
            # Adjacency: check border pixels for touching other colors
            adj_colors = set()
            border_coords = np.argwhere(obj_mask)
            for r, c in border_coords:
                gr, gc = r + slc[0].start, c + slc[1].start
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = gr + dr, gc + dc
                        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                            neighbor_color = grid[nr, nc]
                            if neighbor_color != color:
                                adj_colors.add(int(neighbor_color))
            obj = {
                'color': int(color),
                'area': int(area),
                'bbox': bbox,  # (row_start, row_end, col_start, col_end)
                'centroid': centroid,
                'shape': shape,
                'adjacent_colors': list(adj_colors)
            }
            objects.append(obj)
    return objects

# Example usage:
# grid = train_pairs[0]['input']
# objects = extract_objects(grid)
# for obj in objects:
#     print(obj)


import copy

def recolor(grid, from_color, to_color):
    new_grid = grid.copy()
    new_grid[grid == from_color] = to_color
    return new_grid

def fill_region(grid, region_mask, color):
    new_grid = grid.copy()
    new_grid[region_mask] = color
    return new_grid

def copy_region(grid, region_mask):
    coords = np.argwhere(region_mask)
    if coords.size == 0:
        return None
    minr, minc = coords.min(axis=0)
    maxr, maxc = coords.max(axis=0)
    region = grid[minr:maxr+1, minc:maxc+1] * region_mask[minr:maxr+1, minc:maxc+1]
    return region, (minr, minc)

def paste_region(grid, region, top_left):
    new_grid = grid.copy()
    r0, c0 = top_left
    r1, c1 = r0 + region.shape[0], c0 + region.shape[1]
    new_grid[r0:r1, c0:c1] = region
    return new_grid

def translate_region(grid, region_mask, dx, dy):
    new_grid = grid.copy()
    coords = np.argwhere(region_mask)
    for r, c in coords:
        nr, nc = r + dy, c + dx
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            new_grid[nr, nc] = grid[r, c]
            new_grid[r, c] = 0
    return new_grid

def rotate_region(grid, region_mask, k=1):
    coords = np.argwhere(region_mask)
    if coords.size == 0:
        return grid.copy()
    minr, minc = coords.min(axis=0)
    maxr, maxc = coords.max(axis=0)
    region = grid[minr:maxr+1, minc:maxc+1]
    mask = region_mask[minr:maxr+1, minc:maxc+1]
    region_rot = np.rot90(region, k)
    mask_rot = np.rot90(mask, k)
    new_grid = grid.copy()
    new_grid[minr:maxr+1, minc:maxc+1][mask_rot] = region_rot[mask_rot]
    return new_grid

def mirror_region(grid, region_mask, axis=0):
    coords = np.argwhere(region_mask)
    if coords.size == 0:
        return grid.copy()
    minr, minc = coords.min(axis=0)
    maxr, maxc = coords.max(axis=0)
    region = grid[minr:maxr+1, minc:maxc+1]
    mask = region_mask[minr:maxr+1, minc:maxc+1]
    if axis == 0:
        region_mir = np.flipud(region)
        mask_mir = np.flipud(mask)
    else:
        region_mir = np.fliplr(region)
        mask_mir = np.fliplr(mask)
    new_grid = grid.copy()
    new_grid[minr:maxr+1, minc:maxc+1][mask_mir] = region_mir[mask_mir]
    return new_grid

def remove_isolated(grid, color, threshold=1):
    from scipy.ndimage import label
    mask = (grid == color)
    labeled, num = label(mask)
    new_grid = grid.copy()
    for i in range(1, num+1):
        region = (labeled == i)
        if np.sum(region) < threshold:
            new_grid[region] = 0
    return new_grid

def paint_border(grid, region_mask, color):
    new_grid = grid.copy()
    coords = np.argwhere(region_mask)
    for r, c in coords:
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                if not region_mask[nr, nc]:
                    new_grid[r, c] = color
    return new_grid

# Unit tests
def print_grid(g):
    print("\n".join([" ".join(map(str, row)) for row in g]))
    print()

def test_grid_primitives():
    grid = np.array([
        [1,1,0,0],
        [1,0,2,2],
        [0,0,2,0],
        [3,3,0,0]
    ])
    print('Original:'); print_grid(grid)
    # recolor
    print('Recolor 1->4:'); print_grid(recolor(grid, 1, 4))
    # fill region
    mask = (grid == 2)
    print('Fill region 2 with 5:'); print_grid(fill_region(grid, mask, 5))
    # copy/paste
    region, (r0, c0) = copy_region(grid, grid==1)
    grid2 = paste_region(grid, region, (2,2))
    print('Copy/paste region 1:'); print_grid(grid2)
    # translate
    print('Translate region 2 by (1,1):'); print_grid(translate_region(grid, grid==2, 1, 1))
    # rotate
    print('Rotate region 2 by 90:'); print_grid(rotate_region(grid, grid==2, 1))
    # mirror
    print('Mirror region 2 vertically:'); print_grid(mirror_region(grid, grid==2, axis=0))
    # remove isolated
    grid3 = grid.copy()
    grid3[0,2] = 2
    print('Remove isolated 2s <2:'); print_grid(remove_isolated(grid3, 2, 2))
    # paint border
    print('Paint border of region 1 with 9:'); print_grid(paint_border(grid, grid==1, 9))

test_grid_primitives()


def pixel_diff(input_grid, output_grid):
    diff_mask = (input_grid != output_grid)
    changes = []
    for (r, c), changed in np.ndenumerate(diff_mask):
        if changed:
            changes.append({
                'pos': (r, c),
                'from': int(input_grid[r, c]),
                'to': int(output_grid[r, c])
            })
    return changes

def summarize_pixel_changes(changes):
    summary = {}
    for ch in changes:
        key = (ch['from'], ch['to'])
        summary[key] = summary.get(key, 0) + 1
    return summary

def object_diff(input_grid, output_grid):
    input_objs = extract_objects(input_grid)
    output_objs = extract_objects(output_grid)
    # Simple matching by color and area (could be improved)
    matched = []
    unmatched_input = input_objs.copy()
    unmatched_output = output_objs.copy()
    for in_obj in input_objs:
        for out_obj in output_objs:
            if in_obj['color'] == out_obj['color'] and in_obj['area'] == out_obj['area']:
                matched.append((in_obj, out_obj))
                unmatched_input.remove(in_obj)
                unmatched_output.remove(out_obj)
                break
    return matched, unmatched_input, unmatched_output

def extract_candidate_rules(input_grid, output_grid):
    rules = []
    # Pixel-level color changes
    changes = pixel_diff(input_grid, output_grid)
    color_changes = summarize_pixel_changes(changes)
    for (from_col, to_col), count in color_changes.items():
        if from_col != to_col:
            rules.append(f"Recolor all pixels {from_col} -> {to_col} ({count} pixels)")
    # Object-level changes
    matched, removed, added = object_diff(input_grid, output_grid)
    for obj in removed:
        rules.append(f"Object of color {obj['color']} removed (area {obj['area']})")
    for obj in added:
        rules.append(f"Object of color {obj['color']} added (area {obj['area']})")
    # Check for movement/rotation in matched objects
    for in_obj, out_obj in matched:
        if in_obj['centroid'] != out_obj['centroid']:
            rules.append(f"Object of color {in_obj['color']} moved from {in_obj['centroid']} to {out_obj['centroid']}")
        if in_obj['shape'] == out_obj['shape'] and in_obj['bbox'] != out_obj['bbox']:
            rules.append(f"Object of color {in_obj['color']} translated")
        # Check for rotation (simple: shape swap)
        if in_obj['shape'] == out_obj['shape'][::-1]:
            rules.append(f"Object of color {in_obj['color']} rotated 90 degrees")
    return rules

# Example usage:
# input_grid = np.array([[1,2],[3,4]])
# output_grid = np.array([[1,2],[4,3]])
# print(extract_candidate_rules(input_grid, output_grid))


import itertools

# Define a set of primitive operations as callables with parameter options
def get_primitive_ops():
    return [
        ('recolor', lambda grid: [recolor(grid, c1, c2) for c1 in range(10) for c2 in range(10) if c1 != c2]),
        # Add more primitives as needed, e.g., fill, rotate, etc.
    ]

def apply_program(grid, program):
    g = grid.copy()
    for op, params in program:
        g = op(g, *params)
    return g

# For demonstration, only recolor is used. Expand as needed.
def enumerate_programs(max_length=2):
    # Each program is a sequence of (op, params)
    ops = []
    for c1 in range(10):
        for c2 in range(10):
            if c1 != c2:
                ops.append((recolor, (c1, c2)))
    # Enumerate all sequences up to max_length
    for l in range(1, max_length+1):
        for seq in itertools.product(ops, repeat=l):
            yield seq

def program_succeeds_on_all(program, train_pairs):
    for pair in train_pairs:
        input_grid = pair['input']
        output_grid = pair['output']
        g = input_grid.copy()
        for op, params in program:
            g = op(g, *params)
        if not np.array_equal(g, output_grid):
            return False
    return True

def score_program(program):
    # Simpler (shorter) programs are better
    return len(program)

def synthesize_program(train_pairs, max_length=2):
    best_program = None
    best_score = float('inf')
    for program in enumerate_programs(max_length):
        if program_succeeds_on_all(program, train_pairs):
            s = score_program(program)
            if s < best_score:
                best_score = s
                best_program = program
    return best_program

# Example usage:
# train_pairs = loader.get_train_pairs(0)
# best_prog = synthesize_program(train_pairs, max_length=2)
# print(best_prog)


import heapq
from typing import List, Tuple, Any, Callable
import numpy as np

def beam_search_synthesis(
    train_pairs: List[Dict[str, np.ndarray]], 
    primitives: List[Tuple[Callable, Tuple[Any, ...]]],
    max_length: int = 3,
    beam_width: int = 10,
    early_stop_threshold: float = 0.95
) -> Tuple[List[Tuple[Callable, Tuple[Any, ...]]], float]:
    """
    Enhanced beam search synthesis with early stopping and better scoring.
    
    Args:
        train_pairs: List of input-output pairs for training
        primitives: List of (operation, parameters) tuples
        max_length: Maximum program length to consider
        beam_width: Number of candidates to keep at each step
        early_stop_threshold: Stop if we find a solution with this accuracy
        
    Returns:
        best_program: List of (operation, parameters) tuples
        best_accuracy: Accuracy of the best program found
    """
    # Each candidate: (score, accuracy, program_sequence)
    beam = [(0, 0.0, [])]
    best_program = None
    best_score = float('inf')
    best_accuracy = 0.0

    for step in range(1, max_length + 1):
        candidates = []
        for score, acc, seq in beam:
            for op, params in primitives:
                new_seq = seq + [(op, params)]
                
                # Calculate accuracy and pixel-wise similarity
                matches = 0
                total_similarity = 0.0
                valid_program = True
                
                for pair in train_pairs:
                    try:
                        g = pair['input'].copy()
                        for o, p in new_seq:
                            g_new = o(g, *p)
                            if not isinstance(g_new, np.ndarray) or g_new.shape != g.shape:
                                valid_program = False
                                break
                            g = g_new
                            
                        if valid_program:
                            if np.array_equal(g, pair['output']):
                                matches += 1
                            # Calculate pixel-wise similarity
                            similarity = np.mean(g == pair['output'])
                            total_similarity += similarity
                            
                    except Exception:
                        valid_program = False
                        break
                
                if not valid_program:
                    continue
                
                accuracy = matches / len(train_pairs)
                avg_similarity = total_similarity / len(train_pairs)
                
                # Sophisticated scoring function
                # Prioritize:
                # 1. Number of exact matches (negative for heapq)
                # 2. Average pixel-wise similarity
                # 3. Program simplicity (shorter length)
                prog_score = -matches * 1000 - avg_similarity * 100 + len(new_seq)
                
                heapq.heappush(candidates, (prog_score, accuracy, new_seq))
                
                # Track best program
                if accuracy > best_accuracy or (accuracy == best_accuracy and prog_score < best_score):
                    best_score = prog_score
                    best_program = new_seq
                    best_accuracy = accuracy
                    
                    # Early stopping if we found a good enough solution
                    if accuracy >= early_stop_threshold:
                        return best_program, best_accuracy
        
        # Keep only top-K candidates
        beam = heapq.nsmallest(beam_width, candidates)
        
        # Stop if we can't improve
        if not beam:
            break
            
    return best_program, best_accuracy

# Example usage:
# primitives = [(recolor, (c1, c2)) for c1 in range(10) for c2 in range(10) if c1 != c2]
# train_pairs = loader.get_train_pairs(0)
# best_prog, accuracy = beam_search_synthesis(
#     train_pairs,
#     primitives,
#     max_length=2,
#     beam_width=5,
#     early_stop_threshold=0.95
# )
# print(f"Best program found with accuracy: {accuracy:.2f}")


def apply_program_to_test_inputs(program, test_inputs):
    outputs = []
    for test_grid in test_inputs:
        g = test_grid.copy()
        for op, params in program:
            g = op(g, *params)
        # Ensure output is a list of lists of ints
        outputs.append(g.tolist())
    return outputs

def create_submission(task_ids, test_outputs, filename='/kaggle/input/arc-prize-2025/sample_submission.json'):
    # Format: {task_id: [output1, output2, ...], ...}
    submission = {}
    for tid, outs in zip(task_ids, test_outputs):
        submission[tid] = outs
    with open(filename, 'w') as f:
        import json
        json.dump(submission, f)
    print(f'Submission saved to {filename}')
    return submission

# Example usage:
# loader = ARCAGILoader('arc-agi_test_challenges.json')
# test_inputs = loader.get_test_inputs(0)
# outputs = apply_program_to_test_inputs(best_prog, test_inputs)
# create_submission(['task_id_1'], [outputs])


def local_search_repair(candidate_grid, target_grid, max_steps=10):
    """
    Greedily apply local edits (recolor pixel, fill small region, translate pixel) to minimize pixel difference.
    Stops when no improvement or max_steps reached.
    """
    grid = candidate_grid.copy()
    best_diff = np.sum(grid != target_grid)
    for step in range(max_steps):
        improved = False
        # Try recoloring each differing pixel
        diff_coords = np.argwhere(grid != target_grid)
        for r, c in diff_coords:
            orig = grid[r, c]
            tgt = target_grid[r, c]
            if orig != tgt:
                grid[r, c] = tgt
                new_diff = np.sum(grid != target_grid)
                if new_diff < best_diff:
                    best_diff = new_diff
                    improved = True
                    break  # Greedy: accept first improvement
                else:
                    grid[r, c] = orig  # revert
        if improved:
            continue
        # Try filling small 2x2 regions
        for r, c in diff_coords:
            tgt = target_grid[r, c]
            r0, r1 = max(0, r-1), min(grid.shape[0], r+2)
            c0, c1 = max(0, c-1), min(grid.shape[1], c+2)
            region = grid[r0:r1, c0:c1].copy()
            grid[r0:r1, c0:c1] = tgt
            new_diff = np.sum(grid != target_grid)
            if new_diff < best_diff:
                best_diff = new_diff
                improved = True
                break
            else:
                grid[r0:r1, c0:c1] = region  # revert
        if improved:
            continue
        # Try translating a differing pixel to a nearby location
        for r, c in diff_coords:
            orig = grid[r, c]
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    if grid[nr, nc] != orig:
                        tmp = grid[nr, nc]
                        grid[nr, nc] = orig
                        grid[r, c] = 0
                        new_diff = np.sum(grid != target_grid)
                        if new_diff < best_diff:
                            best_diff = new_diff
                            improved = True
                            break
                        else:
                            grid[nr, nc] = tmp
                            grid[r, c] = orig
            if improved:
                break
        if not improved:
            break  # No further improvement
    return grid

# Example usage:
# repaired = local_search_repair(candidate_grid, target_grid)
# print(np.sum(repaired != target_grid))


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def compute_features(program, input_grid, output_grid, target_grid):
    # Features: program length, pixel similarity, structural similarity
    prog_len = len(program)
    pixel_sim = np.mean(output_grid == target_grid)
    # Structural similarity: number of objects matched (by color/area)
    in_objs = extract_objects(input_grid)
    out_objs = extract_objects(output_grid)
    tgt_objs = extract_objects(target_grid)
    matched = 0
    for o in out_objs:
        for t in tgt_objs:
            if o['color'] == t['color'] and o['area'] == t['area']:
                matched += 1
                break
    struct_sim = matched / max(1, len(tgt_objs))
    return [prog_len, pixel_sim, struct_sim]

def build_scorer(X, y):
    clf = MLPClassifier(hidden_layer_sizes=(16,), max_iter=200, random_state=42)
    clf.fit(X, y)
    return clf

def score_program_ml(clf, program, input_grid, output_grid, target_grid):
    feats = np.array(compute_features(program, input_grid, output_grid, target_grid)).reshape(1, -1)
    prob = clf.predict_proba(feats)[0,1]  # Probability of being correct
    return prob

# Example training data creation (for demonstration):
# X, y = [], []
# for each (program, input_grid, output_grid, target_grid, label):
#     X.append(compute_features(program, input_grid, output_grid, target_grid))
#     y.append(label)  # 1 if correct, 0 if not
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# clf = build_scorer(X_train, y_train)
# print('MLP scorer accuracy:', accuracy_score(y_test, clf.predict(X_test)))

# To rerank candidates: use score_program_ml(clf, ...) as part of the search/scoring phase


import time

def arc_agi_pipeline(
    train_file='/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json',
    test_file='/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json',
    submission_file='/kaggle/input/arc-prize-2025/sample_submission.json',
    beam_width=5,
    max_prog_length=2,
    use_local_repair=True,
    log_details=True
):
    # Load data
    train_loader = ARCAGILoader(train_file)
    test_loader = ARCAGILoader(test_file)
    # For training metrics
    train_metrics = []
    # For submission
    test_task_ids = []
    test_outputs = []
    total_start = time.time()
    
    for idx, test_task in enumerate(test_loader.tasks):
        task_id = test_task.get('id', str(idx))
        test_task_ids.append(task_id)
        # Try to find matching train task (by id or index)
        train_idx = idx if idx < len(train_loader.tasks) else 0
        train_pairs = train_loader.get_train_pairs(train_idx)
        
        # Synthesis (beam search)
        primitives = [(recolor, (c1, c2)) for c1 in range(10) for c2 in range(10) if c1 != c2]
        best_prog, prog_accuracy = beam_search_synthesis(
            train_pairs, 
            primitives, 
            max_length=max_prog_length, 
            beam_width=beam_width
        )
        
        # Evaluate on train
        train_acc = 0
        for pair in train_pairs:
            pred = pair['input'].copy()
            if best_prog is not None:
                try:
                    for op, params in best_prog:
                        pred_new = op(pred, *params)
                        if not isinstance(pred_new, np.ndarray) or pred_new.shape != pred.shape:
                            break
                        pred = pred_new
                    
                    if np.array_equal(pred, pair['output']):
                        train_acc += 1
                    elif use_local_repair and pred.shape == pair['output'].shape:
                        repaired = local_search_repair(pred, pair['output'])
                        if np.array_equal(repaired, pair['output']):
                            train_acc += 1
                except Exception as e:
                    if log_details:
                        print(f"Error in program execution: {str(e)}")
                    continue
            
        train_acc /= len(train_pairs)
        train_metrics.append(train_acc)
        
        # Predict on test inputs
        test_inputs = test_loader.get_test_inputs(idx)
        preds = []
        for test_grid in test_inputs:
            pred = test_grid.copy()
            if best_prog is not None:
                try:
                    for op, params in best_prog:
                        pred_new = op(pred, *params)
                        if not isinstance(pred_new, np.ndarray) or pred_new.shape != pred.shape:
                            break
                        pred = pred_new
                except Exception as e:
                    if log_details:
                        print(f"Error in test prediction: {str(e)}")
            preds.append(pred.tolist())
            
        test_outputs.append(preds)
        if log_details:
            print(f'Task {task_id}: train acc={train_acc:.2f}, synthesis acc={prog_accuracy:.2f}, program={best_prog}')
    
    # Save submission
    create_submission(test_task_ids, test_outputs, filename=submission_file)
    if log_details:
        print(f'Average train accuracy: {np.mean(train_metrics):.3f}')
        print(f'Total time: {time.time()-total_start:.1f}s')
    
    return test_task_ids, test_outputs, train_metrics

# Example usage:
# arc_agi_pipeline()


import numpy as np
import matplotlib.pyplot as plt
import tqdm
import logging
from contextlib import contextmanager
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# ---------------- Error Handler ----------------
@contextmanager
def plot_error_handler(title=""):
    try:
        yield
    except Exception as e:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', fontsize=10, wrap=True)
        plt.axis('off')
        if title:
            plt.title(title, pad=20)
        plt.show()

# ---------------- Helpers ----------------
def create_colormap_legend(ax, colors):
    """Create a color legend showing color indices"""
    legend_elements = [patches.Patch(facecolor=color, label=f'{i}')
                      for i, color in enumerate(colors)]
    ax.legend(handles=legend_elements, title='Color Index',
             loc='center left', bbox_to_anchor=(1, 0.5))

def add_grid_stats(ax, grid, transform):
    """Add statistical information about the grid"""
    stats = [
        f'Shape: {grid.shape}',
        f'Unique colors: {len(np.unique(grid))}',
        f'Max value: {np.max(grid)}',
        f'Non-zero: {np.count_nonzero(grid)}'
    ]
    y_pos = -0.1
    for stat in stats:
        ax.text(0.02, y_pos, stat, transform=transform, fontsize=8)
        y_pos -= 0.08

def plot_grid(grid, ax=None, title="", show_values=False, dpi=100, cell_size=1.0):
    """Simple grid plotter with auto-squeeze"""
    if ax is None:
        fig, ax = plt.subplots()
    
    grid = np.array(grid)
    if grid.ndim > 2:        # fix (1,2,2) or similar shapes
        grid = np.squeeze(grid)
    
    ax.imshow(grid, cmap="tab20", interpolation="nearest")
    if show_values:
        for (i, j), val in np.ndenumerate(grid):
            ax.text(j, i, int(val), ha="center", va="center", color="white", fontsize=6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    return ax

# ---------------- Dummy pipeline for demo ----------------
def arc_agi_pipeline(train_file, test_file, submission_file, **kwargs):
    """Mock pipeline (replace with real implementation)"""
    print("Running dummy arc_agi_pipeline...")
    task_ids = [0]
    # prediction: one test case with 2x2 grid (added batch dim to test squeeze)
    test_outputs = [[[np.array([[[1,2],[3,4]]])]]]  
    train_metrics = {"accuracy": 1.0}
    return task_ids, test_outputs, train_metrics

class ARCAGILoader:
    """Mock loader (replace with real implementation)"""
    def __init__(self, file):
        self.file = file

    def get_test_inputs(self, task_index):
        return [np.array([[[0,1],[2,3]]])]   # shape (1,2,2) for testing squeeze

    def get_train_pairs(self, task_index):
        return [
            {"input": np.array([[0,1],[1,0]]),
             "output": np.array([[1,0],[0,1]])}
        ]

def beam_search_synthesis(pairs, ops, max_length=2, beam_width=3, early_stop_threshold=0.95):
    """Mock synthesis"""
    return [("recolor", (0, 1))], 1.0

def recolor(grid, c1, c2):
    new_grid = grid.copy()
    new_grid[grid == c1] = c2
    return new_grid

# ---------------- Visualization ----------------
def visualize_arc_results(train_file, test_file, submission_file, task_index=0):
    print("Starting ARC visualization pipeline...")
    
    try:
        task_ids, test_outputs, train_metrics = arc_agi_pipeline(
            train_file=train_file,
            test_file=test_file,
            submission_file=submission_file,
            beam_width=3,
            max_prog_length=2,
            use_local_repair=True,
            log_details=True
        )
    except Exception as e:
        print("Pipeline failed:", e)
        return  # stop execution if pipeline fails

    # Define a consistent color palette
    colors = ['black', 'blue', 'red', 'green', 'yellow',
              'gray', 'magenta', 'cyan', 'orange', 'white']

    # Visualize test predictions
    print("\nVisualizing test predictions...")
    loader = ARCAGILoader(test_file)
    first_test_inputs = loader.get_test_inputs(task_index)
    first_pred_outputs = test_outputs[task_index]

    for i, (inp, pred) in enumerate(zip(first_test_inputs, first_pred_outputs)):
        with plot_error_handler(f"Test Case {i+1}"):
            fig = plt.figure(figsize=(15, 6))
            gs = GridSpec(1, 3, width_ratios=[4, 4, 1], figure=fig)
            
            ax1 = fig.add_subplot(gs[0])
            plot_grid(inp, ax=ax1, title=f'Test Input {i+1}', show_values=True, dpi=120, cell_size=0.8)
            add_grid_stats(ax1, inp, ax1.transAxes)
            
            ax2 = fig.add_subplot(gs[1])
            pred_array = np.array(pred)
            plot_grid(pred_array, ax=ax2, title=f'Predicted Output {i+1}', show_values=True, dpi=120, cell_size=0.8)
            add_grid_stats(ax2, pred_array, ax2.transAxes)
            
            ax3 = fig.add_subplot(gs[2])
            create_colormap_legend(ax3, colors)
            ax3.axis('off')
            
            plt.suptitle(f'Test Case {i+1} Analysis', fontsize=14, y=1.05)
            plt.tight_layout()
            plt.show()

    # Visualize training examples
    print("\nVisualizing training examples...")
    try:
        train_loader = ARCAGILoader(train_file)
        train_pairs = train_loader.get_train_pairs(task_index)
        
        for i, pair in enumerate(tqdm.tqdm(train_pairs, desc="Processing training pairs")):
            with plot_error_handler(f"Training Example {i+1}"):
                fig = plt.figure(figsize=(18, 7))
                gs = GridSpec(2, 3, height_ratios=[4, 1], figure=fig)
                
                ax1 = fig.add_subplot(gs[0, 0])
                plot_grid(pair['input'], ax=ax1, title=f'Train Input {i+1}', show_values=True, dpi=120, cell_size=0.8)
                add_grid_stats(ax1, pair['input'], ax1.transAxes)
                
                ax2 = fig.add_subplot(gs[0, 1])
                plot_grid(pair['output'], ax=ax2, title=f'Train Output {i+1}', show_values=True, dpi=120, cell_size=0.8)
                add_grid_stats(ax2, pair['output'], ax2.transAxes)
                
                ax_legend = fig.add_subplot(gs[0, 2])
                create_colormap_legend(ax_legend, colors)
                ax_legend.axis('off')
                
                best_prog, accuracy = beam_search_synthesis(
                    [pair],
                    [(recolor, (c1, c2)) for c1 in range(10) for c2 in range(10) if c1 != c2],
                    max_length=2,
                    beam_width=3,
                    early_stop_threshold=0.95
                )
                
                ax_analysis = fig.add_subplot(gs[1, :])
                ax_analysis.axis('off')
                
                if pair['input'].shape == pair['output'].shape and best_prog is not None:
                    pred = pair['input'].copy()
                    shape_ok = True
                    program_steps = []
                    
                    for op, params in best_prog:
                        try:
                            pred_new = op(pred, *params)
                            if not isinstance(pred_new, np.ndarray):
                                shape_ok = False
                                msg = 'Primitive returned non-array'
                                break
                            if pred_new.shape != pred.shape:
                                shape_ok = False
                                msg = 'Shape changed during program'
                                break
                            pred = pred_new
                            program_steps.append(f"{op.__name__}{params}")
                        except Exception as e:
                            shape_ok = False
                            msg = f'Error: {str(e)}'
                            break
                    
                    if shape_ok:
                        ax3 = fig.add_subplot(gs[0, 2])
                        plot_grid(pred, ax=ax3, title=f'Predicted (Acc: {accuracy:.2f})', show_values=True, dpi=120, cell_size=0.8)
                        add_grid_stats(ax3, pred, ax3.transAxes)
                        
                        analysis_text = [
                            f"Program Accuracy: {accuracy:.2f}",
                            f"Pixel-wise Similarity: {np.mean(pred == pair['output']):.2f}",
                            f"Program Length: {len(best_prog)}",
                            "Program Steps:",
                            *[f"  {i+1}. {step}" for i, step in enumerate(program_steps)]
                        ]
                        ax_analysis.text(0.02, 0.95, '\n'.join(analysis_text), 
                                       fontsize=10, va='top', family='monospace')
                    else:
                        ax_analysis.text(0.5, 0.5, msg, ha='center', va='center', fontsize=12)
                else:
                    ax_analysis.text(0.5, 0.5, 'No valid program found', 
                                   ha='center', va='center', fontsize=12)
                
                plt.suptitle(f'Training Example {i+1} - Detailed Analysis', fontsize=14, y=1.02)
                plt.tight_layout()
                plt.show()
                
    except Exception as e:
        print(f'Could not visualize train outputs: {str(e)}')

# ---------------- Run demo ----------------
train_file = '/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json'
test_file = '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json'
submission_file = '/kaggle/input/arc-prize-2025/sample_submission.json'

visualize_arc_results(train_file, test_file, submission_file)



# Test different transformation patterns
def create_test_cases():
    # Test case 1: Simple color mapping
    test1 = {
        'train': [
            {
                'input': [[1, 1, 2],
                         [2, 1, 1],
                         [2, 2, 1]],
                'output': [[3, 3, 4],
                          [4, 3, 3],
                          [4, 4, 3]]
            }
        ],
        'test': [
            {
                'input': [[1, 2, 1],
                         [2, 2, 2],
                         [1, 1, 2]]
            }
        ]
    }

    # Test case 2: Pattern recognition
    test2 = {
        'train': [
            {
                'input': [[0, 1, 0],
                         [1, 2, 1],
                         [0, 1, 0]],
                'output': [[1, 2, 1],
                          [2, 3, 2],
                          [1, 2, 1]]
            }
        ],
        'test': [
            {
                'input': [[0, 1, 0, 1],
                         [1, 2, 2, 0],
                         [0, 2, 2, 1],
                         [1, 0, 1, 0]]
            }
        ]
    }

    # Test case 3: Object transformation
    test3 = {
        'train': [
            {
                'input': [[0, 1, 1, 0],
                         [1, 2, 2, 1],
                         [1, 2, 2, 1],
                         [0, 1, 1, 0]],
                'output': [[0, 2, 2, 0],
                          [2, 3, 3, 2],
                          [2, 3, 3, 2],
                          [0, 2, 2, 0]]
            }
        ],
        'test': [
            {
                'input': [[0, 1, 1],
                         [1, 2, 1],
                         [0, 1, 0]]
            }
        ]
    }

    # Save test cases to JSON files
    import json
    
    # Save training data
    train_data = {'0': test1, '1': test2, '2': test3}
    with open('custom_train_challenges.json', 'w') as f:
        json.dump(train_data, f)
    
    # Save test data
    test_data = {'0': {'test': test1['test']}, 
                 '1': {'test': test2['test']}, 
                 '2': {'test': test3['test']}}
    with open('custom_test_challenges.json', 'w') as f:
        json.dump(test_data, f)
    
    return 'custom_train_challenges.json', 'custom_test_challenges.json'

# Create and test with custom examples
custom_train_file, custom_test_file = create_test_cases()
print("Generated custom test cases. Running visualization...")

# Visualize results with custom test cases
visualize_arc_results(
    train_file=custom_train_file,
    test_file=custom_test_file,
    submission_file='custom_submission.json'
)


# Enhanced ARC visualization - showing more tasks
import numpy as np
import matplotlib.pyplot as plt
import json
import os

print("Starting enhanced ARC visualization...")

# Now try to load and show MULTIPLE tasks
try:
    file_path = '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json'
    if os.path.exists(file_path):
        print(f"Found file: {file_path}")
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"Total tasks in file: {len(data)}")
        
        # Get first 6 tasks to show
        if isinstance(data, dict):
            task_keys = list(data.keys())[:6]  # Show first 6 tasks
        else:
            task_keys = list(range(min(6, len(data))))
        
        print(f"Will display {len(task_keys)} tasks")
        
        for i, task_key in enumerate(task_keys):
            print(f"\n--- Task {task_key} ---")
            
            # Get task data
            if isinstance(data, dict):
                task = data[task_key]
            else:
                task = data[task_key]
            
            # Show test inputs for this task
            if 'test' in task and len(task['test']) > 0:
                num_tests = len(task['test'])
                print(f"Task {task_key} has {num_tests} test case(s)")
                
                # Show up to 3 test cases per task
                tests_to_show = min(3, num_tests)
                
                fig, axes = plt.subplots(1, tests_to_show, figsize=(4*tests_to_show, 4))
                fig.suptitle(f'Task {task_key} - Test Inputs ({num_tests} total)', fontsize=14)
                
                # Handle single subplot case
                if tests_to_show == 1:
                    axes = [axes]
                
                for test_idx in range(tests_to_show):
                    test_case = task['test'][test_idx]
                    grid = np.array(test_case['input'])
                    
                    # Plot
                    axes[test_idx].imshow(grid, cmap='tab20', interpolation='nearest')
                    axes[test_idx].axis('off')
                    axes[test_idx].set_title(f'Test {test_idx+1}\nShape: {grid.shape}')
                    
                    # Add some statistics
                    unique_colors = len(np.unique(grid))
                    axes[test_idx].text(0.02, 0.98, f'Colors: {unique_colors}', 
                                       transform=axes[test_idx].transAxes, 
                                       fontsize=8, verticalalignment='top',
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                
                plt.tight_layout()
                plt.show()
                
                # Also show training examples if available
                if 'train' in task and len(task['train']) > 0:
                    num_train = len(task['train'])
                    print(f"Task {task_key} has {num_train} training example(s)")
                    
                    # Show up to 2 training examples
                    train_to_show = min(2, num_train)
                    
                    fig, axes = plt.subplots(2, train_to_show, figsize=(4*train_to_show, 6))
                    fig.suptitle(f'Task {task_key} - Training Examples', fontsize=14)
                    
                    # Handle single example case
                    if train_to_show == 1:
                        axes = axes.reshape(-1, 1)
                    
                    for train_idx in range(train_to_show):
                        train_pair = task['train'][train_idx]
                        input_grid = np.array(train_pair['input'])
                        output_grid = np.array(train_pair['output'])
                        
                        # Plot input
                        axes[0, train_idx].imshow(input_grid, cmap='tab20', interpolation='nearest')
                        axes[0, train_idx].axis('off')
                        axes[0, train_idx].set_title(f'Train {train_idx+1} Input\n{input_grid.shape}')
                        
                        # Plot output
                        axes[1, train_idx].imshow(output_grid, cmap='tab20', interpolation='nearest')
                        axes[1, train_idx].axis('off')
                        axes[1, train_idx].set_title(f'Train {train_idx+1} Output\n{output_grid.shape}')
                    
                    plt.tight_layout()
                    plt.show()
                
            else:
                print(f"No test data found in task {task_key}")
                
            # Add separator between tasks
            if i < len(task_keys) - 1:
                print("="*50)
    
    else:
        print(f"File not found: {file_path}")
        
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()

print("\nVisualization complete!")


# Comprehensive ARC Dataset Explorer - Show More Diverse Tasks
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from collections import Counter

print("Starting comprehensive ARC dataset exploration...")

def analyze_task_characteristics(task):
    """Analyze characteristics of a task to categorize it"""
    characteristics = {}
    
    # Analyze training examples
    if 'train' in task:
        input_shapes = [np.array(example['input']).shape for example in task['train']]
        output_shapes = [np.array(example['output']).shape for example in task['train']]
        
        characteristics['num_train_examples'] = len(task['train'])
        characteristics['input_shapes'] = input_shapes
        characteristics['output_shapes'] = output_shapes
        characteristics['max_input_size'] = max([shape[0] * shape[1] for shape in input_shapes])
        characteristics['shape_changes'] = any(inp != out for inp, out in zip(input_shapes, output_shapes))
        
        # Analyze colors
        all_colors = set()
        for example in task['train']:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            all_colors.update(np.unique(input_grid))
            all_colors.update(np.unique(output_grid))
        
        characteristics['num_colors'] = len(all_colors)
        characteristics['max_color'] = max(all_colors) if all_colors else 0
    
    # Analyze test examples
    if 'test' in task:
        test_shapes = [np.array(example['input']).shape for example in task['test']]
        characteristics['num_test_examples'] = len(task['test'])
        characteristics['test_shapes'] = test_shapes
    
    return characteristics

def categorize_tasks(data, num_tasks=20):
    """Categorize tasks by their characteristics"""
    categories = {
        'small_grids': [],      # <= 10x10
        'medium_grids': [],     # 11x20
        'large_grids': [],      # > 20x20
        'many_colors': [],      # > 5 colors
        'shape_changing': [],   # input/output shapes differ
        'complex_patterns': []  # many training examples
    }
    
    if isinstance(data, dict):
        task_items = list(data.items())[:num_tasks]
    else:
        task_items = [(i, task) for i, task in enumerate(data[:num_tasks])]
    
    for task_id, task in task_items:
        chars = analyze_task_characteristics(task)
        
        # Categorize by grid size
        max_size = chars.get('max_input_size', 0)
        if max_size <= 100:  # 10x10
            categories['small_grids'].append((task_id, task, chars))
        elif max_size <= 400:  # 20x20
            categories['medium_grids'].append((task_id, task, chars))
        else:
            categories['large_grids'].append((task_id, task, chars))
        
        # Categorize by colors
        if chars.get('num_colors', 0) > 5:
            categories['many_colors'].append((task_id, task, chars))
        
        # Categorize by shape changes
        if chars.get('shape_changes', False):
            categories['shape_changing'].append((task_id, task, chars))
        
        # Categorize by complexity
        if chars.get('num_train_examples', 0) > 3:
            categories['complex_patterns'].append((task_id, task, chars))
    
    return categories

def visualize_category(category_name, tasks, max_show=3):
    """Visualize tasks from a specific category"""
    print(f"\n{'='*60}")
    print(f"CATEGORY: {category_name.upper().replace('_', ' ')}")
    print(f"Found {len(tasks)} tasks in this category")
    print(f"{'='*60}")
    
    for i, (task_id, task, chars) in enumerate(tasks[:max_show]):
        print(f"\n--- Task {task_id} ({category_name}) ---")
        print(f"Characteristics: {chars}")
        
        # Show test inputs
        if 'test' in task and len(task['test']) > 0:
            num_tests = len(task['test'])
            tests_to_show = min(2, num_tests)
            
            fig, axes = plt.subplots(1, tests_to_show, figsize=(6*tests_to_show, 5))
            fig.suptitle(f'Task {task_id} - Test Inputs\n({category_name})', fontsize=14)
            
            if tests_to_show == 1:
                axes = [axes]
            
            for test_idx in range(tests_to_show):
                test_case = task['test'][test_idx]
                grid = np.array(test_case['input'])
                
                axes[test_idx].imshow(grid, cmap='tab20', interpolation='nearest')
                axes[test_idx].axis('off')
                axes[test_idx].set_title(f'Test {test_idx+1}\nShape: {grid.shape}\nColors: {len(np.unique(grid))}')
                
                # Add grid lines for better visibility
                if grid.shape[0] <= 20 and grid.shape[1] <= 20:
                    axes[test_idx].set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
                    axes[test_idx].set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
                    axes[test_idx].grid(which='minor', color='gray', linewidth=0.5, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        # Show training examples for interesting cases
        if 'train' in task and len(task['train']) > 0 and i == 0:  # Show training for first task only
            num_train = min(3, len(task['train']))
            
            fig, axes = plt.subplots(2, num_train, figsize=(4*num_train, 6))
            fig.suptitle(f'Task {task_id} - Training Examples ({category_name})', fontsize=14)
            
            if num_train == 1:
                axes = axes.reshape(-1, 1)
            
            for train_idx in range(num_train):
                train_pair = task['train'][train_idx]
                input_grid = np.array(train_pair['input'])
                output_grid = np.array(train_pair['output'])
                
                # Plot input
                axes[0, train_idx].imshow(input_grid, cmap='tab20', interpolation='nearest')
                axes[0, train_idx].axis('off')
                axes[0, train_idx].set_title(f'Input {train_idx+1}\n{input_grid.shape}')
                
                # Plot output
                axes[1, train_idx].imshow(output_grid, cmap='tab20', interpolation='nearest')
                axes[1, train_idx].axis('off')
                axes[1, train_idx].set_title(f'Output {train_idx+1}\n{output_grid.shape}')
                
                # Add grid lines for smaller grids
                for ax in [axes[0, train_idx], axes[1, train_idx]]:
                    if input_grid.shape[0] <= 15 and input_grid.shape[1] <= 15:
                        ax.set_xticks(np.arange(-0.5, max(input_grid.shape[1], output_grid.shape[1]), 1), minor=True)
                        ax.set_yticks(np.arange(-0.5, max(input_grid.shape[0], output_grid.shape[0]), 1), minor=True)
                        ax.grid(which='minor', color='gray', linewidth=0.5, alpha=0.3)
            
            plt.tight_layout()
            plt.show()

# Load and analyze the dataset
try:
    file_path = '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json'
    if os.path.exists(file_path):
        print(f"Loading {file_path}...")
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"Analyzing {len(data)} tasks...")
        
        # Categorize tasks
        categories = categorize_tasks(data, num_tasks=50)  # Analyze more tasks
        
        # Print category statistics
        print(f"\nCATEGORY STATISTICS:")
        for cat_name, tasks in categories.items():
            print(f"- {cat_name.replace('_', ' ').title()}: {len(tasks)} tasks")
        
        # Visualize each category
        visualize_category("large_grids", categories['large_grids'], max_show=2)
        visualize_category("many_colors", categories['many_colors'], max_show=2)
        visualize_category("shape_changing", categories['shape_changing'], max_show=2)
        visualize_category("complex_patterns", categories['complex_patterns'], max_show=2)
        
        # Show some interesting statistics
        print(f"\n{'='*60}")
        print("DATASET STATISTICS")
        print(f"{'='*60}")
        
        all_characteristics = []
        if isinstance(data, dict):
            task_items = list(data.items())[:30]
        else:
            task_items = [(i, task) for i, task in enumerate(data[:30])]
        
        grid_sizes = []
        color_counts = []
        shape_changes = 0
        
        for task_id, task in task_items:
            chars = analyze_task_characteristics(task)
            all_characteristics.append(chars)
            
            if 'max_input_size' in chars:
                grid_sizes.append(chars['max_input_size'])
            if 'num_colors' in chars:
                color_counts.append(chars['num_colors'])
            if chars.get('shape_changes', False):
                shape_changes += 1
        
        if grid_sizes:
            print(f"Grid sizes: min={min(grid_sizes)}, max={max(grid_sizes)}, avg={np.mean(grid_sizes):.1f}")
        if color_counts:
            print(f"Color counts: min={min(color_counts)}, max={max(color_counts)}, avg={np.mean(color_counts):.1f}")
        print(f"Tasks with shape changes: {shape_changes}/{len(task_items)} ({100*shape_changes/len(task_items):.1f}%)")
        
    else:
        print(f"File not found: {file_path}")
        
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()

print("\nComprehensive visualization complete!")


# Advanced ARC Pattern Explorer - Complex Transformations
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from collections import Counter
import matplotlib.patches as patches

print("Exploring COMPLEX ARC patterns...")

def analyze_complexity(task):
    """Analyze the complexity of transformations in a task"""
    complexity_score = 0
    features = {}
    
    if 'train' not in task:
        return complexity_score, features
    
    for example in task['train']:
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        # Check for size changes (scaling, cropping, padding)
        if input_grid.shape != output_grid.shape:
            complexity_score += 3
            features['size_change'] = True
        
        # Count pixel changes
        if input_grid.shape == output_grid.shape:
            pixel_changes = np.sum(input_grid != output_grid)
            change_ratio = pixel_changes / (input_grid.shape[0] * input_grid.shape[1])
            if change_ratio > 0.5:
                complexity_score += 2
                features['major_transformation'] = True
        
        # Check for pattern complexity
        unique_input = len(np.unique(input_grid))
        unique_output = len(np.unique(output_grid))
        if unique_output > unique_input + 2:
            complexity_score += 2
            features['color_generation'] = True
        
        # Check for symmetry operations
        if input_grid.shape == output_grid.shape:
            # Check rotations
            for k in [1, 2, 3]:
                if np.array_equal(output_grid, np.rot90(input_grid, k)):
                    complexity_score += 1
                    features['rotation'] = True
                    break
            
            # Check reflections
            if np.array_equal(output_grid, np.flipud(input_grid)) or np.array_equal(output_grid, np.fliplr(input_grid)):
                complexity_score += 1
                features['reflection'] = True
        
        # Check for tiling/repetition patterns
        if output_grid.shape[0] > input_grid.shape[0] * 2 or output_grid.shape[1] > input_grid.shape[1] * 2:
            complexity_score += 3
            features['scaling_tiling'] = True
    
    return complexity_score, features

def find_complex_tasks(data, min_complexity=5, max_tasks=15):
    """Find tasks with high complexity scores"""
    complex_tasks = []
    
    if isinstance(data, dict):
        task_items = list(data.items())
    else:
        task_items = [(i, task) for i, task in enumerate(data)]
    
    for task_id, task in task_items:
        complexity, features = analyze_complexity(task)
        if complexity >= min_complexity:
            complex_tasks.append((task_id, task, complexity, features))
    
    # Sort by complexity and return top tasks
    complex_tasks.sort(key=lambda x: x[2], reverse=True)
    return complex_tasks[:max_tasks]

def visualize_complex_task(task_id, task, complexity, features):
    """Visualize a complex task with detailed analysis"""
    print(f"\n{'='*70}")
    print(f"COMPLEX TASK: {task_id} (Complexity Score: {complexity})")
    print(f"Features: {list(features.keys())}")
    print(f"{'='*70}")
    
    if 'train' not in task or len(task['train']) == 0:
        print("No training data available")
        return
    
    # Show training examples with transformation analysis
    num_examples = min(4, len(task['train']))
    
    fig = plt.figure(figsize=(16, 4 * num_examples))
    
    for i in range(num_examples):
        example = task['train'][i]
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        # Create subplot for this example
        ax_input = plt.subplot(num_examples, 4, i*4 + 1)
        ax_output = plt.subplot(num_examples, 4, i*4 + 2)
        ax_diff = plt.subplot(num_examples, 4, i*4 + 3)
        ax_analysis = plt.subplot(num_examples, 4, i*4 + 4)
        
        # Plot input
        ax_input.imshow(input_grid, cmap='tab20', interpolation='nearest')
        ax_input.set_title(f'Input {i+1}\n{input_grid.shape}')
        ax_input.axis('off')
        
        # Add grid lines for clarity
        if input_grid.shape[0] <= 15 and input_grid.shape[1] <= 15:
            ax_input.set_xticks(np.arange(-0.5, input_grid.shape[1], 1), minor=True)
            ax_input.set_yticks(np.arange(-0.5, input_grid.shape[0], 1), minor=True)
            ax_input.grid(which='minor', color='gray', linewidth=0.5, alpha=0.7)
        
        # Plot output
        ax_output.imshow(output_grid, cmap='tab20', interpolation='nearest')
        ax_output.set_title(f'Output {i+1}\n{output_grid.shape}')
        ax_output.axis('off')
        
        if output_grid.shape[0] <= 15 and output_grid.shape[1] <= 15:
            ax_output.set_xticks(np.arange(-0.5, output_grid.shape[1], 1), minor=True)
            ax_output.set_yticks(np.arange(-0.5, output_grid.shape[0], 1), minor=True)
            ax_output.grid(which='minor', color='gray', linewidth=0.5, alpha=0.7)
        
        # Create difference visualization
        if input_grid.shape == output_grid.shape:
            diff_grid = (input_grid != output_grid).astype(int)
            ax_diff.imshow(diff_grid, cmap='RdYlBu_r', interpolation='nearest')
            ax_diff.set_title(f'Changes\n{np.sum(diff_grid)} pixels')
        else:
            # Show size change visualization
            ax_diff.text(0.5, 0.5, f'Size Change:\n{input_grid.shape}\n→\n{output_grid.shape}', 
                        ha='center', va='center', transform=ax_diff.transAxes, fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        ax_diff.axis('off')
        
        # Analysis text
        analysis_text = []
        analysis_text.append(f"Example {i+1} Analysis:")
        analysis_text.append(f"Input colors: {len(np.unique(input_grid))}")
        analysis_text.append(f"Output colors: {len(np.unique(output_grid))}")
        
        if input_grid.shape == output_grid.shape:
            changes = np.sum(input_grid != output_grid)
            total_pixels = input_grid.shape[0] * input_grid.shape[1]
            change_pct = (changes / total_pixels) * 100
            analysis_text.append(f"Changed: {change_pct:.1f}%")
        else:
            scale_x = output_grid.shape[1] / input_grid.shape[1]
            scale_y = output_grid.shape[0] / input_grid.shape[0]
            analysis_text.append(f"Scale: {scale_x:.1f}x, {scale_y:.1f}x")
        
        # Check for specific patterns
        if input_grid.shape == output_grid.shape:
            # Check for rotations
            for k, name in [(1, "90°"), (2, "180°"), (3, "270°")]:
                if np.array_equal(output_grid, np.rot90(input_grid, k)):
                    analysis_text.append(f"Rotation: {name}")
                    break
            
            # Check for reflections
            if np.array_equal(output_grid, np.flipud(input_grid)):
                analysis_text.append("Vertical flip")
            elif np.array_equal(output_grid, np.fliplr(input_grid)):
                analysis_text.append("Horizontal flip")
            
            # Check for color mappings
            unique_changes = set()
            for r in range(input_grid.shape[0]):
                for c in range(input_grid.shape[1]):
                    if input_grid[r, c] != output_grid[r, c]:
                        unique_changes.add((input_grid[r, c], output_grid[r, c]))
            
            if len(unique_changes) <= 3 and len(unique_changes) > 0:
                analysis_text.append("Color mappings:")
                for old, new in list(unique_changes)[:3]:
                    analysis_text.append(f"  {old} → {new}")
        
        ax_analysis.text(0.05, 0.95, '\n'.join(analysis_text), 
                        transform=ax_analysis.transAxes, fontsize=8, 
                        verticalalignment='top', family='monospace')
        ax_analysis.axis('off')
    
    plt.suptitle(f'Complex Task {task_id} - Detailed Analysis\nComplexity: {complexity}, Features: {list(features.keys())}', 
                 fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()
    
    # Show test cases
    if 'test' in task and len(task['test']) > 0:
        print(f"Test cases for task {task_id}:")
        num_tests = min(3, len(task['test']))
        
        fig, axes = plt.subplots(1, num_tests, figsize=(5*num_tests, 5))
        if num_tests == 1:
            axes = [axes]
        
        for i in range(num_tests):
            test_grid = np.array(task['test'][i]['input'])
            axes[i].imshow(test_grid, cmap='tab20', interpolation='nearest')
            axes[i].set_title(f'Test {i+1}\n{test_grid.shape}\nColors: {len(np.unique(test_grid))}')
            axes[i].axis('off')
            
            if test_grid.shape[0] <= 15 and test_grid.shape[1] <= 15:
                axes[i].set_xticks(np.arange(-0.5, test_grid.shape[1], 1), minor=True)
                axes[i].set_yticks(np.arange(-0.5, test_grid.shape[0], 1), minor=True)
                axes[i].grid(which='minor', color='gray', linewidth=0.5, alpha=0.7)
        
        plt.suptitle(f'Test Cases for Complex Task {task_id}', fontsize=14)
        plt.tight_layout()
        plt.show()

# Load and find complex patterns
try:
    file_path = '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json'
    if os.path.exists(file_path):
        print(f"Loading {file_path} to find complex patterns...")
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Find most complex tasks
        complex_tasks = find_complex_tasks(data, min_complexity=6, max_tasks=8)
        
        print(f"\nFound {len(complex_tasks)} highly complex tasks:")
        for task_id, task, complexity, features in complex_tasks:
            print(f"- Task {task_id}: Complexity {complexity}, Features: {list(features.keys())}")
        
        # Visualize the most complex tasks
        print("\n" + "="*70)
        print("VISUALIZING MOST COMPLEX PATTERNS")
        print("="*70)
        
        for i, (task_id, task, complexity, features) in enumerate(complex_tasks[:5]):  # Show top 5
            visualize_complex_task(task_id, task, complexity, features)
            
            if i < len(complex_tasks[:5]) - 1:
                print("\n" + "-"*50 + "\n")
    
    else:
        print(f"File not found: {file_path}")
        
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()

print("\nComplex pattern exploration complete!")

