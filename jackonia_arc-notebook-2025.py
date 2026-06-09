import json
import numpy as np
from collections import Counter, deque
import itertools
import math
from pathlib import Path
import os





# ---------------- helpers ----------------
def shape(g):
    """Get the (rows, cols) shape of a grid."""
    if not g or not isinstance(g, list):
        return (0, 0)
    rows = len(g)
    cols = len(g[0]) if rows > 0 and isinstance(g[0], list) else 0
    return (rows, cols)

def deep_copy(g):
    """Create a new copy of a grid."""
    return [list(row) for row in g]

def rotate90(g):
    """Rotate a grid 90 degrees clockwise."""
    if not g or not g[0]: return []
    r,c = shape(g)
    return [[g[r-1-j][i] for j in range(r)] for i in range(c)]

def rotate180(g): return rotate90(rotate90(g))
def rotate270(g): return rotate90(rotate180(g))
def flip_h(g): return [list(reversed(row)) for row in g]
def flip_v(g): return list(reversed(g))
def transpose(g): return list(map(list, zip(*g))) if g and g[0] else []

def all_cells(g):
    """Yield all cell values in a grid."""
    for row in g:
        for v in row:
            yield v

def color_counts(g):
    """Return a Counter of {color: count} for a grid."""
    return Counter(all_cells(g))

def majority_border_color(grid):
    """Find the most common color on the grid's border."""
    if not grid or not grid[0]: return 0
    r,c = shape(grid)
    border=[]
    border.extend(grid[0]) # Top row
    if r>1: border.extend(grid[-1]) # Bottom row
    for i in range(1,r-1):
        border.append(grid[i][0]) # Left edge
        if c>1: border.append(grid[i][-1]) # Right edge
    if not border: return 0
    return Counter(border).most_common(1)[0][0]

# ---------------- connected components (object detection) ----------------
def connected_components_by_color(arr):
    """
    Finds all contiguous "objects" in the grid, grouped by color.
    Uses numpy for array operations.
    """
    h,w = arr.shape
    visited = np.zeros_like(arr, dtype=bool)
    comps = [] # List of all found components (objects)
    for i in range(h):
        for j in range(w):
            if visited[i,j]: continue
            color = int(arr[i,j])
            
            # We don't find "objects" of the background color (assume 0)
            if color == 0: 
                visited[i,j] = True
                continue
            
            # Start a Breadth-First Search (BFS) for this object
            q = deque([(i, j)])
            visited[i,j] = True
            pts = [] # Points in this component
            
            while q:
                y,x = q.popleft()
                pts.append((y,x))
                # Check neighbors (up, down, left, right)
                for dy,dx in ((1,0),(-1,0),(0,1),(-1,0)):
                    ny, nx = y+dy, x+dx
                    # Check if neighbor is in bounds, not visited, and same color
                    if 0 <= ny < h and 0 <= nx < w and \
                       not visited[ny,nx] and int(arr[ny,nx]) == color:
                        visited[ny,nx] = True
                        q.append((ny,nx))
            
            if pts:
                # If we found points, save this component
                ys = [p[0] for p in pts]; xs = [p[1] for p in pts]
                ymin,ymax = min(ys), max(ys)
                xmin,xmax = min(xs), max(xs)
                mask = np.zeros_like(arr, dtype=bool)
                for (yy,xx) in pts: mask[yy,xx] = True
                comps.append({
                    "color": int(color),
                    "positions": pts,
                    "bbox": (ymin,ymax,xmin,xmax), # Bounding box
                    "mask": mask
                })
    return comps

# ---------------- translations (moving objects) ----------------
def translate_candidate(crop, dx, dy, R, C, fill):
    """
    Places a small 'crop' grid onto a larger canvas of size (R, C),
    offset by (dx, dy) from the center, using 'fill' as background.
    """
    r,c = shape(crop)
    out = [[fill]*C for _ in range(R)]
    # Calculate top-left corner, starting from center
    start_y = max(0, (R - r)//2 + dy)
    start_x = max(0, (C - c)//2 + dx)
    for i in range(r):
        for j in range(c):
            yi, xj = start_y + i, start_x + j
            if 0 <= yi < R and 0 <= xj < C:
                out[yi][xj] = crop[i][j]
    return out

# ---------------- candidate generators (the "brainstorming") ----------------
def gen_structural(test_input, train_examples):
    """Generate candidates from simple geometric transformations."""
    out = []
    # All 8 symmetries
    for fn in (lambda g: deep_copy(g), rotate90, rotate180, rotate270, 
               flip_h, flip_v, transpose, lambda g: flip_h(transpose(g))):
        try:
            out.append(fn(test_input))
        except:
            pass # Ignore transforms that fail (e.g., transpose on empty)
    return out

def gen_color_variants(test_input, train_examples):
    """Generate candidates from simple color changes."""
    arr = np.array(test_input)
    uniques = sorted(list(set(all_cells(test_input))))
    out = []
    
    # 1. Fill with most dominant color
    counts = Counter(all_cells(test_input))
    if counts:
        dom_color = counts.most_common(1)[0][0]
        out.append([[dom_color]*shape(test_input)[1] for _ in range(shape(test_input)[0])])
        
    # 2. Swap the two most common colors (if 2+ exist)
    if len(counts) >= 2:
        top_two = [c[0] for c in counts.most_common(2)]
        a, b = top_two[0], top_two[1]
        arr2 = arr.copy()
        arr2[arr==a], arr2[arr==b] = b, a
        out.append(arr2.tolist())
    return out
    
def gen_object_based(test_input, train_examples):
    """Generate candidates based on object detection."""
    arr = np.array(test_input)
    comps = connected_components_by_color(arr)
    out = []
    
    # 1. If there's exactly one object, try centering it
    if len(comps) == 1:
        comp = comps[0]
        ymin,ymax,xmin,xmax = comp["bbox"]
        obj = arr[ymin:ymax+1, xmin:xmax+1] # Crop the object
        
        # Create a version of the object with its local background
        obj_with_bg_mask = comp['mask'][ymin:ymax+1, xmin:xmax+1]
        obj_with_bg = np.where(obj_with_bg_mask, obj, majority_border_color(test_input))
        
        R,C = shape(test_input)
        bg = majority_border_color(test_input)
        
        # Place the cropped object (with its background) onto a new canvas
        out.append(translate_candidate(obj_with_bg.tolist(), 0, 0, R, C, bg))

    # 2. If multiple objects, try flipping the whole grid
    if len(comps) > 1:
        out.extend([flip_h(test_input), flip_v(test_input)])
    return out

# ---------------- candidate scoring (the "ranking") ----------------
def grid_difference(a, b):
    """Calculate a simple difference score between two grids."""
    # Convert to numpy for easier comparison
    try:
        a_np = np.array(a); b_np = np.array(b)
    except:
        return 1e9 # Grids are malformed
        
    # Pad the smaller grid to match the larger one
    h = max(a_np.shape[0], b_np.shape[0])
    w = max(a_np.shape[1], b_np.shape[1])
    
    if h == 0 or w == 0: return 0 if h==w else 1e9
    
    a_pad = np.zeros((h,w), dtype=int)
    b_pad = np.zeros((h,w), dtype=int)
    
    if a_np.shape[0] > 0 and a_np.shape[1] > 0:
        a_pad[:a_np.shape[0], :a_np.shape[1]] = a_np
    if b_np.shape[0] > 0 and b_np.shape[1] > 0:
        b_pad[:b_np.shape[0], :b_np.shape[1]] = b_np
        
    # Return the number of differing cells
    return int((a_pad != b_pad).sum())

def candidate_score(candidate, train_examples, test_input):
    """
    Scores a candidate solution. Higher is better.
    This is a heuristic score.
    """
    if not train_examples: return 0.0 # No info to score against
    
    score = 0.0
    cand_shape = shape(candidate)
    cand_counts = color_counts(candidate)
    test_input_shape = shape(test_input) # Get test_input shape
    
    # Compare candidate to each training output
    for ex in train_examples:
        train_in, train_out = ex['input'], ex['output']
        train_in_shape, train_out_shape = shape(train_in), shape(train_out)
        out_counts = color_counts(train_out)
        
        # 1. Reward: Shape matches training output shape
        if cand_shape == train_out_shape:
            score += 1.0
            
        # 2. Reward: Shape *change* matches training shape *change*
        if (cand_shape[0] - test_input_shape[0] == train_out_shape[0] - train_in_shape[0]) and \
           (cand_shape[1] - test_input_shape[1] == train_out_shape[1] - train_in_shape[1]):
            score += 1.0
            
        # 3. Reward: Color histogram is similar to output histogram
        all_keys = set(cand_counts.keys()) | set(out_counts.keys())
        vec1 = np.array([cand_counts.get(k, 0) for k in all_keys], dtype=float)
        vec2 = np.array([out_counts.get(k, 0) for k in all_keys], dtype=float)
        denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        hist_score = (float(np.dot(vec1, vec2) / denom) if denom > 0 else 0.0)
        score += hist_score
        
    # 4. Penalty: Penalize high complexity (too many colors)
    score -= 0.01 * len(cand_counts)
    
    return score

# ---------------- main solver class ----------------
class SingleFileARCSolver:
    def __init__(self, test_input):
        # List of all generator functions to run
        self.generators = [
            gen_structural, 
            gen_color_variants, 
            gen_object_based,
            lambda t,te: [deep_copy(t)] # Always include "copy the input"
        ]
        # A simple fallback in case no candidates are generated
        self.fallback = deep_copy(test_input) 
        self.test_input = test_input # Store test_input for the scorer

    def generate_candidates(self, test_input, train_examples):
        """Run all generators and collect unique candidates."""
        all_cands=[]
        for gen_func in self.generators:
            try:
                # Call the generator
                candidates = gen_func(test_input, train_examples)
                all_cands.extend(candidates)
            except Exception as e:
                # Don't let one bad generator stop the rest
                # print(f"Warning: Generator {gen_func.__name__} failed: {e}")
                pass

        # Also try moving the single biggest object around
        arr = np.array(test_input)
        comps = connected_components_by_color(arr)
        if comps:
            biggest = max(comps, key=lambda x: len(x['positions']))
            ymin,ymax,xmin,xmax = biggest['bbox']
            sub = arr[ymin:ymax+1, xmin:xmax+1].tolist()
            R,C = shape(test_input)
            bg = majority_border_color(test_input)
            # Try 5x5 grid of translations
            for dy in range(-2,3):
                for dx in range(-2,3):
                    all_cands.append(translate_candidate(sub, dx, dy, R, C, bg))
        
        # De-duplicate the list of candidates
        uniq, seen_str = [], set()
        for cand in all_cands:
            try:
                s = str(cand) # Simple way to hash a grid
                if s not in seen_str:
                    uniq.append(cand)
                    seen_str.add(s)
            except:
                pass # Ignore malformed candidates
        return uniq

    def select_two(self, candidates, train_examples):
        """Score all candidates and select the best two."""
        if not candidates:
            # No candidates generated, return the fallback
            return self.fallback, rotate90(self.fallback) # Return two different fallbacks

        # Score all candidates
        scored = sorted(
            [(candidate_score(c, train_examples, self.test_input), c) for c in candidates], 
            key=lambda x: -x[0] # Sort by score, descending
        )
        
        # Get the best candidate
        first = scored[0][1]
        
        # Find the "second best" candidate that is *different* from the first
        best2 = first
        best2_score = -1e9
        
        # Look through the top 25 candidates
        for score, cand in scored[1:min(len(scored), 25)]:
            # Calculate difference from the *first* pick
            diff = grid_difference(first, cand)
            # We want a mix of high score and high difference
            metric = (0.8 * diff) + (0.2 * score) 
            if metric > best2_score:
                best2_score, best2 = metric, cand
                
        # Handle edge case where all candidates were identical
        if grid_difference(first, best2) == 0:
            best2 = rotate90(first) # Create a different second answer
            
        return first, best2

# ---------------- top-level main function ----------------
def main():
    """
    Main function to load tasks, run the solver, and write submission.json
    """
    print("Running single-file heuristic ARC solver...")
    
    # Define file paths
    # Assumes data is in a folder named 'data' in the same directory
    # On Kaggle, this path will be '/kaggle/input/arc-prize-2024/...'
    data_dir = '/kaggle/input/arc-prize-2024'
    if not os.path.exists(data_dir):
        # Try the 2025 path as a fallback
        data_dir = '/kaggle/input/arc-prize-2025'
        if not os.path.exists(data_dir):
            data_dir = '.' # Fallback for local testing
        
    input_file_path = os.path.join(data_dir, 'arc-agi_test_challenges.json')
    output_file_path = 'submission.json' # Will be saved in /kaggle/working/

    try:
        with open(input_file_path, 'r') as f:
            test_data = json.load(f)
        print(f"Loaded {len(test_data)} tasks from: {input_file_path}")
    except FileNotFoundError:
        print(f"FATAL ERROR: Input file not found at '{input_file_path}'.")
        print("Tried '/kaggle/input/arc-prize-2024' and '/kaggle/input/arc-prize-2025'.")
        print("Please check the data path.")
        return
    except json.JSONDecodeError:
        print(f"FATAL ERROR: Could not parse JSON from '{input_file_path}'.")
        return

    submission = {} # This will hold our final answers
    count = 0
    
    for task_id, task in test_data.items():
        train_examples = task.get('train', [])
        test_cases = task.get('test', [])
        
        sol_list = [] # List of solutions for this task
        
        for test_case in test_cases:
            test_input = test_case['input']
            
            # Create a new solver for this specific input
            solver = SingleFileARCSolver(test_input)
            
            # 1. Generate
            candidates = solver.generate_candidates(test_input, train_examples)
            
            # 2. Score and Select
            attempt_1, attempt_2 = solver.select_two(candidates, train_examples)
            
            # 3. Add to solution list
            sol_list.append({"attempt_1": attempt_1, "attempt_2": attempt_2})
            
        submission[task_id] = sol_list
        count += 1
        if count % 50 == 0:
            print(f"Processed {count} / {len(test_data)} tasks...")

    # Write the final submission file
    try:
        with open(output_file_path, 'w') as f:
            json.dump(submission, f)
        print(f"\n*** All done. ***")
        print(f"Successfully saved submission.json with {len(submission)} tasks to '{output_file_path}'")
    except Exception as e:
        print(f"\n*** ERROR ***")
        print(f"Failed to write submission file: {e}")

# This makes the script runnable from the command line
if __name__ == '__main__':
    main()



