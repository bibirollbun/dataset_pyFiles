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


#!/usr/bin/env python3
"""
APEX-AGENT-SIGILAGI DUAL SOLVER v8.0
Combines symbolic reasoning (Logician) with neural synthesis (Intuitionist)
for comprehensive ARC task solving.
"""

import os
import json
import time
import hashlib
import random
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

# Configuration
BASE = Path("/kaggle/input/arc-prize-2025")
OUT_DIR = Path("/kaggle/working")
SUBMISSION_FILE = OUT_DIR / "submission.json"
LEDGER_FILE = OUT_DIR / "solver_ledger.json"

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

class DualARCSolver:
    """Dual-strategy ARC solver combining symbolic and neural approaches."""
    
    def __init__(self):
        self.symbolic_cache = {}
        self.neural_cache = {}
        self.strategy_stats = defaultdict(int)
        
    def safe_npify(self, x):
        """Safely convert input to numpy array."""
        if isinstance(x, np.ndarray):
            return x.astype(int)
        try:
            if not x or (isinstance(x, list) and len(x) == 0):
                return np.array([[0]])
            arr = np.array(x, dtype=int)
            if arr.ndim == 0:
                return np.array([[arr.item()]])
            elif arr.ndim == 1:
                return arr.reshape(1, -1)
            return arr
        except:
            return np.array([[0]])
    
    def grids_equal(self, a, b):
        """Check if two grids are identical."""
        a_arr = self.safe_npify(a)
        b_arr = self.safe_npify(b)
        return a_arr.shape == b_arr.shape and np.array_equal(a_arr, b_arr)
    
    def majority_color(self, a):
        """Find the most common color in grid."""
        arr = self.safe_npify(a)
        if arr.size == 0:
            return 0
        counts = np.bincount(arr.ravel())
        return int(np.argmax(counts))
    
    # ========================
    # SYMBOLIC SOLVER (Logician)
    # ========================
    
    def connected_components(self, a, background=None):
        """Find connected components in grid."""
        arr = self.safe_npify(a)
        if arr.size == 0:
            return []
            
        H, W = arr.shape
        if background is None:
            background = self.majority_color(arr)
            
        visited = np.zeros((H, W), bool)
        components = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for y in range(H):
            for x in range(W):
                if visited[y, x] or arr[y, x] == background:
                    continue
                    
                color = arr[y, x]
                queue = [(y, x)]
                visited[y, x] = True
                pixels = [(y, x)]
                
                while queue:
                    cy, cx = queue.pop(0)
                    for dy, dx in directions:
                        ny, nx = cy + dy, cx + dx
                        if (0 <= ny < H and 0 <= nx < W and 
                            not visited[ny, nx] and arr[ny, nx] == color):
                            visited[ny, nx] = True
                            queue.append((ny, nx))
                            pixels.append((ny, nx))
                
                if pixels:
                    ys, xs = zip(*pixels)
                    y_min, y_max = min(ys), max(ys)
                    x_min, x_max = min(xs), max(xs)
                    components.append({
                        'color': color,
                        'pixels': pixels,
                        'bbox': (y_min, y_max, x_min, x_max),
                        'size': len(pixels)
                    })
        
        return components
    
    # Basic transformations
    def rotate_90(self, a):
        return np.rot90(self.safe_npify(a), 1)
    
    def rotate_180(self, a):
        return np.rot90(self.safe_npify(a), 2)
    
    def rotate_270(self, a):
        return np.rot90(self.safe_npify(a), 3)
    
    def flip_horizontal(self, a):
        return np.fliplr(self.safe_npify(a))
    
    def flip_vertical(self, a):
        return np.flipud(self.safe_npify(a))
    
    def transpose(self, a):
        return self.safe_npify(a).T
    
    def invert_colors(self, a, max_color=9):
        arr = self.safe_npify(a)
        return max_color - arr
    
    def resize_to_shape(self, a, target_shape):
        """Resize array to target shape."""
        arr = self.safe_npify(a)
        if arr.size == 0:
            return np.zeros(target_shape, int)
        
        h, w = arr.shape
        th, tw = target_shape
        
        if (h, w) == (th, tw):
            return arr.copy()
        
        out = np.zeros(target_shape, int)
        for i in range(th):
            for j in range(tw):
                src_i = min(int(i * h / th), h - 1)
                src_j = min(int(j * w / tw), w - 1)
                out[i, j] = arr[src_i, src_j]
        return out
    
    def pad_to_shape(self, a, target_shape, fill=None):
        """Pad array to target shape."""
        arr = self.safe_npify(a)
        h, w = arr.shape
        th, tw = target_shape
        
        if h == th and w == tw:
            return arr.copy()
            
        if fill is None:
            fill = self.majority_color(arr)
            
        out = np.full(target_shape, fill, int)
        start_y = max(0, (th - h) // 2)
        start_x = max(0, (tw - w) // 2)
        end_y = min(th, start_y + h)
        end_x = min(tw, start_x + w)
        
        out[start_y:end_y, start_x:end_x] = arr[:end_y-start_y, :end_x-start_x]
        return out
    
    def learn_color_map(self, train_pairs):
        """Learn color mapping from training examples."""
        color_votes = defaultdict(list)
        
        for input_grid, output_grid in train_pairs:
            inp = self.safe_npify(input_grid)
            out = self.safe_npify(output_grid)
            
            if inp.shape != out.shape:
                continue
                
            for color_in in np.unique(inp):
                mask = inp == color_in
                if mask.any():
                    target_colors = out[mask]
                    most_common = Counter(target_colors).most_common(1)
                    if most_common:
                        color_votes[int(color_in)].append(most_common[0][0])
        
        color_map = {}
        for color_in, votes in color_votes.items():
            if votes:
                color_map[color_in] = Counter(votes).most_common(1)[0][0]
            else:
                color_map[color_in] = color_in  # Identity mapping
                
        return color_map
    
    def apply_color_map(self, a, color_map):
        """Apply color mapping to grid."""
        arr = self.safe_npify(a)
        out = arr.copy()
        for old_color, new_color in color_map.items():
            out[arr == old_color] = new_color
        return out
    
    def detect_symmetry(self, a):
        """Detect symmetry in grid."""
        arr = self.safe_npify(a)
        symmetries = {}
        
        # Horizontal symmetry
        symmetries['horizontal'] = np.array_equal(arr, np.flipud(arr))
        # Vertical symmetry  
        symmetries['vertical'] = np.array_equal(arr, np.fliplr(arr))
        # Diagonal symmetry (for square grids)
        if arr.shape[0] == arr.shape[1]:
            symmetries['diagonal'] = np.array_equal(arr, arr.T)
        
        return symmetries
    
    def complete_symmetry(self, a, symmetry_type):
        """Complete partial symmetry."""
        arr = self.safe_npify(a)
        H, W = arr.shape
        out = arr.copy()
        bg = self.majority_color(arr)
        
        if symmetry_type == 'horizontal' and H > 1:
            mid = H // 2
            for i in range(mid):
                mirror_i = H - 1 - i
                for j in range(W):
                    if out[i, j] == bg and out[mirror_i, j] != bg:
                        out[i, j] = out[mirror_i, j]
                    elif out[mirror_i, j] == bg and out[i, j] != bg:
                        out[mirror_i, j] = out[i, j]
                        
        elif symmetry_type == 'vertical' and W > 1:
            mid = W // 2
            for j in range(mid):
                mirror_j = W - 1 - j
                for i in range(H):
                    if out[i, j] == bg and out[i, mirror_j] != bg:
                        out[i, j] = out[i, mirror_j]
                    elif out[i, mirror_j] == bg and out[i, j] != bg:
                        out[i, mirror_j] = out[i, j]
                        
        return out
    
    def get_symbolic_operations(self, train_pairs):
        """Get library of symbolic operations."""
        ops = {
            'identity': lambda x: x,
            'rotate_90': self.rotate_90,
            'rotate_180': self.rotate_180, 
            'rotate_270': self.rotate_270,
            'flip_h': self.flip_horizontal,
            'flip_v': self.flip_vertical,
            'transpose': self.transpose,
            'invert': self.invert_colors,
        }
        
        # Learn and add color mapping
        try:
            color_map = self.learn_color_map(train_pairs)
            ops['color_map'] = lambda x: self.apply_color_map(x, color_map)
        except:
            pass
        
        # Add symmetry operations
        for sym_type in ['horizontal', 'vertical']:
            ops[f'complete_{sym_type}_symmetry'] = (
                lambda x, st=sym_type: self.complete_symmetry(x, st)
            )
        
        return ops
    
    def score_solution(self, func, train_pairs):
        """Score a solution function on training pairs."""
        correct = 0
        total_accuracy = 0.0
        
        for input_grid, expected_output in train_pairs:
            try:
                prediction = func(input_grid)
                if self.grids_equal(prediction, expected_output):
                    correct += 1
                    total_accuracy += 1.0
                else:
                    # Partial credit for shape match and color accuracy
                    pred_arr = self.safe_npify(prediction)
                    exp_arr = self.safe_npify(expected_output)
                    if pred_arr.shape == exp_arr.shape:
                        accuracy = np.mean(pred_arr == exp_arr)
                        total_accuracy += accuracy
            except Exception:
                continue
                
        return correct, total_accuracy / max(1, len(train_pairs))
    
    def symbolic_solve(self, train_pairs, test_inputs, beam_size=20, max_depth=4):
        """Symbolic solver using beam search."""
        ops = self.get_symbolic_operations(train_pairs)
        op_list = list(ops.items())
        
        # Initialize beam with single operations
        beam = []
        for name, op in op_list:
            correct, accuracy = self.score_solution(op, train_pairs)
            beam.append(([name], op, correct, accuracy))
        
        beam.sort(key=lambda x: (x[2], x[3]), reverse=True)
        beam = beam[:beam_size]
        
        # Check for perfect solutions
        for names, func, correct, _ in beam:
            if correct == len(train_pairs):
                predictions = [func(test_input) for test_input in test_inputs]
                return predictions, names, "perfect"
        
        # Expand beam
        for depth in range(1, max_depth):
            new_beam = []
            
            for name_seq, base_func, _, _ in beam:
                for new_name, new_op in op_list:
                    # Avoid redundant sequences
                    if (len(name_seq) >= 2 and 
                        name_seq[-1] == new_name and name_seq[-2] == new_name):
                        continue
                    
                    new_seq = name_seq + [new_name]
                    composite_func = lambda x, f1=base_func, f2=new_op: f2(f1(x))
                    
                    # Score the composite function
                    correct, accuracy = self.score_solution(composite_func, train_pairs)
                    new_beam.append((new_seq, composite_func, correct, accuracy))
                    
                    # Early return if perfect
                    if correct == len(train_pairs):
                        predictions = [composite_func(test_input) for test_input in test_inputs]
                        return predictions, new_seq, "perfect"
            
            # Update beam
            beam.extend(new_beam)
            beam.sort(key=lambda x: (x[2], x[3]), reverse=True)
            beam = beam[:beam_size]
            
            if not beam:
                break
        
        # Return best found solution
        if beam:
            best_seq, best_func, correct, accuracy = beam[0]
            predictions = [best_func(test_input) for test_input in test_inputs]
            return predictions, best_seq, f"best_{correct}_{len(train_pairs)}"
        
        # Fallback: use identity
        return test_inputs, ["identity"], "fallback"
    
    # ========================
    # NEURAL SOLVER (Intuitionist) 
    # ========================
    
    def neural_solve(self, train_pairs, test_inputs):
        """
        Neural solver - simplified version.
        In practice, this would use a pre-trained neural network.
        """
        # Simple neural-inspired approach: learn patterns and apply
        try:
            # Learn output shape pattern
            output_shapes = [self.safe_npify(o).shape for _, o in train_pairs]
            if output_shapes:
                target_shape = Counter(output_shapes).most_common(1)[0][0]
            else:
                target_shape = (1, 1)
            
            # Learn color transformations
            color_map = self.learn_color_map(train_pairs)
            
            predictions = []
            for test_input in test_inputs:
                # Apply learned color mapping
                result = self.apply_color_map(test_input, color_map)
                
                # Resize to target shape if needed
                result_arr = self.safe_npify(result)
                if result_arr.shape != target_shape:
                    result = self.pad_to_shape(result, target_shape)
                
                predictions.append(result)
            
            return predictions, "neural_pattern", "neural"
            
        except Exception as e:
            # Fallback to symbolic approach
            return self.symbolic_solve(train_pairs, test_inputs)
    
    # ========================
    # DUAL SOLVER INTEGRATION
    # ========================
    
    def solve_task(self, task):
        """Solve a single ARC task using dual strategy."""
        # Extract training pairs and test inputs
        train_pairs = []
        for example in task.get('train', []):
            input_grid = example['input']
            output_grid = example['output'] 
            train_pairs.append((input_grid, output_grid))
        
        test_inputs = [t['input'] for t in task.get('test', [])]
        
        if not train_pairs:
            # No training examples - return test inputs as fallback
            return test_inputs, "no_training", "fallback"
        
        # Strategy 1: Symbolic solver
        symbolic_preds, symbolic_chain, symbolic_status = self.symbolic_solve(
            train_pairs, test_inputs
        )
        
        # Strategy 2: Neural solver  
        neural_preds, neural_chain, neural_status = self.neural_solve(
            train_pairs, test_inputs
        )
        
        # Evaluate both strategies on training data
        symbolic_correct, symbolic_accuracy = self.score_solution(
            lambda x: self.apply_chain(x, symbolic_chain), train_pairs
        )
        
        neural_correct, neural_accuracy = self.score_solution(
            lambda x: self.apply_chain(x, neural_chain), train_pairs
        )
        
        # Choose best strategy
        if symbolic_correct >= neural_correct and symbolic_accuracy >= neural_accuracy:
            final_predictions = symbolic_preds
            final_chain = symbolic_chain
            strategy_used = "symbolic"
            self.strategy_stats["symbolic"] += 1
        else:
            final_predictions = neural_preds
            final_chain = neural_chain  
            strategy_used = "neural"
            self.strategy_stats["neural"] += 1
        
        # Convert predictions to list format
        final_predictions_list = []
        for pred in final_predictions:
            if hasattr(pred, 'tolist'):
                final_predictions_list.append(pred.tolist())
            else:
                final_predictions_list.append(pred)
        
        return final_predictions_list, final_chain, strategy_used
    
    def apply_chain(self, x, chain):
        """Apply a chain of operations to input."""
        ops = self.get_symbolic_operations([])  # Get base operations
        result = self.safe_npify(x)
        
        for op_name in chain:
            if op_name in ops:
                result = ops[op_name](result)
        
        return result
    
    def get_solver_stats(self):
        """Get statistics about solver performance."""
        return dict(self.strategy_stats)

def run_competition():
    """Run the ARC competition with dual solver."""
    solver = DualARCSolver()
    submission = {}
    solver_stats = []
    
    # Find challenge file
    possible_files = [
        BASE / "arc-agi_test.json",
        BASE / "arc-agi_test_challenges.json",
        BASE / "test.json",
    ]
    
    challenge_file = None
    for file_path in possible_files:
        if file_path.exists():
            challenge_file = file_path
            break
    
    if challenge_file is None:
        print("No challenge file found. Creating empty submission.")
        with open(SUBMISSION_FILE, 'w') as f:
            json.dump({}, f)
        return
    
    print(f"Loading challenges from: {challenge_file}")
    with open(challenge_file, 'r') as f:
        challenges = json.load(f)
    
    # Process tasks
    total_tasks = len(challenges)
    print(f"Processing {total_tasks} tasks with dual solver...")
    
    start_time = time.time()
    processed_tasks = 0
    
    if isinstance(challenges, dict):
        items = list(challenges.items())
    else:
        items = list(enumerate(challenges))
    
    for task_id, task_data in items:
        task_start = time.time()
        
        try:
            predictions, chain, strategy = solver.solve_task(task_data)
            submission[str(task_id)] = predictions
            
            # Record solver statistics
            solver_stats.append({
                'task_id': str(task_id),
                'strategy': strategy,
                'chain': chain,
                'processing_time': time.time() - task_start
            })
            
        except Exception as e:
            print(f"Error solving task {task_id}: {e}")
            # Fallback: return test inputs
            test_inputs = task_data.get('test', [])
            fallback = [test_inputs[0]['input']] if test_inputs else [[[0]]]
            submission[str(task_id)] = fallback
            
            solver_stats.append({
                'task_id': str(task_id),
                'strategy': 'error',
                'chain': [],
                'processing_time': time.time() - task_start,
                'error': str(e)
            })
        
        processed_tasks += 1
        if processed_tasks % 10 == 0:
            elapsed = time.time() - start_time
            rate = processed_tasks / elapsed
            print(f"Progress: {processed_tasks}/{total_tasks} | Rate: {rate:.1f} tasks/sec")
    
    # Write submission and statistics
    with open(SUBMISSION_FILE, 'w') as f:
        json.dump(submission, f)
    
    # Write solver statistics
    stats_file = OUT_DIR / "solver_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump({
            'strategy_distribution': solver.get_solver_stats(),
            'task_details': solver_stats,
            'total_processing_time': time.time() - start_time,
            'average_time_per_task': (time.time() - start_time) / total_tasks
        }, f, indent=2)
    
    # Print summary
    total_time = time.time() - start_time
    strategy_stats = solver.get_solver_stats()
    
    print(f"\nğŸ�‰ Dual Solver Complete!")
    print(f"ğŸ“� Submission: {SUBMISSION_FILE}")
    print(f"ğŸ“Š Statistics: {stats_file}")
    print(f"â�±ï¸�  Total time: {total_time:.1f}s")
    print(f"ğŸ“ˆ Strategy distribution:")
    for strategy, count in strategy_stats.items():
        percentage = (count / total_tasks) * 100
        print(f"   - {strategy}: {count} tasks ({percentage:.1f}%)")

if __name__ == "__main__":
    run_competition()

