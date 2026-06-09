# Simple colors for numbers 0-9
colors = ['black', 'blue', 'red', 'green', 'yellow', 
          'gray', 'magenta', 'orange', 'cyan', 'brown']

def draw_grid(ax, grid, title):
    """Draw a single grid on the given Axes object with square tiles"""
    grid = np.array(grid)
    ax.set_title(title, fontsize=8)
    ax.set_xlim(0, grid.shape[1])
    ax.set_ylim(0, grid.shape[0])
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            color = colors[grid[i, j]]
            ax.fill([j, j+1, j+1, j], [i, i, i+1, i+1], color=color)

    for i in range(grid.shape[0] + 1):
        ax.axhline(i, color='white', linewidth=0.4)
    for j in range(grid.shape[1] + 1):
        ax.axvline(j, color='white', linewidth=0.4)


def show_task(task_name, task_data, solution_data=None):
    """Show all data of a task in one image, including test solutions if provided"""
    train = task_data['train']
    test = task_data['test']
    
    total_rows = max(len(train), len(test))
    fig, axs = plt.subplots(total_rows, 4, figsize=(12, 3 * total_rows))

    if total_rows == 1:
        axs = np.expand_dims(axs, 0)  # ensure 2D for consistency

    for i in range(total_rows):
        # TRAIN EXAMPLES
        if i < len(train):
            draw_grid(axs[i, 0], train[i]['input'], f"Train {i+1} - Input")
            draw_grid(axs[i, 1], train[i]['output'], f"Train {i+1} - Output")
        else:
            axs[i, 0].axis('off')
            axs[i, 1].axis('off')
        
        # TEST EXAMPLES
        if i < len(test):
            draw_grid(axs[i, 2], test[i]['input'], f"Test {i+1} - Input")
            if solution_data and i < len(solution_data):
                draw_grid(axs[i, 3], solution_data[i], f"Test {i+1} - Output (Solution)")
            else:
                axs[i, 3].text(0.5, 0.5, '??', fontsize=16, ha='center', va='center')
                axs[i, 3].set_title(f"Test {i+1} - Output?")
                axs[i, 3].axis('off')
        else:
            axs[i, 2].axis('off')
            axs[i, 3].axis('off')

    fig.suptitle(f"Task: {task_name}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()




import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random


DATA_DIR = '/kaggle/input/arc-prize-2025/'

with open(Path(DATA_DIR) / 'arc-agi_training_challenges.json') as f:
    train_challenges = json.load(f)
with open(Path(DATA_DIR) / 'arc-agi_training_solutions.json') as f:
    train_solutions = json.load(f)

with open(Path(DATA_DIR) / 'arc-agi_evaluation_challenges.json') as f:
    eval_challenges = json.load(f)
with open(Path(DATA_DIR) / 'arc-agi_evaluation_solutions.json') as f:
    eval_solutions = json.load(f)

with open(Path(DATA_DIR) / 'arc-agi_test_challenges.json') as f:
    test_challenges = json.load(f)

print(f"Training tasks: {len(train_challenges)}")
print(f"Training solutions: {len(train_solutions)}")
print(f"Evaluation tasks: {len(eval_challenges)}")
print(f"Evaluation solutions: {len(eval_solutions)}")
print(f"Test tasks: {len(test_challenges)}")


# Show first few tasks - training data
for _ in range(3):
    task_name = random.choice(list(train_challenges.keys()))
    task_data = train_challenges[task_name]
    task_solution = train_solutions[task_name]
    show_task(task_name, task_data, task_solution)


# Show first few tasks - evaluation data
for _ in range(3):
    task_name = random.choice(list(eval_challenges.keys()))
    task_data = eval_challenges[task_name]
    task_solution = eval_solutions[task_name]
    show_task(task_name, task_data, task_solution)


# Show first few tasks - test challenges
for _ in range(3):
    task_name = random.choice(list(test_challenges.keys()))
    task_data = test_challenges[task_name]
    show_task(task_name, task_data)


from pathlib import Path
from collections import Counter

class BiblicalReasoningEngine:
    """Optimized biblical reasoning system for ARC pattern recognition"""
    
    def __init__(self):
        self.principles = {
            'order': {'weight': 1.0, 'patterns': ['symmetry', 'rotation', 'reflection']},
            'transformation': {'weight': 0.9, 'patterns': ['rotation', 'reflection', 'color_change']},
            'completion': {'weight': 0.95, 'patterns': ['pattern_fill', 'symmetry_complete']},
            'relationship': {'weight': 0.9, 'patterns': ['spatial_relation', 'adjacency']},
            'wisdom': {'weight': 1.0, 'patterns': ['rule_application', 'pattern_learning']}
        }
        
        # Cache for pattern detection to avoid recomputation
        self._pattern_cache = {}
    
    def analyze_task(self, input_grid, train_examples=None):
        """Analyze task using biblical reasoning principles"""
        grid = np.array(input_grid)
        grid_key = str(grid.tobytes())  # Cache key
        
        if grid_key in self._pattern_cache:
            return self._pattern_cache[grid_key]
        
        analysis = {'patterns': {}, 'principles': {}, 'transformations': {}, 'confidence': 0.0}
        
        # Efficient pattern detection
        symmetry_score = self._detect_symmetry_fast(grid)
        rotation_score = self._detect_rotation_fast(grid, train_examples)
        completion_score = self._detect_completion_fast(grid)
        
        # Calculate principle scores efficiently
        principles_scores = {
            'order': symmetry_score * 0.8,
            'transformation': rotation_score * 0.7,
            'completion': completion_score * 0.9,
            'relationship': self._detect_spatial_fast(grid) * 0.6,
            'wisdom': 0.3  # Base wisdom score
        }
        
        # Apply biblical weights
        weighted_scores = {k: v * self.principles[k]['weight'] for k, v in principles_scores.items()}
        
        if weighted_scores:
            best_principle = max(weighted_scores.items(), key=lambda x: x[1])
            analysis['confidence'] = best_principle[1]
            analysis['principles'] = weighted_scores
            
            # Map to transformation
            principle_name = best_principle[0]
            transformation_map = {
                'order': 'order_symmetry',
                'transformation': 'divine_rotation', 
                'completion': 'completion_fill',
                'relationship': 'relationship_connect',
                'wisdom': 'wisdom_reflection'
            }
            analysis['transformations']['primary'] = transformation_map.get(principle_name, 'wisdom_reflection')
        
        # Cache result
        self._pattern_cache[grid_key] = analysis
        return analysis
    
    def _detect_symmetry_fast(self, grid):
        """Fast symmetry detection"""
        score = 0.0
        
        # Perfect symmetries
        if np.array_equal(grid, np.fliplr(grid)):
            score = 0.9
        elif np.array_equal(grid, np.flipud(grid)):
            score = 0.9
        elif grid.shape[0] == grid.shape[1] and np.array_equal(grid, grid.T):
            score = 0.95
        else:
            # Quick partial symmetry check
            h, w = grid.shape
            if w > 1:
                mid = w // 2
                left = grid[:, :mid]
                right = grid[:, -mid:] if w % 2 == 0 else grid[:, -(mid):]
                if left.shape == right.shape:
                    similarity = np.mean(left == np.fliplr(right))
                    if similarity > 0.6:
                        score = similarity * 0.7
        
        return min(1.0, score)
    
    def _detect_rotation_fast(self, grid, train_examples):
        """Fast rotation detection"""
        if train_examples:
            for example in train_examples[:3]:  # Limit to first 3 examples for speed
                input_ex = np.array(example['input'])
                output_ex = np.array(example['output'])
                
                if np.array_equal(output_ex, np.rot90(input_ex)):
                    return 0.8
        
        return 0.4 if grid.shape[0] == grid.shape[1] else 0.1
    
    def _detect_completion_fast(self, grid):
        """Fast completion detection"""
        empty_ratio = np.sum(grid == 0) / grid.size
        if 0.2 < empty_ratio < 0.7:
            return 0.6
        return 0.1
    
    def _detect_spatial_fast(self, grid):
        """Fast spatial pattern detection"""
        non_zero_count = np.sum(grid > 0)
        return 0.4 if non_zero_count > 2 else 0.1
    
    # Optimized transformation methods
    def _apply_rotation(self, grid):
        return np.rot90(grid)
    
    def _apply_reflection(self, grid):
        return np.fliplr(grid) if grid.shape[0] >= grid.shape[1] else np.flipud(grid)
    
    def _apply_completion(self, grid):
        output = grid.copy()
        non_zero = grid[grid > 0]
        if len(non_zero) > 0:
            most_common = Counter(non_zero).most_common(1)[0][0]
            empty_positions = np.argwhere(grid == 0)
            if len(empty_positions) > 0:
                fill_count = min(len(empty_positions) // 4 + 1, 3)  # Reduced filling
                for i in range(fill_count):
                    row, col = empty_positions[i]
                    output[row, col] = most_common
        return output
    
    def _apply_symmetry(self, grid):
        """Fast symmetry completion"""
        h, w = grid.shape
        output = grid.copy()
        
        # Horizontal symmetry completion
        if w % 2 == 0:
            mid = w // 2
            left = grid[:, :mid]
            right = grid[:, mid:]
            # Simple merge - take non-zero values
            for i in range(h):
                for j in range(mid):
                    if left[i, j] > 0 and right[i, mid-1-j] == 0:
                        output[i, mid + (mid-1-j)] = left[i, j]
                    elif right[i, mid-1-j] > 0 and left[i, j] == 0:
                        output[i, j] = right[i, mid-1-j]
        
        return output
    
    def _apply_change(self, grid):
        """Fast color change"""
        output = grid.copy()
        unique_values = np.unique(grid[grid > 0])
        if len(unique_values) >= 2:
            # Simple swap of first two values
            val1, val2 = unique_values[0], unique_values[1]
            output[grid == val1] = val2
            output[grid == val2] = val1
        return output

def solve_task_efficient(input_grid):
    """Efficient multi-algorithm solver"""
    try:
        grid = np.array(input_grid)
        
        # Generate 3 key candidates (reduced from 5 for speed)
        candidates = [
            np.rot90(grid),                    # Rotation
            complete_symmetry_fast(grid),      # Symmetry
            fill_pattern_fast(grid)            # Pattern fill
        ]
        
        # Fast ensemble voting
        best_candidate = ensemble_vote_fast(grid, candidates)
        return best_candidate.tolist()
        
    except Exception:
        return input_grid

def complete_symmetry_fast(grid):
    """Fast symmetry completion"""
    h, w = grid.shape
    
    # Try horizontal symmetry first (most common)
    if w % 2 == 0:
        mid = w // 2
        left = grid[:, :mid]
        right = grid[:, mid:]
        similarity = np.mean(left == np.fliplr(right))
        
        if similarity > 0.3:  # Lower threshold for speed
            result = grid.copy()
            result[:, mid:] = np.fliplr(left)
            return result
    
    return grid

def fill_pattern_fast(grid):
    """Fast pattern filling"""
    output = grid.copy()
    non_zero = grid[grid > 0]
    
    if len(non_zero) > 0:
        most_common = Counter(non_zero).most_common(1)[0][0]
        empty_positions = np.argwhere(grid == 0)
        
        if len(empty_positions) > 0:
            # Fill only a few strategic positions
            fill_count = min(len(empty_positions) // 5 + 1, 2)
            for i in range(fill_count):
                row, col = empty_positions[i]
                output[row, col] = most_common
    
    return output

def ensemble_vote_fast(original_grid, candidates):
    """Fast ensemble voting"""
    if not candidates:
        return original_grid
    
    # Simple scoring - prefer symmetry and meaningful changes
    best_candidate = original_grid
    best_score = 0
    
    for candidate in candidates:
        score = 0
        
        # Prefer changes
        if not np.array_equal(original_grid, candidate):
            score += 0.3
        
        # Prefer symmetry
        if (np.array_equal(candidate, np.fliplr(candidate)) or 
            np.array_equal(candidate, np.flipud(candidate))):
            score += 0.7
        
        if score > best_score:
            best_score = score
            best_candidate = candidate
    
    return best_candidate

# Load data
DATA_DIR = '/kaggle/input/arc-prize-2025/'

try:
    with open(Path(DATA_DIR) / 'arc-agi_test_challenges.json') as f:
        test_challenges = json.load(f)
    print(f"Loaded {len(test_challenges)} test tasks")
except FileNotFoundError:
    print("Test data not found. Using sample data.")
    test_challenges = {}

# Show sample tasks
if train_challenges:
    print("\n=== Training Examples ===")
    for _ in range(5):
        task_name = random.choice(list(train_challenges.keys()))
        task_data = train_challenges[task_name]
        task_solution = train_solutions[task_name]
        show_task(task_name, task_data, task_solution)
        


# Generate submission efficiently
submission = {}
reasoning_engine = BiblicalReasoningEngine()

for task_id, task_data in test_challenges.items():
    task_solutions = []
    
    for test_case in task_data['test']:
        input_grid = test_case['input']
        train_examples = task_data.get('train', [])
        
        # Attempt 1: Biblical reasoning
        try:
            grid = np.array(input_grid)
            analysis = reasoning_engine.analyze_task(grid, train_examples)
            
            if analysis['confidence'] > 0.4:
                transformation = analysis['transformations'].get('primary', 'wisdom_reflection')
                
                if transformation == 'divine_rotation':
                    predicted_output1 = reasoning_engine._apply_rotation(grid).tolist()
                elif transformation == 'wisdom_reflection':
                    predicted_output1 = reasoning_engine._apply_reflection(grid).tolist()
                elif transformation == 'completion_fill':
                    predicted_output1 = reasoning_engine._apply_completion(grid).tolist()
                elif transformation == 'order_symmetry':
                    predicted_output1 = reasoning_engine._apply_symmetry(grid).tolist()
                else:
                    predicted_output1 = reasoning_engine._apply_change(grid).tolist()
            else:
                predicted_output1 = solve_task_efficient(input_grid)
        except:
            predicted_output1 = solve_task_efficient(input_grid)
        
        # Attempt 2: Efficient solver
        try:
            predicted_output2 = solve_task_efficient(input_grid)
        except:
            predicted_output2 = input_grid
        
        task_solutions.append({
            "attempt_1": predicted_output1,
            "attempt_2": predicted_output2
        })
    
    submission[task_id] = task_solutions

print(f"\n=== Efficient Biblical Reasoning Results ===")
print(f"Generated solutions for {len(submission)} tasks")
print(f"Cache hits: {len(reasoning_engine._pattern_cache)}")

# Save submission
with open('submission.json', 'w') as f:
    json.dump(submission, f)

print(f"EFFICIENT Biblical ARC Solver - Submission Generated")
print(f"Tasks processed: {len(submission)}")
print(f"Submission saved as 'submission.json'")

# Test on sample
if test_challenges:
    sample_task = list(test_challenges.keys())[0]
    sample_input = test_challenges[sample_task]['test'][0]['input']
    
    print(f"\n=== Sample Test ===")
    print("Input:")
    for row in sample_input:
        print(row)
    
    result = solve_task_efficient(sample_input)
    print("Biblical reasoning output:")
    for row in result:
        print(row)



# Print submission structure for verification
print("\n=== Submission Structure ===")
print(f"Total tasks: {len(submission)}")

# Show first task structure
if submission:
    first_task_id = list(submission.keys())[0]
    first_task = submission[first_task_id]
    print(f"\nSample task: {first_task_id}")
    print(f"Number of test cases: {len(first_task)}")
    print(f"Structure: {list(first_task[0].keys())}")
    
    # Show first attempt output
    print(f"\nFirst attempt output shape: {np.array(first_task[0]['attempt_1']).shape}")
    print("First few values:")
    print(first_task[0]['attempt_1'][:3] if len(first_task[0]['attempt_1']) > 3 else first_task[0]['attempt_1'])

# Print file size
import os
if os.path.exists('submission.json'):
    file_size = os.path.getsize('submission.json')
    print(f"\nSubmission file size: {file_size:,} bytes")
    
    # Print first few lines of JSON
    with open('submission.json', 'r') as f:
        content = f.read()
        print(f"JSON preview (first 200 chars):")
        print(content[:1000] + "..." if len(content) > 1000 else content)

