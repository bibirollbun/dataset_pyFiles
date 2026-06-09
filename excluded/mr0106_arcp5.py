# ## ğŸ“¦ 1. Imports & Configuration
# %%
import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import colors
from tqdm.auto import tqdm
import time
import unittest
import warnings

# Configuration
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 100)
plt.style.use('ggplot')


# %% [markdown]
# ## ğŸ�¨ 2. Constants & Visualization
# %%
ARC_COLORMAP = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']
)
ARC_NORM = colors.Normalize(vmin=0, vmax=9)

def plot_task(task, figsize=(10, 5)):
    """Visualize task with inputs and outputs"""
    n_train = len(task['train'])
    n_test = len(task['test'])
    fig, axs = plt.subplots(2, max(n_train, n_test), figsize=figsize)
    
    for i in range(n_train):
        axs[0,i].imshow(task['train'][i]['input'], cmap=ARC_COLORMAP, norm=ARC_NORM)
        axs[0,i].set_title(f'Train Input {i+1}')
        axs[1,i].imshow(task['train'][i]['output'], cmap=ARC_COLORMAP, norm=ARC_NORM)
        axs[1,i].set_title(f'Train Output {i+1}')
    
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ## ğŸ—ƒï¸� 3. Data Loading
# %%
class ARCDataset:
    def __init__(self, base_path='/kaggle/input/arc-prize-2025'):
        self.base_path = base_path
        self._data = {}
    
    def load(self, file_type):
        if file_type not in self._data:
            try:
                with open(f'{self.base_path}/arc-agi_{file_type}.json') as f:
                    self._data[file_type] = json.load(f)
            except Exception as e:
                print(f"Error loading {file_type}: {e}")
                self._data[file_type] = {}
        return self._data[file_type]
    
    @property
    def train(self):
        return self.load('training_challenges')
    
    @property
    def test(self):
        return self.load('test_challenges')


# %% [markdown]
# ## ğŸ§  4. Core Solver Class
# %%
class ARCSolver:
    def __init__(self, timeout=5):
        self.timeout = timeout
        self.strategies = [
            self._solve_color,
            self._solve_rotation,
            self._solve_fallback
        ]
    
    def solve(self, task):
        """Main solving method with timeout"""
        if not isinstance(task, dict) or 'test' not in task:
            return self._empty_solution(task)
            
        start_time = time.time()
        for strategy in self.strategies:
            if time.time() - start_time > self.timeout:
                break
            try:
                solution = strategy(task)
                if solution:
                    return solution
            except Exception:
                continue
        return self._solve_fallback(task)
    
    def _solve_color(self, task):
        """Color transformation strategy"""
        color_map = {}
        for example in task['train']:
            inp = np.array(example['input'])
            out = np.array(example['output'])
            for i, j in zip(inp.flatten(), out.flatten()):
                if i != j:
                    color_map[i] = j
        
        solutions = []
        for test in task['test']:
            grid = np.array(test['input'])
            output = np.copy(grid)
            for old, new in color_map.items():
                output[grid == old] = new
            solutions.append({
                'attempt_1': output.tolist(),
                'attempt_2': grid.tolist()
            })
        return solutions
    
    def _solve_rotation(self, task):
        """Rotation strategy"""
        solutions = []
        for test in task['test']:
            grid = np.array(test['input'])
            solutions.append({
                'attempt_1': np.rot90(grid).tolist(),
                'attempt_2': np.fliplr(grid).tolist()
            })
        return solutions
    
    def _solve_fallback(self, task):
        """Fallback strategy"""
        if not isinstance(task, dict) or 'test' not in task:
            return []
        return [{
            'attempt_1': test['input'],
            'attempt_2': test['input']
        } for test in task['test']]
    
    def _empty_solution(self, task):
        """Handle invalid tasks"""
        return []


# %% [markdown]
# ## ğŸ�† 5. Main Execution
# %%
def main():
    print("ğŸš€ Starting ARC Solution")
    
    # Initialize components
    dataset = ARCDataset()
    solver = ARCSolver()
    
    # Sample visualization
    if dataset.train:
        sample_id = next(iter(dataset.train))
        print(f"\nğŸ”� Visualizing sample task {sample_id}")
        plot_task(dataset.train[sample_id])
    
    # Generate submission
    print("\nğŸ“� Creating submission...")
    submission = {}
    for task_id, task in tqdm(dataset.test.items(), desc="Processing"):
        submission[task_id] = solver.solve(task)
    
    # Save results
    with open('/kaggle/working/submission.json', 'w') as f:
        json.dump(submission, f)
    
    print("\nâœ… Submission created successfully!")
    print("ğŸ“Š File saved to: /kaggle/working/submission.json")

if __name__ == '__main__':
    main()


# %% [markdown]
# ## ğŸ§ª Unit Tests (Final Version)
# %%
class TestARCSolution(unittest.TestCase):
    def setUp(self):
        """Test environment setup"""
        self.solver = ARCSolver()
        self.test_task = {
            'train': [{'input': [[0]], 'output': [[1]]}],
            'test': [{'input': [[0]]}]
        }

    def test_solver_exists(self):
        """Verify main solver function exists"""
        self.assertTrue(callable(self.solver.solve))

    def test_solution_format(self):
        """Verify solution format"""
        result = self.solver.solve(self.test_task)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)

    def test_empty_input(self):
        """Test empty input handling"""
        self.assertEqual(len(self.solver.solve({})), 0)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

