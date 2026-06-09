import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import itertools
from collections import defaultdict

class ARCVisualizer:
    """Visualize ARC tasks and grids"""
    
    def __init__(self):
        self.colors = [
            '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
            '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
        ]
    
    def visualize_grid(self, grid: List[List[int]], title: str = ""):
        """Visualize a single grid"""
        if not grid:
            return
            
        array = np.array(grid)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(array, cmap='tab10', vmin=0, vmax=9)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add cell values
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                ax.text(j, i, str(array[i, j]), ha='center', va='center', 
                       color='white' if array[i, j] in [0, 2, 3, 4, 7, 8] else 'black')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_task(self, task: Dict, task_id: str = ""):
        """Visualize a complete task with train and test pairs"""
        print(f"Task ID: {task_id}")
        
        # Visualize train pairs
        print("Training examples:")
        for i, example in enumerate(task['train']):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
            
            # Input
            input_grid = np.array(example['input'])
            ax1.imshow(input_grid, cmap='tab10', vmin=0, vmax=9)
            ax1.set_title(f'Train {i+1} - Input')
            ax1.set_xticks([])
            ax1.set_yticks([])
            
            # Output
            output_grid = np.array(example['output'])
            ax2.imshow(output_grid, cmap='tab10', vmin=0, vmax=9)
            ax2.set_title(f'Train {i+1} - Output')
            ax2.set_xticks([])
            ax2.set_yticks([])
            
            plt.tight_layout()
            plt.show()
        
        # Visualize test inputs
        print("Test inputs:")
        for i, test_input in enumerate(task['test']):
            plt.figure(figsize=(4, 4))
            test_grid = np.array(test_input['input'])
            plt.imshow(test_grid, cmap='tab10', vmin=0, vmax=9)
            plt.title(f'Test {i+1} - Input')
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.show()

class ARCBaseSolver:
    """Base class for ARC solvers"""
    
    def __init__(self):
        self.visualizer = ARCVisualizer()
    
    def load_data(self, challenges_path: str, solutions_path: Optional[str] = None) -> Tuple[Dict, Dict]:
        """Load challenges and optional solutions"""
        with open(challenges_path, 'r') as f:
            challenges = json.load(f)
        
        solutions = {}
        if solutions_path and Path(solutions_path).exists():
            with open(solutions_path, 'r') as f:
                solutions = json.load(f)
        
        return challenges, solutions
    
    def analyze_task(self, task: Dict) -> Dict:
        """Analyze a task to extract patterns and features"""
        analysis = {
            'input_shapes': [],
            'output_shapes': [],
            'color_changes': [],
            'patterns': [],
            'symmetry': [],
            'operations': []
        }
        
        for example in task['train']:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            analysis['input_shapes'].append(input_grid.shape)
            analysis['output_shapes'].append(output_grid.shape)
            
            # Analyze color transformations
            if input_grid.shape == output_grid.shape:
                changes = np.sum(input_grid != output_grid)
                analysis['color_changes'].append(changes)
            
            # Check for common patterns
            self._detect_patterns(input_grid, output_grid, analysis)
        
        return analysis
    
    def _detect_patterns(self, input_grid: np.ndarray, output_grid: np.ndarray, analysis: Dict):
        """Detect common patterns in transformations"""
        # Check for mirroring
        if np.array_equal(output_grid, np.fliplr(input_grid)):
            analysis['patterns'].append('mirror_horizontal')
        if np.array_equal(output_grid, np.flipud(input_grid)):
            analysis['patterns'].append('mirror_vertical')
        
        # Check for rotation
        for degrees in [90, 180, 270]:
            rotated = np.rot90(input_grid, k=degrees//90)
            if np.array_equal(output_grid, rotated):
                analysis['patterns'].append(f'rotate_{degrees}')
        
        # Check for color mapping
        if input_grid.shape == output_grid.shape:
            unique_pairs = set(zip(input_grid.flatten(), output_grid.flatten()))
            if len(unique_pairs) <= 5:  # Simple color mapping
                analysis['patterns'].append('color_mapping')
    
    def solve_task(self, task: Dict) -> List[Dict]:
        """Solve a task and return predictions for test inputs"""
        predictions = []
        
        for test_input in task['test']:
            input_grid = np.array(test_input['input'])
            
            # Make two attempts
            attempt_1 = self._make_attempt(task, input_grid, attempt=1)
            attempt_2 = self._make_attempt(task, input_grid, attempt=2)
            
            predictions.append({
                "attempt_1": attempt_1.tolist() if isinstance(attempt_1, np.ndarray) else attempt_1,
                "attempt_2": attempt_2.tolist() if isinstance(attempt_2, np.ndarray) else attempt_2
            })
        
        return predictions
    
    def _make_attempt(self, task: Dict, test_input: np.ndarray, attempt: int = 1) -> np.ndarray:
        """Make an attempt to solve the test input"""
        # Basic pattern matching approach
        analysis = self.analyze_task(task)
        
        # Try to apply detected patterns
        if 'mirror_horizontal' in analysis['patterns']:
            return np.fliplr(test_input)
        elif 'mirror_vertical' in analysis['patterns']:
            return np.flipud(test_input)
        elif any('rotate' in pattern for pattern in analysis['patterns']):
            for pattern in analysis['patterns']:
                if 'rotate_90' in pattern:
                    return np.rot90(test_input, k=1)
                elif 'rotate_180' in pattern:
                    return np.rot90(test_input, k=2)
                elif 'rotate_270' in pattern:
                    return np.rot90(test_input, k=3)
        
        # Default: return input unchanged (this will rarely be correct)
        return test_input

class ARCPatternSolver(ARCBaseSolver):
    """Enhanced solver with more pattern recognition"""
    
    def __init__(self):
        super().__init__()
    
    def _make_attempt(self, task: Dict, test_input: np.ndarray, attempt: int = 1) -> np.ndarray:
        """Enhanced pattern matching"""
        train_examples = task['train']
        
        # Try to learn from training examples
        if len(train_examples) > 0:
            # Analyze transformations between input and output
            transformations = self._learn_transformations(train_examples)
            
            # Apply learned transformation
            result = self._apply_transformations(test_input, transformations)
            if result is not None:
                return result
        
        # Fallback strategies
        if attempt == 1:
            # Try common transformations
            return self._try_common_transformations(test_input)
        else:
            # More creative attempt
            return self._creative_attempt(test_input, train_examples)
    
    def _learn_transformations(self, examples: List[Dict]) -> List:
        """Learn transformations from training examples"""
        transformations = []
        
        for example in examples:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            # Check if shapes match
            if input_grid.shape == output_grid.shape:
                # Learn pixel-wise transformations
                transformation = output_grid - input_grid
                transformations.append(('pixel_delta', transformation))
            
            # Learn color mappings
            color_map = {}
            for inp_val, out_val in zip(input_grid.flatten(), output_grid.flatten()):
                if inp_val not in color_map:
                    color_map[inp_val] = out_val
            transformations.append(('color_map', color_map))
        
        return transformations
    
    def _apply_transformations(self, input_grid: np.ndarray, transformations: List) -> Optional[np.ndarray]:
        """Apply learned transformations"""
        for transform_type, transform_data in transformations:
            if transform_type == 'pixel_delta':
                try:
                    return input_grid + transform_data
                except:
                    continue
            elif transform_type == 'color_map':
                result = np.zeros_like(input_grid)
                for i in range(input_grid.shape[0]):
                    for j in range(input_grid.shape[1]):
                        result[i, j] = transform_data.get(input_grid[i, j], input_grid[i, j])
                return result
        
        return None
    
    def _try_common_transformations(self, input_grid: np.ndarray) -> np.ndarray:
        """Try common transformations"""
        transformations = [
            lambda x: x,  # Identity
            lambda x: np.fliplr(x),  # Horizontal flip
            lambda x: np.flipud(x),  # Vertical flip
            lambda x: np.rot90(x, k=1),  # 90° rotation
            lambda x: np.rot90(x, k=2),  # 180° rotation
            lambda x: np.rot90(x, k=3),  # 270° rotation
        ]
        
        # Return the first transformation that changes something
        for transform in transformations:
            result = transform(input_grid)
            if not np.array_equal(result, input_grid):
                return result
        
        return input_grid
    
    def _creative_attempt(self, input_grid: np.ndarray, examples: List[Dict]) -> np.ndarray:
        """Make a more creative attempt based on examples"""
        # Simple approach: return the most common output pattern from examples
        output_shapes = [np.array(ex['output']).shape for ex in examples]
        if output_shapes:
            most_common_shape = max(set(output_shapes), key=output_shapes.count)
            return np.zeros(most_common_shape, dtype=int)
        
        return input_grid

def create_submission(solver: ARCBaseSolver, test_challenges_path: str, output_path: str):
    """Create submission file"""
    with open(test_challenges_path, 'r') as f:
        test_challenges = json.load(f)
    
    submission = {}
    
    for task_id, task in test_challenges.items():
        print(f"Solving task {task_id}...")
        predictions = solver.solve_task(task)
        submission[task_id] = predictions
    
    # Save submission
    with open(output_path, 'w') as f:
        json.dump(submission, f)
    
    print(f"Submission saved to {output_path}")

# Example usage
def main():
    # Initialize solver
    solver = ARCPatternSolver()
    
    # Load data
    challenges, solutions = solver.load_data(
        '/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json',
        '/kaggle/input/arc-prize-2025/arc-agi_training_solutions.json'
    )
    
    # Explore a specific task
    task_id = list(challenges.keys())[0]
    task = challenges[task_id]
    
    # Visualize the task
    solver.visualizer.visualize_task(task, task_id)
    
    # Analyze the task
    analysis = solver.analyze_task(task)
    print("Task analysis:", analysis)
    
    # Try to solve the task
    predictions = solver.solve_task(task)
    print("Predictions:", predictions)
    
    # Create submission (for actual competition)
    # create_submission(solver, 'arc-agi_test_challenges.json', 'submission.json')

if __name__ == "__main__":
    main()




