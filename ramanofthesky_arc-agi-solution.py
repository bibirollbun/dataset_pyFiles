# Import essential libraries for data processing and visualization
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import pandas as pd
from collections import defaultdict, Counter
import os
from pathlib import Path
import copy
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib for better visualization
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("âœ“ Libraries imported successfully")
print("âœ“ Environment configured for ARC-AGI analysis")


import json
from pathlib import Path
import numpy as np
import pandas as pd # Import pandas to handle the CSV file

# 1. Corrected the file path for the Kaggle environment
# The data is in '/kaggle/input/', not the local './' directory.
data_path = Path('/kaggle/input/arc-prize-2025')

# Define data loading function for JSON files
def load_data(file_path):
    """Load JSON data with error handling and validation"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"âœ“ Loaded {file_path.name}: {len(data)} tasks")
        return data
    except FileNotFoundError:
        print(f"âœ— File not found: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"âœ— Invalid JSON format in {file_path}: {e}")
        return {}

# Load all available datasets
print("ğŸ”„ Loading ARC Prize 2025 datasets...")

training_challenges = load_data(data_path / 'arc-agi_training_challenges.json')
training_solutions = load_data(data_path / 'arc-agi_training_solutions.json')
evaluation_challenges = load_data(data_path / 'arc-agi_evaluation_challenges.json')
evaluation_solutions = load_data(data_path / 'arc-agi_evaluation_solutions.json')
test_challenges = load_data(data_path / 'arc-agi_test_challenges.json')

# 2. Correctly load the sample_submission.csv using pandas
try:
    sample_submission = pd.read_csv(data_path / 'sample_submission.csv')
    print(f"âœ“ Loaded sample_submission.csv: {len(sample_submission)} entries")
except FileNotFoundError:
    print(f"âœ— File not found: {data_path / 'sample_submission.csv'}")
    sample_submission = pd.DataFrame() # Create empty DataFrame on failure


print(f"\nğŸ“Š Complete Dataset Summary:")
print(f"Training challenges: {len(training_challenges)} tasks")
print(f"Training solutions: {len(training_solutions)} tasks")
print(f"Evaluation challenges: {len(evaluation_challenges)} tasks")
print(f"Evaluation solutions: {len(evaluation_solutions)} tasks")
print(f"Test challenges: {len(test_challenges)} tasks")
print(f"Sample submission entries: {len(sample_submission)} tasks")

# Validate data consistency
print(f"\nğŸ”� Data Validation:")
print(f"Training data consistency: {len(training_challenges) == len(training_solutions)}")
print(f"Evaluation data consistency: {len(evaluation_challenges) == len(evaluation_solutions)}")

# Display sample task structure
if training_challenges:
    sample_task_id = list(training_challenges.keys())[0]
    sample_task = training_challenges[sample_task_id]
    print(f"\nğŸ“‹ Sample Task Structure (ID: {sample_task_id}):")
    print(f"  Training pairs: {len(sample_task['train'])}")
    print(f"  Test inputs: {len(sample_task['test'])}")
    
    # Show structure of first training pair
    if sample_task['train']:
        first_pair = sample_task['train'][0]
        input_shape = np.array(first_pair['input']).shape
        output_shape = np.array(first_pair['output']).shape
        print(f"  First training input: {input_shape} grid")
        print(f"  First training output: {output_shape} grid")
        print(f"  Input colors: {sorted(list(set(np.array(first_pair['input']).flatten())))}")
        print(f"  Output colors: {sorted(list(set(np.array(first_pair['output']).flatten())))}")


def analyze_dataset_structure(challenges):
    """Analyze the structure and properties of the challenge dataset"""
    analysis = {
        'total_tasks': len(challenges),
        'grid_sizes': {'input': [], 'output': []},
        'colors_used': set(),
        'train_pair_counts': [],
        'test_input_counts': []
    }
    
    for task_id, task in challenges.items():
        # Analyze training pairs
        analysis['train_pair_counts'].append(len(task['train']))
        
        for pair in task['train']:
            input_grid = np.array(pair['input'])
            output_grid = np.array(pair['output'])
            
            analysis['grid_sizes']['input'].append(input_grid.shape)
            analysis['grid_sizes']['output'].append(output_grid.shape)
            analysis['colors_used'].update(input_grid.flatten())
            analysis['colors_used'].update(output_grid.flatten())
        
        # Analyze test inputs
        analysis['test_input_counts'].append(len(task['test']))
        for test_case in task['test']:
            test_input = np.array(test_case['input'])
            analysis['grid_sizes']['input'].append(test_input.shape)
            analysis['colors_used'].update(test_input.flatten())
    
    return analysis

def print_analysis_summary(analysis):
    """Print a comprehensive summary of dataset analysis"""
    print("ğŸ“Š ARC-AGI Dataset Analysis")
    print("=" * 50)
    print(f"Total tasks: {analysis['total_tasks']}")
    print(f"Colors used: {sorted(list(analysis['colors_used']))}")
    print(f"Training pairs per task: {min(analysis['train_pair_counts'])}-{max(analysis['train_pair_counts'])} (avg: {np.mean(analysis['train_pair_counts']):.1f})")
    print(f"Test inputs per task: {min(analysis['test_input_counts'])}-{max(analysis['test_input_counts'])} (avg: {np.mean(analysis['test_input_counts']):.1f})")
    
    # Grid size analysis
    input_sizes = analysis['grid_sizes']['input']
    output_sizes = analysis['grid_sizes']['output']
    
    print(f"\nğŸ“� Grid Size Analysis:")
    print(f"Input grid sizes: {len(set(input_sizes))} unique sizes")
    print(f"  - Smallest: {min(input_sizes)}")
    print(f"  - Largest: {max(input_sizes)}")
    
    if output_sizes:
        print(f"Output grid sizes: {len(set(output_sizes))} unique sizes")
        print(f"  - Smallest: {min(output_sizes)}")
        print(f"  - Largest: {max(output_sizes)}")
    
    # Size distribution
    size_counter = Counter(input_sizes)
    print(f"\nğŸ“ˆ Most common input sizes:")
    for size, count in size_counter.most_common(5):
        print(f"  - {size}: {count} grids")

# Perform analysis
dataset_analysis = analyze_dataset_structure(evaluation_challenges)
print_analysis_summary(dataset_analysis)


# Define ARC-AGI color scheme for professional visualization
ARC_COLORS = [
    '#000000',  # 0: Black
    '#0074D9',  # 1: Blue  
    '#FF4136',  # 2: Red
    '#2ECC40',  # 3: Green
    '#FFDC00',  # 4: Yellow
    '#AAAAAA',  # 5: Grey
    '#F012BE',  # 6: Magenta
    '#FF851B',  # 7: Orange
    '#7FDBFF',  # 8: Sky Blue
    '#870C25',  # 9: Brown
]

def plot_grid(grid, title="", ax=None, show_values=False):
    """
    Professional grid visualization with ARC color scheme
    
    Args:
        grid: 2D array representing the grid
        title: Title for the plot
        ax: Matplotlib axis object (creates new if None)
        show_values: Whether to display numeric values in cells
    """
    grid = np.array(grid)
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Create color map
    cmap = ListedColormap(ARC_COLORS[:10])
    
    # Plot grid
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=9)
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add title
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    # Add values if requested
    if show_values:
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                text = ax.text(j, i, str(grid[i, j]), 
                             ha="center", va="center", 
                             color="white", fontweight='bold', fontsize=8)
    
    return ax

def plot_task(task_data, task_id="", max_pairs=3):
    """
    Visualize a complete ARC task with training pairs and test input
    
    Args:
        task_data: Dictionary containing 'train' and 'test' data
        task_id: Task identifier for title
        max_pairs: Maximum number of training pairs to display
    """
    train_pairs = task_data['train'][:max_pairs]
    test_inputs = task_data['test']
    
    n_pairs = len(train_pairs)
    n_tests = len(test_inputs)
    
    # Calculate subplot layout
    cols = max(2, n_tests + 1)  # At least 2 cols for input/output, more for multiple tests
    rows = max(n_pairs, 1)
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot training pairs
    for i, pair in enumerate(train_pairs):
        if i < rows:
            plot_grid(pair['input'], f"Train {i+1} Input", axes[i, 0])
            plot_grid(pair['output'], f"Train {i+1} Output", axes[i, 1])
    
    # Plot test inputs
    for j, test in enumerate(test_inputs):
        if j < n_tests:
            col_idx = 2 + j if cols > 2 else 1
            row_idx = 0 if n_pairs == 0 else 0
            if col_idx < cols:
                plot_grid(test['input'], f"Test {j+1} Input", axes[row_idx, col_idx])
    
    # Hide unused subplots
    for i in range(rows):
        for j in range(cols):
            if (i >= n_pairs and j < 2) or (j >= 2 + n_tests):
                axes[i, j].set_visible(False)
    
    plt.suptitle(f"ARC Task: {task_id}", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

def quick_task_overview(challenges, n_tasks=3):
    """Display overview of multiple tasks for quick inspection"""
    task_ids = list(challenges.keys())[:n_tasks]
    
    for i, task_id in enumerate(task_ids):
        print(f"\n{'='*60}")
        print(f"TASK {i+1}/{n_tasks}: {task_id}")
        print(f"{'='*60}")
        
        task = challenges[task_id]
        print(f"Training pairs: {len(task['train'])}")
        print(f"Test inputs: {len(task['test'])}")
        
        # Show grid sizes
        for j, pair in enumerate(task['train']):
            input_shape = np.array(pair['input']).shape
            output_shape = np.array(pair['output']).shape
            print(f"  Pair {j+1}: {input_shape} â†’ {output_shape}")
        
        plot_task(task, task_id)

print("âœ“ Visualization system initialized")
print("âœ“ ARC color scheme configured")
print("âœ“ Grid plotting functions ready")


# Demonstrate visualization with sample tasks
print("ğŸ�¨ Demonstrating ARC-AGI Task Visualization")
print("="*50)

# Show a quick overview of the first few tasks
quick_task_overview(evaluation_challenges, n_tasks=2)


class ARCPatternDetector:
    """
    Comprehensive pattern detection system for ARC-AGI tasks
    """
    
    def __init__(self):
        self.transformation_types = [
            'rotation', 'reflection', 'scaling', 'translation',
            'color_change', 'object_extraction', 'pattern_completion',
            'symmetry', 'grid_division', 'object_counting'
        ]
    
    def analyze_grid_properties(self, grid):
        """Extract key properties of a grid"""
        grid = np.array(grid)
        properties = {
            'shape': grid.shape,
            'colors': set(grid.flatten()),
            'color_counts': dict(zip(*np.unique(grid, return_counts=True))),
            'background_color': self._find_background_color(grid),
            'non_zero_positions': np.argwhere(grid != 0).tolist(),
            'symmetries': self._check_symmetries(grid),
            'connected_components': self._find_connected_components(grid)
        }
        return properties
    
    def _find_background_color(self, grid):
        """Identify the background color (most frequent)"""
        colors, counts = np.unique(grid, return_counts=True)
        return colors[np.argmax(counts)]
    
    def _check_symmetries(self, grid):
        """Check for various symmetries in the grid"""
        symmetries = {}
        
        # Horizontal symmetry
        symmetries['horizontal'] = np.array_equal(grid, np.fliplr(grid))
        
        # Vertical symmetry
        symmetries['vertical'] = np.array_equal(grid, np.flipud(grid))
        
        # Rotational symmetry (90, 180, 270 degrees)
        symmetries['rot_90'] = np.array_equal(grid, np.rot90(grid, 1))
        symmetries['rot_180'] = np.array_equal(grid, np.rot90(grid, 2))
        symmetries['rot_270'] = np.array_equal(grid, np.rot90(grid, 3))
        
        return symmetries
    
    def _find_connected_components(self, grid):
        """Find connected components of non-background pixels"""
        background = self._find_background_color(grid)
        binary = (grid != background).astype(int)
        
        # Simple flood-fill to find components
        visited = np.zeros_like(binary)
        components = []
        
        def flood_fill(start_r, start_c, component):
            stack = [(start_r, start_c)]
            while stack:
                r, c = stack.pop()
                if (r < 0 or r >= binary.shape[0] or c < 0 or c >= binary.shape[1] or
                    visited[r, c] or binary[r, c] == 0):
                    continue
                
                visited[r, c] = 1
                component.append((r, c))
                
                # Add neighbors
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    stack.append((r + dr, c + dc))
        
        for r in range(binary.shape[0]):
            for c in range(binary.shape[1]):
                if binary[r, c] and not visited[r, c]:
                    component = []
                    flood_fill(r, c, component)
                    if component:
                        components.append(component)
        
        return components
    
    def compare_input_output(self, input_grid, output_grid):
        """Analyze the transformation from input to output"""
        input_props = self.analyze_grid_properties(input_grid)
        output_props = self.analyze_grid_properties(output_grid)
        
        transformation = {
            'size_change': input_props['shape'] != output_props['shape'],
            'color_change': input_props['colors'] != output_props['colors'],
            'background_change': input_props['background_color'] != output_props['background_color'],
            'component_count_change': len(input_props['connected_components']) != len(output_props['connected_components']),
            'transformations': []
        }
        
        # Check for basic geometric transformations
        input_arr = np.array(input_grid)
        output_arr = np.array(output_grid)
        
        if input_arr.shape == output_arr.shape:
            # Same size transformations
            if np.array_equal(output_arr, np.rot90(input_arr, 1)):
                transformation['transformations'].append('rotate_90')
            elif np.array_equal(output_arr, np.rot90(input_arr, 2)):
                transformation['transformations'].append('rotate_180')
            elif np.array_equal(output_arr, np.rot90(input_arr, 3)):
                transformation['transformations'].append('rotate_270')
            elif np.array_equal(output_arr, np.fliplr(input_arr)):
                transformation['transformations'].append('flip_horizontal')
            elif np.array_equal(output_arr, np.flipud(input_arr)):
                transformation['transformations'].append('flip_vertical')
            else:
                # Check for color transformations
                if input_arr.shape == output_arr.shape:
                    color_mapping = self._find_color_mapping(input_arr, output_arr)
                    if color_mapping:
                        transformation['transformations'].append('color_mapping')
                        transformation['color_mapping'] = color_mapping
        
        return transformation
    
    def _find_color_mapping(self, input_grid, output_grid):
        """Find systematic color transformations"""
        if input_grid.shape != output_grid.shape:
            return None
        
        mapping = {}
        for i in range(input_grid.shape[0]):
            for j in range(input_grid.shape[1]):
                input_color = input_grid[i, j]
                output_color = output_grid[i, j]
                
                if input_color in mapping:
                    if mapping[input_color] != output_color:
                        return None  # Inconsistent mapping
                else:
                    mapping[input_color] = output_color
        
        # Check if mapping is non-trivial
        non_trivial = any(k != v for k, v in mapping.items())
        return mapping if non_trivial else None
    
    def analyze_task(self, task_data):
        """Perform comprehensive analysis of an ARC task"""
        analysis = {
            'task_properties': {},
            'training_transformations': [],
            'common_patterns': [],
            'prediction_strategy': None
        }
        
        # Analyze each training pair
        for i, pair in enumerate(task_data['train']):
            transformation = self.compare_input_output(pair['input'], pair['output'])
            transformation['pair_index'] = i
            analysis['training_transformations'].append(transformation)
        
        # Find common patterns across training pairs
        analysis['common_patterns'] = self._find_common_patterns(analysis['training_transformations'])
        
        # Suggest prediction strategy
        analysis['prediction_strategy'] = self._suggest_strategy(analysis)
        
        return analysis
    
    def _find_common_patterns(self, transformations):
        """Identify patterns that appear across multiple training examples"""
        common = []
        
        # Check for consistent transformations
        if len(transformations) > 1:
            all_transforms = [t['transformations'] for t in transformations]
            
            # Find transformations that appear in all examples
            if all_transforms:
                common_transforms = set(all_transforms[0])
                for transforms in all_transforms[1:]:
                    common_transforms &= set(transforms)
                
                if common_transforms:
                    common.append({
                        'type': 'consistent_transformation',
                        'transformations': list(common_transforms)
                    })
        
        # Check for consistent color mappings
        color_mappings = [t.get('color_mapping') for t in transformations if t.get('color_mapping')]
        if len(color_mappings) > 1:
            # Look for consistent color rules
            consistent_mappings = {}
            for mapping in color_mappings:
                for k, v in mapping.items():
                    if k in consistent_mappings:
                        if consistent_mappings[k] != v:
                            break
                    else:
                        consistent_mappings[k] = v
                else:
                    continue
                break
            else:
                if consistent_mappings:
                    common.append({
                        'type': 'consistent_color_mapping',
                        'mapping': consistent_mappings
                    })
        
        return common
    
    def _suggest_strategy(self, analysis):
        """Suggest a strategy for solving the task based on analysis"""
        patterns = analysis['common_patterns']
        
        if not patterns:
            return "insufficient_pattern_detection"
        
        for pattern in patterns:
            if pattern['type'] == 'consistent_transformation':
                return {
                    'type': 'apply_transformation',
                    'transformations': pattern['transformations']
                }
            elif pattern['type'] == 'consistent_color_mapping':
                return {
                    'type': 'apply_color_mapping',
                    'mapping': pattern['mapping']
                }
        
        return "complex_pattern_requires_manual_analysis"

print("âœ“ Pattern detection system initialized")
print("âœ“ Grid analysis algorithms ready")
print("âœ“ Transformation detection implemented")


# Demonstrate pattern detection on actual ARC tasks
detector = ARCPatternDetector()

def analyze_sample_tasks(challenges, n_tasks=2):
    """Analyze pattern detection on sample tasks"""
    task_ids = list(challenges.keys())[:n_tasks]
    
    for task_id in task_ids:
        print(f"\n{'='*70}")
        print(f"PATTERN ANALYSIS: {task_id}")
        print(f"{'='*70}")
        
        task = challenges[task_id]
        analysis = detector.analyze_task(task)
        
        print(f"Training pairs: {len(task['train'])}")
        print(f"Test inputs: {len(task['test'])}")
        
        # Show transformation analysis for each training pair
        for i, transform in enumerate(analysis['training_transformations']):
            print(f"\nğŸ“‹ Training Pair {i+1}:")
            print(f"  Size change: {transform['size_change']}")
            print(f"  Color change: {transform['color_change']}")
            print(f"  Transformations: {transform['transformations']}")
            
            if 'color_mapping' in transform:
                print(f"  Color mapping: {transform['color_mapping']}")
        
        # Show common patterns
        print(f"\nğŸ”� Common Patterns:")
        if analysis['common_patterns']:
            for pattern in analysis['common_patterns']:
                print(f"  - {pattern['type']}: {pattern}")
        else:
            print("  - No clear common patterns detected")
        
        # Show strategy
        print(f"\nğŸ�¯ Suggested Strategy: {analysis['prediction_strategy']}")

# Run analysis on sample tasks
print("ğŸ”¬ Running Pattern Detection Analysis")
analyze_sample_tasks(evaluation_challenges, n_tasks=3)


class ARCSolver:
    """
    Advanced ARC-AGI solver with multiple solving strategies
    """
    
    def __init__(self):
        self.detector = ARCPatternDetector()
        self.solving_strategies = [
            'geometric_transformation',
            'color_mapping',
            'pattern_completion',
            'object_extraction',
            'size_adjustment',
            'heuristic_fallback'
        ]
    
    def solve_task(self, task_data):
        """
        Generate predictions for a task using multiple strategies
        
        Returns:
            List of predictions (attempt_1, attempt_2) for each test input
        """
        # Analyze the task to understand patterns
        analysis = self.detector.analyze_task(task_data)
        
        predictions = []
        
        for test_case in task_data['test']:
            test_input = test_case['input']
            
            # Generate multiple solution attempts
            attempt_1 = self._apply_primary_strategy(test_input, analysis, task_data)
            attempt_2 = self._apply_secondary_strategy(test_input, analysis, task_data)
            
            predictions.append({
                'attempt_1': attempt_1,
                'attempt_2': attempt_2
            })
        
        return predictions
    
    def _apply_primary_strategy(self, test_input, analysis, task_data):
        """Apply the most likely transformation based on analysis"""
        strategy = analysis['prediction_strategy']
        
        if isinstance(strategy, dict):
            if strategy['type'] == 'apply_transformation':
                return self._apply_geometric_transformations(test_input, strategy['transformations'])
            elif strategy['type'] == 'apply_color_mapping':
                return self._apply_color_mapping(test_input, strategy['mapping'])
        
        # Fallback: try pattern from first training example
        if task_data['train']:
            return self._copy_output_pattern(test_input, task_data['train'][0])
        
        # Ultimate fallback: return input unchanged
        return test_input
    
    def _apply_secondary_strategy(self, test_input, analysis, task_data):
        """Apply alternative strategy for second attempt"""
        # Try different approach for diversity
        
        # Strategy 1: Try size-based transformation
        if task_data['train']:
            first_pair = task_data['train'][0]
            input_shape = np.array(first_pair['input']).shape
            output_shape = np.array(first_pair['output']).shape
            
            if input_shape != output_shape:
                return self._apply_size_transformation(test_input, input_shape, output_shape, first_pair)
        
        # Strategy 2: Try alternative geometric transformation
        alternative_transforms = ['rotate_90', 'flip_horizontal', 'flip_vertical']
        for transform in alternative_transforms:
            result = self._apply_geometric_transformations(test_input, [transform])
            if not np.array_equal(result, test_input):
                return result
        
        # Fallback: return modified input
        return self._apply_simple_modifications(test_input)
    
    def _apply_geometric_transformations(self, grid, transformations):
        """Apply geometric transformations to grid"""
        result = np.array(grid)
        
        for transform in transformations:
            if transform == 'rotate_90':
                result = np.rot90(result, 1)
            elif transform == 'rotate_180':
                result = np.rot90(result, 2)
            elif transform == 'rotate_270':
                result = np.rot90(result, 3)
            elif transform == 'flip_horizontal':
                result = np.fliplr(result)
            elif transform == 'flip_vertical':
                result = np.flipud(result)
        
        return result.tolist()
    
    def _apply_color_mapping(self, grid, color_mapping):
        """Apply color mapping transformation"""
        result = np.array(grid)
        
        for old_color, new_color in color_mapping.items():
            result[result == old_color] = new_color
        
        return result.tolist()
    
    def _copy_output_pattern(self, test_input, training_pair):
        """Copy pattern from training example"""
        # If same size, try to find and apply pattern
        train_input = np.array(training_pair['input'])
        train_output = np.array(training_pair['output'])
        test_array = np.array(test_input)
        
        if train_input.shape == test_array.shape == train_output.shape:
            # Try direct color mapping
            color_mapping = self.detector._find_color_mapping(train_input, train_output)
            if color_mapping:
                return self._apply_color_mapping(test_input, color_mapping)
        
        # Fallback: return training output if it seems reasonable
        if train_output.shape[0] <= 30 and train_output.shape[1] <= 30:
            return train_output.tolist()
        
        return test_input
    
    def _apply_size_transformation(self, test_input, input_shape, output_shape, training_pair):
        """Apply size-based transformation"""
        test_array = np.array(test_input)
        
        # Simple extraction: if output is smaller, try to extract central region
        if all(output_shape[i] <= input_shape[i] for i in range(2)):
            h_start = (test_array.shape[0] - output_shape[0]) // 2
            w_start = (test_array.shape[1] - output_shape[1]) // 2
            
            if h_start >= 0 and w_start >= 0:
                extracted = test_array[h_start:h_start+output_shape[0], 
                                    w_start:w_start+output_shape[1]]
                if extracted.shape == output_shape:
                    return extracted.tolist()
        
        # Fallback: resize or crop/pad to match output shape
        return self._resize_grid(test_input, output_shape)
    
    def _resize_grid(self, grid, target_shape):
        """Resize grid to target shape (simple crop/pad)"""
        grid_array = np.array(grid)
        result = np.zeros(target_shape, dtype=grid_array.dtype)
        
        # Copy what fits
        copy_h = min(grid_array.shape[0], target_shape[0])
        copy_w = min(grid_array.shape[1], target_shape[1])
        
        result[:copy_h, :copy_w] = grid_array[:copy_h, :copy_w]
        
        return result.tolist()
    
    def _apply_simple_modifications(self, grid):
        """Apply simple modifications for fallback"""
        grid_array = np.array(grid)
        
        # Try inverting colors (simple transformation)
        max_color = 9
        inverted = max_color - grid_array
        
        return inverted.tolist()
    
    def evaluate_predictions(self, predictions, ground_truth):
        """Evaluate prediction accuracy against ground truth"""
        if len(predictions) != len(ground_truth):
            return 0.0
        
        correct = 0
        total = len(predictions)
        
        for pred, truth in zip(predictions, ground_truth):
            # Check if either attempt matches ground truth
            attempt_1_correct = np.array_equal(pred['attempt_1'], truth)
            attempt_2_correct = np.array_equal(pred['attempt_2'], truth)
            
            if attempt_1_correct or attempt_2_correct:
                correct += 1
        
        return correct / total if total > 0 else 0.0

print("âœ“ Advanced solver system initialized")
print("âœ“ Multiple solving strategies implemented")
print("âœ“ Evaluation framework ready")


# Initialize solver and test on evaluation data
solver = ARCSolver()

def validate_solver_performance(challenges, solutions, n_tasks=10):
    """
    Validate solver performance on a subset of tasks
    """
    print(f"ğŸ§ª Testing solver on {n_tasks} evaluation tasks")
    print("="*60)
    
    task_ids = list(challenges.keys())[:n_tasks]
    total_score = 0
    total_tasks = 0
    detailed_results = []
    
    for i, task_id in enumerate(task_ids):
        task_data = challenges[task_id]
        ground_truth = solutions[task_id]
        
        # Generate predictions
        predictions = solver.solve_task(task_data)
        
        # Evaluate accuracy
        accuracy = solver.evaluate_predictions(predictions, ground_truth)
        total_score += accuracy
        total_tasks += 1
        
        result = {
            'task_id': task_id,
            'accuracy': accuracy,
            'test_inputs': len(task_data['test']),
            'training_pairs': len(task_data['train'])
        }
        detailed_results.append(result)
        
        print(f"Task {i+1:2d} ({task_id}): {accuracy:.1%} | "
              f"Train: {len(task_data['train'])} | Test: {len(task_data['test'])}")
    
    overall_accuracy = total_score / total_tasks if total_tasks > 0 else 0
    
    print(f"\nğŸ“Š Overall Results:")
    print(f"Tasks tested: {total_tasks}")
    print(f"Overall accuracy: {overall_accuracy:.1%}")
    print(f"Perfect scores: {sum(1 for r in detailed_results if r['accuracy'] == 1.0)}")
    print(f"Partial scores: {sum(1 for r in detailed_results if 0 < r['accuracy'] < 1.0)}")
    print(f"Failed tasks: {sum(1 for r in detailed_results if r['accuracy'] == 0.0)}")
    
    return overall_accuracy, detailed_results

# Run validation
print("ğŸš€ Starting solver validation on evaluation dataset")
accuracy, results = validate_solver_performance(evaluation_challenges, evaluation_solutions, n_tasks=20)


def generate_submission(test_challenges, solver, output_file='submission.json'):
    """
    Generate competition submission file with predictions for all test challenges
    
    Args:
        test_challenges: Dictionary of test tasks
        solver: ARCSolver instance
        output_file: Output file path
    
    Returns:
        Dictionary containing all predictions in competition format
    """
    print(f"ğŸ�¯ Generating predictions for {len(test_challenges)} test tasks")
    print("="*60)
    
    submission = {}
    
    for i, (task_id, task_data) in enumerate(test_challenges.items()):
        if i % 50 == 0:
            print(f"Processing task {i+1}/{len(test_challenges)}: {task_id}")
        
        # Generate predictions for this task
        predictions = solver.solve_task(task_data)
        
        # Format predictions according to competition requirements
        task_predictions = []
        for pred in predictions:
            task_predictions.append({
                'attempt_1': pred['attempt_1'],
                'attempt_2': pred['attempt_2']
            })
        
        submission[task_id] = task_predictions
    
    # Save submission file
    with open(output_file, 'w') as f:
        json.dump(submission, f, indent=2)
    
    print(f"\nâœ… Submission generated successfully!")
    print(f"ğŸ“� Saved to: {output_file}")
    print(f"ğŸ“Š Total tasks: {len(submission)}")
    print(f"ğŸ“Š Total predictions: {sum(len(preds) for preds in submission.values())}")
    
    return submission

def validate_submission_format(submission_file, test_challenges):
    """
    Validate that submission file meets competition requirements
    """
    print(f"ğŸ”� Validating submission format: {submission_file}")
    
    with open(submission_file, 'r') as f:
        submission = json.load(f)
    
    errors = []
    warnings = []
    
    # Check all task IDs are present
    expected_tasks = set(test_challenges.keys())
    submitted_tasks = set(submission.keys())
    
    missing_tasks = expected_tasks - submitted_tasks
    extra_tasks = submitted_tasks - expected_tasks
    
    if missing_tasks:
        errors.append(f"Missing tasks: {list(missing_tasks)[:5]}...")
    if extra_tasks:
        warnings.append(f"Extra tasks: {list(extra_tasks)[:5]}...")
    
    # Check submission format for each task
    for task_id, task_data in test_challenges.items():
        if task_id not in submission:
            continue
        
        expected_outputs = len(task_data['test'])
        submitted_outputs = len(submission[task_id])
        
        if expected_outputs != submitted_outputs:
            errors.append(f"Task {task_id}: expected {expected_outputs} outputs, got {submitted_outputs}")
        
        # Check each prediction has attempt_1 and attempt_2
        for i, pred in enumerate(submission[task_id]):
            if 'attempt_1' not in pred or 'attempt_2' not in pred:
                errors.append(f"Task {task_id} output {i}: missing attempt_1 or attempt_2")
    
    # Print validation results
    if errors:
        print("â�Œ Validation FAILED:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
    else:
        print("âœ… Validation PASSED: Submission format is correct")
    
    if warnings:
        print("âš ï¸�  Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    return len(errors) == 0

# Generate final submission
print("ğŸ�� Creating final competition submission")
final_submission = generate_submission(test_challenges, solver)

# Validate submission format
is_valid = validate_submission_format('submission.json', test_challenges)

