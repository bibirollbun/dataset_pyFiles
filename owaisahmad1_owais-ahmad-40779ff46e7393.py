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


# Install kaggle first
!pip install kaggle

# Download the competition data
!kaggle competitions download -c arc-prize-2025

# Unzip the files
!unzip arc-prize-2025.zip


import os
import json
import requests
from pathlib import Path

# Create data directory
!mkdir -p /kaggle/input/arc-prize-2025

# If you're in Kaggle notebook, use this:
if os.path.exists('/kaggle/input/arc-prize-2025'):
    print("Data directory exists!")
    # List available files
    for file in os.listdir('/kaggle/input/arc-prize-2025'):
        print(f"Found: {file}")


import json
import os
from pathlib import Path

def find_and_load_data():
    """Find and load ARC data with multiple path attempts"""
    
    possible_paths = [
        '/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json',
        './arc-agi_training_challenges.json',
        '../input/arc-prize-2025/arc-agi_training_challenges.json',
        '/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json'
    ]
    
    for file_path in possible_paths:
        if os.path.exists(file_path):
            print(f"Found data at: {file_path}")
            with open(file_path, 'r') as f:
                return json.load(f)
    
    # If no file found, show available files
    print("Available files in current directory:")
    for file in os.listdir('.'):
        print(f"  {file}")
    
    print("\nAvailable files in /kaggle/input:")
    if os.path.exists('/kaggle/input'):
        for comp in os.listdir('/kaggle/input'):
            print(f"  /kaggle/input/{comp}")
            comp_path = f'/kaggle/input/{comp}'
            if os.path.isdir(comp_path):
                for file in os.listdir(comp_path):
                    print(f"    {file}")
    
    raise FileNotFoundError("Could not find ARC data files. Please download them first.")

# Use this instead
try:
    train_challenges = find_and_load_data()
    print("Successfully loaded data!")
    print(f"Number of tasks: {len(train_challenges)}")
except Exception as e:
    print(f"Error: {e}")
    print("\nPlease download the data first using:")
    print("!kaggle competitions download -c arc-prize-2025")
    print("!unzip arc-prize-2025.zip")


# Complete setup script
import os
import subprocess

def setup_arc_environment():
    """Setup the ARC competition environment"""
    
    # Check if we're in Kaggle
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        print("Running in Kaggle notebook - data should be available")
        return True
    
    # Try to download data
    print("Setting up ARC competition environment...")
    
    # Install kaggle API
    try:
        subprocess.run(['pip', 'install', 'kaggle'], check=True)
    except:
        print("Could not install kaggle")
    
    # Download data
    try:
        print("Downloading competition data...")
        result = subprocess.run([
            'kaggle', 'competitions', 'download', '-c', 'arc-prize-2025'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Download successful!")
            # Unzip
            subprocess.run(['unzip', 'arc-prize-2025.zip'], check=True)
            return True
        else:
            print(f"Download failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error downloading: {e}")
        print("\nPlease manually download from:")
        print("https://www.kaggle.com/competitions/arc-prize-2025/data")
        return False

# Run setup
if setup_arc_environment():
    print("Environment ready!")
else:
    print("Please manually download the data files")


# Test if data is loaded properly
def test_data_loading():
    try:
        data = find_and_load_data()
        first_key = list(data.keys())[0]
        first_task = data[first_key]
        
        print("✅ Data loaded successfully!")
        print(f"First task ID: {first_key}")
        print(f"Train pairs: {len(first_task['train'])}")
        print(f"Test inputs: {len(first_task['test'])}")
        
        # Show first train pair
        print("\nFirst train pair:")
        print("Input:")
        for row in first_task['train'][0]['input']:
            print(' '.join(str(x) for x in row))
        print("Output:")  
        for row in first_task['train'][0]['output']:
            print(' '.join(str(x) for x in row))
            
    except Exception as e:
        print(f"❌ Error: {e}")

test_data_loading()


import json
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def analyze_dataset(data):
    """Comprehensive analysis of the ARC dataset"""
    print(f"Total tasks: {len(data)}")
    
    task_stats = {
        'train_pairs': [],
        'test_inputs': [],
        'grid_sizes': [],
        'colors_used': []
    }
    
    for task_id, task in data.items():
        # Count train pairs
        task_stats['train_pairs'].append(len(task['train']))
        
        # Count test inputs  
        task_stats['test_inputs'].append(len(task['test']))
        
        # Analyze grid sizes and colors
        for pair in task['train']:
            input_grid = pair['input']
            output_grid = pair['output']
            
            # Grid sizes
            task_stats['grid_sizes'].append((len(input_grid), len(input_grid[0])))
            task_stats['grid_sizes'].append((len(output_grid), len(output_grid[0])))
            
            # Colors used
            for row in input_grid:
                task_stats['colors_used'].extend(row)
            for row in output_grid:
                task_stats['colors_used'].extend(row)
    
    return task_stats

def plot_statistics(stats):
    """Visualize dataset statistics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Train pairs distribution
    axes[0,0].hist(stats['train_pairs'], bins=10, alpha=0.7)
    axes[0,0].set_title('Train Pairs per Task')
    axes[0,0].set_xlabel('Number of pairs')
    axes[0,0].set_ylabel('Frequency')
    
    # Test inputs distribution
    axes[0,1].hist(stats['test_inputs'], bins=10, alpha=0.7)
    axes[0,1].set_title('Test Inputs per Task')
    axes[0,1].set_xlabel('Number of test inputs')
    
    # Colors distribution
    color_counts = Counter(stats['colors_used'])
    axes[1,0].bar(color_counts.keys(), color_counts.values())
    axes[1,0].set_title('Color Frequency (0-9)')
    axes[1,0].set_xlabel('Color')
    axes[1,0].set_ylabel('Count')
    
    # Grid sizes
    heights = [h for h, w in stats['grid_sizes']]
    widths = [w for h, w in stats['grid_sizes']]
    axes[1,1].scatter(widths, heights, alpha=0.5)
    axes[1,1].set_title('Grid Sizes')
    axes[1,1].set_xlabel('Width')
    axes[1,1].set_ylabel('Height')
    
    plt.tight_layout()
    plt.show()

# Load and analyze
data = find_and_load_data()
stats = analyze_dataset(data)
plot_statistics(stats)


def detect_symmetry(grid):
    """Detect symmetry patterns in grid"""
    grid_np = np.array(grid)
    
    symmetries = {
        'horizontal': np.array_equal(grid_np, grid_np[::-1]),
        'vertical': np.array_equal(grid_np, grid_np[:, ::-1]),
        'diagonal': np.array_equal(grid_np, grid_np.T)
    }
    
    return symmetries

def detect_color_transformations(input_grid, output_grid):
    """Detect color mapping patterns"""
    input_flat = [pixel for row in input_grid for pixel in row]
    output_flat = [pixel for row in output_grid for pixel in row]
    
    # Find color mappings
    color_map = {}
    for i, o in zip(input_flat, output_flat):
        if i not in color_map:
            color_map[i] = o
        elif color_map[i] != o:
            color_map[i] = None  # Inconsistent mapping
    
    # Remove inconsistent mappings
    color_map = {k: v for k, v in color_map.items() if v is not None}
    
    return color_map

def detect_object_movement(input_grid, output_grid):
    """Detect if objects are moving"""
    input_np = np.array(input_grid)
    output_np = np.array(output_grid)
    
    # Find non-zero (colored) positions
    input_objects = np.argwhere(input_np > 0)
    output_objects = np.argwhere(output_np > 0)
    
    if len(input_objects) == len(output_objects):
        # Calculate centroid movement
        input_center = np.mean(input_objects, axis=0)
        output_center = np.mean(output_objects, axis=0)
        movement = output_center - input_center
        
        return {'movement': movement, 'object_count': len(input_objects)}
    
    return None

def analyze_task_patterns(task):
    """Analyze patterns in a single task"""
    patterns = []
    
    for train_pair in task['train']:
        input_grid = train_pair['input']
        output_grid = train_pair['output']
        
        pattern = {
            'symmetry': detect_symmetry(output_grid),
            'color_mapping': detect_color_transformations(input_grid, output_grid),
            'movement': detect_object_movement(input_grid, output_grid),
            'size_change': (len(output_grid) != len(input_grid) or 
                          len(output_grid[0]) != len(input_grid[0]))
        }
        patterns.append(pattern)
    
    return patterns


class BasicARCSolver:
    def __init__(self):
        self.pattern_detectors = [
            self.solve_color_mapping,
            self.solve_symmetry,
            self.solve_object_movement,
            self.solve_size_change
        ]
    
    def solve_color_mapping(self, task):
        """Solve tasks with simple color mappings"""
        # Analyze color patterns from train pairs
        color_maps = []
        for pair in task['train']:
            color_map = detect_color_transformations(pair['input'], pair['output'])
            if color_map:
                color_maps.append(color_map)
        
        # Find consistent color mapping
        if color_maps:
            # Use the most common mapping
            consistent_map = {}
            for color in range(10):
                mappings = [cm.get(color, color) for cm in color_maps]
                if all(m == mappings[0] for m in mappings):
                    consistent_map[color] = mappings[0]
            
            if consistent_map:
                # Apply to test input
                test_input = task['test'][0]
                output = []
                for row in test_input:
                    new_row = [consistent_map.get(pixel, pixel) for pixel in row]
                    output.append(new_row)
                return output
        
        return None
    
    def solve_symmetry(self, task):
        """Solve symmetry-based tasks"""
        symmetries = []
        for pair in task['train']:
            sym = detect_symmetry(pair['output'])
            symmetries.append(sym)
        
        # Check if all outputs have the same symmetry
        if all(s == symmetries[0] for s in symmetries):
            test_input = task['test'][0]
            test_np = np.array(test_input)
            
            if symmetries[0]['horizontal']:
                return test_np[::-1].tolist()
            elif symmetries[0]['vertical']:
                return test_np[:, ::-1].tolist()
            elif symmetries[0]['diagonal']:
                return test_np.T.tolist()
        
        return None
    
    def solve_object_movement(self, task):
        """Solve object movement tasks"""
        movements = []
        for pair in task['train']:
            movement = detect_object_movement(pair['input'], pair['output'])
            if movement:
                movements.append(movement['movement'])
        
        if movements and len(set(tuple(m) for m in movements)) == 1:
            # Consistent movement pattern
            movement = movements[0]
            test_input = np.array(task['test'][0])
            
            # Simple translation (this is basic - needs improvement)
            output = np.zeros_like(test_input)
            non_zero_positions = np.argwhere(test_input > 0)
            
            for pos in non_zero_positions:
                new_pos = pos + movement
                if (0 <= new_pos[0] < output.shape[0] and 
                    0 <= new_pos[1] < output.shape[1]):
                    output[int(new_pos[0]), int(new_pos[1])] = test_input[pos[0], pos[1]]
            
            return output.tolist()
        
        return None
    
    def solve_size_change(self, task):
        """Handle tasks with size changes"""
        # For now, return same size as input
        test_input = task['test'][0]
        return [[0 for _ in range(len(test_input[0]))] for _ in range(len(test_input))]
    
    def solve_task(self, task):
        """Try multiple solving strategies"""
        for detector in self.pattern_detectors:
            solution = detector(task)
            if solution is not None:
                return solution
        
        # Fallback: return same as input
        test_input = task['test'][0]
        return [[0 for _ in range(len(test_input[0]))] for _ in range(len(test_input))]


# Test on a few tasks
def test_solver_on_samples(solver, data, num_samples=5):
    """Test our solver on sample tasks"""
    sample_tasks = list(data.items())[:num_samples]
    
    for task_id, task in sample_tasks:
        print(f"\n=== Testing Task: {task_id} ===")
        
        # Show train examples
        for i, pair in enumerate(task['train']):
            print(f"Train {i+1}:")
            print("Input:")
            for row in pair['input']:
                print(' '.join(str(x) for x in row))
            print("Output:")
            for row in pair['output']:
                print(' '.join(str(x) for x in row))
            print()
        
        # Try to solve
        try:
            prediction = solver.solve_task(task)
            print("Prediction:")
            for row in prediction:
                print(' '.join(str(x) for x in row))
        except Exception as e:
            print(f"Error solving: {e}")

# Initialize and test solver
solver = BasicARCSolver()
test_solver_on_samples(solver, data)




