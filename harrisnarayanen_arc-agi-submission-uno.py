# ARC Prize 2025 - ARC-AGI-2 Challenge
# Neuro-Symbolic Approach for ARC-AGI-2 Dataset
# Competition URL: https://www.kaggle.com/competitions/arc-prize-2025

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, Counter
import itertools
from typing import List, Dict, Tuple, Any
import os
import warnings
import sys
import time
import gc
warnings.filterwarnings('ignore')

# Kaggle environment setup
print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("Starting ARC Prize 2025 submission...")

# Set device (CPU-only for Kaggle code competitions)
device = torch.device('cpu')  # Force CPU for Kaggle code competitions
print(f"Using device: {device}")

# Memory and timeout management
gc.collect()  # Clean up memory at start

# ============================================================================
# NEURAL COMPONENTS - CNN Feature Extractor and Pattern Recognition
# ============================================================================

class ARCFeatureExtractor(nn.Module):
    """CNN-based feature extractor for ARC grids"""
    
    def __init__(self, max_grid_size=30):
        super().__init__()
        self.max_grid_size = max_grid_size
        
        # Color embedding for values 0-9
        self.color_embedding = nn.Embedding(11, 16)  # 11 to handle padding
        
        # Multi-scale convolutions
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        
        # Attention mechanism
        self.attention = nn.Conv2d(128, 1, kernel_size=1)
        
        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 256)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, grid):
        batch_size, h, w = grid.shape
        
        # Embed colors
        x = self.color_embedding(grid)  # (batch, h, w, 16)
        x = x.permute(0, 3, 1, 2)  # (batch, 16, h, w)
        
        # Pad to consistent size
        pad_h = max(0, self.max_grid_size - h)
        pad_w = max(0, self.max_grid_size - w)
        x = F.pad(x, (0, pad_w, 0, pad_h))
        
        # Feature extraction
        x1 = F.relu(self.conv1(x))
        x1 = self.dropout(x1)
        x2 = F.relu(self.conv2(x1))
        x2 = self.dropout(x2)
        x3 = F.relu(self.conv3(x2))
        
        # Attention
        attention_weights = torch.sigmoid(self.attention(x3))
        attended_features = x3 * attention_weights
        
        # Global features
        global_features = self.global_pool(attended_features).squeeze(-1).squeeze(-1)
        global_features = self.fc(global_features)
        
        return {
            'local_features': x3[:, :, :h, :w],  # Crop back to original size
            'global_features': global_features,
            'attention_weights': attention_weights[:, :, :h, :w]
        }

class PatternRecognizer(nn.Module):
    """Recognizes patterns in grid features"""
    
    def __init__(self, feature_dim=128):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Pattern detectors
        self.symmetry_detector = nn.Conv2d(feature_dim, 8, kernel_size=3, padding=1)
        self.shape_detector = nn.Conv2d(feature_dim, 12, kernel_size=3, padding=1)
        self.color_pattern_detector = nn.Conv2d(feature_dim, 16, kernel_size=3, padding=1)
        
        # Global pattern analysis
        self.global_mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
        
    def forward(self, features):
        local_features = features['local_features']
        global_features = features['global_features']
        
        # Local pattern detection
        symmetries = torch.sigmoid(self.symmetry_detector(local_features))
        shapes = torch.sigmoid(self.shape_detector(local_features))
        color_patterns = torch.sigmoid(self.color_pattern_detector(local_features))
        
        # Global pattern analysis
        global_patterns = self.global_mlp(global_features)
        
        return {
            'symmetries': symmetries,
            'shapes': shapes,
            'color_patterns': color_patterns,
            'global_patterns': global_patterns
        }

# ============================================================================
# SYMBOLIC REASONING COMPONENTS
# ============================================================================

class ARCOperations:
    """Core operations for ARC transformations"""
    
    @staticmethod
    def rotate_90(grid):
        return np.rot90(grid, k=1)
    
    @staticmethod
    def rotate_180(grid):
        return np.rot90(grid, k=2)
    
    @staticmethod
    def rotate_270(grid):
        return np.rot90(grid, k=3)
    
    @staticmethod
    def flip_horizontal(grid):
        return np.fliplr(grid)
    
    @staticmethod
    def flip_vertical(grid):
        return np.flipud(grid)
    
    @staticmethod
    def transpose(grid):
        return np.transpose(grid)
    
    @staticmethod
    def identity(grid):
        return grid.copy()
    
    @staticmethod
    def fill_color(grid, old_color, new_color):
        result = grid.copy()
        result[result == old_color] = new_color
        return result
    
    @staticmethod
    def extract_largest_object(grid):
        """Extract the largest connected component"""
        try:
            from scipy import ndimage
            
            result = np.zeros_like(grid)
            for color in range(1, 10):  # Skip background (0)
                mask = (grid == color)
                if not mask.any():
                    continue
                    
                labeled, num_features = ndimage.label(mask)
                if num_features == 0:
                    continue
                    
                # Find largest component
                sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
                max_label = np.argmax(sizes) + 1
                largest_mask = (labeled == max_label)
                result[largest_mask] = color
            
            return result
        except:
            return grid.copy()  # Fallback if scipy not available
    
    @staticmethod
    def scale_grid(grid, factor):
        """Scale grid by integer factor"""
        if factor == 1:
            return grid
        
        h, w = grid.shape
        new_h, new_w = h * factor, w * factor
        result = np.zeros((new_h, new_w), dtype=grid.dtype)
        
        for i in range(h):
            for j in range(w):
                result[i*factor:(i+1)*factor, j*factor:(j+1)*factor] = grid[i, j]
        
        return result

class SymbolicReasoner:
    """Core symbolic reasoning engine"""
    
    def __init__(self):
        self.operations = {
            'identity': ARCOperations.identity,
            'rotate_90': ARCOperations.rotate_90,
            'rotate_180': ARCOperations.rotate_180,
            'rotate_270': ARCOperations.rotate_270,
            'flip_horizontal': ARCOperations.flip_horizontal,
            'flip_vertical': ARCOperations.flip_vertical,
            'transpose': ARCOperations.transpose,
            'extract_largest': ARCOperations.extract_largest_object,
        }
        
        # Color transformation operations
        for old_color in range(10):
            for new_color in range(10):
                if old_color != new_color:
                    op_name = f'fill_{old_color}_to_{new_color}'
                    self.operations[op_name] = lambda g, oc=old_color, nc=new_color: ARCOperations.fill_color(g, oc, nc)
    
    def execute_program(self, program, input_grid):
        """Execute a program (list of operations) on input grid"""
        current_grid = input_grid.copy()
        
        for operation in program:
            if operation in self.operations:
                try:
                    current_grid = self.operations[operation](current_grid)
                except Exception as e:
                    # If operation fails, return current state
                    break
            else:
                # Unknown operation, skip
                continue
        
        return current_grid
    
    def detect_transformation_type(self, input_grid, output_grid):
        """Detect what type of transformation occurred"""
        transformations = []
        
        # Check for direct transformations
        for op_name, op_func in self.operations.items():
            try:
                result = op_func(input_grid)
                if np.array_equal(result, output_grid):
                    transformations.append([op_name])
            except:
                continue
        
        # Check for 2-step transformations
        if len(transformations) == 0:
            for op1_name, op1_func in list(self.operations.items())[:20]:  # Limit search
                try:
                    intermediate = op1_func(input_grid)
                    for op2_name, op2_func in list(self.operations.items())[:20]:
                        try:
                            result = op2_func(intermediate)
                            if np.array_equal(result, output_grid):
                                transformations.append([op1_name, op2_name])
                        except:
                            continue
                except:
                    continue
        
        return transformations

class ProgramSynthesizer:
    """Synthesizes programs from input-output examples"""
    
    def __init__(self, max_program_length=3):
        self.max_length = max_program_length
        self.symbolic_reasoner = SymbolicReasoner()
        
    def synthesize(self, input_output_pairs):
        """Find the best program that fits all input-output pairs"""
        if not input_output_pairs:
            return ['identity']
        
        # Find programs that work for each pair
        all_candidates = []
        
        for input_grid, output_grid in input_output_pairs:
            candidates = self.symbolic_reasoner.detect_transformation_type(
                np.array(input_grid), np.array(output_grid)
            )
            all_candidates.append(set([tuple(c) for c in candidates]))
        
        # Find intersection - programs that work for all pairs
        if all_candidates:
            common_programs = set.intersection(*all_candidates)
            if common_programs:
                # Return the simplest program (shortest)
                best_program = min(common_programs, key=len)
                return list(best_program)
        
        # If no common program found, try the most frequent operations
        operation_counts = defaultdict(int)
        for candidates in all_candidates:
            for program in candidates:
                for op in program:
                    operation_counts[op] += 1
        
        if operation_counts:
            most_common_op = max(operation_counts, key=operation_counts.get)
            return [most_common_op]
        
        return ['identity']
    
    def predict(self, program, test_input):
        """Apply synthesized program to test input"""
        return self.symbolic_reasoner.execute_program(program, np.array(test_input))

# ============================================================================
# MAIN ARC SOLVER
# ============================================================================

class ARCSolver:
    """Main ARC solver combining neural and symbolic components"""
    
    def __init__(self):
        self.feature_extractor = ARCFeatureExtractor().to(device)
        self.pattern_recognizer = PatternRecognizer().to(device)
        self.program_synthesizer = ProgramSynthesizer()
        
        # Set to eval mode (no training in this version)
        self.feature_extractor.eval()
        self.pattern_recognizer.eval()
    
    def solve_task(self, task):
        """Solve a single ARC-AGI-2 task"""
        train_pairs = task['train']
        test_inputs = task['test']
        
        solutions = []
        
        for test_input in test_inputs:
            test_input_grid = test_input['input']
            
            # Try symbolic approach first
            program = self.program_synthesizer.synthesize(
                [(pair['input'], pair['output']) for pair in train_pairs]
            )
            
            prediction = self.program_synthesizer.predict(program, test_input_grid)
            
            # Ensure prediction is valid for ARC-AGI-2 format
            prediction = self.validate_prediction(prediction, test_input_grid)
            
            solutions.append(prediction.tolist())
        
        return solutions
    
    def validate_prediction(self, prediction, reference_input):
        """Validate and fix prediction if needed"""
        # Ensure prediction has valid colors (0-9)
        prediction = np.clip(prediction, 0, 9).astype(int)
        
        # If prediction is empty or invalid, return input
        if prediction.size == 0:
            return np.array(reference_input)
        
        # Ensure reasonable size (not too large)
        if prediction.shape[0] > 30 or prediction.shape[1] > 30:
            return np.array(reference_input)
        
        return prediction

# ============================================================================
# KAGGLE COMPETITION INTERFACE - FIXED FORMAT
# ============================================================================

def load_data(data_path):
    """Load ARC data from JSON files with error handling"""
    try:
        if not os.path.exists(data_path):
            print(f"ERROR: Data file not found at {data_path}")
            print("Available files in input directory:")
            if os.path.exists('/kaggle/input'):
                for root, dirs, files in os.walk('/kaggle/input'):
                    for file in files:
                        print(f"  {os.path.join(root, file)}")
            return None
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        print(f"Successfully loaded {len(data)} tasks")
        return data
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def solve_arc_tasks(data_path, output_path):
    """Main function to solve ARC-AGI 2025 tasks with CORRECT FORMAT"""
    start_time = time.time()
    print("="*50)
    print("ARC-AGI 2025 - Starting Task Solving")
    print("="*50)
    
    # Load the test challenges
    test_data = load_data(data_path)
    if test_data is None:
        print("CRITICAL ERROR: Could not load test data")
        with open(output_path, 'w') as f:
            json.dump({}, f)
        return None
    
    print(f"Loaded {len(test_data)} test tasks from ARC-AGI 2025")
    
    # Initialize solver
    try:
        solver = ARCSolver()
        print("Solver initialized successfully")
    except Exception as e:
        print(f"ERROR initializing solver: {e}")
        with open(output_path, 'w') as f:
            json.dump({}, f)
        return None
    
    # Process each task - CORRECT ARC-AGI 2025 FORMAT
    submission = {}
    total_tasks = len(test_data)
    
    for i, (task_id, task_data) in enumerate(test_data.items()):
        current_time = time.time()
        elapsed = current_time - start_time
        
        print(f"[{elapsed:.1f}s] Task {i+1}/{total_tasks}: {task_id}")
        
        # Timeout protection - leave time for file saving
        if elapsed > 3000:  # 50 minutes
            print(f"TIMEOUT: Stopping at task {i+1} to save submission")
            break
        
        try:
            # Validate task structure
            if 'train' not in task_data or 'test' not in task_data:
                print(f"  WARNING: Invalid task structure for {task_id}")
                # Create minimal valid response - CORRECT FORMAT
                submission[task_id] = [[[0]]]  # Just one attempt per test case
                continue
            
            # Extract training data
            train_pairs = [(pair['input'], pair['output']) for pair in task_data['train']]
            test_cases = task_data['test']
            
            print(f"  Training pairs: {len(train_pairs)}, Test cases: {len(test_cases)}")
            
            # Solve using our neuro-symbolic approach
            try:
                # Synthesize program from training examples
                program = solver.program_synthesizer.synthesize(train_pairs)
                print(f"  Generated program: {program}")
                
                # Apply to each test case - CORRECT FORMAT
                task_predictions = []
                
                for j, test_case in enumerate(test_cases):
                    test_input = test_case['input']
                    
                    try:
                        # Generate prediction
                        prediction = solver.program_synthesizer.predict(program, test_input)
                        
                        # Validate and clean prediction
                        prediction = solver.validate_prediction(prediction, test_input)
                        prediction = prediction.tolist() if hasattr(prediction, 'tolist') else prediction
                        
                        # CORRECT FORMAT: Just add the prediction directly
                        task_predictions.append(prediction)
                        
                    except Exception as e:
                        print(f"    Error on test case {j}: {e}")
                        # Fallback: use input as prediction
                        task_predictions.append(test_input)
                
                # Store predictions for this task - CORRECT FORMAT
                submission[task_id] = task_predictions
                
            except Exception as e:
                print(f"  ERROR in synthesis for {task_id}: {e}")
                # Create fallback predictions - CORRECT FORMAT
                fallback_predictions = []
                for test_case in test_cases:
                    test_input = test_case['input']
                    fallback_predictions.append(test_input)
                submission[task_id] = fallback_predictions
        
        except Exception as e:
            print(f"  CRITICAL ERROR processing {task_id}: {e}")
            # Ultimate fallback - CORRECT FORMAT
            submission[task_id] = [[[0]]]
        
        # Memory cleanup
        if i % 10 == 0:
            gc.collect()
    
    # Save submission with validation - CORRECT FORMAT
    try:
        print(f"\nSaving ARC-AGI 2025 submission...")
        print(f"Tasks processed: {len(submission)}")
        
        # Final validation and cleanup
        validated_submission = {}
        
        for task_id, predictions in submission.items():
            validated_predictions = []
            
            # CORRECT FORMAT: predictions is a list of grids, not nested attempts
            for prediction in predictions:
                # Validate grid format
                def validate_grid(grid):
                    if not isinstance(grid, list) or len(grid) == 0:
                        return [[0]]
                    
                    validated_grid = []
                    for row in grid:
                        if isinstance(row, list) and len(row) > 0:
                            validated_row = []
                            for cell in row:
                                if isinstance(cell, (int, float)):
                                    validated_row.append(int(max(0, min(9, cell))))
                                else:
                                    validated_row.append(0)
                            validated_grid.append(validated_row)
                    
                    return validated_grid if len(validated_grid) > 0 else [[0]]
                
                validated_prediction = validate_grid(prediction)
                validated_predictions.append(validated_prediction)
            
            if len(validated_predictions) > 0:
                validated_submission[task_id] = validated_predictions
            else:
                validated_submission[task_id] = [[[0]]]
        
        # Ensure we have something to submit
        if len(validated_submission) == 0:
            print("WARNING: Creating minimal submission")
            validated_submission = {"dummy_task": [[[0]]]}
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save in compact format
        with open(output_path, 'w') as f:
            json.dump(validated_submission, f, separators=(',', ':'))
        
        # Verify file creation
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✅ Submission saved! Size: {file_size/1024:.2f} KB")
            
            # Quick format check
            with open(output_path, 'r') as f:
                check = json.load(f)
                if len(check) > 0:
                    sample_task = next(iter(check.values()))
                    if isinstance(sample_task, list) and len(sample_task) > 0:
                        sample_prediction = sample_task[0]
                        if isinstance(sample_prediction, list):
                            print("✅ Format validation passed!")
                        else:
                            print("⚠️  Format warning: Check structure")
        else:
            print("❌ Failed to create submission file")
    
    except Exception as e:
        print(f"CRITICAL ERROR saving submission: {e}")
        try:
            # Emergency minimal submission
            with open(output_path, 'w') as f:
                json.dump({"emergency": [[[0]]]}, f)
            print("Created emergency submission")
        except:
            pass
    
    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time:.1f} seconds")
    
    return submission

# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

def test_solver_on_sample():
    """Test the solver on a sample task"""
    # Create a simple test task
    sample_task = {
        'train': [
            {
                'input': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                'output': [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
            }
        ],
        'test': [
            {
                'input': [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
            }
        ]
    }
    
    solver = ARCSolver()
    solutions = solver.solve_task(sample_task)
    
    print("Sample task solutions:")
    for i, solution in enumerate(solutions):
        print(f"Test {i+1}:")
        for row in solution:
            print(row)
        print()

# ============================================================================
# MAIN EXECUTION - KAGGLE SUBMISSION
# ============================================================================

def main():
    """Main execution function with comprehensive error handling"""
    try:
        print("="*60)
        print("ARC Prize 2025 - KAGGLE SUBMISSION STARTING")
        print("="*60)
        
        # Check if we're in Kaggle environment
        if os.path.exists('/kaggle/input'):
            print("✅ Kaggle environment detected")
            
            # Find the correct data path
            data_paths = [
                '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json',
                '/kaggle/input/arc-prize-2025/test.json',
                '/kaggle/input/arc-prize-2025/test_challenges.json'
            ]
            
            data_path = None
            for path in data_paths:
                if os.path.exists(path):
                    data_path = path
                    print(f"✅ Found data at: {path}")
                    break
            
            if data_path is None:
                print("❌ No data file found. Available files:")
                for root, dirs, files in os.walk('/kaggle/input'):
                    for file in files:
                        print(f"  {os.path.join(root, file)}")
                return
            
            # Run the main solution
            submission = solve_arc_tasks(data_path, '/kaggle/working/submission.json')
            
            # Final validation
            if os.path.exists('/kaggle/working/submission.json'):
                print("✅ SUCCESS: Submission file created!")
                validate_submission_format('/kaggle/working/submission.json')
            else:
                print("❌ CRITICAL: Submission file missing!")
        
        else:
            # Local testing
            print("Local environment - running test...")
            test_solver_on_sample()
    
    except Exception as e:
        print(f"CRITICAL ERROR in main(): {e}")
        # Create emergency submission
        try:
            with open('/kaggle/working/submission.json', 'w') as f:
                json.dump({"emergency": [[[0]]]}, f)
            print("Emergency submission created")
        except:
            print("Could not create emergency submission")

# Run main function
if __name__ == "__main__":
    main()

# ============================================================================
# ADDITIONAL UTILITIES
# ============================================================================

def validate_submission_format(submission_path):
    """Validate that submission follows ARC Prize 2025 format"""
    try:
        with open(submission_path, 'r') as f:
            submission = json.load(f)
        
        print("Validating ARC Prize 2025 submission format...")
        
        total_tasks = len(submission)
        valid_tasks = 0
        
        for task_id, task_solutions in submission.items():
            # CORRECT FORMAT: task_solutions should be a list of grids
            if isinstance(task_solutions, list):
                valid_predictions = 0
                for prediction in task_solutions:
                    # Each prediction should be a 2D grid (list of lists)
                    if isinstance(prediction, list) and len(prediction) > 0:
                        if all(isinstance(row, list) for row in prediction):
                            valid_predictions += 1
                
                if valid_predictions == len(task_solutions):
                    valid_tasks += 1
        
        print(f"Validation results:")
        print(f"- Total tasks: {total_tasks}")
        print(f"- Valid tasks: {valid_tasks}")
        print(f"- Success rate: {valid_tasks/total_tasks*100:.1f}%")
        
        if valid_tasks == total_tasks:
            print("✅ Submission format is valid for ARC Prize 2025!")
        else:
            print("❌ Submission format has issues")
        
        return valid_tasks == total_tasks
        
    except Exception as e:
        print(f"Error validating submission: {e}")
        return False

def visualize_task(task, task_id="Unknown"):
    """Visualize an ARC task (for debugging)"""
    print(f"Task: {task_id}")
    print("="*50)
    
    for i, pair in enumerate(task.get('train', [])):
        print(f"Training Example {i+1}:")
        print("Input:")
        for row in pair['input']:
            print(''.join(str(x) for x in row))
        print("Output:")
        for row in pair['output']:
            print(''.join(str(x) for x in row))
        print()
    
    for i, test_case in enumerate(task.get('test', [])):
        print(f"Test {i+1}:")
        print("Input:")
        for row in test_case['input']:
            print(''.join(str(x) for x in row))
        print()




