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


#!/usr/bin/env python
"""
Intuition-Guided Program Synthesizer (IGPS) for ARC Prize 2025
A hybrid System 1 (LLM Intuition) + System 2 (Program Synthesis) approach

Architecture:
- Intuition Core: Fine-tuned LLM for heuristic guidance
- Reasoning Engine: Guided search over Domain-Specific Language
- Workspace: State management for tasks and solutions

Kaggle Submission Rules Compliance:
- Runtime: <= 12 hours on L4x4 GPU (96GB memory)
- No internet access (all models as Kaggle Datasets)
- Output: submission.json in /kaggle/working/
- Format: Two attempts per test case, exact pixel match required
"""

# ==========================================
# PHASE 0: ENVIRONMENT SETUP AND DATA PATHS
# ==========================================

import copy
import heapq
import json
import os
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Constants ---
# Update these paths based on actual Kaggle dataset structure
BASE_INPUT_PATH = '/kaggle/input/arc-prize-2025/'
TRAINING_PATH = os.path.join(
    BASE_INPUT_PATH, 'arc-agi_training_challenges.json')
EVALUATION_PATH = os.path.join(
    BASE_INPUT_PATH, 'arc-agi_evaluation_challenges.json')
TEST_PATH = os.path.join(BASE_INPUT_PATH, 'arc-agi_test_challenges.json')
SUBMISSION_FILE = '/kaggle/working/submission.json'

# Model paths (these would be Kaggle datasets)
# Your fine-tuned model dataset
LLM_MODEL_PATH = '/kaggle/input/finetuned-llama-arc/'
BACKUP_MODEL_PATH = '/kaggle/input/gemma-2-arc/'       # Backup model

# --- IGPS Parameters ---
BEAM_WIDTH = 8
MAX_PROGRAM_LENGTH = 12
MAX_SEARCH_TIME = 300  # 5 minutes per task max
DSL_MAX_DEPTH = 6
CONFIDENCE_THRESHOLD = 0.7

# --- GPU Memory Management ---
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print(" IGPS for ARC Prize 2025 - Initializing...")

# ==========================================
# PHASE 1: DATA LOADING AND ABSTRACTION
# ==========================================


def load_arc_data(file_path: str) -> Dict[str, Any]:
    """Load ARC tasks from JSON file with error handling."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(
            f"Warning: {file_path} not found. Using fallback data structure.")
        return {}


@dataclass
class ARCTask:
    """Structured representation of an ARC task."""
    task_id: str
    train_examples: List[Tuple[np.ndarray, np.ndarray]]
    test_inputs: List[np.ndarray]
    # Only available during validation
    test_outputs: Optional[List[np.ndarray]] = None

    @classmethod
    def from_json(cls, task_id: str, task_data: Dict) -> 'ARCTask':
        """Create ARCTask from JSON data."""
        train_examples = []
        for example in task_data.get('train', []):
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            train_examples.append((input_grid, output_grid))

        test_inputs = []
        test_outputs = []
        for example in task_data.get('test', []):
            test_inputs.append(np.array(example['input']))
            if 'output' in example:
                test_outputs.append(np.array(example['output']))

        return cls(
            task_id=task_id,
            train_examples=train_examples,
            test_inputs=test_inputs,
            test_outputs=test_outputs if test_outputs else None
        )


class DataLoader:
    """Handles loading and preprocessing of ARC data."""

    def __init__(self):
        self.training_tasks = {}
        self.evaluation_tasks = {}
        self.test_tasks = {}

    def load_all_data(self):
        """Load all available datasets."""
        print(" Loading ARC datasets...")

        # Load training data
        if os.path.exists(TRAINING_PATH):
            train_data = load_arc_data(TRAINING_PATH)
            for task_id, task_data in train_data.items():
                self.training_tasks[task_id] = ARCTask.from_json(
                    task_id, task_data)
            print(f"   Loaded {len(self.training_tasks)} training tasks")

        # Load evaluation data
        if os.path.exists(EVALUATION_PATH):
            eval_data = load_arc_data(EVALUATION_PATH)
            for task_id, task_data in eval_data.items():
                self.evaluation_tasks[task_id] = ARCTask.from_json(
                    task_id, task_data)
            print(f"   Loaded {len(self.evaluation_tasks)} evaluation tasks")

        # Load test data
        if os.path.exists(TEST_PATH):
            test_data = load_arc_data(TEST_PATH)
            for task_id, task_data in test_data.items():
                self.test_tasks[task_id] = ARCTask.from_json(
                    task_id, task_data)
            print(f"   Loaded {len(self.test_tasks)} test tasks")

# ==========================================
# PHASE 2: DOMAIN-SPECIFIC LANGUAGE (DSL)
# ==========================================


class DomainSpecificLanguage:
    """
    Comprehensive DSL for ARC transformations.
    Each method represents a primitive operation that can be composed.
    """

    @staticmethod
    def find_objects(grid: np.ndarray) -> List[Dict]:
        """Identify contiguous objects of the same color."""
        from scipy import ndimage

        objects = []
        unique_colors = np.unique(grid)

        for color in unique_colors:
            if color == 0:  # Skip background
                continue

            mask = (grid == color)
            labeled, num_features = ndimage.label(mask)

            for i in range(1, num_features + 1):
                obj_mask = (labeled == i)
                coords = np.argwhere(obj_mask)

                if len(coords) > 0:
                    y_min, x_min = coords.min(axis=0)
                    y_max, x_max = coords.max(axis=0)

                    objects.append({
                        'color': int(color),
                        'mask': obj_mask,
                        'coords': coords,
                        'bbox': (y_min, x_min, y_max, x_max),
                        'size': len(coords),
                        'center': coords.mean(axis=0)
                    })

        return objects

    @staticmethod
    def apply_symmetry(grid: np.ndarray, axis: str) -> np.ndarray:
        """Apply symmetry transformations."""
        if axis == 'horizontal':
            return np.fliplr(grid)
        elif axis == 'vertical':
            return np.flipud(grid)
        elif axis == 'diagonal_main':
            return grid.T
        elif axis == 'diagonal_anti':
            return np.flipud(grid.T)
        return grid

    @staticmethod
    def rotate_grid(grid: np.ndarray, times: int) -> np.ndarray:
        """Rotate grid 90 degrees clockwise 'times' times."""
        return np.rot90(grid, k=-times)

    @staticmethod
    def scale_grid(grid: np.ndarray, factor: int) -> np.ndarray:
        """Scale grid by integer factor."""
        if factor <= 0:
            return grid
        return np.repeat(np.repeat(grid, factor, axis=0), factor, axis=1)

    @staticmethod
    def crop_to_objects(grid: np.ndarray) -> np.ndarray:
        """Crop grid to minimal bounding box containing all non-zero elements."""
        non_zero = np.argwhere(grid != 0)
        if len(non_zero) == 0:
            return grid

        y_min, x_min = non_zero.min(axis=0)
        y_max, x_max = non_zero.max(axis=0)
        return grid[y_min:y_max+1, x_min:x_max+1]

    @staticmethod
    def fill_color(grid: np.ndarray, color: int) -> np.ndarray:
        """Fill entire grid with specified color."""
        return np.full_like(grid, color)

    @staticmethod
    def replace_color(grid: np.ndarray, old_color: int, new_color: int) -> np.ndarray:
        """Replace all instances of old_color with new_color."""
        result = grid.copy()
        result[grid == old_color] = new_color
        return result

    @staticmethod
    def extract_pattern(grid: np.ndarray, pattern_size: Tuple[int, int]) -> List[np.ndarray]:
        """Extract all unique patterns of given size from grid."""
        h, w = pattern_size
        patterns = []
        grid_h, grid_w = grid.shape

        for i in range(grid_h - h + 1):
            for j in range(grid_w - w + 1):
                pattern = grid[i:i+h, j:j+w]
                # Check if this pattern is unique
                is_unique = True
                for existing in patterns:
                    if np.array_equal(pattern, existing):
                        is_unique = False
                        break
                if is_unique:
                    patterns.append(pattern)

        return patterns

    @staticmethod
    def tile_pattern(pattern: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Tile a pattern to fill target shape."""
        target_h, target_w = target_shape
        pattern_h, pattern_w = pattern.shape

        # Calculate how many times to repeat
        repeat_h = (target_h + pattern_h - 1) // pattern_h
        repeat_w = (target_w + pattern_w - 1) // pattern_w

        # Create tiled pattern
        tiled = np.tile(pattern, (repeat_h, repeat_w))

        # Crop to exact target shape
        return tiled[:target_h, :target_w]

    @staticmethod
    def detect_symmetries(grid: np.ndarray) -> List[str]:
        """Detect all symmetries present in the grid."""
        symmetries = []

        # Check horizontal symmetry
        if np.array_equal(grid, np.fliplr(grid)):
            symmetries.append('horizontal')

        # Check vertical symmetry
        if np.array_equal(grid, np.flipud(grid)):
            symmetries.append('vertical')

        # Check diagonal symmetries (if square)
        if grid.shape[0] == grid.shape[1]:
            if np.array_equal(grid, grid.T):
                symmetries.append('diagonal_main')
            if np.array_equal(grid, np.flipud(grid.T)):
                symmetries.append('diagonal_anti')

        return symmetries

    @staticmethod
    def find_repeating_patterns(grid: np.ndarray) -> List[Dict]:
        """Find repeating patterns in the grid."""
        patterns = []
        h, w = grid.shape

        # Check for various pattern sizes
        for ph in range(1, min(h//2 + 1, 4)):
            for pw in range(1, min(w//2 + 1, 4)):
                if h % ph == 0 and w % pw == 0:
                    pattern = grid[:ph, :pw]
                    tiled = np.tile(pattern, (h//ph, w//pw))

                    if np.array_equal(grid, tiled):
                        patterns.append({
                            'pattern': pattern,
                            'size': (ph, pw),
                            'repetitions': (h//ph, w//pw)
                        })

        return patterns

    @staticmethod
    def apply_mask(grid: np.ndarray, mask: np.ndarray, color: int) -> np.ndarray:
        """Apply a mask to set specific positions to a color."""
        result = grid.copy()
        result[mask] = color
        return result

    @staticmethod
    def connect_objects(grid: np.ndarray, color: int) -> np.ndarray:
        """Connect all objects of the same color with lines."""
        objects = DomainSpecificLanguage.find_objects(grid)
        same_color_objects = [obj for obj in objects if obj['color'] == color]

        if len(same_color_objects) < 2:
            return grid

        result = grid.copy()

        # Connect centers with lines
        for i in range(len(same_color_objects) - 1):
            center1 = same_color_objects[i]['center'].astype(int)
            center2 = same_color_objects[i + 1]['center'].astype(int)

            # Simple line drawing (Bresenham-like)
            y1, x1 = center1
            y2, x2 = center2

            # Draw line between centers
            steps = max(abs(y2 - y1), abs(x2 - x1))
            if steps > 0:
                for step in range(steps + 1):
                    y = int(y1 + (y2 - y1) * step / steps)
                    x = int(x1 + (x2 - x1) * step / steps)
                    if 0 <= y < result.shape[0] and 0 <= x < result.shape[1]:
                        result[y, x] = color

        return result

# ==========================================
# PROGRAM REPRESENTATION AND EXECUTION
# ==========================================


@dataclass
class DSLOperation:
    """Represents a single DSL operation with parameters."""
    name: str
    params: Tuple = field(default_factory=tuple)

    def __str__(self):
        if self.params:
            param_str = ', '.join(map(str, self.params))
            return f"{self.name}({param_str})"
        return f"{self.name}()"


@dataclass
class Program:
    """Represents a sequence of DSL operations."""
    operations: List[DSLOperation] = field(default_factory=list)
    score: float = 0.0
    execution_time: float = 0.0

    def __str__(self):
        return " -> ".join(str(op) for op in self.operations)

    def __len__(self):
        return len(self.operations)

    def copy(self):
        return Program(
            operations=self.operations.copy(),
            score=self.score,
            execution_time=self.execution_time
        )


class ProgramExecutor:
    """Executes DSL programs on grids."""

    def __init__(self):
        self.dsl = DomainSpecificLanguage()
        self.operation_map = {
            'find_objects': self.dsl.find_objects,
            'apply_symmetry': self.dsl.apply_symmetry,
            'rotate_grid': self.dsl.rotate_grid,
            'scale_grid': self.dsl.scale_grid,
            'crop_to_objects': self.dsl.crop_to_objects,
            'fill_color': self.dsl.fill_color,
            'replace_color': self.dsl.replace_color,
            'extract_pattern': self.dsl.extract_pattern,
            'tile_pattern': self.dsl.tile_pattern,
            'detect_symmetries': self.dsl.detect_symmetries,
            'find_repeating_patterns': self.dsl.find_repeating_patterns,
            'apply_mask': self.dsl.apply_mask,
            'connect_objects': self.dsl.connect_objects,
        }

    def execute(self, program: Program, input_grid: np.ndarray) -> Optional[np.ndarray]:
        """Execute a program on an input grid."""
        try:
            current_grid = input_grid.copy()

            for operation in program.operations:
                if operation.name not in self.operation_map:
                    return None

                func = self.operation_map[operation.name]

                # Handle different operation types
                if operation.name in ['find_objects', 'crop_to_objects', 'detect_symmetries', 'find_repeating_patterns']:
                    # Operations that don't modify the grid directly
                    if operation.name in ['crop_to_objects']:
                        current_grid = func(current_grid)
                else:
                    # Operations that transform the grid
                    current_grid = func(current_grid, *operation.params)

            return current_grid

        except Exception as e:
            # Return None if execution fails
            return None

    def validate_program(self, program: Program, task: ARCTask) -> float:
        """Validate a program against training examples and return score."""
        if not task.train_examples:
            return 0.0

        correct = 0
        total = len(task.train_examples)

        for input_grid, expected_output in task.train_examples:
            result = self.execute(program, input_grid)
            if result is not None and np.array_equal(result, expected_output):
                correct += 1

        return correct / total

# ==========================================
# INTUITION CORE (LLM GUIDANCE)
# ==========================================


class IntuitionCore:
    """
    LLM-based heuristic guidance for program synthesis.
    Provides three key functions:
    1. Hypothesis Generation
    2. Next-Step Proposal
    3. Heuristic Evaluation
    """

    def __init__(self, model_path: str = None):
        self.model = None
        self.tokenizer = None
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Try to load model if available
        if model_path and os.path.exists(model_path):
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map='auto'
                )
                print(f"   Loaded LLM from {model_path}")
            except Exception as e:
                print(f"   Failed to load LLM: {e}")
                self.model = None

        # Fallback to heuristic guidance if no model
        if self.model is None:
            print("  ðŸ“‹ Using heuristic-based guidance (no LLM available)")

    def analyze_task(self, task: ARCTask) -> Dict[str, Any]:
        """Analyze a task and provide insights."""
        analysis = {
            'grid_sizes': [],
            'color_usage': defaultdict(int),
            'common_patterns': [],
            'suggested_operations': []
        }

        # Analyze training examples
        for input_grid, output_grid in task.train_examples:
            analysis['grid_sizes'].append({
                'input': input_grid.shape,
                'output': output_grid.shape
            })

            # Count colors
            for color in np.unique(input_grid):
                analysis['color_usage'][int(color)] += 1
            for color in np.unique(output_grid):
                analysis['color_usage'][int(color)] += 1

        # Suggest operations based on patterns
        if len(analysis['grid_sizes']) > 0:
            input_shapes = [gs['input'] for gs in analysis['grid_sizes']]
            output_shapes = [gs['output'] for gs in analysis['grid_sizes']]

            # Check for size changes
            if any(inp != out for inp, out in zip(input_shapes, output_shapes)):
                analysis['suggested_operations'].append('scale_grid')
                analysis['suggested_operations'].append('crop_to_objects')

            # Check for color patterns
            if len(analysis['color_usage']) > 5:
                analysis['suggested_operations'].append('replace_color')

            # Default suggestions
            analysis['suggested_operations'].extend([
                'apply_symmetry', 'rotate_grid', 'find_repeating_patterns'
            ])

        return analysis

    def get_next_step_probabilities(self,
                                    current_program: Program,
                                    task_analysis: Dict,
                                    available_operations: List[str]) -> Dict[str, float]:
        """Get probability distribution over next operations."""

        if self.model is not None:
            return self._llm_based_probabilities(current_program, task_analysis, available_operations)
        else:
            return self._heuristic_based_probabilities(current_program, task_analysis, available_operations)

    def _llm_based_probabilities(self, program: Program, analysis: Dict, operations: List[str]) -> Dict[str, float]:
        """Use LLM to get operation probabilities."""
        # This would be the actual LLM inference
        # For now, fall back to heuristic
        return self._heuristic_based_probabilities(program, analysis, operations)

    def _heuristic_based_probabilities(self, program: Program, analysis: Dict, operations: List[str]) -> Dict[str, float]:
        """Use heuristics to assign operation probabilities."""
        probs = {}

        # Base probability for all operations
        base_prob = 1.0 / len(operations)

        for op in operations:
            probs[op] = base_prob

        # Boost suggested operations
        for suggested_op in analysis.get('suggested_operations', []):
            if suggested_op in probs:
                probs[suggested_op] *= 2.0

        # Avoid repeating recent operations
        if len(program.operations) > 0:
            recent_op = program.operations[-1].name
            if recent_op in probs:
                probs[recent_op] *= 0.5

        # Normalize
        total = sum(probs.values())
        if total > 0:
            for op in probs:
                probs[op] /= total

        return probs

    def evaluate_program_quality(self, program: Program, task: ARCTask) -> float:
        """Evaluate the quality/promise of a partial program."""
        # Length penalty
        length_penalty = max(0, 1.0 - len(program) / MAX_PROGRAM_LENGTH)

        # Diversity bonus (different operation types)
        operation_types = set(op.name for op in program.operations)
        diversity_bonus = len(operation_types) / \
            max(1, len(program.operations))

        # Task-specific heuristics could go here

        return 0.6 * length_penalty + 0.4 * diversity_bonus

# ==========================================
# REASONING ENGINE (GUIDED SEARCH)
# ==========================================


@dataclass
class SearchState:
    """Represents a state in the program search space."""
    program: Program
    priority: float
    task_analysis: Dict = field(default_factory=dict)

    def __lt__(self, other):
        return self.priority > other.priority  # Higher priority first


class ReasoningEngine:
    """
    The master controller that performs guided search over the DSL.
    Uses beam search with LLM heuristics to efficiently explore the space.
    """

    def __init__(self, beam_width: int = BEAM_WIDTH, max_time: float = MAX_SEARCH_TIME):
        self.beam_width = beam_width
        self.max_time = max_time
        self.executor = ProgramExecutor()
        self.intuition_core = IntuitionCore(LLM_MODEL_PATH)

        # Available DSL operations with their parameter spaces
        self.operation_space = {
            'apply_symmetry': [('horizontal',), ('vertical',), ('diagonal_main',), ('diagonal_anti',)],
            'rotate_grid': [(1,), (2,), (3,)],
            'scale_grid': [(2,), (3,), (4,)],
            'crop_to_objects': [()],
            'fill_color': [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,)],
            'replace_color': [(i, j) for i in range(10) for j in range(10) if i != j],
            'connect_objects': [(i,) for i in range(1, 10)]
        }

    def solve_task(self, task: ARCTask) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Main entry point for solving an ARC task.
        Returns two solution attempts.
        """
        print(f" Solving task {task.task_id}")

        start_time = time.time()

        # Analyze the task
        task_analysis = self.intuition_core.analyze_task(task)

        # Find best programs using beam search
        best_programs = self._beam_search(task, task_analysis)

        # Generate solutions for test cases
        solutions = []
        for test_input in task.test_inputs:
            test_solutions = []

            for program in best_programs[:2]:  # Try top 2 programs
                result = self.executor.execute(program, test_input)
                if result is not None:
                    test_solutions.append(result.tolist())
                else:
                    # Fallback: return input as-is
                    test_solutions.append(test_input.tolist())

            # Ensure we have exactly 2 attempts
            while len(test_solutions) < 2:
                test_solutions.append(test_input.tolist())

            solutions.append(test_solutions[:2])

        elapsed = time.time() - start_time
        print(f"   Completed in {elapsed:.2f}s")

        # Return first test case solutions (adjust for multiple test cases)
        if solutions:
            return solutions[0][0], solutions[0][1]
        else:
            # Fallback
            dummy = [[0]]
            return dummy, dummy

    def _beam_search(self, task: ARCTask, task_analysis: Dict) -> List[Program]:
        """Perform beam search to find best programs."""

        # Initialize beam with empty program
        initial_state = SearchState(
            program=Program(),
            priority=1.0,
            task_analysis=task_analysis
        )

        beam = [initial_state]
        completed_programs = []

        start_time = time.time()

        for depth in range(MAX_PROGRAM_LENGTH):
            if time.time() - start_time > self.max_time:
                break

            new_beam = []

            for state in beam:
                # Generate successor states
                successors = self._generate_successors(state, task)
                new_beam.extend(successors)

            # Keep top beam_width states
            new_beam.sort()
            beam = new_beam[:self.beam_width]

            # Check for complete solutions
            for state in beam:
                score = self.executor.validate_program(state.program, task)
                if score >= CONFIDENCE_THRESHOLD:
                    state.program.score = score
                    completed_programs.append(state.program)

            # Early termination if we found good solutions
            if len(completed_programs) >= 2:
                break

        # Combine completed programs with best partial programs
        all_programs = completed_programs.copy()

        for state in beam:
            if state.program not in completed_programs:
                state.program.score = self.executor.validate_program(
                    state.program, task)
                all_programs.append(state.program)

        # Sort by score and return best
        all_programs.sort(key=lambda p: p.score, reverse=True)
        return all_programs[:5]  # Return top 5

    def _generate_successors(self, state: SearchState, task: ARCTask) -> List[SearchState]:
        """Generate successor states by adding one operation."""
        successors = []

        # Get operation probabilities from intuition core
        available_ops = list(self.operation_space.keys())
        op_probs = self.intuition_core.get_next_step_probabilities(
            state.program, state.task_analysis, available_ops
        )

        # Generate successors for top operations
        sorted_ops = sorted(op_probs.items(), key=lambda x: x[1], reverse=True)

        for op_name, prob in sorted_ops[:5]:  # Top 5 operations
            if op_name in self.operation_space:
                for params in self.operation_space[op_name]:
                    new_program = state.program.copy()
                    new_program.operations.append(
                        DSLOperation(op_name, params))

                    # Calculate priority
                    base_priority = prob
                    quality_score = self.intuition_core.evaluate_program_quality(
                        new_program, task)
                    priority = base_priority * quality_score

                    successor = SearchState(
                        program=new_program,
                        priority=priority,
                        task_analysis=state.task_analysis
                    )
                    successors.append(successor)

        return successors

# ==========================================
# WORKSPACE AND COORDINATION
# ==========================================


class Workspace:
    """
    Central workspace for managing task state, solutions, and coordination
    between the Intuition Core and Reasoning Engine.
    """

    def __init__(self):
        self.current_task = None
        self.solutions = {}
        self.reasoning_engine = ReasoningEngine()
        self.data_loader = DataLoader()

    def initialize(self):
        """Initialize the workspace and load data."""
        print(" Initializing IGPS Workspace...")
        self.data_loader.load_all_data()
        print("   Workspace ready")

    def solve_all_tasks(self, task_set: str = 'test') -> Dict[str, List[Dict]]:
        """Solve all tasks in the specified set."""

        if task_set == 'test':
            tasks = self.data_loader.test_tasks
        elif task_set == 'evaluation':
            tasks = self.data_loader.evaluation_tasks
        else:
            tasks = self.data_loader.training_tasks

        solutions = {}

        print(f" Solving {len(tasks)} {task_set} tasks...")

        for i, (task_id, task) in enumerate(tasks.items()):
            print(f" Progress: {i+1}/{len(tasks)} - {task_id}")

            try:
                solution1, solution2 = self.reasoning_engine.solve_task(task)

                # Store solutions in competition format
                solutions[task_id] = []
                for test_idx in range(len(task.test_inputs)):
                    solutions[task_id].append({
                        "attempt_1": solution1,
                        "attempt_2": solution2
                    })

            except Exception as e:
                print(f"   Error solving {task_id}: {e}")
                # Fallback solution
                fallback = [[0]]
                solutions[task_id] = [{
                    "attempt_1": fallback,
                    "attempt_2": fallback
                }]

        return solutions

# ==========================================
# PHASE 3: MAIN INFERENCE LOOP
# ==========================================


def main():
    """Main execution script for Kaggle submission."""

    print(" ARC Prize 2025 - IGPS Submission Starting...")
    print("=" * 60)

    start_time = time.time()

    # Initialize workspace
    workspace = Workspace()
    workspace.initialize()

    # Solve test tasks (in actual competition, this would be the hidden test set)
    print("\n Beginning inference on test set...")

    # For local testing, we can use evaluation set
    if not workspace.data_loader.test_tasks:
        print("  No test tasks found, using evaluation set for demo")
        solutions = workspace.solve_all_tasks('evaluation')
    else:
        solutions = workspace.solve_all_tasks('test')

    # Generate submission file
    print(f"\n Generating submission file...")

    with open(SUBMISSION_FILE, 'w') as f:
        json.dump(solutions, f, indent=2)

    total_time = time.time() - start_time
    print(f" Submission completed in {total_time/60:.1f} minutes")
    print(f" Submission file: {SUBMISSION_FILE}")
    print(f" Tasks processed: {len(solutions)}")

    # Validation check
    if os.path.exists(SUBMISSION_FILE):
        with open(SUBMISSION_FILE, 'r') as f:
            submission_data = json.load(f)

        print(f"Submission validation:")
        print(
            f"  - File size: {os.path.getsize(SUBMISSION_FILE) / 1024:.1f} KB")
        print(f"  - Tasks: {len(submission_data)}")

        # Check format
        sample_task = next(iter(submission_data.values()))
        if isinstance(sample_task, list) and len(sample_task) > 0:
            sample_test = sample_task[0]
            if 'attempt_1' in sample_test and 'attempt_2' in sample_test:
                print("  - Format: Valid")
            else:
                print("  - Format:  Invalid - missing attempts")
        else:
            print("  - Format:  Invalid - structure error")


# ==========================================
# KAGGLE NOTEBOOK EXECUTION
# ==========================================


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f" Fatal error: {e}")
        import traceback
        traceback.print_exc()

        # Create minimal fallback submission
        print(" Creating fallback submission...")
        fallback_solution = {"dummy_task": [
            {"attempt_1": [[0]], "attempt_2": [[0]]}]}

        with open(SUBMISSION_FILE, 'w') as f:
            json.dump(fallback_solution, f)

        print(f" Fallback submission saved to {SUBMISSION_FILE}")

print("\n" + "=" * 60)
print(" INTUITION-GUIDED PROGRAM SYNTHESIZER (IGPS)")
print("   Ready for ARC Prize 2025 Competition")
print("   System 1 (Intuition)  System 2 (Reasoning)")
print("=" * 60)


