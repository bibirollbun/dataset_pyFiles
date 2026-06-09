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


import os
import json
import numpy as np
import random
from typing import List, Dict, Any, Tuple, Optional, Callable
from pathlib import Path
from collections import deque

# --- 0. TYPE DEFINITIONS AND UTILITIES ---

Grid = List[List[int]]
Program = List[Tuple[str, List[Any]]]
TaskExample = Dict[str, Grid]
TaskData = List[TaskExample]
Task = Dict[str, TaskData]
SynthExample = Tuple[Program, Grid, Grid] # (Program, Input, Output)

def load_json(file_path: Path) -> Dict[str, Any]:
    """Utility to load JSON data."""
    with open(file_path, "r") as f:
        return json.load(f)

# --- 1. PHASE 1: FOUNDATIONAL SYMBOLIC SYSTEM (DSL & EXECUTION) ---

class DSL_Operation:
    """Base class for all operations in the DSL."""
    def __init__(self, name: str):
        self.name = name

    def apply(self, grid: np.ndarray, *args) -> np.ndarray:
        raise NotImplementedError

class Op_FindConnectedComponent(DSL_Operation):
    """Finds a specific connected component (e.g., largest) and isolates/crops it."""
    def __init__(self):
        super().__init__("FindCC")
    
    # Simple BFS/DFS based Connected Component labeling (avoids scipy dependency)
    def _label_grid(self, grid: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
        rows, cols = grid.shape
        labeled_array = np.zeros_like(grid, dtype=int)
        next_label = 1
        sizes = {}

        for r in range(rows):
            for c in range(cols):
                if grid[r, c] > 0 and labeled_array[r, c] == 0:
                    component_size = 0
                    q = deque([(r, c)])
                    labeled_array[r, c] = next_label
                    
                    while q:
                        cr, cc = q.popleft()
                        component_size += 1
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < rows and 0 <= nc < cols and \
                                grid[nr, nc] == grid[cr, cc] and labeled_array[nr, nc] == 0:
                                labeled_array[nr, nc] = next_label
                                q.append((nr, nc))
                    
                    sizes[next_label] = component_size
                    next_label += 1
        return labeled_array, sizes

    def apply(self, grid: np.ndarray, method: str = 'largest') -> np.ndarray:
        labeled_array, sizes = self._label_grid(grid)
        if not sizes: return np.zeros_like(grid)

        # 1. Select the component label
        if method == 'largest':
            target_label = max(sizes, key=sizes.get)
        else: # Default or error handling
            target_label = 1 

        # 2. Isolate
        mask = labeled_array == target_label
        result_grid = np.zeros_like(grid)
        result_grid[mask] = grid[mask]
        
        # 3. Crop (find bounding box)
        if np.any(mask):
            coords = np.argwhere(mask)
            r_min, c_min = coords.min(axis=0)
            r_max, c_max = coords.max(axis=0)
            return result_grid[r_min:r_max+1, c_min:c_max+1]
            
        return result_grid

class Op_ResizeAndFill(DSL_Operation):
    """Resizes the grid to a target size (rows, cols) and fills with a color."""
    def __init__(self):
        super().__init__("ResizeFill")

    def apply(self, grid: np.ndarray, target_r: int, target_c: int, fill_color: int) -> np.ndarray:
        new_grid = np.full((target_r, target_c), fill_color, dtype=np.uint8)
        
        # Calculate offset to center the existing content (complex, simplified here)
        r_start = (target_r - grid.shape[0]) // 2
        c_start = (target_c - grid.shape[1]) // 2
        
        r_end = r_start + grid.shape[0]
        c_end = c_start + grid.shape[1]
        
        # Clamp to bounds to handle inputs larger than target size (cropping)
        r_insert_end = min(r_end, target_r)
        c_insert_end = min(c_end, target_c)
        
        r_slice = slice(r_start, r_insert_end)
        c_slice = slice(c_start, c_insert_end)
        
        r_source_end = min(grid.shape[0], target_r - r_start)
        c_source_end = min(grid.shape[1], target_c - c_start)
        
        new_grid[r_slice, c_slice] = grid[:r_source_end, :c_source_end]
        return new_grid

class DSLExecutor:
    """Executes a sequence of DSL operations (a 'program')."""
    
    # P0: Define the complete set of DSL operators
    OP_MAP: Dict[str, Callable] = {
        "Identity": lambda g: g,
        "FindCC": Op_FindConnectedComponent().apply,
        "ResizeFill": Op_ResizeAndFill().apply,
        # Add more: Rotate90, FlipH, ColorSwap(c1, c2), BoundingBox, etc.
    }
    
    def execute_program(self, initial_grid: Grid, program_steps: Program) -> Grid:
        grid_np = np.array(initial_grid, dtype=np.uint8)
        
        for op_name, args in program_steps:
            if op_name in self.OP_MAP:
                try:
                    grid_np = self.OP_MAP[op_name](grid_np, *args)
                    # Enforce color and size constraints
                    grid_np = np.clip(grid_np, 0, 9)
                except Exception:
                    # Return a simple failure grid on execution error
                    return [[-1]] # Using -1 to denote execution failure
            else:
                return [[-2]] # Unknown op error

        return grid_np.astype(int).tolist()

# --- 2. PHASE 2: SYNTHETIC DATA GENERATION (Neuro-Symbolic Training Data) ---

class SyntheticDataGenerator:
    """Generates (Program, Input, Output) triples for training the neural model."""
    def __init__(self, executor: DSLExecutor):
        self.executor = executor
        self.max_grid_size = 10 
        self.max_program_length = 3
        self.colors = list(range(10))

    def generate_random_grid(self) -> Grid:
        """Generates a random, small grid."""
        r = random.randint(3, self.max_grid_size)
        c = random.randint(3, self.max_grid_size)
        # Randomly sample colors, majority 0 (empty)
        return np.random.choice(self.colors, size=(r, c), p=[0.7] + [0.3/9]*9).tolist()

    def generate_random_program(self) -> Program:
        """Generates a random sequence of DSL operations."""
        program_length = random.randint(1, self.max_program_length)
        program: Program = []
        for _ in range(program_length):
            op_name = random.choice(list(self.executor.OP_MAP.keys()))
            args: List[Any] = []

            # Hardcoded random args for simplicity
            if op_name == "ResizeFill":
                args = [random.randint(3, 15), random.randint(3, 15), random.choice(self.colors)]
            elif op_name == "FindCC":
                 args = [random.choice(['largest', 'smallest'])]
            # Add logic for other operations...
            
            program.append((op_name, args))
        return program

    def generate_example(self) -> Optional[SynthExample]:
        """Generates one valid (Program, Input, Output) example."""
        for _ in range(100): # Max attempts
            input_grid = self.generate_random_grid()
            program = self.generate_random_program()
            
            output_grid = self.executor.execute_program(input_grid, program)
            
            # Check for successful execution and non-trivial output
            if output_grid != [[-1]] and output_grid != [[-2]]:
                 # Check if the output is different from the input (non-trivial)
                if not np.array_equal(np.array(input_grid), np.array(output_grid)):
                    return (program, input_grid, output_grid)
        return None

# --- 3. PHASE 3: INFERENCE ARCHITECTURE (Neuro-Symbolic Beam Search) ---

class ProgramProposerModel:
    """
    MOCK-UP of the Transformer/Neural Network (Phase 2).
    In reality, this would be a loaded PyTorch/TensorFlow model.
    """
    def __init__(self, dsl_executor: DSLExecutor):
        self.ops = list(dsl_executor.OP_MAP.keys())
        # Mock-up of learned tokens/arguments
        self.tokens = self.ops + [str(i) for i in range(20)] + ['largest', 'smallest']
        
    def encode_task(self, examples: TaskData) -> np.ndarray:
        """Mocks the task encoding step (CNN/Transformer over I/O grids)."""
        # Returns a mock feature vector (e.g., flattened grids)
        input_sizes = [np.array(e['input']).size for e in examples]
        output_sizes = [np.array(e['output']).size for e in examples]
        return np.array(input_sizes + output_sizes)

    def predict_next_token_probs(self, encoded_features: np.ndarray, program_prefix: Program) -> Dict[str, float]:
        """
        Mocks the neural model predicting the probability distribution of the next token (Op or Argument).
        This is the core of the PSvNN (Program Synthesis via Neural Networks).
        """
        # --- SIMPLE MOCK-UP LOGIC ---
        
        # If the prefix is empty, prioritize high-level operations
        if not program_prefix:
            return {op: 1.0 / len(self.ops) for op in self.ops}

        # If the last token was an Op, suggest arguments next
        last_op, last_args = program_prefix[-1]
        
        if last_op == 'ResizeFill' and len(last_args) < 3:
            # Mock high probability for small numbers (grid sizes/colors)
            return {str(i): 0.1 for i in range(1, 10)}
            
        # Mock high probability for starting a new operation
        return {op: 0.8/len(self.ops) for op in self.ops}
    
    def is_program_complete(self, program: Program) -> bool:
        """Mocks the neural prediction of when the program should terminate."""
        return len(program) >= 1 and len(program) <= 3 # Simple length heuristic

class ARC_Synthesizer:
    """Integrates the neural proposer with the symbolic search (Phase 3)."""
    def __init__(self, executor: DSLExecutor, model: ProgramProposerModel):
        self.executor = executor
        self.model = model
        self.beam_width = 10 # Key hyperparameter for Phase 3

    def synthesize_program(self, task_examples: TaskData) -> Optional[Program]:
        """Performs Beam Search to find the optimal program."""
        
        encoded_features = self.model.encode_task(task_examples)
        
        # Beam: Stores (Program, Probability, Is_Valid)
        beam: List[Tuple[Program, float, bool]] = [([], 1.0, False)]
        
        best_valid_program: Optional[Program] = None
        
        # Search for a maximum program length (e.g., 5 operations)
        for search_step in range(5):
            new_beam: List[Tuple[Program, float, bool]] = []
            
            for current_program, current_prob, is_valid in beam:
                if is_valid:
                    new_beam.append((current_program, current_prob, True))
                    continue # Keep valid programs in the beam

                if self.model.is_program_complete(current_program):
                    # Check validity of the complete program
                    if self._check_program_validity(current_program, task_examples):
                        # Found a valid program! Return immediately or record best
                        return current_program 
                    continue

                # Get the next token probabilities from the mock neural model
                next_tokens = self.model.predict_next_token_probs(encoded_features, current_program)
                
                # Get top K tokens (Beam Search expansion)
                sorted_tokens = sorted(next_tokens.items(), key=lambda item: item[1], reverse=True)[:self.beam_width * 2]

                for token, token_prob in sorted_tokens:
                    new_program: Program = list(current_program)
                    
                    if token in self.model.ops:
                        # Start a new operation
                        new_program.append((token, []))
                    elif token.isdigit():
                        # Append an argument to the latest operation
                        op_name, args = new_program[-1]
                        args.append(int(token))
                        new_program[-1] = (op_name, args)
                    else: # Other symbolic arguments (e.g., 'largest')
                        op_name, args = new_program[-1]
                        args.append(token)
                        new_program[-1] = (op_name, args)
                        
                    new_prob = current_prob * token_prob
                    
                    # Add to the next beam
                    new_beam.append((new_program, new_prob, False))

            # Select the top K programs for the next iteration
            new_beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:self.beam_width]
            beam = new_beam
            
            # If the search space is exhausted, stop.
            if not beam: break
        
        # If no exact match was found, return the best effort (or None)
        return best_valid_program

    def _check_program_validity(self, program: Program, task_examples: TaskData) -> bool:
        """Validates if the program holds across all training examples."""
        for example in task_examples:
            predicted = self.executor.execute_program(example['input'], program)
            
            # Check for execution error or failure
            if predicted == [[-1]] or predicted == [[-2]]:
                return False
                
            # Check for exact match
            if not np.array_equal(np.array(predicted, dtype=object), np.array(example['output'], dtype=object)):
                return False
        return True
        
    def predict_grid(self, task: Task) -> List[Dict[str, Grid]]:
        """Applies the synthesized program to the test input."""
        
        program = self.synthesize_program(task['train'])
        
        predictions = []
        
        for test_example in task['test']:
            input_grid = test_example['input']
            
            predicted_output: Grid = [[0, 0], [0, 0]] # Default fallback
            
            if program is not None:
                predicted_output_list = self.executor.execute_program(input_grid, program)
                
                # Handle execution failures or empty/invalid grids
                if predicted_output_list not in ([[-1]], [[-2]]):
                    predicted_output = predicted_output_list
            
            # Store the prediction in the required two-attempt format
            predictions.append({
                "attempt_1": predicted_output,
                "attempt_2": predicted_output
            })
                
        return predictions

# --- 4. KAGGLE EXECUTION PIPELINE ---

def load_arc_data(base_path: str = '/kaggle/input/arc-prize-2025/') -> Dict[str, Task]:
    """Loads all test tasks from the ARC Prize 2025 competition data."""
    TEST_FILE = Path(base_path) / "arc-agi_test_challenges.json"
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Required test file not found at {TEST_FILE}. Check the input path.")
    
    print(f"Loading test challenges from: {TEST_FILE}")
    test_data = load_json(TEST_FILE)
    print(f"Successfully loaded {len(test_data)} test tasks.")
    return test_data

def generate_submission(synthesizer: ARC_Synthesizer) -> Dict[str, List[Dict[str, List[Grid]]]]:
    """Generates the final submission JSON structure by processing all test tasks."""
    
    test_tasks = load_arc_data()
    submission_data = {}
    
    for task_id, task_data in test_tasks.items():
        print(f"Processing Task: {task_id}")
        task: Task = task_data
        
        task_predictions = synthesizer.predict_grid(task)
        submission_data[task_id] = task_predictions
        
    return submission_data

def main():
    """Main execution function."""
    print("--- ARC Prize 2025: Neuro-Symbolic Synthesizer Start ---")
    
    # 1. Initialize DSL Executor (Phase 1)
    executor = DSLExecutor()
    
    # 2. (Simulated) Phase 2: Synthetic Data Generation & Model Training
    # NOTE: In a real submission, this step would involve a massive offline training process.
    # We only include the generation framework for completeness.
    generator = SyntheticDataGenerator(executor)
    synthetic_data = [generator.generate_example() for _ in range(100)] 
    print(f"Simulated generation of {len(synthetic_data)} synthetic examples.")
    
    # 3. Initialize Neuro-Symbolic Model (Phase 3)
    # The Proposer Model must be initialized before the Synthesizer
    proposer_model = ProgramProposerModel(executor)
    synthesizer = ARC_Synthesizer(executor, proposer_model)
    
    # 4. Generate Submission
    print("\nStarting Guided Program Synthesis on Test Data...")
    final_submission = generate_submission(synthesizer)
    
    # 5. Save to submission.json
    submission_file_path = "submission.json"
    with open(submission_file_path, 'w') as f:
        json.dump(final_submission, f, indent=4)
        
    print("\n--- Submission Generation Complete ---")
    print(f"Submission saved to: {submission_file_path}")

if __name__ == "__main__":
    np.random.seed(42) 
    random.seed(42)
    main()

