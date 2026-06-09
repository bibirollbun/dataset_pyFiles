import numpy as np
import json
import os
from typing import Dict, Any, List, Tuple
from pathlib import Path

# Paths
INPUT_PATH = Path("/kaggle/input/arc-prize-2025") if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ else Path("./arc_data")
OUTPUT_PATH = Path("/kaggle/working") if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ else Path("./output")
OUTPUT_PATH.mkdir(exist_ok=True)

class MinimalSolver:
    """Minimal solver - finds the simplest transformation that works."""
    
    def solve(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> callable:
        """Find transformation from examples."""
        
        if not examples:
            return lambda x: x
        
        # Try transformations in order of complexity
        transformations = [
            # 1. Identity
            lambda x: x,
            
            # 2. Simple value replacement
            *[self._make_replace(v1, v2) for v1 in range(10) for v2 in range(10) if v1 != v2],
            
            # 3. Flips and rotations (if square)
            lambda x: np.flipud(x),
            lambda x: np.fliplr(x),
            lambda x: np.rot90(x, 1),
            lambda x: np.rot90(x, 2),
            lambda x: np.rot90(x, 3),
            lambda x: x.T if x.shape[0] == x.shape[1] else x,
            
            # 4. Fill patterns
            *[self._make_fill_value(v) for v in range(10)],
        ]
        
        # Find first transformation that works on all examples
        for transform in transformations:
            if self._test_transform(transform, examples):
                return transform
        
        # Nothing worked - return identity
        return lambda x: x
    
    def _make_replace(self, old_val: int, new_val: int) -> callable:
        """Create a value replacement function."""
        def replace(grid):
            result = grid.copy()
            result[grid == old_val] = new_val
            return result
        return replace
    
    def _make_fill_value(self, fill_val: int) -> callable:
        """Create a function that fills non-zero with a value."""
        def fill(grid):
            result = grid.copy()
            result[result != 0] = fill_val
            return result
        return fill
    
    def _test_transform(self, transform: callable, examples: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        """Test if transformation works on all examples."""
        for inp, expected in examples:
            try:
                output = transform(inp)
                if not np.array_equal(output, expected):
                    return False
            except:
                return False
        return True

def solve_task(task_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Solve a single ARC task."""
    # Extract training examples
    examples = []
    for case in task_data.get('train', []):
        if 'output' in case:
            inp = np.array(case['input'])
            out = np.array(case['output'])
            examples.append((inp, out))
    
    # Find transformation
    solver = MinimalSolver()
    transform = solver.solve(examples)
    
    # Apply to test cases
    predictions = []
    for test_case in task_data.get('test', []):
        test_input = np.array(test_case['input'])
        
        try:
            result = transform(test_input)
        except:
            result = test_input
        
        predictions.append({
            'attempt_1': result.tolist(),
            'attempt_2': result.tolist()  # Same prediction twice
        })
    
    return predictions

def main():
    """Main execution."""
    print("ğŸ§  Minimal GTMÃ˜ ARC Solver")
    print("=" * 50)
    print("Finding simplest transformation that works\n")
    
    # Load test data
    test_path = INPUT_PATH / 'arc-agi_test_challenges.json'
    if not test_path.exists():
        print(f"â�Œ Test file not found: {test_path}")
        return
    
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    
    print(f"ğŸ“Š Processing {len(test_data)} test tasks...")
    
    # Process all tasks
    submission = {}
    for i, (task_id, task_data) in enumerate(test_data.items()):
        if i % 50 == 0:
            print(f"Progress: {i}/{len(test_data)}")
        
        predictions = solve_task(task_data)
        submission[task_id] = predictions
    
    # Save submission
    output_file = OUTPUT_PATH / "submission.json"
    with open(output_file, 'w') as f:
        json.dump(submission, f, separators=(',', ':'))
    
    print(f"\nâœ… Submission saved to {output_file}")
    print("Done!")

if __name__ == "__main__":
    main()

