# %% [code]
# ======================
# ğŸ› ï¸� CONFIGURATION SETUP
# ======================
import os
import json
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Union
import time
from tqdm.auto import tqdm
from scipy.ndimage import binary_dilation, generate_binary_structure
from sklearn.cluster import KMeans
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

class Config:
    """Optimized configuration for ARC-AGI solver with GPU support"""
    DATA_PATH = '/kaggle/input/arc-prize-2025'
    SUBMISSION_FILE = '/kaggle/working/submission.json'
    MAX_ATTEMPTS = 2  # Competition requires exactly 2 attempts
    TIMEOUT_PER_TASK = 4.5  # Leaves buffer for notebook overhead
    DEBUG = False
    
    # Dynamic strategy parameters
    INITIAL_STRATEGY_WEIGHTS = [0.35, 0.25, 0.15, 0.1, 0.08, 0.05, 0.02]
    COLOR_PRIORITY = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]  # Ordered by frequency
    
    # GPU optimization
    USE_GPU = True  # Auto-detected below
    MAX_GRID_SIZE = 30  # Skip processing very large grids
    
    # Caching
    CACHE_SIZE = 500
    SHAPE_CACHE_SIZE = 1000

# Detect GPU availability
try:
    import torch
    Config.USE_GPU = torch.cuda.is_available()
except:
    Config.USE_GPU = False



# %% [code]
# ======================
# ğŸ§  CORE SOLVER CLASS
# ======================
class ARCSolver:
    """Advanced solver for ARC-AGI competition with multi-strategy approach"""
    
    def __init__(self):
        # Core strategies in execution order
        self.strategies = [
            self._solve_color_mapping,
            self._solve_grid_transforms,
            self._solve_pattern_extrapolation,
            self._solve_object_detection,
            self._solve_cluster_analysis,
            self._solve_composition,
            self._solve_fallback
        ]
        
        # Dynamic weights that adjust during execution
        self.strategy_weights = Config.INITIAL_STRATEGY_WEIGHTS.copy()
        self.strategy_success = [0] * len(self.strategies)
        self.strategy_attempts = [0] * len(self.strategies)
        
        # Enhanced caching system
        self._cache = {}
        self._shape_cache = {}
        self._task_type_cache = {}
        
        # GPU acceleration setup
        self.gpu_enabled = Config.USE_GPU
        if self.gpu_enabled:
            self._init_gpu_operations()

    def _init_gpu_operations(self):
        """Initialize GPU-accelerated operations"""
        try:
            import torch
            self.torch = torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Precompile common operations
            self.gpu_ops = {
                'rotate90': lambda x: torch.rot90(x, 1, [1, 2]),
                'flip_lr': lambda x: torch.flip(x, [2]),
                'flip_ud': lambda x: torch.flip(x, [1]),
            }
        except:
            self.gpu_enabled = False


# %% [code]
# ======================
# ğŸ�¯ MAIN SOLVING METHODS
# ======================
def solve_task(self, task: Dict) -> List[Dict]:
    """Main solving entry point with enhanced caching"""
    if not isinstance(task, dict) or 'test' not in task:
        return self._default_attempts()
        
    # Analyze task type for dynamic strategy adjustment
    task_type = self._analyze_task_type(task)
    self._adjust_weights_for_task_type(task_type)
    
    solutions = []
    for test_case in task['test']:
        if Config.DEBUG:
            print(f"\nProcessing test case with input shape {np.array(test_case['input']).shape}")
            
        attempts = self._solve_single_case(task, test_case['input'])
        
        # Ensure exactly 2 attempts as required by competition
        while len(attempts) < 2:
            attempts.append(self._generate_educated_guess(task, test_case['input']))
        attempts = attempts[:2]
        
        solutions.append({
            "attempt_1": attempts[0],
            "attempt_2": attempts[1]
        })
        
        if Config.DEBUG:
            self._visualize_attempts(test_case['input'], attempts)
    
    return solutions

def _solve_single_case(self, task: Dict, test_input: List[List[int]]) -> List[List[List[int]]]:
    """Core solving logic with three-phase approach"""
    attempts = []
    start_time = time.time()
    
    # Convert to numpy array once
    test_grid = np.array(test_input)
    
    # Phase 1: Weighted strategy execution
    for idx, (strategy, weight) in enumerate(zip(self.strategies, self.strategy_weights)):
        if self._timeout(start_time):
            break
            
        solution = strategy(task, test_grid)
        self.strategy_attempts[idx] += 1
        
        if solution and solution not in attempts:
            if self._validate_solution(solution, task):
                attempts.append(solution)
                self.strategy_success[idx] += 1
                if self._max_attempts_reached(attempts):
                    return attempts

    # Phase 2: Optimized strategy combinations
    if len(attempts) < Config.MAX_ATTEMPTS:
        for combo in [(0,1), (0,2), (1,2), (0,3), (1,3)]:  # Extended combinations
            if self._timeout(start_time):
                break
                
            intermediate = self.strategies[combo[0]](task, test_grid)
            if intermediate:
                solution = self.strategies[combo[1]](task, intermediate)
                if solution and solution not in attempts:
                    if self._validate_solution(solution, task):
                        attempts.append(solution)
                        if self._max_attempts_reached(attempts):
                            return attempts

    # Phase 3: Fallback with educated guesses
    while len(attempts) < Config.MAX_ATTEMPTS:
        educated_guess = self._generate_educated_guess(task, test_grid)
        if educated_guess not in attempts:
            attempts.append(educated_guess)
    
    return attempts[:Config.MAX_ATTEMPTS]


# %% [code]
# ======================
# ğŸ”� TASK ANALYSIS METHODS
# ======================
def _analyze_task_type(self, task: Dict) -> str:
    """Classify task type for dynamic strategy adjustment"""
    train_inputs = [ex['input'] for ex in task['train']]
    cache_key = f"task_type_{hash(str(train_inputs))}"
    if cache_key in self._task_type_cache:
        return self._task_type_cache[cache_key]
        
    # Extract features from training examples
    features = {
        'color_changes': 0,
        'grid_transforms': 0,
        'object_based': 0,
        'pattern_repeats': 0
    }
    
    for example in task['train']:
        inp = np.array(example['input'])
        out = np.array(example['output'])
        
        # Check color mapping
        if np.any(inp != out):
            features['color_changes'] += 1
            
        # Check grid transforms
        for transformed in self._generate_transforms(inp):
            if np.array_equal(transformed, out):
                features['grid_transforms'] += 1
                break
                
        # Check for objects
        if len(self._find_objects(inp)) > 0 and len(self._find_objects(out)) > 0:
            features['object_based'] += 1
            
        # Check repeating patterns
        if self._detect_repeating_pattern({'train': [example]}):
            features['pattern_repeats'] += 1
            
    # Determine dominant feature
    dominant = max(features, key=features.get)
    task_type = {
        'color_changes': 'color',
        'grid_transforms': 'grid',
        'object_based': 'object',
        'pattern_repeats': 'pattern'
    }[dominant]
    
    self._task_type_cache[cache_key] = task_type
    return task_type

def _adjust_weights_for_task_type(self, task_type: str):
    """Dynamically adjust strategy weights based on task type"""
    type_weights = {
        'color': [0.5, 0.2, 0.1, 0.1, 0.05, 0.03, 0.02],
        'grid': [0.2, 0.5, 0.1, 0.1, 0.05, 0.03, 0.02],
        'object': [0.1, 0.1, 0.2, 0.4, 0.1, 0.05, 0.05],
        'pattern': [0.1, 0.1, 0.5, 0.1, 0.1, 0.05, 0.05]
    }
    
    # Blend with current weights
    blend_factor = 0.3
    new_weights = type_weights.get(task_type, Config.INITIAL_STRATEGY_WEIGHTS)
    self.strategy_weights = [
        (1-blend_factor)*current + blend_factor*new
        for current, new in zip(self.strategy_weights, new_weights)
    ]
    
    # Normalize
    total = sum(self.strategy_weights)
    self.strategy_weights = [w/total for w in self.strategy_weights]


# %% [code]
# ======================
# âœ… VALIDATION METHODS
# ======================
def _validate_solution(self, solution: List[List[int]], task: Dict) -> bool:
    """Enhanced validation with shape and color checks"""
    if not solution:
        return False
        
    try:
        sol_arr = np.array(solution)
        shape_key = tuple(sol_arr.shape)
        
        # Shape validation
        if not self._valid_shape(shape_key, task):
            return False
            
        # Exact match check
        if self._exact_match(sol_arr, task):
            return True
            
        # Color validation
        if not self._valid_colors(sol_arr, task):
            return False
            
        # Pattern consistency check
        if not self._consistent_pattern(sol_arr, task):
            return False
            
        return True
    except:
        return False

def _valid_shape(self, shape_key: Tuple, task: Dict) -> bool:
    """Check if shape matches training examples"""
    if shape_key not in self._shape_cache:
        train_shapes = {tuple(np.array(ex['output']).shape) for ex in task['train']}
        self._shape_cache[shape_key] = shape_key in train_shapes
        if len(self._shape_cache) > Config.SHAPE_CACHE_SIZE:
            self._shape_cache.pop(next(iter(self._shape_cache)))
    return self._shape_cache[shape_key]

def _exact_match(self, sol_arr: np.ndarray, task: Dict) -> bool:
    """Check for exact match with any training output"""
    for example in task['train']:
        if np.array_equal(sol_arr, np.array(example['output'])):
            return True
    return False

def _valid_colors(self, sol_arr: np.ndarray, task: Dict) -> bool:
    """Validate solution colors against training examples"""
    sol_colors = set(sol_arr.flatten())
    train_colors = {c for ex in task['train'] for c in np.array(ex['output']).flatten()}
    return sol_colors.issubset(train_colors | {0})

def _consistent_pattern(self, sol_arr: np.ndarray, task: Dict) -> bool:
    """Check if solution maintains patterns from training examples"""
    # Implement pattern consistency checks
    return True  # Placeholder for actual implementation


# %% [code]
# ======================
# ğŸ�¨ COLOR MAPPING STRATEGY
# ======================
def _solve_color_mapping(self, task: Dict, test_input: Union[List[List[int]], np.ndarray]) -> Optional[List[List[int]]]:
    """Enhanced color transformation strategy with dynamic weights"""
    if isinstance(test_input, list):
        test_input = np.array(test_input)
        
    cache_key = f"color_{hash(str([ex['input'] for ex in task['train']]))}"
    if cache_key in self._cache:
        return self._apply_mapping(test_input, self._cache[cache_key]).tolist()
        
    try:
        color_map = self._build_color_map(task)
        if not color_map:
            return None
            
        self._cache[cache_key] = color_map
        if len(self._cache) > Config.CACHE_SIZE:
            self._cache.pop(next(iter(self._cache)))
            
        return self._apply_mapping(test_input, color_map).tolist()
    except:
        return None

def _build_color_map(self, task: Dict) -> Dict:
    """Build weighted color mapping with edge prioritization"""
    color_map = defaultdict(list)
    edge_boost = 1.5
    
    for example in task['train']:
        inp = np.array(example['input'])
        out = np.array(example['output'])
        h, w = inp.shape
        
        for i in range(h):
            for j in range(w):
                if inp[i,j] != out[i,j]:
                    # Boost edge pixels
                    weight = edge_boost if (i == 0 or i == h-1 or j == 0 or j == w-1) else 1.0
                    color_map[(inp[i,j], out[i,j])].append(weight)
    
    if not color_map:
        return {}
        
    # Select most frequent mappings with weighting
    best_map = {}
    for (src, dst), weights in color_map.items():
        total_weight = sum(weights)
        if src not in best_map or total_weight > sum(color_map[(src, best_map[src])]):
            best_map[src] = dst
            
    return best_map

def _apply_mapping(self, grid: np.ndarray, mapping: Dict) -> np.ndarray:
    """Apply color mapping efficiently"""
    output = np.copy(grid)
    for old, new in mapping.items():
        output[grid == old] = new
    return output


# %% [code]
# ======================
# â™»ï¸� GRID TRANSFORM STRATEGY
# ======================
def _solve_grid_transforms(self, task: Dict, test_input: Union[List[List[int]], np.ndarray]) -> Optional[List[List[int]]]:
    """Enhanced grid transformation strategy with GPU support"""
    if isinstance(test_input, list):
        test_input = np.array(test_input)
        
    cache_key = f"grid_{hash(str([ex['input'] for ex in task['train']]))}"
    if cache_key in self._cache:
        return self._cache[cache_key]
        
    try:
        transforms = self._generate_transforms(test_input)
        
        for example in task['train']:
            target = np.array(example['output'])
            for transformed in transforms:
                if np.array_equal(transformed, target):
                    self._cache[cache_key] = transformed.tolist()
                    if len(self._cache) > Config.CACHE_SIZE:
                        self._cache.pop(next(iter(self._cache)))
                    return transformed.tolist()
        
        return None
    except:
        return None

def _generate_transforms(self, grid: np.ndarray) -> List[np.ndarray]:
    """Generate all possible grid transformations with GPU acceleration"""
    if self.gpu_enabled:
        try:
            grid_tensor = self.torch.tensor(grid, device=self.device).unsqueeze(0).float()
            transforms = [
                self.gpu_ops['rotate90'](grid_tensor),
                self.gpu_ops['rotate90'](self.gpu_ops['rotate90'](grid_tensor)),
                self.gpu_ops['rotate90'](self.gpu_ops['rotate90'](self.gpu_ops['rotate90'](grid_tensor))),
                self.gpu_ops['flip_lr'](grid_tensor),
                self.gpu_ops['flip_ud'](grid_tensor),
                grid_tensor.transpose(1, 2),
                9 - grid_tensor
            ]
            return [t.squeeze(0).cpu().numpy().astype(int) for t in transforms]
        except:
            pass
            
    # CPU fallback
    return [
        np.rot90(grid, 1),
        np.rot90(grid, 2),
        np.rot90(grid, 3),
        np.fliplr(grid),
        np.flipud(grid),
        grid.T,
        9 - grid
    ]


# %% [code]
# ======================
# ğŸ–¼ï¸� VISUALIZATION METHODS
# ======================
def _visualize_attempts(self, input_grid: List[List[int]], attempts: List[List[List[int]]]):
    """Debug visualization (runs only when DEBUG=True)"""
    if not Config.DEBUG:
        return
        
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 1 + len(attempts), figsize=(12, 4))
        axes[0].imshow(np.array(input_grid), cmap='viridis')
        axes[0].set_title('Input')
        
        for i, attempt in enumerate(attempts, 1):
            axes[i].imshow(np.array(attempt), cmap='viridis')
            axes[i].set_title(f'Attempt {i}')
            
        plt.tight_layout()
        plt.show()
    except:
        pass


# %% [code]
# ======================
# â�±ï¸� UTILITY METHODS
# ======================
def _timeout(self, start_time: float) -> bool:
    """Check if timeout exceeded with safety margin"""
    return time.time() - start_time > (Config.TIMEOUT_PER_TASK * 0.9)

def _max_attempts_reached(self, attempts: List) -> bool:
    """Check if max attempts reached"""
    return len(attempts) >= Config.MAX_ATTEMPTS

def _generate_educated_guess(self, task: Dict, test_input: Union[List[List[int]], np.ndarray]) -> List[List[int]]:
    """Enhanced fallback generator with task-aware heuristics"""
    if isinstance(test_input, list):
        test_input = np.array(test_input)
        
    try:
        # Try most common output shape
        output_shapes = [np.array(ex['output']).shape for ex in task['train']]
        if len(set(output_shapes)) == 1:
            output_arrays = [np.array(ex['output']) for ex in task['train']]
            common_output = np.round(np.mean(output_arrays, axis=0)).astype(int)
            return common_output.tolist()
        
        # Try most common color change
        changes = []
        for example in task['train']:
            inp = np.array(example['input'])
            out = np.array(example['output'])
            changes.extend((out - inp).flatten())
        
        if changes:
            common_change = Counter(changes).most_common(1)[0][0]
            output = test_input + common_change
            return np.clip(output, 0, 9).tolist()
        
        # Fallback to input with most common output color
        output_colors = [c for ex in task['train'] for c in np.array(ex['output']).flatten()]
        if output_colors:
            common_color = Counter(output_colors).most_common(1)[0][0]
            return np.full_like(test_input, common_color).tolist()
        
        return test_input.tolist()
    except:
        return test_input.tolist()

def _default_attempts(self) -> List[Dict]:
    """Competition-required default response"""
    return [{"attempt_1": [[0]], "attempt_2": [[0]]}]


import os
import json
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Union
import time
from tqdm.auto import tqdm

class Config:
    DATA_PATH = '/kaggle/input/arc-prize-2025'
    SUBMISSION_FILE = '/kaggle/working/submission.json'
    MAX_ATTEMPTS = 2
    TIMEOUT_PER_TASK = 4.5
    DEBUG = False
    INITIAL_STRATEGY_WEIGHTS = [0.35, 0.25, 0.15, 0.1, 0.08, 0.05, 0.02]
    CACHE_SIZE = 500
    SHAPE_CACHE_SIZE = 1000
    USE_GPU = False

class ARCSolver:
    def __init__(self):
        self.strategies = [
            self._solve_color_mapping,
            self._solve_grid_transforms,
            self._solve_pattern_extrapolation,
            self._solve_object_detection,
            self._solve_cluster_analysis,
            self._solve_composition,
            self._solve_fallback
        ]
        self.strategy_weights = Config.INITIAL_STRATEGY_WEIGHTS.copy()
        self.strategy_success = [0] * len(self.strategies)
        self.strategy_attempts = [0] * len(self.strategies)
        self._cache = {}
        self._shape_cache = {}
        self._task_type_cache = {}
        self.gpu_enabled = Config.USE_GPU
        if self.gpu_enabled:
            self._init_gpu_operations()

    def _solve_color_mapping(self, task, test_input):
        """Fixed color mapping strategy with correct f-string syntax"""
        try:
            if isinstance(test_input, list):
                test_input = np.array(test_input)
            
            # Corrected f-string with proper parentheses
            train_inputs = str([ex['input'] for ex in task['train']])
            cache_key = f"color_{hash(train_inputs)}"
            
            if cache_key in self._cache:
                return self._apply_mapping(test_input, self._cache[cache_key]).tolist()
                
            color_map = self._build_color_map(task)
            if not color_map:
                return None
                
            self._cache[cache_key] = color_map
            if len(self._cache) > Config.CACHE_SIZE:
                self._cache.pop(next(iter(self._cache)))
                
            return self._apply_mapping(test_input, color_map).tolist()
        except:
            return None

    def _solve_grid_transforms(self, task, test_input):
        try:
            if isinstance(test_input, list):
                test_input = np.array(test_input)
                
            transforms = self._generate_transforms(test_input)
            for example in task['train']:
                target = np.array(example['output'])
                for transformed in transforms:
                    if np.array_equal(transformed, target):
                        return transformed.tolist()
            return None
        except:
            return None

    def _solve_pattern_extrapolation(self, task, test_input):
        return None
    
    def _solve_object_detection(self, task, test_input):
        return None
    
    def _solve_cluster_analysis(self, task, test_input):
        return None
    
    def _solve_composition(self, task, test_input):
        return None
    
    def _solve_fallback(self, task, test_input):
        try:
            return test_input.tolist() if hasattr(test_input, 'tolist') else test_input
        except:
            return [[0]]

    def _build_color_map(self, task):
        color_map = defaultdict(list)
        for example in task['train']:
            inp = np.array(example['input'])
            out = np.array(example['output'])
            for (i,j), val in np.ndenumerate(inp):
                if inp[i,j] != out[i,j]:
                    weight = 1.5 if (i == 0 or i == inp.shape[0]-1 or j == 0 or j == inp.shape[1]-1) else 1.0
                    color_map[(inp[i,j], out[i,j])].append(weight)
        return {k[0]: k[1] for k, v in color_map.items()}

    def _apply_mapping(self, grid, mapping):
        output = np.copy(grid)
        for old, new in mapping.items():
            output[grid == old] = new
        return output

    def _generate_transforms(self, grid):
        return [
            np.rot90(grid, 1),
            np.rot90(grid, 2),
            np.rot90(grid, 3),
            np.fliplr(grid),
            np.flipud(grid),
            grid.T,
            9 - grid
        ]

    def _init_gpu_operations(self):
        try:
            import torch
            self.torch = torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except:
            self.gpu_enabled = False

    def solve_task(self, task):
        if not isinstance(task, dict) or 'test' not in task:
            return [{"attempt_1": [[0]], "attempt_2": [[0]]}]
            
        solutions = []
        for test_case in task['test']:
            attempts = []
            test_grid = np.array(test_case['input']) if isinstance(test_case['input'], list) else test_case['input']
            
            for strategy in self.strategies:
                solution = strategy(task, test_grid)
                if solution is not None:
                    attempts.append(solution)
                    if len(attempts) >= Config.MAX_ATTEMPTS:
                        break
            
            while len(attempts) < Config.MAX_ATTEMPTS:
                attempts.append(self._solve_fallback(task, test_grid))
            
            solutions.append({
                "attempt_1": attempts[0],
                "attempt_2": attempts[1] if len(attempts) > 1 else attempts[0]
            })
        
        return solutions

def main():
    print("ğŸ�† ARC-SOLV Competition Ready")
    print(f"âš¡ Version: GPU-{'ENABLED' if Config.USE_GPU else 'DISABLED'}")
    print("-" * 50)
    
    try:
        with open(f'{Config.DATA_PATH}/arc-agi_test_challenges.json') as f:
            test_data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Data loading failed: {str(e)}")
    
    solver = ARCSolver()
    submission = {}
    start_time = time.time()
    
    for task_id, task in tqdm(test_data.items(), desc="Processing Tasks"):
        submission[task_id] = solver.solve_task(task)
    
    with open(Config.SUBMISSION_FILE, 'w') as f:
        json.dump(submission, f)
    
    print(f"\nâœ… Processed {len(submission)} tasks in {time.time()-start_time:.2f}s")
    print(f"âš¡ Average time per task: {(time.time()-start_time)/len(submission):.4f}s")

if __name__ == "__main__":
    main()

