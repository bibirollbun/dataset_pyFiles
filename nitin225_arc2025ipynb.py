# ========================================
# ARC Prize 2025 ç»¼å�ˆæ±‚è§£ç³»ç»Ÿ - Kaggleé€‚é…�ç‰ˆ (JSONè¾“å‡º)
# ========================================

# å¯¼å…¥å¿…è¦�çš„åº“
import json
import numpy as np
import pandas as pd
from pathlib import Path
import time
import logging 
import os
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# è®¾ç½®æ—¥å¿—
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("=== ARC Prize 2025 ç»¼å�ˆæ±‚è§£ç³»ç»Ÿ ===")
print(f"å½“å‰�å·¥ä½œç›®å½•: {os.getcwd()}")

# æ£€æŸ¥å¤šä¸ªå�¯èƒ½çš„æ•°æ�®ç›®å½•è·¯å¾„
possible_data_dirs = [
    '../data',
    './data',
    '/kaggle/input/arc-prize-2025',
    '../input/arc-prize-2025',
    'data',
    '/kaggle/input'
]

data_dir = None
for dir_path in possible_data_dirs:
    if os.path.exists(dir_path):
        data_dir = dir_path
        print(f"[OK] æ‰¾åˆ°æ•°æ�®ç›®å½•: {data_dir}")
        print(f"æ•°æ�®ç›®å½•å†…å®¹: {os.listdir(data_dir)}")
        break
    else:
        print(f"[ERROR] æ•°æ�®ç›®å½•ä¸�å­˜åœ¨: {dir_path}")

if not data_dir:
    print("[WARNING] æœªæ‰¾åˆ°ä»»ä½•æ•°æ�®ç›®å½•ï¼Œå°†ä½¿ç”¨é»˜è®¤è·¯å¾„ './data'")
    data_dir = './data'

# ========================================
# ARCDataLoaderç±»
# ========================================
class ARCDataLoader:
    """ARCæ•°æ�®åŠ è½½å™¨"""
    
    def __init__(self, data_dir: str = './data'):
        self.data_dir = Path(data_dir)
        self.tasks = {}
        print(f"[INIT] ARCDataLoaderåˆ�å§‹åŒ–ï¼Œæ•°æ�®ç›®å½•: {self.data_dir}")
        
    def load_tasks(self) -> Dict[str, Any]:
        """åŠ è½½æ‰€æœ‰æµ‹è¯•ä»»åŠ¡"""
        print(f"[LOAD] å¼€å§‹åŠ è½½æµ‹è¯•ä»»åŠ¡ï¼Œæ•°æ�®ç›®å½•: {self.data_dir}")
        
        # æ£€æŸ¥å¤šç§�å�¯èƒ½çš„æµ‹è¯•æ–‡ä»¶å�� - ä¼˜å…ˆä½¿ç”¨evaluationæ•°æ�®è¿›è¡Œæœ¬åœ°æµ‹è¯•
        possible_files = [
            'arc-agi_evaluation_challenges.json',  # ä¼˜å…ˆä½¿ç”¨evaluationæ•°æ�®ï¼ŒåŒ…å�«æ­£ç¡®ç­”æ¡ˆ
            'arc-agi_test_challenges.json',
            'test_challenges.json', 
            'test.json',
            'arc-agi_test.json'
        ]
        
        test_file = None
        for filename in possible_files:
            file_path = self.data_dir / filename
            print(f"[CHECK] æ£€æŸ¥æ–‡ä»¶: {file_path}")
            if file_path.exists():
                test_file = file_path
                print(f"[OK] æ‰¾åˆ°æµ‹è¯•æ–‡ä»¶: {test_file}")
                break
            else:
                print(f"[ERROR] æ–‡ä»¶ä¸�å­˜åœ¨: {file_path}")
        
        if test_file and test_file.exists():
            try:
                print(f"ğŸ“– æ­£åœ¨è¯»å�–æ–‡ä»¶: {test_file}")
                # å°�è¯•å¤šç§�ç¼–ç �æ–¹å¼�
                encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
                for encoding in encodings:
                    try:
                        with open(test_file, 'r', encoding=encoding) as f:
                            self.tasks = json.load(f)
                        print(f"[OK] ä½¿ç”¨ {encoding} ç¼–ç �æˆ�åŠŸåŠ è½½ {len(self.tasks)} ä¸ªä»»åŠ¡")
                        break
                    except (UnicodeDecodeError, UnicodeEncodeError) as e:
                        print(f"[WARNING] {encoding} ç¼–ç �å¤±è´¥: {e}")
                        continue
                else:
                    # å¦‚æ�œæ‰€æœ‰ç¼–ç �éƒ½å¤±è´¥ï¼Œå°�è¯•äºŒè¿›åˆ¶æ¨¡å¼�
                    with open(test_file, 'rb') as f:
                        content = f.read().decode('utf-8', errors='ignore')
                        self.tasks = json.loads(content)
                    print(f"[OK] ä½¿ç”¨äºŒè¿›åˆ¶æ¨¡å¼�æˆ�åŠŸåŠ è½½ {len(self.tasks)} ä¸ªä»»åŠ¡")
                
                # æ˜¾ç¤ºå‰�å‡ ä¸ªä»»åŠ¡IDä½œä¸ºéªŒè¯�
                task_ids = list(self.tasks.keys())[:5]
                print(f"ğŸ“‹ å‰�5ä¸ªä»»åŠ¡ID: {task_ids}")
                
            except Exception as e:
                print(f"[ERROR] è¯»å�–æ–‡ä»¶å¤±è´¥: {e}")
                logger.error(f"è¯»å�–æµ‹è¯•æ–‡ä»¶å¤±è´¥: {e}")
                self.tasks = self._create_sample_tasks()
        else:
            print(f"[WARNING] æœªæ‰¾åˆ°ä»»ä½•æµ‹è¯•æ–‡ä»¶ï¼Œåˆ›å»ºç¤ºä¾‹ä»»åŠ¡")
            logger.warning(f"æµ‹è¯•æ–‡ä»¶ä¸�å­˜åœ¨ï¼Œå°�è¯•çš„è·¯å¾„: {[self.data_dir / f for f in possible_files]}")
            self.tasks = self._create_sample_tasks()
                
        logger.info(f"æœ€ç»ˆåŠ è½½äº† {len(self.tasks)} ä¸ªä»»åŠ¡")
        return self.tasks
    
    def _create_sample_tasks(self) -> Dict[str, Any]:
        """åˆ›å»ºç¤ºä¾‹ä»»åŠ¡ç”¨äº�æµ‹è¯•"""
        return {
            "sample_task": {
                "train": [
                    {
                        "input": [[0, 1], [1, 0]],
                        "output": [[1, 0], [0, 1]]
                    }
                ],
                "test": [
                    {"input": [[0, 1, 0], [1, 0, 1], [0, 1, 0]]}
                ]
            }
        }

# ========================================
# PatternAnalyzerç±»
# ========================================
class PatternAnalyzer:
    """æ¨¡å¼�åˆ†æ��å™¨"""
    
    def analyze_grid(self, grid: List[List[int]]) -> Dict[str, Any]:
        """åˆ†æ��ç½‘æ ¼æ¨¡å¼�"""
        grid = np.array(grid)
        
        return {
            'shape': grid.shape,
            'unique_colors': len(np.unique(grid)),
            'color_counts': dict(zip(*np.unique(grid, return_counts=True))),
            'symmetry': self._check_symmetry(grid),
            'patterns': self._detect_patterns(grid)
        }
    
    def _check_symmetry(self, grid: np.ndarray) -> Dict[str, bool]:
        """æ£€æŸ¥å¯¹ç§°æ€§"""
        return {
            'horizontal': np.array_equal(grid, np.flipud(grid)),
            'vertical': np.array_equal(grid, np.fliplr(grid)),
            'diagonal': np.array_equal(grid, grid.T) if grid.shape[0] == grid.shape[1] else False
        }
    
    def _detect_patterns(self, grid: np.ndarray) -> List[str]:
        """æ£€æµ‹æ¨¡å¼�"""
        patterns = []
        
        # æ£€æŸ¥é‡�å¤�æ¨¡å¼�
        if self._has_repeating_pattern(grid):
            patterns.append('repeating')
            
        # æ£€æŸ¥è¾¹ç•Œæ¨¡å¼�
        if self._has_border_pattern(grid):
            patterns.append('border')
            
        return patterns
    
    def _has_repeating_pattern(self, grid: np.ndarray) -> bool:
        """æ£€æŸ¥æ˜¯å�¦æœ‰é‡�å¤�æ¨¡å¼�"""
        h, w = grid.shape
        
        # æ£€æŸ¥2x2é‡�å¤�
        if h >= 4 and w >= 4:
            for i in range(0, h-1, 2):
                for j in range(0, w-1, 2):
                    if i+3 < h and j+3 < w:
                        block1 = grid[i:i+2, j:j+2]
                        block2 = grid[i+2:i+4, j+2:j+4]
                        if np.array_equal(block1, block2):
                            return True
        return False
    
    def _has_border_pattern(self, grid: np.ndarray) -> bool:
        """æ£€æŸ¥æ˜¯å�¦æœ‰è¾¹ç•Œæ¨¡å¼�"""
        h, w = grid.shape
        if h < 3 or w < 3:
            return False
            
        # æ£€æŸ¥è¾¹ç•Œæ˜¯å�¦ä¸�å†…éƒ¨ä¸�å�Œ
        border = np.concatenate([
            grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]
        ])
        interior = grid[1:-1, 1:-1].flatten()
        
        return len(set(border)) != len(set(interior))

# ========================================
# TransformationEngineç±»
# ========================================
class TransformationEngine:
    """å�˜æ�¢å¼•æ“�"""
    
    def __init__(self):
        self.transformations = [
            self._flip_horizontal,
            self._flip_vertical,
            self._rotate_90,
            self._rotate_180,
            self._rotate_270,
            self._transpose,
            self._invert_colors,
            self._shift_colors,
            self._fill_pattern,
            self._extract_objects
        ]
    
    def apply_transformations(self, grid: List[List[int]]) -> List[List[List[int]]]:
        """åº”ç”¨æ‰€æœ‰å�˜æ�¢"""
        grid = np.array(grid)
        results = []
        
        for transform in self.transformations:
            try:
                result = transform(grid)
                if result is not None:
                    results.append(result.tolist())
            except Exception as e:
                logger.debug(f"å�˜æ�¢å¤±è´¥: {transform.__name__}: {e}")
                
        return results
    
    def _flip_horizontal(self, grid: np.ndarray) -> np.ndarray:
        return np.flipud(grid)
    
    def _flip_vertical(self, grid: np.ndarray) -> np.ndarray:
        return np.fliplr(grid)
    
    def _rotate_90(self, grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid)
    
    def _rotate_180(self, grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, 2)
    
    def _rotate_270(self, grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, 3)
    
    def _transpose(self, grid: np.ndarray) -> np.ndarray:
        if grid.shape[0] == grid.shape[1]:
            return grid.T
        return grid
    
    def _invert_colors(self, grid: np.ndarray) -> np.ndarray:
        max_color = np.max(grid)
        return max_color - grid
    
    def _shift_colors(self, grid: np.ndarray) -> np.ndarray:
        unique_colors = np.unique(grid)
        if len(unique_colors) <= 1:
            return grid
        
        result = grid.copy()
        for i, color in enumerate(unique_colors):
            next_color = unique_colors[(i + 1) % len(unique_colors)]
            result[grid == color] = next_color
        return result
    
    def _fill_pattern(self, grid: np.ndarray) -> np.ndarray:
        result = grid.copy()
        h, w = grid.shape
        
        # ç®€å�•çš„å¡«å……æ¨¡å¼�ï¼šå°†0æ›¿æ�¢ä¸ºå‘¨å›´æœ€å¸¸è§�çš„é¢œè‰²
        for i in range(h):
            for j in range(w):
                if grid[i, j] == 0:
                    neighbors = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < h and 0 <= nj < w and (di != 0 or dj != 0):
                                neighbors.append(grid[ni, nj])
                    
                    if neighbors:
                        most_common = Counter(neighbors).most_common(1)[0][0]
                        if most_common != 0:
                            result[i, j] = most_common
        
        return result
    
    def _extract_objects(self, grid: np.ndarray) -> np.ndarray:
        """æ��å�–å¯¹è±¡"""
        # ç®€å�•çš„å¯¹è±¡æ��å�–ï¼šæ‰¾åˆ°é��é›¶åŒºåŸŸ
        non_zero_mask = grid != 0
        if not np.any(non_zero_mask):
            return grid
        
        # æ‰¾åˆ°è¾¹ç•Œæ¡†
        rows, cols = np.where(non_zero_mask)
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()
        
        # æ��å�–è¾¹ç•Œæ¡†å†…çš„å†…å®¹
        extracted = grid[min_row:max_row+1, min_col:max_col+1]
        return extracted
    
    def infer_transformation(self, input_grid: List[List[int]], output_grid: List[List[int]], input_patterns=None, output_patterns=None) -> str:
        """æ�¨æ–­è¾“å…¥åˆ°è¾“å‡ºçš„å�˜æ�¢ç±»å�‹"""
        input_array = np.array(input_grid)
        output_array = np.array(output_grid)
        
        # æ£€æŸ¥å°ºå¯¸å�˜åŒ–
        if input_array.shape != output_array.shape:
            return 'resize'
        
        # æ£€æŸ¥æ˜¯å�¦ä¸ºæ�’ç­‰å�˜æ�¢
        if np.array_equal(input_array, output_array):
            return 'identity'
        
        # æ£€æŸ¥æ—‹è½¬å�˜æ�¢
        for k in [1, 2, 3]:
            if np.array_equal(np.rot90(input_array, k), output_array):
                return f'rotate_{k*90}'
        
        # æ£€æŸ¥ç¿»è½¬å�˜æ�¢
        if np.array_equal(np.flipud(input_array), output_array):
            return 'flip_horizontal'
        if np.array_equal(np.fliplr(input_array), output_array):
            return 'flip_vertical'
        
        # æ£€æŸ¥è½¬ç½®
        if input_array.shape[0] == input_array.shape[1] and np.array_equal(input_array.T, output_array):
            return 'transpose'
        
        # æ£€æŸ¥é¢œè‰²å�˜æ�¢
        if np.array_equal(input_array.shape, output_array.shape):
            unique_input = set(input_array.flatten())
            unique_output = set(output_array.flatten())
            if len(unique_input) == len(unique_output):
                return 'color_mapping'
        
        return 'complex_transformation'
    
    def apply_transformation(self, grid: List[List[int]], transformation: str) -> Optional[List[List[int]]]:
        """åº”ç”¨æŒ‡å®šçš„å�˜æ�¢"""
        grid_array = np.array(grid)
        
        try:
            if transformation == 'identity':
                return grid
            elif transformation == 'rotate_90':
                return self._rotate_90(grid_array).tolist()
            elif transformation == 'rotate_180':
                return self._rotate_180(grid_array).tolist()
            elif transformation == 'rotate_270':
                return self._rotate_270(grid_array).tolist()
            elif transformation == 'flip_horizontal':
                return self._flip_horizontal(grid_array).tolist()
            elif transformation == 'flip_vertical':
                return self._flip_vertical(grid_array).tolist()
            elif transformation == 'transpose':
                return self._transpose(grid_array).tolist()
            elif transformation == 'invert_colors':
                return self._invert_colors(grid_array).tolist()
            elif transformation == 'shift_colors':
                return self._shift_colors(grid_array).tolist()
            elif transformation == 'fill_pattern':
                return self._fill_pattern(grid_array).tolist()
            elif transformation == 'extract_objects':
                return self._extract_objects(grid_array).tolist()
            else:
                return None
        except Exception as e:
            logger.debug(f"å�˜æ�¢åº”ç”¨å¤±è´¥ {transformation}: {e}")
            return None

# ========================================
# EnsembleSolverç±»
# ========================================
class EnsembleSolver:
    """é›†æˆ�æ±‚è§£å™¨"""
    
    def __init__(self):
        self.pattern_analyzer = PatternAnalyzer()
        self.transformation_engine = TransformationEngine()
        self.solution_cache = {}
    
    def solve_task(self, task: Dict[str, Any]) -> List[Dict[str, List[List[int]]]]:
        """æ±‚è§£å�•ä¸ªä»»åŠ¡ - å¤„ç�†æ‰€æœ‰æµ‹è¯•è¾“å…¥"""
        try:
            train_examples = task.get('train', [])
            test_examples = task.get('test', [])
            
            if not train_examples or not test_examples:
                # å¦‚æ�œæ²¡æœ‰æµ‹è¯•æ•°æ�®ï¼Œè¿”å›�é»˜è®¤è¾“å‡º
                return [{
                    "attempt_1": [[0, 0], [0, 0]],
                    "attempt_2": [[0, 0], [0, 0]]
                }]
            
            # åˆ†æ��è®­ç»ƒæ ·æœ¬
            patterns = self._analyze_training_examples(train_examples)
            
            # ä¸ºæ¯�ä¸ªæµ‹è¯•è¾“å…¥ç”Ÿæˆ�è§£å†³æ–¹æ¡ˆ
            all_solutions = []
            for test_example in test_examples:
                test_input = test_example.get('input', [])
                
                if not test_input:
                    # å¦‚æ�œæµ‹è¯•è¾“å…¥ä¸ºç©ºï¼Œä½¿ç”¨é»˜è®¤è¾“å‡º
                    solution = {
                        "attempt_1": [[0, 0], [0, 0]],
                        "attempt_2": [[0, 0], [0, 0]]
                    }
                else:
                    # ç”Ÿæˆ�å€™é€‰è§£
                    candidates = self._generate_candidates(test_input, patterns)
                    
                    # é€‰æ‹©æœ€ä½³è§£
                    best_solutions = self._select_best_solutions(candidates, patterns)
                    
                    # ç¡®ä¿�æœ‰ä¸¤ä¸ªè§£å†³æ–¹æ¡ˆ
                    while len(best_solutions) < 2:
                        if best_solutions:
                            best_solutions.append(best_solutions[0])  # å¤�åˆ¶ç¬¬ä¸€ä¸ªè§£
                        else:
                            default_output = self._generate_default_output(test_input)
                            best_solutions.append(default_output)
                    
                    solution = {
                        "attempt_1": best_solutions[0],
                        "attempt_2": best_solutions[1]
                    }
                
                all_solutions.append(solution)
            
            return all_solutions
            
        except Exception as e:
            logger.error(f"æ±‚è§£ä»»åŠ¡å¤±è´¥: {e}")
            # è¿”å›�é»˜è®¤è§£å†³æ–¹æ¡ˆ
            return [{
                "attempt_1": [[0, 0], [0, 0]],
                "attempt_2": [[0, 0], [0, 0]]
            }]
    
    def _analyze_training_examples(self, train_examples: List[Dict]) -> Dict[str, Any]:
        """åˆ†æ��è®­ç»ƒæ ·æœ¬"""
        input_patterns = []
        output_patterns = []
        transformations = []
        
        for example in train_examples:
            input_grid = example['input']
            output_grid = example['output']
            
            input_analysis = self.pattern_analyzer.analyze_grid(input_grid)
            output_analysis = self.pattern_analyzer.analyze_grid(output_grid)
            
            input_patterns.append(input_analysis)
            output_patterns.append(output_analysis)
            
            # å°�è¯•æ‰¾åˆ°å�˜æ�¢å…³ç³»
            transform = self._find_transformation(input_grid, output_grid)
            transformations.append(transform)
        
        return {
            'input_patterns': input_patterns,
            'output_patterns': output_patterns,
            'transformations': transformations
        }
    
    def _find_transformation(self, input_grid: List[List[int]], output_grid: List[List[int]]) -> str:
        """æ‰¾åˆ°è¾“å…¥åˆ°è¾“å‡ºçš„å�˜æ�¢"""
        input_array = np.array(input_grid)
        output_array = np.array(output_grid)
        
        # æ£€æŸ¥ç®€å�•å�˜æ�¢
        if np.array_equal(output_array, np.flipud(input_array)):
            return 'flip_horizontal'
        elif np.array_equal(output_array, np.fliplr(input_array)):
            return 'flip_vertical'
        elif np.array_equal(output_array, np.rot90(input_array)):
            return 'rotate_90'
        elif input_array.shape[0] == input_array.shape[1] and np.array_equal(output_array, input_array.T):
            return 'transpose'
        elif output_array.shape != input_array.shape:
            return 'resize'
        else:
            return 'complex'
    
    def _generate_candidates(self, test_input: List[List[int]], patterns: Dict[str, Any]) -> List[List[List[int]]]:
        """ç”Ÿæˆ�å€™é€‰è§£"""
        candidates = []
        
        # è®¾ç½®å½“å‰�æ¨¡å¼�ä¿¡æ�¯ï¼Œä¾›resizeå�˜æ�¢ä½¿ç”¨
        self._current_patterns = patterns
        
        # åŸºäº�å�˜æ�¢ç”Ÿæˆ�å€™é€‰è§£
        transformations = patterns.get('transformations', [])
        most_common_transform = Counter(transformations).most_common(1)
        
        if most_common_transform:
            transform_name = most_common_transform[0][0]
            candidate = self._apply_named_transformation(test_input, transform_name)
            if candidate is not None:
                candidates.append(candidate)
        
        # åº”ç”¨æ‰€æœ‰å�¯èƒ½çš„å�˜æ�¢
        all_transforms = self.transformation_engine.apply_transformations(test_input)
        candidates.extend(all_transforms)
        
        # å¦‚æ�œæ²¡æœ‰å€™é€‰è§£ï¼Œç”Ÿæˆ�é»˜è®¤è§£
        if not candidates:
            candidates.append(self._generate_default_output(test_input))
        
        return candidates
    
    def _apply_named_transformation(self, grid: List[List[int]], transform_name: str) -> Optional[List[List[int]]]:
        """åº”ç”¨æŒ‡å®šçš„å�˜æ�¢"""
        grid_array = np.array(grid)
        
        try:
            if transform_name == 'flip_horizontal':
                return np.flipud(grid_array).tolist()
            elif transform_name == 'flip_vertical':
                return np.fliplr(grid_array).tolist()
            elif transform_name == 'rotate_90':
                return np.rot90(grid_array).tolist()
            elif transform_name == 'transpose' and grid_array.shape[0] == grid_array.shape[1]:
                return grid_array.T.tolist()
            elif transform_name == 'resize':
                return self._apply_resize_transformation(grid)
            else:
                return None
        except Exception:
            return None
    
    def _select_best_solutions(self, candidates: List[List[List[int]]], patterns: Dict[str, Any]) -> List[List[List[int]]]:
        """é€‰æ‹©æœ€ä½³è§£"""
        if not candidates:
            return []
        
        # ç®€å�•çš„è¯„åˆ†æœºåˆ¶
        scored_candidates = []
        
        for candidate in candidates:
            score = self._score_candidate(candidate, patterns)
            scored_candidates.append((score, candidate))
        
        # æŒ‰åˆ†æ•°æ�’åº�
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # è¿”å›�å�»é‡�å��çš„å‰�å‡ ä¸ªè§£
        unique_solutions = []
        seen = set()
        
        for score, candidate in scored_candidates:
            candidate_str = str(candidate)
            if candidate_str not in seen:
                unique_solutions.append(candidate)
                seen.add(candidate_str)
                if len(unique_solutions) >= 2:
                    break
        
        return unique_solutions
    
    def _score_candidate(self, candidate: List[List[int]], patterns: Dict[str, Any]) -> float:
        """ä¸ºå€™é€‰è§£è¯„åˆ†"""
        score = 0.0
        
        try:
            candidate_analysis = self.pattern_analyzer.analyze_grid(candidate)
            output_patterns = patterns.get('output_patterns', [])
            
            if output_patterns:
                # ä¸�è®­ç»ƒè¾“å‡ºçš„ç›¸ä¼¼æ€§
                for output_pattern in output_patterns:
                    if candidate_analysis['shape'] == output_pattern['shape']:
                        score += 1.0
                    if candidate_analysis['unique_colors'] == output_pattern['unique_colors']:
                        score += 0.5
            
            # å¤�æ�‚åº¦æƒ©ç½š
            if candidate_analysis['unique_colors'] > 5:
                score -= 0.2
            
        except Exception:
            score = 0.0
        
        return score
    
    def _apply_resize_transformation(self, grid: List[List[int]]) -> List[List[int]]:
        """åº”ç”¨å°ºå¯¸å�˜æ�¢"""
        # ä»�è®­ç»ƒæ ·æœ¬ä¸­æ�¨æ–­ç›®æ ‡å°ºå¯¸
        if hasattr(self, '_current_patterns') and self._current_patterns:
            output_patterns = self._current_patterns.get('output_patterns', [])
            if output_patterns:
                # åˆ†æ��æ‰€æœ‰è¾“å‡ºå°ºå¯¸ï¼Œå¯»æ‰¾æ¨¡å¼�
                output_shapes = [pattern.get('shape', (len(grid), len(grid[0]))) for pattern in output_patterns]
                
                # å°�è¯•æ‰¾åˆ°å°ºå¯¸è§„å¾‹
                target_shape = self._predict_output_size(grid, output_shapes)
                target_h, target_w = target_shape
                return self._resize_grid_to_target(grid, (target_h, target_w))
        
        # å¦‚æ�œæ²¡æœ‰æ¨¡å¼�ä¿¡æ�¯ï¼Œè¿”å›�å�Ÿç½‘æ ¼
        return grid
    
    def _resize_grid_to_target(self, grid: List[List[int]], target_shape: Tuple[int, int]) -> List[List[int]]:
        """å°†ç½‘æ ¼è°ƒæ•´åˆ°ç›®æ ‡å°ºå¯¸"""
        target_h, target_w = target_shape
        current_h, current_w = len(grid), len(grid[0]) if grid else 0
        
        if current_h == target_h and current_w == target_w:
            return grid
        
        # æ™ºèƒ½ç¼©æ”¾ç­–ç•¥
        grid_array = np.array(grid)
        result = np.zeros((target_h, target_w), dtype=int)
        
        # è®¡ç®—ç¼©æ”¾å› å­�
        scale_h = current_h / target_h if target_h > 0 else 1
        scale_w = current_w / target_w if target_w > 0 else 1
        
        for i in range(target_h):
            for j in range(target_w):
                # è®¡ç®—æº�åŒºåŸŸ
                src_i_start = int(i * scale_h)
                src_i_end = int((i + 1) * scale_h)
                src_j_start = int(j * scale_w)
                src_j_end = int((j + 1) * scale_w)
                
                # ç¡®ä¿�ä¸�è¶…å‡ºè¾¹ç•Œ
                src_i_end = min(src_i_end, current_h)
                src_j_end = min(src_j_end, current_w)
                src_i_start = min(src_i_start, current_h - 1)
                src_j_start = min(src_j_start, current_w - 1)
                
                # æ��å�–åŒºåŸŸå¹¶é€‰æ‹©æœ€å¸¸è§�çš„é¢œè‰²
                if src_i_start < current_h and src_j_start < current_w:
                    region = grid_array[src_i_start:src_i_end, src_j_start:src_j_end]
                    if region.size > 0:
                        flat_region = region.flatten()
                        non_zero = flat_region[flat_region != 0]
                        if len(non_zero) > 0:
                            result[i, j] = Counter(non_zero).most_common(1)[0][0]
                        else:
                            result[i, j] = Counter(flat_region).most_common(1)[0][0]
        
        return result.tolist()
    
    def _predict_output_size(self, grid: List[List[int]], training_shapes: List[tuple]) -> tuple:
        """åŸºäº�ARC-AGIè§„åˆ™çš„å°ºå¯¸é¢„æµ‹ç®—æ³• - é‡�ç‚¹å…³æ³¨å†…å®¹åˆ†æ��"""
        input_height, input_width = len(grid), len(grid[0])
        
        if not training_shapes:
            return (input_height, input_width)
        
        # æ”¶é›†æ‰€æœ‰å”¯ä¸€çš„è¾“å‡ºå°ºå¯¸
        unique_outputs = list(set(training_shapes))
        
        # ç­–ç•¥1: å›ºå®šå°ºå¯¸æ¨¡å¼�ï¼ˆæ‰€æœ‰è®­ç»ƒæ ·æœ¬è¾“å‡ºç›¸å�Œï¼‰
        if len(unique_outputs) == 1:
            return unique_outputs[0]
        
        # ç­–ç•¥2: åŸºäº�ç½‘æ ¼å†…å®¹çš„æ™ºèƒ½åˆ†æ��
        # è¿™æ˜¯ARC-AGIä»»åŠ¡çš„æ ¸å¿ƒ - è¾“å‡ºå°ºå¯¸é€šå¸¸ä¸�è¾“å…¥å†…å®¹çš„æŸ�ç§�æ¨¡å¼�ç›¸å…³
        content_based_size = self._analyze_grid_content_for_size(grid, unique_outputs)
        if content_based_size:
            return content_based_size
        
        # ç­–ç•¥3: åŸºäº�è®­ç»ƒæ ·æœ¬çš„ç»Ÿè®¡åˆ†æ��
        # è®¡ç®—æœ€å¸¸è§�çš„å°ºå¯¸
        from collections import Counter
        size_counts = Counter(training_shapes)
        most_common_size = size_counts.most_common(1)[0][0]
        
        # å¦‚æ�œæŸ�ä¸ªå°ºå¯¸å‡ºç�°é¢‘ç�‡è¶…è¿‡40%ï¼Œä¼˜å…ˆé€‰æ‹©
        if size_counts[most_common_size] >= len(training_shapes) * 0.4:
            return most_common_size
        
        # ç­–ç•¥4: å°ºå¯¸èŒƒå›´åˆ†æ��
        # é€‰æ‹©åœ¨å�ˆç�†èŒƒå›´å†…çš„å°ºå¯¸ï¼ˆä¸�ä¼šå¤ªå¤§æˆ–å¤ªå°�ï¼‰
        reasonable_sizes = []
        for size in unique_outputs:
            h, w = size
            # æ�’é™¤å¼‚å¸¸å¤§çš„å°ºå¯¸ï¼ˆè¶…è¿‡è¾“å…¥å°ºå¯¸çš„ä¸€å�Šï¼‰
            if h <= input_height // 2 and w <= input_width // 2:
                # æ�’é™¤å¼‚å¸¸å°�çš„å°ºå¯¸ï¼ˆå°�äº�3x3ï¼Œé™¤é��æ‰€æœ‰å°ºå¯¸éƒ½å¾ˆå°�ï¼‰
                if h >= 3 and w >= 3:
                    reasonable_sizes.append(size)
        
        if reasonable_sizes:
            # ä»�å�ˆç�†å°ºå¯¸ä¸­é€‰æ‹©æœ€æ�¥è¿‘å¹³å�‡å€¼çš„
            avg_h = sum(s[0] for s in reasonable_sizes) / len(reasonable_sizes)
            avg_w = sum(s[1] for s in reasonable_sizes) / len(reasonable_sizes)
            
            best_size = reasonable_sizes[0]
            min_distance = float('inf')
            
            for size in reasonable_sizes:
                distance = abs(size[0] - avg_h) + abs(size[1] - avg_w)
                if distance < min_distance:
                    min_distance = distance
                    best_size = size
            
            return best_size
        
        # ç­–ç•¥5: å¦‚æ�œæ‰€æœ‰å°ºå¯¸éƒ½å¾ˆå°�ï¼Œé€‰æ‹©æœ€å¤§çš„
        if all(h <= 10 and w <= 10 for h, w in unique_outputs):
            return max(unique_outputs, key=lambda x: x[0] * x[1])
        
        # æœ€å��çš„å¤‡é€‰ï¼šè¿”å›�æœ€å¸¸è§�çš„å°ºå¯¸
        return most_common_size
    

    
    def _analyze_grid_content_for_size(self, grid: List[List[int]], possible_sizes: List[tuple]) -> Optional[tuple]:
        """åŸºäº�ç½‘æ ¼å†…å®¹åˆ†æ��é¢„æµ‹è¾“å‡ºå°ºå¯¸ - é’ˆå¯¹é��æ•´é™¤å…³ç³»çš„ARCä»»åŠ¡"""
        try:
            grid_array = np.array(grid)
            h, w = grid_array.shape
            
            # åˆ†æ��1: æ£€æµ‹ç½‘æ ¼ä¸­çš„å¯¹è±¡å’ŒåŒºåŸŸ
            unique_values = np.unique(grid_array[grid_array != 0])
            num_colors = len(unique_values)
            
            # åˆ†æ��2: æ£€æµ‹è¿�é€šåŒºåŸŸæ•°é‡�
            # è¿™å�¯èƒ½ä¸�è¾“å‡ºå°ºå¯¸ç›¸å…³
            connected_regions = self._count_connected_regions(grid_array)
            
            # åˆ†æ��3: åŸºäº�é¢œè‰²åˆ†å¸ƒé€‰æ‹©å°ºå¯¸
            # æŸ�äº›ARCä»»åŠ¡çš„è¾“å‡ºå°ºå¯¸ä¸�é¢œè‰²ç§�ç±»æ•°é‡�ç›¸å…³
            for target_h, target_w in possible_sizes:
                total_cells = target_h * target_w
                
                # ç­–ç•¥3.1: é¢œè‰²æ•°é‡�åŒ¹é…�
                if abs(num_colors - total_cells) <= 1:
                    return (target_h, target_w)
                
                # ç­–ç•¥3.2: è¿�é€šåŒºåŸŸæ•°é‡�åŒ¹é…�
                if abs(connected_regions - total_cells) <= 1:
                    return (target_h, target_w)
            
            # åˆ†æ��4: åŸºäº�ç½‘æ ¼ç‰¹å¾�çš„å�¯å�‘å¼�é€‰æ‹©
            # è®¡ç®—ç½‘æ ¼çš„"å¤�æ�‚åº¦"æ�¥é€‰æ‹©å�ˆé€‚çš„è¾“å‡ºå°ºå¯¸
            complexity_score = self._calculate_grid_complexity(grid_array)
            
            # æ ¹æ�®å¤�æ�‚åº¦é€‰æ‹©è¾“å‡ºå°ºå¯¸
            if complexity_score > 0.7:  # é«˜å¤�æ�‚åº¦
                # é€‰æ‹©è¾ƒå¤§çš„è¾“å‡ºå°ºå¯¸
                return max(possible_sizes, key=lambda x: x[0] * x[1])
            elif complexity_score < 0.3:  # ä½�å¤�æ�‚åº¦
                # é€‰æ‹©è¾ƒå°�çš„è¾“å‡ºå°ºå¯¸
                return min(possible_sizes, key=lambda x: x[0] * x[1])
            
            # åˆ†æ��5: åŸºäº�ç½‘æ ¼å†…å®¹çš„æ¨¡å¼�è¯†åˆ«
            # å°�è¯•è¯†åˆ«å�¯èƒ½çš„å¯¹è±¡æˆ–ç»“æ�„
            pattern_score = {}
            for target_h, target_w in possible_sizes:
                score = self._score_size_based_on_content(grid_array, target_h, target_w)
                pattern_score[(target_h, target_w)] = score
            
            # é€‰æ‹©å¾—åˆ†æœ€é«˜çš„å°ºå¯¸
            if pattern_score:
                best_size = max(pattern_score.keys(), key=lambda x: pattern_score[x])
                if pattern_score[best_size] > 0.1:  # å�ªæœ‰å½“å¾—åˆ†è¶³å¤Ÿé«˜æ—¶æ‰�è¿”å›�
                    return best_size
            
            return None
            
        except Exception:
            return None
    
    def _count_connected_regions(self, grid_array: np.ndarray) -> int:
        """è®¡ç®—è¿�é€šåŒºåŸŸæ•°é‡�"""
        try:
            from scipy import ndimage
            # å°†é��é›¶åŒºåŸŸæ ‡è®°ä¸ºè¿�é€šåŒºåŸŸ
            labeled_array, num_features = ndimage.label(grid_array != 0)
            return num_features
        except ImportError:
            # å¦‚æ�œæ²¡æœ‰scipyï¼Œä½¿ç”¨ç®€å�•çš„æ–¹æ³•
            visited = np.zeros_like(grid_array, dtype=bool)
            regions = 0
            
            def dfs(i, j):
                if (i < 0 or i >= grid_array.shape[0] or j < 0 or j >= grid_array.shape[1] or 
                    visited[i, j] or grid_array[i, j] == 0):
                    return
                visited[i, j] = True
                for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    dfs(i + di, j + dj)
            
            for i in range(grid_array.shape[0]):
                for j in range(grid_array.shape[1]):
                    if not visited[i, j] and grid_array[i, j] != 0:
                        dfs(i, j)
                        regions += 1
            
            return regions
    
    def _calculate_grid_complexity(self, grid_array: np.ndarray) -> float:
        """è®¡ç®—ç½‘æ ¼å¤�æ�‚åº¦"""
        try:
            # å¤�æ�‚åº¦åŸºäº�å¤šä¸ªå› ç´ 
            h, w = grid_array.shape
            
            # å› ç´ 1: é¢œè‰²å¤šæ ·æ€§
            unique_colors = len(np.unique(grid_array[grid_array != 0]))
            color_diversity = unique_colors / 9.0  # å�‡è®¾æœ€å¤š9ç§�é¢œè‰²
            
            # å› ç´ 2: ç©ºé—´åˆ†å¸ƒçš„å�‡åŒ€æ€§
            non_zero_positions = np.where(grid_array != 0)
            if len(non_zero_positions[0]) > 0:
                spread_h = (np.max(non_zero_positions[0]) - np.min(non_zero_positions[0])) / h
                spread_w = (np.max(non_zero_positions[1]) - np.min(non_zero_positions[1])) / w
                spatial_spread = (spread_h + spread_w) / 2
            else:
                spatial_spread = 0
            
            # å› ç´ 3: å¯†åº¦
            density = np.count_nonzero(grid_array) / (h * w)
            
            # ç»¼å�ˆå¤�æ�‚åº¦
            complexity = (color_diversity * 0.4 + spatial_spread * 0.3 + density * 0.3)
            return min(1.0, complexity)
            
        except Exception:
            return 0.5  # é»˜è®¤ä¸­ç­‰å¤�æ�‚åº¦
    
    def _score_size_based_on_content(self, grid_array: np.ndarray, target_h: int, target_w: int) -> float:
        """åŸºäº�å†…å®¹ä¸ºç‰¹å®šå°ºå¯¸è¯„åˆ†"""
        try:
            score = 0.0
            
            # è¯„åˆ†1: åŸºäº�é¢œè‰²æ•°é‡�ä¸�ç›®æ ‡å°ºå¯¸çš„å…³ç³»
            unique_colors = len(np.unique(grid_array[grid_array != 0]))
            target_cells = target_h * target_w
            
            if unique_colors > 0:
                color_ratio = min(unique_colors, target_cells) / max(unique_colors, target_cells)
                score += color_ratio * 0.5
            
            # è¯„åˆ†2: åŸºäº�ç½‘æ ¼å¯†åº¦ä¸�ç›®æ ‡å°ºå¯¸çš„é€‚é…�æ€§
            density = np.count_nonzero(grid_array) / (grid_array.shape[0] * grid_array.shape[1])
            
            # é«˜å¯†åº¦é€‚å�ˆè¾ƒå¤§è¾“å‡ºï¼Œä½�å¯†åº¦é€‚å�ˆè¾ƒå°�è¾“å‡º
            size_factor = (target_h * target_w) / 100.0  # å½’ä¸€åŒ–
            density_match = 1 - abs(density - size_factor)
            score += max(0, density_match) * 0.3
            
            # è¯„åˆ†3: åŸºäº�å°ºå¯¸çš„å�ˆç�†æ€§
            # å��å¥½ä¸­ç­‰å¤§å°�çš„è¾“å‡ºï¼ˆä¸�å¤ªå¤§ä¹Ÿä¸�å¤ªå°�ï¼‰
            if 6 <= target_cells <= 40:
                score += 0.2
            
            return score
            
        except Exception:
            return 0.0
    
    def _check_repeating_pattern(self, grid_array: np.ndarray, pattern_h: int, pattern_w: int) -> bool:
        """æ£€æŸ¥ç½‘æ ¼ä¸­æ˜¯å�¦å­˜åœ¨æŒ‡å®šå°ºå¯¸çš„é‡�å¤�æ¨¡å¼�"""
        try:
            h, w = grid_array.shape
            if h < pattern_h or w < pattern_w:
                return False
            
            # æ��å�–ç¬¬ä¸€ä¸ªæ¨¡å¼�å�—
            pattern = grid_array[:pattern_h, :pattern_w]
            
            # æ£€æŸ¥æ˜¯å�¦åœ¨å…¶ä»–ä½�ç½®é‡�å¤�
            matches = 0
            total_checks = 0
            
            for i in range(0, h - pattern_h + 1, pattern_h):
                for j in range(0, w - pattern_w + 1, pattern_w):
                    block = grid_array[i:i+pattern_h, j:j+pattern_w]
                    total_checks += 1
                    if np.array_equal(pattern, block):
                        matches += 1
            
            # å¦‚æ�œè‡³å°‘50%çš„å�—åŒ¹é…�ï¼Œè®¤ä¸ºå­˜åœ¨é‡�å¤�æ¨¡å¼�
            return matches >= total_checks * 0.5 if total_checks > 0 else False
            
        except Exception:
            return False
    
    def _generate_default_output(self, test_input: List[List[int]]) -> List[List[int]]:
        """ç”Ÿæˆ�é»˜è®¤è¾“å‡º"""
        if not test_input:
            return [[0]]
        
        # è¿”å›�ä¸�è¾“å…¥ç›¸å�Œå¤§å°�çš„é›¶çŸ©é˜µ
        h, w = len(test_input), len(test_input[0]) if test_input else 1
        return [[0] * w for _ in range(h)]

# ========================================
# ä¸»æ‰§è¡Œä»£ç �
# ========================================

# 1. åŠ è½½æ•°æ�®
print("æ­£åœ¨åŠ è½½æ•°æ�®...")
data_loader = ARCDataLoader(data_dir)
tasks = data_loader.load_tasks()

if not tasks:
    print("[ERROR] æ²¡æœ‰åŠ è½½åˆ°ä»»åŠ¡æ•°æ�®")
    # åˆ›å»ºä¸€ä¸ªç©ºçš„æ��äº¤æ–‡ä»¶é�¿å…�é”™è¯¯
    submission_data = {}
else:
    print(f"[OK] æˆ�åŠŸåŠ è½½ {len(tasks)} ä¸ªä»»åŠ¡")

# 2. åˆ�å§‹åŒ–æ±‚è§£å™¨
solver = EnsembleSolver()
submission_data = {}

# 3. å¤„ç�†æ¯�ä¸ªä»»åŠ¡
print("å¼€å§‹å¤„ç�†ä»»åŠ¡...")
for i, (task_id, task_data) in enumerate(tasks.items()):
    try:
        print(f"å¤„ç�†ä»»åŠ¡ {i+1}/{len(tasks)}: {task_id}")
        
        # æ±‚è§£ä»»åŠ¡ - ç�°åœ¨è¿”å›�æ­£ç¡®æ ¼å¼�çš„è§£å†³æ–¹æ¡ˆåˆ—è¡¨
        solutions = solver.solve_task(task_data)
        
        # å­˜å‚¨è§£å†³æ–¹æ¡ˆåˆ—è¡¨ï¼ˆæ¯�ä¸ªæµ‹è¯•ç”¨ä¾‹ä¸€ä¸ªè§£å†³æ–¹æ¡ˆï¼‰
        submission_data[task_id] = solutions
        
        print(f"[OK] ä»»åŠ¡ {task_id}: ç”Ÿæˆ�äº† {len(solutions)} ä¸ªæµ‹è¯•è¾“å‡º")
        
    except Exception as e:
        print(f"[ERROR] ä»»åŠ¡ {task_id} å¤„ç�†å¤±è´¥: {e}")
        # ä½¿ç”¨é»˜è®¤è§£å†³æ–¹æ¡ˆ
        submission_data[task_id] = [{
            "attempt_1": [[0, 0], [0, 0]],
            "attempt_2": [[0, 0], [0, 0]]
        }]

print(f"[OK] å®Œæˆ�å¤„ç�† {len(submission_data)} ä¸ªä»»åŠ¡")

# 4. åˆ›å»ºJSONæ ¼å¼�çš„æ��äº¤æ–‡ä»¶
print("åˆ›å»ºæ��äº¤æ–‡ä»¶...")

# ç¡®ä¿�åœ¨å½“å‰�å·¥ä½œç›®å½•åˆ›å»ºæ–‡ä»¶
submission_path = './submission.json'

# ç›´æ�¥ä¿�å­˜ä¸ºJSONæ ¼å¼�
with open(submission_path, 'w') as f:
    json.dump(submission_data, f, indent=2)

# éªŒè¯�æ–‡ä»¶æ˜¯å�¦åˆ›å»ºæˆ�åŠŸ
if os.path.exists(submission_path):
    print(f"[OK] æ��äº¤æ–‡ä»¶å·²æˆ�åŠŸåˆ›å»º: {submission_path}")
else:
    print("[ERROR] æ��äº¤æ–‡ä»¶åˆ›å»ºå¤±è´¥")
    
# ä¹Ÿå°�è¯•åœ¨å¤šä¸ªå�¯èƒ½çš„ä½�ç½®åˆ›å»ºæ–‡ä»¶
try:
    # åœ¨å·¥ä½œç›®å½•åˆ›å»º
    with open('submission.json', 'w') as f:
        json.dump(submission_data, f, indent=2)
    print("[OK] åœ¨å½“å‰�ç›®å½•åˆ›å»ºäº†submission.json")
except Exception as e:
    print(f"[ERROR] åˆ›å»ºæ–‡ä»¶å¤±è´¥: {e}")

# åˆ—å‡ºå½“å‰�ç›®å½•çš„æ–‡ä»¶ï¼Œç¡®è®¤æ–‡ä»¶å­˜åœ¨
print(f"å½“å‰�ç›®å½•æ–‡ä»¶: {os.listdir('.')}")  
if 'submission.json' in os.listdir('.'):
    print("[OK] ç¡®è®¤submission.jsonæ–‡ä»¶å­˜åœ¨")

print(f"[OK] æ��äº¤æ–‡ä»¶å·²ä¿�å­˜: {submission_path}")
print(f"[INFO] æ��äº¤æ–‡ä»¶å¤§å°�: {os.path.getsize(submission_path)} bytes")
print(f"[INFO] æ��äº¤ä»»åŠ¡æ•°: {len(submission_data)}")

# 5. æ˜¾ç¤ºæ��äº¤æ–‡ä»¶é¢„è§ˆ
print("\næ��äº¤æ–‡ä»¶é¢„è§ˆ:")
sample_tasks = list(submission_data.keys())[:3]  # æ˜¾ç¤ºå‰�3ä¸ªä»»åŠ¡
for task_id in sample_tasks:
    test_outputs = submission_data[task_id]  # è�·å�–æ‰€æœ‰æµ‹è¯•è¾“å‡º
    print(f"ä»»åŠ¡ {task_id}: {len(test_outputs)} ä¸ªæµ‹è¯•è¾“å‡º")
    for i, output in enumerate(test_outputs):
        attempt1 = output["attempt_1"]
        attempt2 = output["attempt_2"]
        print(f"  æµ‹è¯•è¾“å‡º {i+1}:")
        print(f"    å°�è¯•1: {len(attempt1)}x{len(attempt1[0]) if attempt1 else 0} ç½‘æ ¼")
        print(f"    å°�è¯•2: {len(attempt2)}x{len(attempt2[0]) if attempt2 else 0} ç½‘æ ¼")

print("\n=== å¤„ç�†å®Œæˆ� ===")
print(f"æ€»ä»»åŠ¡æ•°: {len(tasks)}")
print(f"æˆ�åŠŸå¤„ç�†: {len(submission_data)}")
print(f"æ��äº¤æ–‡ä»¶: {submission_path}")
print("å‡†å¤‡æ��äº¤åˆ°ç«�èµ›ï¼�")

# ========================================
# AllInAllç±» - EnsembleSolverçš„åˆ«å��
# ========================================
class AllInAll(EnsembleSolver):
    """AllInAllæ±‚è§£å™¨ - EnsembleSolverçš„åˆ«å��"""
    
    def __init__(self):
        super().__init__()
    
    def solve_task(self, train_data: List[Dict], test_input: List[List[int]]) -> List[List[List[int]]]:
        """æ±‚è§£å�•ä¸ªä»»åŠ¡çš„æµ‹è¯•è¾“å…¥"""
        # æ�„é€ ä»»åŠ¡æ•°æ�®æ ¼å¼�
        task_data = {
            'train': train_data,
            'test': [{'input': test_input}]
        }
        
        # è°ƒç”¨çˆ¶ç±»æ–¹æ³•
        solutions = super().solve_task(task_data)
        
        # æ��å�–å€™é€‰è§£
        candidates = []
        if solutions and len(solutions) > 0:
            solution = solutions[0]  # å�–ç¬¬ä¸€ä¸ªæµ‹è¯•è¾“å‡º
            if 'attempt_1' in solution:
                candidates.append(solution['attempt_1'])
            if 'attempt_2' in solution:
                candidates.append(solution['attempt_2'])
        
        # å¦‚æ�œæ²¡æœ‰å€™é€‰è§£ï¼Œè¿”å›�é»˜è®¤è§£
        if not candidates:
            candidates = [[[0, 0], [0, 0]], [[1, 1], [1, 1]]]
        
        return candidates

