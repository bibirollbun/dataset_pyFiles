#!/usr/bin/env python3
"""
æœ€ç»ˆKaggle ARCæ±‚è§£å™¨ - ç»“å�ˆåŸºç¡€æ¨¡å¼�è¯†åˆ«å’Œå�¥å£®é”™è¯¯å¤„ç�†
åŸºäº�åŸºç¡€æ±‚è§£å™¨åœ¨ç®€å�•æ¨¡å¼�ä¸Š90%å‡†ç¡®ç�‡çš„æˆ�åŠŸç»�éªŒ
ç›®æ ‡ï¼šåœ¨çœŸå®�ARCæ•°æ�®ä¸Šå®�ç�°5-15%çš„å‡†ç¡®ç�‡çª�ç ´
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
import os
import sys
from pathlib import Path
import time
from dataclasses import dataclass
from enum import Enum

class TransformType(Enum):
    """å�˜æ�¢ç±»å�‹æ�šä¸¾"""
    IDENTITY = "identity"
    FLIP_H = "flip_horizontal"
    FLIP_V = "flip_vertical"
    ROTATE_90 = "rotate_90"
    ROTATE_180 = "rotate_180"
    ROTATE_270 = "rotate_270"
    TRANSPOSE = "transpose"
    COLOR_MAP = "color_map"
    EXTRACT_CORNER = "extract_corner"
    SCALE_DOWN = "scale_down"
    FILL_PATTERN = "fill_pattern"
    EXTRACT_OBJECTS = "extract_objects"

@dataclass
class SolverResult:
    """æ±‚è§£å™¨ç»“æ�œ"""
    predictions: List[List[List[int]]]
    confidence: float
    transform_used: Optional[str] = None
    solve_time: float = 0.0
    error: Optional[str] = None

class FinalKaggleARCSolver:
    """æœ€ç»ˆKaggle ARCæ±‚è§£å™¨"""
    
    def __init__(self):
        self.name = "FinalKaggleARCSolver"
        self.version = "1.0"
        self.transforms = self._initialize_transforms()
        self.pattern_cache = {}
        
    def _initialize_transforms(self) -> Dict[str, Callable]:
        """åˆ�å§‹åŒ–å�˜æ�¢å‡½æ•°åº“"""
        return {
            TransformType.IDENTITY.value: self._identity,
            TransformType.FLIP_H.value: self._flip_horizontal,
            TransformType.FLIP_V.value: self._flip_vertical,
            TransformType.ROTATE_90.value: self._rotate_90,
            TransformType.ROTATE_180.value: self._rotate_180,
            TransformType.ROTATE_270.value: self._rotate_270,
            TransformType.TRANSPOSE.value: self._transpose,
            TransformType.COLOR_MAP.value: self._color_map,
            TransformType.EXTRACT_CORNER.value: self._extract_corner,
            TransformType.SCALE_DOWN.value: self._scale_down,
            TransformType.FILL_PATTERN.value: self._fill_pattern,
            TransformType.EXTRACT_OBJECTS.value: self._extract_objects,
        }
    
    def solve_task(self, task: Dict[str, Any]) -> SolverResult:
        """æ±‚è§£ARCä»»åŠ¡"""
        start_time = time.time()
        
        try:
            # æ��å�–è®­ç»ƒå’Œæµ‹è¯•æ•°æ�®
            train_data = task.get('train', [])
            test_data = task.get('test', [])
            
            if not train_data or not test_data:
                return SolverResult(
                    predictions=[],
                    confidence=0.0,
                    error="ç¼ºå°‘è®­ç»ƒæˆ–æµ‹è¯•æ•°æ�®"
                )
            
            # åˆ†æ��è®­ç»ƒæ ·æœ¬ï¼Œæ‰¾åˆ°æœ€ä½³å�˜æ�¢
            best_transform, confidence = self._find_best_transform(train_data)
            
            if best_transform is None:
                # å¦‚æ�œæ‰¾ä¸�åˆ°å�˜æ�¢ï¼Œè¿”å›�è¾“å…¥çš„å‰¯æœ¬
                predictions = []
                for test_case in test_data:
                    input_grid = test_case.get('input', [])
                    if input_grid:
                        predictions.append([row[:] for row in input_grid])
                    else:
                        predictions.append([[0]])
                
                return SolverResult(
                    predictions=predictions,
                    confidence=0.1,
                    transform_used="identity_fallback",
                    solve_time=time.time() - start_time
                )
            
            # åº”ç”¨æœ€ä½³å�˜æ�¢åˆ°æµ‹è¯•ç”¨ä¾‹
            predictions = []
            for test_case in test_data:
                input_grid = test_case.get('input', [])
                if input_grid:
                    try:
                        prediction = self.transforms[best_transform](input_grid)
                        predictions.append(prediction)
                    except Exception as e:
                        # å�˜æ�¢å¤±è´¥æ—¶è¿”å›�è¾“å…¥å‰¯æœ¬
                        predictions.append([row[:] for row in input_grid])
                else:
                    predictions.append([[0]])
            
            return SolverResult(
                predictions=predictions,
                confidence=confidence,
                transform_used=best_transform,
                solve_time=time.time() - start_time
            )
            
        except Exception as e:
            # å…¨å±€é”™è¯¯å¤„ç�†
            predictions = []
            for test_case in test_data:
                input_grid = test_case.get('input', [])
                if input_grid:
                    predictions.append([row[:] for row in input_grid])
                else:
                    predictions.append([[0]])
            
            return SolverResult(
                predictions=predictions,
                confidence=0.0,
                error=str(e),
                solve_time=time.time() - start_time
            )
    
    def _find_best_transform(self, train_data: List[Dict]) -> Tuple[Optional[str], float]:
        """æ‰¾åˆ°æœ€ä½³å�˜æ�¢"""
        transform_scores = {}
        
        for transform_name, transform_func in self.transforms.items():
            correct_count = 0
            total_count = len(train_data)
            
            for sample in train_data:
                input_grid = sample.get('input', [])
                expected_output = sample.get('output', [])
                
                if not input_grid or not expected_output:
                    continue
                
                try:
                    result = transform_func(input_grid)
                    if self._grids_equal(result, expected_output):
                        correct_count += 1
                except:
                    continue
            
            if total_count > 0:
                accuracy = correct_count / total_count
                transform_scores[transform_name] = accuracy
        
        if not transform_scores:
            return None, 0.0
        
        # æ‰¾åˆ°æœ€é«˜åˆ†æ•°çš„å�˜æ�¢
        best_transform = max(transform_scores.keys(), key=lambda k: transform_scores[k])
        best_score = transform_scores[best_transform]
        
        # å�ªæœ‰å½“å‡†ç¡®ç�‡å¤§äº�0æ—¶æ‰�è¿”å›�å�˜æ�¢
        if best_score > 0:
            return best_transform, best_score
        else:
            return None, 0.0
    
    def _grids_equal(self, grid1: List[List[int]], grid2: List[List[int]]) -> bool:
        """æ¯”è¾ƒä¸¤ä¸ªç½‘æ ¼æ˜¯å�¦ç›¸ç­‰"""
        if len(grid1) != len(grid2):
            return False
        
        for i in range(len(grid1)):
            if len(grid1[i]) != len(grid2[i]):
                return False
            for j in range(len(grid1[i])):
                if grid1[i][j] != grid2[i][j]:
                    return False
        
        return True
    
    # åŸºç¡€å�˜æ�¢å‡½æ•°
    def _identity(self, grid: List[List[int]]) -> List[List[int]]:
        """æ�’ç­‰å�˜æ�¢"""
        return [row[:] for row in grid]
    
    def _flip_horizontal(self, grid: List[List[int]]) -> List[List[int]]:
        """æ°´å¹³ç¿»è½¬"""
        return [row[::-1] for row in grid]
    
    def _flip_vertical(self, grid: List[List[int]]) -> List[List[int]]:
        """å�‚ç›´ç¿»è½¬"""
        return grid[::-1]
    
    def _rotate_90(self, grid: List[List[int]]) -> List[List[int]]:
        """é¡ºæ—¶é’ˆæ—‹è½¬90åº¦"""
        rows, cols = len(grid), len(grid[0])
        return [[grid[rows-1-j][i] for j in range(rows)] for i in range(cols)]
    
    def _rotate_180(self, grid: List[List[int]]) -> List[List[int]]:
        """æ—‹è½¬180åº¦"""
        return [row[::-1] for row in grid[::-1]]
    
    def _rotate_270(self, grid: List[List[int]]) -> List[List[int]]:
        """é¡ºæ—¶é’ˆæ—‹è½¬270åº¦"""
        rows, cols = len(grid), len(grid[0])
        return [[grid[j][cols-1-i] for j in range(rows)] for i in range(cols)]
    
    def _transpose(self, grid: List[List[int]]) -> List[List[int]]:
        """è½¬ç½®"""
        rows, cols = len(grid), len(grid[0])
        return [[grid[j][i] for j in range(rows)] for i in range(cols)]
    
    def _color_map(self, grid: List[List[int]]) -> List[List[int]]:
        """é¢œè‰²æ˜ å°„ï¼ˆç®€å�•ç‰ˆæœ¬ï¼š0->1, 1->0ï¼‰"""
        result = []
        for row in grid:
            new_row = []
            for cell in row:
                if cell == 0:
                    new_row.append(1)
                elif cell == 1:
                    new_row.append(0)
                else:
                    new_row.append(cell)
            result.append(new_row)
        return result
    
    def _extract_corner(self, grid: List[List[int]]) -> List[List[int]]:
        """æ��å�–å·¦ä¸Šè§’2x2åŒºåŸŸ"""
        if len(grid) < 2 or len(grid[0]) < 2:
            return [[0, 0], [0, 0]]
        
        return [[grid[0][0], grid[0][1]], [grid[1][0], grid[1][1]]]
    
    def _scale_down(self, grid: List[List[int]]) -> List[List[int]]:
        """ç¼©å°�ä¸€å�Š"""
        rows, cols = len(grid), len(grid[0])
        new_rows, new_cols = max(1, rows // 2), max(1, cols // 2)
        
        result = []
        for i in range(new_rows):
            row = []
            for j in range(new_cols):
                # å�–2x2åŒºåŸŸçš„å·¦ä¸Šè§’å€¼
                row.append(grid[i*2][j*2])
            result.append(row)
        
        return result
    
    def _fill_pattern(self, grid: List[List[int]]) -> List[List[int]]:
        """å¡«å……æ¨¡å¼�ï¼ˆå°†0å¡«å……ä¸º1ï¼‰"""
        return [[1 if cell == 0 else cell for cell in row] for row in grid]
    
    def _extract_objects(self, grid: List[List[int]]) -> List[List[int]]:
        """æ��å�–å¯¹è±¡ï¼ˆä¿�ç•™é��é›¶å€¼ï¼‰"""
        return [[cell if cell != 0 else 0 for cell in row] for row in grid]

class KaggleSubmissionGenerator:
    """Kaggleæ��äº¤æ–‡ä»¶ç”Ÿæˆ�å™¨"""
    
    def __init__(self, solver: FinalKaggleARCSolver):
        self.solver = solver
    
    def generate_submission(self, test_data_paths, output_path: str = "submission.json") -> bool:
        """ç”Ÿæˆ�Kaggleæ��äº¤æ–‡ä»¶"""
        try:
            # å¤šè·¯å¾„æ•°æ�®åŠ è½½ï¼ˆå�‚è€ƒv12æ ¼å¼�ï¼‰
            test_data = self._load_test_data(test_data_paths)
            
            if not test_data:
                print(f"[ERROR] æ— æ³•åŠ è½½æµ‹è¯•æ•°æ�®")
                return False
            
            submission = {}
            total_tasks = len(test_data)
            
            print(f"[INFO] å¼€å§‹å¤„ç�† {total_tasks} ä¸ªæµ‹è¯•ä»»åŠ¡...")
            
            # æ˜¾ç¤ºå‰�å‡ ä¸ªä»»åŠ¡IDä½œä¸ºéªŒè¯�
            task_ids = list(test_data.keys())[:5]
            print(f"ğŸ“‹ å‰�5ä¸ªä»»åŠ¡ID: {task_ids}")
            
            for i, (task_id, task_data) in enumerate(test_data.items()):
                print(f"å¤„ç�†ä»»åŠ¡ {i+1}/{total_tasks}: {task_id}")
                
                try:
                    # æ±‚è§£ä»»åŠ¡
                    result = self.solver.solve_task(task_data)
                    
                    # æ ¼å¼�åŒ–é¢„æµ‹ç»“æ�œ
                    task_predictions = []
                    for j, prediction in enumerate(result.predictions):
                        # ç¡®ä¿�é¢„æµ‹æ˜¯æœ‰æ•ˆçš„ç½‘æ ¼
                        if prediction and len(prediction) > 0:
                            task_predictions.append({
                                "attempt_1": prediction,
                                "attempt_2": prediction  # æ��ä¾›ä¸¤ä¸ªç›¸å�Œçš„å°�è¯•
                            })
                        else:
                            # æ��ä¾›é»˜è®¤é¢„æµ‹
                            task_predictions.append({
                                "attempt_1": [[0]],
                                "attempt_2": [[0]]
                            })
                    
                    submission[task_id] = task_predictions
                    
                    if result.error:
                        print(f"  è­¦å‘Šï¼š{result.error}")
                    else:
                        print(f"  æˆ�åŠŸï¼šä½¿ç”¨å�˜æ�¢ {result.transform_used}ï¼Œç½®ä¿¡åº¦ {result.confidence:.2f}")
                        
                except Exception as e:
                    print(f"  é”™è¯¯ï¼šå¤„ç�†ä»»åŠ¡ {task_id} æ—¶å‡ºé”™: {e}")
                    # æ��ä¾›é»˜è®¤é¢„æµ‹
                    submission[task_id] = [{
                        "attempt_1": [[0]],
                        "attempt_2": [[0]]
                    }]
            
            # ä¿�å­˜æ��äº¤æ–‡ä»¶
            with open(output_path, 'w') as f:
                json.dump(submission, f, indent=2)
            
            print(f"\nâœ“ æ��äº¤æ–‡ä»¶å·²ä¿�å­˜åˆ°: {output_path}")
            print(f"âœ“ å¤„ç�†äº† {len(submission)} ä¸ªä»»åŠ¡")
            
            return True
            
        except Exception as e:
            print(f"ç”Ÿæˆ�æ��äº¤æ–‡ä»¶æ—¶å‡ºé”™: {e}")
            return False
    
    def _load_test_data(self, test_data_paths) -> Dict[str, Any]:
        """å¤šè·¯å¾„æ•°æ�®åŠ è½½ï¼ˆå�‚è€ƒv12æ ¼å¼�ï¼‰"""
        # å¦‚æ�œä¼ å…¥çš„æ˜¯å­—ç¬¦ä¸²ï¼Œè½¬æ�¢ä¸ºåˆ—è¡¨
        if isinstance(test_data_paths, str):
            possible_paths = [
                test_data_paths,
                os.path.join(os.getcwd(), test_data_paths)
            ]
        else:
            possible_paths = test_data_paths
        
        # æ·»åŠ é¢�å¤–çš„Kaggleç‰¹å®šè·¯å¾„
        possible_paths.extend([
            "/kaggle/input/arc-prize-2025/arc-agi_test-challenges.json",
            "/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json",
            "/kaggle/working/arc-agi_test-challenges.json",
            "/kaggle/working/arc-agi_test_challenges.json",
            "./arc-agi_test-challenges.json",
            "./arc-agi_test_challenges.json",
            "../input/arc-prize-2025/arc-agi_test-challenges.json",
            "../input/arc-prize-2025/arc-agi_test_challenges.json"
        ])
        
        # å°�è¯•å¤šç§�ç¼–ç �æ–¹å¼�
        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
        
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    print(f"å°�è¯•åŠ è½½æµ‹è¯•æ•°æ�®: {path}")
                    
                    # å°�è¯•ä¸�å�Œçš„ç¼–ç �
                    for encoding in encodings:
                        try:
                            with open(path, 'r', encoding=encoding) as f:
                                data = json.load(f)
                            print(f"[OK] ä½¿ç”¨ {encoding} ç¼–ç �æˆ�åŠŸåŠ è½½æµ‹è¯•æ•°æ�®: {path}")
                            print(f"[INFO] åŠ è½½äº† {len(data)} ä¸ªä»»åŠ¡")
                            return data
                        except (UnicodeDecodeError, UnicodeEncodeError):
                            continue
                    
                    # å¦‚æ�œæ‰€æœ‰ç¼–ç �éƒ½å¤±è´¥ï¼Œå°�è¯•äºŒè¿›åˆ¶æ¨¡å¼�
                    with open(path, 'rb') as f:
                        content = f.read().decode('utf-8', errors='ignore')
                        data = json.loads(content)
                    print(f"[OK] ä½¿ç”¨äºŒè¿›åˆ¶æ¨¡å¼�æˆ�åŠŸåŠ è½½æµ‹è¯•æ•°æ�®: {path}")
                    print(f"[INFO] åŠ è½½äº† {len(data)} ä¸ªä»»åŠ¡")
                    return data
            except Exception as e:
                print(f"[WARNING] åŠ è½½ {path} å¤±è´¥: {e}")
                continue
        
        print("[WARNING] æ— æ³•æ‰¾åˆ°æµ‹è¯•æ•°æ�®æ–‡ä»¶ï¼Œåˆ›å»ºç¤ºä¾‹æ•°æ�®")
        return self._create_sample_test_data()
    
    def _create_sample_test_data(self) -> Dict[str, Any]:
        """åˆ›å»ºç¤ºä¾‹æµ‹è¯•æ•°æ�®"""
        return {
            "sample_task_1": {
                "train": [
                    {
                        "input": [[1, 0], [0, 1]],
                        "output": [[0, 1], [1, 0]]
                    }
                ],
                "test": [
                    {
                        "input": [[1, 1], [0, 0]]
                    }
                ]
            }
        }

def main():
    """ä¸»å‡½æ•°"""
    print("=== ARC Prize 2025 æœ€ç»ˆKaggleæ±‚è§£å™¨ ===")
    print(f"å½“å‰�å·¥ä½œç›®å½•: {os.getcwd()}")
    
    # æ£€æŸ¥å¤šä¸ªå�¯èƒ½çš„æ•°æ�®ç›®å½•è·¯å¾„
    possible_data_dirs = [
        '../data',
        './data',
        '/kaggle/input/arc-prize-2025',
        '../input/arc-prize-2025',
        'data',
        '/kaggle/input',
        '.',
        '..',
        '../input',
        './input'
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
    
    # æ£€æŸ¥å¤šç§�å�¯èƒ½çš„æµ‹è¯•æ–‡ä»¶å��
    possible_files = [
        'arc-agi_test-challenges.json',
        'arc-agi_test_challenges.json',
        'arc-agi_evaluation-challenges.json',
        'arc-agi_evaluation_challenges.json',
        'test_challenges.json', 
        'test.json',
        'arc-agi_test.json'
    ]
    
    # ç”Ÿæˆ�æ‰€æœ‰å�¯èƒ½çš„æ–‡ä»¶è·¯å¾„
    all_possible_paths = []
    for dir_path in possible_data_dirs:
        for filename in possible_files:
            all_possible_paths.append(os.path.join(dir_path, filename))
    
    # æ·»åŠ é¢�å¤–çš„Kaggleç‰¹å®šè·¯å¾„
    all_possible_paths.extend([
        "/kaggle/input/arc-prize-2025/arc-agi_test-challenges.json",
        "/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json",
        "/kaggle/working/arc-agi_test-challenges.json",
        "/kaggle/working/arc-agi_test_challenges.json"
    ])
    
    # åˆ›å»ºæ±‚è§£å™¨
    print("æ­£åœ¨åˆ�å§‹åŒ–æ±‚è§£å™¨...")
    solver = FinalKaggleARCSolver()
    generator = KaggleSubmissionGenerator(solver)
    
    # ç”Ÿæˆ�æ��äº¤æ–‡ä»¶
    print("\nç”Ÿæˆ�Kaggleæ��äº¤æ–‡ä»¶...")
    success = generator.generate_submission(
        all_possible_paths,  # ä¼ é€’æ‰€æœ‰å�¯èƒ½çš„æ–‡ä»¶è·¯å¾„
        "submission.json"    # ä½¿ç”¨æ ‡å‡†çš„submission.jsonæ–‡ä»¶å��
    )
    
    if success:
        print("\nâœ“ æœ€ç»ˆKaggleæ±‚è§£å™¨å‡†å¤‡å®Œæˆ�ï¼�")
        print("âœ“ åŸºäº�90%ç®€å�•æ¨¡å¼�å‡†ç¡®ç�‡çš„æˆ�åŠŸç»�éªŒ")
        print("âœ“ é›†æˆ�å�¥å£®é”™è¯¯å¤„ç�†å’Œå¤šè·¯å¾„æ•°æ�®åŠ è½½")
        print("âœ“ ç›®æ ‡ï¼šåœ¨çœŸå®�ARCæ•°æ�®ä¸Šå®�ç�°5-15%å‡†ç¡®ç�‡çª�ç ´")
        
        # éªŒè¯�æ–‡ä»¶æ˜¯å�¦åˆ›å»ºæˆ�åŠŸ
        if os.path.exists("submission.json"):
            print(f"[OK] æ��äº¤æ–‡ä»¶å·²æˆ�åŠŸåˆ›å»º: submission.json")
            print(f"[INFO] æ��äº¤æ–‡ä»¶å¤§å°�: {os.path.getsize('submission.json')} bytes")
            
            # æ˜¾ç¤ºæ��äº¤æ–‡ä»¶é¢„è§ˆ
            try:
                with open("submission.json", "r") as f:
                    submission_data = json.load(f)
                print(f"[INFO] æ��äº¤ä»»åŠ¡æ•°: {len(submission_data)}")
                
                print("\næ��äº¤æ–‡ä»¶é¢„è§ˆ:")
                sample_tasks = list(submission_data.keys())[:3]  # æ˜¾ç¤ºå‰�3ä¸ªä»»åŠ¡
                for task_id in sample_tasks:
                    print(f"ä»»åŠ¡ {task_id}: å·²ç”Ÿæˆ�é¢„æµ‹")
            except Exception as e:
                print(f"[ERROR] è¯»å�–æ��äº¤æ–‡ä»¶å¤±è´¥: {e}")
        
        # ä¹Ÿå°�è¯•åœ¨å¤šä¸ªå�¯èƒ½çš„ä½�ç½®åˆ›å»ºæ–‡ä»¶
        try:
            # åœ¨å·¥ä½œç›®å½•åˆ›å»º
            with open('submission.json', 'r') as f:
                submission_data = json.load(f)
            print(f"[OK] ç¡®è®¤submission.jsonæ–‡ä»¶å­˜åœ¨ä¸”å�¯è¯»å�–ï¼ŒåŒ…å�« {len(submission_data)} ä¸ªä»»åŠ¡")
        except Exception as e:
            print(f"[WARNING] è¯»å�–æ–‡ä»¶å¤±è´¥: {e}")
            
            # å·²ç»�åœ¨ä¸Šé�¢æ˜¾ç¤ºäº†æ��äº¤æ–‡ä»¶é¢„è§ˆ

    else:
        print("\nâœ— æ��äº¤æ–‡ä»¶ç”Ÿæˆ�å¤±è´¥")

if __name__ == "__main__":
    main()

