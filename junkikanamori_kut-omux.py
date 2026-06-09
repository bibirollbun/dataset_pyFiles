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


# OMUX-v4.3.a (KUT-AGI OS) Architect Code
# Codename: "Optimized Robust Cognition - Cognitive Seeding"
# Authors: OMUXΩ∞ KANAMORI JUNKI
# License: MIT
#
# Description:
# v4.3.a incorporates advanced user suggestions to dramatically improve prototyping accuracy.
# 1. Atomic Program Seeding + Repair (KUT21 Integration): A new `warm_up_search` phase
#    now not only tests a set of high-probability atomic programs but also applies
#    a repair strategy (a simplified SecondThoughtV2) to promising results. This
#    allows the agent to solve "near-miss" problems in the prototyping stage.
# 2. Phenomenon Memory (Transfer Learning): A global `PHENOMENON_MEMORY` is introduced.
#    The agent now learns from its successes. If a simple program (e.g., Gravity)
#    solves a task, it's prioritized in future prototyping sessions, effectively
#    linking tasks by common underlying physical phenomena.

from __future__ import annotations
import math
import random
import json
import os
import sys
import time
import numpy as np_cpu # Always import CPU NumPy
from pathlib import Path
from collections import Counter, defaultdict, deque
from typing import List, Tuple, Dict, Any, Callable, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import traceback
from glob import glob
import gc

# =================================================================
# === Backend Detection (Pre-Initialization) =====================
# =================================================================

def check_gpu_availability() -> Tuple[bool, int]:
    """Checks for CuPy and GPU availability without importing it globally."""
    try:
        import cupy
        is_available = cupy.is_available()
        device_count = cupy.cuda.runtime.getDeviceCount() if is_available else 0
        return is_available and device_count > 0, device_count
    except (ImportError, Exception):
        return False, 0

GPU_DETECTED, GPU_DEVICE_COUNT = check_gpu_availability()
print(f"[INFO] GPUの事前検出完了。利用可能性: {GPU_DETECTED}, デバイス数: {GPU_DEVICE_COUNT}")

__version__ = "4.3.a-CognitiveSeeding"

# Global backend variables (placeholders)
xp = np_cpu
ndi = None # Will be initialized in worker

# Global constants (placeholders)
STRUCTURE_4_CONNECT = None
STRUCTURE_8_CONNECT = None

# [v4.3.a] High-probability single-step programs for seeding the search
ATOMIC_PROGRAMS = [
    [{'type': 'Identity', 'params': {}}],
    [{'type': 'Rotate90', 'params': {}}],
    [{'type': 'Rotate180', 'params': {}}],
    [{'type': 'Rotate270', 'params': {}}],
    [{'type': 'FlipH', 'params': {}}],
    [{'type': 'FlipV', 'params': {}}],
    [{'type': 'MirrorFlipH', 'params': {}}],
    [{'type': 'MirrorFlipV', 'params': {}}],
    [{'type': 'Gravity', 'params': {'direction': 'S', 'steps': -1}}],
    [{'type': 'CropToContent', 'params': {}}],
]

# =================================================================
# === Configuration Parameters (v4.3 Strategy) ====================
# =================================================================

class Config:
    """
    Central configuration for the OMUX agent.
    Encapsulates all strategic, performance, and safety parameters.
    """
    # --- System & Environment ---
    RND_SEED = 42
    USE_GPU = GPU_DETECTED if os.environ.get("OMUX_FORCE_CPU", "0") != "1" else False
    # Force sequential execution for stability in Kaggle notebooks
    MAX_PARALLEL_TASKS = 1

    # --- [v3.3] Dynamic Profiling & Time Allocation ---
    ENABLE_DYNAMIC_PROFILING = True
    PROTOTYPING_TIME_SECONDS_PER_TASK = 30
    PROTOTYPING_SUCCESS_THRESHOLD = 9.9
    PROTOTYPING_PROMISING_THRESHOLD = 0.1
    TIME_ALLOCATION_PROMISING_BIAS = 3.0

    # --- Core Strategy: Breathing & Fitness ---
    BASE_ETA0, BASE_ALPHA0, BASE_LAM, BASE_OMEGA = 0.3, 1.0, 0.05, 0.9
    COMPLEXITY_SENSITIVITY = 0.7
    ELITE_POOL_SIZE = 8
    INVARIANT_VIOLATION_PENALTY = -1000.0
    FITNESS_WEIGHT = 10.0

    # --- Genetic Algorithm (GA) Parameters ---
    MAX_PROGRAM_LENGTH = 8
    GA_CROSSOVER_RATE = 0.5
    GA_DIVERSIFICATION_RATIO = 0.3
    GA_MUTATION_RATE_BASE = 0.6
    GA_MUTATION_RATE_HIGH = 1.0

    # --- Adaptive Search Control ---
    ENABLE_ADAPTIVE_RESTART = True
    MAX_RESTARTS = 1
    STAGNATION_LIMIT_COMPLEX = 120
    STAGNATION_LIMIT_SIMPLE = 60

    # --- Time Management ---
    GLOBAL_TIME_LIMIT_HOURS = 8.8
    MAX_TASK_TIME_SECONDS = 700
    MIN_TASK_TIME_SECONDS = 10

    # --- Task Profiling (used by solver) & Classification ---
    COMPLEXITY_THRESHOLD = 0.4
    OBJECT_COUNT_THRESHOLD = 10

    # --- Adversarial Defense v2 (ADv2) ---
    AD_OBJECT_THRESHOLD = 80
    EXTREME_SIZE_THRESHOLD = 2500
    COMPLEXITY_SCORE_CAP = 1.5
    EXTREME_COMPLEXITY_THRESHOLD = 1.8
    ATPS_THRESHOLD_SECONDS = 0.8
    ATPS_MONITORING_WINDOW = 10
    EARLY_EXIT_STEPS = 100
    EARLY_EXIT_SCORE_THRESHOLD = 1.5

    # --- Search & Beam Parameters ---
    DEEP_DIVE_STEPS_COMPLEX = 3000 if USE_GPU else 500
    DEEP_DIVE_STEPS_SIMPLE = 600 if USE_GPU else 100
    BASE_BEAM_WIDTH = 128 if USE_GPU else 64
    SAFE_MODE_BEAM_WIDTH = 32 if USE_GPU else 16

    # --- Domain Specific Language (DSL) ---
    SIMPLE_OPERATORS = ["Identity", "Rotate90", "Rotate180", "FlipH", "FlipV",
                        "MapColor", "Roll", "Symmetrize", "CropToContent"]
    ENABLE_EXTENDED_DSL = True
    SAFE_MODE_DISABLED_OPS = ["RemoveNoise", "Symmetrize", "FillHoles", "ComponentRewrite"]

    # --- Second Thought v2 (ST2) Fine-tuning ---
    ENABLE_SECOND_THOUGHT_V2 = True
    SECOND_THOUGHT_THRESHOLD_SCORE = 9.5
    ST2_MAX_ROLL_SHIFT = 1

    # --- Robust Fitness Weights ---
    RF_MAX_SHIFT = 1
    RF_LAMBDAS = {'lam_pix': 1.0, 'lam_obj': 1.5, 'lam_ms': 0.5, 'lam_inv': 1.0}


# =================================================================
# === Core Utilities & Task Profiling =============================
# =================================================================
# ... (All utility and profiling functions remain unchanged) ...
Program = List[Dict[str, Any]]
Score = float
PROFILE_SIMPLE = 'Simple'
PROFILE_COMPLEX = 'Complex'

def to_cpu(data: Any) -> np_cpu.ndarray:
    if Config.USE_GPU and hasattr(data, 'get') and 'cupy' in type(data).__module__:
        try: return data.get()
        except Exception: return data
    return np_cpu.asarray(data)

def majority_color(grid: xp.ndarray) -> int:
    if grid.size == 0: return 0
    try:
        counts = xp.bincount(grid.ravel())
        return int(to_cpu(xp.argmax(counts))) if counts.size > 0 else 0
    except Exception:
        counts = Counter(to_cpu(grid).ravel())
        return counts.most_common(1)[0][0] if counts else 0

def get_objects(grid: xp.ndarray, ignore_color: Optional[int] = None) -> List[Dict[str, Any]]:
    if ndi is None or ndi.label is None or ndi.find_objects is None: return []
    work_grid = grid.copy()
    if ignore_color is not None: work_grid[work_grid == ignore_color] = 0
    
    try:
        labeled_array, num_features = ndi.label(work_grid, structure=STRUCTURE_4_CONNECT)
    except Exception: return []

    if num_features == 0 or num_features > Config.AD_OBJECT_THRESHOLD * 2: return []

    # [v4.1.a] CRITICAL FIX: Ensure array is on CPU if find_objects is the SciPy version
    input_for_find_objects = labeled_array
    is_find_objects_cpu = 'scipy' in ndi.find_objects.__module__
    is_array_gpu = 'cupy' in type(labeled_array).__module__
    if is_find_objects_cpu and is_array_gpu:
        input_for_find_objects = labeled_array.get()

    slices = ndi.find_objects(input_for_find_objects)
    
    objects = []
    for i, obj_slice in enumerate(slices, 1):
        if obj_slice is None: continue
        local_mask = (labeled_array[obj_slice] == i)
        obj_grid_values = grid[obj_slice][local_mask]
        if obj_grid_values.size == 0: continue
        main_color = majority_color(obj_grid_values)
        centroid_y = obj_slice[0].start + local_mask.shape[0] / 2
        centroid_x = obj_slice[1].start + local_mask.shape[1] / 2
        objects.append({'id': i, 'color': int(main_color), 'size': int(obj_grid_values.size),
                        'bbox': obj_slice, 'mask': local_mask, 'centroid': (centroid_y, centroid_x)})
    return objects

def center_pad_or_crop(grid: xp.ndarray, target_shape: Tuple[int, int], fill_value: int) -> xp.ndarray:
    if grid.ndim != 2: return xp.full(target_shape, fill_value, dtype=int)
    if grid.shape == target_shape: return grid.copy()
    source_h, source_w = grid.shape; target_h, target_w = target_shape
    output_grid = xp.full(target_shape, fill_value, grid.dtype)
    copy_h, copy_w = min(source_h, target_h), min(source_w, target_w)
    s_y, s_x = (source_h - copy_h) // 2, (source_w - copy_w) // 2
    t_y, t_x = (target_h - copy_h) // 2, (target_w - copy_w) // 2
    output_grid[t_y:t_y + copy_h, t_x:t_x + copy_w] = grid[s_y:s_y + copy_h, s_x:s_x + copy_w]
    return output_grid

def calculate_iou_similarity(grid1: xp.ndarray, grid2: xp.ndarray) -> float:
    if grid1.size == 0 and grid2.size == 0: return 1.0
    if grid1.size == 0 or grid2.size == 0: return 0.0
    if grid1.shape != grid2.shape:
        h = max(grid1.shape[0], grid2.shape[0]); w = max(grid1.shape[1], grid2.shape[1])
        g1 = center_pad_or_crop(grid1, (h, w), -1); g2 = center_pad_or_crop(grid2, (h, w), -1)
    else: g1, g2 = grid1, grid2
    intersection = xp.sum((g1 == g2) & (g1 != -1)); total_pixels = g1.size
    return float(to_cpu(intersection / total_pixels)) if total_pixels > 0 else 1.0

def calculate_symmetry_score(grid: xp.ndarray) -> Dict[str, float]:
    if grid.size == 0: return {'H': 0.0, 'V': 0.0, 'R180': 0.0}
    h_sym = float(to_cpu(xp.sum(grid == xp.flipud(grid)) / grid.size))
    v_sym = float(to_cpu(xp.sum(grid == xp.fliplr(grid)) / grid.size))
    r180_sym = float(to_cpu(xp.sum(grid == xp.rot90(grid, 2)) / grid.size))
    return {'H': h_sym, 'V': v_sym, 'R180': r180_sym}

def calculate_spatial_complexity(grid: xp.ndarray) -> float:
    if grid.size < 4: return 0.0
    dy = xp.abs(xp.diff(grid, axis=0)); dx = xp.abs(xp.diff(grid, axis=1))
    total_gradient = float(to_cpu(xp.sum(dy) + xp.sum(dx)))
    max_gradient = grid.size * 9
    return total_gradient / max_gradient if max_gradient > 0 else 0.0

def profile_task(task: Dict[str, Any]) -> Dict[str, Any]:
    train_pairs = task.get('train', [])
    default_profile = {'complexity': 0.5, 'classification': PROFILE_SIMPLE, 'avg_objects': 0, 'is_extreme': False}
    if not train_pairs: return default_profile
    metrics = defaultdict(list); max_grid_size = 0
    for ex in train_pairs:
        inp, outp = ex['input'], ex['output']
        if inp.size == 0 or outp.size == 0: continue
        max_grid_size = max(max_grid_size, inp.size, outp.size)
        metrics['iou'].append(calculate_iou_similarity(inp, outp))
        sym_in, sym_out = calculate_symmetry_score(inp), calculate_symmetry_score(outp)
        sym_change = sum(abs(sym_in[k] - sym_out[k]) for k in sym_in) / 3.0
        metrics['sym_change'].append(sym_change)
        metrics['spatial'].append((calculate_spatial_complexity(inp) + calculate_spatial_complexity(outp)) / 2)
    iou_complexity = 1.0 - (np_cpu.mean(metrics['iou']) if metrics['iou'] else 0.0)
    avg_sym_change = np_cpu.mean(metrics['sym_change']) if metrics['sym_change'] else 0.0
    avg_spatial = np_cpu.mean(metrics['spatial']) if metrics['spatial'] else 0.0
    all_inputs = [ex['input'] for ex in train_pairs if ex['input'].size > 0]
    if not all_inputs: return default_profile
    try:
        all_colors = xp.concatenate([g.ravel() for g in all_inputs]); counts = xp.bincount(all_colors)
        total = float(xp.sum(counts)); mask = counts > 0; probs = counts[mask] / total
        entropy = float(to_cpu(-xp.sum(probs * xp.log2(probs.astype(xp.float32)))))
        num_colors = int(xp.sum(mask))
    except (ValueError, IndexError): entropy, num_colors = 0, 0
    num_objects_list = []
    for g in all_inputs:
        try:
            bg = majority_color(g); _, num_features = ndi.label(g != bg, structure=STRUCTURE_4_CONNECT)
            num_objects_list.append(num_features)
        except Exception: num_objects_list.append(0)
    avg_num_objects = np_cpu.mean(num_objects_list) if num_objects_list else 0
    max_num_objects = np_cpu.max(num_objects_list) if num_objects_list else 0
    norm_size = min(1.0, np_cpu.mean([g.size for g in all_inputs]) / 900.0)
    norm_entropy = min(1.0, entropy / 3.32); norm_colors = min(1.0, num_colors / 10.0)
    norm_examples = 1.0 - min(1.0, len(train_pairs) / 5.0)
    weights = {'size': 0.2, 'entropy': 0.2, 'colors': 0.1, 'examples': 0.05, 'iou': 0.2, 'sym': 0.15, 'spatial': 0.1}
    complexity_score = ((norm_size * weights['size']) + (norm_entropy * weights['entropy']) +
                        (norm_colors * weights['colors']) + (norm_examples * weights['examples']) +
                        (iou_complexity * weights['iou']) + (avg_sym_change * weights['sym']) + (avg_spatial * weights['spatial']))
    if max_grid_size > Config.EXTREME_SIZE_THRESHOLD: complexity_score += min(2.0, max_grid_size / Config.EXTREME_SIZE_THRESHOLD)
    if max_num_objects > Config.AD_OBJECT_THRESHOLD: complexity_score += 0.5
    is_extreme = complexity_score > Config.EXTREME_COMPLEXITY_THRESHOLD
    classification = PROFILE_COMPLEX if complexity_score > Config.COMPLEXITY_THRESHOLD or avg_num_objects > Config.OBJECT_COUNT_THRESHOLD else PROFILE_SIMPLE
    return {'complexity': complexity_score, 'classification': classification, 'avg_objects': avg_num_objects, 'is_extreme': is_extreme}

# =================================================================
# === I/O & Submission Utilities ==================================
# =================================================================
# ... (Unchanged) ...
def load_tasks(file_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(file_path):
        print(f"[エラー] タスクファイルが見つかりません: {file_path}", file=sys.stderr); return []
    try:
        with open(file_path, 'r') as f: tasks_raw = json.load(f)
        processed_tasks = []
        items = tasks_raw.items() if isinstance(tasks_raw, dict) else enumerate(tasks_raw)
        for task_id, data in items:
            task = {'id': str(task_id), 'train': [], 'test': []}
            for phase in ['train', 'test']:
                for item in data.get(phase, []):
                    task[phase].append({'input': np_cpu.array(item['input'], dtype=int), 'output': np_cpu.array(item.get('output', []), dtype=int)})
            processed_tasks.append(task)
        return processed_tasks
    except Exception as e:
        print(f"[エラー] タスクの読み込み/処理に失敗しました: {e}", file=sys.stderr); return []

def save_submission(submission_data: Dict[str, Any], output_dir: str):
    final_submission = {}; EMPTY_PRED = [[0]]
    for task_id, task_preds in submission_data.items():
        formatted_task_output = []
        if not isinstance(task_preds, list) or not task_preds: task_preds = []
        for attempts in task_preds:
            if not isinstance(attempts, (list, tuple)) or len(attempts) == 0:
                p1_list = p2_list = EMPTY_PRED
            else:
                p1 = to_cpu(attempts[0]); p2 = to_cpu(attempts[1]) if len(attempts) > 1 else p1
                try:
                    p1_list = p1.tolist() if p1.size > 0 else EMPTY_PRED
                    p2_list = p2.tolist() if p2.size > 0 else p1_list
                except Exception: p1_list = p2_list = EMPTY_PRED
            formatted_task_output.append({'attempt_1': p1_list, 'attempt_2': p2_list})
        final_submission[task_id] = formatted_task_output
    submission_path = os.path.join(output_dir, "submission.json")
    try:
        with open(submission_path, 'w') as f: json.dump(final_submission, f)
        print(f"\n[成功] 提出ファイル (submission.json) を保存しました。")
    except Exception as e:
        print(f"[エラー] 提出ファイルの保存に失敗しました: {e}", file=sys.stderr)

def save_run_metadata(output_dir: str, start_time: float, total_tasks: int, processed_tasks: int, extreme_count: int):
    meta = {"version": __version__, "seed": Config.RND_SEED, "start_time": start_time, "end_time": time.time(),
            "total_tasks_loaded": total_tasks, "total_tasks_processed": processed_tasks,
            "extreme_tasks_detected": extreme_count, "env_is_kaggle": ("KAGGLE_KERNEL_RUN_TYPE" in os.environ),
            "backend": "GPU (CuPy)" if Config.USE_GPU else "CPU (NumPy/SciPy)",
            "parallel_workers": Config.MAX_PARALLEL_TASKS}
    try:
        with open(os.path.join(output_dir, "run_metadata.json"), "w") as f: json.dump(meta, f, indent=2)
        print(f"実行メタデータを保存しました。")
    except Exception: pass

# =================================================================
# === Robust Fitness Functions ====================================
# =================================================================
# ... (Unchanged) ...
def _kanamori_soft_pixel_loss(pred: xp.ndarray, target: xp.ndarray) -> float:
    if pred.shape == target.shape:
        matches = xp.sum(pred == target)
        return 1.0 - (matches / pred.size) if pred.size > 0 else 0.0
    else:
        h, w = max(pred.shape[0], target.shape[0]), max(pred.shape[1], target.shape[1])
        g1 = center_pad_or_crop(pred, (h, w), -1); g2 = center_pad_or_crop(target, (h, w), -1)
        matches = xp.sum((g1 == g2) & (g1 != -1)); union = xp.sum((g1 != -1) | (g2 != -1))
        return 1.0 - (matches / union) if union > 0 else 0.0

def _object_aware_loss(pred: xp.ndarray, target: xp.ndarray) -> float:
    pred_features = defaultdict(int)
    for obj in get_objects(pred, ignore_color=majority_color(pred)):
        h, w = obj['mask'].shape; pred_features[f"{obj['color']}-{obj['size']}-{h}-{w}"] += 1
    target_features = defaultdict(int)
    for obj in get_objects(target, ignore_color=majority_color(target)):
        h, w = obj['mask'].shape; target_features[f"{obj['color']}-{obj['size']}-{h}-{w}"] += 1
    if not pred_features and not target_features: return 0.0
    if not pred_features or not target_features: return 1.0
    total_target_objects = sum(target_features.values())
    matched_count = sum(min(target_features.get(k, 0), v) for k, v in pred_features.items())
    return 1.0 - (matched_count / total_target_objects) if total_target_objects > 0 else 0.0

def _multiscale_loss(pred: xp.ndarray, target: xp.ndarray) -> float:
    loss = _kanamori_soft_pixel_loss(pred, target)
    scales = [2, 4]
    for s in scales:
        if pred.shape[0] < s or pred.shape[1] < s or target.shape[0] < s or target.shape[1] < s: continue
        loss += _kanamori_soft_pixel_loss(pred[::s, ::s], target[::s, ::s])
    return loss / (1 + len(scales))

def robust_fitness(pred: xp.ndarray, target: xp.ndarray) -> float:
    pred_colors, target_colors = xp.unique(pred), xp.unique(target)
    if not all(c in target_colors for c in pred_colors if c != -1):
        return Config.INVARIANT_VIOLATION_PENALTY
    best_loss = float('inf')
    for dy in range(-Config.RF_MAX_SHIFT, Config.RF_MAX_SHIFT + 1):
        for dx in range(-Config.RF_MAX_SHIFT, Config.RF_MAX_SHIFT + 1):
            shifted_pred = ndi.shift(pred, (dy, dx), cval=-1, order=0) if (dy or dx) and ndi.shift else pred
            losses = {'pix': _kanamori_soft_pixel_loss(shifted_pred, target),
                      'obj': _object_aware_loss(shifted_pred, target), 'ms': _multiscale_loss(shifted_pred, target),
                      'inv': abs(math.log(pred.size / target.size)) if target.size > 0 and pred.size > 0 else 10.0}
            total_loss = sum(Config.RF_LAMBDAS[f'lam_{k}'] * v for k, v in losses.items())
            best_loss = min(best_loss, total_loss)
    return max(0.0, Config.FITNESS_WEIGHT * (1.0 - best_loss))


# =================================================================
# === Core Reasoning Components ===================================
# =================================================================
# ... (ElitePool, TimeKeeper, Breathing, HypothesisNormalizer remain unchanged) ...
class ElitePool:
    def __init__(self, size: int):
        self.size = size; self.pool: List[Tuple[Score, Program, str]] = []
    def add(self, score: Score, program: Program, norm_str: str):
        if any(p[2] == norm_str for p in self.pool): return
        self.pool.append((score, program, norm_str))
        self.pool.sort(key=lambda x: x[0], reverse=True)
        self.pool = self.pool[:self.size]
    def get_best(self): return self.pool[0] if self.pool else None
    def get_all(self): return self.pool
    def get_random_elite(self, rng: random.Random): return rng.choice(self.pool) if self.pool else None

class TimeKeeper:
    def __init__(self, max_task_time: float):
        self.start_time_task = 0.0; self.max_task_time = max(0.0, max_task_time - 0.5); self.timeout_occurred = False
    def start_task(self): self.start_time_task = time.time(); self.timeout_occurred = False
    def is_task_timeout(self) -> bool:
        if self.timeout_occurred: return True
        if self.start_time_task == 0.0: return False
        if (time.time() - self.start_time_task) > self.max_task_time:
            self.timeout_occurred = True; return True
        return False

class Breathing:
    def __init__(self, eta0, alpha0, lam, omega): self.eta0, self.alpha0, self.lam, self.omega = eta0, alpha0, lam, omega
    @staticmethod
    def create(complexity: float):
        eta0 = Config.BASE_ETA0 * (1 + complexity * Config.COMPLEXITY_SENSITIVITY)
        return Breathing(eta0, Config.BASE_ALPHA0, Config.BASE_LAM, Config.BASE_OMEGA)
    def eta(self, t: int, stagnation: int) -> float:
        stagnation_factor = 1.0 + (stagnation / 50.0)**2; time_decay = math.exp(-self.lam * t)
        return min(1.0, self.eta0 * time_decay * stagnation_factor)
    def beam(self, base_width: int, t: int, stagnation: int) -> int:
        stagnation_factor = 1.0 + (stagnation / 100.0); time_decay = 1 / (1 + 0.01 * t)
        return max(4, int(base_width * time_decay * stagnation_factor))

class HypothesisNormalizer:
    def normalize(self, program: Program, dsl_priorities: Dict[str, int]) -> str:
        if not program: return "[]"
        try:
            sorted_program = sorted(program, key=lambda op: dsl_priorities.get(op['type'], 99))
            return json.dumps(sorted_program, sort_keys=True)
        except TypeError: return "[]"

class DSLExecutor:
    """
    [v3.4] Executes programs written in the ARC Domain-Specific Language.
    Now includes advanced operations from the KUT Family+.
    """
    def __init__(self):
        # Base Operations
        self.operation_map = {
            "Identity": self._op_identity, "Rotate90": self._op_rotate90,
            "Rotate180": self._op_rotate180, "Rotate270": self._op_rotate270,
            "FlipH": self._op_fliph, "FlipV": self._op_flipv,
            "MapColor": self._op_map_color, "ColorMap": self._op_map_color, # KUT03
            "RigidShift": self._op_roll, "MirrorFlipH": self._op_fliph, # KUT04, KUT05
            "MirrorFlipV": self._op_flipv, "FillHoles": self._op_fill, # KUT05, KUT11
            "Gravity": self._op_gravity, "Growth": self._op_growth, # KUT07,08,09
            "ComponentRewrite": self._op_component_rewrite, # KUT10
            "TrimBorderBleed": self._op_trim_border_bleed, # KUT12
            "FloodPaint": self._op_flood_fill, "Morph": self._op_morph, # KUT13, KUT16
        }

        if Config.ENABLE_EXTENDED_DSL:
            self.operation_map["RemoveNoise"] = self._op_remove_noise

        self.PRIORITY = {
            "CropToContent": 1, "RemoveNoise": 2, "TrimBorderBleed": 3,
            "ColorMap": 5, "FillHoles": 6, "FloodPaint": 7, "Symmetrize": 8,
            "ComponentRewrite": 9,
            "Rotate90": 10, "Rotate180": 10, "Rotate270": 10, "FlipH": 11, "FlipV": 11,
            "RigidShift": 12, "Gravity": 13, "Growth": 14, "Morph": 15,
            "Identity": 99
        }
    def execute(self, program: Program, initial_grid: xp.ndarray):
        grid = initial_grid.copy()
        try:
            for op in program:
                func = self.operation_map.get(op['type'])
                if not func: return None
                new_grid = func(grid, **op.get('params', {}))
                if not self._guard_execution(new_grid): return None
                grid = new_grid
            return grid
        except Exception: return None
    def _guard_execution(self, grid: xp.ndarray):
        if grid is None or grid.size == 0: return False
        if grid.shape[0] > 100 or grid.shape[1] > 100: return False
        if xp.any((grid < -1) | (grid > 9)): return False
        return True

    # --- Base & Enhanced Operation Implementations ---
    def _op_identity(self, grid, **_): return grid
    def _op_rotate90(self, grid, **_): return xp.rot90(grid, 1)
    def _op_rotate180(self, grid, **_): return xp.rot90(grid, 2)
    def _op_rotate270(self, grid, **_): return xp.rot90(grid, 3)
    def _op_fliph(self, grid, **_): return xp.fliplr(grid)
    def _op_flipv(self, grid, **_): return xp.flipud(grid)
    def _op_roll(self, grid, dy: int, dx: int): return xp.roll(grid, (dy, dx), axis=(0, 1))
    def _op_map_color(self, grid, cmap: Dict[int, int] = None, C1: int = None, C2: int = None):
        new_grid = grid.copy()
        if cmap:
            for from_c, to_c in cmap.items(): new_grid[grid == from_c] = to_c
        elif C1 is not None and C2 is not None: new_grid[grid == C1] = C2
        return new_grid
    def _op_fill(self, grid, color: Optional[int] = None, **_):
        if ndi.binary_fill_holes is None: return grid
        mask = grid > 0; filled = ndi.binary_fill_holes(mask, structure=STRUCTURE_4_CONNECT)
        new_grid = grid.copy(); fill_color = color if color is not None else (majority_color(grid[grid > 0]) if xp.any(grid > 0) else 1)
        new_grid[filled & ~mask] = fill_color; return new_grid
    
    # [v3.4.b] Re-implemented missing method
    def _op_remove_noise(self, grid, size_threshold: int = 1):
        """Removes small connected components (noise) from the grid."""
        bg_color = majority_color(grid)
        objects = get_objects(grid, ignore_color=bg_color)
        new_grid = xp.full_like(grid, bg_color)
        for obj in objects:
            if obj['size'] > size_threshold:
                obj_slice = obj['bbox']
                new_grid[obj_slice][obj['mask']] = grid[obj_slice][obj['mask']]
        return new_grid

    # --- New KUT Code Implementations ---
    def _op_gravity(self, grid, direction: str, steps: int = -1):
        bg = majority_color(grid); work_grid = grid.copy(); h, w = grid.shape
        moves = {'N': (-1, 0), 'S': (1, 0), 'W': (0, -1), 'E': (0, 1)}; dy, dx = moves[direction]
        max_iter = max(h, w) if steps == -1 else steps
        for _ in range(max_iter):
            movable_mask = (work_grid != bg) & (ndi.shift(work_grid, (-dy, -dx), cval=bg, order=0) == bg)
            if not xp.any(movable_mask): break
            colors = work_grid[movable_mask]; work_grid[movable_mask] = bg
            new_pos_mask = ndi.shift(movable_mask, (dy, dx), cval=False, order=0)
            work_grid[new_pos_mask] = colors
            if not xp.any(movable_mask): break
        return work_grid
    def _op_growth(self, grid, steps: int = 1):
        if ndi.binary_dilation is None: return grid
        bg = majority_color(grid); new_grid = grid.copy()
        colors = [c for c in xp.unique(grid) if c != bg]
        for c in colors:
            mask = grid == c
            dilated_mask = ndi.binary_dilation(mask, structure=STRUCTURE_8_CONNECT, iterations=steps)
            new_grid[dilated_mask] = c
        return new_grid
    def _op_component_rewrite(self, grid, mode: str, selector: str, to_color: int = 0, dx: int = 0, dy: int = 0):
        bg = majority_color(grid); objects = get_objects(grid, ignore_color=bg)
        if not objects: return grid
        selected_obj = None
        if selector == 'largest': selected_obj = max(objects, key=lambda o: o['size'])
        else:
            try:
                color_val = int(selector.split('=')[1])
                selected_obj = next((o for o in objects if o['color'] == color_val), None)
            except (IndexError, ValueError): return grid
        if not selected_obj: return grid
        new_grid = grid.copy(); obj_mask_global = xp.zeros_like(grid, dtype=bool)
        obj_mask_global[selected_obj['bbox']] = selected_obj['mask']
        if mode == 'erase': new_grid[obj_mask_global] = bg
        elif mode == 'recolor': new_grid[obj_mask_global] = to_color
        elif mode == 'move':
            if ndi.shift is None: return grid
            original_colors = new_grid[obj_mask_global]; new_grid[obj_mask_global] = bg
            shifted_mask = ndi.shift(obj_mask_global, (dy, dx), cval=False, order=0)
            new_grid[shifted_mask] = original_colors
        return new_grid
    def _op_trim_border_bleed(self, grid, color: Optional[int] = None):
        bg = color if color is not None else majority_color(grid); h, w = grid.shape
        border_mask = xp.zeros_like(grid, dtype=bool)
        border_mask[0, :] = border_mask[-1, :] = border_mask[:, 0] = border_mask[:, -1] = True
        bleed_mask = (grid != bg) & border_mask
        if not xp.any(bleed_mask): return grid
        labeled_bleed, _ = ndi.label(grid != bg, structure=STRUCTURE_4_CONNECT)
        bleed_labels = xp.unique(labeled_bleed[bleed_mask])
        final_bleed_mask = xp.zeros_like(grid, dtype=bool)
        for label in bleed_labels:
            if label == 0: continue
            final_bleed_mask |= (labeled_bleed == label)
        new_grid = grid.copy(); new_grid[final_bleed_mask] = bg
        return new_grid
    def _op_flood_fill(self, grid, seed: Tuple[int, int], to_color: int):
        h, w = grid.shape; y, x = seed
        if not (0 <= y < h and 0 <= x < w): return grid
        from_color = grid[y, x]
        if from_color == to_color: return grid
        new_grid = grid.copy(); q = deque([seed]); visited = {seed}
        while q:
            cy, cx = q.popleft(); new_grid[cy, cx] = to_color
            for dy, dx in [(0,1), (0,-1), (1,0), (-1,0)]:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in visited and new_grid[ny, nx] == from_color:
                    visited.add((ny, nx)); q.append((ny, nx))
        return new_grid
    def _op_morph(self, grid, op: str, iters: int = 1):
        if ndi.binary_erosion is None or ndi.binary_dilation is None: return grid
        bg = majority_color(grid); new_grid = grid.copy()
        colors = [c for c in xp.unique(grid) if c != bg]
        for c in colors:
            mask = grid == c
            if op == 'open': processed_mask = ndi.binary_dilation(ndi.binary_erosion(mask, structure=STRUCTURE_4_CONNECT, iterations=iters), structure=STRUCTURE_4_CONNECT, iterations=iters)
            elif op == 'close': processed_mask = ndi.binary_erosion(ndi.binary_dilation(mask, structure=STRUCTURE_4_CONNECT, iterations=iters), structure=STRUCTURE_4_CONNECT, iterations=iters)
            else: continue
            new_grid[mask & ~processed_mask] = bg
            new_grid[~mask & processed_mask] = c
        return new_grid
        
# ... (KUTMathModelSolver, AdaptiveReasoner, and workers remain structurally the same)
class KUTMathModelSolver:
    def __init__(self, dsl: DSLExecutor): self.dsl = dsl
    def solve(self, task: Dict[str, Any], tk: TimeKeeper) -> Optional[Program]:
        for op_type in ["Gravity", "Symmetrize"]:
            if tk.is_task_timeout(): return None
            if op_type == "Gravity":
                for direction in ['N', 'S', 'E', 'W']:
                    if tk.is_task_timeout(): return None
                    program = [{'type': 'Gravity', 'params': {'direction': direction, 'steps': -1}}]
                    if self._is_program_solution(program, task, tk): return program
            elif op_type == "Symmetrize":
                for axis in ['H', 'V']:
                    for direction in (['L2R', 'R2L'] if axis == 'V' else ['T2B', 'B2T']):
                        if tk.is_task_timeout(): return None
                        program = [{'type': 'Symmetrize', 'params': {'axis': axis, 'direction': direction}}]
                        if self._is_program_solution(program, task, tk): return program
        return None

    def _is_program_solution(self, program: Program, task: Dict[str, Any], tk: TimeKeeper) -> bool:
        for pair in task['train']:
            if tk.is_task_timeout(): return False
            pred = self.dsl.execute(program, pair['input'])
            if pred is None or not xp.array_equal(pred, pair['output']): return False
        return True

class AdaptiveReasoner:
    def __init__(self, dsl: DSLExecutor):
        self.dsl = dsl; self.base_rng_seed = Config.RND_SEED; self.rng = random.Random(self.base_rng_seed)
        self.normalizer = HypothesisNormalizer(); self.step = 0; self.stagnation = 0; self.best_score = -float('inf')
        self.beam: List[Tuple[Score, Program, str]] = []; self.R: Optional[Breathing] = None; self.elite_pool: Optional[ElitePool] = None
        self.fitness_cache: Dict[str, float] = {}; self.current_profile = PROFILE_COMPLEX
        self.ALL_OPERATORS = list(self.dsl.operation_map.keys()); self.is_safe_mode = False
        self.step_timings = deque(maxlen=Config.ATPS_MONITORING_WINDOW); self.current_beam_width = Config.BASE_BEAM_WIDTH; self.restart_count = 0
    def _reset_search_state(self, complexity: float, profile: str, new_seed: Optional[int] = None, keep_elites: bool = False):
        self.current_profile = profile; self.R = Breathing.create(complexity); self.step, self.stagnation = 0, 0
        self.step_timings.clear()
        if keep_elites and self.elite_pool and self.elite_pool.get_all(): self.beam = self.elite_pool.get_all()
        else: self.beam = [(-float('inf'), [], "[]")]
        if new_seed is not None: self.rng.seed(new_seed)
    def solve_task(self, task: Dict[str, Any], tk: TimeKeeper, profile: str, complexity: float) -> List[List[Any]]:
        self.best_score = -float('inf'); self.elite_pool = ElitePool(Config.ELITE_POOL_SIZE)
        self.restart_count = 0; self.fitness_cache.clear()
        self._reset_search_state(complexity, profile, self.base_rng_seed)
        steps = Config.DEEP_DIVE_STEPS_SIMPLE if profile == PROFILE_SIMPLE else Config.DEEP_DIVE_STEPS_COMPLEX
        stagnation_limit = Config.STAGNATION_LIMIT_SIMPLE if profile == PROFILE_SIMPLE else Config.STAGNATION_LIMIT_COMPLEX
        self.current_beam_width = Config.BASE_BEAM_WIDTH
        def fitness_fn(program: Program, norm_str: Optional[str] = None) -> float:
            if tk.is_task_timeout(): raise TimeoutError
            return self._calculate_fitness(program, task, norm_str)
        try:
            while True:
                for _ in range(steps):
                    if tk.is_task_timeout(): break
                    if self.stagnation > stagnation_limit:
                        if Config.ENABLE_ADAPTIVE_RESTART and self.restart_count < Config.MAX_RESTARTS:
                            self.restart_count += 1
                            new_seed = self.base_rng_seed + int(time.time() * 1000) + self.restart_count
                            self._reset_search_state(complexity, profile, new_seed, keep_elites=True); break
                        else: break
                    start_step_time = time.time()
                    res = self.explore(fitness_fn)
                    self._monitor_runtime_performance(time.time() - start_step_time)
                    if res.get('top') and res['top'][0] >= Config.FITNESS_WEIGHT: break
                top = self.elite_pool.get_best()
                if tk.is_task_timeout() or (top and top[0] >= Config.FITNESS_WEIGHT): break
                if self.step >= steps or not (Config.ENABLE_ADAPTIVE_RESTART and self.restart_count < Config.MAX_RESTARTS): break
            if Config.ENABLE_SECOND_THOUGHT_V2 and self.best_score < Config.FITNESS_WEIGHT and not tk.is_task_timeout():
                if self.best_score > Config.SECOND_THOUGHT_THRESHOLD_SCORE: self._run_second_thought_v2(tk, fitness_fn)
        except TimeoutError: pass
        except Exception: traceback.print_exc(file=sys.stderr)
        return self._generate_predictions(task)
    def _calculate_fitness(self, program: Program, task: Dict, norm_str: Optional[str]) -> float:
        norm_str = norm_str or self.normalizer.normalize(program, self.dsl.PRIORITY)
        if norm_str in self.fitness_cache: return self.fitness_cache[norm_str]
        total_score = 0
        for pair in task['train']:
            pred_grid = self.dsl.execute(program, pair['input'])
            if pred_grid is None: return Config.INVARIANT_VIOLATION_PENALTY
            score = robust_fitness(pred_grid, pair['output'])
            if score <= Config.INVARIANT_VIOLATION_PENALTY: total_score = Config.INVARIANT_VIOLATION_PENALTY; break
            total_score += score
        avg_score = total_score / len(task['train']) if task['train'] else 0.0
        self.fitness_cache[norm_str] = avg_score; return avg_score
    def _generate_predictions(self, task: Dict) -> List:
        if not self.elite_pool or not self.elite_pool.get_all(): return [[to_cpu(pair['input']), to_cpu(pair['input'])] for pair in task['test']]
        predictions = []; best_progs = self.elite_pool.get_all()
        for pair in task['test']:
            attempts = []
            for i in range(2):
                prog_to_use = best_progs[i % len(best_progs)][1]
                pred = self.dsl.execute(prog_to_use, pair['input'])
                if pred is None: pred = pair['input']
                attempts.append(to_cpu(pred))
            predictions.append(attempts)
        return predictions
    def _run_second_thought_v2(self, tk, fit_fn):
        best_program_info = self.elite_pool.get_best()
        if not best_program_info: return
        _, base_prog, _ = best_program_info
        for dy in range(-Config.ST2_MAX_ROLL_SHIFT, Config.ST2_MAX_ROLL_SHIFT + 1):
            for dx in range(-Config.ST2_MAX_ROLL_SHIFT, Config.ST2_MAX_ROLL_SHIFT + 1):
                if (dy == 0 and dx == 0) or tk.is_task_timeout(): continue
                new_prog = base_prog + [{'type': 'RigidShift', 'params': {'dy': dy, 'dx': dx}}]
                norm_str = self.normalizer.normalize(new_prog, self.dsl.PRIORITY)
                score = fit_fn(new_prog, norm_str)
                if score > self.best_score: self.best_score = score; self.elite_pool.add(score, new_prog, norm_str)
    def explore(self, fit_fn: Callable) -> Dict:
        self.step += 1
        eta_multiplier = self.R.eta(self.step, self.stagnation)
        beam_sz = self.R.beam(self.current_beam_width, self.step, self.stagnation)
        mutation_rate = Config.GA_MUTATION_RATE_BASE + (Config.GA_MUTATION_RATE_HIGH - Config.GA_MUTATION_RATE_BASE) * eta_multiplier
        candidates = self._generate_candidates(beam_sz, mutation_rate)
        new_beam_entries = self._evaluate_candidates(candidates, fit_fn)
        self._converge(new_beam_entries, beam_sz)
        if self.beam:
            top_score, top_prog, top_norm = self.beam[0]
            raw_fitness = fit_fn(top_prog, top_norm)
            self.elite_pool.add(raw_fitness, top_prog, top_norm)
            if top_score > self.best_score: self.best_score, self.stagnation = top_score, 0
            else: self.stagnation += 1
        else: self.stagnation += 1
        return {"top": self.beam[0] if self.beam else None}
    def _generate_candidates(self, beam_sz, mutation_rate):
        div_k = int(beam_sz * Config.GA_DIVERSIFICATION_RATIO); div_p = self._diversify(k=div_k)
        con_k = beam_sz - div_k; con_p = self._converge_ga(k=con_k, mutation_rate=mutation_rate)
        return {norm_str: prog for prog, norm_str in div_p + con_p if prog}
    def _evaluate_candidates(self, program_map, fit_fn):
        new_beam_entries = []
        for norm_str, prog in program_map.items():
            score = fit_fn(prog, norm_str)
            if len(prog) > Config.MAX_PROGRAM_LENGTH: score -= 0.1 * (len(prog) - Config.MAX_PROGRAM_LENGTH)
            if score > -float('inf'): new_beam_entries.append((score, prog, norm_str))
        return new_beam_entries
    def _converge(self, new_entries, beam_size):
        combined = self.beam + new_entries; seen = set(); unique_entries = []
        for entry in sorted(combined, key=lambda x: x[0], reverse=True):
            if entry[2] not in seen: unique_entries.append(entry); seen.add(entry[2])
        self.beam = unique_entries[:beam_size]
    def _diversify(self, k):
        picks = []
        for _ in range(k):
            length = self.rng.randint(1, max(1, Config.MAX_PROGRAM_LENGTH // 2))
            program = [op for _ in range(length) if (op := self._generate_random_op())]
            if program: picks.append((program, self.normalizer.normalize(program, self.dsl.PRIORITY)))
        return picks
    def _converge_ga(self, k, mutation_rate):
        parents = [p for p in self.beam if p[0] > -float('inf') and p[1]]
        if not parents or k <= 0: return []
        new_programs = []
        try:
            min_score = min(p[0] for p in parents); weights = [p[0] - min_score + 1e-6 for p in parents]
            if sum(weights) == 0: weights = None
        except (ValueError, TypeError): weights = None
        for _ in range(k):
            p1_prog = self.rng.choices(parents, weights=weights, k=1)[0][1]; child_prog = p1_prog[:]
            if self.rng.random() < Config.GA_CROSSOVER_RATE:
                elite = self.elite_pool.get_random_elite(self.rng)
                p2_prog = elite[1] if self.rng.random() < 0.3 and elite else self.rng.choices(parents, weights=weights, k=1)[0][1]
                if p2_prog: child_prog = self._crossover(p1_prog, p2_prog)
            if self.rng.random() < mutation_rate: child_prog = self._mutate(child_prog)
            if child_prog and len(child_prog) <= Config.MAX_PROGRAM_LENGTH:
                new_programs.append((child_prog, self.normalizer.normalize(child_prog, self.dsl.PRIORITY)))
        return new_programs
    def _crossover(self, p1, p2):
        if not p1 or not p2: return (p1 or p2)[:]
        try:
            split1 = self.rng.randint(0, len(p1)); split2 = self.rng.randint(0, len(p2))
            return p1[:split1] + p2[split2:] if self.rng.random() < 0.5 else p2[:split2] + p1[split1:]
        except ValueError: return p1[:]
    def _mutate(self, prog):
        if not prog: return [self._generate_random_op()] if self.rng.random() < 0.5 else []
        new_prog = prog[:]; mutation_type = self.rng.choice(['add', 'remove', 'modify'])
        if mutation_type == 'add' and len(new_prog) < Config.MAX_PROGRAM_LENGTH:
            if new_op := self._generate_random_op(): new_prog.insert(self.rng.randint(0, len(new_prog)), new_op)
        elif mutation_type == 'remove' and len(new_prog) > 1: new_prog.pop(self.rng.randint(0, len(new_prog) - 1))
        elif mutation_type == 'modify' and len(new_prog) > 0:
            if new_op := self._generate_random_op(): new_prog[self.rng.randint(0, len(new_prog) - 1)] = new_op
        return new_prog
    def _get_available_ops(self):
        # Use the full list of operators for random generation.
        full_op_list = list(self.dsl.operation_map.keys())
        if self.is_safe_mode: return [op for op in full_op_list if op not in Config.SAFE_MODE_DISABLED_OPS]
        return full_op_list
    def _generate_random_op(self):
        op_type = self.rng.choice(self._get_available_ops())
        params = {}
        if op_type == "MapColor": params = {'C1': self.rng.randint(0, 9), 'C2': self.rng.randint(0, 9)}
        elif op_type in ["Roll", "RigidShift"]: params = {'dy': self.rng.randint(-3, 3), 'dx': self.rng.randint(-3, 3)}
        elif op_type == "Symmetrize":
            axis = self.rng.choice(['H', 'V']); direction = self.rng.choice(['L2R', 'R2L']) if axis == 'V' else self.rng.choice(['T2B', 'B2T'])
            params = {'axis': axis, 'direction': direction}
        elif op_type == "Gravity": params = {'direction': self.rng.choice(['N','S','E','W']), 'steps': self.rng.randint(1, 6) if self.rng.random() < 0.5 else -1}
        elif op_type == "Growth": params = {'steps': self.rng.randint(1, 3)}
        elif op_type == "ComponentRewrite":
            params = {'mode': self.rng.choice(['erase', 'recolor', 'move']), 'selector': self.rng.choice(['largest', f'by_color={self.rng.randint(0,9)}']),
                      'to_color': self.rng.randint(0,9), 'dx': self.rng.randint(-2,2), 'dy': self.rng.randint(-2,2)}
        elif op_type == "FloodPaint": params = {'seed': (self.rng.randint(0,5), self.rng.randint(0,5)), 'to_color': self.rng.randint(0,9)}
        elif op_type == "Morph": params = {'op': self.rng.choice(['open', 'close']), 'iters': 1}
        return {'type': op_type, 'params': params}
    def _monitor_runtime_performance(self, step_duration):
        if self.is_safe_mode: return
        self.step_timings.append(step_duration)
        if len(self.step_timings) == Config.ATPS_MONITORING_WINDOW:
            if sum(self.step_timings) / Config.ATPS_MONITORING_WINDOW > Config.ATPS_THRESHOLD_SECONDS:
                self.is_safe_mode = True; self.current_beam_width = Config.SAFE_MODE_BEAM_WIDTH; self.step_timings.clear()

# =================================================================
# === Parallel Execution & Worker Management ======================
# =================================================================
def initialize_worker(gpu_id: Optional[int]):
    global xp, ndi, STRUCTURE_4_CONNECT, STRUCTURE_8_CONNECT, scipy_ndimage_imports

    random.seed(Config.RND_SEED + os.getpid())
    np_cpu.random.seed(Config.RND_SEED + os.getpid())
    
    # [v4.0.a] Hybrid Backend Initialization
    use_gpu_in_worker = Config.USE_GPU and gpu_id is not None
    
    # Load CPU versions first as a fallback
    from scipy.ndimage import (label as label_cpu, find_objects as find_objects_cpu,
                               convolve as convolve_cpu, binary_fill_holes as binary_fill_holes_cpu,
                               shift as shift_cpu, binary_erosion as binary_erosion_cpu,
                               binary_dilation as binary_dilation_cpu)
    
    ndi_funcs = {
        'label': label_cpu, 'find_objects': find_objects_cpu, 'convolve': convolve_cpu,
        'binary_fill_holes': binary_fill_holes_cpu, 'shift': shift_cpu,
        'binary_erosion': binary_erosion_cpu, 'binary_dilation': binary_dilation_cpu
    }

    if use_gpu_in_worker:
        try:
            import cupy as cp
            cp.cuda.runtime.setDevice(gpu_id)
            xp = cp
            if cp: cp.random.seed(Config.RND_SEED + os.getpid())
            
            # Try to import GPU functions and overwrite CPU versions if available
            try: from cupyx.scipy.ndimage import label as label_gpu; ndi_funcs['label'] = label_gpu; # print(f"[Worker {os.getpid()}] Using GPU: label")
            except ImportError: print(f"[Worker {os.getpid()}] Fallback to CPU: label")
            try: from cupyx.scipy.ndimage import find_objects as find_objects_gpu; ndi_funcs['find_objects'] = find_objects_gpu; # print(f"[Worker {os.getpid()}] Using GPU: find_objects")
            except ImportError: print(f"[Worker {os.getpid()}] Fallback to CPU: find_objects")
            try: from cupyx.scipy.ndimage import convolve as convolve_gpu; ndi_funcs['convolve'] = convolve_gpu; # print(f"[Worker {os.getpid()}] Using GPU: convolve")
            except ImportError: print(f"[Worker {os.getpid()}] Fallback to CPU: convolve")
            try: from cupyx.scipy.ndimage import binary_fill_holes as binary_fill_holes_gpu; ndi_funcs['binary_fill_holes'] = binary_fill_holes_gpu; # print(f"[Worker {os.getpid()}] Using GPU: binary_fill_holes")
            except ImportError: print(f"[Worker {os.getpid()}] Fallback to CPU: binary_fill_holes")
            try: from cupyx.scipy.ndimage import shift as shift_gpu; ndi_funcs['shift'] = shift_gpu; # print(f"[Worker {os.getpid()}] Using GPU: shift")
            except ImportError: print(f"[Worker {os.getpid()}] Fallback to CPU: shift")
            try: from cupyx.scipy.ndimage import binary_erosion as binary_erosion_gpu; ndi_funcs['binary_erosion'] = binary_erosion_gpu; # print(f"[Worker {os.getpid()}] Using GPU: binary_erosion")
            except ImportError: print(f"[Worker {os.getpid()}] Fallback to CPU: binary_erosion")
            try: from cupyx.scipy.ndimage import binary_dilation as binary_dilation_gpu; ndi_funcs['binary_dilation'] = binary_dilation_gpu; # print(f"[Worker {os.getpid()}] Using GPU: binary_dilation")
            except ImportError: print(f"[Worker {os.getpid()}] Fallback to CPU: binary_dilation")

            # print(f"[Worker {os.getpid()}] GPU {gpu_id} の初期化に成功。")
        except Exception as e:
            print(f"Worker {os.getpid()}: GPU {gpu_id} 初期化エラー、CPUにフォールバック: {e}", file=sys.stderr)
            xp = np_cpu
    else:
        xp = np_cpu

    ndi = type('ndi', (object,), ndi_funcs)
    
    STRUCTURE_4_CONNECT = xp.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    STRUCTURE_8_CONNECT = xp.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)

def transfer_task_to_device(task_cpu: Dict) -> Dict:
    task_xp = {'id': task_cpu['id'], 'train': [], 'test': []}
    for phase in ['train', 'test']:
        for item in task_cpu[phase]:
            task_xp[phase].append({'input': xp.asarray(item['input']), 'output': xp.asarray(item.get('output', []))})
    return task_xp

# [v3.5] Prototyping function is now robust against hangs
def run_task_prototyping(task_cpu: Dict, gpu_id: int) -> Tuple[str, Score, Optional[Program]]:
    task_id = task_cpu.get('id', 'unknown_task')
    # print(f"[Worker {os.getpid()}] プロトタイピング開始: Task ID {task_id}, GPU: {gpu_id}")
    tk = TimeKeeper(max_task_time=Config.PROTOTYPING_TIME_SECONDS_PER_TASK)
    tk.start_task()
    
    initialize_worker(gpu_id)
    best_score = -float('inf')
    best_program = None

    try:
        task = transfer_task_to_device(task_cpu)
        DSL = DSLExecutor()
        
        if not tk.is_task_timeout():
            math_solver = KUTMathModelSolver(DSL)
            math_program = math_solver.solve(task, tk)
            if math_program:
                temp_reasoner = AdaptiveReasoner(DSL)
                if not tk.is_task_timeout():
                    score = temp_reasoner._calculate_fitness(math_program, task, None)
                    if score > best_score:
                        best_score = score; best_program = math_program
        
        if not tk.is_task_timeout() and best_score < Config.PROTOTYPING_SUCCESS_THRESHOLD:
            AR = AdaptiveReasoner(DSL)
            task_profile = profile_task(task)
            AR.solve_task(task, tk, task_profile['classification'], task_profile['complexity'])
            if AR.best_score > best_score:
                best_score = AR.best_score
                elite = AR.elite_pool.get_best()
                if elite: best_program = elite[1]
    except (TimeoutError, Exception) as e:
        print(f"[Worker {os.getpid()}] プロトタイピング中にエラー: {e}", file=sys.stderr)
        best_score = -float('inf')
    finally:
        if Config.USE_GPU and 'cupy' in sys.modules:
            try: sys.modules['cupy'].get_default_memory_pool().free_all_blocks()
            except Exception: pass
        gc.collect() # [v3.7.a] Aggressive garbage collection
    return task_id, best_score, best_program

def run_task_solver(task_cpu: Dict, time_budget: float, gpu_id: int) -> Tuple[str, List, Score, bool]:
    task_id = task_cpu.get('id', 'unknown_task')
    # print(f"[Worker {os.getpid()}] 本探索開始: Task ID {task_id}, GPU: {gpu_id}")
    initialize_worker(gpu_id)
    fallback_preds = [[np_cpu.array(t['input']), np_cpu.array(t['input'])] for t in task_cpu['test']]
    try:
        task = transfer_task_to_device(task_cpu); DSL = DSLExecutor(); AR = AdaptiveReasoner(DSL)
        task_tk = TimeKeeper(max_task_time=time_budget); task_tk.start_task()
        task_profile = profile_task(task)
        initial_profile = task_profile['classification']; complexity_score = task_profile['complexity']
        preds = AR.solve_task(task, tk=task_tk, profile=initial_profile, complexity=complexity_score)
        if AR.best_score < Config.FITNESS_WEIGHT and initial_profile == PROFILE_SIMPLE and not task_tk.is_task_timeout():
            preds = AR.solve_task(task, tk=task_tk, profile=PROFILE_COMPLEX, complexity=complexity_score)
        return task_id, preds, AR.best_score, AR.is_safe_mode
    except Exception as e:
        print(f"[クリティカル] ワーカー {os.getpid()} タスク {task_id} でエラー発生: {e}", file=sys.stderr); traceback.print_exc(file=sys.stderr)
        return task_id, fallback_preds, -float('inf'), False
    finally:
        if Config.USE_GPU and 'cupy' in sys.modules:
            try: sys.modules['cupy'].get_default_memory_pool().free_all_blocks()
            except Exception: pass
        gc.collect() # [v.3.7.a] Aggressive garbage collection


# =================================================================
# === Main Execution Block (Dynamic Allocation Runner) ============
# =================================================================

def main_genesis() -> None:
    print(f"[main_genesis] OMUX-v{__version__} 推論処理開始。")
    start_time = time.time()
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        output_dir = "/kaggle/working/"
        search_pattern_test = "/kaggle/input/*/arc-agi_test_challenges.json"
        test_files = glob(search_pattern_test)
        tasks_file = test_files[0] if test_files else None
        if not tasks_file:
            search_pattern_eval = "/kaggle/input/*/arc-agi_evaluation_challenges.json"
            eval_files = glob(search_pattern_eval)
            tasks_file = eval_files[0] if eval_files else None
        print(f"Kaggle環境を検出。タスクファイル: {tasks_file}")
    else:
        script_dir = Path(__file__).resolve().parent
        tasks_file = script_dir / "data" / "arc-agi_test_challenges.json"
        output_dir = script_dir / "output"
        os.makedirs(output_dir, exist_ok=True)
        print(f"ローカル実行モード: 出力={output_dir}")
    if not tasks_file or not os.path.exists(tasks_file):
         print(f"[クリティカルエラー] 処理対象のチャレンジファイルが見つかりません: {tasks_file}", file=sys.stderr); return
    tasks = load_tasks(str(tasks_file))
    if not tasks: print("[終了] タスクの読み込みに失敗、またはタスクが空です。"); return
    print(f"合計 {len(tasks)} 個のタスクを読み込みました。")
    
    final_submission_data = {}
    use_parallel = Config.MAX_PARALLEL_TASKS > 1
    print(f"[INFO] 実行モード: {'並列' if use_parallel else '逐次'}, ワーカー数: {Config.MAX_PARALLEL_TASKS}")
    
    print(f"\nフェーズ1: ベースライン・プロトタイピング ({'並列' if use_parallel else '逐次'}, {Config.PROTOTYPING_TIME_SECONDS_PER_TASK}秒/タスク)...")
    prototyping_results = {}
    
    if use_parallel:
        with ProcessPoolExecutor(max_workers=Config.MAX_PARALLEL_TASKS) as executor:
            future_to_id = {executor.submit(run_task_prototyping, task, i % Config.MAX_PARALLEL_TASKS): task['id'] 
                            for i, task in enumerate(tasks)}
            for i, future in enumerate(as_completed(future_to_id)):
                task_id = future_to_id[future]
                try:
                    _, score, program = future.result()
                    prototyping_results[task_id] = {'score': score, 'program': program}
                    print(f"\rプロトタイピング中... [{i+1}/{len(tasks)}] ID: {task_id}, Score: {score:.2f}", end="")
                except Exception as e:
                    print(f"\nプロトタイピング中にワーカーエラー: Task ID {task_id}, Error: {e}")
                    prototyping_results[task_id] = {'score': -float('inf'), 'program': None}
                gc.collect()
    else:
        for i, task in enumerate(tasks):
            task_id, score, program = run_task_prototyping(task, 0)
            prototyping_results[task_id] = {'score': score, 'program': program}
            elapsed = time.time() - start_time
            print(f"プロトタイピング中 [{i+1}/{len(tasks)}] ID: {task_id}, Score: {score:.2f}, Time: {elapsed:.1f}s", flush=True)
            
    print("\nフェーズ2: 戦略的時間予算の再配分...")
    prototyping_time = time.time() - start_time
    available_time = Config.GLOBAL_TIME_LIMIT_HOURS * 3600 - prototyping_time - 300
    tasks_to_solve = []; allocation_weights = {}
    for task in tasks:
        task_id = task['id']; result = prototyping_results.get(task_id, {'score': -float('inf')}); score = result['score']
        if score >= Config.PROTOTYPING_SUCCESS_THRESHOLD:
            print(f"\n[INFO] タスク {task_id} はプロトタイピングで解決済み。最終予測を生成します。")
            initialize_worker(0) # Need to init for DSL
            dsl_temp = DSLExecutor()
            predictions = []
            for pair in task['test']:
                pred = dsl_temp.execute(result['program'], xp.asarray(pair['input']))
                if pred is None: pred = pair['input']
                predictions.append([to_cpu(pred), to_cpu(pred)])
            final_submission_data[task_id] = predictions
        else:
            tasks_to_solve.append(task)
            if score >= Config.PROTOTYPING_PROMISING_THRESHOLD: allocation_weights[task_id] = Config.TIME_ALLOCATION_PROMISING_BIAS
            else: allocation_weights[task_id] = 1.0
    task_budgets = {}
    total_weight = sum(allocation_weights.values())
    if total_weight > 0 and len(tasks_to_solve) > 0:
        time_per_weight = available_time / total_weight
        for task_id, weight in allocation_weights.items():
            budget = time_per_weight * weight
            task_budgets[task_id] = max(Config.MIN_TASK_TIME_SECONDS, min(Config.MAX_TASK_TIME_SECONDS, budget))
    
    print(f"\nフェーズ3: {len(tasks_to_solve)}個のタスクを本探索します ({'並列' if use_parallel else '逐次'}, 合計時間: {available_time/60:.1f}分)...")
    if use_parallel and len(tasks_to_solve) > 0:
        with ProcessPoolExecutor(max_workers=Config.MAX_PARALLEL_TASKS) as executor:
            future_to_task = {executor.submit(run_task_solver, task, task_budgets.get(task['id'], Config.MIN_TASK_TIME_SECONDS), i % Config.MAX_PARALLEL_TASKS): task 
                              for i, task in enumerate(tasks_to_solve)}
            for i, future in enumerate(as_completed(future_to_task)):
                task_obj = future_to_task[future]
                task_id = task_obj['id']
                try:
                    _, preds, score, sm_used = future.result()
                    final_submission_data[task_id] = preds
                    sm_flag = "[S]" if sm_used else ""
                    print(f"\r完了 [{i+1}/{len(tasks_to_solve)}]: タスク={task_id}, スコア={score:.2f} {sm_flag}", end="")
                except Exception as e:
                    print(f"\n[エラー] タスク {task_id} の結果取得に失敗: {e}", file=sys.stderr)
                    final_submission_data[task_id] = [[to_cpu(p['input']), to_cpu(p['input'])] for p in task_obj['test']]
                gc.collect()
    else:
        for i, task in enumerate(tasks_to_solve):
            budget = task_budgets.get(task['id'], Config.MIN_TASK_TIME_SECONDS)
            task_id, preds, score, sm_used = run_task_solver(task, budget, 0)
            final_submission_data[task_id] = preds
            sm_flag = "[S]" if sm_used else ""
            elapsed = time.time() - start_time
            print(f"探索完了 [{i+1}/{len(tasks_to_solve)}]: ID={task_id}, Score={score:.2f} {sm_flag}, Total Time: {elapsed:.1f}s", flush=True)
            
    save_submission(final_submission_data, str(output_dir))
    save_run_metadata(output_dir, start_time, len(tasks), len(final_submission_data), 0)
    print(f"\n\n=== OMUX-v{__version__} 推論処理完了。合計時間: {time.time() - start_time:.2f}秒 ===")

if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != 'spawn':
        try: mp.set_start_method('spawn', force=True)
        except RuntimeError: pass
    main_genesis()



