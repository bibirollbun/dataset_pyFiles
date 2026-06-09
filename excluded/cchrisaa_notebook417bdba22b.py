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
# coding: utf-8

"""
ARC Prize 2025 Competition Notebook - FIXED VERSION
Enhanced Hybrid Neuro-Symbolic Solver
"""

# --- Standard and Data Science Imports ---
import json
import os
import time
import random
import sys
import traceback
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tqdm import tqdm
import gc
from collections import defaultdict

# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

# --- Visualization Imports ---
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ==========================================
# CONFIGURATION
# ==========================================
DEBUG_TASK_COUNT = None 

# --- Paths ---
try:
    DATA_PATH = Path('/kaggle/input/arc-prize-2025')
    WORK_PATH = Path('/kaggle/working/')
except:
    print("Kaggle paths not found, using local paths.")
    DATA_PATH = Path('./data')
    WORK_PATH = Path('./working')
    WORK_PATH.mkdir(exist_ok=True)

TRAINING_CHALLENGES = DATA_PATH / 'arc-agi_training_challenges.json'
EVALUATION_CHALLENGES = DATA_PATH / 'arc-agi_evaluation_challenges.json'
TEST_CHALLENGES = DATA_PATH / 'arc-agi_test_challenges.json'

PREPROCESSED_DATA_PATH = WORK_PATH / 'preprocessed_arc_training_enhanced.pt'
MODEL_SAVE_PATH = WORK_PATH / 'arc_neuro_symbolic_model_enhanced.pth'
SUBMISSION_PATH = WORK_PATH / 'submission.json'

# --- Model Hyperparameters ---
OBJECT_FEATURE_DIM = 24
RELATIONSHIP_FEATURE_DIM = 12
HIDDEN_DIM = 256  # Reduced for stability
NUM_RULES = 40
TRANSFORMER_HEADS = 8
TRANSFORMER_LAYERS = 4  # Reduced for stability
MAX_OBJECTS = 100
MAX_RELATIONSHIPS = 200

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-4
EPOCHS = 30
BATCH_SIZE = 32
NUM_WORKERS = 0  # Set to 0 to avoid multiprocessing issues
GRADIENT_CLIP = 1.0
WARMUP_EPOCHS = 3

# --- GPU Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"ğŸš€ Using device: {device}")
if NUM_GPUS > 0:
    print(f"ğŸ”¥ Found {NUM_GPUS} GPUs")

# ==========================================
# 1. SYMBOLIC PERCEPTION
# ==========================================

@dataclass
class ObjectRelationship:
    """Represents spatial and visual relationships between objects."""
    obj1_idx: int
    obj2_idx: int
    distance: float
    relative_position: str
    color_match: bool
    size_ratio: float
    alignment: str
    touching: bool
    
@dataclass
class ARCObject:
    """Dataclass with geometric properties."""
    shape: np.ndarray
    color: int
    position: Tuple[int, int]
    size: int
    width: int
    height: int
    center: Tuple[float, float]
    bounding_box: Tuple[int, int, int, int]
    
    # Geometric properties
    is_line: bool = False
    is_rectangle: bool = False
    is_symmetric: bool = False
    is_square: bool = False
    aspect_ratio: float = 1.0
    is_l_shape: bool = False
    is_cross: bool = False
    has_holes: bool = False
    num_corners: int = 0
    orientation: float = 0.0
    compactness: float = 1.0
    
    def __post_init__(self):
        self._analyze_geometric_properties()
    
    def _analyze_geometric_properties(self):
        """Analyze geometric properties."""
        self.aspect_ratio = self.width / max(1, self.height)
        self.is_square = self.width == self.height
        
        if self.width == 1 or self.height == 1:
            self.is_line = True
            
        if np.sum(self.shape) == self.width * self.height:
            self.is_rectangle = True
            
        if (np.array_equal(self.shape, np.fliplr(self.shape)) or 
            np.array_equal(self.shape, np.flipud(self.shape))):
            self.is_symmetric = True
            
        # Simplified shape detection
        self.is_l_shape = self._check_l_shape()
        self.is_cross = self._check_cross_shape()
        
        # Hole detection
        filled = ndimage.binary_fill_holes(self.shape)
        if not np.array_equal(filled, self.shape):
            self.has_holes = True
            
        self.num_corners = self._count_corners()
        self.orientation = 0.0  # Simplified
        
        perimeter = self._calculate_perimeter()
        if perimeter > 0:
            self.compactness = (4 * np.pi * self.size) / (perimeter ** 2)
    
    def _check_l_shape(self) -> bool:
        """Simple L-shape check."""
        if self.size < 3 or self.is_rectangle:
            return False
        # Basic heuristic
        return self.size < 0.75 * self.width * self.height
    
    def _check_cross_shape(self) -> bool:
        """Simple cross shape check."""
        if self.width < 3 or self.height < 3:
            return False
        h_center = self.height // 2
        v_center = self.width // 2
        return (np.sum(self.shape[h_center, :]) > 2 and 
                np.sum(self.shape[:, v_center]) > 2 and
                self.size < 0.8 * self.width * self.height)
    
    def _count_corners(self) -> int:
        """Count approximate corners."""
        corners = 0
        h, w = self.shape.shape
        for i in range(1, h-1):
            for j in range(1, w-1):
                if self.shape[i, j]:
                    neighbors = sum([
                        self.shape[i-1, j], self.shape[i+1, j],
                        self.shape[i, j-1], self.shape[i, j+1]
                    ])
                    if neighbors == 2:
                        corners += 1
        return min(corners, 20)  # Cap at 20
    
    def _calculate_perimeter(self) -> int:
        """Calculate perimeter."""
        if self.shape.size == 0:
            return 0
        # Use simple edge detection
        padded = np.pad(self.shape, 1, mode='constant', constant_values=0)
        edges = 0
        h, w = self.shape.shape
        for i in range(h):
            for j in range(w):
                if self.shape[i, j]:
                    # Count edges with background
                    edges += 4 - sum([
                        padded[i, j+1], padded[i+2, j+1],
                        padded[i+1, j], padded[i+1, j+2]
                    ])
        return edges

class MultiScaleObjectParser:
    """Parser with multi-scale perception."""
    def __init__(self, min_object_size: int = 1):
        self.min_object_size = min_object_size
        self.block_sizes = [2, 3, 4]
    
    def parse_grid(self, grid: np.ndarray) -> Tuple[List[ARCObject], List[ObjectRelationship]]:
        """Parse grid and extract relationships."""
        # Parse objects
        objects = self._parse_connected_components(grid)
        
        # Extract relationships
        relationships = self._extract_relationships(objects)
        
        return objects, relationships
    
    def _parse_connected_components(self, grid: np.ndarray) -> List[ARCObject]:
        """Parse connected components."""
        objects = []
        for color_val in range(1, 10):
            color_mask = (grid == color_val)
            if not color_mask.any():
                continue
            
            labeled_mask, num_objects = ndimage.label(color_mask)
            
            for obj_id in range(1, num_objects + 1):
                obj_mask = (labeled_mask == obj_id)
                if np.sum(obj_mask) < self.min_object_size:
                    continue
                
                coords = np.where(obj_mask)
                if len(coords[0]) == 0:
                    continue
                    
                min_row, max_row = np.min(coords[0]), np.max(coords[0])
                min_col, max_col = np.min(coords[1]), np.max(coords[1])
                
                shape_mask = obj_mask[min_row:max_row+1, min_col:max_col+1]
                
                objects.append(ARCObject(
                    shape=shape_mask,
                    color=color_val,
                    position=(min_row, min_col),
                    size=np.sum(obj_mask),
                    width=shape_mask.shape[1],
                    height=shape_mask.shape[0],
                    center=(np.mean(coords[0]), np.mean(coords[1])),
                    bounding_box=(min_row, min_col, max_row, max_col)
                ))
        return objects
    
    def _extract_relationships(self, objects: List[ARCObject]) -> List[ObjectRelationship]:
        """Extract spatial relationships."""
        relationships = []
        
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects[i+1:], i+1):
                # Calculate distance
                dist = np.sqrt((obj1.center[0] - obj2.center[0])**2 + 
                              (obj1.center[1] - obj2.center[1])**2)
                
                # Relative position
                dy = obj2.center[0] - obj1.center[0]
                dx = obj2.center[1] - obj1.center[1]
                if abs(dy) > abs(dx):
                    rel_pos = 'below' if dy > 0 else 'above'
                else:
                    rel_pos = 'right' if dx > 0 else 'left'
                
                # Check if touching
                touching = self._objects_touching(obj1, obj2)
                
                # Alignment
                if abs(dy) < 1:
                    alignment = 'horizontal'
                elif abs(dx) < 1:
                    alignment = 'vertical'
                elif abs(abs(dy) - abs(dx)) < 1:
                    alignment = 'diagonal'
                else:
                    alignment = 'none'
                
                relationships.append(ObjectRelationship(
                    obj1_idx=i,
                    obj2_idx=j,
                    distance=dist,
                    relative_position=rel_pos,
                    color_match=(obj1.color == obj2.color),
                    size_ratio=obj1.size / max(1, obj2.size),
                    alignment=alignment,
                    touching=touching
                ))
        
        return relationships
    
    def _objects_touching(self, obj1: ARCObject, obj2: ARCObject) -> bool:
        """Check if two objects are touching."""
        bb1 = obj1.bounding_box
        bb2 = obj2.bounding_box
        
        # Check if bounding boxes are adjacent
        return (abs(bb1[2] - bb2[0]) <= 1 or abs(bb1[0] - bb2[2]) <= 1 or
                abs(bb1[3] - bb2[1]) <= 1 or abs(bb1[1] - bb2[3]) <= 1)

# ==========================================
# 2. BRIDGE: OBJECTS TO TENSORS
# ==========================================

def objects_to_tensor(objects: List[ARCObject], relationships: List[ObjectRelationship], 
                     grid_shape: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert objects and relationships to tensors."""
    if not objects:
        return torch.zeros(1, OBJECT_FEATURE_DIM), torch.zeros(1, RELATIONSHIP_FEATURE_DIM)
    
    # Object features
    object_features = []
    max_h, max_w = max(1, grid_shape[0]), max(1, grid_shape[1])
    
    for obj in objects:
        features = [
            obj.position[1] / max_w,
            obj.position[0] / max_h,
            obj.color / 9.0,
            obj.size / (max_w * max_h),
            obj.width / max_w,
            obj.height / max_h,
            float(obj.is_line),
            float(obj.is_rectangle),
            float(obj.is_symmetric),
            float(obj.is_square),
            float(obj.is_l_shape),
            float(obj.is_cross),
            float(obj.has_holes),
            obj.aspect_ratio / 10.0,
            obj.num_corners / 20.0,
            obj.orientation / np.pi,
            obj.compactness,
            obj.center[0] / max_h,
            obj.center[1] / max_w,
            obj.bounding_box[0] / max_h,
            obj.bounding_box[1] / max_w,
            obj.bounding_box[2] / max_h,
            obj.bounding_box[3] / max_w,
            0.0  # Reserved
        ]
        object_features.append(features)
    
    # Relationship features
    relationship_features = []
    for rel in relationships:
        features = [
            rel.obj1_idx / max(1, len(objects)),
            rel.obj2_idx / max(1, len(objects)),
            rel.distance / np.sqrt(max_h**2 + max_w**2),
            float(rel.relative_position == 'above'),
            float(rel.relative_position == 'below'),
            float(rel.relative_position == 'left'),
            float(rel.relative_position == 'right'),
            float(rel.color_match),
            rel.size_ratio / 10.0,
            float(rel.alignment == 'horizontal'),
            float(rel.alignment == 'vertical'),
            float(rel.touching)
        ]
        relationship_features.append(features)
    
    if not relationship_features:
        relationship_features = [[0.0] * RELATIONSHIP_FEATURE_DIM]
    
    return (torch.tensor(object_features, dtype=torch.float32),
            torch.tensor(relationship_features, dtype=torch.float32))

# ==========================================
# 3. NEURAL COMPONENTS
# ==========================================

class DifferentiableRuleExecutor(nn.Module):
    """Executes weighted combination of rules."""
    def __init__(self, rules):
        super().__init__()
        self.rules = rules
    
    def forward(self, x, rule_weights):
        batch_outputs = []
        for i in range(x.shape[0]):
            # Apply each rule and weight the outputs
            weighted_outputs = []
            for j, rule in enumerate(self.rules):
                rule_output = rule(x[i])
                weighted_output = rule_weights[i, j] * rule_output
                weighted_outputs.append(weighted_output)
            
            # Sum weighted outputs
            combined = torch.stack(weighted_outputs, dim=0).sum(dim=0)
            batch_outputs.append(combined)
        
        return torch.stack(batch_outputs, dim=0)

class TransformerTaskEmbedder(nn.Module):
    """Basic transformer for task embedding."""
    def __init__(self, d_model=HIDDEN_DIM, nhead=TRANSFORMER_HEADS, num_layers=TRANSFORMER_LAYERS):
        super().__init__()
        self.d_model = d_model
        self.pos_embedding = nn.Parameter(torch.randn(MAX_OBJECTS, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask):
        batch_size, seq_len, _ = src.shape
        
        if seq_len > MAX_OBJECTS:
            src = src[:, :MAX_OBJECTS, :]
            src_mask = src_mask[:, :MAX_OBJECTS]
            seq_len = MAX_OBJECTS
        
        # Add positional encoding
        src = src + self.pos_embedding[:seq_len, :].unsqueeze(0)
        
        # Transform
        output = self.transformer(src, src_key_padding_mask=src_mask)
        output = self.norm(output)
        
        # Pool to single vector
        output = output.masked_fill(src_mask.unsqueeze(-1), 0)
        lengths = (~src_mask).sum(dim=1, keepdim=True).float()
        pooled = output.sum(dim=1) / lengths.clamp(min=1)
        
        return pooled

class OutputShapePredictor(nn.Module):
    """Predicts output grid dimensions."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Sigmoid()
        )
    
    def forward(self, task_embedding):
        # Returns scale factors (0.5 to 2.0)
        return self.predictor(task_embedding) * 1.5 + 0.5

class RuleWeightPredictor(nn.Module):
    """Predicts rule weights."""
    def __init__(self, num_rules, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, num_rules),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, task_embedding):
        return self.mlp(task_embedding)

class EnhancedARCNeuroSymbolicModel(nn.Module):
    """Main model."""
    def __init__(self, rules, input_dim=OBJECT_FEATURE_DIM, hidden_dim=HIDDEN_DIM, num_rules=NUM_RULES):
        super().__init__()
        # Encoders
        self.obj_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.rel_encoder = nn.Linear(RELATIONSHIP_FEATURE_DIM, hidden_dim)
        
        # Task understanding
        self.task_embedder = TransformerTaskEmbedder(d_model=hidden_dim)
        
        # Prediction heads
        self.rule_predictor = RuleWeightPredictor(num_rules, hidden_dim)
        self.rule_executor = DifferentiableRuleExecutor(rules)
        self.shape_predictor = OutputShapePredictor(hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(input_dim, input_dim)

    def forward(self, obj_features, obj_mask, rel_features=None, rel_mask=None):
        # Encode objects
        encoded_objs = self.obj_encoder(obj_features)
        
        # Get task embedding
        task_embedding = self.task_embedder(encoded_objs, obj_mask)
        
        # Predict transformations
        rule_weights = self.rule_predictor(task_embedding)
        shape_scales = self.shape_predictor(task_embedding)
        
        # Apply rules
        transformed = self.rule_executor(obj_features, rule_weights)
        
        # Final projection
        output = self.output_proj(transformed)
        
        return output, rule_weights, shape_scales

# ==========================================
# 4. RULE LIBRARY
# ==========================================

# Basic movement rules
def identity(t): return t.clone()
def move_right(t): 
    c = t.clone()
    c[:, 0] = torch.clamp(c[:, 0] + 0.1, 0, 1)
    return c
def move_left(t):
    c = t.clone()
    c[:, 0] = torch.clamp(c[:, 0] - 0.1, 0, 1)
    return c
def move_down(t):
    c = t.clone()
    c[:, 1] = torch.clamp(c[:, 1] + 0.1, 0, 1)
    return c
def move_up(t):
    c = t.clone()
    c[:, 1] = torch.clamp(c[:, 1] - 0.1, 0, 1)
    return c

# Color rules
def cycle_color_forward(t):
    c = t.clone()
    c[:, 2] = torch.fmod(c[:, 2] * 9 + 1, 10) / 9.0
    return c
def cycle_color_backward(t):
    c = t.clone()
    c[:, 2] = torch.fmod(c[:, 2] * 9 - 1 + 10, 10) / 9.0
    return c
def set_color_by_size(t):
    c = t.clone()
    c[:, 2] = torch.clamp(c[:, 3] * 9, 0, 1)
    return c
def set_color_by_position(t):
    c = t.clone()
    c[:, 2] = torch.clamp((c[:, 0] + c[:, 1]) / 2, 0, 1)
    return c

# Geometric transformations
def rotate_90(t):
    r = t.clone()
    # Swap position coordinates
    old_x, old_y = t[:, 0].clone(), t[:, 1].clone()
    r[:, 0] = 1 - old_y
    r[:, 1] = old_x
    # Swap width/height
    r[:, 4], r[:, 5] = t[:, 5], t[:, 4]
    return r

def mirror_horizontal(t):
    m = t.clone()
    m[:, 0] = 1.0 - m[:, 0]
    return m
def mirror_vertical(t):
    m = t.clone()
    m[:, 1] = 1.0 - m[:, 1]
    return m

# Scaling
def scale_up(t):
    s = t.clone()
    s[:, 3:6] = torch.clamp(s[:, 3:6] * 1.5, 0, 1)
    return s
def scale_down(t):
    s = t.clone()
    s[:, 3:6] = torch.clamp(s[:, 3:6] * 0.7, 0, 1)
    return s

# Alignment
def center_all(t):
    c = t.clone()
    c[:, 0] = 0.5
    c[:, 1] = 0.5
    return c
def align_top(t):
    a = t.clone()
    a[:, 1] = 0.1
    return a
def align_bottom(t):
    a = t.clone()
    a[:, 1] = 0.9
    return a
def align_left(t):
    a = t.clone()
    a[:, 0] = 0.1
    return a
def align_right(t):
    a = t.clone()
    a[:, 0] = 0.9
    return a

# Conditional rules
def move_if_small(t):
    mask = (t[:, 3] < 0.1).float().unsqueeze(1)
    movement = torch.zeros_like(t)
    movement[:, 0] = 0.1
    return t + mask * movement
def move_if_large(t):
    mask = (t[:, 3] > 0.3).float().unsqueeze(1)
    movement = torch.zeros_like(t)
    movement[:, 1] = 0.1
    return t + mask * movement
def delete_if_corner(t):
    corner_mask = ((t[:, 0] < 0.2) | (t[:, 0] > 0.8)) & ((t[:, 1] < 0.2) | (t[:, 1] > 0.8))
    d = t.clone()
    d[corner_mask] = 0
    return d

# Pattern rules
def spread_horizontal(t):
    s = t.clone()
    n = t.shape[0]
    if n > 1:
        positions = torch.linspace(0.1, 0.9, n, device=t.device)
        s[:, 0] = positions
    return s
def spread_vertical(t):
    s = t.clone()
    n = t.shape[0]
    if n > 1:
        positions = torch.linspace(0.1, 0.9, n, device=t.device)
        s[:, 1] = positions
    return s
def grid_arrange(t):
    g = t.clone()
    n = t.shape[0]
    grid_size = int(np.sqrt(n)) + 1
    for i in range(n):
        row = i // grid_size
        col = i % grid_size
        g[i, 0] = (col + 0.5) / grid_size
        g[i, 1] = (row + 0.5) / grid_size
    return g

# Size modifications
def grow_all(t):
    g = t.clone()
    g[:, 3:6] = torch.clamp(g[:, 3:6] * 1.2, 0, 1)
    return g
def shrink_all(t):
    s = t.clone()
    s[:, 3:6] = torch.clamp(s[:, 3:6] * 0.8, 0, 1)
    return s

# Additional rules for diversity
def swap_xy(t):
    s = t.clone()
    s[:, 0], s[:, 1] = t[:, 1], t[:, 0]
    return s
def set_color_red(t):
    c = t.clone()
    c[:, 2] = 2.0 / 9.0  # Red
    return c
def set_color_blue(t):
    c = t.clone()
    c[:, 2] = 1.0 / 9.0  # Blue
    return c
def set_color_green(t):
    c = t.clone()
    c[:, 2] = 3.0 / 9.0  # Green
    return c

# Group operations
def group_by_color(t):
    if t.shape[0] <= 1:
        return t
    g = t.clone()
    # Sort by color
    sorted_indices = torch.argsort(t[:, 2])
    g = g[sorted_indices]
    # Spread horizontally
    return spread_horizontal(g)

def push_apart(t):
    p = t.clone()
    n = t.shape[0]
    if n > 1:
        center = torch.mean(t[:, :2], dim=0)
        for i in range(n):
            diff = t[i, :2] - center
            p[i, :2] = torch.clamp(t[i, :2] + 0.1 * diff, 0, 1)
    return p

def pull_together(t):
    p = t.clone()
    n = t.shape[0]
    if n > 1:
        center = torch.mean(t[:, :2], dim=0)
        for i in range(n):
            diff = center - t[i, :2]
            p[i, :2] = torch.clamp(t[i, :2] + 0.1 * diff, 0, 1)
    return p

# Final rules to reach 40
def diagonal_arrange(t):
    d = t.clone()
    n = t.shape[0]
    for i in range(n):
        pos = i / max(1, n - 1)
        d[i, 0] = pos * 0.8 + 0.1
        d[i, 1] = pos * 0.8 + 0.1
    return d

def reverse_diagonal(t):
    d = t.clone()
    n = t.shape[0]
    for i in range(n):
        pos = i / max(1, n - 1)
        d[i, 0] = pos * 0.8 + 0.1
        d[i, 1] = (1 - pos) * 0.8 + 0.1
    return d

# Complete rule list
enhanced_rules = [
    identity, move_right, move_left, move_down, move_up,
    cycle_color_forward, cycle_color_backward, set_color_by_size, set_color_by_position,
    rotate_90, mirror_horizontal, mirror_vertical,
    scale_up, scale_down, center_all, align_top, align_bottom,
    align_left, align_right, move_if_small, move_if_large, delete_if_corner,
    spread_horizontal, spread_vertical, grid_arrange,
    grow_all, shrink_all, swap_xy, set_color_red, set_color_blue,
    set_color_green, group_by_color, push_apart, pull_together,
    diagonal_arrange, reverse_diagonal,
    identity, identity, identity, identity  # Padding
]
NUM_RULES = len(enhanced_rules)

# ==========================================
# 5. DATA LOADING
# ==========================================

def preprocess_and_save_data_enhanced(tasks, save_path):
    """Preprocess tasks."""
    samples = []
    parser = MultiScaleObjectParser()
    
    print(f"ğŸ“Š Pre-processing {len(tasks)} tasks...")
    
    for task_id, task in tqdm(tasks.items()):
        for example in task['train']:
            try:
                input_grid = np.array(example['input'])
                output_grid = np.array(example['output'])
                
                # Parse grids
                input_objects, input_relationships = parser.parse_grid(input_grid)
                target_objects, target_relationships = parser.parse_grid(output_grid)
                
                # Convert to tensors
                input_obj_tensor, input_rel_tensor = objects_to_tensor(
                    input_objects, input_relationships, input_grid.shape
                )
                target_obj_tensor, target_rel_tensor = objects_to_tensor(
                    target_objects, target_relationships, output_grid.shape
                )
                
                samples.append({
                    'input_obj_tensor': input_obj_tensor,
                    'input_rel_tensor': input_rel_tensor,
                    'target_obj_tensor': target_obj_tensor,
                    'target_rel_tensor': target_rel_tensor,
                    'task_id': task_id,
                    'input_grid_shape': input_grid.shape,
                    'output_grid_shape': output_grid.shape,
                    'input_objects': input_objects,
                    'target_objects': target_objects
                })
                
            except Exception as e:
                print(f"âš ï¸� Error processing task {task_id}: {e}")
                continue
    
    torch.save(samples, save_path)
    print(f"âœ… Saved {len(samples)} samples")

class ARCDataset(Dataset):
    def __init__(self, preprocessed_file):
        # Use weights_only=False since we trust our own saved data
        self.samples = torch.load(preprocessed_file, weights_only=False)
        print(f"ğŸ“š Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def arc_collate_fn(batch):
    """Collate function."""
    input_objs = [item['input_obj_tensor'] for item in batch]
    target_objs = [item['target_obj_tensor'] for item in batch]
    
    # Pad objects
    max_obj_in = max(t.shape[0] for t in input_objs)
    max_obj_out = max(t.shape[0] for t in target_objs)
    
    padded_input_objs = torch.zeros(len(batch), max_obj_in, OBJECT_FEATURE_DIM)
    input_obj_masks = torch.ones(len(batch), max_obj_in, dtype=torch.bool)
    padded_target_objs = torch.zeros(len(batch), max_obj_out, OBJECT_FEATURE_DIM)
    target_obj_masks = torch.ones(len(batch), max_obj_out, dtype=torch.bool)
    
    for i in range(len(batch)):
        padded_input_objs[i, :input_objs[i].shape[0], :] = input_objs[i]
        input_obj_masks[i, :input_objs[i].shape[0]] = False
        padded_target_objs[i, :target_objs[i].shape[0], :] = target_objs[i]
        target_obj_masks[i, :target_objs[i].shape[0]] = False
    
    other_data = {
        k: [item[k] for item in batch] 
        for k in batch[0] if not k.endswith('_tensor')
    }
    
    return (padded_input_objs, input_obj_masks,
            padded_target_objs, target_obj_masks, other_data)

# ==========================================
# 6. TRAINING
# ==========================================

class MatchingLoss(nn.Module):
    def __init__(self, feature_weight=1.0, position_weight=2.0, color_weight=3.0):
        super().__init__()
        self.fw = feature_weight
        self.pw = position_weight
        self.cw = color_weight
    
    def forward(self, pred_objs, target_objs, pred_mask, target_mask):
        total_loss = 0
        
        for i in range(pred_objs.shape[0]):
            pred = pred_objs[i][~pred_mask[i]]
            target = target_objs[i][~target_mask[i]]
            
            if pred.shape[0] == 0 and target.shape[0] == 0:
                continue
            if pred.shape[0] == 0 or target.shape[0] == 0:
                total_loss += 10.0
                continue
            
            # Cost matrix
            pos_cost = torch.cdist(pred[:, :2], target[:, :2], p=2) * self.pw
            col_cost = torch.cdist(pred[:, 2:3], target[:, 2:3], p=1) * self.cw
            feat_cost = torch.cdist(pred[:, 3:], target[:, 3:], p=1) * self.fw
            
            cost_matrix = pos_cost + col_cost + feat_cost
            
            # Hungarian matching
            with torch.no_grad():
                row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())
            
            total_loss += cost_matrix[row_ind, col_ind].sum()
        
        return total_loss / pred_objs.shape[0]

def train_epoch(model, dataloader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0
    
    for batch_data in tqdm(dataloader, desc="Training"):
        try:
            inputs, input_masks, targets, target_masks, _ = batch_data
            
            inputs = inputs.to(device)
            input_masks = input_masks.to(device)
            targets = targets.to(device)
            target_masks = target_masks.to(device)
            
            optimizer.zero_grad()
            
            with autocast(device_type='cuda', enabled=device.type == 'cuda'):
                predictions, _, _ = model(inputs, input_masks)
                loss = criterion(predictions, targets, input_masks, target_masks)
            
            if torch.isnan(loss):
                continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
        except Exception as e:
            print(f"âš ï¸� Batch error: {e}")
            continue
    
    return total_loss / len(dataloader)

# ==========================================
# 7. RECONSTRUCTION
# ==========================================

def reconstruct_grid_enhanced(
    predicted_features: torch.Tensor,
    original_objects: List[ARCObject],
    shape_scales: torch.Tensor,
    input_shape: Tuple[int, int]
) -> np.ndarray:
    """Reconstruct grid from predictions."""
    # Determine output shape
    scale_h, scale_w = shape_scales.cpu().numpy()
    output_shape = (
        int(input_shape[0] * scale_h),
        int(input_shape[1] * scale_w)
    )
    
    grid = np.zeros(output_shape, dtype=int)
    
    num_objects = min(len(predicted_features), len(original_objects))
    
    for i in range(num_objects):
        pred_feat = predicted_features[i].cpu().numpy()
        orig_obj = original_objects[i]
        
        # Extract predicted properties
        pred_x = int(pred_feat[0] * output_shape[1])
        pred_y = int(pred_feat[1] * output_shape[0])
        pred_color = max(1, min(9, int(round(pred_feat[2] * 9))))
        
        # Use original shape
        shape_to_place = orig_obj.shape
        
        # Place on grid
        y_start = max(0, pred_y)
        x_start = max(0, pred_x)
        y_end = min(y_start + shape_to_place.shape[0], output_shape[0])
        x_end = min(x_start + shape_to_place.shape[1], output_shape[1])
        
        if y_start < output_shape[0] and x_start < output_shape[1]:
            sub_shape = shape_to_place[:y_end - y_start, :x_end - x_start]
            grid_slice = grid[y_start:y_end, x_start:x_end]
            np.copyto(grid_slice, pred_color, where=sub_shape)
    
    return grid

# ==========================================
# 8. INFERENCE
# ==========================================

def solve_task_enhanced(model, task, parser, device):
    """Solve a single task."""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for test_example in task['test']:
            test_grid = np.array(test_example['input'])
            
            # Parse
            test_objects, test_relationships = parser.parse_grid(test_grid)
            
            # Convert to tensor
            obj_tensor, rel_tensor = objects_to_tensor(
                test_objects, test_relationships, test_grid.shape
            )
            
            # Prepare batch
            obj_inputs = obj_tensor.unsqueeze(0).to(device)
            obj_mask = torch.zeros(1, obj_inputs.shape[1], dtype=torch.bool).to(device)
            
            # Forward
            with autocast(device_type='cuda', enabled=device.type == 'cuda'):
                output_tensor, _, shape_scales = model(obj_inputs, obj_mask)
            
            # Reconstruct
            predicted_grid = reconstruct_grid_enhanced(
                output_tensor[0], test_objects, shape_scales[0],
                test_grid.shape
            )
            
            predictions.append(predicted_grid.tolist())
    
    return predictions

def create_submission(model, test_tasks, parser, device, save_path):
    """Create submission file."""
    submission = {}
    
    print(f"ğŸ�¯ Generating predictions for {len(test_tasks)} tasks...")
    
    for task_id, task in tqdm(test_tasks.items()):
        try:
            predictions = solve_task_enhanced(model, task, parser, device)
            submission[task_id] = [
                {"attempt_1": pred, "attempt_2": pred} for pred in predictions
            ]
        except Exception as e:
            print(f"âš ï¸� Error on task {task_id}: {e}")
            # Fallback
            submission[task_id] = [
                {"attempt_1": ex['input'], "attempt_2": ex['input']} 
                for ex in task['test']
            ]
    
    with open(save_path, 'w') as f:
        json.dump(submission, f)
    print(f"âœ… Submission saved to {save_path}")

# ==========================================
# 9. MAIN
# ==========================================

def main():
    print("ğŸš€ Fixed ARC Neuro-Symbolic Solver")
    print("=" * 50)
    
    # Load data
    force_regenerate = False  # Set to True only if you want to regenerate
    if not PREPROCESSED_DATA_PATH.exists() or force_regenerate:
        print("ğŸ“� Loading training data...")
        try:
            with open(TRAINING_CHALLENGES, 'r') as f:
                training_tasks = json.load(f)
            if DEBUG_TASK_COUNT:
                training_tasks = {k: training_tasks[k] 
                                for k in list(training_tasks)[:DEBUG_TASK_COUNT]}
            preprocess_and_save_data_enhanced(training_tasks, PREPROCESSED_DATA_PATH)
        except FileNotFoundError:
            print(f"â�Œ Training data not found at {TRAINING_CHALLENGES}")
            return
    
    # Initialize model
    model = EnhancedARCNeuroSymbolicModel(rules=enhanced_rules, num_rules=NUM_RULES)
    model.to(device)
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = MatchingLoss()
    scaler = GradScaler(enabled=device.type == 'cuda')
    
    # Dataset
    dataset = ARCDataset(PREPROCESSED_DATA_PATH)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=arc_collate_fn,
        pin_memory=(device.type == 'cuda')
    )
    
    # Training
    print(f"\nğŸ�‹ï¸� Training for {EPOCHS} epochs...")
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        print(f"\nğŸ“… Epoch {epoch + 1}/{EPOCHS}")
        
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, scaler, device)
        scheduler.step()
        
        print(f"  Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_loss': best_loss
            }, MODEL_SAVE_PATH)
            print(f"  ğŸ’¾ Saved best model")
        
        if epoch % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    print("\nâœ… Training complete!")
    
    # Generate submission
    print("\nğŸ“� Generating submission...")
    try:
        # Load best model
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        with open(TEST_CHALLENGES, 'r') as f:
            test_tasks = json.load(f)
        
        parser = MultiScaleObjectParser()
        create_submission(model, test_tasks, parser, device, SUBMISSION_PATH)
        
    except FileNotFoundError:
        print(f"âš ï¸� Test data not found")
    
    print("\nğŸ�‰ Done!")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nğŸ›‘ Interrupted")
    except Exception as e:
        print(f"\nâ�Œ Error: {e}")
        traceback.print_exc()

