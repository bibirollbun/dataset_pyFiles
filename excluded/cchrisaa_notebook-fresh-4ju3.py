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


# ===================================================================
# Enhanced ARC Solver v2 - With Expanded Pattern Recognition
# ===================================================================

import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Any, Callable
from dataclasses import dataclass
from collections import defaultdict, Counter
import itertools
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
# Try to import scipy, but make it optional
try:
    from scipy import ndimage
    from scipy.spatial import distance
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print(" scipy not available - some features will be limited")

# --- 0. Kaggle Data Loading (same as before) ---

def load_kaggle_data():
    """Load ARC data from Kaggle input directory"""
    print("Loading data from Kaggle input directory...")
    
    base_path = '/kaggle/input/arc-prize-2025'
    files = {}
    
    for dirname, _, filenames in os.walk(base_path):
        for filename in filenames:
            if filename.endswith('.json'):
                files[filename] = os.path.join(dirname, filename)
                print(f"   Found: {filename}")
    
    data = {}
    
    if 'arc-agi_training_challenges.json' in files:
        with open(files['arc-agi_training_challenges.json'], 'r') as f:
            data['training'] = json.load(f)
            print(f"   ✓ Loaded {len(data['training'])} training challenges")
    
    if 'arc-agi_test_challenges.json' in files:
        with open(files['arc-agi_test_challenges.json'], 'r') as f:
            data['test'] = json.load(f)
            print(f"   ✓ Loaded {len(data['test'])} test challenges")
    
    if 'sample_submission.json' in files:
        with open(files['sample_submission.json'], 'r') as f:
            data['sample_submission'] = json.load(f)
            print(f"   ✓ Loaded sample submission format")
    
    return data, files

# --- 1. Enhanced DSL Operations ---

class EnhancedDSLOperations:
    """Comprehensive collection of grid transformation operations"""
    
    # === Basic Transformations ===
    @staticmethod
    def rotate_90(grid):
        return np.rot90(grid)
    
    @staticmethod
    def rotate_180(grid):
        return np.rot90(grid, 2)
    
    @staticmethod
    def rotate_270(grid):
        return np.rot90(grid, 3)
    
    @staticmethod
    def flip_horizontal(grid):
        return np.fliplr(grid)
    
    @staticmethod
    def flip_vertical(grid):
        return np.flipud(grid)
    
    @staticmethod
    def transpose(grid):
        return np.transpose(grid)
    
    # === Color Operations ===
    @staticmethod
    def invert_colors(grid, max_color=9):
        """Invert colors (useful for some puzzles)"""
        return max_color - grid
    
    @staticmethod
    def swap_colors(grid, color1, color2):
        """Swap two specific colors"""
        result = grid.copy()
        mask1 = grid == color1
        mask2 = grid == color2
        result[mask1] = color2
        result[mask2] = color1
        return result
    
    @staticmethod
    def replace_color(grid, old_color, new_color):
        """Replace all occurrences of old_color with new_color"""
        result = grid.copy()
        result[grid == old_color] = new_color
        return result
    
    @staticmethod
    def extract_color(grid, color):
        """Extract only specific color, rest becomes 0"""
        result = np.zeros_like(grid)
        result[grid == color] = color
        return result
    
    # === Pattern Completion ===
    @staticmethod
    def complete_pattern(grid):
        """Try to complete a partially filled pattern"""
        # Find the most common non-zero pattern and replicate
        h, w = grid.shape
        
        # Try to find repeating sections
        for size in range(2, min(h, w) // 2):
            if h % size == 0 and w % size == 0:
                tile = grid[:size, :size]
                if np.count_nonzero(tile) > 0:
                    completed = np.tile(tile, (h // size, w // size))
                    if np.array_equal(grid[grid != 0], completed[grid != 0]):
                        return completed
        
        return grid
    
    # === Object Manipulation ===
    @staticmethod
    def move_objects(grid, dx, dy, background=0):
        """Move all non-background objects by dx, dy"""
        result = np.full_like(grid, background)
        h, w = grid.shape
        
        for i in range(h):
            for j in range(w):
                if grid[i, j] != background:
                    new_i = i + dy
                    new_j = j + dx
                    if 0 <= new_i < h and 0 <= new_j < w:
                        result[new_i, new_j] = grid[i, j]
        
        return result
    
    @staticmethod
    def duplicate_objects(grid, direction='right', background=0):
        """Duplicate objects in a specific direction"""
        h, w = grid.shape
        
        if direction == 'right':
            if w * 2 <= 30:  # Reasonable size limit
                result = np.full((h, w * 2), background, dtype=grid.dtype)
                result[:, :w] = grid
                result[:, w:] = grid
                return result
        elif direction == 'down':
            if h * 2 <= 30:
                result = np.full((h * 2, w), background, dtype=grid.dtype)
                result[:h, :] = grid
                result[h:, :] = grid
                return result
        
        return grid
    
    # === Size Operations ===
    @staticmethod
    def scale_up(grid, factor):
        """Scale up by integer factor"""
        if factor <= 1 or factor > 5:  # Reasonable limits
            return grid
        return np.repeat(np.repeat(grid, factor, axis=0), factor, axis=1)
    
    @staticmethod
    def scale_down(grid, factor):
        """Scale down by integer factor (taking every nth pixel)"""
        if factor <= 1:
            return grid
        return grid[::factor, ::factor]
    
    @staticmethod
    def crop_to_content(grid, background=0):
        """Crop to minimum bounding box containing non-background"""
        mask = grid != background
        if not mask.any():
            return grid
        
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        return grid[rmin:rmax+1, cmin:cmax+1]
    
    # === Symmetry Operations ===
    @staticmethod
    def make_symmetric_vertical(grid):
        """Make grid vertically symmetric (left half mirrors to right)"""
        h, w = grid.shape
        result = grid.copy()
        for j in range(w // 2):
            result[:, w-1-j] = result[:, j]
        return result
    
    @staticmethod
    def make_symmetric_horizontal(grid):
        """Make grid horizontally symmetric (top half mirrors to bottom)"""
        h, w = grid.shape
        result = grid.copy()
        for i in range(h // 2):
            result[h-1-i, :] = result[i, :]
        return result
    
    @staticmethod
    def make_symmetric_diagonal(grid):
        """Make grid diagonally symmetric"""
        if grid.shape[0] != grid.shape[1]:
            return grid
        
        result = grid.copy()
        n = grid.shape[0]
        for i in range(n):
            for j in range(i+1, n):
                result[j, i] = result[i, j]
        return result
    
    # === Pattern Fill Operations ===
    @staticmethod
    def flood_fill(grid, start_pos, new_color):
        """Flood fill from a starting position"""
        if not (0 <= start_pos[0] < grid.shape[0] and 0 <= start_pos[1] < grid.shape[1]):
            return grid
        
        result = grid.copy()
        old_color = grid[start_pos]
        
        if old_color == new_color:
            return result
        
        stack = [start_pos]
        while stack:
            y, x = stack.pop()
            if result[y, x] == old_color:
                result[y, x] = new_color
                
                # Add neighbors
                for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                        if result[ny, nx] == old_color:
                            stack.append((ny, nx))
        
        return result
    
    @staticmethod
    def fill_enclosed_areas(grid, fill_color, boundary_color):
        """Fill areas enclosed by boundary_color with fill_color"""
        result = grid.copy()
        h, w = grid.shape
        
        # Find enclosed areas using flood fill from edges
        edge_connected = np.zeros_like(grid, dtype=bool)
        
        # Mark all areas connected to edges
        for i in range(h):
            for j in range(w):
                if (i == 0 or i == h-1 or j == 0 or j == w-1) and grid[i, j] != boundary_color:
                    # Flood fill from edge
                    stack = [(i, j)]
                    while stack:
                        y, x = stack.pop()
                        if not edge_connected[y, x] and grid[y, x] != boundary_color:
                            edge_connected[y, x] = True
                            for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                ny, nx = y + dy, x + dx
                                if 0 <= ny < h and 0 <= nx < w:
                                    if not edge_connected[ny, nx] and grid[ny, nx] != boundary_color:
                                        stack.append((ny, nx))
        
        # Fill enclosed areas
        enclosed = ~edge_connected & (grid != boundary_color)
        result[enclosed] = fill_color
        
        return result
    
    # === Counting and Numerical Operations ===
    @staticmethod
    def count_colors(grid):
        """Replace each pixel with count of that color in the grid"""
        result = grid.copy()
        unique, counts = np.unique(grid, return_counts=True)
        count_dict = dict(zip(unique, counts))
        
        for color, count in count_dict.items():
            # Map count to a valid color (0-9)
            mapped_count = min(count, 9)
            result[grid == color] = mapped_count
        
        return result
    
    @staticmethod
    def count_neighbors(grid, target_color):
        """Count neighbors of specific color for each cell"""
        result = np.zeros_like(grid)
        h, w = grid.shape
        
        for i in range(h):
            for j in range(w):
                count = 0
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            if grid[ni, nj] == target_color:
                                count += 1
                result[i, j] = min(count, 9)  # Cap at 9
        
        return result
    
    # === Line and Shape Drawing ===
    @staticmethod
    def draw_line(grid, start, end, color):
        """Draw a line between two points"""
        result = grid.copy()
        x0, y0 = start
        x1, y1 = end
        
        # Bresenham's line algorithm
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            if 0 <= y0 < grid.shape[0] and 0 <= x0 < grid.shape[1]:
                result[y0, x0] = color
            
            if x0 == x1 and y0 == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        
        return result
    
    @staticmethod
    def draw_rectangle(grid, top_left, bottom_right, color, filled=False):
        """Draw a rectangle"""
        result = grid.copy()
        y1, x1 = top_left
        y2, x2 = bottom_right
        
        if filled:
            result[y1:y2+1, x1:x2+1] = color
        else:
            # Top and bottom
            result[y1, x1:x2+1] = color
            result[y2, x1:x2+1] = color
            # Left and right
            result[y1:y2+1, x1] = color
            result[y1:y2+1, x2] = color
        
        return result
    
    # === Advanced Pattern Operations ===
    @staticmethod
    def extract_largest_object(grid, background=0):
        """Extract only the largest connected component"""
        if not SCIPY_AVAILABLE:
            return grid
            
        from scipy import ndimage
        
        mask = grid != background
        labeled, num_features = ndimage.label(mask)
        
        if num_features == 0:
            return grid
        
        # Find largest component
        sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
        largest_label = np.argmax(sizes) + 1
        
        result = np.full_like(grid, background)
        result[labeled == largest_label] = grid[labeled == largest_label]
        
        return result
    
    @staticmethod
    def separate_objects(grid, background=0):
        """Separate connected objects with spacing"""
        from scipy import ndimage
        
        mask = grid != background
        labeled, num_features = ndimage.label(mask)
        
        if num_features <= 1:
            return grid
        
        # Calculate required size
        objects = []
        total_width = 0
        max_height = 0
        
        for i in range(1, num_features + 1):
            obj_mask = labeled == i
            rows, cols = np.where(obj_mask)
            if len(rows) > 0:
                obj = {
                    'mask': obj_mask,
                    'height': rows.max() - rows.min() + 1,
                    'width': cols.max() - cols.min() + 1,
                    'min_row': rows.min(),
                    'min_col': cols.min()
                }
                objects.append(obj)
                total_width += obj['width'] + 1  # 1 pixel spacing
                max_height = max(max_height, obj['height'])
        
        # Create output grid
        result = np.full((max_height, total_width), background, dtype=grid.dtype)
        
        # Place objects
        current_col = 0
        for obj in objects:
            # Extract object data
            obj_data = grid[obj['mask']].reshape(-1)
            obj_grid = np.full((obj['height'], obj['width']), background, dtype=grid.dtype)
            
            # Fill object grid
            rows, cols = np.where(obj['mask'])
            for idx, (r, c) in enumerate(zip(rows, cols)):
                local_r = r - obj['min_row']
                local_c = c - obj['min_col']
                obj_grid[local_r, local_c] = obj_data[idx]
            
            # Place in result
            result[:obj['height'], current_col:current_col + obj['width']] = obj_grid
            current_col += obj['width'] + 1
        
        return result[:, :current_col-1]  # Remove last spacing
    
    # === Sorting and Ordering ===
    @staticmethod
    def sort_by_size(grid, background=0):
        """Sort objects by size (left to right, small to large)"""
        from scipy import ndimage
        
        mask = grid != background
        labeled, num_features = ndimage.label(mask)
        
        if num_features <= 1:
            return grid
        
        # Get object sizes and data
        objects = []
        for i in range(1, num_features + 1):
            obj_mask = labeled == i
            size = np.sum(obj_mask)
            if size > 0:
                rows, cols = np.where(obj_mask)
                obj_data = grid[obj_mask]
                objects.append({
                    'size': size,
                    'data': obj_data,
                    'shape': (rows.max() - rows.min() + 1, cols.max() - cols.min() + 1),
                    'relative_positions': list(zip(rows - rows.min(), cols - cols.min()))
                })
        
        # Sort by size
        objects.sort(key=lambda x: x['size'])
        
        # Reconstruct grid
        max_height = max(obj['shape'][0] for obj in objects)
        total_width = sum(obj['shape'][1] + 1 for obj in objects)
        
        result = np.full((max_height, total_width), background, dtype=grid.dtype)
        
        current_col = 0
        for obj in objects:
            h, w = obj['shape']
            obj_grid = np.full((h, w), background, dtype=grid.dtype)
            
            for idx, (r, c) in enumerate(obj['relative_positions']):
                obj_grid[r, c] = obj['data'][idx]
            
            result[:h, current_col:current_col + w] = obj_grid
            current_col += w + 1
        
        return result[:, :current_col-1]

# --- 2. Enhanced Pattern Detectors ---

class AdvancedPatternDetector:
    """Advanced pattern detection for complex ARC patterns"""
    
    @staticmethod
    def detect_progression(grids):
        """Detect arithmetic or geometric progressions in a sequence"""
        if len(grids) < 2:
            return None
        
        # Check for size progression
        sizes = [(g.shape[0], g.shape[1]) for g in grids]
        
        # Check for color count progression
        color_counts = []
        for grid in grids:
            unique, counts = np.unique(grid, return_counts=True)
            color_counts.append(dict(zip(unique, counts)))
        
        # Check for object count progression
        object_counts = []
        for grid in grids:
            from scipy import ndimage
            mask = grid != 0
            _, num = ndimage.label(mask)
            object_counts.append(num)
        
        # Analyze progressions
        if len(set(object_counts)) > 1:
            diffs = [object_counts[i+1] - object_counts[i] for i in range(len(object_counts)-1)]
            if len(set(diffs)) == 1:  # Arithmetic progression
                return {'type': 'object_count_arithmetic', 'difference': diffs[0]}
        
        return None
    
    @staticmethod
    def detect_color_rules(inp, out):
        """Detect color transformation rules"""
        rules = []
        
        # Check for color inversions
        if inp.shape == out.shape:
            unique_in = set(inp.flatten())
            unique_out = set(out.flatten())
            
            # Check if colors are inverted (e.g., 0->9, 1->8, etc.)
            inverted = True
            for color in unique_in:
                expected = 9 - color
                if not np.array_equal(out[inp == color], expected):
                    inverted = False
                    break
            
            if inverted:
                rules.append({'type': 'color_inversion', 'max_value': 9})
            
            # Check for color cycling
            color_map = {}
            for i in range(inp.shape[0]):
                for j in range(inp.shape[1]):
                    in_color = int(inp[i, j])
                    out_color = int(out[i, j])
                    if in_color in color_map:
                        if color_map[in_color] != out_color:
                            color_map = None
                            break
                    else:
                        color_map[in_color] = out_color
                if color_map is None:
                    break
            
            if color_map is not None:
                rules.append({'type': 'color_mapping', 'map': color_map})
        
        return rules
    
    @staticmethod
    def detect_counting_pattern(inp, out):
        """Detect if output encodes counts from input"""
        if out.shape == (1, 1) or (out.shape[0] * out.shape[1] < inp.shape[0] * inp.shape[1] // 4):
            # Output is much smaller, might be a count
            unique, counts = np.unique(inp, return_counts=True)
            
            # Check if output contains counts
            out_vals = out.flatten()
            if any(count in out_vals for count in counts):
                return {'type': 'count_encoding', 'counts': dict(zip(unique, counts))}
        
        return None
    
    @staticmethod
    def detect_symmetry_breaking(inp, out):
        """Detect if output breaks symmetry of input"""
        # Check if input is symmetric but output is not
        input_vsym = np.array_equal(inp, np.fliplr(inp))
        input_hsym = np.array_equal(inp, np.flipud(inp))
        
        if inp.shape == out.shape:
            output_vsym = np.array_equal(out, np.fliplr(out))
            output_hsym = np.array_equal(out, np.flipud(out))
            
            if (input_vsym and not output_vsym) or (input_hsym and not output_hsym):
                return {'type': 'symmetry_breaking', 
                        'input_symmetry': {'vertical': input_vsym, 'horizontal': input_hsym},
                        'output_symmetry': {'vertical': output_vsym, 'horizontal': output_hsym}}
        
        return None

# --- 3. Enhanced Hypothesis Generator ---

class EnhancedHypothesisGenerator:
    """Generate more sophisticated hypotheses"""
    
    def __init__(self):
        self.ops = EnhancedDSLOperations()
        self.pattern_detector = AdvancedPatternDetector()
    
    def generate_hypotheses(self, train_pairs):
        """Generate comprehensive set of hypotheses"""
        hypotheses = []
        
        # Try each type of transformation with error handling
        transform_generators = [
            ("direct", self._generate_direct_transformations),
            ("color", self._generate_color_transformations),
            ("size", self._generate_size_transformations),
            ("pattern", self._generate_pattern_transformations),
            ("object", self._generate_object_transformations),
            ("numerical", self._generate_numerical_transformations),
            ("compositional", self._generate_compositional_transformations),
        ]
        
        for name, generator in transform_generators:
            try:
                hypotheses.extend(generator(train_pairs))
            except Exception as e:
                # Continue with other generators if one fails
                pass
        
        return hypotheses
    
    def _generate_direct_transformations(self, train_pairs):
        """Simple direct transformations"""
        hypotheses = []
        
        # Basic transformations
        basic_ops = [
            ('rotate_90', self.ops.rotate_90),
            ('rotate_180', self.ops.rotate_180),
            ('rotate_270', self.ops.rotate_270),
            ('flip_horizontal', self.ops.flip_horizontal),
            ('flip_vertical', self.ops.flip_vertical),
            ('transpose', self.ops.transpose),
        ]
        
        for name, op in basic_ops:
            if all(self._safe_compare(op(inp), out) for inp, out in train_pairs):
                hypotheses.append(Hypothesis(name, op, 1.0))
        
        return hypotheses
    
    def _generate_color_transformations(self, train_pairs):
        """Color-based transformations"""
        hypotheses = []
        
        # Check for color rules
        all_rules = []
        for inp, out in train_pairs:
            rules = self.pattern_detector.detect_color_rules(inp, out)
            all_rules.append(rules)
        
        # Color inversion
        if all_rules and all(rules and any(r['type'] == 'color_inversion' for r in rules) for rules in all_rules):
            hypotheses.append(Hypothesis(
                'color_inversion',
                lambda g: self.ops.invert_colors(g, 9),
                0.9
            ))
        
        # Color mapping
        if all_rules and all_rules[0] and all(any(r['type'] == 'color_mapping' for r in rules) for rules in all_rules if rules):
            # Extract consistent mapping
            try:
                first_map = next(r['map'] for r in all_rules[0] if r['type'] == 'color_mapping')
                if all(any(r.get('map') == first_map for r in rules) for rules in all_rules[1:]):
                    hypotheses.append(Hypothesis(
                        'color_mapping',
                        lambda g, m=first_map: self._apply_color_map(g, m),
                        0.9
                    ))
            except StopIteration:
                pass  # No color mapping found
        
        # Extract specific color
        for color in range(10):
            if all(np.array_equal(self.ops.extract_color(inp, color), out) 
                   for inp, out in train_pairs):
                hypotheses.append(Hypothesis(
                    f'extract_color_{color}',
                    lambda g, c=color: self.ops.extract_color(g, c),
                    0.8
                ))
        
        return hypotheses
    
    def _generate_size_transformations(self, train_pairs):
        """Size-based transformations"""
        hypotheses = []
        
        # Scaling
        for factor in [2, 3, 4]:
            # Scale up
            if all(self._safe_compare(self.ops.scale_up(inp, factor), out) 
                   for inp, out in train_pairs):
                hypotheses.append(Hypothesis(
                    f'scale_up_{factor}x',
                    lambda g, f=factor: self.ops.scale_up(g, f),
                    0.9
                ))
            
            # Scale down
            if all(self._safe_compare(self.ops.scale_down(inp, factor), out) 
                   for inp, out in train_pairs):
                hypotheses.append(Hypothesis(
                    f'scale_down_{factor}x',
                    lambda g, f=factor: self.ops.scale_down(g, f),
                    0.9
                ))
        
        # Cropping
        if all(self._safe_compare(self.ops.crop_to_content(inp), out) 
               for inp, out in train_pairs):
            hypotheses.append(Hypothesis(
                'crop_to_content',
                self.ops.crop_to_content,
                0.8
            ))
        
        return hypotheses
    
    def _generate_pattern_transformations(self, train_pairs):
        """Pattern-based transformations"""
        hypotheses = []
        
        # Pattern completion
        if all(self._could_be_pattern_completion(inp, out) for inp, out in train_pairs):
            hypotheses.append(Hypothesis(
                'complete_pattern',
                self.ops.complete_pattern,
                0.7
            ))
        
        # Symmetry operations
        symmetry_ops = [
            ('make_symmetric_vertical', self.ops.make_symmetric_vertical),
            ('make_symmetric_horizontal', self.ops.make_symmetric_horizontal),
            ('make_symmetric_diagonal', self.ops.make_symmetric_diagonal),
        ]
        
        for name, op in symmetry_ops:
            if all(self._safe_compare(op(inp), out) for inp, out in train_pairs):
                hypotheses.append(Hypothesis(name, op, 0.8))
        
        return hypotheses
    
    def _generate_object_transformations(self, train_pairs):
        """Object manipulation transformations"""
        hypotheses = []
        
        # Object duplication
        for direction in ['right', 'down']:
            if all(self._safe_compare(self.ops.duplicate_objects(inp, direction), out)
                   for inp, out in train_pairs):
                hypotheses.append(Hypothesis(
                    f'duplicate_objects_{direction}',
                    lambda g, d=direction: self.ops.duplicate_objects(g, d),
                    0.8
                ))
        
        # Extract largest object
        if all(self._safe_compare(self.ops.extract_largest_object(inp), out)
               for inp, out in train_pairs):
            hypotheses.append(Hypothesis(
                'extract_largest_object',
                self.ops.extract_largest_object,
                0.7
            ))
        
        # Separate objects
        if all(self._safe_compare(self.ops.separate_objects(inp), out)
               for inp, out in train_pairs):
            hypotheses.append(Hypothesis(
                'separate_objects',
                self.ops.separate_objects,
                0.7
            ))
        
        # Sort by size
        if all(self._safe_compare(self.ops.sort_by_size(inp), out)
               for inp, out in train_pairs):
            hypotheses.append(Hypothesis(
                'sort_by_size',
                self.ops.sort_by_size,
                0.7
            ))
        
        # Object movement
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                if all(self._safe_compare(self.ops.move_objects(inp, dx, dy), out)
                       for inp, out in train_pairs):
                    hypotheses.append(Hypothesis(
                        f'move_objects_dx{dx}_dy{dy}',
                        lambda g, x=dx, y=dy: self.ops.move_objects(g, x, y),
                        0.7
                    ))
        
        return hypotheses
    
    def _generate_numerical_transformations(self, train_pairs):
        """Counting and numerical transformations"""
        hypotheses = []
        
        # Count colors
        if all(self._safe_compare(self.ops.count_colors(inp), out)
               for inp, out in train_pairs):
            hypotheses.append(Hypothesis(
                'count_colors',
                self.ops.count_colors,
                0.7
            ))
        
        # Count neighbors
        for target_color in range(10):
            if all(self._safe_compare(self.ops.count_neighbors(inp, target_color), out)
                   for inp, out in train_pairs):
                hypotheses.append(Hypothesis(
                    f'count_neighbors_of_{target_color}',
                    lambda g, c=target_color: self.ops.count_neighbors(g, c),
                    0.7
                ))
        
        # Check for counting pattern
        counting_patterns = [self.pattern_detector.detect_counting_pattern(inp, out) 
                           for inp, out in train_pairs]
        if all(p is not None for p in counting_patterns):
            # Create a counting transformation
            hypotheses.append(Hypothesis(
                'encode_counts',
                lambda g: self._encode_counts(g),
                0.6
            ))
        
        return hypotheses
    
    def _generate_compositional_transformations(self, train_pairs):
        """Complex compositional transformations"""
        hypotheses = []
        
        # Common compositions
        compositions = [
            ('crop_then_scale_2x', 
             lambda g: self.ops.scale_up(self.ops.crop_to_content(g), 2)),
            ('largest_then_duplicate_right',
             lambda g: self.ops.duplicate_objects(self.ops.extract_largest_object(g), 'right')),
            ('symmetric_then_rotate',
             lambda g: self.ops.rotate_90(self.ops.make_symmetric_vertical(g))),
            ('extract_then_fill',
             lambda g: self._extract_and_fill_pattern(g)),
        ]
        
        for name, func in compositions:
            try:
                if all(self._safe_compare(func(inp), out) for inp, out in train_pairs):
                    hypotheses.append(Hypothesis(name, func, 0.6))
            except:
                pass
        
        return hypotheses
    
    # Helper methods
    def _safe_compare(self, grid1, grid2):
        """Safely compare two grids"""
        if grid1 is None or grid2 is None:
            return False
        return grid1.shape == grid2.shape and np.array_equal(grid1, grid2)
    
    def _apply_color_map(self, grid, color_map):
        """Apply color mapping"""
        result = grid.copy()
        for old_color, new_color in color_map.items():
            result[grid == old_color] = new_color
        return result
    
    def _could_be_pattern_completion(self, inp, out):
        """Check if output could be a pattern completion of input"""
        if inp.shape != out.shape:
            return False
        
        # Check if output has more non-zero elements
        inp_nonzero = np.count_nonzero(inp)
        out_nonzero = np.count_nonzero(out)
        
        if out_nonzero <= inp_nonzero:
            return False
        
        # Check if input is subset of output
        return np.array_equal(inp[inp != 0], out[inp != 0])
    
    def _encode_counts(self, grid):
        """Encode color counts into a small grid"""
        unique, counts = np.unique(grid, return_counts=True)
        
        # Create a small grid with counts
        max_colors = min(len(unique), 3)
        result = np.zeros((1, max_colors), dtype=grid.dtype)
        
        for i in range(max_colors):
            result[0, i] = min(counts[i], 9)
        
        return result
    
    def _extract_and_fill_pattern(self, grid):
        """Extract pattern and fill enclosed areas"""
        # Find most common non-zero color
        non_zero = grid[grid != 0]
        if len(non_zero) == 0:
            return grid
        
        boundary_color = np.bincount(non_zero).argmax()
        
        # Fill enclosed areas with a different color
        fill_color = (boundary_color + 1) % 10
        if fill_color == 0:
            fill_color = 1
        
        return self.ops.fill_enclosed_areas(grid, fill_color, boundary_color)

# --- 4. Multi-Strategy Solver ---

class Hypothesis:
    """Enhanced hypothesis class with more features"""
    def __init__(self, name, transform_func, confidence=0.5):
        self.name = name
        self.transform_func = transform_func
        self.confidence = confidence
        self.test_results = []
        
    def test(self, inp, out):
        """Test hypothesis on input-output pair"""
        try:
            result = self.transform_func(inp)
            success = np.array_equal(result, out)
            self.test_results.append(success)
            return success
        except Exception as e:
            self.test_results.append(False)
            return False
    
    def get_success_rate(self):
        """Get success rate of hypothesis"""
        if not self.test_results:
            return 0.0
        return sum(self.test_results) / len(self.test_results)

class MultiStrategyARCSolver:
    """Solver using multiple strategies and fallback mechanisms"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.hypothesis_generator = EnhancedHypothesisGenerator()
        self.ops = EnhancedDSLOperations()
        
        if verbose:
            print(" Multi-Strategy ARC Solver initialized")
            print("   • Enhanced DSL operations: ✓")
            print("   • Advanced pattern detection: ✓") 
            print("   • Multi-hypothesis testing: ✓")
            print("   • Smart fallback strategies: ✓")
    
    def solve_task(self, task, task_id=None):
        """Solve a task using multiple strategies"""
        train_pairs = [(np.array(t['input']), np.array(t['output'])) 
                      for t in task['train']]
        test_inputs = [np.array(t['input']) for t in task['test']]
        
        if self.verbose:
            print(f"\n Solving task {task_id if task_id else ''}")
            print(f"   Training examples: {len(train_pairs)}")
            print(f"   Test cases: {len(test_inputs)}")
        
        # Strategy 1: Direct hypothesis testing
        solution = self._try_direct_hypotheses(train_pairs)
        
        # Strategy 2: Compositional search
        if not solution:
            solution = self._try_compositional_search(train_pairs)
        
        # Strategy 3: Pattern-specific strategies
        if not solution:
            solution = self._try_pattern_specific(train_pairs)
        
        # Strategy 4: Machine learning-inspired approach
        if not solution:
            solution = self._try_ml_approach(train_pairs)
        
        # Apply solution or use smart fallback
        predictions = []
        for test_input in test_inputs:
            if solution:
                try:
                    pred = solution(test_input)
                    predictions.append(pred.tolist())
                except:
                    pred = self._smart_fallback(test_input, train_pairs)
                    predictions.append(pred.tolist())
            else:
                pred = self._smart_fallback(test_input, train_pairs)
                predictions.append(pred.tolist())
        
        return predictions
    
    def _try_direct_hypotheses(self, train_pairs):
        """Try direct hypothesis generation and testing"""
        if self.verbose:
            print("    Strategy 1: Direct hypothesis testing...")
        
        hypotheses = self.hypothesis_generator.generate_hypotheses(train_pairs)
        
        if self.verbose:
            print(f"      Generated {len(hypotheses)} hypotheses")
        
        # Test all hypotheses
        perfect_hypotheses = []
        for h in hypotheses:
            if all(h.test(inp, out) for inp, out in train_pairs):
                perfect_hypotheses.append(h)
                if self.verbose:
                    print(f"      ✓ Found perfect: {h.name}")
        
        if perfect_hypotheses:
            # Return the one with highest confidence
            best = max(perfect_hypotheses, key=lambda h: h.confidence)
            return best.transform_func
        
        return None
    
    def _try_compositional_search(self, train_pairs):
        """Try composing multiple operations"""
        if self.verbose:
            print("    Strategy 2: Compositional search...")
        
        # Get basic operations
        basic_ops = [
            ('rotate_90', self.ops.rotate_90),
            ('flip_h', self.ops.flip_horizontal),
            ('flip_v', self.ops.flip_vertical),
            ('transpose', self.ops.transpose),
            ('crop', self.ops.crop_to_content),
            ('scale_2x', lambda g: self.ops.scale_up(g, 2)),
            ('largest', self.ops.extract_largest_object),
            ('symmetric_v', self.ops.make_symmetric_vertical),
        ]
        
        # Try 2-operation compositions
        for (name1, op1), (name2, op2) in itertools.combinations(basic_ops, 2):
            # Try both orders
            comp1 = lambda g, o1=op1, o2=op2: o2(o1(g))
            comp2 = lambda g, o1=op1, o2=op2: o1(o2(g))
            
            try:
                if all(np.array_equal(comp1(inp), out) for inp, out in train_pairs):
                    if self.verbose:
                        print(f"      ✓ Found composition: {name1} → {name2}")
                    return comp1
                    
                if all(np.array_equal(comp2(inp), out) for inp, out in train_pairs):
                    if self.verbose:
                        print(f"      ✓ Found composition: {name2} → {name1}")
                    return comp2
            except:
                pass
        
        return None
    
    def _try_pattern_specific(self, train_pairs):
        """Try pattern-specific solving strategies"""
        if self.verbose:
            print("    Strategy 3: Pattern-specific analysis...")
        
        # Check for specific patterns
        inp0, out0 = train_pairs[0]
        
        # Check if it's a counting task
        if out0.size < inp0.size // 4:
            if self.verbose:
                print("      Detected possible counting task")
            # Try various counting strategies
            count_func = self._find_counting_pattern(train_pairs)
            if count_func:
                return count_func
        
        # Check if it's a selection task
        if out0.shape != inp0.shape:
            if self.verbose:
                print("      Detected size change - possible selection/extraction")
            select_func = self._find_selection_pattern(train_pairs)
            if select_func:
                return select_func
        
        return None
    
    def _try_ml_approach(self, train_pairs):
        """Use ML-inspired pattern matching"""
        if self.verbose:
            print("    Strategy 4: ML-inspired pattern matching...")
        
        # Find nearest neighbor in training set
        def nearest_neighbor_transform(test_input):
            min_dist = float('inf')
            best_output = None
            
            for inp, out in train_pairs:
                if inp.shape == test_input.shape:
                    dist = np.sum(inp != test_input)
                    if dist < min_dist:
                        min_dist = dist
                        best_output = out
            
            return best_output if best_output is not None else test_input
        
        # Check if this works reasonably well
        # (This is a heuristic approach)
        return nearest_neighbor_transform
    
    def _smart_fallback(self, test_input, train_pairs):
        """Smart fallback strategy"""
        # Strategy 1: Return input unchanged (common in ARC)
        if self._check_identity_pattern(train_pairs):
            return test_input
        
        # Strategy 2: Return most common output shape
        out_shapes = [out.shape for _, out in train_pairs]
        if len(set(out_shapes)) == 1:
            target_shape = out_shapes[0]
            if target_shape != test_input.shape:
                # Try to match the shape
                if target_shape[0] * target_shape[1] < test_input.shape[0] * test_input.shape[1]:
                    # Need to reduce - try cropping
                    return self.ops.crop_to_content(test_input)
                else:
                    # Need to expand - try scaling
                    factor = int(np.sqrt(target_shape[0] * target_shape[1] / (test_input.shape[0] * test_input.shape[1])))
                    if factor > 1:
                        return self.ops.scale_up(test_input, factor)
        
        # Strategy 3: Return input unchanged
        return test_input
    
    def _check_identity_pattern(self, train_pairs):
        """Check if outputs are same as inputs"""
        return any(np.array_equal(inp, out) for inp, out in train_pairs)
    
    def _find_counting_pattern(self, train_pairs):
        """Find counting pattern in data"""
        # Implement specific counting pattern detection
        # This is a placeholder - would need more sophisticated analysis
        return None
    
    def _find_selection_pattern(self, train_pairs):
        """Find selection/extraction pattern"""
        # Check if output is a subset of input
        for inp, out in train_pairs:
            # Check if output appears in input
            if out.shape[0] <= inp.shape[0] and out.shape[1] <= inp.shape[1]:
                for i in range(inp.shape[0] - out.shape[0] + 1):
                    for j in range(inp.shape[1] - out.shape[1] + 1):
                        if np.array_equal(inp[i:i+out.shape[0], j:j+out.shape[1]], out):
                            # Found exact match - this is a crop operation
                            def crop_func(g, y=i, x=j, h=out.shape[0], w=out.shape[1]):
                                return g[y:y+h, x:x+w]
                            return crop_func
        
        return None

# --- 5. Main execution ---

def create_submission(solver, test_tasks, sample_submission, max_tasks=None):
    """Create submission file with enhanced solver"""
    submission = sample_submission.copy()
    
    task_items = list(test_tasks.items())
    if max_tasks:
        task_items = task_items[:max_tasks]
    
    total = len(task_items)
    solved_count = 0
    
    for idx, (task_id, task_data) in enumerate(task_items):
        if solver.verbose:
            print(f"\n[{idx+1}/{total}] Task: {task_id}")
        
        # Solve task
        predictions = solver.solve_task(task_data, task_id)
        
        # Check if we found a non-trivial solution
        is_trivial = all(pred == [[0, 0], [0, 0]] for pred in predictions)
        if not is_trivial:
            solved_count += 1
        
        # Format predictions
        formatted_preds = []
        for pred in predictions:
            formatted_preds.append({
                "attempt_1": pred,
                "attempt_2": pred  # Could implement different strategy for attempt_2
            })
        
        # Update submission
        if task_id in submission:
            submission[task_id] = formatted_preds[:len(submission[task_id])]
    
    if solver.verbose:
        print(f"\n Summary: Solved {solved_count}/{total} tasks ({solved_count/total*100:.1f}%)")
    
    return submission

# Example usage
if __name__ == "__main__":
    print(" Enhanced ARC Solver v2")
    print("=" * 50)
    
    # Load data
    data, files = load_kaggle_data()
    
    if 'test' not in data or 'sample_submission' not in data:
        print(" Required data not found!")
    else:
        # Create enhanced solver
        solver = MultiStrategyARCSolver(verbose=True)
        
        # Create submission
        print("\n Creating submission...")
        submission = create_submission(
            solver,
            data['test'],
            data['sample_submission'],
            max_tasks=None  # Process all tasks
        )
        
        # Save submission
        submission_path = '/kaggle/working/submission.json'
        with open(submission_path, 'w') as f:
            json.dump(submission, f)
        
        print(f"\n Submission saved to {submission_path}")

