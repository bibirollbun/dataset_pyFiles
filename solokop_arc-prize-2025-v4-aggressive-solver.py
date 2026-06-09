"""ARC Prize 2025 - Hybrid Solver
Combines program synthesis (induction) with pattern matching (transduction).
Based on 2024 winners' insight: each approach solves different tasks.
"""
import json
import numpy as np
from pathlib import Path
from collections import Counter
from scipy import ndimage
from itertools import product, combinations

# Data loading
try:
    DATA_DIR = Path('/kaggle/input/arc-prize-2025')
    train_challenges = json.load(open(DATA_DIR / 'arc-agi_training_challenges.json'))
    train_solutions = json.load(open(DATA_DIR / 'arc-agi_training_solutions.json'))
    test_challenges = json.load(open(DATA_DIR / 'arc-agi_test_challenges.json'))
    print(f"Kaggle: {len(train_challenges)} train, {len(test_challenges)} test")
except:
    DATA_DIR = Path('../datasets')
    train_challenges = json.load(open(DATA_DIR / 'arc-agi_training_challenges.json'))
    train_solutions = json.load(open(DATA_DIR / 'arc-agi_training_solutions.json'))
    test_challenges = json.load(open(DATA_DIR / 'arc-agi_test_challenges.json'))
    print(f"Local: {len(train_challenges)} train, {len(test_challenges)} test")

# Utils
def g2a(grid): return np.array(grid)
def a2g(arr): return arr.tolist()
def eq(g1, g2):
    try:
        a1, a2 = g2a(g1), g2a(g2)
        return a1.shape == a2.shape and np.array_equal(a1, a2)
    except: return False

# Geometric transforms
class Geo:
    @staticmethod
    def rot90(g): return a2g(np.rot90(g2a(g), k=-1))
    @staticmethod
    def rot180(g): return a2g(np.rot90(g2a(g), k=2))
    @staticmethod
    def rot270(g): return a2g(np.rot90(g2a(g), k=1))
    @staticmethod
    def flip_h(g): return a2g(np.fliplr(g2a(g)))
    @staticmethod
    def flip_v(g): return a2g(np.flipud(g2a(g)))
    @staticmethod
    def transpose(g): return a2g(g2a(g).T)

# Scaling transforms
class Scale:
    @staticmethod
    def up_2x(g):
        a = g2a(g)
        return a2g(np.repeat(np.repeat(a, 2, 0), 2, 1))
    @staticmethod
    def up_3x(g):
        a = g2a(g)
        return a2g(np.repeat(np.repeat(a, 3, 0), 3, 1))
    @staticmethod
    def down_2x(g):
        a = g2a(g)
        if a.shape[0] % 2 == 0 and a.shape[1] % 2 == 0:
            return a2g(a[::2, ::2])
        return g
    @staticmethod
    def down_3x(g):
        a = g2a(g)
        if a.shape[0] % 3 == 0 and a.shape[1] % 3 == 0:
            return a2g(a[::3, ::3])
        return g

# Tiling transforms
class Tile:
    @staticmethod
    def t_2x2(g): return a2g(np.tile(g2a(g), (2, 2)))
    @staticmethod
    def t_3x3(g): return a2g(np.tile(g2a(g), (3, 3)))
    @staticmethod
    def t_2x1(g): return a2g(np.tile(g2a(g), (2, 1)))
    @staticmethod
    def t_1x2(g): return a2g(np.tile(g2a(g), (1, 2)))

# Color transforms
class Color:
    @staticmethod
    def invert(g): return a2g(9 - g2a(g))
    @staticmethod
    def majority(g):
        a = g2a(g).flatten()
        a = a[a != 0]
        if len(a) == 0: return g
        return a2g(np.full_like(g2a(g), Counter(a).most_common(1)[0][0]))
    @staticmethod
    def swap_01(g):
        a = g2a(g).copy()
        a[a == 0] = 10
        a[a == 1] = 0
        a[a == 10] = 1
        return a2g(a)

# Border transforms
class Border:
    @staticmethod
    def remove(g):
        a = g2a(g)
        if a.shape[0] <= 2 or a.shape[1] <= 2: return g
        return a2g(a[1:-1, 1:-1])
    @staticmethod
    def add(g):
        a = g2a(g)
        bordered = np.zeros((a.shape[0]+2, a.shape[1]+2), dtype=a.dtype)
        bordered[1:-1, 1:-1] = a
        return a2g(bordered)
    @staticmethod
    def crop(g):
        a = g2a(g)
        mask = a != 0
        if not mask.any(): return g
        rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
        if not rows.any() or not cols.any(): return g
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return a2g(a[rmin:rmax+1, cmin:cmax+1])

# Gravity transforms
class Gravity:
    @staticmethod
    def down(g):
        a = g2a(g).copy()
        for col in range(a.shape[1]):
            column = a[:, col]
            non_zero = column[column != 0]
            a[:, col] = np.concatenate([np.zeros(len(column)-len(non_zero), dtype=a.dtype), non_zero])
        return a2g(a)
    @staticmethod
    def up(g):
        a = g2a(g).copy()
        for col in range(a.shape[1]):
            column = a[:, col]
            non_zero = column[column != 0]
            a[:, col] = np.concatenate([non_zero, np.zeros(len(column)-len(non_zero), dtype=a.dtype)])
        return a2g(a)
    @staticmethod
    def left(g):
        a = g2a(g).copy()
        for row in range(a.shape[0]):
            row_data = a[row, :]
            non_zero = row_data[row_data != 0]
            a[row, :] = np.concatenate([non_zero, np.zeros(len(row_data)-len(non_zero), dtype=a.dtype)])
        return a2g(a)
    @staticmethod
    def right(g):
        a = g2a(g).copy()
        for row in range(a.shape[0]):
            row_data = a[row, :]
            non_zero = row_data[row_data != 0]
            a[row, :] = np.concatenate([np.zeros(len(row_data)-len(non_zero), dtype=a.dtype), non_zero])
        return a2g(a)

# Object transforms
class Obj:
    @staticmethod
    def extract_largest(g):
        a = g2a(g)
        mask = a != 0
        if not mask.any(): return g
        labeled, num = ndimage.label(mask)
        if num == 0: return g
        sizes = [(labeled == i).sum() for i in range(1, num+1)]
        largest = np.argmax(sizes) + 1
        obj_mask = labeled == largest
        rows, cols = np.where(obj_mask)
        if len(rows) == 0: return g
        result = a[rows.min():rows.max()+1, cols.min():cols.max()+1].copy()
        result[~obj_mask[rows.min():rows.max()+1, cols.min():cols.max()+1]] = 0
        return a2g(result)
    @staticmethod
    def hollow(g):
        a = g2a(g).copy()
        if a.shape[0] <= 2 or a.shape[1] <= 2: return g
        a[1:-1, 1:-1] = 0
        return a2g(a)
    @staticmethod
    def outline(g):
        a = g2a(g)
        mask = a != 0
        if not mask.any(): return g
        dilated = ndimage.binary_dilation(mask)
        result = a.copy()
        result[dilated & ~mask] = 1
        return a2g(result)
    @staticmethod
    def fill_holes(g):
        a = g2a(g)
        mask = a != 0
        if not mask.any(): return g
        filled = ndimage.binary_fill_holes(mask)
        result = a.copy()
        result[filled & ~mask] = 1
        return a2g(result)
    @staticmethod
    def count_objects(g):
        a = g2a(g)
        mask = a != 0
        if not mask.any(): return g
        labeled, num = ndimage.label(mask)
        result = np.full_like(a, num)
        return a2g(result)

# Pattern transforms
class Pattern:
    @staticmethod
    def mirror_h(g):
        """Mirror horizontally (left half to right)."""
        a = g2a(g)
        h, w = a.shape
        if w % 2 != 0: return g
        left = a[:, :w//2]
        result = np.concatenate([left, np.fliplr(left)], axis=1)
        return a2g(result)
    @staticmethod
    def mirror_v(g):
        """Mirror vertically (top half to bottom)."""
        a = g2a(g)
        h, w = a.shape
        if h % 2 != 0: return g
        top = a[:h//2, :]
        result = np.concatenate([top, np.flipud(top)], axis=0)
        return a2g(result)
    @staticmethod
    def extend_h(g):
        """Extend pattern horizontally."""
        a = g2a(g)
        return a2g(np.tile(a, (1, 2)))
    @staticmethod
    def extend_v(g):
        """Extend pattern vertically."""
        a = g2a(g)
        return a2g(np.tile(a, (2, 1)))
    @staticmethod
    def compress_h(g):
        """Compress horizontally (take every other column)."""
        a = g2a(g)
        if a.shape[1] < 2: return g
        return a2g(a[:, ::2])
    @staticmethod
    def compress_v(g):
        """Compress vertically (take every other row)."""
        a = g2a(g)
        if a.shape[0] < 2: return g
        return a2g(a[::2, :])

# Grid operations
class Grid:
    @staticmethod
    def overlay(g):
        """Overlay all non-zero values."""
        a = g2a(g)
        result = np.zeros_like(a)
        for i in range(1, 10):
            result[a == i] = i
        return a2g(result)
    @staticmethod
    def mask_nonzero(g):
        """Convert all non-zero to 1."""
        a = g2a(g)
        result = (a != 0).astype(a.dtype)
        return a2g(result)
    @staticmethod
    def isolate_color(g, color=1):
        """Keep only specific color."""
        a = g2a(g)
        result = np.where(a == color, a, 0)
        return a2g(result)
    @staticmethod
    def remove_color(g, color=0):
        """Remove specific color."""
        a = g2a(g)
        result = np.where(a == color, 0, a)
        return a2g(result)

# Transduction: Direct pattern matching
class Transduction:
    @staticmethod
    def detect_color_map(task):
        """Detect consistent color mapping."""
        mappings = []
        for ex in task['train']:
            inp, out = g2a(ex['input']), g2a(ex['output'])
            if inp.shape == out.shape:
                mapping = {}
                for i in range(10):
                    inp_mask = inp == i
                    if inp_mask.any():
                        out_colors = out[inp_mask]
                        if len(set(out_colors)) == 1:
                            mapping[i] = out_colors[0]
                if mapping:
                    mappings.append(mapping)
        if len(mappings) > 0 and all(m == mappings[0] for m in mappings):
            return mappings[0]
        return None
    
    @staticmethod
    def apply_color_map(grid, mapping):
        result = g2a(grid).copy()
        for old_c, new_c in mapping.items():
            result[result == old_c] = new_c
        return a2g(result)
    
    @staticmethod
    def solve(task):
        """Try direct pattern matching."""
        color_map = Transduction.detect_color_map(task)
        if color_map:
            try:
                result = Transduction.apply_color_map(task['test'][0]['input'], color_map)
                if all(eq(Transduction.apply_color_map(ex['input'], color_map), ex['output']) for ex in task['train']):
                    return result
            except: pass
        return None

# Induction: Program synthesis
class Induction:
    def __init__(self):
        # All atomic transforms
        self.transforms = {
            # Geometric
            'rot90': Geo.rot90, 'rot180': Geo.rot180, 'rot270': Geo.rot270,
            'flip_h': Geo.flip_h, 'flip_v': Geo.flip_v, 'transpose': Geo.transpose,
            # Scaling
            'up_2x': Scale.up_2x, 'up_3x': Scale.up_3x,
            'down_2x': Scale.down_2x, 'down_3x': Scale.down_3x,
            # Tiling
            't_2x2': Tile.t_2x2, 't_3x3': Tile.t_3x3, 't_2x1': Tile.t_2x1, 't_1x2': Tile.t_1x2,
            # Color
            'invert': Color.invert, 'majority': Color.majority, 'swap_01': Color.swap_01,
            # Border
            'remove': Border.remove, 'add': Border.add, 'crop': Border.crop,
            # Gravity
            'g_down': Gravity.down, 'g_up': Gravity.up, 'g_left': Gravity.left, 'g_right': Gravity.right,
            # Object
            'largest': Obj.extract_largest, 'hollow': Obj.hollow, 'outline': Obj.outline,
            'fill_holes': Obj.fill_holes, 'count_obj': Obj.count_objects,
            # Pattern
            'mirror_h': Pattern.mirror_h, 'mirror_v': Pattern.mirror_v,
            'extend_h': Pattern.extend_h, 'extend_v': Pattern.extend_v,
            'compress_h': Pattern.compress_h, 'compress_v': Pattern.compress_v,
            # Grid
            'overlay': Grid.overlay, 'mask_nz': Grid.mask_nonzero,
        }
        print(f"Induction: {len(self.transforms)} transforms")
    
    def try_transform(self, transform, task):
        try:
            if all(eq(transform(ex['input']), ex['output']) for ex in task['train']):
                return transform(task['test'][0]['input'])
        except: pass
        return None
    
    def solve(self, task):
        """Try program synthesis."""
        # Single transforms
        for name, transform in self.transforms.items():
            result = self.try_transform(transform, task)
            if result is not None: return result

        # 2-step compositions (geometric)
        geo = ['rot90', 'rot180', 'rot270', 'flip_h', 'flip_v', 'transpose']
        for op1_name in geo:
            for op2_name in geo:
                try:
                    op1, op2 = self.transforms[op1_name], self.transforms[op2_name]
                    composed = lambda g, o1=op1, o2=op2: o2(o1(g))
                    result = self.try_transform(composed, task)
                    if result is not None: return result
                except: continue

        # Border + geo
        for border in ['remove', 'crop']:
            for geo_op in geo:
                try:
                    b, g = self.transforms[border], self.transforms[geo_op]
                    composed = lambda grid, bb=b, gg=g: gg(bb(grid))
                    result = self.try_transform(composed, task)
                    if result is not None: return result
                except: continue

        # Scale + geo
        for scale in ['up_2x', 'up_3x', 'down_2x', 'down_3x']:
            for geo_op in geo:
                try:
                    s, g = self.transforms[scale], self.transforms[geo_op]
                    composed = lambda grid, ss=s, gg=g: gg(ss(grid))
                    result = self.try_transform(composed, task)
                    if result is not None: return result
                except: continue

        # Pattern + geo
        for pattern in ['mirror_h', 'mirror_v', 'extend_h', 'extend_v']:
            for geo_op in geo:
                try:
                    p, g = self.transforms[pattern], self.transforms[geo_op]
                    composed = lambda grid, pp=p, gg=g: gg(pp(grid))
                    result = self.try_transform(composed, task)
                    if result is not None: return result
                except: continue

        # 3-step: Border + Scale + Geo
        for border in ['remove', 'crop']:
            for scale in ['up_2x', 'down_2x']:
                for geo_op in ['rot90', 'flip_h', 'flip_v']:
                    try:
                        b, s, g = self.transforms[border], self.transforms[scale], self.transforms[geo_op]
                        composed = lambda grid, bb=b, ss=s, gg=g: gg(ss(bb(grid)))
                        result = self.try_transform(composed, task)
                        if result is not None: return result
                    except: continue

        return None

# Hybrid solver: Combine both approaches
class HybridSolver:
    def __init__(self):
        self.transduction = Transduction()
        self.induction = Induction()
    
    def solve(self, task):
        # Try transduction first (faster)
        result = self.transduction.solve(task)
        if result is not None:
            return result
        
        # Try induction (program synthesis)
        result = self.induction.solve(task)
        if result is not None:
            return result
        
        # Fallback
        return task['test'][0]['input']

# Main
solver = HybridSolver()

# Validate
print("\nValidating...")
correct = 0
for task_id in list(train_challenges.keys())[:100]:
    task = train_challenges[task_id]
    solution = train_solutions[task_id]
    prediction = solver.solve(task)
    if eq(prediction, solution[0]):
        correct += 1
print(f"Validation: {correct}% on 100 samples")

# Generate submission
print("\nGenerating submission...")
submission = {}
for task_id, task in test_challenges.items():
    test_predictions = []
    for test_case in task['test']:
        mini_task = {'train': task['train'], 'test': [test_case]}
        prediction = solver.solve(mini_task)
        test_predictions.append({'attempt_1': prediction, 'attempt_2': prediction})
    submission[task_id] = test_predictions

with open('submission.json', 'w') as f:
    json.dump(submission, f)

print(f"Done: {len(submission)} tasks")
print(f"Size: {Path('submission.json').stat().st_size / 1024:.1f} KB")


