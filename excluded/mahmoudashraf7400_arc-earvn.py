
# Import libraries 
import json
import numpy as np
from collections import Counter
from itertools import product
import random
import copy
print("Imports ready!")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




# CELL 1: FULL FIXED CODE – YOUR EARVN + SUBMISSION FIX
import json
import numpy as np
from collections import deque
import os

TEST_DATA_PATH = "/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json"

class Smart_Object:
    def __init__(self, color, pixels):
        self.color = color
        self.pixels = pixels
        self.area = len(pixels)
        self.bbox = self._calculate_bbox()
        self.height = self.bbox[2] - self.bbox[0] + 1
        self.width = self.bbox[3] - self.bbox[1] + 1
        self.is_filled_rectangle = self.area == (self.height * self.width)
        self.is_square = self.height == self.width and self.is_filled_rectangle
        self.centroid = self._calculate_centroid()
   
    def _calculate_bbox(self):
        if not self.pixels: return (0, 0, 0, 0)
        rows = [p[0] for p in self.pixels]
        cols = [p[1] for p in self.pixels]
        return (min(rows), min(cols), max(rows), max(cols))
       
    def _calculate_centroid(self):
        if not self.pixels: return (0, 0)
        rows = [p[0] for p in self.pixels]
        cols = [p[1] for p in self.pixels]
        return (round(sum(rows) / self.area), round(sum(cols) / self.area))
   
    def to_grid(self, grid_shape):
        new_grid = np.zeros(grid_shape, dtype=int)
        for r, c in self.pixels:
            if r < grid_shape[0] and c < grid_shape[1]:
                 new_grid[r, c] = self.color
        return new_grid.tolist()

class ARCSolver:
    def __init__(self, file_path):
        self.file_path = file_path
        self.tasks = {}
        self.submission = {}

    def load_tasks(self):
        try:
            if os.path.isfile(self.file_path):
                with open(self.file_path, 'r') as f:
                    self.tasks = json.load(f)
                print(f"Loaded {len(self.tasks)} test tasks")
                return True
            else:
                print(f"File not found: {self.file_path}")
                return False
        except Exception as e:
            print(f"Load error: {e}")
            return False

    # === HELPER: Make grid ARC-legal ===
    def make_legal(self, grid):
        if not grid or not isinstance(grid, list) or not grid[0]:
            return [[0]]
        grid = [[int(max(0, min(9, c))) for c in row] for row in grid]
        if not grid: return [[0]]
        max_len = max(len(row) for row in grid)
        return [row + [0] * (max_len - len(row)) for row in grid]

    # --- TRANSFORMATIONS (unchanged) ---
    def _rotate_90(self, grid): return np.rot90(grid, k=3).tolist()
    def _flip_vertical(self, grid): return np.flipud(grid).tolist()
    def _color_swap(self, grid, a, b):
        g = grid.copy()
        g[grid == a] = -1
        g[grid == b] = a
        g[grid == -1] = b
        return g.tolist()
    def _recolor_all_non_background_to_9(self, grid):
        g = grid.copy()
        g[g != 0] = 9
        return g.tolist()
    def _trim_borders(self, grid):
        if grid.shape[0] > 2 and grid.shape[1] > 2:
            return grid[1:-1, 1:-1].tolist()
        return grid.tolist()
    def _invert_colors(self, grid): return (9 - grid).tolist()
    def _crop_to_content(self, grid):
        rows = np.any(grid, axis=1)
        cols = np.any(grid, axis=0)
        if not np.any(rows) or not np.any(cols): return grid.tolist()
        r_idx = np.where(rows)[0]
        c_idx = np.where(cols)[0]
        return grid[r_idx[0]:r_idx[-1]+1, c_idx[0]:c_idx[-1]+1].tolist()
    def _scale_grid(self, grid, factor):
        if factor <= 0: return grid.tolist()
        return np.repeat(np.repeat(grid, factor, axis=0), factor, axis=1).tolist()

    # --- OBJECT-BASED (unchanged) ---
    def _get_connected_components(self, grid):
        rows, cols = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        objects = []
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != 0 and not visited[r, c]:
                    color = grid[r, c]
                    pixels = []
                    queue = deque([(r, c)])
                    visited[r, c] = True
                    while queue:
                        cr, cc = queue.popleft()
                        pixels.append((cr, cc))
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and grid[nr, nc] == color:
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                    if pixels:
                        objects.append(Smart_Object(color, pixels))
        return objects

    def _keep_largest_component(self, grid):
        objs = self._get_connected_components(grid)
        if not objs: return grid.tolist()
        largest = max(objs, key=lambda o: o.area)
        return largest.to_grid(grid.shape)

    def _filter_by_size_one(self, grid):
        objs = self._get_connected_components(grid)
        new_grid = np.zeros_like(grid, dtype=int)
        for obj in objs:
            if obj.area > 1:
                for r, c in obj.pixels:
                    new_grid[r, c] = obj.color
        return new_grid.tolist()

    def _fill_holes(self, grid, fill_color=9):
        rows, cols = grid.shape
        padded = np.pad(grid, 1, constant_values=0)
        exterior = np.zeros_like(padded, dtype=bool)
        queue = deque([(0,0)])
        exterior[0,0] = True
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows+2 and 0 <= nc < cols+2 and padded[nr,nc] == 0 and not exterior[nr,nc]:
                    exterior[nr,nc] = True
                    queue.append((nr,nc))
        hole_mask = (grid == 0) & (~exterior[1:-1,1:-1])
        result = grid.copy()
        result[hole_mask] = fill_color
        return result.tolist()

    def _move_smallest_to_largest_center(self, grid):
        objs = self._get_connected_components(grid)
        if len(objs) < 2: return grid.tolist()
        largest = max(objs, key=lambda o: o.area)
        smallest = min(objs, key=lambda o: o.area)
        if largest.pixels == smallest.pixels: return grid.tolist()
        dr = largest.centroid[0] - smallest.centroid[1]
        dc = largest.centroid[1] - smallest.centroid[1]
        result = grid.copy()
        for r, c in smallest.pixels:
            result[r, c] = 0
        for r, c in smallest.pixels:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                result[nr, nc] = smallest.color
        return result.tolist()

    def _intersect_largest_two_objects(self, grid):
        objs = self._get_connected_components(grid)
        if len(objs) < 2: return grid.tolist()
        o1, o2 = sorted(objs, key=lambda o: o.area, reverse=True)[:2]
        inter = set(o1.pixels) & set(o2.pixels)
        result = np.zeros_like(grid, dtype=int)
        for r, c in inter:
            result[r, c] = o1.color
        return result.tolist()

    def _subtract_smallest_from_largest(self, grid):
        objs = self._get_connected_components(grid)
        if len(objs) < 2: return grid.tolist()
        largest = max(objs, key=lambda o: o.area)
        smallest = min(objs, key=lambda o: o.area)
        diff = set(largest.pixels) - set(smallest.pixels)
        result = np.zeros_like(grid, dtype=int)
        for r, c in diff:
            result[r, c] = largest.color
        return result.tolist()

    def _tile_largest_component(self, grid):
        objs = self._get_connected_components(grid)
        if not objs: return grid.tolist()
        obj = max(objs, key=lambda o: o.area)
        min_r, min_c, max_r, max_c = obj.bbox
        pattern = grid[min_r:max_r+1, min_c:max_c+1]
        h, w = pattern.shape
        gh, gw = grid.shape
        if h == 0 or w == 0: return grid.tolist()
        result = np.zeros_like(grid, dtype=int)
        for r in range(gh):
            for c in range(gw):
                result[r, c] = pattern[r % h, c % w]
        return result.tolist()

    def _test_rule(self, task, rule_func):
        correct = 0
        for pair in task['train']:
            inp = np.array(pair['input'])
            out = np.array(pair['output'])
            try:
                pred = np.array(rule_func(inp))
                if pred.shape == out.shape and np.array_equal(pred, out):
                    correct += 1
            except: pass
        return correct, len(task['train'])

    def infer_and_solve_combined_rules(self, task):
        # === SCALING ===
        if task['train']:
            i0 = np.array(task['train'][0]['input'])
            o0 = np.array(task['train'][0]['output'])
            ih, iw = i0.shape
            oh, ow = o0.shape
            if oh % ih == 0 and ow % iw == 0:
                f = oh // ih
                if f == ow // iw and f > 1:
                    def scale(g): return self._scale_grid(g, f)
                    if self._test_rule(task, scale)[0] == len(task['train']):
                        return scale

        # === COLOR SWAP ===
        if task['train']:
            colors_in = set(np.unique(i0)) - {0}
            colors_out = set(np.unique(o0)) - {0}
            if len(colors_in) == 1 and len(colors_out) == 1:
                a, b = list(colors_in)[0], list(colors_out)[0]
                def swap(g): return self._color_swap(g, a, b)
                if self._test_rule(task, swap)[0] == len(task['train']):
                    return swap

        # === EXHAUSTIVE SEARCH (1-3 steps) ===
        rules = [
            self._rotate_90, self._flip_vertical, self._trim_borders, self._crop_to_content,
            self._invert_colors, self._filter_by_size_one, self._keep_largest_component,
            self._fill_holes, self._move_smallest_to_largest_center,
            self._intersect_largest_two_objects, self._subtract_smallest_from_largest,
            self._tile_largest_component, self._recolor_all_non_background_to_9
        ]

        for r in rules:
            if self._test_rule(task, r)[0] == len(task['train']):
                return r

        for r1 in rules:
            for r2 in rules:
                def combo(g, r1=r1, r2=r2): return r2(np.array(r1(g)))
                if self._test_rule(task, combo)[0] == len(task['train']):
                    return combo

        for r1 in rules:
            for r2 in rules:
                for r3 in rules:
                    def combo3(g, r1=r1, r2=r2, r3=r3): return r3(np.array(r2(np.array(r1(g)))))
                    if self._test_rule(task, combo3)[0] == len(task['train']):
                        return combo3

        return lambda g: g.tolist()

    def run(self):
        if not self.load_tasks():
            return
        for task_id, task in self.tasks.items():
            rule = self.infer_and_solve_combined_rules(task)
            preds = []
            for test_case in task.get('test', []):
                inp = np.array(test_case['input'])
                pred_grid = rule(inp)
                legal_grid = self.make_legal(pred_grid)
                preds.append({"attempt_1": legal_grid})
            self.submission[task_id] = preds

        with open('submission.json', 'w') as f:
            json.dump(self.submission, f, separators=(',', ':'))
        print(f"SUBMISSION READY: {len(self.submission)} tasks")

# === RUN ===
solver = ARCSolver(TEST_DATA_PATH)
solver.run()


# VALIDATOR
def validate():
    with open('submission.json') as f:
        sub = json.load(f)
    if len(sub) != 100:
        print(f"Wrong tasks: {len(sub)}")
        return
    print("100 TASKS – FORMAT OK – SUBMIT NOW!")

validate()




