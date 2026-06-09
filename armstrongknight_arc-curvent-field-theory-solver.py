import json
import numpy as np
from scipy.ndimage import label, convolve, zoom, center_of_mass
from collections import Counter
import os


# FIELD OPERATORS
def compute_gradient(grid):
    grid = np.array(grid, dtype=float)
    dy = np.roll(grid, -1, axis=0) - grid
    dx = np.roll(grid, -1, axis=1) - grid
    return dy, dx

def compute_curl(grid):
    grid = np.array(grid, dtype=float)
    dy, dx = compute_gradient(grid)
    return np.roll(dx, -1, axis=0) - np.roll(dy, -1, axis=1)


# OBJECT DETECTION
def find_connected_components(grid, include_background=False):
    grid = np.array(grid)
    objects = []
    colors = np.unique(grid)
    if not include_background:
        colors = colors[colors != 0]
    for color in colors:
        mask = (grid == color)
        labeled, num = label(mask)
        for i in range(1, num + 1):
            component_mask = (labeled == i)
            indices = np.argwhere(component_mask)
            if len(indices) > 0:
                min_r, min_c = indices.min(axis=0)
                max_r, max_c = indices.max(axis=0)
                objects.append({
                    'color': int(color),
                    'mask': component_mask,
                    'position': (min_r, min_c),
                    'size': np.sum(component_mask),
                })
    return objects


# TRANSFORMATION INFERENCE
def infer_tiling_pattern(inp, out):
    inp, out = np.array(inp), np.array(out)
    ih, iw = inp.shape
    oh, ow = out.shape
    if oh % ih != 0 or ow % iw != 0:
        return None
    th, tw = oh // ih, ow // iw
    if th == 0 or tw == 0:
        return None
    if np.array_equal(np.tile(inp, (th, tw)), out):
        return ('tile', (th, tw))
    flip_modes = [
        ('row_flip_lr', lambda i, j: np.fliplr(inp) if i % 2 == 1 else inp),
        ('col_flip_lr', lambda i, j: np.fliplr(inp) if j % 2 == 1 else inp),
        ('checkerboard_lr', lambda i, j: np.fliplr(inp) if (i + j) % 2 == 1 else inp),
    ]
    for mode_name, tile_func in flip_modes:
        result = np.zeros((oh, ow), dtype=inp.dtype)
        for i in range(th):
            for j in range(tw):
                result[i*ih:(i+1)*ih, j*iw:(j+1)*iw] = tile_func(i, j)
        if np.array_equal(result, out):
            return ('tiling', (th, tw, mode_name))
    return None

def infer_self_reference(inp, out):
    inp, out = np.array(inp), np.array(out)
    ih, iw = inp.shape
    oh, ow = out.shape
    if oh % ih != 0 or ow % iw != 0:
        return None
    sh, sw = oh // ih, ow // iw
    if sh == 0 or sw == 0 or (sh == 1 and sw == 1):
        return None
    result = np.zeros((oh, ow), dtype=inp.dtype)
    for i in range(ih):
        for j in range(iw):
            if inp[i, j] != 0:
                result[i*sh:(i+1)*sh, j*sw:(j+1)*sw] = inp
    if np.array_equal(result, out):
        return ('self_reference', (sh, sw))
    return None

def infer_color_replacement(inp, out):
    inp, out = np.array(inp), np.array(out)
    if inp.shape != out.shape:
        return None
    objects = find_connected_components(inp)
    color_changes = {}
    for obj in objects:
        mask = obj['mask']
        out_colors = out[mask]
        unique_out = np.unique(out_colors)
        if len(unique_out) == 1 and unique_out[0] != obj['color']:
            old_c, new_c = obj['color'], int(unique_out[0])
            if old_c in color_changes and color_changes[old_c] != new_c:
                return None
            color_changes[old_c] = new_c
    if color_changes:
        return ('color_replace', color_changes)
    return None


# DIRECT TRANSFORMS
def detect_direct_transform(inp, out):
    inp, out = np.array(inp), np.array(out)
    if inp.shape == out.shape:
        if np.array_equal(out, np.fliplr(inp)): return ('flip_lr', None)
        if np.array_equal(out, np.flipud(inp)): return ('flip_ud', None)
        if np.array_equal(out, np.rot90(inp, 1)): return ('rot90', 1)
        if np.array_equal(out, np.rot90(inp, 2)): return ('rot180', 2)
        if np.array_equal(out, np.rot90(inp, 3)): return ('rot270', 3)
    return None

def detect_scale(inp, out):
    inp, out = np.array(inp), np.array(out)
    ih, iw = inp.shape
    oh, ow = out.shape
    if oh % ih == 0 and ow % iw == 0:
        sy, sx = oh // ih, ow // iw
        if sy > 0 and sx > 0:
            tiled = np.repeat(np.repeat(inp, sy, axis=0), sx, axis=1)
            if np.array_equal(tiled, out):
                return ('scale', (sy, sx))
    return None

def detect_value_map(inp, out):
    inp, out = np.array(inp), np.array(out)
    if inp.shape != out.shape:
        return None
    mapping = {}
    for i in range(inp.shape[0]):
        for j in range(inp.shape[1]):
            iv, ov = int(inp[i,j]), int(out[i,j])
            if iv in mapping and mapping[iv] != ov:
                return None
            mapping[iv] = ov
    if all(k == v for k,v in mapping.items()):
        return None
    return ('value_map', mapping)

def detect_region_extract(inp, out):
    inp, out = np.array(inp), np.array(out)
    ih, iw = inp.shape
    oh, ow = out.shape
    if oh > ih or ow > iw:
        return None
    for y in range(ih - oh + 1):
        for x in range(iw - ow + 1):
            if np.array_equal(inp[y:y+oh, x:x+ow], out):
                return ('extract', (y, x, oh, ow))
    return None


# MAIN SOLVER
class ARCSolver:
    def __init__(self):
        self.transform = None
        self.params = None
        self.output_shape = None
    
    def learn(self, train_pairs):
        for pair in train_pairs:
            inp, out = pair['input'], pair['output']
            for detector in [detect_scale, detect_direct_transform, infer_tiling_pattern,
                            infer_self_reference, detect_value_map, detect_region_extract,
                            infer_color_replacement]:
                try:
                    result = detector(inp, out)
                    if result:
                        self.transform = result[0]
                        self.params = result[1]
                        return
                except:
                    pass
        self.transform = 'unknown'
        self.output_shape = np.array(train_pairs[0]['output']).shape
    
    def predict(self, test_input):
        inp = np.array(test_input)
        
        if self.transform == 'scale':
            sy, sx = self.params
            return np.repeat(np.repeat(inp, sy, axis=0), sx, axis=1).tolist()
        if self.transform == 'flip_lr':
            return np.fliplr(inp).tolist()
        if self.transform == 'flip_ud':
            return np.flipud(inp).tolist()
        if self.transform in ['rot90', 'rot180', 'rot270']:
            k = {'rot90': 1, 'rot180': 2, 'rot270': 3}[self.transform]
            return np.rot90(inp, k).tolist()
        if self.transform == 'tile':
            th, tw = self.params
            return np.tile(inp, (th, tw)).tolist()
        if self.transform == 'tiling':
            th, tw, mode = self.params
            ih, iw = inp.shape
            result = np.zeros((ih * th, iw * tw), dtype=int)
            for i in range(th):
                for j in range(tw):
                    tile = inp
                    if 'flip_lr' in mode and ((i % 2 == 1 and 'row' in mode) or (j % 2 == 1 and 'col' in mode) or ((i+j) % 2 == 1 and 'checker' in mode)):
                        tile = np.fliplr(inp)
                    result[i*ih:(i+1)*ih, j*iw:(j+1)*iw] = tile
            return result.tolist()
        if self.transform == 'self_reference':
            sh, sw = self.params
            ih, iw = inp.shape
            result = np.zeros((ih * sh, iw * sw), dtype=int)
            for i in range(ih):
                for j in range(iw):
                    if inp[i, j] != 0:
                        result[i*sh:(i+1)*sh, j*sw:(j+1)*sw] = inp
            return result.tolist()
        if self.transform == 'value_map':
            result = inp.copy()
            for old_val, new_val in self.params.items():
                result[inp == old_val] = new_val
            return result.tolist()
        if self.transform == 'extract':
            y, x, h, w = self.params
            return inp[y:y+h, x:x+w].tolist()
        if self.transform == 'color_replace':
            result = inp.copy()
            for old, new in self.params.items():
                result[inp == old] = new
            return result.tolist()
        if self.output_shape:
            oh, ow = self.output_shape
            return inp[:oh, :ow].tolist()
        return inp.tolist()
    
    def solve_task(self, task):
        self.learn(task['train'])
        return [self.predict(t['input']) for t in task['test']]


# CREATE SUBMISSION
test_path = '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json'
with open(test_path, 'r') as f:
    test_challenges = json.load(f)

print(f'Loaded {len(test_challenges)} test challenges')

solver = ARCSolver()
submission = {}

for task_id, task in test_challenges.items():
    preds = solver.solve_task(task)
    submission[task_id] = [
        {'attempt_1': pred, 'attempt_2': pred}
        for pred in preds
    ]

with open('submission.json', 'w') as f:
    json.dump(submission, f)

print(f'Saved submission.json with {len(submission)} tasks')

