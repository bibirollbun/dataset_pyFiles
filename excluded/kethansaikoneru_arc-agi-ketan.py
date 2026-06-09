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



import os, json, math
from copy import deepcopy
from collections import Counter, deque
import numpy as np
import pandas as pd


def is_grid(g):
    if not isinstance(g, list) or not g: return False
    if not all(isinstance(r, list) for r in g): return False
    if any(not all(isinstance(x, int) for x in r) for r in g): return False
    return len({len(r) for r in g}) == 1

def grid_shape(g):
    return None if not is_grid(g) else (len(g), len(g[0]))

def copy_grid(g):
    return deepcopy(g)

def grids_equal(a,b):
    if not (is_grid(a) and is_grid(b)): return False
    if grid_shape(a) != grid_shape(b): return False
    return all(a[i] == b[i] for i in range(len(a)))

def mode_of_grid(g):
    flat = [c for r in g for c in r]
    return Counter(flat).most_common(1)[0][0]


def rotate_grid(g, k=1):
    if not is_grid(g): return g
    k = k % 4
    out = copy_grid(g)
    for _ in range(k):
        rows, cols = len(out), len(out[0])
        new = [[0]*rows for _ in range(cols)]
        for i in range(rows):
            for j in range(cols):
                new[j][rows-1-i] = out[i][j]
        out = new
    return out

def flip_grid_h(g):
    return [list(reversed(row)) for row in g] if is_grid(g) else g

def flip_grid_v(g):
    return list(reversed(g)) if is_grid(g) else g

def transpose_grid(g):
    if not is_grid(g): return g
    rows, cols = len(g), len(g[0])
    return [[g[r][c] for r in range(rows)] for c in range(cols)]


def pad_to_shape(g, shape, pad_value=0):
    if not is_grid(g): return g
    h,w = shape
    out = [[pad_value]*w for _ in range(h)]
    for i in range(min(h, len(g))):
        for j in range(min(w, len(g[0]))):
            out[i][j] = g[i][j]
    return out

def crop_bbox(g, bg=None):
    if not is_grid(g): return g
    if bg is None: bg = mode_of_grid(g)
    R,C = len(g), len(g[0])
    minr, minc = R, C
    maxr, maxc = -1, -1
    for i in range(R):
        for j in range(C):
            if g[i][j] != bg:
                minr = min(minr,i); maxr = max(maxr,i)
                minc = min(minc,j); maxc = max(maxc,j)
    if maxr == -1:
        return [[bg]]
    return [row[minc:maxc+1] for row in g[minr:maxr+1]]

def maybe_tile_to_shape(cand, target_shape):
    if not is_grid(cand): return cand
    th, tw = target_shape; ch, cw = len(cand), len(cand[0])
    reps_h = math.ceil(th / ch); reps_w = math.ceil(tw / cw)
    tiled = []
    for rh in range(reps_h):
        for r in range(ch):
            row = []
            for rw in range(reps_w):
                row.extend(cand[r])
            tiled.append(row[:tw])
    return tiled[:th]


def derive_color_map(src, tgt):
    if not (is_grid(src) and is_grid(tgt)): return None
    if grid_shape(src) != grid_shape(tgt): return None
    m = {}
    for i in range(len(src)):
        for j in range(len(src[0])):
            a, b = src[i][j], tgt[i][j]
            if a in m and m[a] != b:
                return None
            m[a] = b
    return m

def apply_color_map_to_grid(g, mapping):
    if not is_grid(g): return g
    return [[mapping.get(cell, cell) for cell in row] for row in g]


def connected_components(grid):
    if not is_grid(grid): return []
    H, W = len(grid), len(grid[0])
    visited = [[False]*W for _ in range(H)]
    comps = []
    dirs = [(1,0),(-1,0),(0,1),(0,-1)]
    for i in range(H):
        for j in range(W):
            if visited[i][j]: continue
            color = grid[i][j]
            q = deque([(i,j)]); visited[i][j] = True
            mask = []; minr, minc, maxr, maxc = i,j,i,j
            while q:
                r,c = q.popleft()
                mask.append((r,c))
                minr = min(minr, r); minc = min(minc, c)
                maxr = max(maxr, r); maxc = max(maxc, c)
                for dr,dc in dirs:
                    nr, nc = r+dr, c+dc
                    if 0<=nr<H and 0<=nc<W and not visited[nr][nc] and grid[nr][nc]==color:
                        visited[nr][nc] = True
                        q.append((nr,nc))
            bh,bw = maxr-minr+1, maxc-minc+1
            shape = [[0]*bw for _ in range(bh)]
            for (r,c) in mask:
                shape[r-minr][c-minc] = 1
            comps.append({'color': color, 'mask': mask, 'bbox': (minr,minc,maxr,maxc), 'shape': shape, 'size': len(mask)})
    return comps

def shapes_equal(s1, s2):
    return is_grid(s1) and is_grid(s2) and grids_equal(s1,s2)

def derive_region_map_by_shape(src, tgt):
    if not (is_grid(src) and is_grid(tgt)): return None
    if grid_shape(src) != grid_shape(tgt): return None
    scomps = connected_components(src)
    tcomps = connected_components(tgt)
    mapping = {}
    used_t = set()
    for sc in sorted(scomps, key=lambda x:(-x['size'], x['color'])):
        matched = False
        for ti, tc in enumerate(tcomps):
            if ti in used_t: continue
            if sc['size'] == tc['size'] and shapes_equal(sc['shape'], tc['shape']):
                if sc['color'] in mapping and mapping[sc['color']] != tc['color']: return None
                mapping[sc['color']] = tc['color']; used_t.add(ti); matched = True; break
        if not matched:
            for ti, tc in enumerate(tcomps):
                if ti in used_t: continue
                if shapes_equal(sc['shape'], tc['shape']):
                    if sc['color'] in mapping and mapping[sc['color']] != tc['color']: return None
                    mapping[sc['color']] = tc['color']; used_t.add(ti); matched = True; break
        if not matched:
            return None
    return mapping



def load_arc_files(input_dir='/kaggle/input/arc-prize-2025'):
    files = {
        'training_challenges': os.path.join(input_dir, 'arc-agi_training_challenges.json'),  # Fixed: underscore not dash
        'training_solutions': os.path.join(input_dir, 'arc-agi_training_solutions.json'),   # Fixed: underscore not dash
        'evaluation_challenges': os.path.join(input_dir, 'arc-agi_evaluation_challenges.json'),
        'evaluation_solutions': os.path.join(input_dir, 'arc-agi_evaluation_solutions.json'),
        'test_challenges': os.path.join(input_dir, 'arc-agi_test_challenges.json'),
        'sample_submission': os.path.join(input_dir, 'sample_submission.json')
    }
    loaded = {}
    for k,p in files.items():
        if os.path.exists(p):
            with open(p,'r') as fh: loaded[k] = json.load(fh)
            print('Loaded', k, '->', p, 'entries =', len(loaded[k]))
        else:
            print('File not found:', p)
    return loaded

# Enhanced pattern detection functions
def try_simple_patterns(examples):
    """Try simple direct patterns like copy, fill, etc."""
    # Check if output is just the input
    for inp, out in examples:
        if grids_equal(inp, out):
            return {'type': 'identity', 'mapping': {}}
    
    # Check if output is filled with single color
    single_colors = []
    for inp, out in examples:
        out_colors = set(c for row in out for c in row)
        if len(out_colors) == 1:
            single_colors.append(list(out_colors)[0])
        else:
            single_colors = []
            break
    
    if single_colors and len(set(single_colors)) == 1:
        return {'type': 'fill', 'color': single_colors[0], 'out_shape': grid_shape(examples[0][1])}
    
    return None

def try_object_manipulation(examples):
    """Try object-based transformations"""
    for inp, out in examples:
        if not (is_grid(inp) and is_grid(out)): continue
        
        inp_comps = connected_components(inp)
        out_comps = connected_components(out)
        
        # Check if it's just moving objects
        if len(inp_comps) == len(out_comps):
            inp_shapes = [comp['shape'] for comp in inp_comps]
            out_shapes = [comp['shape'] for comp in out_comps]
            
            if all(any(shapes_equal(i_shape, o_shape) for o_shape in out_shapes) for i_shape in inp_shapes):
                return {'type': 'object_move', 'preserve_shapes': True}
    
    return None

def try_size_based_patterns(examples):
    """Try patterns based on size changes"""
    size_ratios = []
    for inp, out in examples:
        inp_shape = grid_shape(inp)
        out_shape = grid_shape(out)
        if inp_shape and out_shape:
            r_ratio = out_shape[0] / inp_shape[0]
            c_ratio = out_shape[1] / inp_shape[1]
            size_ratios.append((r_ratio, c_ratio))
    
    if size_ratios and all(r == size_ratios[0] for r in size_ratios):
        ratio = size_ratios[0]
        if ratio[0] == ratio[1] and ratio[0] in [2, 3, 4, 0.5, 0.25]:
            return {'type': 'scale', 'ratio': ratio[0]}
    
    return None

# Your existing transformation functions (keeping them)
def try_sequence_matches_all(examples, crop_flag, rotate_k, flip_mode):
    mapping_global = {}; out_shape = None
    for inp, out in examples:
        if not (is_grid(inp) and is_grid(out)): return None
        cand = crop_bbox(inp, bg=mode_of_grid(inp)) if crop_flag else copy_grid(inp)
        cand = rotate_grid(cand, rotate_k)
        if flip_mode == 'h': cand = flip_grid_h(cand)
        elif flip_mode == 'v': cand = flip_grid_v(cand)
        cand_padded = pad_to_shape(cand, grid_shape(out), pad_value=mode_of_grid(cand))
        mapping = derive_color_map(cand_padded, out)
        if mapping is None: return None
        for k,v in mapping.items():
            if k in mapping_global and mapping_global[k] != v: return None
            mapping_global[k] = v
        out_shape = grid_shape(out)
    return {'mapping': mapping_global, 'out_shape': out_shape, 'rotate': rotate_k, 'flip': flip_mode, 'crop': crop_flag, 'transpose': False, 'tile': False}

def try_extended_sequences(examples):
    for crop_flag in (False, True):
        for rotate_k in (0,1,2,3):
            for flip_mode in ('none','h','v'):
                res = try_sequence_matches_all(examples, crop_flag, rotate_k, flip_mode)
                if res: res['type']='transform'; return res
                
                mapping_global = {}
                ok = True; out_shape = None
                for inp,out in examples:
                    if not (is_grid(inp) and is_grid(out)): ok=False; break
                    cand = crop_bbox(inp, bg=mode_of_grid(inp)) if crop_flag else copy_grid(inp)
                    cand = rotate_grid(cand, rotate_k)
                    if flip_mode=='h': cand = flip_grid_h(cand)
                    elif flip_mode=='v': cand = flip_grid_v(cand)
                    cand = transpose_grid(cand)
                    cand_padded = pad_to_shape(cand, grid_shape(out), pad_value=mode_of_grid(cand))
                    mapping = derive_color_map(cand_padded, out)
                    if mapping is None:
                        mapping = derive_region_map_by_shape(cand_padded, out)
                        if mapping is None: ok=False; break
                    for k,v in mapping.items():
                        if k in mapping_global and mapping_global[k] != v: ok=False; break
                        mapping_global[k]=v
                    out_shape = grid_shape(out)
                if ok and mapping_global:
                    return {'mapping': mapping_global, 'out_shape': out_shape, 'rotate': rotate_k, 'flip': flip_mode, 'crop': crop_flag, 'transpose': True, 'tile': False, 'type':'transform'}
                
                mapping_global = {}; ok = True; out_shape = None
                for inp,out in examples:
                    if not (is_grid(inp) and is_grid(out)): ok=False; break
                    cand = crop_bbox(inp, bg=mode_of_grid(inp)) if crop_flag else copy_grid(inp)
                    cand = rotate_grid(cand, rotate_k)
                    if flip_mode=='h': cand = flip_grid_h(cand)
                    elif flip_mode=='v': cand = flip_grid_v(cand)
                    cand_tiled = maybe_tile_to_shape(cand, grid_shape(out))
                    cand_padded = pad_to_shape(cand_tiled, grid_shape(out), pad_value=mode_of_grid(cand_tiled))
                    mapping = derive_color_map(cand_padded, out)
                    if mapping is None:
                        mapping = derive_region_map_by_shape(cand_padded, out)
                        if mapping is None: ok=False; break
                    for k,v in mapping.items():
                        if k in mapping_global and mapping_global[k] != v: ok=False; break
                        mapping_global[k]=v
                    out_shape = grid_shape(out)
                if ok and mapping_global:
                    return {'mapping': mapping_global, 'out_shape': out_shape, 'rotate': rotate_k, 'flip': flip_mode, 'crop': crop_flag, 'transpose': False, 'tile': True, 'type':'transform'}
    return None


def get_examples_from_task(task):
    exs = []
    for ex in task.get('train', []):
        inp = ex.get('input')
        out = ex.get('output')
        # Handle different output formats
        if isinstance(out, list) and out and isinstance(out[0], list) and isinstance(out[0][0], int):
            # output is already a grid
            pass
        elif isinstance(out, list) and out and isinstance(out[0], list) and isinstance(out[0][0], list):
            # output is list of grids, take first one
            out = out[0]
        exs.append((inp, out))
    return exs

# Enhanced solver with more strategies
def infer_program_for_task(task):
    examples = get_examples_from_task(task)
    if not examples: return None
    
    # Try simple patterns first
    prog = try_simple_patterns(examples)
    if prog and verify_program_on_examples(prog, examples): return prog
    
    # Try size-based patterns
    prog = try_size_based_patterns(examples)
    if prog and verify_program_on_examples(prog, examples): return prog
    
    # Try object manipulation
    prog = try_object_manipulation(examples)
    if prog and verify_program_on_examples(prog, examples): return prog
    
    # Try your existing sequence primitives
    prog = try_extended_sequences(examples)
    if prog and verify_program_on_examples(prog, examples): return prog
    
    return None

def verify_program_on_examples(prog, examples):
    if prog is None: return False
    for inp, out in examples:
        if not is_grid(inp) or not is_grid(out): return False
        pred = apply_inferred_program(prog, inp)
        if not is_grid(pred) or not grids_equal(pred, out):
            return False
    return True

# Enhanced program application
def apply_inferred_program(prog, inp):
    if prog is None or not is_grid(inp): return None
    
    prog_type = prog.get('type', 'transform')
    
    if prog_type == 'identity':
        return copy_grid(inp)
    elif prog_type == 'fill':
        out_shape = prog.get('out_shape', grid_shape(inp))
        color = prog.get('color', 0)
        return [[color]*out_shape[1] for _ in range(out_shape[0])]
    elif prog_type == 'scale':
        ratio = prog.get('ratio', 1)
        if ratio > 1:
            # Scale up
            result = []
            for row in inp:
                scaled_row = []
                for cell in row:
                    scaled_row.extend([cell] * int(ratio))
                for _ in range(int(ratio)):
                    result.append(scaled_row[:])
            return result
        elif ratio < 1:
            # Scale down
            step = int(1/ratio)
            result = []
            for i in range(0, len(inp), step):
                row = []
                for j in range(0, len(inp[0]), step):
                    row.append(inp[i][j])
                result.append(row)
            return result
        else:
            return copy_grid(inp)
    else:
        # Your existing transform logic
        cand = copy_grid(inp)
        if prog.get('crop'):
            cand = crop_bbox(cand, bg=mode_of_grid(cand))
        cand = rotate_grid(cand, prog.get('rotate',0))
        flip = prog.get('flip')
        if flip == 'h': cand = flip_grid_h(cand)
        elif flip == 'v': cand = flip_grid_v(cand)
        if prog.get('transpose'): cand = transpose_grid(cand)
        if prog.get('tile'): 
            out_shape = prog.get('out_shape')
            if out_shape: cand = maybe_tile_to_shape(cand, out_shape)
        out_shape = prog.get('out_shape')
        if out_shape: cand = pad_to_shape(cand, out_shape, pad_value=mode_of_grid(cand))
        pred = apply_color_map_to_grid(cand, prog.get('mapping', {}))
        return pred



arc = load_arc_files()

# Test on training data first to see if we can solve anything
print("Testing on TRAINING data...")
training_challenges = arc.get('training_challenges', {})
training_solutions = arc.get('training_solutions', {})

solved_training = 0
total_training = 0
training_results = []

for tid, task in list(training_challenges.items())[:50]:  # Test first 50
    total_training += 1
    prog = infer_program_for_task(task)
    
    # Get test input
    test_input = None
    try:
        test_input = task['test'][0]['input']
    except:
        continue
        
    pred = apply_inferred_program(prog, test_input) if (test_input is not None and prog is not None) else None
    
    # Get ground truth
    true_out = None
    if tid in training_solutions:
        s = training_solutions[tid]
        if isinstance(s, list) and s and is_grid(s[0]): 
            true_out = s[0]
        elif is_grid(s): 
            true_out = s
    
    is_correct = pred is not None and true_out is not None and grids_equal(pred, true_out)
    if is_correct:
        solved_training += 1
        
    training_results.append({
        'task_id': tid,
        'solved': is_correct,
        'program_type': prog.get('type') if prog else None,
        'pred_shape': grid_shape(pred) if pred else None,
        'true_shape': grid_shape(true_out) if true_out else None
    })

print(f"TRAINING: Solved {solved_training}/{total_training} = {solved_training/total_training*100:.2f}%")



solved_tasks = [r for r in training_results if r['solved']]
failed_tasks = [r for r in training_results if not r['solved']]

print("\nSOLVED TASKS (first 5):")
for r in solved_tasks[:5]:
    print(f"  {r['task_id']}: {r['program_type']}")

print("\nFAILED TASKS (first 5):")
for r in failed_tasks[:5]:
    print(f"  {r['task_id']}: prog_type={r['program_type']}, pred_shape={r['pred_shape']}, true_shape={r['true_shape']}")



print("\nTesting on EVALUATION data...")
evaluation_challenges = arc.get('evaluation_challenges', {})
evaluation_solutions = arc.get('evaluation_solutions', {})

solved_eval = 0
total_eval = 0

for tid, task in evaluation_challenges.items():
    total_eval += 1
    prog = infer_program_for_task(task)
    
    test_input = None
    try:
        test_input = task['test'][0]['input']
    except:
        continue
        
    pred = apply_inferred_program(prog, test_input) if (test_input is not None and prog is not None) else None
    
    true_out = None
    if tid in evaluation_solutions:
        s = evaluation_solutions[tid]
        if isinstance(s, list) and s and is_grid(s[0]): 
            true_out = s[0]
        elif is_grid(s): 
            true_out = s
    
    if pred is not None and true_out is not None and grids_equal(pred, true_out):
        solved_eval += 1

print(f"EVALUATION: Solved {solved_eval}/{total_eval} = {solved_eval/total_eval*100 if total_eval else 0:.2f}%")



test_challenges = arc.get('test_challenges', {})
submission = {}

for task_id, task in test_challenges.items():
    task_predictions = []
    prog = infer_program_for_task(task)
    
    for test_case in task.get('test', []):
        test_input = test_case.get('input')
        pred = apply_inferred_program(prog, test_input) if (test_input and prog) else None
        
        if pred and is_grid(pred):
            # Submit the prediction and a fallback
            task_predictions.append({'attempt_1': pred, 'attempt_2': pred})
        else:
            # Submit empty grids as fallback
            task_predictions.append({'attempt_1': [[0]], 'attempt_2': [[0]]})
    
    submission[task_id] = task_predictions

print(f"\nGenerated submission for {len(submission)} test tasks")

# Save submission
with open('/kaggle/working/submission.json', 'w') as f:
    json.dump(submission, f)
    
print("Saved submission to /kaggle/working/submission.json")




