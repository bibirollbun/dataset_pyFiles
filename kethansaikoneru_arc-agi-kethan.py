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

# wrapper to try transpose/tile/region-based mapping and others
def try_extended_sequences(examples):
    for crop_flag in (False, True):
        for rotate_k in (0,1,2,3):
            for flip_mode in ('none','h','v'):
                # simple sequence
                res = try_sequence_matches_all(examples, crop_flag, rotate_k, flip_mode)
                if res: res['type']='transform'; return res
                # transpose variant
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
                # tile variant
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



def extract_block(grid, r0, c0, h, w):
    if not is_grid(grid): return None
    return [row[c0:c0+w] for row in grid[r0:r0+h]]

def try_concat_tile_examples(examples, max_block=6):
    for crop_flag in (False, True):
        for rotate_k in (0,1,2,3):
            for flip_mode in ('none','h','v'):
                global_map = {}; out_shape = None; ok_all=True
                for inp,out in examples:
                    if not (is_grid(inp) and is_grid(out)): ok_all=False; break
                    cand = crop_bbox(inp, bg=mode_of_grid(inp)) if crop_flag else copy_grid(inp)
                    cand = rotate_grid(cand, rotate_k)
                    if flip_mode=='h': cand = flip_grid_h(cand)
                    elif flip_mode=='v': cand = flip_grid_v(cand)
                    Hc,Wc = grid_shape(cand); Ho,Wo = grid_shape(out)
                    found_local=False
                    for bh in range(1, min(Hc, max_block)+1):
                        for bw in range(1, min(Wc, max_block)+1):
                            block = extract_block(cand, 0, 0, bh, bw)
                            if block is None: continue
                            tiled = maybe_tile_to_shape(block, (Ho,Wo))
                            padded = pad_to_shape(tiled, (Ho,Wo), pad_value=mode_of_grid(tiled))
                            mapping = derive_color_map(padded, out)
                            if mapping is None:
                                mapping = derive_region_map_by_shape(padded, out)
                                if mapping is None: continue
                            consistent = True
                            for k,v in mapping.items():
                                if k in global_map and global_map[k] != v: consistent=False; break
                            if not consistent: continue
                            for k,v in mapping.items(): global_map[k]=v
                            out_shape = grid_shape(out); found_local=True; break
                        if found_local: break
                    if not found_local:
                        ok_all=False; break
                if ok_all and global_map:
                    return {'mapping': global_map, 'out_shape': out_shape, 'rotate': rotate_k, 'flip': flip_mode, 'crop': crop_flag, 'tile': True, 'type': 'concat_tile'}
    return None

def try_row_col_shifts(examples, max_shift=None):
    for crop_flag in (False, True):
        for rotate_k in (0,1,2,3):
            for flip_mode in ('none','h','v'):
                mapping_global = {}; out_shape = None; ok_all=True
                for inp,out in examples:
                    if not (is_grid(inp) and is_grid(out)): ok_all=False; break
                    cand = crop_bbox(inp, bg=mode_of_grid(inp)) if crop_flag else copy_grid(inp)
                    cand = rotate_grid(cand, rotate_k)
                    if flip_mode=='h': cand = flip_grid_h(cand)
                    elif flip_mode=='v': cand = flip_grid_v(cand)
                    Ho,Wo = grid_shape(out); Hc,Wc = grid_shape(cand)
                    if Hc == Ho and Wc == Wo:
                        mapping = derive_color_map(cand, out)
                        if mapping is None:
                            mapping = derive_region_map_by_shape(cand, out)
                        if mapping is None: ok_all=False; break
                        for k,v in mapping.items():
                            if k in mapping_global and mapping_global[k] != v: ok_all=False; break
                            mapping_global[k]=v
                        out_shape = grid_shape(out)
                        if not ok_all: break
                        continue
                    found=False
                    max_r = Hc if max_shift is None else min(Hc, max_shift)
                    max_c = Wc if max_shift is None else min(Wc, max_shift)
                    for shift in range(0, max_r):
                        shifted = cand[-shift:] + cand[:-shift]
                        padded = pad_to_shape(shifted, (Ho,Wo), pad_value=mode_of_grid(shifted))
                        mapping = derive_color_map(padded, out)
                        if mapping is not None:
                            for k,v in mapping.items():
                                if k in mapping_global and mapping_global[k] != v: ok_all=False; break
                                mapping_global[k]=v
                            out_shape = grid_shape(out); found=True; break
                    if found: continue
                    for shift in range(0, max_c):
                        shifted = [row[-shift:]+row[:-shift] for row in cand]
                        padded = pad_to_shape(shifted, (Ho,Wo), pad_value=mode_of_grid(shifted))
                        mapping = derive_color_map(padded, out)
                        if mapping is not None:
                            for k,v in mapping.items():
                                if k in mapping_global and mapping_global[k] != v: ok_all=False; break
                                mapping_global[k]=v
                            out_shape = grid_shape(out); found=True; break
                    if not found:
                        ok_all=False; break
                if ok_all and mapping_global:
                    return {'mapping': mapping_global, 'out_shape': out_shape, 'rotate': rotate_k, 'flip': flip_mode, 'crop': crop_flag, 'type': 'shift'}
    return None



def get_examples_from_task(task):
    exs = []
    for ex in task.get('train', []):
        inp = ex.get('input'); out = ex.get('output')
        if isinstance(out, list) and out and is_grid(out[0]):
            out = out[0]
        exs.append((inp, out))
    return exs

def verify_program_on_examples(prog, examples):
    if prog is None: return False
    for inp, out in examples:
        if not is_grid(inp) or not is_grid(out): return False
        # apply inferred program
        pred = apply_inferred_program(prog, inp)
        if not is_grid(pred) or not grids_equal(pred, out):
            return False
    return True

def infer_program_for_task(task):
    examples = get_examples_from_task(task)
    if not examples: return None
    # try sequence primitives
    prog = try_extended_sequences(examples)
    if prog and verify_program_on_examples(prog, examples): return prog
    # try region-based variants (crop/rot/flip/transpose/tile handled inside)
    prog = None
    # reuse try_extended_sequences results already had region fallback; next try concat/tile
    prog = try_concat_tile_examples(examples)
    if prog and verify_program_on_examples(prog, examples): return prog
    prog = try_row_col_shifts(examples)
    if prog and verify_program_on_examples(prog, examples): return prog
    return None

def apply_inferred_program(prog, inp):
    if prog is None or not is_grid(inp): return None
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



def load_arc_files(input_dir='/kaggle/input/arc-prize-2025'):
    files = {
        'training_challenges': os.path.join(input_dir, 'arc-agi_training-challenges.json'),
        'training_solutions': os.path.join(input_dir, 'arc-agi_training-solutions.json'),
        'evaluation_challenges': os.path.join(input_dir, 'arc-agi_evaluation-challenges.json'),
        'evaluation_solutions': os.path.join(input_dir, 'arc-agi_evaluation-solutions.json'),
        'test_challenges': os.path.join(input_dir, 'arc-agi_test-challenges.json'),
        'sample_submission': os.path.join(input_dir, 'sample_submission.json')
    }
    loaded = {}
    for k,p in files.items():
        if os.path.exists(p):
            with open(p,'r') as fh: loaded[k] = json.load(fh)
            print('Loaded', k, '->', p, 'entries =', len(loaded[k]))
    return loaded

arc = load_arc_files()
evaluation_challenges = arc.get('evaluation_challenges', {})
evaluation_solutions = arc.get('evaluation_solutions', {})

# evaluate
mismatches = []
solved = 0; total = 0
for tid, task in (evaluation_challenges or {}).items():
    total += 1
    prog = infer_program_for_task(task)
    test_input = None
    try:
        test_input = task['test'][0]['input']
    except Exception:
        test_input = None
    pred = apply_inferred_program(prog, test_input) if (test_input is not None and prog is not None) else None
    true_out = None
    if evaluation_solutions and tid in evaluation_solutions:
        s = evaluation_solutions[tid]
        if isinstance(s, list) and s and is_grid(s[0]): true_out = s[0]
        elif is_grid(s): true_out = s
    if pred is not None and true_out is not None and grids_equal(pred, true_out):
        solved += 1
    else:
        mismatches.append({'tid':tid, 'prog': prog, 'pred_shape': grid_shape(pred) if is_grid(pred) else None, 'true_shape': grid_shape(true_out) if is_grid(true_out) else None})
print(f"Program-learner solved {solved}/{total} = {solved/total if total else 0:.4f}")
with open('/kaggle/working/mismatches_sample.json','w') as fh:
    json.dump(mismatches[:200], fh)
print('Wrote /kaggle/working/mismatches_sample.json (sample)')



def show_task_examples(tid):
    task = evaluation_challenges.get(tid) or {}
    print('Task ID:', tid)
    for i,ex in enumerate(task.get('train',[])):
        print('Train', i, 'input shape:', grid_shape(ex['input']), 'output shape:', grid_shape(ex['output'][0]) if isinstance(ex['output'], list) and ex['output'] and is_grid(ex['output'][0]) else grid_shape(ex['output']))
    try:
        print('Test input shape:', grid_shape(task['test'][0]['input']))
    except Exception:
        print('No test input found')
    import pprint; pprint.pprint(task)
# use: show_task_examples('0934a4d8')



# %% Cell 12 — Optional LLM->code fallback (disabled). Read comments and enable only if you add an API key to Kaggle Secrets.
# This cell contains a template using the OpenAI library style. It does NOT run unless you set `USE_LLM=True` and add secrets.
USE_LLM = False
if USE_LLM:
    # INSTRUCTIONS:
    # 1) Put your API key into Kaggle Secrets: go to "Settings -> Secrets" in the notebook and add OPENAI_API_KEY
    # 2) Install/openai (if missing): pip install openai (Kaggle image usually has it)
    # 3) Set USE_LLM=True and run this cell.
    import openai, os, json, textwrap
    openai.api_key = os.environ.get('OPENAI_API_KEY')  # Kaggle secrets populate env var
    def ask_llm_for_code(examples, max_tokens=1024):
        # build a concise prompt that asks for JSON {"code": "..."} with a def transform(inp):...
        # Keep examples small; do at most 4
        prompt_parts = ["You will be given up to 4 input->output grid pairs. Return ONLY a JSON object with key \"code\" whose value is Python source defining def transform(inp): returning a nested list of ints. No imports, no file/network calls."]
        for i,(inp,out) in enumerate(examples[:4]):
            prompt_parts.append(f"EX{i+1} INPUT:\n{json.dumps(inp)}\nEX{i+1} OUTPUT:\n{json.dumps(out)}")
        prompt_parts.append("Return: {\"code\": \"def transform(inp):\\n    ...\"}")
        prompt = "\n\n".join(prompt_parts)
        # use ChatCompletions v1 -> use the modern API call style if your openai version supports it
        resp = openai.ChatCompletion.create(model="gpt-4-0613", messages=[{"role":"user","content":prompt}], max_tokens=max_tokens, temperature=0)
        txt = resp['choices'][0]['message']['content']
        try:
            obj = json.loads(txt)
            return obj.get('code')
        except Exception:
            # attempt to extract code block
            return txt
    # safe exec snippet (basic)
    def safe_exec_and_verify(code_str, examples):
        forbidden = ['import ','open(', 'subprocess', 'os.', 'sys.', 'eval(', 'exec(', '__', 'socket', 'requests']
        if any(tok in code_str for tok in forbidden):
            return None, "forbidden tokens"
        safe_globals = {"__builtins__": {}}
        safe_locals = {}
        try:
            exec(code_str, safe_globals, safe_locals)
        except Exception as e:
            return None, f"exec error: {e}"
        transform = safe_locals.get('transform')
        if transform is None:
            return None, "no transform"
        from types import SimpleNamespace
        for inp,out in examples:
            try:
                pred = transform(inp)
            except Exception as e:
                return None, f"runtime error on example: {e}"
            if not is_grid(pred) or not grids_equal(pred, out):
                return None, "failed training examples"
        return transform, "ok"





