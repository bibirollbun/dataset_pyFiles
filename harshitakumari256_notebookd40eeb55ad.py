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


import json
import re
import torch
import numpy as np
from pathlib import Path
from collections import Counter
from itertools import product
from skimage.transform import rotate
from scipy.ndimage import binary_fill_holes
from transformers import AutoModelForCausalLM, AutoTokenizer


DATA_DIR = Path('/kaggle/input/arc-prize-2025')
TEST_FILE = DATA_DIR / 'arc-agi_test_challenges.json'


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def safe_truncate(grid, max_size=8):
    return [row[:max_size] for row in grid[:max_size]]



from pprint import pprint
data = load_json(TEST_FILE)
items = list(data.items())
N = 5
first_n = items[:1]
pprint(first_n)



def infer_color_map(inp, out):
    cmap = {}
    for ex_in, ex_out in zip(inp, out):
        in_arr = np.array(ex_in)
        out_arr = np.array(ex_out)
        if in_arr.shape != out_arr.shape:
            continue
            
        for c in np.unique(in_arr):
            if c not in cmap:
                matches = out_arr[in_arr == c]
                if matches.size > 0:
                    cmap[c] = Counter(matches.flatten()).most_common(1)[0][0]
    return cmap

def apply_color_map(grid, cmap):
    arr = np.array(grid)
    return np.vectorize(lambda x: cmap.get(x, x))(arr).tolist()



def infer_rotation(inp, out):
    for angle in [0, 90, 180, 270]:
        if all(np.array_equal(rotate(np.array(ei), angle, preserve_range=True).astype(int), eo)
               for ei, eo in zip(inp, out)):
            return angle
    return None

def apply_rotation(grid, angle):
    if angle is None:
        return grid
    arr = np.array(grid)
    return rotate(arr, angle, preserve_range=True).astype(int).tolist()



def infer_scale(inp, out):
    dx_vals, dy_vals = [], []
    for ei, eo in zip(inp, out):
        h1, w1 = len(ei), len(ei[0])
        h2, w2 = len(eo), len(eo[0])
        dx_vals.append(w2 // w1)
        dy_vals.append(h2 // h1)
    dx = max(1, Counter(dx_vals).most_common(1)[0][0])
    dy = max(1, Counter(dy_vals).most_common(1)[0][0])
    return dx, dy

def apply_scale(grid, dx, dy):
    arr = np.array(grid)
    return np.kron(arr, np.ones((dy, dx), dtype=int)).tolist()



PRIMITIVES = [
    ("rot90", lambda g: np.rot90(np.array(g), k=1).tolist()),
    ("rot180", lambda g: np.rot90(np.array(g), k=2).tolist()),
    ("rot270", lambda g: np.rot90(np.array(g), k=3).tolist()),
    ("flipH", lambda g: np.fliplr(np.array(g)).tolist()),
    ("flipV", lambda g: np.flipud(np.array(g)).tolist()),
    ("scale2x", lambda g: np.kron(np.array(g), np.ones((2,2), dtype=int)).tolist()),
    ("tile2x2", lambda g: np.tile(np.array(g), (2,2)).tolist()),
    ("fill_holes", lambda g: binary_fill_holes(np.array(g)).astype(int).tolist())
]

def synthesize_dsl(demo_pairs, max_depth=2):
    for depth in range(1, max_depth+1):
        for pipeline in product(PRIMITIVES, repeat=depth):
            funcs = [fn for _, fn in pipeline]
            if all(run_pipeline(inp, funcs) == out for inp, out in demo_pairs):
                return funcs
    return None



def run_pipeline(grid, pipeline):
    current = grid
    for fn in pipeline:
        current = fn(current)
    return current


model = AutoModelForCausalLM.from_pretrained(
    "/kaggle/input/qwen-3/transformers/1.7b-base/1",
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(
    "/kaggle/input/qwen-3/transformers/1.7b-base/1"
)



import re
import torch

def evaluate(task):
    demo_pairs = [(ex['input'], ex['output']) for ex in task['train']]
    examples = "\n\n".join(
        f"Input {i}:\n{safe_truncate(inp)}\nOutput {i}:\n{safe_truncate(out)}"
        for i, (inp, out) in enumerate(demo_pairs)
    )
    prompt = f"""You are a puzzle solving wizard. You are given a puzzle 
    from the abstraction and reasoning 
    corpus developed by Francois Chollet.

{examples}

Respond *only* with a fenced Python code block in the given format, make sure to name the function -> solve(this is mandatory that you name the function "solve", passing grid parameter), and do not generate any other text except the specified format given for code generation given below:

```python
# your minimal code here
```"""

    messages = [
        {"role": "system", "content": "You are an expert problem solver."},
        {"role": "user",   "content": prompt}
    ]
    with torch.no_grad():
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated = model.generate(
            **inputs,
            max_new_tokens=256,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.0,  
            top_p=1.0,
            do_sample=False
        )
        response = tokenizer.decode(generated[0], skip_special_tokens=True)
    blocks = re.findall(
        r'```python\s*\n([\s\S]*?)\n```',
        response,
        flags=re.DOTALL
    )
    if not blocks:
        raise RuntimeError("Model response did not contain a ```python``` block.")
    code = blocks[-1].strip()
    namespace = {}
    exec(code, namespace)
    solve_func = namespace.get('solve')
    if not callable(solve_func):
        raise RuntimeError("No callable solve(grid) found in returned code.")

    for inp, out in demo_pairs:
        if solve_func([row.copy() for row in inp]) != out:
            raise RuntimeError(f"Validation failed for input {inp!r}")

    return solve_func



def ensemble(task):
    demo_pairs = [(ex['input'], ex['output']) for ex in task['train']]
    
    same_size_pairs = [(i, o) for i, o in demo_pairs 
                      if np.array(i).shape == np.array(o).shape]
    if same_size_pairs:
        cmap = infer_color_map(*zip(*same_size_pairs))
        if cmap and all(apply_color_map(inp, cmap) == out 
                       for inp, out in same_size_pairs):
            return lambda g: apply_color_map(g, cmap)
    
    angle = infer_rotation(*zip(*demo_pairs))
    if angle is not None:
        return lambda g: apply_rotation(g, angle)
    
    dx, dy = infer_scale(*zip(*demo_pairs))
    if all(apply_scale(inp, dx, dy) == out for inp, out in demo_pairs):
        return lambda g: apply_scale(g, dx, dy)
    
    dsl_pipeline = synthesize_dsl(demo_pairs)
    if dsl_pipeline is not None:
        return lambda g: run_pipeline(g, dsl_pipeline)
    
    return evaluate(task)


if __name__ == "__main__":
    test_data = load_json(TEST_FILE)
    submission = {}

    for task_id, task in test_data.items():
        try:
            solver1 = ensemble(task)
            preds1 = [solver1(tc['input']) for tc in task['test']]
            del solver1
            torch.cuda.empty_cache()
            solver2 = ensemble(task)
            preds2 = [solver2(tc['input']) for tc in task['test']]
            del solver2
            torch.cuda.empty_cache()
            submission[task_id] = [
                {"attempt_1": p1, "attempt_2": p2}
                for p1, p2 in zip(preds1, preds2)
            ]

        except Exception as e:
            print(f"Failed task {task_id}: {str(e)[:200]}")
            fallback = [tc['input'] for tc in task['test']]
            submission[task_id] = [
                {"attempt_1": grid, "attempt_2": grid}
                for grid in fallback
            ]
    with open('submission.json', 'w') as f:
        json.dump(submission, f)

    print("Submission saved with", len(submission), "tasks processed")





