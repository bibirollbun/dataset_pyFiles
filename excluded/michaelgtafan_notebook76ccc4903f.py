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


# Load and examine the utility code
with open('/kaggle/input/google-code-golf-2025/code_golf_utils/code_golf_utils.py', 'r') as f:
    utils_code = f.read()
print("=== CODE GOLF UTILS ===")
print(utils_code)
print("\n" + "="*50 + "\n")

# Load a sample task to understand structure
import json
with open('/kaggle/input/google-code-golf-2025/task001.json', 'r') as f:
    sample_task = json.load(f)
print("=== SAMPLE TASK STRUCTURE (task001) ===")
print(json.dumps(sample_task, indent=2))


# COMPREHENSIVE ARC CODE GOLF ANALYSIS & AUTOMATION
import json,glob
from collections import Counter

# Load all 400 tasks
tasks={}
for f in sorted(glob.glob('/kaggle/input/google-code-golf-2025/task*.json')):
    with open(f)as fp:tasks[f.split('/')[-1].split('.')[0]]=json.load(fp)
print(f"✓ Loaded {len(tasks)} tasks\n")

# Analyze structure
print("=== TASK STRUCTURE ANALYSIS ===")
s=list(tasks.values())[0]
print(f"Keys: {list(s.keys())}")
if 'train' in s:
    print(f"Train examples: {len(s['train'])}")
    print(f"Test examples: {len(s.get('test',[]))}")
    if s['train']:
        ex=s['train'][0]
        print(f"Example keys: {list(ex.keys())}")
        if 'input' in ex:
            inp=ex['input']
            print(f"Input type: {type(inp)}")
            if isinstance(inp,list)and inp:
                print(f"Grid shape: {len(inp)}x{len(inp[0]) if inp else 0}")

print("\n=== CODE GOLF TIPS ===")
print("""1. Import: from X import*; use j=json.load inline
2. Lambda: lambda x:transformed_x
3. List comp: [f(x)for x in data]
4. Numpy: np.array ops, rot90, flip for transforms
5. Single-letter vars
6. Ternary: x if c else y
7. Grid ops: zip(*g)=transpose, g[::-1]=vertical flip
8. ARC patterns: color map, symmetry, pattern fill""")

print("\n=== TEST FRAMEWORK ===")
print("""def test(solve,tid):
    task=tasks[tid]
    c=0
    for ex in task.get('train',[]):
        try:
            if solve(ex['input'])==ex['output']:c+=1
        except:pass
    return c,len(task.get('train',[]))

def test_all(solve):
    return{tid:test(solve,tid)for tid in tasks if test(solve,tid)[0]>0}""")

print(f"\n=== SUMMARY ===")
print(f"✓ {len(tasks)} ARC tasks loaded")
print(f"✓ Util: /kaggle/input/.../code_golf_utils.py")
print(f"✓ solve=lambda inp:out")
print(f"✓ Minimize chars!\n")


# AUTOMATED CODE GOLF SOLVER FOR ALL 400 ARC TASKS
import json,glob,numpy as np
from collections import Counter
import time

# Load all tasks
tasks={}
for f in sorted(glob.glob('/kaggle/input/google-code-golf-2025/task*.json')):
    with open(f)as fp:tasks[f.split('/')[-1].split('.')[0]]=json.load(fp)

print(f"Loaded {len(tasks)} tasks\n")

# Define compact solution patterns (code golf style) with character counts
solution_templates = [
    # Identity/copy - 10 chars
    (lambda x:x, "lambda x:x", 10),
    # Transpose - 36 chars
    (lambda x:[list(r)for r in zip(*x)], "lambda x:[list(r)for r in zip(*x)]", 36),
    # Vertical flip - 18 chars
    (lambda x:x[::-1], "lambda x:x[::-1]", 18),
    # Horizontal flip - 32 chars
    (lambda x:[r[::-1]for r in x], "lambda x:[r[::-1]for r in x]", 32),
    # Rotate 90 CW - 44 chars
    (lambda x:[list(r)[::-1]for r in zip(*x)], "lambda x:[list(r)[::-1]for r in zip(*x)]", 44),
    # Rotate 180 - 39 chars
    (lambda x:[r[::-1]for r in x[::-1]], "lambda x:[r[::-1]for r in x[::-1]]", 39),
    # Rotate 270 CW - 45 chars
    (lambda x:[list(r)for r in zip(*x[::-1])], "lambda x:[list(r)for r in zip(*x[::-1])]", 45),
    # Numpy rotate 90
    (lambda x:np.rot90(x).tolist(), "lambda x:np.rot90(x).tolist()", 32),
    # Numpy rotate 180  
    (lambda x:np.rot90(x,2).tolist(), "lambda x:np.rot90(x,2).tolist()", 34),
    # Numpy rotate 270
    (lambda x:np.rot90(x,3).tolist(), "lambda x:np.rot90(x,3).tolist()", 34),
    # Numpy flip vertical
    (lambda x:np.flipud(x).tolist(), "lambda x:np.flipud(x).tolist()", 33),
    # Numpy flip horizontal
    (lambda x:np.fliplr(x).tolist(), "lambda x:np.fliplr(x).tolist()", 33),
]

print(f"Testing {len(solution_templates)} solution patterns...\n")

# Test function
def test_solution(solve, task_data):
    passed = 0
    total = len(task_data.get('train', []))
    for ex in task_data.get('train', []):
        try:
            result = solve(ex['input'])
            if result == ex['output']:
                passed += 1
        except:
            pass
    return passed, total

# Run all solutions on all tasks
results = {}
for tid, task_data in tasks.items():
    best_sol_idx = None
    best_score = 0
    best_char_count = float('inf')
    
    for i, (sol_func, sol_code, char_count) in enumerate(solution_templates):
        passed, total = test_solution(sol_func, task_data)
        
        # Update best if: more passed, or same passed but fewer chars
        if passed > best_score or (passed == best_score and passed > 0 and char_count < best_char_count):
            best_score = passed
            best_sol_idx = i
            best_char_count = char_count
    
    results[tid] = {
        'sol_idx': best_sol_idx,
        'sol_code': solution_templates[best_sol_idx][1] if best_sol_idx is not None else None,
        'char_count': best_char_count if best_char_count != float('inf') else 0,
        'passed': best_score,
        'total': len(task_data.get('train', [])),
        'test_passed': best_score == len(task_data.get('train', [])) and best_score > 0
    }

print(f"{'='*60}")
print("RESULTS SUMMARY")
print(f"{'='*60}")
print(f"Total tasks processed: {len(results)}")
solved_count = sum(1 for r in results.values() if r['test_passed'])
partial_count = sum(1 for r in results.values() if r['passed']>0 and not r['test_passed'])
unsolved_count = sum(1 for r in results.values() if r['passed']==0)

print(f"Fully solved (all train passed): {solved_count}")
print(f"Partially solved: {partial_count}")
print(f"Unsolved: {unsolved_count}")
print(f"\nSolve rate: {solved_count/len(results)*100:.1f}%")

# Show some solved examples
print(f"\n{'='*60}")
print("TOP 10 SOLVED TASKS (shortest code)")
print(f"{'='*60}")
solved_tasks = [(tid, r) for tid, r in results.items() if r['test_passed']]
solved_tasks.sort(key=lambda x: x[1]['char_count'])

for tid, r in solved_tasks[:10]:
    print(f"\n{tid}: {r['passed']}/{r['total']} tests passed")
    print(f"  Solution ({r['char_count']} chars): {r['sol_code']}")

# Save results to working directory
with open('/kaggle/working/code_golf_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print("Full results saved to: /kaggle/working/code_golf_results.json")
print(f"{'='*60}")




