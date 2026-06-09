import os
import json
import zipfile
from tqdm import tqdm
from rich import print
import numpy as np
from collections import defaultdict
from scipy.ndimage import label, binary_dilation, binary_erosion

def check_code_solution(solution_code, task_data):
    try:
        exec(solution_code, globals())
        all_examples = task_data['train'] + task_data['test'] + task_data['arc-gen']
        for example in all_examples[:3]:
            input_grid = example['input']
            expected = example['output']
            actual = p(input_grid)
            if actual != expected:
                return False
        return "Success"
    except Exception as e:
        return "Failed"

submisisons = [
    "/kaggle/input/25-solved-stater-neurips-2025-google-code-golf/submission"
]

simple_solution="""def p(g):
    return g"""

def get_bytes(code: str) -> int:
    return len(code.encode('utf-8'))

solved = 0
total_score = 0
os.makedirs("/kaggle/working/submission", exist_ok=True)


import re
import math
from functools import reduce


def remove_spaces(s):
    for c in ['[', ']', '(', ')', '{', '}', '=', '!', '<', '>', '+', '-', '/', '*', '%']:
        s = s.replace(' ' + c, c)
        s = s.replace(c + ' ', c)
    return s


def minimize_indentation(s):

    leading_spaces = [
        len(m.group(1))
        for m in re.finditer(r'^( +)(?=\S)', s, flags=re.MULTILINE)
    ]
    if not leading_spaces:
        return s

    unit = reduce(math.gcd, leading_spaces)
    if unit <= 1:
        return s

    def _shrink(match: re.Match) -> str:
        count = len(match.group(1))
        return ' ' * (count // unit)

    return re.sub(r'^( +)', _shrink, s, flags=re.MULTILINE)


n_tasks = 401

for task_num in tqdm(range(1, n_tasks)):
    task_id = f"{task_num:03d}"
    task_code = []
    task_data_path = f"/kaggle/input/google-code-golf-2025/task{task_id}.json"
    task_data = json.load(open(task_data_path))

    is_solved = False
    
    for submission_path in submisisons:
        task_code_path = f"{submission_path}/task{task_id}.py"
        if not os.path.exists(task_code_path):
            print('err')
            continue
        with open(task_code_path, 'r') as f:
            solution_code = f.read()
            solution_code = remove_spaces(solution_code)
            solution_code = minimize_indentation(solution_code)
        if check_code_solution(solution_code, task_data) == "Success":
            task_code.append({"code": solution_code, "bytes": get_bytes(solution_code)})
            is_solved = True

    if not task_code:
        task_code.append({"code": simple_solution, "bytes": get_bytes(simple_solution)})
        score = 0.001
    else:
        task_code_details = min(task_code, key=lambda x: x['bytes'])
        score = max(1, 2500 - task_code_details['bytes'])
        if is_solved:  # Only count as solved if it passes verification
            solved += 1
            print(f"[green]{task_id} - solved[/green]")

    total_score += score

    best_code = min(task_code, key=lambda x: x['bytes'])['code']
    with open(f"/kaggle/working/submission/task{task_id}.py", "w") as f:
        f.write(best_code)

# Create the submission zip
with zipfile.ZipFile("/kaggle/working/submission.zip", "w") as zipf:
    for task_num in range(1, n_tasks):
        task_id = f"{task_num:03d}"
        zipf.write(f"/kaggle/working/submission/task{task_id}.py", 
                   arcname=f"task{task_id}.py")

print(f"[green]Total solved: {solved} / 400[/green]")
print(f"[blue]LB Score: {total_score:.3f}[/blue]")


%%writefile task289.py
p=lambda g:(n:=len(set(sum(g,[]))-{0}),[[x for x in r for _ in range(n)]for r in g for _ in range(n)])[1]




