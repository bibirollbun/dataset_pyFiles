import sys
sys.path.append("/kaggle/input/google-code-golf-2025/code_golf_utils")
from code_golf_utils import *


import copy
import importlib.util
import json
import re
import sys
import traceback
import numpy as np
import os
import shutil
from tqdm import tqdm 

def simple_verify_program(task_num, examples):
    task_name, task_path = "task_with_imports", "/kaggle/working/task.py"
    spec = importlib.util.spec_from_file_location(task_name, task_path)
    if spec is None:
        print("Error: Unable to import task.py.")
        return
    module = sys.modules[task_name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "p"):
        print("Error: Unable to locate function p() in task.py.")
        return
    program = getattr(module, "p")
    if not callable(program):
        print("Error: Function p() in task.py is not callable.")
        return

    def verify(example_subset):
        right, wrong, expected, error = 0, 0, None, ""
        for example in example_subset:
            example_copy = copy.deepcopy(example)
            try:
                result = program(example_copy["input"])
                result = json.dumps(result)
                result = result.replace("true", "1").replace("false", "0")
                unsafe_chars = re.compile(r"[^0-9,\[\]\s\.]")
                if unsafe_chars.search(result):
                    raise ValueError(f"Invalid output from user code: {result[:500]}")
                result = json.loads(result)
                user_output = np.array(result)
                label_output = np.array(example_copy["output"])
                if numpy.array_equal(user_output, label_output):
                    right += 1
                else:
                    expected = copy.deepcopy(example)
                    wrong += 1
            except:
                error = traceback.format_exc()
                wrong += 1
                #if error: print(f"Error: {error}")
        return right, wrong, expected
    
    arc_agi_right, arc_agi_wrong, arc_agi_expected = verify(examples["train"] + examples["test"])
    arc_gen_right, arc_gen_wrong, arc_gen_expected = verify(examples["arc-gen"])
    #print(f"Results on ARC-AGI examples: {arc_agi_right} pass, {arc_agi_wrong} fail")
    #print(f"Results on ARC-GEN examples: {arc_gen_right} pass, {arc_gen_wrong} fail")
    return arc_agi_right, arc_agi_wrong, arc_gen_right, arc_gen_wrong


import os
import zipfile
from io import BytesIO
import warnings
from itertools import combinations, permutations

# Ignore SyntaxWarning globally
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Paths containing submission.zip
paths = ["/kaggle/input/get-started-step-by-step-toolkit-for-newcomers",
         "/kaggle/input/python-minifier-applied",
         "/kaggle/input/road-to-400-collaboration-39-unsolved-tasks", 
         "/kaggle/input/road-to-400-collaboration", 
         "/kaggle/input/code-golf-ensemble-local-score-391-400-dsl",
         "/kaggle/input/oh-barnacles"
        ]

for idx, path in enumerate(paths):
    dest_folder = f"/kaggle/working/submission_{idx}"
    os.makedirs(dest_folder, exist_ok=True)
    
    zip_path = os.path.join(path, "submission.zip")
    if not os.path.exists(zip_path):
        print(f"Zip not found: {zip_path}")
        continue

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_folder)

    print(f"Extracted all files from {zip_path} → {dest_folder}")


src = "/kaggle/input/gcgc-data-solutions"
dst = f"/kaggle/working/submission_{len(paths)}"
os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    if f.endswith(".py"):
        shutil.copy(os.path.join(src, f), dst)
paths.append(src)


from tqdm import tqdm

submission = "/kaggle/working/submission"
os.makedirs(submission, exist_ok=True)

lb_score = 0
unsolved_tasks = list()
# Process each task
num_tasks = 400
for task_num in range(1, num_tasks + 1):
    task_name = f"task{task_num:03d}.py"
    smallest_size = None
    smallest_idx = 0
    
    # Check all submission.zip in paths
    for idx in range(len(paths)):
        folder = f"submission_{idx}"
        task_path = os.path.join(folder, task_name)

        if os.path.exists(task_path):
            # Check the solution
            shutil.copy(task_path, "task.py")
            examples = load_examples(task_num)
            agi_right, agi_wrong, agen_right, gen_wrong = simple_verify_program(task_num, examples)
            
            if agi_wrong + gen_wrong == 0:
                size = os.path.getsize(task_path)
                if smallest_size is None or size < smallest_size:
                    smallest_size = size
                    smallest_idx = idx
    
    smallest_file = os.path.join(f"submission_{smallest_idx}", task_name)
    if os.path.exists(smallest_file):
        # Add smallest valid file to final zip
        if smallest_size != None:
            mark = "" 
        else:
            agi_right, agi_wrong, agen_right, gen_wrong = simple_verify_program(task_num, examples)
            mark = f" >>> agi_wrong: {agi_wrong}, gen_wrong: {gen_wrong}"
        print("Task_id = {}: the best solution ({} bytes) = {}".format(task_num, smallest_size, paths[smallest_idx]) + mark)
        shutil.copy(smallest_file, os.path.join(submission, task_name))
        
        if smallest_size == None: #No correct answer found
            unsolved_tasks.append(task_num)
            lb_score += 0.001
        else:
            lb_score += max(1, 2500 - smallest_size)
    else:
        unsolved_tasks.append(task_num)
        print("Task_id = {}: No solution found".format(task_num))


import zipfile

submission_zip = f"{submission}.zip"

with zipfile.ZipFile(submission_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
    task_count = 0
    for task_num in range(1, 401):
        task_id = f"{task_num:03d}"
        src_path = f"{submission}/task{task_id}.py"
        
        if os.path.exists(src_path):
            zipf.write(src_path, arcname=f"task{task_id}.py")
            task_count += 1

print(f"Created submission zip with {task_count} tasks: {submission_zip}")

# Display zip file size
zip_size = os.path.getsize(submission_zip)
print(f"Submission zip size: {zip_size:,} bytes ({zip_size/1024:.1f} KB)")


print("LB Score: ", lb_score)
print(f"{len(unsolved_tasks)} Unsolved Tasks: {unsolved_tasks}")


# Task visualization: https://www.kaggle.com/code/jacekwl/a-bit-more-of-code-golf-255-400-visualization

for idx in range(1,len(unsolved_tasks)+1): # only display 100 at once to not cause any memory issues
    task_num = unsolved_tasks[idx-1]
    task_id = f"{task_num:03d}"
    examples = load_examples(task_num)
    show_examples(examples['train'][:1] + examples['test'][:1] )
    plt.figure(idx).suptitle(
        f"task{task_id}" + [' : solved', ' : unsolved'][task_num in unsolved_tasks],
        fontsize=16,color=['green', 'red'][task_num in unsolved_tasks])

