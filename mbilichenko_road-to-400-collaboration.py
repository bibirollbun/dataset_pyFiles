import zipfile, json, os, copy
import sys
sys.path.append("/kaggle/input/google-code-golf-2025/code_golf_utils")
from code_golf_utils import *


LOCAL_SCORE = False
source = "/kaggle/input/google-code-golf-2025-tasks-pub"


from zlib import compress

def zip_src(src):
    compression_level = 9  # Max Compression
    
    # We prefer that compressed source not end in a quotation mark
    while (compressed := compress(src, compression_level))[-1] == ord('"'): 
        src += b"#"
    
    def sanitize(b_in):
        """Clean up problematic bytes in compressed b-string"""
        b_out = bytearray()
        for b in b_in:
            if b == 0:         
                b_out += b"\\x00"
            elif b == ord("\r"): 
                b_out += b"\\r"
            elif b == ord("\\"): 
                b_out += b"\\\\"
            else: 
                b_out.append(b)
        return b"" + b_out
    
    compressed = sanitize(compressed)
    
    delim = b'"""' if ord("\n") in compressed or ord('"') in compressed else b'"'
    
    return b"#coding:L1\nimport zlib\nexec(zlib.decompress(bytes(" + \
        delim + compressed + delim + \
        b',"L1")))'



import os

total_save = 0
processed_tasks = 0

submission = "/kaggle/working/submission"
os.makedirs(submission, exist_ok=True)

# Process tasks 1-400
for task_num in range(1, 401):
    path_in = f"{source}/task{task_num:03d}.py"
    path_out = f"{submission}/task{task_num:03d}.py"
    
    if not os.path.exists(path_in):
        continue
    
    try:
        with open(path_in, "rb") as task_in:
            task_src = task_in.read()

        # Only compress if file has content
        if len(task_src) > 0:
            zipped_src = zip_src(task_src)
            improvement = len(task_src) - len(zipped_src)
            
            # Use compressed version if it saves space
            if improvement > 0:
                task_src = zipped_src
                total_save += improvement
            
            with open(path_out, "wb") as task_out:
                task_out.write(task_src)
            
            processed_tasks += 1
    
    except Exception as e:
        print(f"Error processing task{task_num:03d}: {e}")
        continue

print(f"Processed {processed_tasks} tasks")
print(f"Saved {total_save}b with zlib compression")


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


def check(solution, task_num, valall=False):
    task_data = load_examples(task_num)
    try:
        namespace = {}
        exec(solution, namespace)
        if 'p' not in namespace: return False
        all_examples = task_data['train'] + task_data['test'] + task_data['arc-gen']
        examples_to_check = all_examples if valall else all_examples[:3]
        for example in examples_to_check:
            input_grid = copy.deepcopy(example['input'])
            expected = example['output']
            try:
                actual = namespace['p'](input_grid)
                if actual != expected:
                    return False
            except:
                return False
        return True
    except Exception as e:
        return False

if LOCAL_SCORE:
    score = 0
    for task_num in range(1, 401):
        try:
            solution = open(f"{submission}/task" + str(task_num).zfill(3) + '.py','rb').read()
            if check(solution, task_num, valall=True):
                s = max([0.1,2500-len(solution)])
                score += s
                #print(task_num, '*'* 40)
            else: print(task_num, ":L")
        except: pass
    print('Score:', score)

