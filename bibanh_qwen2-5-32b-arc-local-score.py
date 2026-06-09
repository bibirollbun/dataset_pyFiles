import os
import json
import zipfile
from tqdm import tqdm
from rich import print
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
    "/kaggle/input/liah-public-best"
]


simple_solution="""def p(g):
    return g"""

def get_bytes(code: str) -> int:
    return len(code.encode('utf-8'))

solved = 0
total_score = 0
os.makedirs("/kaggle/working/submission", exist_ok=True)

for task_num in tqdm(range(1, 401)):
    task_id = f"{task_num:03d}"
    task_code = []
    task_data_path = f"/kaggle/input/google-code-golf-2025/task{task_id}.json"
    task_data = json.load(open(task_data_path))

    is_solved = False
    for submission_path in submisisons:
        task_code_path = f"{submission_path}/task{task_id}.py"
        if not os.path.exists(task_code_path):
            continue
        with open(task_code_path, 'r', encoding="utf-8", errors="replace") as f:
            solution_code = f.read()
        if check_code_solution(solution_code, task_data) == "Success":
            task_code.append({"code": solution_code, "bytes": get_bytes(solution_code)})
            is_solved = True

    if not task_code:
        task_code.append({"code": simple_solution, "bytes": get_bytes(simple_solution)})
        score = 0.001
    else:
        task_code_details = min(task_code, key=lambda x: x['bytes'])
        score = max(1, 2500 - task_code_details['bytes'])
        solved += 1
        print(f"[green]{task_id} - solved[/green]")

    total_score += score

    best_code = min(task_code, key=lambda x: x['bytes'])['code']
    with open(f"/kaggle/working/submission/task{task_id}.py", "w") as f:
        f.write(best_code)



print(f"[green]Total solved: {solved} / 400[/green]")
print(f"[blue]LB Score: {total_score:.3f}[/blue]")


source = "/kaggle/input/liah-public-best"


from zlib import compress

def zip_src(src):
 compression_level = 9 # Max Compression

 # We prefer that compressed source not end in a quotation mark
 while (compressed := compress(src, compression_level))[-1] == ord('"'): src += b"#"

 def sanitize(b_in):
  """Clean up problematic bytes in compressed b-string"""
  b_out = bytearray()
  for b in b_in:
   if   b==0:         b_out += b"\\x00"
   elif b==ord("\r"): b_out += b"\\r"
   elif b==ord("\\"): b_out += b"\\\\"
   else: b_out.append(b)
  return b"" + b_out

 compressed = sanitize(compressed)

 delim = b'"""' if ord("\n") in compressed or ord('"') in compressed else b'"'

 return b"#coding:L1\nimport zlib\nexec(zlib.decompress(bytes(" + \
  delim + compressed + delim + \
  b',"L1")))'


import os

total_save = 0

submission = "/kaggle/working/submission"

os.makedirs(submission, exist_ok=True)

for task_num in range(1, 401):
 path_in  = f"{source}/task{task_num:03d}.py"
 path_out = f"{submission}/task{task_num:03d}.py" 
    
 if not os.path.exists(path_in):continue

 with open(path_in, "rb") as task_in:
  task_src = task_in.read()
  
 zipped_src = zip_src(task_src)

 improvement = len(task_src) - len(zipped_src)

 if improvement > 0:
  task_src = zipped_src
  total_save += improvement

 
 with open(path_out, "wb") as task_out:
  task_out.write(task_src)

print(f"Saved {total_save}b with zlib")



import zipfile

with zipfile.ZipFile(f"{submission}.zip", "w") as zipf:
 for task_num in range(1, 401):
  task_id = f"{task_num:03d}"
  src_path = f"{submission}/task{task_id}.py"
  if os.path.exists(src_path):
   zipf.write(src_path, arcname=f"task{task_id}.py")

