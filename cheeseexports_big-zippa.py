sources = [
 "/kaggle/input/arc-code-golf-starter-solutions",
 # Author: https://www.kaggle.com/cristianocalcagno
 "/kaggle/input/google-code-golf-2025-submit",
 # Author: https://www.kaggle.com/tonylica
]


failing = [
 "/kaggle/input/google-code-golf-2025-submit/task071.py",
 "/kaggle/input/google-code-golf-2025-submit/task074.py",
 "/kaggle/input/google-code-golf-2025-submit/task343.py",
]


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


! pip install python-minifier

import os
from python_minifier import minify
import warnings

warnings.filterwarnings("ignore")

total_save = 0
score = 0
submission = "/kaggle/working/submission"

os.makedirs(submission, exist_ok=True)

for task_num in range(1, 401):
 
 path_out = f"{submission}/task{task_num:03d}.py" 

 task_src = b"#" * 1_000_000
 for source in sources:
  path_in  = f"{source}/task{task_num:03d}.py"
  if not os.path.exists(path_in):continue
  if path_in in failing:continue

  with open(path_in, "r") as task_in:
   new_src = task_in.read()
   task_src = min([task_src, minify(new_src).encode(), new_src.encode()], key=len)
 zipped_src = zip_src(task_src)

 improvement = len(task_src) - len(zipped_src)

 if improvement > 0:
  task_src = zipped_src
  total_save += improvement
 score += max(1, 2500-len(task_src))
 with open(path_out, "wb") as task_out:
  task_out.write(task_src)

print(f"Saved {total_save}b with zlib")
print(f"Projected score: {score}")


import zipfile

with zipfile.ZipFile(f"{submission}.zip", "w") as zipf:
 for task_num in range(1, 401):
  task_id = f"{task_num:03d}"
  src_path = f"{submission}/task{task_id}.py"
  if os.path.exists(src_path):
   zipf.write(src_path, arcname=f"task{task_id}.py")

