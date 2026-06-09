task_num = 0  # Task 0 is just an illustrative example (and not eligible for points)


import sys
sys.path.append("/kaggle/input/google-code-golf-2025/code_golf_utils")
from code_golf_utils import *
show_legend()


examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])


%%writefile task.py
def p(g):
 for r, row in enumerate(g):
  for c, color in enumerate(row):
   if r and c and color==5 and g[r-1][c-1] not in [0,5]: g[r][c]=0
 return g


verify_program(task_num, examples)

