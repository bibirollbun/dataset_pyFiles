task_num = 243
import sys
sys.path.append("/kaggle/input/google-code-golf-2025/code_golf_utils")
from code_golf_utils import *
show_legend()


examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])


%%writefile task.py
p=lambda g,k=79:-k*g or p([[x or v==1for x,v in zip(r,[0]+r)]for*r,in zip(*g[::-1])],k-1)


verify_program(task_num, examples)




