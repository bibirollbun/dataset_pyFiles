import sys
sys.path.append("/kaggle/input/google-code-golf-2025/code_golf_utils")
from code_golf_utils import *


task_num = 343
show_examples(load_examples(task_num)['train'])
show_examples(load_examples(task_num)['arc-gen'][:3])


example=load_examples(task_num)["train"][0];g=example["input"]
q=3 # This q value of is the all-important pattern size. Once calculated, the solution is simple.
[(r[:q]*15)[:15]for r in g]


q=3
example["output"]=[(r[:q]*15)[:15]for r in g]
show_examples([example])


%%writefile task.py
p=lambda g:[(r[:3]*15)[:15]for r in g]


verify_program(task_num, load_examples(task_num))


[r for r in zip(*g)if any(r)]


[*filter(any,[*zip(*g)])]


p=lambda g,q=1:p(g,q+1)if((k:=[*filter(any,[*zip(*g)])])[:q]*9)[:len(k)]!=k else[(r[:q]*15)[:15]for r in g]


example=load_examples(task_num)["train"][2];
solution={"input":example["input"],"output":p(example["input"])}
show_examples([example,solution])


%%writefile task.py
p=lambda g,q=1:p(g,q+1)if((k:=[*filter(any,[*zip(*g)])])[:q]*9)[:len(k)]!=k else[(r[:q]*15)[:15]for r in g]


verify_program(task_num, load_examples(task_num))


p=lambda g,q=1:p(g,q+1)if((k:=[*filter(any,zip(*g))])[:q]*9)[:len(k)]!=k else[(r[:q]*15)[:15]for r in g]


p=lambda g:[(r[:(6,8)[False]]*15)[:15]for r in g]


p=lambda g:[(r[:(8,6)[((k:=[*filter(any,zip(*g))])[:6]*9)[:len(k)]==k]]*5)[:15]for r in g]


p=lambda g:[(r[:(6,8)[(t:=[*zip(*g)])[:4]in[t[4:8],t[8:12]]]]*15)[:15]for r in g]


p=lambda g:[*zip(*((t:=[*zip(*g)])[:(6,8)[t[:4]in[t[4:8],t[8:12]]]]*15)[:15])]


p=lambda g:[*zip(*((t:=[*zip(*g)])[:6+2*(t[:4]in[t[4:8],t[8:12]])]*3)[:15])]


p=lambda g:[(r[:6+2*(r[:4]in(r[4:8],r[8:12]))]*3)[:15]for r in g]

