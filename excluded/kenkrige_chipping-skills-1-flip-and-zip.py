import sys
sys.path.append("/kaggle/input/google-code-golf-2025/code_golf_utils")
from code_golf_utils import *


task_num = 234
show_examples(load_examples(task_num)['train'])
show_examples(load_examples(task_num)['arc-gen'][9:12])


example=load_examples(task_num)["arc-gen"][9];g=example["input"]
w=len(g[0])
[r for r in g if w-1!=r.count(0)]


def p(g):w=len(g[0]);u=[r for r in g if w-1!=r.count(0)];return[[0]*w]*(len(g)-len(u))+u


example=load_examples(task_num)["arc-gen"][9];
solution={"input":example["input"],"output":p(example["input"])}
show_examples([example,solution])


example=load_examples(task_num)["train"][0];
solution={"input":example["input"],"output":p(example["input"])}
show_examples([example,solution])


max([*filter(any,g)][0])


def p(g):
    for _ in'12':
        u=[r for r in g if r.count(max([*filter(any,g)][0]))!=1]
        g=((len(g)-len(u))*g[0:1]+u)[::-1]
    return g


example=load_examples(task_num)["train"][0];
solution={"input":example["input"],"output":p(example["input"])}
show_examples([example,solution])


example=load_examples(task_num)["train"][1];
solution={"input":example["input"],"output":p(example["input"])}
show_examples([example,solution])


h=len(g);w=len(g[0]);R=range
t=[[g[r][c]for r in R(h)]for c in R(w)]
T=[*zip(*g)]
print(len(g),len(g[0]))
print(len(t),len(t[0]))
print(len(T),len(T[0]))


def p(g):
    for _ in[0]*4:
        u=[r for r in g if r.count(max([*filter(any,g)][0]))!=1]
        g=[*zip(*((len(g)-len(u))*g[0:1]+u))][::-1]
    return g


example=load_examples(task_num)["train"][1];
solution={"input":example["input"],"output":p(example["input"])}
show_examples([example,solution])


%%writefile task.py
def p(g):
    for _ in[0]*4:
        u=[r for r in g if r.count(max([*filter(any,g)][0]))!=1]
        g=[*zip(*((len(g)-len(u))*g[0:1]+u))][::-1]
    return[[*r]for r in g]


%%writefile task.py
p=lambda g:[(g:=[*zip(*((len(g)-len(u:=[r for r in g if r.count(max(next(filter(any,g))))!=1]))*g[:1]+u))][::-1])for _ in[0]*4][3]


verify_program(task_num, load_examples(task_num))




