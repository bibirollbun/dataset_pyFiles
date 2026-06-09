import sys
sys.path.append("/kaggle/input/google-code-golf-2025/code_golf_utils")
from code_golf_utils import *
show_legend()
import warnings
warnings.simplefilter("ignore")


task_num = 15
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])


%%writefile task.py
def p(j,A=enumerate):
 c=[k[:]for k in j]
 for E,k in A(j):
  for W,l in A(k):
   for J in(-1,0,1):
    for a in(-1,0,1):
     if l and J|a and(l==2and J*a or l==1and not J*a):
      C=E+J;e=W+a
      if 0<=C<len(j)and 0<=e<len(j[0])and c[C][e]<1:c[C][e]=4+3*(l&1)
 return c


verify_program(task_num, examples)


%%writefile task.py
def p(j,A=enumerate):
 c=[k[:]for k in j]
 for E,k in A(j):
  for W,l in A(k):
   for J in(-1,0,1):
    for a in(-1,0,1):
     if l and J|a and(l==2and J*a or l==1and not J*a):
      C=E+J;e=W+a
      if 0<=C<9and 0<=e<9and c[C][e]<1:c[C][e]=4+3*(l&1)
 return c


verify_program(task_num, examples)


import pandas as pd
import numpy as np
df_list=[]
for task_num in range(1,400):
 examples = load_examples(task_num)
 h_list=[]
 w_list=[]
 t_list=["task"+str(task_num).zfill(3)]
 for arcgen in examples["arc-gen"]:
  g=arcgen["input"]
  h,w=len(g),len(g[0])
  h_list.append(h)
  w_list.append(w)
 if len(set(h_list))==1:
  t_list.append(h_list[0])
 if len(set(w_list))==1:
  t_list.append(w_list[0])
 df_list.append(t_list)
import pandas as pd
df=pd.DataFrame(df_list,columns=["task_num","hight","width"]).fillna(0)
df["hight"]=df["hight"].astype(int)
df["width"]=df["width"].astype(int)


df[df["hight"]!=0]


df.to_csv("task_hw,csv",index=False)

