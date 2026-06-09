task_num = 154
import sys
sys.path.append("/kaggle/input/google-code-golf-2025/code_golf_utils")
from code_golf_utils import *
show_legend()


examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])


%%writefile task.py
def p(g,k=0,R=range(15)):
 *g,=map(list,zip(*g))
 if max(r.count(2)for r in g)<5:
  b,*_,d=(j for r in g for j in R if r[j]==2)
  for r in g:
   for j in R:
    if r[j]>2and(d<j or j<b):r[j]=0;r[2*(b,d)[j>=b+d+1>>1]-j]=5
 return k*g or p(g,1)


verify_program(task_num, examples)


%%writefile task.py
import re
p=lambda g:[(g:=[[*map(int,re.sub('(.*[^0].*)020(.*)020(.*)',lambda m:'0'*(l:=len(m[1]))+'020'+m[1][::-1]+m[2][l:]+'020'+m[3],str(r)[1::3]))]for r in zip(*g[::-1])])for _ in g*4]and g


verify_program(task_num, examples)


%%writefile task.py
import re;p=lambda g:[g:=[[*map(int,re.sub(r'(.*5.*)(020)(.*)\2(.*)',lambda m:'0'*len(x:=m[1])+m[2]+x[::-1]+m[3][len(x):]+m[2]+m[4],str(r)[1::3]))]for r in zip(*g[::-1])]for _ in g*4][3]


verify_program(task_num, examples)




