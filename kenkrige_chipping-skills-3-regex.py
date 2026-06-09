import sys
sys.path.append("/kaggle/input/google-code-golf-2025/code_golf_utils")
from code_golf_utils import *


task_num = 96
show_examples(load_examples(task_num)['test']+load_examples(task_num)['arc-gen'][:2])


from IPython.display import display, Image
display(Image(filename='/kaggle/input/task096-pics/task096.png', width=800))


import pandas as pd
data =[
    [0,'Blue',1,1,0],
    [1,'Turquoise',8,3,0],
    [2,'Green',3,5,1],
    [3,'Purple',6,7,3],
    [4,'Red',2,9,3],
    [5,'Yellow',4,11,1],
      ]
df = pd.DataFrame(data, columns=['Square','Colour','Col Code' ,'Side', 'Middle Gap'])
df


S = {d[0]:((d[4]+1)//2,d[2]) for d in data}
S


R=range # Some golfer's housekeeping
b=1 # Background colour blue.
x=2*max(S)+1 # The size of the square output grid
[[b if(z[0]<(p:=S[z[1]])[0])else p[1]for j in R(x)if(z:=sorted([abs(j-x//2),abs(i-x//2)]))]for i in R(x)]


example=load_examples(task_num)["test"][0] # Grab the input for our example
show_examples([example])
g = example["input"]


import re
t=re.sub(r',\s','',str(g+[*zip(*g)]))# Prep the input for easier pattern search
t+=t[::-1] # This creates 4 versions of the input, seeing it from top, left, bottom and right
t 


b= int(max(t,key=t.count)) # Background colour as integer
b


c=4 # This is the colour. We will find yellow patterns.
re.findall(f'{c}+',t)


f=re.findall(f'{c}{c}[^]){c}]+{c}',t) 
print(f,len(f[0])-3) # We subtract the 3 yellows from our search result to get the length of the gap in the yellw square.


show_examples([example])


# Initialise the dict with square 0 in background colour.
# This because not all examples have a tiny square at the centre.
B={0:(0,b)}
# Loop through all colours except the background
for c in{*R(10)}-{b}:
    if(v:=re.findall(f'{c}+',t)): # The corner size
        m=len((re.findall(f'{c}{c}[^]){c}]+{c}',t)+[3*'0'])[0])-3 # The gap size
        l=len(max(v,key=len))*(1,2)[m>0] # Squares [0,1] do not have a gap(m=0) so no need to double corner length.
        S[(l+m)//2]=((m+1)//2,c) # Compileng the dict
x=2*max(S)+1 # Output grid main dimension
x, S


%%writefile task.py
import re
def p(g,R=range,f=re.findall):
 t=re.sub(r',\s','',str(g+[*zip(*g)]));t+=t[::-1];b=int(max(t,key=t.count));B={0:(0,b)}
 for c in{*R(10)}-{b}:
  if(v:=f(f'{c}+',t)):
   m=len((f(f'{c}{c}[^]){c}]+{c}',t)+[3*'0'])[0])-3
   l=len(max(v,key=len))*(1,2)[m>0]
   B[(l+m)//2]=((m+1)//2,c)
 x=2*max(B)+1
 return[[b if(z[0]<(p:=B[z[1]])[0])else p[1]for j in R(x)if(z:=sorted([abs(j-x//2),abs(i-x//2)]))]for i in R(x)]


verify_program(task_num, load_examples(task_num))




