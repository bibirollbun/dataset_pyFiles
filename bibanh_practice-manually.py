!unzip /kaggle/input/ensemble-of-ensemble-of-ensemble/submission.zip -d ./submission


import sys
sys.path.append("/kaggle/input/google-code-golf-2025/code_golf_utils")
from code_golf_utils import *
show_legend()


import multiprocessing
import time

def run_with_timeout(func, args=(), kwargs={}, timeout=5):
    def wrapper(queue):
        try:
            result = func(*args, **kwargs)
            queue.put(result)
        except Exception as e:
            queue.put(e)

    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=wrapper, args=(queue,))
    p.start()
    p.join(timeout)
    
    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutError(f"Function timed out after {timeout} seconds")
    
    result = queue.get()
    if isinstance(result, Exception):
        raise result
    return result


import re
import ast

default_code = '''
def p(input):
    return 
'''

def update(task_num):
    file_num = f"{task_num:03}"
    filename = f"task{file_num}.py"

    with open('task.py', "r", encoding="utf-8") as f:
        code = f.read()
        
    print(f'Task num: {task_num} | File num: {file_num}')
    examples = load_examples(task_num)
    
    try:
        run_with_timeout(verify_program, args=(task_num, examples), timeout=10)  # timeout 10s
        with open('./submission/' + filename, "w", encoding="utf-8") as f:
            f.write(code)
    except TimeoutError:
        print("TIMEOUT")
        with open('./submission/' + filename, "w", encoding="utf-8") as f:
            f.write(default_code)
        print("="*60)
    except Exception as e:
        print(f"ERROR: {e}")
        with open('./submission/' + filename, "w", encoding="utf-8") as f:
            f.write(default_code)
        print("="*60)
    print(f'File {file_num} updated')
    print("="*60)


from tqdm import tqdm
ret = []
for task_num in tqdm(range(1, 401)):
    text = ""
    examples = load_examples(task_num)
    total_samples = len(examples['train']) + len(examples['test'])
    examples['train'] += examples['test']
    for i in range(total_samples):
        text += f'\nExample {i + 1}:\n - Input:\n'
        text += '],\n '.join(str(examples['train'][i]['input']).split('], '))
        text += '\n - Output:\n'
        text += '],\n '.join(str(examples['train'][i]['output']).split('], '))
        text += '\n'
    ret.append(text)
print(len(ret))


task_num = 1
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])



%%writefile task.py
def p(m):
 s=len(m)
 return[[m[i%s][j%s]if m[i//s][j//s]else 0for j in range(s*s)]for i in range(s*s)]



verify_program(task_num, examples)



update(task_num)


task_num = 2
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])



%%writefile task.py
from collections import*
def p(a):
 if not a or not a[0]:return[]
 r,c=len(a),len(a[0])
 q=deque([(i,j)for i in range(r)for j in range(c)if(i*j==0 or i==r-1 or j==c-1)and a[i][j]==0])
 while q:
  x,y=q.popleft();a[x][y]=1
  for dx,dy in[(0,1),(0,-1),(1,0),(-1,0)]:
   nx,ny=x+dx,y+dy
   if 0<=nx<r and 0<=ny<c and a[nx][ny]==0:q.append((nx,ny));a[nx][ny]=1
 return[[4 if v==0 else 0 if v==1 else v for v in row]for row in a]



verify_program(task_num, examples)



update(task_num)


task_num = 3
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])
print(ret[task_num - 1])


%%writefile task.py

def p(m):
 s,n=next((j,v)for j,v in enumerate(m[-1])if v)
 o=[[0]*10 for _ in range(10)]
 for i in range(10):
  for j in range(s,10):
   d=j-s
   if d%2==0:o[i][j]=n
   elif(i==0 and d%4==1)or(i==9 and d%4==3):o[i][j]=5
 return o



verify_program(task_num, examples)



update(task_num)


task_num = 6
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])



%%writefile task.py
def p(m):
 o=[]
 for r in m:
  try:
   s=r.index(5);l,R=r[:s],r[s+1:]
   o+=[[2if l[i]==R[i]==1else l[i]+R[i]if l[i]==R[i]else 0for i in range(s)]]
  except:o+=[[]]
 return o



verify_program(task_num, examples)



update(task_num)


task_num = 10
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])



%%writefile task.py
def p(m):
 n=len(m);c=[sum(r[j]==5for r in m)for j in range(n)]
 s=sorted({v for v in c if v},reverse=1)
 M={v:i+1for i,v in enumerate(s)}
 o=[r[:]for r in m]
 for i in range(n):
  for j in range(n):
   if o[i][j]==5:o[i][j]=M.get(c[j],0)
 return o



verify_program(task_num, examples)



update(task_num)


task_num = 32
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])



%%writefile task.py
def p(m):
 t=len(m);s=[[0]*len(m[0])for _ in m]
 for j in range(len(m[0])):
  c=[m[i][j]for i in range(t)if m[i][j]]
  for i,x in enumerate(c):s[t-len(c)+i][j]=x
 return s



verify_program(task_num, examples)



update(task_num)


task_num = 37
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])



%%writefile task.py
def p(m):
 R,C=len(m),len(m[0])
 D,O={},[r[:]for r in m]
 for r in range(R):
  for c in range(C):
   v=m[r][c]
   if v:D.setdefault(v,[]).append((r,c))
 for v in D:
  (y1,x1),(y2,x2)=D[v]
  dy=1if y2>y1 else-1
  dx=1if x2>x1 else-1
  for i in range(abs(y2-y1)+1):O[y1+i*dy][x1+i*dx]=v
 return O




verify_program(task_num, examples)



update(task_num)


task_num = 55
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])
print(ret[task_num - 1])


%%writefile task.py
def p(m):
 R,C=len(m),len(m[0])
 O=[r[:]for r in m]
 h1,h2=[r for r in range(R)if all(v==8 for v in m[r])]
 v1,v2=[c for c in range(C)if all(m[r][c]==8 for r in range(R))]
 for r in range(R):
  for c in range(C):
   if not O[r][c]:
    if r<h1 and v1<c<v2:O[r][c]=2
    elif h1<r<h2 and c<v1:O[r][c]=4
    elif h1<r<h2 and v1<c<v2:O[r][c]=6
    elif h1<r<h2 and c>v2:O[r][c]=3
    elif r>h2 and v1<c<v2:O[r][c]=1
 return O



verify_program(task_num, examples)



update(task_num)


task_num = 100
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])



%%writefile task.py
from collections import*
def p(m):
 r,c=len(m),len(m[0])
 v=set();a=0;z=0
 for i in range(r):
  for j in range(c):
   if m[i][j]<1and(i,j)not in v:
    q=deque([(i,j)]);v|={(i,j)}
    A={(i,j)};b=set();e=1;en=1
    while q:
     x,y=q.popleft()
     for u,vv in[(0,1),(1,0),(-1,0),(0,-1)]:
      X,Y=x+u,y+vv
      if not(0<=X<r>Y>=0):en=0;continue
      if(X,Y)in A:continue
      A|={(X,Y)}
      if m[X][Y]:b|={m[X][Y]}
      else:q.append((X,Y));v|={(X,Y)}
      e+=1
    if en and e>a and len(b)==1:a=e;z=b.pop()
 return [[z]*2]*2




verify_program(task_num, examples)



update(task_num)


task_num = 200
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])



%%writefile task.py

def p(m):
 s,n=next((j,v)for j,v in enumerate(m[-1])if v)
 o=[[0]*10 for _ in range(10)]
 for i in range(10):
  for j in range(s,10):
   d=j-s
   if d%2==0:o[i][j]=n
   elif(i==0 and d%4==1)or(i==9 and d%4==3):o[i][j]=5
 return o



verify_program(task_num, examples)



update(task_num)


task_num = 300
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])



%%writefile task.py

def p(m):
 a=[(i,j)for i,r in enumerate(m)for j,v in enumerate(r)if v]
 if not a:return[]
 from collections import Counter as C
 v=C(m[i][j]for i,j in a).most_common(1)[0][0]
 x=[(i,j)for i,j in a if m[i][j]==v]
 r0,c0=min(i for i,_ in x),min(j for _,j in x)
 r1,c1=max(i for i,_ in x)+1,max(j for _,j in x)+1
 return[m[i][c0:c1]for i in range(r0,r1)]




verify_program(task_num, examples)



update(task_num)


task_num = 301
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])



%%writefile task.py

def p(m):
 from collections import Counter as C
 a=[v for r in m for v in r if v]
 b=dict(C(a).most_common())
 w=len(m[0])
 r=[[0]*w for _ in range(len(m))]
 for i,k in enumerate(sorted(b,key=b.get,reverse=True)):
  r[-1-i][-b[k]:]=[k]*b[k]
 return r




verify_program(task_num, examples)



update(task_num)


task_num = 302
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])



%%writefile task.py

def p(m):
 a,b=len(m),len(m[0]);v=[[0]*b for _ in m];r=[]
 def f(i,j):
  q=[(i,j)];v[i][j]=1;c=[(i,j)];d=1
  while q:
   x,y=q.pop()
   for u,vv in[(0,1),(1,0),(0,-1),(-1,0)]:
    nx,ny=x+u,y+vv
    if not(0<=nx<a and 0<=ny<b):d=0;continue
    if m[nx][ny]<1 and not v[nx][ny]:v[nx][ny]=1;q+=[(nx,ny)];c+=[(nx,ny)]
  return c if d else[]
 for i in range(a):
  for j in range(b-1,-1,-1):
   if m[i][j]<1 and not v[i][j]:r+=[f(i,j)]
 r.sort(key=len,reverse=1)
 for i,x in enumerate(r):
  t=min(8,max(6,len(x)**.5+.5//1+5))
  for y in x:m[y[0]][y[1]]=t
 return m



verify_program(task_num, examples)



update(task_num)


task_num = 304
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])



%%writefile task.py
from collections import Counter
def p(m):
 n=len(m)
 c=Counter(e for r in m for e in r).most_common(1)[0][0]
 o=[[0]*n*n for _ in range(n*n)]
 for i in range(n):
  for j in range(n):
   if m[i][j]==c:
    for a in range(n):
     for b in range(n):o[i*n+a][j*n+b]=m[a][b]
 return o



verify_program(task_num, examples)



update(task_num)


task_num = 305
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])



%%writefile task.py
def p(m):
 n=len(m)
 f=[x for r in m for x in r if x]
 if not f:return m
 b=sorted(set(f))
 s=len(b)
 o=[[0]*n for _ in[0]*n]
 for i in range(n):
  for j in range(n):o[i][j]=b[(i+j)%s]
 return o



verify_program(task_num, examples)



update(task_num)


task_num = 309
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])



%%writefile task.py
def p(m):
 for i in m:
  for j in range(len(i)):
   if i[j]==7:i[j]=5
 return m


verify_program(task_num, examples)


update(task_num)


task_num = 310
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])



%%writefile task.py

from collections import Counter
def p(m):
 c=Counter(e for r in m for e in r if e).most_common()
 if not c: return []
 l=c[-1][0];rs=re=-1
 for i,r in enumerate(m):
  if l in r:
   if rs<0:rs=i
   re=i
 cs=ce=-1
 for i in range(len(m[0])):
  if any(m[j][i]==l for j in range(rs,re+1)):
   if cs<0:cs=i
   ce=i
 return[r[cs:ce+1]for r in m[rs:re+1]]



verify_program(task_num, examples)


update(task_num)


task_num = 312
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])



%%writefile task.py
def p(m):
 for r in m:
  for v in r:
   if v and v-5:
    r[:]=[v*(x==5)+x*(x!=5)for x in r];break
 return m




verify_program(task_num, examples)


update(task_num)


task_num = 315
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])



%%writefile task.py
def p(m):
 n=len(m)
 o=[[0]*n*n for _ in range(n*n)]
 for i in range(n):
  for j in range(n):
   if m[i][j]==2:
    for a in range(n):
     for b in range(n):o[i*n+a][j*n+b]=m[a][b]
 return o




verify_program(task_num, examples)


update(task_num)


task_num = 316
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])



%%writefile task.py
def p(m):
 n=3;v=[]
 for c in zip(*m):
  for x in c:
   if x:v+=[x];break
 v+=[0]*(n*n-len(v))
 return[v[i*n:i*n+n][::1-2*(i%2)]for i in range(n)]



verify_program(task_num, examples)


update(task_num)


task_num = 318
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])



%%writefile task.py
def p(m):
    return [[3 if m[r][c] or m[r+5][c] else 0 for c in range(4)] for r in range(4)]


verify_program(task_num, examples)


update(task_num)


task_num = 320
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])



%%writefile task.py
def p(m):
 r=len(m);c=len(m[0]);r_=[i[:]for i in m]
 for j in range(c):
  nz=[i for i in range(r)if m[i][j]];l=len(nz)//2
  for i in range(l):r_[nz[-1-i]][j]=8
 return r_



verify_program(task_num, examples)


update(task_num)


task_num = 321
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])



%%writefile task.py
def p(m):
    s = [0] + [c + 1 for c in range(len(m[0])) if any(r[c] == 2 for r in m)]
    return [[next((m[r][j + o] for o in s if m[r][j + o]), 0) for j in range(4)] for r in range(4)]


verify_program(task_num, examples)


update(task_num)


task_num = 322
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])



%%writefile task.py
def p(m):
 for c in range(len(m[0])):
  for r in range(len(m)):
   if m[r][c]:break
  else:continue
  for i in range(r,len(m)):m[i][c]=m[r][c]
 return m



verify_program(task_num, examples)


update(task_num)


task_num = 323
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])



%%writefile task.py
def p(m):
 r,c=len(m),len(m[0])
 o=[i[:]for i in m]
 i,j=[(i,j)for i in range(r)for j in range(c)if m[i][j]][0]
 for a,b in(((-1,1),(2,2)),((1,-1),(2,2))):
  x,y=i,j
  while 1:
   for _ in[0]*b[0]:
    x+=a[0]
    if 0<=x<r:o[x][y]=5
    else:break
   else:
    for _ in[0]*b[1]:
     y+=a[1]
     if 0<=y<c:o[x][y]=5
     else:break
    else:continue
   break
 return o



verify_program(task_num, examples)


update(task_num)


task_num = 325
examples = load_examples(task_num)
show_examples(examples['train'] + examples['test'])



%%writefile task.py
def p(m):
 r,c,n=len(m),len(m[0]),0
 v=[[0]*c for _ in m]
 def d(y,x):
  s=[(y,x)]
  v[y][x]=1
  while s:
   y,x=s.pop()
   for dy,dx in[(-1,0),(1,0),(0,-1),(0,1)]:
    ny,nx=y+dy,x+dx
    if 0<=ny<r and 0<=nx<c and m[ny][nx] and not v[ny][nx]:
     v[ny][nx]=1
     s+=[(ny,nx)]
 for y in range(r):
  for x in range(c):
   if m[y][x]and not v[y][x]:
    n+=1;d(y,x)
 return[[8*(i==j)for j in range(n)]for i in range(n)]



verify_program(task_num, examples)


update(task_num)


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

submission_path = "/kaggle/working/submission"


simple_solution="""def p(g):
    return g"""

def get_bytes(code: str) -> int:
    return len(code.encode('utf-8'))

solved = 0
total_score = 0
os.makedirs("/kaggle/working/submission", exist_ok=True)

for task_num_id in tqdm(range(1, 401)):
    task_id = f"{task_num_id:03d}"
    task_code = []
    task_data_path = f"/kaggle/input/google-code-golf-2025/task{task_id}.json"
    task_data = json.load(open(task_data_path))

    is_solved = False
    task_code_path = f"{submission_path}/task{task_id}.py"
    if not os.path.exists(task_code_path):
        continue
    with open(task_code_path, 'r') as f:
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
        print(f"[green]{task_id} - solved[/green] with score: {score}")
    total_score += score

    best_code = min(task_code, key=lambda x: x['bytes'])['code']
    with open(f"/kaggle/working/submission/task{task_id}.py", "w") as f:
        f.write(best_code)


with zipfile.ZipFile("/kaggle/working/submission.zip", "w") as zipf:
    for task_num_id in range(1, 401):
        task_id = f"{task_num_id:03d}"
        zipf.write(f"/kaggle/working/submission/task{task_id}.py", 
                   arcname=f"task{task_id}.py")

print(f"[green]Total solved: {solved} / 400[/green]")
print(f"[blue]LB Score: {total_score:.3f}[/blue]")







