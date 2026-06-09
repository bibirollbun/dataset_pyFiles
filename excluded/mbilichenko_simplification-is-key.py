import sys
sys.path.append("/kaggle/input/google-code-golf-2025/code_golf_utils")
from code_golf_utils import *


task_num = 165


%%writefile task.py
def p(x,e=range):
 from collections import Counter as j,deque;o,h=len(x),len(x[0]);p=j(sum(x,[])).most_common()[0][0];n=[[0]*h for k in e(o)];i=[]
 for k in e(o):
  for l in e(h):
   if n[k][l]or x[k][l]==p:continue
   t=x[k][l];s=deque([(k,l)]);n[k][l]=1;r=[(k,l)]
   while s:
    y,m=s.popleft()
    for(u,b)in[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
     if 0<=y+u<o and 0<=m+b<h and not n[y+u][m+b]and x[y+u][m+b]==t:n[y+u][m+b]=1;s.append((y+u,m+b));r.append((y+u,m+b))
   c,c=[k for(k,l)in r],[k for(l,k)in r];i.append((t,r,(min(c),min(c),max(c),max(c))))
 if not i:return x
 i.sort(key=lambda n:len(n[1]),reverse=1);q=i[0];l=i[1:]
 if not l:return x
 r=j(k for(k,l,q)in l).most_common()[0][0];n={};[n.setdefault(k,[]).append(l)for(k,l)in q[1]];[k.sort()for k in n.values()];n={};[n.setdefault(m,[]).append(y)for(k,l,u)in l if k==r for(y,m)in l];[k.sort()for k in n.values()];u=[k for(k,l)in{k:[l for(l,y)in q[1]if y==k]for(l,k)in q[1]}.items()if k in n and n[k][-1]>=l[-1]and any(x[y][k]==p for y in e(l[0],o)if y not in l)];u and[x[l].__setitem__(k,r)for k in sorted(u)for l in e(min(l for(l,y)in q[1]if y==k),o)if l not in[l for(l,y)in q[1]if y==k]and x[l][k]==p];return x


verify_program(task_num, load_examples(task_num))


show_examples(load_examples(task_num)['train'])
show_examples(load_examples(task_num)['arc-gen'][101:104])


%%writefile task.py
def p(g):
 a=sum(g,[]);s=next(x for x in a[::-1] if x);f=(set(a)-{0,s}).pop()
 for j,c in enumerate(zip(*g)):
  if f in c and s in c[(i:=19-c[::-1].index(f))+1:]:
   for k in range(i,20): g[k][j]=g[k][j] or s
 return g


verify_program(task_num, load_examples(task_num))


task_num = 273


%%writefile task.py
l=enumerate
r=range
def p(u):
 u=[e[:]for e in u];f=[[e for(e,f)in l(e)if f==4]for e in u]
 for(q,m)in l(f):
  for(f,e)in l(m):
   for f in m[f+1:]:
    for p in r(q+1,len(u)):
     if u[p][e]==4and u[p][f]==4:
      for p in r(q+1,p):u[p][e+1:f]=[2]*(f-e-1)
 return u


verify_program(task_num, load_examples(task_num))


show_examples(load_examples(task_num)['train'])
show_examples(load_examples(task_num)['arc-gen'][101:104])


%%writefile task.py
def p(a):
 d={}
 for i,r in enumerate(a):
  if 4 in r:
   x=r.index(4);y=r.index(4,x+1)
   for t in a[d.setdefault((x,y),i)+1:i]:t[x+1:y]=[2]*~(x-y)
 return a


verify_program(task_num, load_examples(task_num))





from zlib import compress

def zip_src(src):
    compression_level = 9  # Max Compression
    
    # We prefer that compressed source not end in a quotation mark
    while (compressed := compress(src, compression_level))[-1] == ord('"'): 
        src += b"#"
    
    def sanitize(b_in):
        """Clean up problematic bytes in compressed b-string"""
        b_out = bytearray()
        for b in b_in:
            if b == 0:         
                b_out += b"\\x00"
            elif b == ord("\r"): 
                b_out += b"\\r"
            elif b == ord("\\"): 
                b_out += b"\\\\"
            else: 
                b_out.append(b)
        return b"" + b_out
    
    compressed = sanitize(compressed)
    
    delim = b'"""' if ord("\n") in compressed or ord('"') in compressed else b'"'
    
    return b"#coding:L1\nimport zlib\nexec(zlib.decompress(bytes(" + \
        delim + compressed + delim + \
        b',"L1")))'


import os
source = "/kaggle/input/google-golf-code-tasks-dataset-com"
submission = "/kaggle/working/submission"
os.makedirs(submission, exist_ok=True)
os.chdir(submission)


# Copy tasks into submission folder
processed_tasks = 0
for task_num in range(1, 401):
    path_in = f"{source}/task{task_num:03d}.py"
    path_out = f"{submission}/task{task_num:03d}.py"
    
    if not os.path.exists(path_in):
        continue
    
    try:
        with open(path_in, "rb") as fin:
            code = fin.read()
        with open(path_out, "wb") as fout:
            fout.write(code)
        processed_tasks += 1
    except Exception as e:
        print(f"Error processing task{task_num:03d}: {e}")

print(f"Processed {processed_tasks} tasks")


import os
source = "/kaggle/input/google-golf-code-tasks-dataset-com"
submission = "/kaggle/working/submission"

total_save = 0
processed_tasks = 0

os.makedirs(submission, exist_ok=True)

# Process tasks 1-400
for task_num in range(1, 401):
    path_in = f"{source}/task{task_num:03d}.py"
    path_out = f"{submission}/task{task_num:03d}.py"
    
    if not os.path.exists(path_in):
        continue
    
    try:
        with open(path_in, "rb") as task_in:
            task_src = task_in.read()

        # Only compress if file has content
        if len(task_src) > 0:
            zipped_src = zip_src(task_src)
            improvement = len(task_src) - len(zipped_src)
            
            # Use compressed version if it saves space
            if improvement > 0:
                task_src = zipped_src
                total_save += improvement
            
            with open(path_out, "wb") as task_out:
                task_out.write(task_src)
            
            processed_tasks += 1
    
    except Exception as e:
        print(f"Error processing task{task_num:03d}: {e}")
        continue

print(f"Processed {processed_tasks} tasks")
print(f"Saved {total_save}b with zlib compression")


import zipfile

submission_zip = f"{submission}.zip"

with zipfile.ZipFile(submission_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
    task_count = 0
    for task_num in range(1, 401):
        task_id = f"{task_num:03d}"
        src_path = f"{submission}/task{task_id}.py"
        
        if os.path.exists(src_path):
            zipf.write(src_path, arcname=f"task{task_id}.py")
            task_count += 1

print(f"Created submission zip with {task_count} tasks: {submission_zip}")

# Display zip file size
zip_size = os.path.getsize(submission_zip)
print(f"Submission zip size: {zip_size:,} bytes ({zip_size/1024:.1f} KB)")

