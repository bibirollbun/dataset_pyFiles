# ğŸ�† Google Code Golf 2025 Tony Li's SubmissionğŸ�†

import os
import zipfile

# Input and output paths
source = "/kaggle/input/google-code-golf-2025-submit"
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


%%writefile task327.py
p=lambda g,l=[0]*3:[l:=[*map(max,[0]+l*2,r+[0]*3)]for r in g+[l]*3]


%%writefile task040.py
p=lambda g,e=enumerate:(a:=g[0][0],[[(a,g[9][9])[(j,i)[a==g[0][9]]>4]*(v==3)or v for j,v in e(r)]for i,r in e(g)])[1]


%%writefile task362.py
def p(g):c=str(g).count('5');m=max(g);r=range(10);return[[m[0]*(i==g.index(m)+c or j==g[0].index(m[0])-c)for j in r]for i in r]


%%writefile task014.py
p=lambda g,f=filter:(s:=sum(g,[]),v:=min({*s}-{0},key=s.count),[*zip(*f(any,zip(*f(any,[[v*(x==v)for x in r]for r in g]))))])[2]


%%writefile task294.py
def p(g):
 for i in range(1,9):
  for j in range(1,9):
   if g[i+1][j]*g[i][j+1]*g[i-1][j]*g[i][j-1]:g[i][j]=2
 return g



%%writefile task095.py
p=lambda g,A=enumerate:[[C|any(any(R[A and A-1:A+2])for R in g[B and B-1:B+2])for A,C in A(r)]for B,r in A(g)]


%%writefile task063.py
def p(g):c=len(g)-1;e=enumerate;return[[v|3*(i*j*(c-i)*(c-j)and sum(r[1:-1])*sum(t[j]for t in g[1:-1])<1)for j,v in e(r)]for i,r in e(g)]



%%writefile task036.py
def p(g):A=sum(g,[]);C=A.index;D=A[::-1].index;B=max(A,key=lambda v:C(v)+D(v));G,H=C(B)//30,(899-D(B))//30;E=g[G:H+1];F=[A for(A,C)in enumerate(zip(*E))if B in C];return[A[F[0]:F[-1]+1]for A in E]


%%writefile task070.py
def p(g):e=enumerate;x,y=zip(*((i,j)for i,r in e(g)for j,v in e(r)if v>7));return[[v+2*((min(x)<=i<=max(x))&(min(y)<=j<=max(y))&v&1)for j,v in e(r)]for i,r in e(g)]


%%writefile task243.py
def p(g,e=enumerate):
 for i,r in e(g):
  for j,a in e(r):
   if a<1and 1in(g[i-(i>0)][j],r[j-(j>0)],g[i+(i+1<len(g))][j],r[j+(j+1<len(r))]):r[j]=1;return p(g)
 return g



# Create submission zip file
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

print("ğŸ�‰ Submission Complete! ğŸ�‰")

