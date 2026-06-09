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


%%writefile task178.py
p=lambda g:[[b for a,b in zip([0]+a,a)if a^b]for a,b in zip(g,[[]]+g)if a!=b]


%%writefile task085.py
p=lambda g:[((t:=0)or[(t:=t^v)&v for v in b],b)[a!=c]for a,b,c in zip([0]+g,g,g[1:]+[0])]


%%writefile task377.py
p=lambda g,F=lambda d:[*zip(*[b for a,b in zip([0]+d,d)if a!=b])]:F(F(g))


%%writefile task346.py
def p(g):return[[next(r[j+1]for t,b,r in zip(g,g[2:],g[1:])for j in range(len(r)-2)if t[j]==t[j+1]==b[j+1]==r[j]==r[j+2]==b[j+2]>0)]]


%%writefile task091.py
def p(g):s=[r for r in g if 5 in r];t=s[0];i=g.index(t);return[x[(q:=t.index(5)):t.index(5,q+1)+1]for x in g[i-1:i-~len(s)]]



%%writefile task338.py
def p(g):
 n=len(g)
 def f(i,j):
  if i<n>j and g[i][j]<1:g[i][j]=1;f(i+1,j);f(i,j+1)
 for i in range(n):f(i,0);f(0,i)
 return[[3*(c<1)for c in r]for r in g]


%%writefile task359.py
p=lambda d:[[max(S:=r+C,key=S.count)for*C,in zip(*d)]for r in d]


%%writefile task008.py
def p(g):
 e=enumerate
 for _ in[0]*4:
  b=max(i for i,r in e(g)if 2in r);d=min(i for i,r in e(g)if 8in r);g=[*zip(*((d>b)*(d-b-1)*[[0]*len(g[0])]+(g[:b+1]+g[d:],g)[b>=d]))][::-1]
 return g


%%writefile task110.py
def p(g):R=range;f=lambda A:next(i for i in R(1,30)if 1>any(a*b*(a^b)for r in A for a,b in zip(r,r[i:])));t=f(g);s=f(zip(*g));return[[max(max(r[j%t::t])for r in g[i%s::s])for j in R(29)]for i in R(29)]


%%writefile task092.py
def p(g):
 o=[*map(list,g)]
 for _ in 0,0:
  for r,t in zip(o,g):
   for v in{*t}-{0}:a=t.index(v);b=len(t)-t[::-1].index(v);r[a:b]=[v]*(b-a)
  o=[*map(list,zip(*o))];g=[*zip(*g)]
 return o



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

