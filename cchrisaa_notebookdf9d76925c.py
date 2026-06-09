# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import os,json,zipfile
from collections import deque,Counter
import numpy as np

# Expanded ultra-compact pattern library
P={
'i':'p=lambda g:g',
'h':'p=lambda g:[r[::-1]for r in g]',
'v':'p=lambda g:g[::-1]',
'1':'p=lambda g:[list(r)for r in zip(*g[::-1])]',
'2':'p=lambda g:[r[::-1]for r in g[::-1]]',
'3':'p=lambda g:[list(r)for r in zip(*g)][::-1]',
't':'p=lambda g:[list(r)for r in zip(*g)]',
'c':'p=lambda g:[r[1:-1]for r in g[1:-1]]',
'd':'p=lambda g:[r[::2]for r in g[::2]]',
'u':'p=lambda g:[x for r in g for x in([v for v in r for _ in(0,1)])for _ in(0,1)]',
'x':'p=lambda g:[r+r for r in(g+g)]',
'H':'p=lambda g:[r+r[::-1]for r in g]',
'V':'p=lambda g:g+g[::-1]',
'f':'p=lambda g:[[max(set(sum(g,[])),key=lambda x:sum(r.count(x)for r in g))]*len(g[0])]*len(g)',
's':'p=lambda g:[[0if i<len(g)-1and j<len(g[0])-1and g[i][j]!=g[i+1][j+1]and g[i+1][j+1]else g[i][j]for j in range(len(g[0]))]for i in range(len(g))]',
'o':'p=lambda g:[[g[i][j]if g[i][j]and any(i+a<0 or i+a>=len(g)or j+b<0 or j+b>=len(g[0])or not g[i+a][j+b]for a,b in((0,1),(0,-1),(1,0),(-1,0)))else 0for j in range(len(g[0]))]for i in range(len(g))]',
'F':'p=lambda g,c=1:[[c]*len(g[0])]*len(g)',
'M':'p=lambda g,m={}:[[m.get(x,x)for x in r]for r in g]',
}

class FinalDetector:
 """Final optimized pattern detector with bug fixes"""
 
 def detect(s,examples):
  """Main detection pipeline"""
  if not examples:return'i',{},0
  
  e=[(np.array(i),np.array(o))for i,o in examples]
  best=('i',{},0)
  
  # Fix: Ensure confidence never exceeds 1.0
  def update_best(result):
   nonlocal best
   conf=min(result[2],1.0)  # Cap at 1.0
   if conf>best[2]:
    best=(result[0],result[1],conf)
    return conf==1.0
   return False
  
  # Strategy 1: Geometric transformations
  result=s._geometric(e)
  if update_best(result):return best
  
  # Strategy 2: Scaling operations
  result=s._scaling(e)
  if update_best(result):return best
  
  # Strategy 3: Mirror/tile operations
  result=s._mirror(e)
  if update_best(result):return best
  
  # Strategy 4: Color operations
  result=s._color(e)
  if update_best(result):return best
  
  # Strategy 5: Object extraction
  result=s._objects(e)
  if update_best(result):return best
  
  # Strategy 6: Pattern finding
  result=s._patterns(e)
  if update_best(result):return best
  
  # Strategy 7: Subgrid operations
  result=s._subgrid(e)
  if update_best(result):return best
  
  # Strategy 8: Composite transformations
  result=s._composite(e)
  if update_best(result):return best
  
  return best
 
 def _geometric(s,e):
  """Geometric transformations"""
  tests=[
   ('i',lambda i,o:np.array_equal(i,o)),
   ('h',lambda i,o:np.array_equal(np.fliplr(i),o)),
   ('v',lambda i,o:np.array_equal(np.flipud(i),o)),
   ('1',lambda i,o:np.array_equal(np.rot90(i),o)),
   ('2',lambda i,o:np.array_equal(np.rot90(i,2),o)),
   ('3',lambda i,o:np.array_equal(np.rot90(i,3),o)),
   ('t',lambda i,o:i.shape[0]==i.shape[1]and np.array_equal(i.T,o)),
  ]
  
  best=('i',{},0)
  for name,test in tests:
   matches=sum(1 for i,o in e if test(i,o))
   conf=matches/len(e)
   if conf>best[2]:best=(name,{},conf)
  
  return best
 
 def _scaling(s,e):
  """Scaling operations"""
  best=('i',{},0)
  
  # Scale up 2x
  matches=sum(1 for i,o in e if o.shape==(i.shape[0]*2,i.shape[1]*2)and np.array_equal(np.repeat(np.repeat(i,2,0),2,1),o))
  conf=matches/len(e)
  if conf>best[2]:best=('u',{},conf)
  
  # Scale down 2x
  matches=sum(1 for i,o in e if i.shape[0]%2==0 and i.shape[1]%2==0 and o.shape==(i.shape[0]//2,i.shape[1]//2)and np.array_equal(i[::2,::2],o))
  conf=matches/len(e)
  if conf>best[2]:best=('d',{},conf)
  
  # Crop center
  matches=sum(1 for i,o in e if i.shape[0]>2 and i.shape[1]>2 and o.shape==(i.shape[0]-2,i.shape[1]-2)and np.array_equal(i[1:-1,1:-1],o))
  conf=matches/len(e)
  if conf>best[2]:best=('c',{},conf)
  
  return best
 
 def _mirror(s,e):
  """Mirror and tile operations"""
  best=('i',{},0)
  
  # Horizontal mirror
  matches=sum(1 for i,o in e if o.shape==(i.shape[0],i.shape[1]*2)and np.array_equal(np.hstack([i,np.fliplr(i)]),o))
  conf=matches/len(e)
  if conf>best[2]:best=('H',{},conf)
  
  # Vertical mirror
  matches=sum(1 for i,o in e if o.shape==(i.shape[0]*2,i.shape[1])and np.array_equal(np.vstack([i,np.flipud(i)]),o))
  conf=matches/len(e)
  if conf>best[2]:best=('V',{},conf)
  
  # Tile 2x2
  matches=sum(1 for i,o in e if o.shape==(i.shape[0]*2,i.shape[1]*2)and np.array_equal(np.tile(i,(2,2)),o))
  conf=matches/len(e)
  if conf>best[2]:best=('x',{},conf)
  
  return best
 
 def _color(s,e):
  """Color-based operations"""
  # Color mapping
  global_map=None
  valid=True
  
  for inp,out in e:
   if inp.shape!=out.shape:
    valid=False;break
   
   local_map={}
   for r in range(inp.shape[0]):
    for c in range(inp.shape[1]):
     iv,ov=int(inp[r,c]),int(out[r,c])
     if iv in local_map:
      if local_map[iv]!=ov:
       valid=False;break
     else:
      local_map[iv]=ov
   
   if not valid:break
   
   if global_map is None:
    global_map=local_map
   elif global_map!=local_map:
    valid=False;break
  
  if valid and global_map and not all(k==v for k,v in global_map.items()):
   return('M',{'m':global_map},1.0)
  
  # Fill with single color
  fill_val=None
  matches=0
  for inp,out in e:
   if inp.shape==out.shape:
    unique=np.unique(out)
    if len(unique)==1:
     val=int(unique[0])
     if fill_val is None:fill_val=val
     if val==fill_val:matches+=1
  
  if matches==len(e)and fill_val is not None:
   return('F',{'c':fill_val},1.0)
  
  # Flood with dominant color
  matches=sum(1 for i,o in e if i.shape==o.shape and len(np.unique(i))>0 and np.all(o==np.bincount(i.flatten()).argmax()))
  conf=matches/len(e)
  if conf>0:return('f',{},conf)
  
  return('i',{},0)
 
 def _objects(s,e):
  """Object extraction patterns"""
  best=('i',{},0)
  
  # Extract specific colors
  for color in range(1,10):
   matches=0
   for inp,out in e:
    if inp.shape==out.shape:
     expected=np.where(inp==color,inp,0)
     if np.array_equal(expected,out):matches+=1
   
   conf=matches/len(e)
   if conf>best[2]:best=('i',{},conf)
  
  # Extract largest object
  matches=0
  for inp,out in e:
   objects=s._find_objects(inp)
   if objects:
    largest=max(objects,key=len)
    if s._matches_extraction(inp,out,largest):matches+=1
  
  conf=matches/len(e)
  if conf>best[2]:best=('i',{},conf)
  
  return best
 
 def _patterns(s,e):
  """Pattern-based operations"""
  best=('i',{},0)
  
  # Outline detection
  matches=0
  for inp,out in e:
   if inp.shape==out.shape:
    h,w=inp.shape
    expected=np.zeros_like(inp)
    for r in range(h):
     for c in range(w):
      if inp[r,c]!=0:
       edge=any(r+dr<0 or r+dr>=h or c+dc<0 or c+dc>=w or inp[r+dr,c+dc]==0 for dr,dc in[(0,1),(0,-1),(1,0),(-1,0)])
       if edge:expected[r,c]=inp[r,c]
    if np.array_equal(expected,out):matches+=1
  
  conf=matches/len(e)
  if conf>best[2]:best=('o',{},conf)
  
  # Staircase pattern
  matches=0
  for inp,out in e:
   if inp.shape==out.shape:
    h,w=inp.shape
    expected=inp.copy()
    for r in range(h-1):
     for c in range(w-1):
      if inp[r,c]!=inp[r+1,c+1]and inp[r+1,c+1]!=0:
       expected[r,c]=0
    if np.array_equal(expected,out):matches+=1
  
  conf=matches/len(e)
  if conf>best[2]:best=('s',{},conf)
  
  return best
 
 def _subgrid(s,e):
  """Subgrid extraction/padding"""
  best=('i',{},0)
  
  # Output is subregion of input
  matches=0
  for inp,out in e:
   if s._contains(inp,out):matches+=1
  
  conf=matches/len(e)
  if conf>best[2]:best=('i',{},conf)
  
  # Input is subregion of output
  matches=0
  for inp,out in e:
   if s._contains(out,inp):matches+=1
  
  conf=matches/len(e)
  if conf>best[2]:best=('i',{},conf)
  
  return best
 
 def _composite(s,e):
  """Composite transformations"""
  best=('i',{},0)
  
  # Try each rotation
  for rot in range(4):
   matches=0
   for inp,out in e:
    rotated=np.rot90(inp,rot)
    if np.array_equal(rotated,out):matches+=1
   
   conf=matches/len(e)
   if conf>best[2]:
    pattern=['i','1','2','3'][rot]
    best=(pattern,{},conf)
  
  # Try flip + rotation combinations
  for flip_type in['h','v']:
   for rot in range(4):
    matches=0
    for inp,out in e:
     if flip_type=='h':flipped=np.fliplr(inp)
     else:flipped=np.flipud(inp)
     rotated=np.rot90(flipped,rot)
     if np.array_equal(rotated,out):matches+=1
    
    conf=matches/len(e)
    if conf>best[2]:
     # Use simpler single pattern
     if conf>=0.8:
      pattern=['i','1','2','3'][rot]if rot>0 else flip_type
      best=(pattern,{},conf)
  
  return best
 
 def _find_objects(s,grid):
  """Find connected components"""
  visited=np.zeros_like(grid,dtype=bool)
  objects=[]
  
  for r in range(grid.shape[0]):
   for c in range(grid.shape[1]):
    if grid[r,c]!=0 and not visited[r,c]:
     obj=[]
     queue=[(r,c)]
     color=grid[r,c]
     
     while queue:
      cr,cc=queue.pop(0)
      if cr<0 or cr>=grid.shape[0]or cc<0 or cc>=grid.shape[1]:continue
      if visited[cr,cc]or grid[cr,cc]!=color:continue
      
      visited[cr,cc]=True
      obj.append((cr,cc))
      
      for dr,dc in[(0,1),(0,-1),(1,0),(-1,0)]:
       queue.append((cr+dr,cc+dc))
     
     objects.append(obj)
  
  return objects
 
 def _matches_extraction(s,inp,out,obj):
  """Check if output matches object extraction"""
  if not obj:return False
  
  rows,cols=zip(*obj)
  minr,maxr,minc,maxc=min(rows),max(rows),min(cols),max(cols)
  
  if out.shape!=(maxr-minr+1,maxc-minc+1):return False
  
  for r,c in obj:
   if out[r-minr,c-minc]!=inp[r,c]:return False
  
  # Check non-object cells are zero
  for r in range(out.shape[0]):
   for c in range(out.shape[1]):
    if (r+minr,c+minc)not in obj and out[r,c]!=0:
     return False
  
  return True
 
 def _contains(s,large,small):
  """Check if large contains small"""
  if small.shape[0]>large.shape[0]or small.shape[1]>large.shape[1]:
   return False
  
  for r in range(large.shape[0]-small.shape[0]+1):
   for c in range(large.shape[1]-small.shape[1]+1):
    if np.array_equal(large[r:r+small.shape[0],c:c+small.shape[1]],small):
     return True
  
  return False

def apply(pattern,params,grid):
 """Apply pattern"""
 G=np.array(grid)
 
 if pattern=='i':return grid
 if pattern=='h':return np.fliplr(G).tolist()
 if pattern=='v':return np.flipud(G).tolist()
 if pattern=='1':return np.rot90(G).tolist()
 if pattern=='2':return np.rot90(G,2).tolist()
 if pattern=='3':return np.rot90(G,3).tolist()
 if pattern=='t':return G.T.tolist()if G.shape[0]==G.shape[1]else grid
 if pattern=='c':return G[1:-1,1:-1].tolist()if G.shape[0]>2 and G.shape[1]>2 else grid
 if pattern=='d':return G[::2,::2].tolist()
 if pattern=='u':return np.repeat(np.repeat(G,2,0),2,1).tolist()
 if pattern=='H':return np.hstack([G,np.fliplr(G)]).tolist()
 if pattern=='V':return np.vstack([G,np.flipud(G)]).tolist()
 if pattern=='x':return np.tile(G,(2,2)).tolist()
 if pattern=='f':
  if G.size>0:
   counts=np.bincount(G.flatten())
   if len(counts)>0:
    return np.full(G.shape,np.argmax(counts)).tolist()
 if pattern=='F':
  c=params.get('c',1)if params else 1
  return[[c]*G.shape[1]for _ in range(G.shape[0])]
 if pattern=='M':
  m=params.get('m',{})if params else{}
  return[[m.get(int(c),int(c))for c in r]for r in G.tolist()]
 if pattern=='o':
  h,w=G.shape
  out=np.zeros_like(G)
  for r in range(h):
   for c in range(w):
    if G[r,c]!=0:
     if any(r+dr<0 or r+dr>=h or c+dc<0 or c+dc>=w or G[r+dr,c+dc]==0 for dr,dc in[(0,1),(0,-1),(1,0),(-1,0)]):
      out[r,c]=G[r,c]
  return out.tolist()
 if pattern=='s':
  h,w=G.shape
  out=G.copy()
  for r in range(h-1):
   for c in range(w-1):
    if G[r,c]!=G[r+1,c+1]and G[r+1,c+1]!=0:
     out[r,c]=0
  return out.tolist()
 
 return grid

def gen(pattern,params=None):
 """Generate code"""
 if pattern=='M'and params and'm'in params:
  m=params['m']
  items=list(m.items())
  if len(items)==1:
   o,n=items[0]
   return f'p=lambda g:[[{n}if x=={o}else x for x in r]for r in g]'
  if len(items)==2:
   (a,b),(c,d)=items
   return f'p=lambda g:[[{b}if x=={a}else{d}if x=={c}else x for x in r]for r in g]'
  d=','.join(f'{k}:{v}'for k,v in m.items())
  return f'p=lambda g:(lambda m={{{d}}}:[[m.get(x,x)for x in r]for r in g])()'
 
 if pattern=='F'and params:
  return f'p=lambda g:[[{params.get("c",1)}]*len(g[0])]*len(g)'
 
 return P.get(pattern,P['i'])

def solve(task):
 """Solve task"""
 examples=[(x['input'],x['output'])for x in task.get('train',[])]
 detector=FinalDetector()
 pattern,params,conf=detector.detect(examples)
 tests=task.get('test',[])
 outputs=[apply(pattern,params,t['input'])for t in tests]
 return outputs,pattern,params,conf

def main():
 """Main"""
 td=os.environ.get('TASKS_DIR','/kaggle/input/google-code-golf-2025')
 od=os.environ.get('OUTPUT_DIR','solutions')
 os.makedirs(od,exist_ok=True)
 
 ts,sm=0,[]
 
 for i in range(400):
  tid=f'task{i:03d}'
  tp=os.path.join(td,f'{tid}.json')
  
  if not os.path.exists(tp):continue
  
  try:
   with open(tp)as f:tk=json.load(f)
   
   outs,pt,pr,cf=solve(tk)
   cd=gen(pt,pr)
   sc=max(1,2500-len(cd))
   
   with open(os.path.join(od,f'{tid}.py'),'w')as f:
    f.write(f'# \n{cd}\n')
   
   with open(os.path.join(od,f'{tid}.out.json'),'w')as f:
    pl={'output':outs[0]}if len(outs)==1 else{'outputs':outs}
    json.dump(pl,f)
   
   ts+=sc
   sm.append({'id':tid,'p':pt,'c':cf,'s':sc})
   
   print(f"{tid}:{pt}(c={cf:.3f},s={sc})")
  
  except Exception as e:
   print(f"Error {tid}:{e}")
 
 with zipfile.ZipFile('submission.zip','w')as z:
  for r,_,fs in os.walk(od):
   for fn in fs:z.write(os.path.join(r,fn),os.path.join(os.path.basename(r),fn))
 
 # Summary
 detected=sum(1 for s in sm if s['p']!='i'or s['c']==1.0)
 print(f"\n{'='*50}")
 print(f"TOTAL SCORE: {ts}")
 print(f"PATTERNS DETECTED: {detected}/{len(sm)}")
 print(f"DETECTION RATE: {detected*100/len(sm):.1f}%")

if __name__=='__main__':main()

