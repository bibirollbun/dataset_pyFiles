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
from collections import deque
import numpy as np

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
'u':'p=lambda g:[x for r in g for x in([v for v in r for _ in(0,1)]) for _ in(0,1)]',
'x':'p=lambda g:[r+r for r in(g+g)]',
'H':'p=lambda g:[r+r[::-1]for r in g]',
'V':'p=lambda g:g+g[::-1]',
'f':'p=lambda g:[[max(set(sum(g,[])),key=lambda x:sum(r.count(x)for r in g))]*len(g[0])]*len(g)if g else g',
's':'p=lambda g:[[0 if(i<len(g)-1 and j<len(g[0])-1 and g[i][j]!=g[i+1][j+1]and g[i+1][j+1])else g[i][j]for j in range(len(g[0]))]for i in range(len(g))]',
'o':'p=lambda g:[[g[i][j]if g[i][j]and any(i+a<0 or i+a>=len(g)or j+b<0 or j+b>=len(g[0])or not g[i+a][j+b]for a,b in((0,1),(0,-1),(1,0),(-1,0)))else 0 for j in range(len(g[0]))]for i in range(len(g))]',
'F':'p=lambda g,c=1:[[c]*len(g[0])]*len(g)',
'M':'p=lambda g,m={}:[[m.get(x,x)for x in r]for r in g]'
}

class D:
    def d(self,ex):
        if not ex:return'i',{},0.0
        e=[(np.array(i),np.array(o))for i,o in ex];b=('i',{},0.0)
        def u(r):
            nonlocal b
            c=min(float(r[2]),1.0)
            if c>b[2]:b=(r[0],r[1],c);return c==1.0
            return 0
        for s in(self.g,self.sc,self.m,self.k,self.ob,self.p,self.sg,self.cp):
            if u(s(e)):return b
        return b

    def g(self,e):
        T=[('i',lambda i,o:np.array_equal(i,o)),
           ('h',lambda i,o:np.array_equal(np.fliplr(i),o)),
           ('v',lambda i,o:np.array_equal(np.flipud(i),o)),
           ('1',lambda i,o:np.array_equal(np.rot90(i),o)),
           ('2',lambda i,o:np.array_equal(np.rot90(i,2),o)),
           ('3',lambda i,o:np.array_equal(np.rot90(i,3),o)),
           ('t',lambda i,o:i.shape[0]==i.shape[1]and np.array_equal(i.T,o))]
        b=('i',{},0.0)
        for n,t in T:
            m=sum(t(i,o)for i,o in e);c=m/len(e)
            if c>b[2]:b=(n,{},c)
        return b

    def sc(self,e):
        b=('i',{},0.0)
        m=sum(o.shape==(i.shape[0]*2,i.shape[1]*2)and np.array_equal(np.repeat(np.repeat(i,2,0),2,1),o)for i,o in e);c=m/len(e)
        if c>b[2]:b=('u',{},c)
        m=sum(i.shape[0]%2==0 and i.shape[1]%2==0 and o.shape==(i.shape[0]//2,i.shape[1]//2)and np.array_equal(i[::2,::2],o)for i,o in e);c=m/len(e)
        if c>b[2]:b=('d',{},c)
        m=sum(i.shape[0]>2 and i.shape[1]>2 and o.shape==(i.shape[0]-2,i.shape[1]-2)and np.array_equal(i[1:-1,1:-1],o)for i,o in e);c=m/len(e)
        if c>b[2]:b=('c',{},c)
        return b

    def m(self,e):
        b=('i',{},0.0)
        m=sum(o.shape==(i.shape[0],i.shape[1]*2)and np.array_equal(np.hstack([i,np.fliplr(i)]),o)for i,o in e);c=m/len(e)
        if c>b[2]:b=('H',{},c)
        m=sum(o.shape==(i.shape[0]*2,i.shape[1])and np.array_equal(np.vstack([i,np.flipud(i)]),o)for i,o in e);c=m/len(e)
        if c>b[2]:b=('V',{},c)
        m=sum(o.shape==(i.shape[0]*2,i.shape[1]*2)and np.array_equal(np.tile(i,(2,2)),o)for i,o in e);c=m/len(e)
        if c>b[2]:b=('x',{},c)
        return b

    def k(self,e):
        gm=None;ok=1
        for I,O in e:
            if I.shape!=O.shape:ok=0;break
            uI=np.unique(I);uO=np.unique(O);lm={}
            for v in uI:
                msk=I==v;ov=O[msk]
                if ov.size and len(np.unique(ov))==1:lm[int(v)]=int(ov[0])
                else:ok=0;break
            if not ok:break
            gm=lm if gm is None else(gm if gm==lm else(0,0,ok:=0)[0])
            if not ok:break
        if ok and gm and not all(k==v for k,v in gm.items()):return('M',{'m':{int(k):int(v)for k,v in gm.items()}},1.0)
        m=0;fv=None
        for I,O in e:
            if I.shape==O.shape:
                u=np.unique(O)
                if len(u)==1:
                    v=int(u[0])
                    if fv is None:fv=v
                    if v==fv:m+=1
        if m==len(e)and fv is not None:return('F',{'c':fv},1.0)
        m=0
        for I,O in e:
            if I.shape==O.shape and I.size:
                fl=I.ravel()
                if fl.min()<0:continue
                bc=np.bincount(fl); 
                if bc.size and np.all(O==np.argmax(bc)):m+=1
        c=m/len(e)
        return('f',{},c)if c else('i',{},0.0)

    def ob(self,e):
        b=('i',{},0.0)
        for col in range(1,10):
            m=0;vals=set()
            for I,O in e:
                if I.shape==O.shape:
                    E=np.where(I==col,I,0)
                    if np.array_equal(E,O):m+=1
                vals|=set(np.unique(I).tolist())
            c=m/len(e)
            if c>b[2]and m:
                b=('M',{'m':{int(v):(int(v) if int(v)==col else 0)for v in vals}},c)
        m=0
        for I,O in e:
            objs=self._fo(I)
            if objs:
                L=max(objs,key=len)
                if self._bx(I,O,L):m+=1
        c=m/len(e)
        if c>b[2]:b=('i',{},c)
        return b

    def p(self,e):
        b=('i',{},0.0)
        m=0
        for I,O in e:
            if I.shape==O.shape:
                h,w=I.shape;E=np.zeros_like(I)
                for r in range(h):
                    for c in range(w):
                        if I[r,c]!=0 and(any(r+dr<0 or r+dr>=h or c+dc<0 or c+dc>=w or I[r+dr,c+dc]==0 for dr,dc in((0,1),(0,-1),(1,0),(-1,0)))):
                            E[r,c]=I[r,c]
                if np.array_equal(E,O):m+=1
        c=m/len(e)
        if c>b[2]:b=('o',{},c)
        m=0
        for I,O in e:
            if I.shape==O.shape:
                h,w=I.shape;E=I.copy()
                for r in range(h-1):
                    for c in range(w-1):
                        if I[r,c]!=I[r+1,c+1]and I[r+1,c+1]!=0:E[r,c]=0
                if np.array_equal(E,O):m+=1
        c=m/len(e)
        if c>b[2]:b=('s',{},c)
        return b

    def sg(self,e):
        b=('i',{},0.0)
        m=sum(self._ct(I,O)for I,O in e);c=m/len(e)
        if c>b[2]:b=('i',{},c)
        m=sum(self._ct(O,I)for I,O in e);c=m/len(e)
        if c>b[2]:b=('i',{},c)
        return b

    def cp(self,e):
        b=('i',{},0.0);N=['i','1','2','3']
        for r in range(4):
            m=sum(np.array_equal(np.rot90(I,r),O)for I,O in e);c=m/len(e)
            if c>b[2]:b=(N[r],{},c)
        for fl in('h','v'):
            for r in range(4):
                m=0
                for I,O in e:
                    F=np.fliplr(I)if fl=='h'else np.flipud(I)
                    R=np.rot90(F,r)
                    if np.array_equal(R,O):m+=1
                c=m/len(e)
                if c>b[2]:b=((N[r]if r else fl),{},c if c>=0.8 else c)
        return b

    def _fo(self,G):
        v=np.zeros_like(G,bool);o=[];h,w=G.shape
        for r in range(h):
            for c in range(w):
                if G[r,c]!=0 and not v[r,c]:
                    col=G[r,c];q=deque([(r,c)]);obj=[]
                    while q:
                        y,x=q.popleft()
                        if 0<=y<h and 0<=x<w and not v[y,x] and G[y,x]==col:
                            v[y,x]=1;obj.append((y,x))
                            for dy,dx in((0,1),(0,-1),(1,0),(-1,0)):q.append((y+dy,x+dx))
                    if obj:o.append(obj)
        return o

    def _bx(self,I,O,obj):
        if not obj:return 0
        rs,cs=zip(*obj);a,b,c,d=min(rs),max(rs),min(cs),max(cs)
        sh=(b-a+1,d-c+1)
        if O.shape!=sh:return 0
        for r,x in obj:
            if O[r-a,x-c]!=I[r,x]:return 0
        for r in range(sh[0]):
            for x in range(sh[1]):
                if (r+a,x+c)not in obj and O[r,x]!=0:return 0
        return 1

    def _ct(self,L,S):
        if S.shape[0]>L.shape[0]or S.shape[1]>L.shape[1]:return 0
        H,W=L.shape;h,w=S.shape
        for r in range(H-h+1):
            for c in range(W-w+1):
                if np.array_equal(L[r:r+h,c:c+w],S):return 1
        return 0

def A(p,pr,g):
    G=np.array(g)
    if p=='i':return g
    if p=='h':return np.fliplr(G).tolist()
    if p=='v':return np.flipud(G).tolist()
    if p=='1':return np.rot90(G).tolist()
    if p=='2':return np.rot90(G,2).tolist()
    if p=='3':return np.rot90(G,3).tolist()
    if p=='t':return G.T.tolist()if G.size else g
    if p=='c':return G[1:-1,1:-1].tolist()if G.shape[0]>2 and G.shape[1]>2 else g
    if p=='d':return G[::2,::2].tolist()
    if p=='u':return np.repeat(np.repeat(G,2,0),2,1).tolist()
    if p=='H':return np.hstack([G,np.fliplr(G)]).tolist()
    if p=='V':return np.vstack([G,np.flipud(G)]).tolist()
    if p=='x':return np.tile(G,(2,2)).tolist()
    if p=='f':
        if G.size:
            fl=G.ravel()
            if fl.min()<0:return g
            bc=np.bincount(fl)
            if bc.size:return np.full(G.shape,int(np.argmax(bc))).tolist()
    if p=='F':
        c=pr.get('c',1)if pr else 1
        return [[c]*G.shape[1]for _ in range(G.shape[0])]
    if p=='M':
        m=pr.get('m',{})if pr else{}
        return [[int(m.get(int(x),int(x)))for x in r]for r in G.tolist()]
    if p=='o':
        h,w=G.shape;O=np.zeros_like(G)
        for r in range(h):
            for c in range(w):
                if G[r,c]!=0 and any(r+dr<0 or r+dr>=h or c+dc<0 or c+dc>=w or G[r+dr,c+dc]==0 for dr,dc in((0,1),(0,-1),(1,0),(-1,0))):O[r,c]=G[r,c]
        return O.tolist()
    if p=='s':
        h,w=G.shape;O=G.copy()
        for r in range(max(0,h-1)):
            for c in range(max(0,w-1)):
                if G[r,c]!=G[r+1,c+1]and G[r+1,c+1]!=0:O[r,c]=0
        return O.tolist()
    return g

def Gc(p,pr=None):
    if p=='M'and pr and'm'in pr:
        m=pr['m'];it=list(m.items())
        if len(it)==1:
            o,n=it[0];return f'p=lambda g:[[{n} if x=={o} else x for x in r] for r in g]'
        if len(it)==2:
            (a,b),(c,d)=it;return f'p=lambda g:[[{b} if x=={a} else {d} if x=={c} else x for x in r] for r in g]'
        d=','.join(f'{k}:{v}'for k,v in m.items());return f'p=lambda g:(lambda m={{{d}}}:[[m.get(x,x) for x in r] for r in g])()'
    if p=='F'and pr:return f'p=lambda g:[[{pr.get("c",1)}]*len(g[0])]*len(g)'
    return P.get(p,P['i'])

def S(t):
    ex=[(x['input'],x['output'])for x in t.get('train',[])]
    dt=D();p,pr,c=dt.d(ex);ts=t.get('test',[]);outs=[A(p,pr,z['input'])for z in ts]
    return outs,p,pr,c

def main():
    td=os.environ.get('TASKS_DIR','/kaggle/input/google-code-golf-2025');od=os.environ.get('OUTPUT_DIR','solutions')
    os.makedirs(od,exist_ok=True)
    T=0;sm=[]
    for i in range(400):
        tid=f'task{i:03d}';tp=os.path.join(td,f'{tid}.json')
        if not os.path.exists(tp):continue
        try:
            with open(tp)as f:t=json.load(f)
            outs,p,pr,c=S(t);cd=Gc(p,pr);sc=max(1,2500-len(cd))
            with open(os.path.join(od,f'{tid}.py'),'w')as f:f.write('#\n'+cd+'\n')
            with open(os.path.join(od,f'{tid}.out.json'),'w')as f:json.dump({'output':outs[0]}if len(outs)==1 else{'outputs':outs},f)
            T+=sc;sm.append({'id':tid,'p':p,'c':c,'s':sc})
            print(f"{tid}:{p}(c={c:.3f},s={sc})")
        except Exception as e:print(f"Error {tid}:{e}")
    with zipfile.ZipFile('submission.zip','w')as z:
        for r,_,fs in os.walk(od):
            for fn in fs:z.write(os.path.join(r,fn),os.path.join(os.path.basename(r),fn))
    det=sum(1 for s in sm if s['p']!='i' or s['c']==1.0);tot=len(sm)if sm else 0
    print('='*50);print(f"TOTAL SCORE: {T}");print(f"PATTERNS DETECTED: {det}/{tot}");print(f"DETECTION RATE: {(det*100/tot)if tot else 0.0:.1f}%")

if __name__=='__main__':main()


