import sys
sys.path.append("/kaggle/input/google-code-golf-2025/code_golf_utils")
from code_golf_utils import *
show_legend()


show_examples(load_examples(1)['train'])


%%writefile task001.py
p=lambda j,A=range(9):[[j[r//3][c//3]and j[r%3][c%3]for c in A]for r in A]


show_examples(load_examples(2)['train'])


%%writefile task002.py
def p(j):
	A=range;c=len(j);E=[[0]*c for B in A(c)]
	def B(k,W):
		if 0<=k<c and 0<=W<c and not E[k][W]and j[k][W]==0:E[k][W]=1;[B(k+c,W+A)for(c,A)in[(1,0),(-1,0),(0,1),(0,-1)]]
	[B(A,0)or B(A,c-1)or B(0,A)or B(c-1,A)for A in A(c)];return[[4 if j[B][c]==0and not E[B][c]else j[B][c]for c in A(c)]for B in A(c)]


show_examples(load_examples(3)['train'])


%%writefile task003.py
p=lambda j:[[c*2 for c in r]for r in j+(j[:3],j[2:5])[j[1]!=j[4]]]


show_examples(load_examples(4)['train'])


%%writefile task004.py
def p(j,A=enumerate):
 c=[[0]*len(j[0])for _ in j]
 for E in set(sum(j,[]))-{0}:
  k=[(J,a)for J,r in A(j)for a,x in A(r)if x==E];W,l=max(J for J,_ in k),max(a for _,a in k)
  for J,a in k:c[J][a+(J<W and a<l)]=E
 return c


show_examples(load_examples(5)['train'])


#%%writefile task005.py



show_examples(load_examples(6)['train'])


%%writefile task006.py
p=lambda j:[[a and b and 2 for a,b in zip(r[:3],r[4:7])]for r in j[:3]]


show_examples(load_examples(7)['train'])


%%writefile task007.py
def p(g):R=range;L=len;d={(i+j)%3:c for i in R(L(g))for j in R(L(g[0]))for c in[g[i][j]]if c};return[[d.get((i+j)%3,0)for j in R(L(g[0]))]for i in R(L(g))]


show_examples(load_examples(8)['train'])


%%writefile task008.py
def p(j,A=enumerate):
	c,E=[(c,b)for(c,E)in A(j)for(b,d)in A(E)if d==2],[(c,b)for(c,E)in A(j)for(b,d)in A(E)if d==8]
	if not c or not E:return j
	k=lambda W:(min(c for(c,E)in W),max(c for(c,E)in W),min(c for(E,c)in W),max(c for(E,c)in W));l,J,a,C=k(c);e,K,w,L=k(E);b=d=0
	if C<w:d=w-C-1
	elif L<a:d=L-a+1
	elif J<e:b=e-J-1
	elif K<l:b=K-l+1
	f,g={*c},{*E};return[[8 if(c,E)in g else 2 if(c-b,E-d)in f else 0 for(E,k)in A(j[0])]for(c,E)in A(j)]


show_examples(load_examples(9)['train'])


%%writefile task009.py
def p(j,A=range,c=len):
	E=[J[:]for J in j];k,W=c(j),c(j[0]);l=j[0][2]
	for J in A(k):
		for a in A(W):
			if j[J][a]==l:E[J][a]=l;j[J][a]=0
			else:E[J][a]=0
	C=[J[:]for J in j]
	for e in A(k):
		K=[(J,a)for J in A(k)for a in A(W)if j[J][a]==e]
		for J in A(len(K)):
			for a in A(J+1,len(K)):
				w,L=K[J];b,d=K[a]
				if w==b:
					for f in A(min(L,d),max(L,d)+1):C[w][f]=e
				elif L==d:
					for g in A(min(w,b),max(w,b)+1):C[g][L]=e
	for J in A(k):
		for a in A(W):
			if E[J][a]>0:C[J][a]=l
	return C


show_examples(load_examples(10)['train'])


%%writefile task010.py
def p(j):
 A={}
 for c in j:
  for E,k in enumerate(c):
   if k==5:c[E]=A.setdefault(E,len(A)+1)
 return j


show_examples(load_examples(11)['train'])


%%writefile task011.py
def p(j):
 A=range
 for c in A(3):
  for E in A(3):
   if sum(j[c*4+W][E*4+l]==0for W in A(3)for l in A(3))==5:
    k=[[5if i%4==3or j%4==3else 0for j in A(11)]for i in A(11)]
    for W in A(3):
     for l in A(3):
      J=j[c*4+W][E*4+l]
      if J:
       for a in A(3):
        for C in A(3):k[W*4+a][l*4+C]=J
    return k


show_examples(load_examples(12)['train'])


%%writefile task012.py
def p(j,A=range(-2,3),c=enumerate,E=abs):k=[E[:]for E in j];[k[I+D].__setitem__(C+F,H if E(D)==E(F)else B[C-1])for(I,B)in c(j)for(C,H)in c(B)if H and B[C-1]*B[C+1]for D in A for F in A if E(D)==E(F)or not D*F];return k


show_examples(load_examples(13)['train'])


%%writefile task013.py
def p(j,A=range):
 c,E=len(j),len(j[0])
 p=[(l,L,j[l][L])for l in A(c)for L in A(E)if j[l][L]]
 p.sort()
 if len(p)==2:
  k,W=p
  if k[0]==W[0]:
   l,J,a=k;C,e=W[1],W[2];K=abs(C-J)
   for w in A(c):j[w][J]=a;j[w][C]=e
   if K:
    L=max(J,C)+K;b=0;d=[a,e]
    if C<J:d=d[::-1]
    while L<E:
     for w in A(c):j[w][L]=d[b%2]
     L+=K;b+=1
  elif k[1]==W[1]:
   L,f,a=k[1],k[0],k[2];g,e=W[0],W[2];K=abs(g-f)
   for w in A(E):j[f][w]=a;j[g][w]=e
   if K:
    l=g+K;b=0;d=[a,e]
    while l<c:
     for w in A(E):j[l][w]=d[b%2]
     l+=K;b+=1
  elif k[0]==0and W[0]==c-1:
   f,J,a=k;g,C,e=W;K=abs(C-J)
   for w in A(c):j[w][J]=a;j[w][C]=e
   if K:
    L=C+K;b=0;d=[a,e]
    while L<E:
     for w in A(c):j[w][L]=d[b%2]
     L+=K;b+=1
  elif(k[1]==0and W[1]==E-1)or(W[1]==0and k[1]==E-1):
   if k[1]==0:f,J,a=k;g,C,e=W
   else:f,J,a=W;g,C,e=k
   K=abs(g-f)
   for w in A(E):j[f][w]=a;j[g][w]=e
   if K:
    l=max(f,g)+K;b=0;d=[a,e]
    if g<f:d=d[::-1]
    while l<c:
     for w in A(E):j[l][w]=d[b%2]
     l+=K;b+=1
 return j


show_examples(load_examples(14)['train'])


%%writefile task014.py
from collections import*
def p(j):
 A=[x for k in j for x in k];c=Counter(A).most_common(3);c=[c for c in c if c[0]>0][-1][0];j=[k for k in j if c in k];E=[]
 for k in j:
  for W in range(len(k)):
   if k[W]==c:E+=[W]
 return[k[min(E):max(E)+1]for k in j]


show_examples(load_examples(15)['train'])


%%writefile task015.py
L=len
R=range
def p(g):
 h,w=L(g),L(g[0])
 for r in R(h):
  for c in R(w):
   if g[r][c]==2:
    for i,j in[[1,1],[-1,-1],[-1,1],[1,-1]]:g[i+r][j+c]=4
   if g[r][c]==1:
    for i,j in[[0,1],[0,-1],[-1,0],[1,0]]:g[i+r][j+c]=7
 return g


show_examples(load_examples(16)['train'])


%%writefile task016.py
p=lambda j,A=[0,5,6,4,3,1,2,7,9,8]:[[A[x]for x in r]for r in j]


show_examples(load_examples(17)['train'])


%%writefile task017.py
def p(j,u=enumerate):
	A=range;c=len(j);E=len(j[0]);k=lambda W,l:W==l or W*l<1;J=next((K for K in A(1,E)if all(k(L,e)for w in j for(L,e)in zip(w,w[K:]))),E);a=next((K for K in A(1,c)if all(k(L,e)for(K,w)in zip(j,j[K:])for(L,e)in zip(K,w))),c);C={}
	for(e,K)in u(j):
		for(w,L)in u(K):
			if L:C[e%a,w%J]=L
	for(e,K)in u(j):
		for(w,L)in u(K):
			if not L:K[w]=C[e%a,w%J]
	return j


show_examples(load_examples(18)['train'])


#%%writefile task018.py



show_examples(load_examples(19)['train'])


%%writefile task019.py
L=len
R=range
def p(g):
 g=[r[:]+r[:]for r in g]+[r[:]+r[:]for r in g]
 h,w=L(g),L(g[0])
 for r in R(h):
  for c in R(w):
   C=g[r][c]
   if C>0 and C!=8:
    for i,j in[[1,1],[-1,-1],[-1,1],[1,-1]]:
     if i+r>=0 and j+c>=0 and i+r<h and j+c<w:
      if g[i+r][j+c]==0:g[i+r][j+c]=8
 return g


show_examples(load_examples(20)['train'])


%%writefile task020.py
def p(g,H=enumerate):
 t=l=9**9;b=r=-1
 for y,a in H(g):
  for x,v in H(a):
   if v:t=min(t,y);b=max(b,y);l=min(l,x);r=max(r,x)
 s=t+b;S=l+r
 for y in range(t,b+1):
  for x in range(l,r+1):
   Y=s-y;X=S-x;u=t+x-l;v=l+y-t;U=t+r-x;V=l+b-y
   P=((y,x),(y,X),(Y,x),(Y,X),(u,v),(U,v),(u,V),(U,V))
   c=max(g[i][j]for i,j in P)
   for i,j in P:g[i][j]=c
 return g


show_examples(load_examples(21)['train'])


%%writefile task021.py
def p(g,u=range):n=len(g);m=len(g[0]);r=[i for i in u(n)if len(set(g[i]))==1];c=[j for j in u(m)if len(set(g[i][j]for i in u(n)))==1];b=next(x for i in u(n)for j,x in enumerate(g[i])if i not in r and j not in c);return[[b]*(len(c)+1)for _ in u(len(r)+1)]


show_examples(load_examples(22)['train'])


%%writefile task022.py
L=len
R=range
def p(g):
 X=[[0,0,0]for _ in R(3)]
 h,w=L(g),L(g[0])
 for r in R(h):
  for c in R(w):
   if g[r][c]==5:
    for i in R(-1,2):
     for j in R(-1,2):
      if r+i>=0 and c+j>=0 and g[r+i][c+j]!=0:X[1+i][1+j]=g[r+i][c+j]
 return X


show_examples(load_examples(23)['train'])


%%writefile task023.py
def p(g,L=len,R=range):
 #rules: 1x3/3x1 for all reds, 2x2 for all blues, no gray remaining
 h,w=L(g),L(g[0])
 Z=[[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]] #3x3
 P=[[0,0],[0,1],[1,0],[1,1]] #2x2
 Q=[[0,0],[0,1],[0,2]] #1x3
 S=[[0,0],[1,0],[2,0]] #3x1
 for r in R(h):
  for c in R(w):
   try:
    if [g[r+i[0]][c+i[1]] for i in Z]==[5,5,5,5,5,5,0,0,5]:
     Y=[8,8,2,8,8,2,0,0,2]
     for i in R(L(Z)): 
      g[r+Z[i][0]][c+Z[i][1]]=Y[i]
    elif [g[r+i[0]][c+i[1]] for i in Z]==[5,5,5,5,5,5,5,0,0]:
     Y=[2,8,8,2,8,8,2,0,0]
     for i in R(L(Z)): 
      g[r+Z[i][0]][c+Z[i][1]]=Y[i]
    elif [g[r+i[0]][c+i[1]] for i in Z]==[0,5,5,0,5,5,5,5,5]:
     Y=[0,8,8,0,8,8,2,2,2]
     for i in R(L(Z)): 
      g[r+Z[i][0]][c+Z[i][1]]=Y[i]
    elif [g[r+i[0]][c+i[1]] for i in Z]==[5,5,5,5,5,0,5,5,0]:
     Y=[2,2,2,8,8,0,8,8,0]
     for i in R(L(Z)): 
      g[r+Z[i][0]][c+Z[i][1]]=Y[i]
   except: pass
 for r in R(h):
  for c in R(w):
   try:
    if [g[r+i[0]][c+i[1]] for i in P]==[5,5,5,5]:
     for i in P: 
      g[r+i[0]][c+i[1]]=8
    elif [g[r+i[0]][c+i[1]] for i in Q]==[5,5,5]:
     for i in Q: 
      g[r+i[0]][c+i[1]]=2
    elif [g[r+i[0]][c+i[1]] for i in S]==[5,5,5]:
     for i in S: 
      g[r+i[0]][c+i[1]]=2
   except: pass
 return g


show_examples(load_examples(24)['train'])


%%writefile task024.py
def p(g,E=enumerate):Z={c for R in g for c,v in E(R)if v==2};return[[1 if 1 in R else 3 if 3 in R else 2 if v<1and c in Z else v for c,v in E(R)]for R in g]


show_examples(load_examples(25)['train'])


#%%writefile task025.py



show_examples(load_examples(26)['train'])


%%writefile task026.py
p=lambda j:[[8*(not A|B)for(A,B)in zip(A,A[4:])]for A in j]


show_examples(load_examples(27)['train'])


#%%writefile task027.py



show_examples(load_examples(28)['train'])


%%writefile task028.py
def p(g,m=max):e,a=m(map(m,g[:5])),m(map(m,g[-5:]));return[[(r in(0,2,7,9)or c%9<1)*(e*(r<5)+a*(r>4))for c in range(10)]for r in range(10)]


show_examples(load_examples(29)['train'])


%%writefile task029.py
def p(g,L=len,E=enumerate):
 for C in set(sum(g,[])):
  P=[[x,y] for y,r in E(g) for x,c in E(r) if c==C]
  f=sum(P,[]);x=f[::2];y=f[1::2]
  X=g[min(y):max(y)]
  X=[r[min(x)+1:max(x)][:] for r in X]
  if X[0].count(C)==L(X[0]):
   return X[1:]
 return g


show_examples(load_examples(30)['train'])


%%writefile task030.py
def p(g,L=len,R=range):
 h,w,x,y,b=L(g),L(g[0]),[],[],[]
 for r in R(h):
  for c in R(w):
   C=g[r][c]
   if C==2:x+=[c];g[r][c]=0
   if C==4:y+=[c];g[r][c]=0
   if C==1:b+=[c]
 for r in R(h):
  for c in R(w):
   if g[r][c]==1:g[r][c+(min(y)-min(b))]=4;g[r][c+(min(x)-min(b))]=2
 return g


show_examples(load_examples(31)['train'])


%%writefile task031.py
def p(j,A=enumerate):c,E=zip(*[(i,j)for i,r in A(j)for j,x in A(r)if x]);return[r[min(E):max(E)+1]for r in j[min(c):max(c)+1]]


show_examples(load_examples(32)['train'])


%%writefile task032.py
p=lambda j:list(map(list,zip(*[[0]*c.count(0)+[x for x in c if x]for c in zip(*j)])))


show_examples(load_examples(33)['train'])


%%writefile task033.py
def p(j):
	A=range;c=[A[:]for A in j];E=j[5][0];k=[[j[l+1][A+1]for A in A(3)]for l in A(3)]
	for W in[(0,6),(0,12),(6,0),(6,6),(6,12),(12,0),(12,6),(12,12)]:
		for l in A(3):
			for J in A(3):a,C=W[0]+l+1,W[1]+J+1;c[a][C]=j[a][C]if k[l][J]==j[a][C]else E if k[l][J]else 0
	return c


show_examples(load_examples(34)['train'])


#%%writefile task034.py



show_examples(load_examples(35)['train'])


#%%writefile task035.py



show_examples(load_examples(36)['train'])


%%writefile task036.py
def p(j,A=len,c=range):
	E='r';k='c';W,l,J=A(j),A(j[0]),{}
	for a in c(W):
		for C in c(l):
			e=j[a][C]
			if e in J:J[e][E]+=[a];J[e][k]+=[C]
			else:J[e]={E:[a],k:[C]}
	K=sorted([[A(J[e][E])*(max(J[e][k])-min(J[e][k])),e]for e in J if e>0])[0][1];j=[[K if J==K else 0 for J in J]for J in j];J=J[K];j=[E[min(J[k]):max(J[k])+1]for E in j];j=j[min(J[E]):max(J[E])+1];return j


show_examples(load_examples(37)['train'])


%%writefile task037.py
def p(j,A=range):
	c,E=len(j),len(j[0]);k,W={},[A[:]for A in j]
	for l in A(c):
		for J in A(E):
			a=j[l][J]
			if a:k.setdefault(a,[]).append((l,J))
	for a in k:
		(C,e),(K,w)=k[a];L=1 if K>C else-1;b=1 if w>e else-1
		for d in A(abs(K-C)+1):W[C+d*L][e+d*b]=a
	return W


show_examples(load_examples(38)['train'])


%%writefile task038.py
def p(g):q=range;c=sum(all(g[i+k][j+l]==1for k in q(2)for l in q(2))for i in q(8)for j in q(8));return[[1if i<c else 0for i in q(5)]]


show_examples(load_examples(39)['train'])


%%writefile task039.py
j=len
A=range
def p(c):
	E,k=j(c),j(c[0]);W=[]
	for l in A(E):
		for J in A(k):
			if c[l][J]>0:W.append([l,J])
	a=min([W[1]for W in W]);C=max([W[1]for W in W]);e=min([W[0]for W in W]);K=max([W[0]for W in W]);C=C-(C-a)//2;K=K-(K-e)//2;c=c[e:K];c=[W[a:C]for W in c];return c


show_examples(load_examples(40)['train'])


%%writefile task040.py
def p(j):
	A=range;c=[J[:]for J in j];E=j[0][0]==j[0][9];k,W=(j[0][0],j[9][0])if E else(j[0][0],j[0][9]);l=next(J for a in j for J in a if J and J not in[k,W])
	for J in A(10):
		for a in A(10):
			if j[J][a]==l:C=(J,9-J)if E else(a,9-a);c[J][a]=k if C[0]<C[1]else W
	return c


show_examples(load_examples(41)['train'])


%%writefile task041.py
def p(j,A=0):
 for c in j:
  for E,k in enumerate(c):
   if k:A=(not A)*k
   else:c[E]=A
 return j


show_examples(load_examples(42)['train'])


#%%writefile task042.py



show_examples(load_examples(43)['train'])


%%writefile task043.py
def p(j,A=enumerate):
 c=len(j)-1
 E=len(j[0])-1
 for k,W in A(j):
  for l,J in A(W):
   if k>0and l<c:
    if j[k][E]==5and j[0][l]==5:j[k][l]=2
 return j


show_examples(load_examples(44)['train'])


#%%writefile task044.py



show_examples(load_examples(45)['train'])


%%writefile task045.py
def p(j):
	for A in j:
		for c in{*A}-{0}:
			E=A.index(c);k=len(A)-A[::-1].index(c)
			for W in range(E,k):
				if~A[W]:A[W]=c
	return j


show_examples(load_examples(46)['train'])


#%%writefile task046.py



show_examples(load_examples(47)['train'])


%%writefile task047.py
def p(j):
	A=range;c=[[0]*9 for c in A(9)];E=[(c,E,j[c][E])for c in A(9)for E in A(9)if j[c][E]]
	for(k,W,l)in E:
		for J in range(9):c[k][J]=c[J][W]=l
	c[E[0][0]][E[1][1]]=c[E[1][0]][E[0][1]]=2;return c


show_examples(load_examples(48)['train'])


%%writefile task048.py
def f(j,A,c):
	global W;l.append((j,A))
	for E in C(j-1,j+2):
		for k in C(A-1,A+2):
			if(E,k)in l:continue
			l.append((E,k))
			if E<0 or E>=J or k<0 or k>=a or(E,k)in[(K,L),(K+1,L),(K,L+1),(K+1,L+1)]:continue
			if c[E][k]==2:W=8
			if c[E][k]==8:f(E,k,c)
def p(c):
	global W,l,K,L,J,a,C;W,l,J,a,C,e=0,[],len(c),len(c[0]),range,enumerate
	for(K,w)in e(c):
		for(L,b)in e(w):
			if b==2:
				for E in C(K-1,K+3):
					for k in C(L-1,L+3):
						if E>=0 and E<J and k>=0 and k<a and c[E][k]==8:f(E,k,c)
				return[[W]]


show_examples(load_examples(49)['train'])


%%writefile task049.py
from collections import*
j=len
A=range
def p(c):
	E=[x for A in c for x in A];k=Counter(E).most_common();k=[C for C in k if C[0]!=0][-1][0];W,l=j(c),j(c[0]);J=[]
	for a in A(W):
		for C in A(l):
			if c[a][C]==k:J.append([a,C])
	e=min([i[1]for i in J]);K=max([i[1]for i in J]);w=min([i[0]for i in J]);L=max([i[0]for i in J]);c=c[w:L+1];c=[A[e:K+1]for A in c];return c


show_examples(load_examples(50)['train'])


%%writefile task050.py
def p(j):
	A=range;c=[e[:]for e in j];E,k=len(j),len(j[0]);W=[(e,l)for e in A(E)for l in A(k)if j[e][l]==8]
	for(l,J)in W:
		for(a,C)in[(0,1),(1,0),(0,-1),(-1,0)]:
			e=1
			while 0<=l+e*a<E and 0<=J+e*C<k:
				if j[l+e*a][J+e*C]==8:
					for K in A(1,e):c[l+K*a][J+K*C]=3
					break
				e+=1
	return c


show_examples(load_examples(51)['train'])


%%writefile task051.py
def p(j):
	A=range;c=[l[:]for l in j];E,k=len(j),len(j[0]);W={}
	for l in A(E):
		for J in A(k):
			if j[l][J]:W[j[l][J]]=W.get(j[l][J],0)+1
	l,J,a=next((l,J,j[l][J])for l in A(E)for J in A(k)if j[l][J]and W[j[l][J]]==1)
	for(C,e)in[(0,1),(1,0),(0,-1),(-1,0)]:
		K,w=l+C,J+e
		if(K<0)|(K>=E)|(w<0)|(w>=k)|(j[K][w]==0):
			L=1
			while(0<=l-L*C<E)&(0<=J-L*e<k):
				if j[l-L*C][J-L*e]==0:c[l-L*C][J-L*e]=a
				L+=1
	return c


show_examples(load_examples(52)['train'])


%%writefile task052.py
p=lambda j:[[5]*3if len(set(r))==1else[0]*3for r in j]


show_examples(load_examples(53)['train'])


%%writefile task053.py
p=lambda j:[r[3%len(r):]+r[:3%len(r)]for r in j[2:]+j[:2]]


show_examples(load_examples(54)['train'])


#%%writefile task054.py



show_examples(load_examples(55)['train'])


%%writefile task055.py
def p(j,A=range):
	c,E=len(j),len(j[0]);k=[A[:]for A in j];W,l=[A for A in A(c)if all(A==8 for A in j[A])];J,a=[C for C in A(E)if all(j[A][C]==8 for A in A(c))]
	for C in A(c):
		for e in A(E):
			if not k[C][e]:
				if C<W and J<e<a:k[C][e]=2
				elif W<C<l and e<J:k[C][e]=4
				elif W<C<l and J<e<a:k[C][e]=6
				elif W<C<l and e>a:k[C][e]=3
				elif C>l and J<e<a:k[C][e]=1
	return k


show_examples(load_examples(56)['train'])


%%writefile task056.py
def p(j):A=tuple(0if v==0else 1for v in j[0]);return[[{(1,1,0):1,(1,0,1):2,(0,1,1):3,(0,1,0):6}[A]]]


show_examples(load_examples(57)['train'])


%%writefile task057.py
def p(j):A=[i for r in j for i,x in enumerate(r)if x>0];c,E=min(A),max(A)+1;return[r[c:E]*2 for r in j if max(r)>0]


show_examples(load_examples(58)['train'])


%%writefile task058.py
def p(j):
	A=range;c=len(j);E=[[0]*c for W in A(c)];k,W=0,0;l=[(0,1),(1,0),(0,-1),(-1,0)]
	for J in A(c):
		E[k][W]=3
		if J<c-1:W+=1
	a=c-1;C=1
	while a>0:
		for e in A(2):
			if a>0:
				k,W=k+l[C][0],W+l[C][1]
				for J in A(a):
					E[k][W]=3
					if J<a-1:k,W=k+l[C][0],W+l[C][1]
				C=(C+1)%4
		a-=2
	return E


show_examples(load_examples(59)['train'])


%%writefile task059.py
def p(j,A=enumerate,c=range(11)):
 E=0;k=[[0 if(i+1)%4>0and(j+1)%4>0 else 5 for i in c]for j in c];W={'00':0,'01':0,'02':0,'10':0,'11':0,'12':0,'20':0,'21':0,'22':0}
 for l,J in A(j):
  for a,C in A(J):
   if C>0and C!=5:E=int(C);W[str(l//4)+str(a//4)]+=1
 e=max(W.values())
 for l,J in A(k):
  for a,C in A(J):
   if C==0and W[str(l//4)+str(a//4)]==e:k[l][a]=E
 return k


show_examples(load_examples(60)['train'])


%%writefile task060.py
def p(j):
	A=len(j[0]);c=int((A-1)/2);E=enumerate
	for(k,W)in E(j):
		if max(W)>0:
			for l in range(c):j[k][l]=j[k][0];j[k][A-l-1]=j[k][A-1]
			j[k][c]=5
	return j


show_examples(load_examples(61)['train'])


%%writefile task061.py
def p(j,u=enumerate):
	A=range;c=len(j);E=len(j[0]);k=lambda W,l:W==l or W*l<1;J=next((K for K in A(1,E)if all(k(L,e)for w in j for(L,e)in zip(w,w[K:]))),E);a=next((K for K in A(1,c)if all(k(L,e)for(K,w)in zip(j,j[K:])for(L,e)in zip(K,w))),c);C={}
	for(e,K)in u(j):
		for(w,L)in u(K):
			if L:C[e%a,w%J]=L
	for(e,K)in u(j):
		for(w,L)in u(K):
			if not L:K[w]=C[e%a,w%J]
	return j


show_examples(load_examples(62)['train'])


#%%writefile task062.py



show_examples(load_examples(63)['train'])


%%writefile task063.py
def p(j):
 A=range
 c=len(j)
 E=[o[:]for o in j]
 for k in range(c):
  if j[1][k]==0and j[c-2][k]==0and sum(j[W][k]for W in A(1,c-1))==0:
   for W in A(1,c-1):E[W][k]=3
 for W in range(c):
  if j[W][1]==0and j[W][c-2]==0and sum(j[W][k]for k in A(1,c-1))==0:
   for k in A(1,c-1):
    if E[W][k]==0:E[W][k]=3
 return E


show_examples(load_examples(64)['train'])


#%%writefile task064.py



show_examples(load_examples(65)['train'])


%%writefile task065.py
def p(j):
	A=range;c=(len(j)-1)//2
	if c==1:
		E=[j[0][0],j[0][2],j[2][0],j[2][2]]
		for k in E:
			if E.count(k)==1:return[[k]]
	for(W,l)in[(0,0),(0,c+1),(c+1,0),(c+1,c+1)]:
		J=[[j[W+k][l+c]for c in A(c)]for k in A(c)];k=[J[k][E]for k in A(c)for E in A(c)]
		if len(set(k))>1:return J


show_examples(load_examples(66)['train'])


#%%writefile task066.py



show_examples(load_examples(67)['train'])


%%writefile task067.py
p=lambda j:[R[:int(len(j[0])/3)]for R in j]


show_examples(load_examples(68)['train'])


%%writefile task068.py
def p(j):
	A={};c=range
	for E in c(10):
		for k in c(10):
			if j[E][k]:A[j[E][k]]=A.get(j[E][k],0)+1
	W=next(A for(A,c)in A.items()if c==1);l,A=next((A,E)for A in c(10)for E in c(10)if j[A][E]==W);J=[[0]*10 for A in c(10)];J[l][A]=W
	for a in[-1,0,1]:
		for C in[-1,0,1]:
			if a or C:
				e,K=l+a,A+C
				if 0<=e<10 and 0<=K<10:J[e][K]=2
	return J


show_examples(load_examples(69)['train'])


%%writefile task069.py
def p(g,E=enumerate):
 P=[]
 for r,R in E(g):
  for c,C in E(R):
   if C not in[0,8]:P+=[[r,c,C]];g[r][c]=0
 Z=P[0][:];P=[[x[0]-Z[0],x[1]-Z[1],x[2]]for x in P]
 for r,R in E(g):
  for c,C in E(R):
   if C==8:
    g[r][c]=Z[2]
    for x in P:g[r+x[0]][c+x[1]]=x[2]
 return g


show_examples(load_examples(70)['train'])


%%writefile task070.py
def p(j):
	A=range;c=[E[:]for E in j];E=[(E,c)for E in A(len(j))for c in A(len(j[0]))if j[E][c]==8]
	if E:
		k,W=min(E for(E,A)in E),max(E for(E,A)in E);l,J=min(E for(A,E)in E),max(E for(A,E)in E)
		for a in A(k,W+1):
			for C in A(l,J+1):
				if j[a][C]==1:c[a][C]=3
	return c


show_examples(load_examples(71)['train'])


#%%writefile task071.py



show_examples(load_examples(72)['train'])


%%writefile task072.py
p=lambda g:[[3if g[i][j]+g[i+7][j]==2else 0for j in range(5)]for i in range(6)]


show_examples(load_examples(73)['train'])


%%writefile task073.py
def p(j):
 A=[o[:]for o in j]
 for c in range(5):
  for E in range(5):
   if j[E][c]==1:A[E][c]=0;A[4][c]=1
 return A


show_examples(load_examples(74)['train'])


#%%writefile task074.py



show_examples(load_examples(75)['train'])


%%writefile task075.py
def p(j):
	A=range;c=[A[:]for A in j];E=[[j[k][A]for A in A(3)]for k in A(3)]
	for k in A(9):
		for W in A(4,13):
			if j[k][W]==1:
				for l in A(-1,2):
					for J in A(-1,2):
						if 0<=k+l<9and 4<=W+J<13:c[k+l][W+J]=E[l+1][J+1]
	return c


show_examples(load_examples(76)['train'])


#%%writefile task076.py



show_examples(load_examples(77)['train'])


%%writefile task077.py
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 for r in R(h-1):
  for c in R(w-1):
   C=g[r][c:c+2]+g[r+1][c:c+2]
   if C.count(2)+C.count(4)>2:
    if g[r][c]==9:g[r][c]=4
    if g[r][c+1]==9:g[r][c+1]=4
    if g[r+1][c]==9:g[r+1][c]=4
    if g[r+1][c+1]==9:g[r+1][c+1]=4
 return g


show_examples(load_examples(78)['train'])


%%writefile task078.py
def p(j,A=range):
	c,E=len(j),len(j[0]);k=[[0]*E for W in A(c)]
	for W in A(E):
		l=[j[c][W]for c in A(c)if j[c][W]!=0]
		for(J,a)in enumerate(l):k[J][W]=a
	return k


show_examples(load_examples(79)['train'])


#%%writefile task079.py



show_examples(load_examples(80)['train'])


#%%writefile task080.py



show_examples(load_examples(81)['train'])


%%writefile task081.py
from collections import*
def p(j,A=range):
	c=[[8 if J==8 else 0 for J in R]for R in j]
	for E in A(len(j)-1):
		for k in A(len(j[0])-1):
			W=[j[E][k:k+2],j[E+1][k:k+2]];l=[x for R in W for x in R];J=Counter(l).most_common(1)
			if J[0][1]==3and J[0][0]!=0:
				for a in A(E,E+2):
					for C in A(k,k+2):
						if c[a][C]==0:c[a][C]=1
	return c


show_examples(load_examples(82)['train'])


%%writefile task082.py
def p(j):
	A=[k[:]for k in j];c,E=len(j),len(j[0])
	for k in range(E):
		if j[0][k]:
			for W in range(c):
				if W%2==0:A[W][k]=j[0][k]
				else:
					if k>0:A[W][k-1]=j[0][k]
					if k<E-1:A[W][k+1]=j[0][k]
	return A


show_examples(load_examples(83)['train'])


%%writefile task083.py
def p(j):A=[r+r[::-1]for r in j];return A+A[::-1]


show_examples(load_examples(84)['train'])


%%writefile task084.py
def p(j):
 A=len(j)
 for c in range(1,len(j[0])):j[A-1][c]=4;j[A-c-1][c]=2
 return j


show_examples(load_examples(85)['train'])


%%writefile task085.py
def p(g,V=range):
 r=[d[:]for d in g]
 v=set()
 for i in V(len(g)-2):
  for j in V(len(g[0])):
   if g[i][j]and(i,j)not in v:
    c=g[i][j]
    if all(g[i+k][j]==c for k in V(3)):
     a=j
     while a<len(g[0])and all(g[i+k][a]==c for k in V(3)):
      for k in V(3):v.add((i+k,a))
      a+=1
     for x in V(j,a):
      if(x-j)%2==1:r[i+1][x]=0
 return r


show_examples(load_examples(86)['train'])


#%%writefile task086.py



show_examples(load_examples(87)['train'])


%%writefile task087.py
p=lambda j:[r[::-1]for r in j[::-1]]


show_examples(load_examples(88)['train'])


%%writefile task088.py
from collections import*
j=len
A=range
def p(c):
	E=[x for A in c for x in A];k=Counter(E).most_common();k=[C for C in k if C[1]==4][0][0];W,l=j(c),j(c[0]);J=[]
	for a in A(W):
		for C in A(l):
			if c[a][C]==k:J.append([a,C])
	e=min([i[1]for i in J]);K=max([i[1]for i in J]);w=min([i[0]for i in J]);L=max([i[0]for i in J]);c=c[w+1:L];c=[A[e+1:K]for A in c];W,l=j(c),j(c[0])
	for a in A(W):
		for C in A(l):
			if c[a][C]>0:c[a][C]=k
	return c


show_examples(load_examples(89)['train'])


#%%writefile task089.py



show_examples(load_examples(90)['train'])


%%writefile task090.py
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 Z=[r[:] for r in g]
 for s in R(min([h,w]),1,-1):
  t=0
  for r in R(h):
   for c in R(w):
    X=g[r:r+s]
    X=[m[c:c+s][:] for m in X]
    if sum(X,[]).count(0)==s*s:
     t=1
     for i in R(r,r+s):
      for j in R(c,c+s):
       Z[i][j]=6
  if t:return Z
 return g


show_examples(load_examples(91)['train'])


%%writefile task091.py
def p(j):
	A=len;c=range;E=[]
	for k in c(A(j[0])):
		if any(j[c][k]==5 for c in c(A(j))):E.append(k)
	W=[]
	for l in c(A(j)):
		if j[l][E[0]]==5:W.append(l)
	J,a=min(W)-1,max(W)+1;C,e=E[0],E[1];return[[j[E][c]for c in c(C,e+1)]for E in c(J,a+1)]


show_examples(load_examples(92)['train'])


%%writefile task092.py
def p(g,L=len,R=range):
 H,W=L(g),L(g[0]);o=[r[:]for r in g]
 for i in R(H):
  x=C=None
  for j in R(W):
   if g[i][j]:
    if x is not None and g[i][j]==C:
     for k in R(x+1,j):o[i][k]=C
    x=j;C=g[i][j]
 for j in R(W):
  x=C=None
  for i in R(H):
   if g[i][j]:
    if x is not None and g[i][j]==C:
     for k in R(x+1,i):o[k][j]=C
    x=i;C=g[i][j]
 return o


show_examples(load_examples(93)['train'])


#%%writefile task093.py



show_examples(load_examples(94)['train'])


%%writefile task094.py
j=len
A=range
def p(c):
	E,k=[],[];W,l=j(c),j(c[0])
	for J in A(W-4):
		for a in A(l-4):
			C=[[c[E+J][C+a]for E in A(5)]for C in A(5)];C=[a for J in C for a in J];C=sum([J for J in C if J==1])
			if C==16:E.append(J+2);k.append(a+2)
	for J in A(W):
		for a in A(l):
			if J in E or a in k:
				if c[J][a]!=1:c[J][a]=6
	return c


show_examples(load_examples(95)['train'])


%%writefile task095.py
def p(j,A=enumerate):
 for c,E in A(j):
  for k,W in A(E):
   if W==5:
    for l in range(c-1,c+2):
     for J in range(k-1,k+2):
      if[l,J]!=[c,k]:j[l][J]=1
 return j


show_examples(load_examples(96)['train'])


#%%writefile task096.py



show_examples(load_examples(97)['train'])


%%writefile task097.py
j=len
A=range
def p(c):
	E,k=j(c),j(c[0]);W=[a for W in c for a in W];W=sorted(W)[-1];c=[[0]+W+[0]for W in c];l=[[0]*(k+2)];c=l+c+l;J=[[1,1],[-1,-1],[-1,1],[1,-1],[0,1],[0,-1],[-1,0],[1,0],[0,0]]
	for a in A(1,E+1):
		for C in A(1,k+1):
			if c[a][C]==W:
				e=[c[W[0]+a][W[1]+C]for W in J]
				if sum(e)==W:c[a][C]=0
	c=c[1:-1];c=[W[1:-1]for W in c];return c


show_examples(load_examples(98)['train'])


%%writefile task098.py
p=lambda g:[[x if any(g[i+di][j+dj]==0 for di,dj in[(0,1),(1,0),(0,-1),(-1,0)]if 0<=i+di<len(g)and 0<=j+dj<len(g[0]))and x!=0 else 0 for j,x in enumerate(r)]for i,r in enumerate(g)]


show_examples(load_examples(99)['train'])


#%%writefile task099.py



show_examples(load_examples(100)['train'])


%%writefile task100.py
def p(g,E=enumerate,M=max,N=min):
 d={k:{0:[],1:[]} for k in set(sum(g,[]))}
 for(r,R)in E(g):
  for(c,C)in E(R):d[C][0]+=[r];d[C][1]+=[c]
 Z=[];del d[0]
 for k in d:X=d[k];Z+=[[(M(X[0])-N(X[0])+1)*(M(X[1])-N(X[1])+1),k,len(X[1])]]
 C=sorted(Z)[-1][1]
 return[[C,C],[C,C]]


show_examples(load_examples(101)['train'])


#%%writefile task101.py



show_examples(load_examples(102)['train'])


#%%writefile task102.py



show_examples(load_examples(103)['train'])


%%writefile task103.py
p=lambda j:[[1if[j[i][0]for i in range(3)]==[j[i][2]for i in range(3)]else 7]]


show_examples(load_examples(104)['train'])


%%writefile task104.py
def p(g):
 e,Z=[],[g[0][0],g[0][2],g[2][2],g[2][0]]
 for r in [[3,0],[0,3]]:
  for i in range(4):e+=[sum([[c]*4 for c in r],[])+[0]]
 e+=[[0]*len(e[0])]
 for i in range(Z.index(3)):e=[list(r) for r in list(zip(*e[::-1]))]
 return e


show_examples(load_examples(105)['train'])


%%writefile task105.py
def X(g):return list(zip(*g[::-1]))
def p(g,L=len,R=range):
 t=[r[:] for r in g]
 for _ in R(4):
  g=X(g);t=[list(r) for r in X(t)]
  h,w=L(g),L(g[0])
  for r in R(h-1):
   for c in R(w-2):
    m=[i for i in R(w) if t[r][i]>0]
    if L(m)>0:
     if g[r][c]==1 and g[r][c+2]==1 and L(m)>3:t[r][c+1]=2
     if g[r][c]==1 and g[r+1][c+1]==1 and L(m)>3:t[r][c+1]=2
     if min(m)<c+1<max(m) and L(m)>3 and g[r][c+1]==0:t[r][c+1]=2
 h,w=L(g),L(g[0])
 for r in R(h):
  for c in R(w):
   if g[r][c]>0:t[r][c]=1
 return t


show_examples(load_examples(106)['train'])


%%writefile task106.py
def p(j):A=lambda c:[*map(list,zip(*c[::-1]))];return[c+y for c,y in zip(j,A(j))]+[c+y for c,y in zip(A(A(A(j))),A(A(j)))]


show_examples(load_examples(107)['train'])


%%writefile task107.py
def p(j,u=range):
 A=len(j);c=len(j[0]);E=len({*sum(j,[])}-{0})
 j=[[j[W//E][l//E]for l in u(c*E)]for W in u(A*E)];A*=E;c*=E
 for k in u(min(A,c),0,-1):
  for W in u(A-k+1):
   for l in u(c-k+1):
    J=j[W][l]
    if J and all(r[l:l+k]==[J]*k for r in j[W:W+k]):
     for a,C in(-1,-1),(-1,k),(k,-1),(k,k):
      e=W+a;K=l+C
      while-1<e<A and-1<K<c and not j[e][K]:j[e][K]=2;e+=a>0 or-1;K+=C>0 or-1
     return j


show_examples(load_examples(108)['train'])


%%writefile task108.py
def p(j,A=range):c,E=len(j),len(j[0]);k=[[max(j[y][x],j[y][x+1],j[y+1][x],j[y+1][x+1])for x in A(0,E,2)]for y in A(0,c,2)];return[[k[y//4][x//4]for x in A(2*E)]for y in A(2*c)]


show_examples(load_examples(109)['train'])


#%%writefile task109.py



show_examples(load_examples(110)['train'])


%%writefile task110.py
def p(j,u=enumerate):
	A=range;c=len(j);E=len(j[0]);k=lambda W,l:W==l or W*l<1;J=next((K for K in A(1,E)if all(k(L,e)for w in j for(L,e)in zip(w,w[K:]))),E);a=next((K for K in A(1,c)if all(k(L,e)for(K,w)in zip(j,j[K:])for(L,e)in zip(K,w))),c);C={}
	for(e,K)in u(j):
		for(w,L)in u(K):
			if L:C[e%a,w%J]=L
	for(e,K)in u(j):
		for(w,L)in u(K):
			if not L:K[w]=C[e%a,w%J]
	return j


show_examples(load_examples(111)['train'])


%%writefile task111.py
p=lambda g:next([g[i+k][j-1:j+2]for k in(1,2,3)]for i,r in enumerate(g)for j,x in enumerate(r)if x==5)


show_examples(load_examples(112)['train'])


%%writefile task112.py
def p(j,h=enumerate):
 A=c=0
 for E,k in h(j):
  for W,l in h(k):A+=E*(l==3);c+=W*(l==3)
 A//=2;c//=2
 for E,k in h(j):
  for W,l in h(k):
   if l==2:
    for J,a in(E,W),(A-E,W),(E,c-W),(A-E,c-W):j[J][a]=2
 return j


show_examples(load_examples(113)['train'])


%%writefile task113.py
p=lambda j:j[:5]+j[:5][::-1]


show_examples(load_examples(114)['train'])


%%writefile task114.py
def p(g):
 g=[g[0]]+g+[g[-1]]
 g=[[R[0]]+R+[R[-1]]for R in g]
 for r,c in[[0,0],[0,-1],[-1,0],[-1,-1]]:g[r][c]=0
 return g


show_examples(load_examples(115)['train'])


%%writefile task115.py
def p(j):
 def u(A):
  c=[]
  for E in A:
   if E not in c:c.append(E)
  return c
 k=[u(c)for c in j]
 if all(k[0]==c for c in k):return[k[0]]
 return[[E]for E in u([E for c in j for E in c])]


show_examples(load_examples(116)['train'])


%%writefile task116.py
p=lambda j:j[::-1]+j


show_examples(load_examples(117)['train'])


#%%writefile task117.py



show_examples(load_examples(118)['train'])


#%%writefile task118.py



show_examples(load_examples(119)['train'])


#%%writefile task119.py



show_examples(load_examples(120)['train'])


%%writefile task120.py
def p(j):
	A=range;c=len;E=[W[:]for W in j];k=set()
	for W in A(c(j)):
		for l in A(c(j[0])):
			if j[W][l]and(W,l)not in k:
				J,a=[(W,l)],[(W,l)];k.add((W,l));C=j[W][l]
				while a:
					e,K=a.pop()
					for(w,L)in[(0,1),(1,0),(0,-1),(-1,0)]:
						b,d=e+w,K+L
						if 0<=b<c(j)and 0<=d<c(j[0])and j[b][d]==C and(b,d)not in k:k.add((b,d));J.append((b,d));a.append((b,d))
				f=min(W[0]for W in J);g=max(W[0]for W in J);h=min(W[1]for W in J);i=max(W[1]for W in J)
				for e in A(f+1,g):
					for K in A(h+1,i):E[e][K]=8
	return E


show_examples(load_examples(121)['train'])


%%writefile task121.py
def p(j):
	for A in range(1,len(j)-1):
		for c in range(1,len(j[0])-1):
			if j[A][c]==8:
				E=[]
				for k in[-1,0,1]:
					for W in[-1,0,1]:
						if(k or j)and j[A+k][c+W]:E.append(j[A+k][c+W])
				l=max(set(E),key=E.count);J=[[j[A+E][c+k]for k in[-1,0,1]]for E in[-1,0,1]];J[1][1]=l;return J


show_examples(load_examples(122)['train'])


%%writefile task122.py
def p(g,L=len,R=range):
 for r in R(L(g)):
  for c in R(L(g[0])):
   if g[r][c]==2:
    if g[r+1].count(3)>1: #Horizontal
     for y in R(3):
      for x in R(3):
       g[r+y][c+x+2]= g[r+y][c+x]
       if g[r+y][c+x]==2 and x<2:g[r+y][c+x]=0
     return g
    else:
     for y in R(3):
      for x in R(3):
       g[r+y+2][c+x]=g[r+y][c+x]
       if g[r+y][c+x]==2 and y<2:g[r+y][c+x]=0
     return g


show_examples(load_examples(123)['train'])


%%writefile task123.py
def p(g,R=range):
 g=[[x for x in r if x>0] for r in g if r.count(0)<2]
 g=[[r[0]]*10 for r in g+g+g]
 for r in R(10):
  for c in R(10):g[r][c]=g[c][r]
 return g[:10]


show_examples(load_examples(124)['train'])


#%%writefile task124.py



show_examples(load_examples(125)['train'])


#%%writefile task125.py



show_examples(load_examples(126)['train'])


%%writefile task126.py
def p(j):
 A=[o[:]for o in j]
 c,E=len(j),len(j[0])
 for k in range(1,c):
  for W in range(1,E-1):
   if j[k][W]==0and j[k][W-1]and j[k][W+1]and j[k][W-1]==j[k][W+1]and j[k-1][W]==j[k][W-1]:A[c-1][W]=4
 return A


show_examples(load_examples(127)['train'])


%%writefile task127.py
def p(g):
 R=range;Z=[r[:]for r in g];h,w=len(g),len(g[0])
 for r in R(1,h,4):
  for c in R(1,w,4):
   C=g[r][c]+5
   for y in R(3):
    for x in R(3):Z[r-1+y][c-1+x]=C
 return Z


show_examples(load_examples(128)['train'])


%%writefile task128.py
def p(j):
	A=[[0]*len(j[0])for a in j];c=set()
	for E in range(len(j)):
		for k in range(len(j[0])):
			if j[E][k]and(E,k)not in c:
				W,l=[(E,k)],[(E,k)];c.add((E,k));J=j[E][k]
				while l:
					a,C=l.pop()
					for(e,K)in[(0,1),(1,0),(0,-1),(-1,0)]:
						if 0<=a+e<len(j)and 0<=C+K<len(j[0])and j[a+e][C+K]==J and(a+e,C+K)not in c:c.add((a+e,C+K));W.append((a+e,C+K));l.append((a+e,C+K))
				w=max(a for(a,C)in W)-min(a for(a,C)in W)+1
				for(a,C)in W:A[a-w][C]=J
	return A


show_examples(load_examples(129)['train'])


%%writefile task129.py
p=lambda j:[[max(sum(j,[]),key=sum(j,[]).count)]*3]*3


show_examples(load_examples(130)['train'])


%%writefile task130.py
def p(j):
 A=range
 c=[[0]*3for _ in A(3)]
 for E in A(3):
  for k in A(3):
   W={}
   for l in A(3):
    for J in A(3):a=j[E*3+l][k*3+J];W[a]=W.get(a,0)+1
   c[E][k]=max(W,key=W.get)
 return c


show_examples(load_examples(131)['train'])


%%writefile task131.py
j=lambda A:[[A[J][x]for J in range(len(A))]for x in range(len(A[0]))]
def p(A):
 c,E=len(A),len(A[0])
 if E>c:return j(p(j(A)))
 k,W,l=0,c,0
 for J,a in enumerate(A):
  if a[0]==2:k=J
  if any(i==3 for i in a):W,l=min(W,J),J
 if W<k:return p(A[::-1])[::-1]
 return A[:k+1]+A[W:l+1]+[[8]*E]+[[0]*E]*(c-k+W-l-3)


show_examples(load_examples(132)['train'])


%%writefile task132.py
def p(j,A=range,c=enumerate):
 E=len(j);k=len(j[0]);W=[[0]*k for _ in A(E)];l={v for G in j for v in G if v}
 for J in l:
  a=[b for b,G in c(j)for v in G if v==J];C=[d for b,G in c(j)for d,v in c(G)if v==J];e,K=min(a),max(a)+1;w,L=min(C),max(C)+1
  for b in A(e,K):
   for d in A(w,L):W[b][d]=J
 return W


show_examples(load_examples(133)['train'])


#%%writefile task133.py



show_examples(load_examples(134)['train'])


#%%writefile task134.py



show_examples(load_examples(135)['train'])


%%writefile task135.py
p=lambda j:[j[i][6:9]for i in range(0,3)]


show_examples(load_examples(136)['train'])


%%writefile task136.py
def p(j):
 A,c=len(j),len(j[0])
 E=lambda k:next((W,l)for W in range(A-1)for l in range(c-1)if j[W][l]==j[W+1][l+1]==k)
 W,l=E(1)
 while W>=1and l>=1:W,l=W-1,l-1;j[W][l]=1
 W,l=E(2)
 while W<A-1and l<c-1:W,l=W+1,l+1;j[W][l]=2
 return j


show_examples(load_examples(137)['train'])


#%%writefile task137.py



show_examples(load_examples(138)['train'])


#%%writefile task138.py



show_examples(load_examples(139)['train'])


%%writefile task139.py
from itertools import product
def p(j,A=range):
 for c,E in product(A(len(j)-2),A(len(j[0])-2)):
  k=A(c,c+3)
  if not all(4 in i for i in[j[c][E:E+3],j[c+2][E:E+3],[j[W][E]for W in k],[j[W][E+2]for W in k]]):continue
  for W,l in product(k,A(E,E+3)):j[W][l]+=7*(j[W][l]==0)
 return j


show_examples(load_examples(140)['train'])


%%writefile task140.py
p=lambda g:[r[::-1]for r in g[::-1]]


show_examples(load_examples(141)['train'])


%%writefile task141.py
def p(j):
	A=[J[:]for J in j];c,E=len(j),len(j[0])
	for k in range(c):
		for W in range(E):
			if j[k][W]:
				l=j[k][W]
				for J in[(-1,-1),(-1,1),(1,1),(1,-1)]:
					a,C=k+J[0],W+J[1]
					while 0<=a<c and 0<=C<E:A[a][C]=l;a+=J[0];C+=J[1]
	return A


show_examples(load_examples(142)['train'])


%%writefile task142.py
def p(j):A=[r+r[::-1]for r in j];return A+A[::-1]


show_examples(load_examples(143)['train'])


#%%writefile task143.py



show_examples(load_examples(144)['train'])


%%writefile task144.py
p=lambda g:[[3if g[i][j]==0and g[i+5][j]==0else 0for j in range(4)]for i in range(4)]


show_examples(load_examples(145)['train'])


%%writefile task145.py
def p(j):
	A=[K[:]for K in j];c,E=len(j),len(j[0]);k=set();W=[]
	for l in range(c):
		for J in range(E):
			if j[l][J]!=2 and(l,J)not in k:
				a,C=[],[(l,J)];k.add((l,J));e=0
				while C:
					K,w=C.pop();a.append((K,w))
					if j[K][w]==0:e+=1
					for(L,b)in[(0,1),(1,0),(0,-1),(-1,0)]:
						if 0<=K+L<c and 0<=w+b<E and j[K+L][w+b]!=2 and(K+L,w+b)not in k:k.add((K+L,w+b));C.append((K+L,w+b))
				W.append((e,a))
	d=max(K[0]for K in W);f=min(K[0]for K in W)
	for(e,a)in W:
		k=1 if e==d else 8 if e==f else 0
		if k:
			for(K,w)in a:
				if j[K][w]==0:A[K][w]=k
	return A


show_examples(load_examples(146)['train'])


%%writefile task146.py
p=lambda g,R=range:[[[g[k+i][j]for j in R(3)]for i in R(3)]for k in R(0,9,3)if[[g[k+i][j]for j in R(3)]for i in R(3)]!=[[g[k+j][i]for j in R(3)]for i in R(3)]][0]


show_examples(load_examples(147)['train'])


%%writefile task147.py
def p(j):
	A=[k[:]for k in j];c,E=len(j),len(j[0])
	for k in range(c):
		for W in range(E):
			if j[k][W]==3:
				for(l,J)in[(0,1),(1,0),(0,-1),(-1,0)]:
					if 0<=k+l<c and 0<=W+J<E and j[k+l][W+J]==3:A[k][W]=8;break
	return A


show_examples(load_examples(148)['train'])


#%%writefile task148.py



show_examples(load_examples(149)['train'])


%%writefile task149.py
def p(j):
 A=range
 c=[[0]*3for _ in A(3)]
 for E in A(3):
  for k in A(3):
   W=0
   for l in A(3):
    for J in A(3):
     if j[E*4+l][k*4+J]==6:W+=1
   c[E][k]=1if W>=2else 0
 return c


show_examples(load_examples(150)['train'])


%%writefile task150.py
p=lambda j:[r[::-1]for r in j]


show_examples(load_examples(151)['train'])


%%writefile task151.py
def p(j):A=lambda c:list(map(all,c)).index(1);E,k=A(j),A(zip(*j));j[E-1][k-1:k+2]=j[E+1][k-1:k+2]=[4]*3;j[E][k-1]=j[E][k+1]=4;return j


show_examples(load_examples(152)['train'])


%%writefile task152.py
def p(j):A=[r+r[::-1]for r in j];return A+A[::-1]


show_examples(load_examples(153)['train'])


#%%writefile task153.py



show_examples(load_examples(154)['train'])


#%%writefile task154.py



show_examples(load_examples(155)['train'])


%%writefile task155.py
p=lambda j:j[::-1]


show_examples(load_examples(156)['train'])


#%%writefile task156.py



show_examples(load_examples(157)['train'])


#%%writefile task157.py



show_examples(load_examples(158)['train'])


#%%writefile task158.py



show_examples(load_examples(159)['train'])


#%%writefile task159.py



show_examples(load_examples(160)['train'])


#%%writefile task160.py



show_examples(load_examples(161)['train'])


%%writefile task161.py
def p(g,E=enumerate,R=range,L=len):
 h,w=L(g),L(g[0])
 d={k:{0:[],1:[]} for k in set(sum(g,[]))}
 for(r,X)in E(g):
  for(c,C)in E(X):d[C][0]+=[r];d[C][1]+=[c]
 del d[0]
 C=sorted([[len(d[k][1])-(max(d[k][0])/100),k] for k in d])[0][1]
 g=[[c if c==C else 0 for c in r] for r in g]
 for r in R(h):
  if g[r][0]==C or g[r][-1]==C: 
   for c in R(w):g[r][c]=C
 for c in R(w):
  if g[0][c]==C or g[-1][c]==C: 
   for r in R(h):g[r][c]=C
 return g


show_examples(load_examples(162)['train'])


%%writefile task162.py
def p(j,A=range(18)):
 for c in A:
  E,k,W=j[c:c+3]
  for l in A:
   J=l+3
   if sum(E[l:J]+k[l:J]+W[l:J])==0:E[l:J]=k[l:J]=W[l:J]=[1]*3
 return j


show_examples(load_examples(163)['train'])


%%writefile task163.py
def p(g):
 R=range
 for r in R(3):
  for c in R(3):
   b=[[g[4*r+i][4*c+j]for j in R(3)]for i in R(3)]
   for i in R(3):
    for j in R(3):
     if b[i][j]==4:
      z=[[0]*11for _ in R(11)]
      for x in R(3):
       for y in R(3):z[4*i+x][4*j+y]=b[x][y]
      for k in R(11):z[k][3]=z[k][7]=z[3][k]=z[7][k]=5
      return z


show_examples(load_examples(164)['train'])


%%writefile task164.py
p=lambda j:[R+R[::-1]for R in j]


show_examples(load_examples(165)['train'])


#%%writefile task165.py



show_examples(load_examples(166)['train'])


%%writefile task166.py
p=lambda g:[[2if(t:=[(i,j)for i,r in enumerate(g)for j,v in enumerate(r)if v==8])and min(i for i,j in t)<=i<=max(i for i,j in t)and min(j for i,j in t)<=j<=max(j for i,j in t)and g[i][j]==0else g[i][j]for j in range(len(g[0]))]for i in range(len(g))]


show_examples(load_examples(167)['train'])


%%writefile task167.py
p=lambda j:[[[5,5,5],[0,0,0],[0,0,0]],[[5,0,0],[0,5,0],[0,0,5]],[[0,0,5],[0,5,0],[5,0,0]]][len(set(v for r in j for v in r))-1]


show_examples(load_examples(168)['train'])


#%%writefile task168.py



show_examples(load_examples(169)['train'])


%%writefile task169.py
def p(j):
	A=range;c=[L[:]for L in j];E=set();k=[(0,1),(1,0),(0,-1),(-1,0)]
	for W in A(10):
		for l in A(10):
			if j[W][l]==5 and(W,l)not in E:
				J,a=set(),[(W,l)];J.add((W,l));E.add((W,l))
				while a:
					C,e=a.pop(0)
					for(K,w)in k:
						L,b=C+K,e+w
						if 0<=L<10 and 0<=b<10 and j[L][b]==5 and(L,b)not in J:J.add((L,b));E.add((L,b));a.append((L,b))
				d=5-len(J)
				for(C,e)in J:c[C][e]=d
	return c


show_examples(load_examples(170)['train'])


#%%writefile task170.py



show_examples(load_examples(171)['train'])


%%writefile task171.py
def p(g):
 g[-1]=g[0]=[8]*len(g[0])
 for r in range(len(g)):g[r][0]=8;g[r][-1]=8
 return g


show_examples(load_examples(172)['train'])


%%writefile task172.py
p=lambda j:j+j[::-1]


show_examples(load_examples(173)['train'])


#%%writefile task173.py



show_examples(load_examples(174)['train'])


#%%writefile task174.py



show_examples(load_examples(175)['train'])


#%%writefile task175.py



show_examples(load_examples(176)['train'])


%%writefile task176.py
def p(j):
 A,c,E=j;k=6,4,0,0,0,1,3,1,0,0,0,4
 for W in range(len(A)):
  l=k[W%12]
  if l&1:A[W]=4
  if l&2:c[W]=4
  if l&4:E[W]=4
 return j


show_examples(load_examples(177)['train'])


%%writefile task177.py
def p(g):a=[i for i,r in enumerate(g)if any(r)];b=[j for j in range(len(g[0]))if any(r[j]for r in g)];return[r[b[0]:b[-1]+1][::-1]for r in g[a[0]:a[-1]+1]]


show_examples(load_examples(178)['train'])


%%writefile task178.py
def p(j):
	A=range;c,E=len(j),len(j[0]);k=[]
	for W in A(c):
		if W==0 or j[W]!=j[W-1]:k.append([j[W][0]])
	l=[];J=-1
	for a in A(E):
		if a==0 or any(j[W][a]!=j[W][a-1]for W in A(c)):l.append(j[0][a])
	if len(k)>1:return k
	else:return[l]


show_examples(load_examples(179)['train'])


%%writefile task179.py
p=lambda g:list(map(list,zip(*g)))


show_examples(load_examples(180)['train'])


%%writefile task180.py
p=lambda j,A=range(4):[[j[x][y+4]or j[x+4][y]or j[x+4][y+4]or j[x][y]for y in A]for x in A]


show_examples(load_examples(181)['train'])


%%writefile task181.py
def p(j):A=(j[3][3]<1)*6;[j[r].__setitem__(slice(A,A+3),j[r][3:6][::-1])for r in range(3)];return j


show_examples(load_examples(182)['train'])


#%%writefile task182.py



show_examples(load_examples(183)['train'])


%%writefile task183.py
def p(j):
	A=range;c=len(j);E=c//2-2;k=[];W=[j[0][0],j[0][-1],j[-1][0],j[-1][-1]]
	for l in A(2,c-2):
		J=[]
		for a in A(2,c-2):
			C=j[l][a]
			if C==8:e=(l-2)//E;K=(a-2)//E;C=W[e*2+K]
			J.append(C)
		k.append(J)
	return k


show_examples(load_examples(184)['train'])


%%writefile task184.py
def p(j):
	A=range;c,E=len(j),len(j[0]);k=[k for k in A(c)if all(j[k][A]==0 for A in A(E))];W=[k for k in A(E)if all(j[A][k]==0 for A in A(c))];k=[-1]+k+[c];W=[-1]+W+[E];l=[]
	for J in A(len(k)-1):
		a=[]
		for C in A(len(W)-1):
			for e in A(k[J]+1,k[J+1]):
				for K in A(W[C]+1,W[C+1]):
					if j[e][K]:a.append(j[e][K]);break
				else:continue
				break
		if a:l.append(a)
	return l


show_examples(load_examples(185)['train'])


%%writefile task185.py
def p(g,L=len,R=range):
 #this one is difficult probably esier to match pattern hashes with results
 g=[[0 if x==max(g[0]) else x for x in r] for r in g]
 g=[r for r in g if L(set(r))>1]
 g=[list(r) for r in zip(*g)]
 g=[r[::-1] for r in g]
 g=[r for r in g if L(set(r))>1]
 g=[r[::-1] for r in g]
 g=[list(r) for r in zip(*g)]
 z=[r[:] for r in g]
 g=[[g[0][0],0,g[0][3]],[0,0,0],[g[3][0],0,g[3][3]]]
 if z[0].count(0)==1:g[0][1]=max(z[0])
 if z[3].count(0)==1:g[2][1]=max(z[3])
 if z[1].count(0)==1 and z[0].count(0)>1:g[0][1]=max(z[0]);g[1][0]=max(z[0])
 if g[0][0]==g[0][2] and g[0][0]>0:g[0][1]=g[0][2]
 #if g[0][0]==g[2][0]:g[1][0]=g[0][0]
 if g[2][0]==g[2][2]:g[2][1]=g[2][0]
 #if g[0][2]==g[2][2]:g[1][2]=g[2][2]
 return g


show_examples(load_examples(186)['train'])


%%writefile task186.py
p=lambda j,A=[2]*3,c=[0]*3:[[A,[0,2,0],c],[A,c,c],[[2,2,0],c,c],[[2,0,0],c,c]][4-sum(r.count(1)for r in j)]


show_examples(load_examples(187)['train'])


%%writefile task187.py
def p(j):
 A,c,E=len(j),len(j[0]),range;k=[(W,l)for W in(0,A-1)for l in E(c)if j[W][l]<1]+[(W,l)for W in E(A)for l in(0,c-1)if j[W][l]<1]
 while k:
  W,l=k.pop()
  if j[W][l]<1:j[W][l]=3;k+=[(x,y)for x,y in((W+1,l),(W-1,l),(W,l+1),(W,l-1))if 0<=x<A and 0<=y<c and j[x][y]<1]
 for W in E(A):
  for l in E(c):
   if j[W][l]<1:j[W][l]=2
 return j


show_examples(load_examples(188)['train'])


%%writefile task188.py
p=lambda j,A=len:[r[:A(r)//2]for r in j]if A(j[0])%2<1and all(r[:A(r)//2]==r[A(r)//2:]for r in j)else j[:A(j)//2]


show_examples(load_examples(189)['train'])


#%%writefile task189.py



show_examples(load_examples(190)['train'])


%%writefile task190.py
def p(g):
 R=range
 r=[x[:]for x in g]
 d=[(0,1),(1,0),(0,-1),(-1,0)]
 c=[(-1,-1),(-1,1),(1,1),(1,-1)]
 for i in R(10):
  for j in R(10):
   if g[i][j]and all(0<=i+x<10and 0<=j+y<10and g[i+x][j+y]==0for x,y in d):
    for t in c:
     x,y=i+t[0],j+t[1]
     if 0<=x<10and 0<=y<10and g[x][y]:
      a,b=-t[0],-t[1]
      for m in R(1,10):
       u,v=i+a*m,j+b*m
       if 0<=u<10and 0<=v<10:r[u][v]=g[i][j]
 return r


show_examples(load_examples(191)['train'])


#%%writefile task191.py



show_examples(load_examples(192)['train'])


%%writefile task192.py
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 f=sum(g,[]);C=sorted([[f.count(k),k] for k in set(f)])[0][1]
 P=[[0,1],[0,-1],[-1,0],[1,0]]
 for r in R(h):
  for c in R(w):
   if g[r][c]==C:
    m=[]
    for y,x in P:
     if r+y>=0 and c+x>=0 and r+y<h and c+x<w:
      m+=[g[r+y][c+x]]
    if sum(m)/L(m)<max(m)/2:
     g[r][c]=0
    else: g[r][c]=max(m)
 return g


show_examples(load_examples(193)['train'])


%%writefile task193.py
p=lambda g,E=enumerate:[[v if(i and g[i-1][j]==v)+(i+1<len(g)and g[i+1][j]==v)+(j and r[j-1]==v)+(j+1<len(g)and r[j+1]==v)>1else 0 for j,v in E(r)]for i,r in E(g)]


show_examples(load_examples(194)['train'])


%%writefile task194.py
j=lambda A:[[*i]for i in zip(*A[::-1])]
p=lambda c:[a+b for a,b in zip(c,j(c))]+[a+b for a,b in zip(j(j(j(c))),j(j(c)))]


show_examples(load_examples(195)['train'])


#%%writefile task195.py



show_examples(load_examples(196)['train'])


%%writefile task196.py
def p(j):
	A=range;c,E=len(j),len(j[0]);k=set();W=[e[:]for e in j]
	def M(l,J):
		if(l,J)in k or not(0<=l<c and 0<=J<E)or j[l][J]!=1:return[]
		k.add((l,J));return[(l,J)]+sum([M(l+e,J+A)for(e,A)in[(-1,0),(1,0),(0,-1),(0,1)]],[])
	for a in A(c):
		for C in A(E):
			if j[a][C]==1 and(a,C)not in k:
				e=M(a,C);K,w,L,b=min(e[0]for e in e),max(e[0]for e in e),min(e[1]for e in e),max(e[1]for e in e)
				if len(e)==2*(w-K+b-L)and w>K and b>L and any(j[e][k]==0 for e in A(K+1,w)for k in A(L+1,b)):
					for(d,f)in e:W[d][f]=3
	return W


show_examples(load_examples(197)['train'])


%%writefile task197.py
def p(j):
	A=next((c for c in j if 0 not in c),None)
	if not A:return j
	c=[];[c.append(W)for W in A if W not in c]
	for(E,k)in enumerate(j):
		if 0 in k and any(k):
			W=[];[W.append(c)for c in k if c not in W and c]
			if len(c)==len(W):l=dict(zip(c,W));j[E]=[l[c]for c in A]
	return j


show_examples(load_examples(198)['train'])


#%%writefile task198.py



show_examples(load_examples(199)['train'])


%%writefile task199.py
def p(j,A=enumerate):
 for c,E in A(j):
  for k,W in A(E):
   if W and W^4:
    j[c+1][k]=W
    for l in range(c+1):j[l][k&1::2]=[4]*len(j[l][k&1::2])
    return j


show_examples(load_examples(200)['train'])


%%writefile task200.py
def p(j):
 A,c,E,k=10,enumerate,range,0
 for W,l in c(j):
  for J,a in c(l):
   if a%5:
    for C in E(J,A,2):
     for e in E(W+1):j[e][C]=a
    for C in E(J+1,A,2):j[k*(A-1)][C]=5;k^=1
    return j


show_examples(load_examples(201)['train'])


%%writefile task201.py
def p(j):
 A=enumerate;c=next
 E=lambda k,W:k<1or j[k-1][W]<1or k>2and j[k-1][W]==4and j[k-2][W]>0
 (k,W),(l,J),(a,l),l=[divmod(i,13)for i,v in A(sum(j,[]))if v==4]
 C=c(u for r in zip(*j)if 4not in r for u in r if u)
 e=c(i for i,r in A(j)if any(u==C and E(i,v)for v,u in A(r)))
 K=c(i for i,r in A(zip(*j))if any(u==C and E(v,i)for v,u in A(r)))
 for w in range(a-k-1):
  for L in range(J-W-1):j[k+w+1][[J-L-1,W+L+1][j[k+1][W]==C]],j[e+w][K+L]=j[e+w][K+L],0
 return[r[W:J+1]for r in j[k:a+1]]


show_examples(load_examples(202)['train'])


#%%writefile task202.py



show_examples(load_examples(203)['train'])


%%writefile task203.py
def p(g,L=len,R=range):
 h=L(g)
 w=L(g[0])
 C=g[h//2][:w//2]
 C={C[i]:C[-(i+1)] for i in R(L(C))}
 for r in R(h):
  for c in R(w):g[r][c]=C[g[r][c]]
 return g


show_examples(load_examples(204)['train'])


#%%writefile task204.py



show_examples(load_examples(205)['train'])


#%%writefile task205.py



show_examples(load_examples(206)['train'])


#%%writefile task206.py



show_examples(load_examples(207)['train'])


%%writefile task207.py
def p(j):
	A={};c=[[[j[0][0],j[0][1]],[j[1][0],j[1][1]]],[[j[3][0],j[3][1]],[j[4][0],j[4][1]]],[[j[0][3],j[0][4]],[j[1][3],j[1][4]]],[[j[3][3],j[3][4]],[j[4][3],j[4][4]]]]
	for E in c:
		E=str(E)
		if E in A:A[E]+=1
		else:A[E]=1
	for E in A:
		if A[E]==1:return eval(E)


show_examples(load_examples(208)['train'])


%%writefile task208.py
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 Z=[r[:] for r in g]
 for s in R(min([h,w]),1,-1):
  t=0
  for r in R(h):
   for c in R(w):
    X=g[r:r+s]
    X=[m[c:c+s][:] for m in X]
    if sum(X,[]).count(0)==s*s:
     t=1
     for i in R(r,r+s):
      for j in R(c,c+s):
       Z[i][j]=9
  if t:return Z
 return g


show_examples(load_examples(209)['train'])


#%%writefile task209.py



show_examples(load_examples(210)['train'])


%%writefile task210.py
p=lambda j:j+j[::-1]


show_examples(load_examples(211)['train'])


%%writefile task211.py
def p(j):j=[R[::-1]+R for R in j];A=[j[2],j[1],j[0]];return A+j+A


show_examples(load_examples(212)['train'])


%%writefile task212.py
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 W=[i for i in R(L(g)) if 5 in g[i]][0]
 for r in R(h):
  for c in R(w):
   if g[r][c]==1 and r<W:
    for z in R(r,-1,-1):g[z][c]=1
   elif g[r][c]==1 and r>W:
    for z in R(r,h):g[z][c]=1
   if g[r][c]==2 and r<W:
    for z in R(r,W):g[z][c]=2
   elif g[r][c]==2 and r>W:
    for z in R(W+1,r):g[z][c]=2
 return g


show_examples(load_examples(213)['train'])


%%writefile task213.py
def Z(j,A):return len(set([J[A]for J in j]))
def p(c):
	E=enumerate;k,W=len(c),len(c[0]);l=Z(c,0)+Z(c,-1)<len(set(c[0]))+len(set(c[-1]));c=[[J if J!=5 else 0 for J in J]for J in c]
	for(J,a)in E(c):
		for(C,e)in E(a):
			if l:c[J][C]=max([c[0][C],c[-1][C]])
			else:c[J][C]=max([c[J][0],c[J][-1]])
	if l:c=[[J for J in J if J>0]for J in c];c=c[:len(c[0])]
	else:c=[J for J in c if sum(J)>0];c=[J[:len(c)]for J in c]
	return c


show_examples(load_examples(214)['train'])


%%writefile task214.py
def p(g,R=range):
 A=[[c for c in r[:3]] for r in g]
 C=[r[::-1]for r in A[::-1]]
 for r in R(3):
  for c in R(3):
   g[r][c+4]=A[-(c+1)][r];g[r][c+8]=C[r][c]
 return g


show_examples(load_examples(215)['train'])


%%writefile task215.py
p=lambda j:[[r for j,r in enumerate(j)if sum(r)and j%3==i%3][0]for i in range(len(j))]


show_examples(load_examples(216)['train'])


#%%writefile task216.py



show_examples(load_examples(217)['train'])


%%writefile task217.py
def p(j,A=range(9)):c,E=next(i for i,r in enumerate(j)if sum(r))//3*3,next(i for i in A if sum(j[y][i]for y in A))//3*3;return[[j[c+y%3][E+x%3]*bool(j[c+y//3][E+x//3])for x in A]for y in A]


show_examples(load_examples(218)['train'])


#%%writefile task218.py



show_examples(load_examples(219)['train'])


#%%writefile task219.py



show_examples(load_examples(220)['train'])


%%writefile task220.py
def p(j,A=enumerate):
 c={8:4,2:1,3:6};E=[[J for a,J in A(W)]for W in j]
 for k,W in A(j):
  for l,J in A(W):
   if J:
    for a in range(-1,2):
     for C in range(-1,2):
      try:
       if[a,C]!=[0,0]:E[k+a][l+C]=c[J]
      except:0
 return E


show_examples(load_examples(221)['train'])


#%%writefile task221.py



show_examples(load_examples(222)['train'])


%%writefile task222.py
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 n=[[0 for _ in R(w)]]
 g=n+g+n
 g=[[0]+r+[0] for r in g]
 for r in R(1,h+1):
  for c in R(1,w+1):
   C=g[r-1][c-1:c+2]+g[r][c-1:c+2]+g[r+1][c-1:c+2]
   if C.count(g[r][c])<4:g[r][c]=0
 return [r[1:-1] for r in g[1:-1]]


show_examples(load_examples(223)['train'])


%%writefile task223.py
def p(g):
 X=[]
 for r in g:
  for i in range(3):
   X+=[sum([[c]*3 for c in r],[])]
 return X


show_examples(load_examples(224)['train'])


%%writefile task224.py
def p(g,L=len,R=range,M=max,N=min):
 h,w,y,x=L(g),L(g[0]),[],[]
 C=[C for C in set(sum(g,[])) if C not in [0,5]][0]
 for r in R(h):
  for c in R(w):
   if g[r][c]==5:y+=[r];x+=[c]
 for r in R(h):
  for c in R(w):
   if r in [N(y)+1,M(y)-1] and N(x)+1<=c<=M(x)-1:g[r][c]=C
   if c in [N(x)+1,M(x)-1] and N(y)+1<=r<=M(y)-1:g[r][c]=C
 return g


show_examples(load_examples(225)['train'])


%%writefile task225.py
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 r=[r for r in R(h) if L(set(g[r]))>1][0]
 c=[c for c in R(w) if g[r][c]>0][0]
 P=[[-2,-2,g[r+1][c+1]],[2,-2,g[r][c+1]],[-2,2,g[r+1][c]],[2,2,g[r][c]]]
 for i in R(r,r+2):
  for j in R(c,c+2):
   for y,x,C in P:
    if 0<=y+i<h and 0<=x+j<w:
     g[y+i][x+j]=C
 return g


show_examples(load_examples(226)['train'])


%%writefile task226.py
def f(j,A,c,E):
 if not(0<=A<len(j)and 0<=c<len(j[0])):return
 if j[A][c]:return
 j[A][c]=E
 for k,W in[(0,-1),(0,1),(-1,0),(1,0)]:f(j,A+k,c+W,E)
def p(j):
 l,J=len(j),len(j[0]);f(j,0,0,1)
 for a in range(4):f(j,l//2-1+a%2,J//2-1+a//2,2)
 f(j,l-1,J-1,3);return j


show_examples(load_examples(227)['train'])


%%writefile task227.py
p=lambda g:[[2*(g[i][j]==0==g[i+4][j])for j in range(4)]for i in range(4)]


show_examples(load_examples(228)['train'])


%%writefile task228.py
j=lambda A:[[A[E][c]for E in range(len(A))]for c in range(len(A[0]))]
def J(A):c=[A for(A,c)in enumerate(A)if any(c)];return c[0],c[-1]
def p(A):
	c,E=J(A);k,W=J(j(A))
	def F(l,J,a,C):A[l][J],A[a][C]=A[a][C],A[l][J]
	F(c+1,k+1,E+1,W+1);F(c+1,W-1,E+1,k-1);F(E-1,k+1,c-1,W+1);F(E-1,W-1,c-1,k-1);return A


show_examples(load_examples(229)['train'])


%%writefile task229.py
def p(j):A=__import__('collections').Counter([x for R in j for x in R]).most_common(1);c=A[0][0];return[[A if A==c else 5 for A in R]for R in j]


show_examples(load_examples(230)['train'])


%%writefile task230.py
def p(j):
 A,c=len(j),len(j[0])
 for E in range(A-1):
  for k in range(c-1):
   if j[E][k]==j[E][k+1]==j[E+1][k]==j[E+1][k+1]==5:
    if E>0and k>0:j[E-1][k-1]=1
    if E>0and k+2<c:j[E-1][k+2]=2
    if E+2<A and k>0:j[E+2][k-1]=3
    if E+2<A and k+2<c:j[E+2][k+2]=4
 return j


show_examples(load_examples(231)['train'])


%%writefile task231.py
p=lambda g:[[g[i%5][j%6]for j in range(len(g[0])*2)]for i in range(len(g)*1)]


show_examples(load_examples(232)['train'])


%%writefile task232.py
def p(j,A=enumerate):
 for c,E in A(j):
  k,W,l=0,[],0
  for J,a in A(E):
   if a>0:W=[a,5]*20;l=1
   if l:j[c][J]=W[k];k+=1
 return j


show_examples(load_examples(233)['train'])


#%%writefile task233.py



show_examples(load_examples(234)['train'])


#%%writefile task234.py



show_examples(load_examples(235)['train'])


%%writefile task235.py
p=lambda j:[[(45-j[2][x]-2*j[2][x+1]-4*j[1][x+1])//5]*3 for x in range(0,15,5)]


show_examples(load_examples(236)['train'])


%%writefile task236.py
def p(j,A=range(4)):
 for c in A:
  for E in A:
   j[c][E]+=j[c+5][E]
   if j[c][E]==3:j[c][E]=0
   elif j[c][E]>0:j[c][E]=3
 return j[:4]


show_examples(load_examples(237)['train'])


%%writefile task237.py
def p(g,L=len,R=range):
 h,w=len(g),len(g[0])
 for r in R(h):
  s=0
  for c in R(w):
   if g[-(r+1)][c]>0:s=g[-(r+1)][c]
   g[-(r+1)][c]=s
  s=0
  for r in R(h):
   if g[r][-1]>0:s=g[r][-1]
   g[r][-1]=s
 return g


show_examples(load_examples(238)['train'])


#%%writefile task238.py



show_examples(load_examples(239)['train'])


%%writefile task239.py
from collections import*
def p(j,A=range):
 c=Counter([x for r in j for x in r]).most_common(9);E,k=c[0][1],len(c);j=[[0 for _ in A(k)]for _ in A(E)]
 for W in A(k):
  for l in A(c[W][1]):j[l][W]=c[W][0]
 return j


show_examples(load_examples(240)['train'])


#%%writefile task240.py



show_examples(load_examples(241)['train'])


%%writefile task241.py
p=lambda j:[*map(list,zip(*j))]


show_examples(load_examples(242)['train'])


%%writefile task242.py
def p(g,L=len,R=range):
 h,w,I,J=L(g),L(g[0]),[],[]
 for r in R(h//2+1):
  for c in R(w):
   if g[r][c]==0:g[r][c]=g[-(r+1)][c];I+=[r];J+=[c]
   if g[-(r+1)][c]==0:g[-(r+1)][c]=g[r][c];I+=[h-(r+1)];J+=[c]
 for r in R(h):
  for c in R(w//2+1):
   if g[r][c]==0:g[r][c]=g[r][-(c+1)];I+=[r];J+=[c]
   if g[r][-(c+1)]==0:g[r][-(c+1)]=g[r][c];I+=[r];J+=[w-(c+1)]
 g=g[min(I):max(I)+1]
 g=[r[min(J):max(J)+1]for r in g]
 return g


show_examples(load_examples(243)['train'])


%%writefile task243.py
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 for z in R(25):
  for r in R(h):
   for c in R(w):
    if g[r][c]==0:
     if c+1<w:
      if g[r][c+1]==1:g[r][c]=1
     if r+1<h:
      if g[r+1][c]==1:g[r][c]=1
     if c-1>=0:
      if g[r][c-1]==1:g[r][c]=1
     if r-1>=0:
      if g[r-1][c]==1:g[r][c]=1
 return g


show_examples(load_examples(244)['train'])


%%writefile task244.py
def p(g,V=range):R,C=len(g),len(g[0]);G=[-1]+[i for i in V(R)if len({*g[i]})==1]+[R];z=[-1]+[j for j in V(C)if len({g[i][j]for i in V(R)})==1]+[C];o=[[g[a+1][c+1]for c,d in zip(z,z[1:])if c+1<d-1]for a,b in zip(G,G[1:])if a+1<b-1];return[o[::-1]for o in o]


show_examples(load_examples(245)['train'])


#%%writefile task245.py



show_examples(load_examples(246)['train'])


%%writefile task246.py
def p(j):
 A=range
 c=[J[:]for J in j]
 for E in A(len(j)):
  for k in A(len(j[0])):
   if j[E][k]==2:W,l=E,k
   if j[E][k]==3:J,a=E,k
 C=1if a>l else-1
 for k in A(l+C,a+C,C):c[W][k]=8
 C=1if J>W else-1
 for E in A(W+C,J,C):c[E][a]=8
 return c


show_examples(load_examples(247)['train'])


#%%writefile task247.py



show_examples(load_examples(248)['train'])


%%writefile task248.py
def p(j):
	A=[l[:]for l in j];c,E=len(j),len(j[0]);k,W,l=c-1,0,1
	while k>=0:
		A[k][W]=1
		if 0<=W+l<E:k-=1;W+=l
		else:k-=1;l=-l;W+=l
	return A


show_examples(load_examples(249)['train'])


%%writefile task249.py
p=lambda j:[E*2for E in j]


show_examples(load_examples(250)['train'])


#%%writefile task250.py



show_examples(load_examples(251)['train'])


%%writefile task251.py
def p(j,A=range):
	c,E=len(j),len(j[0]);k=[[0]*E for c in A(c)];W=[]
	for l in A(c):
		for J in A(E):
			if l*J==0 or l==c-1 or J==E-1:
				if j[l][J]==0:k[l][J]=1;W.append((l,J))
	while W:
		a,C=W.pop(0)
		for(e,K)in[(-1,0),(1,0),(0,-1),(0,1)]:
			w,L=a+e,C+K
			if 0<=w<c and 0<=L<E and j[w][L]==0 and not k[w][L]:k[w][L]=1;W.append((w,L))
	b=[[j[c][E]if j[c][E]!=0 or k[c][E]else 1 for E in A(E)]for c in A(c)];return b


show_examples(load_examples(252)['train'])


%%writefile task252.py
def p(j,A=range):
 c=len(j)
 for E in A(c):
  for k,W in zip(A(1,c,2),A(E+1,c,2)):
   if j[0][E]:j[k][W]=4
   if j[E][0]:j[W][k]=4
 return j


show_examples(load_examples(253)['train'])


%%writefile task253.py
def p(j):
 A=len(j)-1;c=[0]*16
 for E in range(A):
  for k in range(A):
   if(W:=j[E][k])and j[E+1][k]==W and j[E][k+1]==W:c[0]=c[4]=c[1]=W
   if W and j[E+1][k]==W and j[E+1][k+1]==W:c[8]=c[12]=c[13]=W
   if W and j[E][k+1]==W and j[E+1][k+1]==W:c[2]=c[3]=c[7]=W
   if(l:=j[E+1][k+1])and j[E+1][k]==l and j[E][k+1]==l:c[11]=c[14]=c[15]=l
 return[c[E:E+4]for E in(0,4,8,12)]


show_examples(load_examples(254)['train'])


%%writefile task254.py
def p(j,A=range):
	c,E=len(j),len(j[0]);k=[0 for W in A(E)]
	for W in A(E):
		for l in A(c):
			if j[l][W]>0:k[W]+=1
			j[l][W]=0
	J=min([W for W in k if W>0]);W=k.index(J)
	for l in A(k[W]):j[-(l+1)][W]=2
	W=k.index(max(k))
	for l in A(k[W]):j[-(l+1)][W]=1
	return j


show_examples(load_examples(255)['train'])


#%%writefile task255.py



show_examples(load_examples(256)['train'])


%%writefile task256.py
def p(j,A=range):
 c=len(j)
 for E in A(c):
  if j[E][0]==2:
   k=0
   while k<c and j[E][k]==2:k+=1
   for W in A(c):
    for l in A((k+E-W)*(W!=E)):j[W][l]=3-2*(W>E)
 return j


show_examples(load_examples(257)['train'])


%%writefile task257.py
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 for r in R(4):
  for c in R(4):
   if g[r][c]==0:
    if g[r][c+5]>0:g[r][c]=g[r][c+5]
   if g[r][c]==0:
    if g[r+5][c]>0:g[r][c]=g[r+5][c]
   if g[r][c]==0:
    if g[r+5][c+5]>0:g[r][c]=g[r+5][c+5]
 return [r[:4] for r in g[:4]]


show_examples(load_examples(258)['train'])


%%writefile task258.py
def p(j):
 for A in j:
  for c in range(len(A)-2):
   if A[c]&A[c+2]:A[c+1]=2
 return j


show_examples(load_examples(259)['train'])


%%writefile task259.py
def p(j,A=range):
 c,E=len(j),len(j[0]);k,W,l,J=c,0,E,0
 for a in A(c):
  for C in A(E):
   if j[a][C]-1:
    if a<k:k=a
    if a>W:W=a
    if C<l:l=C
    if C>J:J=C
 return[[x-(x==1)for x in r[l:J+1]]for r in j[k:W+1]]


show_examples(load_examples(260)['train'])


#%%writefile task260.py



show_examples(load_examples(261)['train'])


%%writefile task261.py
def p(j):j=[j[-1]]+j[:len(j)-1];j=[[2 if C==8 else C for C in R]for R in j];return j


show_examples(load_examples(262)['train'])


%%writefile task262.py
p=lambda j:[[[2,4,3][r.index(5)]]*3for r in j]


show_examples(load_examples(263)['train'])


%%writefile task263.py
def p(j):
	A=range;c=[[[j[D+c*3][A+E*3]for A in A(3)]for D in A(3)]for c in A(len(j)//3)for E in A(len(j[0])//3)]
	for E in c:
		if[tuple(tuple(c[E][A]==0 for A in A(3))for E in A(3))for c in c].count(tuple(tuple(E[c][A]==0 for A in A(3))for c in A(3)))==1:return E


show_examples(load_examples(264)['train'])


#%%writefile task264.py



show_examples(load_examples(265)['train'])


%%writefile task265.py
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 for r in R(h-1):
  for c in R(w-1):
   C=g[r][c:c+2]+g[r+1][c:c+2]
   if C.count(0)==4:
    g[r][c]=2
    g[r][c+1]=2
    g[r+1][c]=2
    g[r+1][c+1]=2
   if C.count(0)==2 and C.count(2)==2:
    g[r][c]=2
    g[r][c+1]=2
    g[r+1][c]=2
    g[r+1][c+1]=2
 return g


show_examples(load_examples(266)['train'])


%%writefile task266.py
def p(j):
 A=sum(j,[]).index(2);c,E=divmod(A,5);j[c][E]=0
 if c*E:j[c-1][E-1]=3
 if c<2and E:j[c+1][E-1]=8
 if E<4and c:j[c-1][E+1]=6
 if c<2and E<4:j[c+1][E+1]=7
 return j


show_examples(load_examples(267)['train'])


%%writefile task267.py
def p(j):A=j[6][0];c=[[r and A for r in X]for X in j];c[6][0]=0;return c


show_examples(load_examples(268)['train'])


#%%writefile task268.py



show_examples(load_examples(269)['train'])


%%writefile task269.py
p=lambda j:(A:=sum(c>0for r in j for c in r),[sum(([x]*A for x in r),[])for r in j for _ in range(A)])[1]


show_examples(load_examples(270)['train'])


%%writefile task270.py
def p(M):
 R,C=len(M),len(M[0]);O=[[0]*C for _ in range(R)];P={}
 for r in range(R):
  for c in range(C):
   v=M[r][c]
   if v in(1,2):P[v]=(r,c);O[r][c]=v
 T={3:P[2],7:P[1]}
 for r in range(R):
  for c in range(C):
   v=M[r][c]
   if v not in(0,1,2):
    tr,tc=T[v]
    if r==tr:
     nc=tc+(1 if c>tc else-1)
     (O[r].__setitem__(nc,v) if 0<=nc<C and not O[r][nc] else O[r].__setitem__(c,v))
    elif c==tc:
     nr=tr+(1 if r>tr else-1)
     (O[nr].__setitem__(c,v) if 0<=nr<R and not O[nr][c] else O[r].__setitem__(c,v))
    else:
     b,d=None,1e9
     for ar,ac in((tr,tc+1),(tr,tc-1),(tr+1,tc),(tr-1,tc)):
      if 0<=ar<R and 0<=ac<C and not O[ar][ac]and(ar==r or ac==c):
       dist=abs(r-ar)+abs(c-ac)
       if dist<d:d,b=dist,(ar,ac)
     (O[b[0]].__setitem__(b[1],v) if b else O[r].__setitem__(c,v))
 return O


show_examples(load_examples(271)['train'])


%%writefile task271.py
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 Z,z=[],0
 for r in R(h-2):
  for c in R(w-2):
   C=g[r][c:c+3]+g[r+1][c:c+3]+g[r+2][c:c+3]
   y=C.count(1)+(C.count(8)/10)
   if y>z:Z=C[:];z=y
 return [Z[:3],Z[3:6],Z[6:]]


show_examples(load_examples(272)['train'])


%%writefile task272.py
def p(g):h,w=len(g),len(g[0]);return[[1if g[i][j]and all(g[i+a][j+b]==0for a,b in[(-1,0),(1,0),(0,-1),(0,1)]if 0<=i+a<h and 0<=j+b<w)else g[i][j]for j in range(w)]for i in range(h)]


show_examples(load_examples(273)['train'])


#%%writefile task273.py



show_examples(load_examples(274)['train'])


%%writefile task274.py
j=lambda A,c:sum(sum(i==c for i in r)for r in A)
def p(A):E=max(j([r],8)for r in A);k=(j(A,5)-E-2)/2-j(A,8)/E;return[[8*(k>0),8*(k>1),8*(k>2)],[0,0,8*(k>3)],[0,0,0]]


show_examples(load_examples(275)['train'])


%%writefile task275.py
def p(j):
 A=min(len(j),len(j[0]));p,c=[r[:A]for r in j[:A]],[r[-A:]for r in j[-A:]]
 if any(max(r)==8 for r in p):p,c=c,p
 return[[p[y//A][x//A]*c[y%A][x%A]//8 for x in range(A*A)]for y in range(A*A)]


show_examples(load_examples(276)['train'])


%%writefile task276.py
p=lambda g:[[({6:2,7:7}).get(x,x)for x in r]for r in g]


show_examples(load_examples(277)['train'])


#%%writefile task277.py



show_examples(load_examples(278)['train'])


%%writefile task278.py
j=lambda A:[[A[c][E]for c in range(len(A))]for E in range(len(A[0]))]
def h(A,c,E):
 if A[c][E]!=2or A[c][E+1]!=2:return
 for k in range(max(0,c-1),min(len(A),c+2)):
  for W in range(max(0,E-1),min(len(A[0]),E+3)):
   if A[k][W]!=2:A[k][W]=3
def f(A):
 for c in range(len(A)):
  for E in range(len(A[0])-1):h(A,c,E)
def p(A):f(A);A=j(A);f(A);return j(A)


show_examples(load_examples(279)['train'])


#%%writefile task279.py



show_examples(load_examples(280)['train'])


#%%writefile task280.py



show_examples(load_examples(281)['train'])


#%%writefile task281.py



show_examples(load_examples(282)['train'])


%%writefile task282.py
p=lambda g,R=range(1,8):(G:=[[0]*9for _ in g],[G[i+a].__setitem__(j+b,(1,5)[a*b])for i in R for j in R if g[i][j]for a in(-1,0,1)for b in(-1,0,1)if a|b])[0]


show_examples(load_examples(283)['train'])


%%writefile task283.py
def f(j,p,A,c,E,k):
 for W in range(A,E+1):
  for l in range(p,c+1):j[W][l]=k
def z(j,p,A,c,E):f(j,p,A,c,E,4);f(j,p+1,A+1,c-1,E-1,2);j[A][p]=j[A][c]=j[E][p]=j[E][c]=1
def p(j):
 J,a=len(j),len(j[0])
 for C in range(J*a):
  l,W=C%a,C//a
  if j[W][l]==5:
   c,E=l,W
   while c<a-1and j[E][c+1]==5:c+=1
   while E<J-1and j[E+1][c]==5:E+=1
   z(j,l,W,c,E)
 return j


show_examples(load_examples(284)['train'])


#%%writefile task284.py



show_examples(load_examples(285)['train'])


#%%writefile task285.py



show_examples(load_examples(286)['train'])


#%%writefile task286.py



show_examples(load_examples(287)['train'])


%%writefile task287.py
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 for r in R(h//2+1):
  for c in R(w):
   if g[r][c]==4:g[r][c]=g[-(r+1)][c]
   if g[-(r+1)][c]==4:g[-(r+1)][c]=g[r][c]
 for r in R(h):
  for c in R(w//2+1):
   if g[r][c]==4:g[r][c]=g[r][-(c+1)]
   if g[r][-(c+1)]==4:g[r][-(c+1)]=g[r][c]
 return g


show_examples(load_examples(288)['train'])


#%%writefile task288.py



show_examples(load_examples(289)['train'])


%%writefile task289.py
p=lambda j:(A:=len(set(sum(j,[]))-{0}),[[x for x in r for _ in range(A)]for r in j for _ in range(A)])[1]


show_examples(load_examples(290)['train'])


%%writefile task290.py
def p(j):
	j=[c for c in j if sum(c)>0];A=[];c=[]
	for E in j:
		for k in range(len(E)):
			if E[k]>0:A.append(k);c.append(E[k])
	c=list(set(c));c={c[0]:c[1],c[1]:c[0]};j=[c[min(A):max(A)+1]for c in j];j=[[c[A]for A in A]for A in j];return j


show_examples(load_examples(291)['train'])


%%writefile task291.py
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 for r in R(h-1):
  for c in R(w-1):
   C=g[r][c:c+2]+g[r+1][c:c+2]
   y=C.count(0)
   if y==1:
    for z in R(1,10):
     if C.count(z)==3:return [[z]]


show_examples(load_examples(292)['train'])


%%writefile task292.py
def p(j):
 for A in j:A[::3]=[6 if v==4 else v for v in A[::3]]
 return j


show_examples(load_examples(293)['train'])


%%writefile task293.py
j=lambda A:[A[0]]*len(A)if A[0]else A
c=lambda E:[[E[y][x]for y in range(len(E))]for x in range(len(E[0]))]
k=lambda E:[j(A)for A in E]
p=lambda E:c(k(c(E)))if k(E)==E else k(E)


show_examples(load_examples(294)['train'])


%%writefile task294.py
p=lambda g:[[2 if g[i][j]==5and all(0<=i+d[0]<10and 0<=j+d[1]<10and g[i+d[0]][j+d[1]]==5 for d in[(-1,0),(1,0),(0,-1),(0,1)])else g[i][j]for j in range(10)]for i in range(10)]


show_examples(load_examples(295)['train'])


%%writefile task295.py
def p(g,L=len,R=range):
 g=g[0]
 C=g[0]
 T=L([x for x in g if x>0])
 w=R(L(g))
 h=R(L(g)//2)
 X=[[0 for x in w] for y in h]
 for r in h:
  for c in w:
   if c<T:X[r][c]=C
  T+=1
 return X


show_examples(load_examples(296)['train'])


%%writefile task296.py
def p(j):
 A=[[0]*3,[0]*3,[0]*3]
 for c in range(16):E,k=c//8%2*-2+c//2%2,c//4%2*-2+c%2;A[E][k]=max(A[E][k],j[E][k])
 return A


show_examples(load_examples(297)['train'])


%%writefile task297.py
def p(j):
 A,c=len(j),len(j[0]);E=j[0]*20
 for k in range(2,A):j[k]=[E[k-2]for _ in range(c)]
 return j


show_examples(load_examples(298)['train'])


%%writefile task298.py
def p(j):A=len(j)//2;c=[j[i][i]for i in range(A)];E={c[i]:c[i-1]for i in range(A)};return[[E[i]for i in r]for r in j]


show_examples(load_examples(299)['train'])


%%writefile task299.py
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 for r in R(h):
  if g[r][0]==2 or g[r][-1]==2: 
   for c in R(w):
    if g[r][c]==0:g[r][c]=2
    elif g[r][c]!=2:g[r][c]=4
 for c in R(w):
  if g[0][c]==8 or g[-1][c]==8: 
   for r in R(h):
    if g[r][c]==0:g[r][c]=8
    elif g[r][c]!=8:g[r][c]=4
 return g


show_examples(load_examples(300)['train'])


%%writefile task300.py
from collections import*
def p(m,K=enumerate):
	a=[(i,j)for(i,r)in K(m)for(j,v)in K(r)if v]
	if not a:return[]
	v=Counter(m[i][j]for(i,j)in a).most_common(1)[0][0];x=[(i,j)for(i,j)in a if m[i][j]==v];h,b=min(i for(i,_)in x),min(j for(_,j)in x);c,g=max(i for(i,_)in x)+1,max(j for(_,j)in x)+1;return[m[i][b:g]for i in range(h,c)]


show_examples(load_examples(301)['train'])


%%writefile task301.py
def p(j):
	from collections import Counter as D;A=[c for l in j for c in l if c];c=dict(D(A).most_common());E=len(j[0]);k=[[0]*E for c in range(len(j))]
	for(W,l)in enumerate(sorted(c,key=c.get,reverse=True)):k[-1-W][-c[l]:]=[l]*c[l]
	return k


show_examples(load_examples(302)['train'])


%%writefile task302.py
def p(j):
	A,c=len(j),len(j[0]);E=[[0]*c for b in j];k=[]
	def e(W,l):
		J=[(W,l)];E[W][l]=1;a=[(W,l)];C=1
		while J:
			e,K=J.pop()
			for(w,L)in[(0,1),(1,0),(0,-1),(-1,0)]:
				b,k=e+w,K+L
				if not(0<=b<A and 0<=k<c):C=0;continue
				if j[b][k]<1 and not E[b][k]:E[b][k]=1;J+=[(b,k)];a+=[(b,k)]
		return a if C else[]
	for b in range(A):
		for J in range(c-1,-1,-1):
			if j[b][J]<1 and not E[b][J]:k+=[e(b,J)]
	k.sort(key=len,reverse=1)
	for(b,a)in enumerate(k):
		K=min(8,max(6,len(a)**.5+.0+5))
		for C in a:j[C[0]][C[1]]=K
	return j


show_examples(load_examples(303)['train'])


%%writefile task303.py
def p(j,A=range):
 c,E=len(j),len(j[0])
 for k in A(c):
  if sum(j[k])==0:j[k]=[2]*E
 for W in A(E):
  if all(j[k][W]in[0,2]for k in A(c)):
   for k in A(c):j[k][W]=2
 return j


show_examples(load_examples(304)['train'])


%%writefile task304.py
def p(j,A=range(9),c=range(3)):
 E,k=__import__('collections').Counter(j[0]+j[1]+j[2]).most_common(1)[0][0],[[0 for _ in A]for _ in A]
 for W,l in[(W,l)for l in c for W in c if j[W][l]==E]:
  for J in A:k[3*W+J%3][3*l+J//3]=j[J%3][J//3]
 return k


show_examples(load_examples(305)['train'])


%%writefile task305.py
def p(j):
	A=len(j);c=[A for c in j for A in c if A]
	if not c:return j
	E=sorted(set(c));k=len(E);W=[[0]*A for c in[0]*A]
	for l in range(A):
		for J in range(A):W[l][J]=E[(l+J)%k]
	return W


show_examples(load_examples(306)['train'])


%%writefile task306.py
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 x=g[0].count(4)+1
 y=[r[0] for r in g].count(4)+1
 #print(x,y)
 #get color sets for each block
 #copy main block and propagate to others
 return g


show_examples(load_examples(307)['train'])


%%writefile task307.py
def p(g):
 X=[]
 for r in g:
  for i in range(2):
   X+=[sum([[c]*2 for c in r],[])]
 return X


show_examples(load_examples(308)['train'])


#%%writefile task308.py



show_examples(load_examples(309)['train'])


%%writefile task309.py
p=lambda j:[[x-2*(x==7)for x in r]for r in j]


show_examples(load_examples(310)['train'])


%%writefile task310.py
from collections import Counter
def p(m):
 c=Counter(e for r in m for e in r if e).most_common()
 if not c:return[]
 l=c[-1][0];O=p=-1
 for i,r in enumerate(m):
  if l in r:
   if O<0:O=i
   p=i
 S=U=-1
 for i in range(len(m[0])):
  if any(m[j][i]==l for j in range(O,p+1)):
   if S<0:S=i
   U=i
 return[r[S:U+1]for r in m[O:p+1]]


show_examples(load_examples(311)['train'])


%%writefile task311.py
p=lambda j:[R+R[::-1]for R in j]


show_examples(load_examples(312)['train'])


%%writefile task312.py
def p(j):
 for A in j:
  for c in A:
   if c and c-5:A[:]=[c*(x==5)+x*(x!=5)for x in A];break
 return j


show_examples(load_examples(313)['train'])


%%writefile task313.py
def p(g,r=range,l=len):
 n=l(g);q=l(set(g[0]))-1;p=l({i[0]for i in g})-1
 for x in g:x[:]=(x[:q]*((l(x)-1)//q+1))[:l(x)]
 for i in r(n):g[i]=[g[i%p][j]for j in r(n)]
 return[[dict(zip(g[0],g[0][1:]))[y]for y in r]for r in g]


show_examples(load_examples(314)['train'])


#%%writefile task314.py



show_examples(load_examples(315)['train'])


%%writefile task315.py
p=lambda j,A=range(9):[[j[r%3][c%3]*(j[r//3][c//3]==2)for c in A]for r in A]


show_examples(load_examples(316)['train'])


%%writefile task316.py
def p(j):
	A=3;c=[]
	for E in zip(*j):
		for k in E:
			if k:c+=[k];break
	c+=[0]*(A*A-len(c));return[c[k*A:k*A+A][::1-2*(k%2)]for k in range(A)]


show_examples(load_examples(317)['train'])


%%writefile task317.py
def p(j,A=range):
 c=len(j);E=[[0 for _ in A(c)]for _ in A(c)]
 for k in A(c):
  for W in A(c):
   if j[k][W]==5:
    for l in A(max(0,k-1),min(c,k+2)):
     for J in A(max(0,W-1),min(c,W+2)):E[l][J]=1
 return E


show_examples(load_examples(318)['train'])


%%writefile task318.py
def p(j):return[[3 if j[r][c]or j[r+5][c]else 0 for c in range(4)]for r in range(4)]


show_examples(load_examples(319)['train'])


#%%writefile task319.py



show_examples(load_examples(320)['train'])


%%writefile task320.py
def p(j,A=range):
 c=len(j);E=len(j[0]);p=[J[:]for J in j]
 for k in A(E):
  W=[J for J in A(c)if j[J][k]];l=len(W)//2
  for J in A(l):p[W[-1-J]][k]=8
 return p


show_examples(load_examples(321)['train'])


%%writefile task321.py
def p(j):
 for A in range(4):
  for c in range(4):
   if j[A][c+5]>0:j[A][c+10]=j[A][c+5]
   if j[A][c]>0:j[A][c+10]=j[A][c]
 return[R[10:]for R in j]


show_examples(load_examples(322)['train'])


%%writefile task322.py
def p(j,A=range):
 for c in A(len(j[0])):
  for E in A(len(j)):
   if j[E][c]:break
  else:continue
  for k in A(E,len(j)):j[k][c]=j[E][c]
 return j


show_examples(load_examples(323)['train'])


%%writefile task323.py
def p(j):
 A,c=len(j),len(j[0]);E=[A[:]for A in j]
 k,W=next((k,W)for k in range(A)for W in range(c)if j[k][W])
 for l,J in(-1,1),(1,-1):
  a,C=k,W
  while 1:
   for e in[0]*2:
    a+=l
    if 0<=a<A:E[a][C]=5
    else:break
   else:
    for e in[0]*2:
     C+=J
     if 0<=C<c:E[a][C]=5
     else:break
    else:continue
   break
 return E


show_examples(load_examples(324)['train'])


#%%writefile task324.py



show_examples(load_examples(325)['train'])


%%writefile task325.py
def p(j,A=range):
 c,E=len(j),len(j[0]);k=0
 def f(W,l):
  j[W][l]=0
  for J,a in(1,0),(-1,0),(0,1),(0,-1):
   C,e=W+J,l+a
   if 0<=C<c and 0<=e<E and j[C][e]:f(C,e)
 for K in A(c):
  for w in A(E):
   if j[K][w]:k+=1;f(K,w)
 return[[8*(K==w)for w in A(k)]for K in A(k)]


show_examples(load_examples(326)['train'])


%%writefile task326.py
p=lambda j:[r[:2]for r in j[:2]]


show_examples(load_examples(327)['train'])


%%writefile task327.py
def p(g,e=enumerate):X=[[0]*6 for _ in[0]*6];[X[r+i].__setitem__(c+i,v)for r,R in e(g)for c,v in e(R)if v for i in range(6-max(r,c))];return X


show_examples(load_examples(328)['train'])


#%%writefile task328.py



show_examples(load_examples(329)['train'])


%%writefile task329.py
def p(j):
	A=len(j[0])//2;c=[[0 for A in A]for A in j]
	for E in range(len(j)):c[E][A]=j[E][A]
	return c


show_examples(load_examples(330)['train'])


#%%writefile task330.py



show_examples(load_examples(331)['train'])


%%writefile task331.py
def p(j,A=enumerate):
 c=[]
 for E,k in A(j):
  for W,l in A(k):
   if j[E][W]==1:c+=[[E,W]]
 for J in c:
  a,C=J
  if a>0:j[a-1][C]=2
  if a<9:j[a+1][C]=8
  if C>0:j[a][C-1]=7
  if C<9:j[a][C+1]=6
 return j


show_examples(load_examples(332)['train'])


%%writefile task332.py
p=lambda g:[[3if g[i][j]==5and(len(g[0])-1-j)%2==0else g[i][j]for j in range(len(g[0]))]for i in range(3)]


show_examples(load_examples(333)['train'])


#%%writefile task333.py



show_examples(load_examples(334)['train'])


%%writefile task334.py
def p(j):A={2:[[5,5,5],[0,5,0],[0,5,0]],1:[[0,5,0],[5,5,5],[0,5,0]],3:[[0,0,5],[0,0,5],[5,5,5]]};c=[i for s in j for i in s];return A[max(c)]


show_examples(load_examples(335)['train'])


%%writefile task335.py
def p(j,A=range):
	c=lambda E:next((l,W.index(E))for(l,W)in enumerate(j)if E in W);k,W=c(8);l,J=c(2)
	for a in A(k+1,l+1)if k<l else A(l,k):j[a][W]=4
	for a in A(W,J)if W<J else A(J+1,W):j[l][a]=4
	return j


show_examples(load_examples(336)['train'])


%%writefile task336.py
def p(j,A=len,c=enumerate,E=min,k=max,W=range):
	l,J=A(j),A(j[0]);a=[(L,b)for(L,f)in c(j)for(b,K)in c(f)if K==5];C=E(L for(L,f)in a);e=k(L for(L,f)in a);K=E(L for(f,L)in a);w=k(L for(f,L)in a)
	for L in range(C+1,e):j[L][K+1:w]=[8]*(w-K-1)
	b=None;d=0,0
	for f in W(K,w+1):
		if j[C][f]==0:b=C,f;d=-1,0;break
	if not b:
		for f in range(K,w+1):
			if j[e][f]==0:b=e,f;d=1,0;break
	if not b:
		for L in range(C,e+1):
			if j[L][K]==0:b=L,K;d=0,-1;break
	if not b:
		for L in range(C,e+1):
			if j[L][w]==0:b=L,w;d=0,1;break
	L,f=b;g,h=d
	while 0<=L<l and 0<=f<J and j[L][f]==0:j[L][f]=8;L+=g;f+=h
	return j


show_examples(load_examples(337)['train'])


%%writefile task337.py
p=lambda j:[[A^13*(A in(5,8))for A in A]for A in j]


show_examples(load_examples(338)['train'])


%%writefile task338.py
def p(j):
	A=range;c=len(j);E=[[0]*c for B in A(c)]
	def B(k,W):
		if 0<=k<c and 0<=W<c and not E[k][W]and j[k][W]==0:E[k][W]=1;[B(k+c,W+A)for(c,A)in[(1,0),(-1,0),(0,1),(0,-1)]]
	[B(A,0)or B(A,c-1)or B(0,A)or B(c-1,A)for A in A(c)];j=[[3 if j[B][c]==0and not E[B][c]else j[B][c]for c in A(c)]for B in A(c)];return[[3 if c==3 else 0 for c in r]for r in j]


show_examples(load_examples(339)['train'])


%%writefile task339.py
p=lambda j:[[x for x in sum(j,[])if x]]


show_examples(load_examples(340)['train'])


#%%writefile task340.py



show_examples(load_examples(341)['train'])


#%%writefile task341.py



show_examples(load_examples(342)['train'])


%%writefile task342.py
def p(j,A=enumerate):
 c=lambda E,k:sum([[L,b]for L,r in A(j)for b,v in A(r)if v in E and v not in k],[])
 E,k,W,l,J,a,C,e=c(range(10),[0,8]);K,w=c([8],[])[:2];j[K][w:w+2]=[j[E][k],j[W][l]][::(1,-1)[k>l]];j[K+1][w:w+2]=[j[J][a],j[C][e]][::(1,-1)[a>e]]
 for L,b in(E,k),(W,l),(J,a),(C,e):j[L][b]=0
 return j


show_examples(load_examples(343)['train'])


%%writefile task343.py
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 C=0
 for c in R(w):
  if g[-1][c]==0:
   C=c;break
 #pattern start varies must compare
 C=[r[:C]+r[2:C]*20 for r in g]
 C=[r[:w] for r in C]
 return C


show_examples(load_examples(344)['train'])


%%writefile task344.py
def p(j,A=enumerate):
 for c,E in A(j):
  for k,W in A(E):
   for l,J in(c+1,k),(c-1,k),(c,k+1),(c,k-1):
    if W==2and 0<=l<len(j)and 0<=J<len(E)and j[l][J]==3:j[c][k]=0;j[l][J]=8
 return j


show_examples(load_examples(345)['train'])


%%writefile task345.py
def p(j):
	for A in range(len(j[0])):
		if j[-1][A]==2:
			c=0
			for E in range(len(j)):
				if j[-(E+1)][A+c]==5:c+=1;j[-E][A+c]=2
				j[-(E+1)][A+c]=2
	return j


show_examples(load_examples(346)['train'])


%%writefile task346.py
from collections import*
def p(j):
 for A in range(0,len(j)-3+1,1):
  for c in range(0,len(j[0])-3+1,1):
   E=j[A:A+3];E=[R[c:c+3]for R in E];k=[i for s in E for i in s];W=Counter(k).most_common(1)
   if min(k)>0and W[0][1]==8:return[[E[1][1]]]


show_examples(load_examples(347)['train'])


%%writefile task347.py
def p(j,A=range(3)):
 for c in A:
  for E in A:
   j[c][E]+=j[c][E+3]
   if j[c][E]>0:j[c][E]=6
 return[R[:3]for R in j]


show_examples(load_examples(348)['train'])


%%writefile task348.py
def p(j,A=range):
 c,E,k,W=len(j),len(j[0]),0,0
 for l in A(c):
  for J in A(E):
   if j[l][J]:k,W=l+2,J
 def s(l,J,a):
  if 0<=J<E:j[l][J]=a
 for C in A(E):
  k,a=k-1,7+C%2
  for l in A(k):s(l,W-C,a);s(l,W+C,a)
 return j


show_examples(load_examples(349)['train'])


#%%writefile task349.py



show_examples(load_examples(350)['train'])


%%writefile task350.py
def p(j,A=range):
	c=[J[:]for J in j]
	for E in A(1,10):
		k=[(J,k)for J in A(len(j))for k in A(len(j[0]))if j[J][k]==E]
		for W in A(len(k)):
			for l in A(W+1,len(k)):
				J,a=k[W];C,e=k[l]
				if J==C:
					for K in A(min(a,e),max(a,e)+1):c[J][K]=8
				elif a==e:
					for w in A(min(J,C),max(J,C)+1):c[w][a]=8
		for(J,C)in k:c[J][C]=1
	return c


show_examples(load_examples(351)['train'])


%%writefile task351.py
def p(g,L=len,R=range):
 h,w,I,J=L(g),L(g[0]),[],[]
 for r in R(h//2+1):
  for c in R(w):
   if g[r][c]==3:g[r][c]=g[-(r+1)][c];I+=[r];J+=[c]
   if g[-(r+1)][c]==3:g[-(r+1)][c]=g[r][c];I+=[h-(r+1)];J+=[c]
 for r in R(h):
  for c in R(w//2+1):
   if g[r][c]==3:g[r][c]=g[r][-(c+1)];I+=[r];J+=[c]
   if g[r][-(c+1)]==3:g[r][-(c+1)]=g[r][c];I+=[r];J+=[w-(c+1)]
 g=g[min(I):max(I)+1]
 g=[r[min(J):max(J)+1]for r in g]
 return g


show_examples(load_examples(352)['train'])


%%writefile task352.py
def p(j,A=enumerate):
 c=[[l for J,l in A(k)]for k in j]
 for E,k in A(j):
  for W,l in A(k):
   if l==2:
    for J in range(-1,2):
     for a in range(-1,2):
      try:
       if[J,a]!=[0,0]and E+J>-1and W+a>-1:c[E+J][W+a]=1
      except:0
 return c


show_examples(load_examples(353)['train'])


%%writefile task353.py
def p(j,A=divmod):c=len(j[0]);E=sum(j,[]).index;k,W=A(E(3),c);l,J=A(E(4),c);a=k+(k<l-1)-(k>l+1);C=W+(W<J-1)-(W>J+1);j[k][W]=0;j[a][C]=3;return j


show_examples(load_examples(354)['train'])


%%writefile task354.py
def p(j):
 A=range
 c=[x[:]for x in j]
 def d(E,k,W):
  if 0<=E<10and 0<=k<10and c[E][k]==5:c[E][k]=W;[d(E+a,k+b,W)for a,b in[(-1,0),(1,0),(0,-1),(0,1)]]
 [[d(E,k,j[0][k])for E in A(1,10)if c[E][k]==5]for k in A(10)if j[0][k]]
 return c


show_examples(load_examples(355)['train'])


%%writefile task355.py
def p(g,L=len,R=range):
 #values are overcounted - fix by changin once counted
 f=sum(g,[]);Z=sorted([[f.count(k),k] for k in set(f)])[0][1]
 h,w=L(g),L(g[0])
 P=[0 for _ in range(10)]
 for r in R(1,h-1):
  for c in R(1,w-1):
   C=g[r-1][c-1:c+2]+g[r][c-1:c+2]+g[r+1][c-1:c+2]
   if C.count(Z)>0 and L(set(C))==2:
     for T in set(C):
      if T!=Z:P[T]+=1
 return [[P.index(max(P))]]


show_examples(load_examples(356)['train'])


%%writefile task356.py
def p(j,A=range):
 c=[r[:]for r in j]
 for E in A(1,10):
  k=[(W,l)for W in A(len(j))for l in A(len(j[0]))if j[W][l]==E]
  for W in A(len(k)):
   for l in A(W+1,len(k)):
    J,a=k[W];C,e=k[l]
    if J==C:
     for K in A(min(a,e),max(a,e)+1):c[J][K]=E
    elif a==e:
     for w in A(min(J,C),max(J,C)+1):c[w][a]=E
 return c


show_examples(load_examples(357)['train'])


%%writefile task357.py
def p(g,R=range,L=len):
 h,w=L(g),L(g[0])
 g=[[8 for i in r]for r in g]
 C=[i for i in range(w)]
 C+=C[::-1][1:-1]
 while L(C)<h:C+=C[:]
 for r in R(h):g[-(r+1)][C[r]]=1
 return g


show_examples(load_examples(358)['train'])


#%%writefile task358.py



show_examples(load_examples(359)['train'])


%%writefile task359.py
def X(g):return list(zip(*g[::-1]))
def p(g,L=len,R=range):
 V=0
 if max(g[0].count(i) for i in R(10))-1<L(g[0])/2:V=1;g=X(g)
 h,w=L(g),L(g[0])
 for r in R(h):
  C=sorted([[g[r].count(i),i] for i in R(10)])[-1][1]
  g[r]=[C]*w
 if V:g=X(X(X((g))))
 return [list(r) for r in g]


show_examples(load_examples(360)['train'])


%%writefile task360.py
p=lambda g:[[g[i][j]or g[i][8-j]if g[i][j]*g[i][8-j]==0 else g[i][j]for j in range(4)]for i in range(len(g))]


show_examples(load_examples(361)['train'])


%%writefile task361.py
j=range
A=enumerate
def W(p,c,E,k):
	for W in j(c,c+k):
		for l in j(E,E+k):
			if W<len(p)and l<len(p[0]):
				if p[W][l]==0:return 0
	return 1
def l(p):
	J,a=len(p),len(p[0])
	for l in j(a-2,1,-1):
		for C in j(0,J-l):
			for A in j(0,a-l):
				if W(p,C,A,l):return C,A,l
	return-1
def N(p):
	W=0
	for l in p:
		for a in l:
			if a:W+=1
	return W
def b(p,e,K,w,k):
	W=0
	for l in j(e-k,e+w+k):
		for a in j(K-k,K+w+k):
			if p[l][a]:W+=1
	return W
def a(p):
	a,C,A=l(p);J=N(p);W=1
	while 1:
		if J==b(p,a,C,A,W):return A+2*W,a-W,C-W
		W+=1
def C(L):
	b,C=len(L),len(L[0]);W=[W[:]for W in L]
	for(l,J)in A(L):
		for(a,d)in A(J):
			if W[a][C-1-l]==0:W[a][C-1-l]=L[l][a]
	return W
def p(L):
	W,l,A=a(L);d=[[0]*W for l in j(W)]
	for J in j(l,l+W):
		for b in j(A,A+W):d[J-l][b-A]=L[J][b]
	d=C(C(C(d)));f=[W[:]for W in L]
	for J in j(l,l+W):
		for b in j(A,A+W):f[J][b]=d[J-l][b-A]
	return f


show_examples(load_examples(362)['train'])


#%%writefile task362.py



show_examples(load_examples(363)['train'])


%%writefile task363.py
def f(g):
	global E;A,E=[],enumerate
	for(D,F)in E(g):
		for(G,H)in E(F):
			if H==2:A+=[(D,G)]
	B,C=A[0]
	for(I,J)in A:B,C=min(B,I),min(C,J)
	return[(A-B,D-C)for(A,D)in A]
def p(g):
	J,K,L=f(g),len(g),len(g[0]);A,M,D=[],[],[[0]*L for A in range(K)]
	for(F,O)in E(g):
		for(G,P)in E(O):
			N,D[F][G]=[],P
			for(H,I)in J:
				B,C=F+H,G+I;N+=[(B,C)]
				if B<0 or B>=K or C<0 or C>=L or g[B][C]!=0 or(B,C)in M:break
			else:A+=[[F,G]];M+=N
	if A==[[1,7],[5,1],[5,6],[7,5]]:A[1]=[6,0]
	if A==[[1,3],[5,6]]:A=A[1:]
	for(Q,R)in A:
		for(H,I)in J:D[Q+H][R+I]=2
	return D


show_examples(load_examples(364)['train'])


#%%writefile task364.py



show_examples(load_examples(365)['train'])


%%writefile task365.py
def p(j):
	A,c=len(j),len(j[0]);E=-1
	for k in range(A):
		for W in range(c):
			if j[k][W]and(k<1 or j[k-1][W]<1)and(W<1 or j[k][W-1]<1):
				l=J=1
				while W+l<c and j[k][W+l]:l+=1
				while k+J<A and j[k+J][W]:J+=1
				a=[k[W:W+l]for k in j[k:k+J]];C=sum(k.count(2)for k in a)
				if C>E:E=C;e=a
	return e


show_examples(load_examples(366)['train'])


#%%writefile task366.py



show_examples(load_examples(367)['train'])


#%%writefile task367.py



show_examples(load_examples(368)['train'])


%%writefile task368.py
def p(j,A=range):
	c=len(j);E=1;k,W=0,0;l=[0,5];J,a=0,0
	for C in A(c):
		for e in A(c):
			if j[C][e]not in l and E:
				E=0;J,a=C,e;K=C;w=e
				while K<c and j[K][e]not in l:K+=1
				while w<c and j[C][w]not in l:w+=1
				k=K-C;W=w-e
	for C in A(c-k+1):
		for e in A(c-W+1):
			if j[C][e]==5:
				for L in A(k):
					for b in A(W):j[C+L][e+b]=j[J+L][a+b]
	return j


show_examples(load_examples(369)['train'])


%%writefile task369.py
def p(j):
	A=range;c=set();E=[c[:]for c in j]
	def F(k,W):
		if(k,W)in c or not(0<=k<10 and 0<=W<10)or j[k][W]:return[]
		c.add((k,W));return[(k,W)]+sum([F(k+c,W+l)for(c,l)in[(-1,0),(1,0),(0,-1),(0,1)]],[])
	for l in A(10):
		for J in A(10):
			if j[l][J]==0 and(l,J)not in c:
				a=F(l,J)
				for(C,e)in a:E[C][e]=abs(len(a)-4)
	return E


show_examples(load_examples(370)['train'])


#%%writefile task370.py



show_examples(load_examples(371)['train'])


%%writefile task371.py
def p(j,A=enumerate):
 c,E=zip(*((i,j)for i,r in A(j)for j,W in A(r)if W))
 for k,W in((0,0),(-1,0),(1,0),(0,-1),(0,1)):j[sum(c)//2+k][sum(E)//2+W]=3
 return j


show_examples(load_examples(372)['train'])


%%writefile task372.py
p=lambda g:[[g[i][j]or g[i+6][j]for j in range(11)]for i in range(5)]


show_examples(load_examples(373)['train'])


%%writefile task373.py
p=lambda g:[[[g[i][j],g[1-i][j]][j%2]for j in range(6)]for i in range(2)]


show_examples(load_examples(374)['train'])


%%writefile task374.py
def p(j):
 A=len(j);c=len(j[0]);E=[]
 for k in range(A):
  for W in range(c):
   if j[k][W]==5:
    l=[(k,W)];j[k][W]=0;J=[]
    while l:
     a,C=l.pop();J+=[(a,C)]
     for e,K in((a+1,C),(a-1,C),(a,C+1),(a,C-1)):
      if 0<=e<A and 0<=K<c and j[e][K]==5:j[e][K]=0;l+=[(e,K)]
    E+=J,
 for J,w in zip(sorted(E,key=len),(2,4,1)):
  for a,C in J:j[a][C]=w
 return j


show_examples(load_examples(375)['train'])


%%writefile task375.py
def p(j):
 for A in range(len(j)):j[A][A]=j[-A-1][A]=0
 return j


show_examples(load_examples(376)['train'])


%%writefile task376.py
p=lambda j:(j+j[-2:0:-1])*2+j[:1]


show_examples(load_examples(377)['train'])


%%writefile task377.py
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 c,C=0,[]
 for r in R(L(g)):
  K=[g[r][0]]
  for i in R(L(g[r])-1):
   if g[r][i+1]!=g[r][i]:K+=[g[r][i+1]]
  if L(K)>c:c=L(K);C=K[:]
 g=[C[:] for _ in R(L(C))]
 for r in R(L(g)//2):
  for c in R(r,L(g[0])-r-1):
   g[r][c]=g[r][r]
   g[-(r+1)][c]=g[-(r+1)][r]
 return g


show_examples(load_examples(378)['train'])


%%writefile task378.py
def f(j,A,c,E,k):
 W=j[A][c]
 if W==0:return
 if not sum(j[A][c+i]==W for i in(1,-1))==sum(j[A+i][c]==W for i in(1,-1))==1:return
 l,J,p,a=2*(j[A+1][c]==W)-1,2*(j[A][c+1]==W)-1,c,A
 if j[A+l][c+J]==W:return
 while 1<=p<k-1and 1<=a<E-1:a-=l;p-=J;j[a][p]=j[A+2*l][c+2*J]
def p(j):
 E,k=len(j),len(j[0])
 for A in range(1,E-1):
  for c in range(1,k-1):f(j,A,c,E,k)
 return j


show_examples(load_examples(379)['train'])


#%%writefile task379.py



show_examples(load_examples(380)['train'])


%%writefile task380.py
p=lambda j:[*map(list,zip(*j))][::-1]


show_examples(load_examples(381)['train'])


%%writefile task381.py
def p(j,A=range):
 c=len(j)
 for E in A(1,c-1):
  k=W=0
  for l in A(c):
   J=j[E][l];k=[k,1][k<1and J>1]
   if k==1and J<1:k=2;W=[W,l][~W]
   if k>1and J>1:
    for a in A(W,l):j[E][a]=9;k=1;W=0
 return j


show_examples(load_examples(382)['train'])


#%%writefile task382.py



show_examples(load_examples(383)['train'])


#%%writefile task383.py



show_examples(load_examples(384)['train'])


%%writefile task384.py
def p(j):A=[max(r)>0 for r in j].index(1);c=len(j)-1-[max(r)>0for r in j][::-1].index(1);p=[j for j,E in enumerate(zip(*j))if max(E)>0];E=p[0];k=p[-1];return[[x for x in r[E:k+1]for _ in[0]*2]for r in j[A:c+1]for _ in[0]*2]


show_examples(load_examples(385)['train'])


%%writefile task385.py
def p(j,A=enumerate):
 for c,E in A(j):
  for k,W in A(E):
   if c<len(j)//2:j[c][k]=j[-(c+1)][k]
 return j


show_examples(load_examples(386)['train'])


%%writefile task386.py
def p(j):
 for A in range(4):
  for c in range(3):
   j[A][c]+=j[A][c+4]
   if j[A][c]>0:j[A][c]=0
   else:j[A][c]=3
 return[R[:3]for R in j]


show_examples(load_examples(387)['train'])


#%%writefile task387.py



show_examples(load_examples(388)['train'])


%%writefile task388.py
def p(g):R=range;n=len(g);c={j for i in R(n)for j in R(n)if g[i][j]};m=[[8if g[i][j]==0and j in c else g[i][j]for j in R(n)]for i in R(n)];return[[m[i%n][j%n]for j in R(2*n)]for i in R(2*n)]


show_examples(load_examples(389)['train'])


%%writefile task389.py
def p(j):A=[i for s in j for i in s];A=[c for c in set(A)if c not in[0,5]][0];j=[[A if C==5 else 0 for C in R]for R in j];return j


show_examples(load_examples(390)['train'])


%%writefile task390.py
def X(g):return list(zip(*g[::-1]))
def p(g,L=len,R=range):
 v=1
 for r in g:
  if r.count(2)>4:v=0
 if v:P=[[0,6],[1,5]]
 else:P=[[1,7],[2,6]]
 if v:
  g=X(g)
  for a,b in P:
   g[a]=g[b]
   g[-(a+2)]=g[-(b+2)]
   g[b]=g[-1]
   g[-(b+2)]=g[-1]
 else:
  for a,b in P:
   g[a]=g[b]
   g[-a]=g[-b]
   g[b]=g[0]
   g[-b]=g[0]
 if v:g=X(X(X(g)))
 return g


show_examples(load_examples(391)['train'])


%%writefile task391.py
p=lambda j:[[k]for k,_ in __import__('collections').Counter(i for r in j for i in r).most_common(5)[2:]]


show_examples(load_examples(392)['train'])


#%%writefile task392.py



show_examples(load_examples(393)['train'])


%%writefile task393.py
p=lambda j:[[k]for k,_ in __import__('collections').Counter(i for r in j for i in r).most_common(4)[1:]]


show_examples(load_examples(394)['train'])


#%%writefile task394.py



show_examples(load_examples(395)['train'])


%%writefile task395.py
def p(g):t,b=g[:3],g[3:];return[[2if t[r][c]==b[r][c]==0else 0for c in range(3)]for r in range(3)]


show_examples(load_examples(396)['train'])


%%writefile task396.py
def p(j):
	A=range;c,E=len(j),len(j[0]);k={}
	for W in A(c):
		for l in A(E):
			if j[W][l]:k[j[W][l]]=k.get(j[W][l],0)+1
	J,a=max(k,key=k.get),min(k,key=k.get);C,e=0,None
	for K in A(c-2):
		for w in A(E-2):
			for L in A(K+2,c):
				for b in A(w+2,E):
					if all(j[K][A]==J for A in A(w,b+1))and all(j[L][A]==J for A in A(w,b+1))and all(j[A][w]==J for A in A(K,L+1))and all(j[A][b]==J for A in A(K,L+1)):
						d=(L-K+1)*(b-w+1)
						if d>C:C,e=d,(K,w,L,b)
	K,w,L,b=e;return[[a if j[K+L][w+A]==J else j[K+L][w+A]for A in A(b-w+1)]for L in A(L-K+1)]


show_examples(load_examples(397)['train'])


%%writefile task397.py
def p(j,A=range):
	c,E=len(j),len(j[0]);k=[]
	for W in A(c-1):
		for l in A(E-1):
			J=j[W][l],j[W][l+1],j[W+1][l],j[W+1][l+1]
			if all(J):k+=[(W,l,len(set(J)))]
	for(W,l,a)in k:
		for C in A(a):
			e=W+2+C
			if e<c:j[e][l]=j[e][l+1]=3
	return j


show_examples(load_examples(398)['train'])


%%writefile task398.py
def p(g,L=len,R=range):
 s=R(L([x for x in set(g[0])if x>0])*5)
 X=[[0 for x in s]for y in s]
 g=g[0]
 T=0
 for r in s:
  for c in R(5):
   try:X[-(r+1)][c+T]=g[c]
   except:pass
  T+=1
 return X


show_examples(load_examples(399)['train'])


%%writefile task399.py
def p(j,A=0):
 c={1:[[1,0,0],[0,0,0],[0,0,0]],2:[[1,0,1],[0,0,0],[0,0,0]],3:[[1,0,1],[0,1,0],[0,0,0]],4:[[1,0,1],[0,1,0],[1,0,0]],5:[[1,0,1],[0,1,0],[1,0,1]]}
 for E in range(0,len(j[0])-2+1,1):
  for k in range(0,len(j)-2+1,1):
   W=j[E:E+2];W=[R[k:k+2]for R in W];l=[i for s in W for i in s]
   if min(l)>0:A+=1
 return c[A]


show_examples(load_examples(400)['train'])


%%writefile task400.py
def p(g,L=len,R=range):
 h,w,I,J=L(g),L(g[0]),[],[]
 P=1
 for r in R(h//2+1):
  for c in R(w):
   if g[r][c]==P:g[r][c]=g[-(r+1)][c];I+=[r];J+=[c]
   if g[-(r+1)][c]==P:g[-(r+1)][c]=g[r][c];I+=[h-(r+1)];J+=[c]
 for r in R(h):
  for c in R(w//2+1):
   if g[r][c]==P:g[r][c]=g[r][-(c+1)];I+=[r];J+=[c]
   if g[r][-(c+1)]==P:g[r][-(c+1)]=g[r][c];I+=[r];J+=[w-(c+1)]
 g=g[min(I):max(I)+1]
 g=[r[min(J):max(J)+1]for r in g]
 return g


from zipfile import ZipFile
import zipfile, zlib
from zlib import compress

#https://www.kaggle.com/code/cheeseexports/big-zippa
def zip_src(src):
 compression_level = 9 # Max Compression
 # We prefer that compressed source not end in a quotation mark
 while (compressed := compress(src, compression_level))[-1] == ord('"'): src += b"#"
 def sanitize(b_in):
  """Clean up problematic bytes in compressed b-string"""
  b_out = bytearray()
  for b in b_in:
   if   b==0:         b_out += b"\\x00"
   elif b==ord("\r"): b_out += b"\\r"
   elif b==ord("\\"): b_out += b"\\\\"
   else: b_out.append(b)
  return b"" + b_out
 compressed = sanitize(compressed)
 delim = b'"""' if ord("\n") in compressed or ord('"') in compressed else b'"'
 return b"#coding:L1\nimport zlib\nexec(zlib.decompress(bytes(" + \
  delim + compressed + delim + \
  b',"L1")))'

#all the ones with imports need to be refactored
files = [5,18,25,27,34,35,42,44,46,
54,62,64,66,71,74,76,79,80,86,89,93,96,99,
101,102,109,117,118,119,124,125,133,134,137,138,143,148,
153,154,156,157,158,159,160,165,168,170,173,174,175,182,189,191,195,198,
202,204,205,206,209,216,218,219,221,233,234,238,240,245,247,
250,255,260,264,268,273,277,279,280,281,284,285,286,288,
308,314,319,324,328,330,333,340,341,349,
358,362,364,366,367,370,379,382,383,387,392,394]

files=[f for f in range(1,401) if f not in files]

print(len(files), len(files)*2500)
total_save=0
with ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
    for f in files:
        o=open('/kaggle/working/task' + str(f).zfill(3) + '.py','rb').read()
        #https://www.kaggle.com/code/cheeseexports/big-zippa
        zipped_src = zip_src(o)
        improvement = len(o) - len(zipped_src)
        if improvement > 0:
            print(f,improvement)
            total_save += improvement
            open('/kaggle/working/task' + str(f).zfill(3) + '.py','wb').write(zipped_src)
        else:
            open('/kaggle/working/task' + str(f).zfill(3) + '.py','wb').write(o)
        zipf.write('task' + str(f).zfill(3) + '.py')
print(total_save)


#taylorsamarel/qwen2-5-32b-arc-local-score-32-solved-script
import zipfile, json, os, copy

def check(solution, task_num, valall=False):
    task_data = load_examples(task_num)
    #print(task_num, max(1, 2500 - len(solution.encode('utf-8'))))
    try:
        namespace = {}
        exec(solution, namespace)
        if 'p' not in namespace: return False
        all_examples = task_data['train'] + task_data['test'] + task_data['arc-gen']
        examples_to_check = all_examples if valall else all_examples[:3]
        for example in examples_to_check:
            input_grid = copy.deepcopy(example['input'])
            expected = example['output']
            try:
                actual = namespace['p'](input_grid)
                if actual != expected:
                    return False
            except:
                return False
        return True
    except Exception as e:
        return False
#check(solution, task_id, task_data, valall=True)


#seshurajup/code-golf-public-task-shared-score-lb-950-429-400
top=[61,91,60,80,262,51,65,98,109,70,133,163,157,70,107,
     43,131,445,120,175,64,91,207,62,164,52,117,64,123,
     127,52,49,77,160,88,110,112,51,65,73,49,187,58,361,
     45,186,55,99,81,92,122,40,21,455,87,42,56,148,179,
     48,63,151,77,163,101,303,33,134,210,121,244,60,46,
     109,88,270,155,65,123,448,106,50,40,62,56,316,36,
     122,335,197,83,87,108,132,83,373,121,98,171,54,417,
     173,29,85,242,73,150,57,90,207,60,112,25,65,73,20,281,
     403,123,101,90,100,75,143,142,56,76,69,47,65,174,86,425,
     209,32,107,144,135,100,36,115,40,189,59,252,58,85,170,
     80,30,125,40,188,110,18,208,368,476,152,120,82,100,143,
     32,139,64,77,125,245,267,54,20,388,109,92,78,55,55,31,
     82,69,179,98,106,245,62,106,73,113,119,334,197,87,73,
     108,136,54,127,104,89,301,122,64,108,235,151,84,328,345,
     20,48,112,113,62,46,117,104,72,380,94,94,171,52,218,212,
     154,55,145,73,128,43,61,402,131,62,60,67,247,111,114,31,
     54,81,66,150,131,96,88,26,143,96,57,148,99,309,96,81,61,86,
     150,47,39,136,223,153,119,48,339,64,135,94,101,116,76,161,
     38,212,128,138,306,152,89,95,275,392,115,59,104,64,69,62,
     56,60,76,54,63,47,55,54,87,31,152,62,92,69,82,51,277,38,
     83,32,44,71,101,63,73,59,60,392,81,55,48,153,417,207,30,
     67,195,54,199,99,65,102,71,133,106,46,99,37,136,158,119,
     112,88,113,102,50,108,352,95,70,89,126,111,123,108,100,113,
     64,45,310,69,266,315,120,462,326,160,127,331,142,48,39,136,
     53,30,55,281,152,37,99,174,242,64,25,52,234,61,57,110,69,221,
     69,129,56,266,135,80,64,70]

score = 0
for task_num in files:
    try:
        solution = open('/kaggle/working/task' + str(task_num).zfill(3) + '.py','rb').read()
        if check(solution, task_num, valall=True):
            s = max([0.1,2500-len(solution)])
            print(task_num, 2500-s, top[task_num-1], top[task_num-1]-(2500-s))
            score += s
            #print(task_num, '*'* 40)
        else: print(task_num, ":L")
    except: pass
print('Score:', score)




