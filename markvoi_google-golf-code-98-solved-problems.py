import sys
sys.path.append("/kaggle/input/google-code-golf-2025/code_golf_utils")
from code_golf_utils import *
show_legend()


%%writefile task.py
p=lambda g:[[x&y for x in a for y in b]for a in g for b in g]


example_num = 1
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(g):
 def f(i,j):
  try:
   if g[i][j]<1:g[i][j]=1;f(i+1,j);f(i-1,j);f(i,j+1);f(i,j-1)
  except:0
 for i in range(len(g)):f(i,0);f(i,-1);f(0,i);f(-1,i)
 return[[(4,0,0,3)[a]for a in r]for r in g]


example_num = 2
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda g:[[c*2 for c in r]for r in g+(g[:3],g[2:5])[g[1]!=g[4]]]


example_num = 3
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(g,p=enumerate):
 a=[[0]*len(g[0])for _ in g]
 for q in{*sum(g,[])}-{0}:
  f=[(y,x)for y,r in p(g)for x,v in p(r)if v==q];k,l=map(max,zip(*f))
  for y,x in f:a[y][x+(y<k)*(x<l)]=q
 return a


example_num = 4
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda j:[[x and y and 2for x,y in zip(r,r[4:])]for r in j[:3]]


example_num = 6
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(g):a,b=range(len(g)),range(len(g[0]));d={(i+j)%3:g[i][j]for i in a for j in b if g[i][j]};return[[d.get((i+j)%3,0)for j in b]for i in a]


example_num = 7
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
S,C,B=enumerate,max,min
def p(J):
	a=b=m=len(J);c=d=n=len(J[0]);e=g=0
	for y,r in S(J):
		for x,v in S(r):
			if v&3:a=B(a,y);e=C(e,y);c=B(c,x);g=C(g,x)
			if v&8:b=B(b,y);d=B(d,x)
	Y=(g<d)*(d+~g)+(d+1<c)*(d-c+2);X=(e<b)*(b+~e)+(b+1<a)*(b-a+2);X*=Y<1
	return[[J[y][x]&8or 0<=y-X<m and 0<=x-Y<n and J[y-X][x-Y]&3for x in range(n)]for y in range(m)]


example_num = 8
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(m):
 I=range(k:=len(m));h=m[2][2];r=eval(str(n:=[[v*(v!=h)for v in R]for R in m]));E=lambda t:(t.index(c),k-t[::-1].index(c))
 for c in I:
  for i in I:
   if c in(t:=n[i]):a,b=E(t);r[i][a:b]=[c]*(b-a)
   if c in(t:=[r[i]for r in n]):
    for j in range(*E(t)):r[j][i]=c
 return[[(h,r[i][j])[m[i][j]!=h]for j in I]for i in I]


example_num = 9
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda g:(a:={})or[[x if x-5 else a.setdefault(i,len(a)+1) for i,x in enumerate(r)]for r in g]


example_num = 10
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda g,A=range:(lambda c,e:[[5if x%4==3or y%4==3else g[c*4+x//4][e*4+y//4]for y in A(11)]for x in A(11)])(*next((c,e)for c in A(3)for e in A(3)if sum(g[c*4+w][e*4+l]==0 for w in A(3)for l in A(3))==5))


example_num = 11
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(g):
 n=a=b=e=d=0;c=[]
 if(n:=sum(g[0])or sum(g[-1])):g=[*zip(*g[::-1])]
 w=len(g[0])
 for i in range(len(g)):
  if g[i][0]+g[i][-1]and b<2:g[i]=[g[i][0]+g[i][-1]]*w;b+=1;c+=g[i][:1];d=i
  a+=0<b<2
 for i in range(d+a,len(g),a):g[i]=[c[e]]*w;e^=1
 if n:g=[*map(list,zip(*g))][::-1]
 return g


example_num = 13
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
from collections import*
def p(j):
 A=[x for k in j for x in k];c=Counter(A).most_common(3);c=[c for c in c if c[0]>0][-1][0];j=[k for k in j if c in k];E=[]
 for k in j:
  for W in range(len(k)):
   if k[W]==c:E+=[W]
 return[k[min(E):max(E)+1]for k in j]


example_num = 14
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(g,r=range(9)):
 s=[(i,j,v)for i in r for j in r if 0<(v:=g[i][j])<3]
 for i,j,v in s:
  for a in-1,0,1:
   for b in-1,0,1:
    if a|b and(a*b!=0)==(v>1):g[i+a][j+b]=7-3*(v>1)
 return g


example_num = 15
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda j,s='0564312798':[[int(s[x])for x in r]for r in j]


example_num = 16
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda g:[[(l:=g[0][0])]*-~sum(x!=l for x in g[0])]*-~sum(r[0]!=l for r in g)


example_num = 21
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda g,E=enumerate:(Z:={i for R in g for i,x in E(R)if x==2})and 0 or[[1in R and 1or 3 in R and 3or v or(i in Z)*2for i,v in E(R)]for R in g]


example_num = 24
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda j:[[8*(a|b==0)for a,b in zip(a,a[4:])]for a in j]


example_num = 26
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda g:[*map(list,zip(*map(sorted,zip(*g))))]


example_num = 32
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
r=range(8);p=lambda g:[([1]*sum(g[i][j]*g[i+1][j]*g[i][j+1]*g[i+1][j+1]==1 for j in r for i in r)+[0]*5)[:5]]


example_num = 38
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda j,a=0:[[x and(a:=(a==0)*x)or x or a for x in r]for r in j]


example_num = 41
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(g):
 for i in g:
  if i[0]==i[-1]:i[:]=i[:1]*len(i)
 return g


example_num = 45
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda j:[[(r.count(r[0])>2)*5]*3for r in j]


example_num = 52
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda g:g[-1:]+g[:-1]


example_num = 53
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda j:(x:=j[0],[[{6:1,5:2,3:3,2:6}[(x[0]!=0)*4+(x[1]!=0)*2+(x[2]!=0)]]])[1]


example_num = 56
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda j:(A:={i for r in j for i,x in enumerate(r)if x>0},[r[min(A):max(A)+1]*2for r in j if max(r)>0])[1]


example_num = 57
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(j):
 for w in j:
  if w[0]:w[:]=w[:1]*5+[5]+w[-1:]*5
 return j


example_num = 60
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda g:[a[:len(g)]for a in g]


example_num = 67
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda g,r=range:[[3*(g[i][j]+g[i+7][j]==2)for j in r(5)]for i in r(6)]


example_num = 72
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(g):
 for i,x in enumerate(g[2]):
  if x:g[2][i],g[-1][i]=0,1
 return g


example_num = 73
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(g):
 for i,c in enumerate(g[0][1:-1],1):
  if c:g[1][i-1:i+2:2]=c,c
 g[2:]=g[:-2];g[2:]=g[:-2];return g


example_num = 82
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda g:[i+i[::-1]for i in g+g[::-1]]


example_num = 83
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(g):
 for i in range(1,len(g[0])):
  g[-1][i]=4;g[i-1][-i]=2
 return g


example_num = 84
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda g:[i[::-1]for i in g[::-1]]


example_num = 87
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
A=enumerate;p=lambda g:[F[C:E+1]for(B,C)in[next((B,A.index(5))for(B,A)in A(g)if 5 in A)]for(D,E)in[next((~B+len(g),~A[::-1].index(5)+len(A))for(B,A)in A(g[::-1])if 5 in A)]for F in g[max(0,B-1):D+2]]


example_num = 91
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(g):
 a=[0]*10;R=range(10)
 for i in R:
  for j in R:
   r=g[i];v=r[j]
   if v:a[v]=r.count(v)*sum(t[j]==v for t in g)
 return[[a.index(max(a))]*2]*2


example_num = 100
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda g:[[1]]if g==g[::-1]else[[7]]


example_num = 103
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(g):
 a=[[3]*4+[0]*5]*4+[[0]*4+[3]*4+[0]]*4+[[0]*9]
 for k in g[0][0],g[0][2],g[2][2],g[2][0]:
  if k:break
  a=[*map(list,zip(*a[::-1]))]
 return a


example_num = 104
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
z=lambda g:[*map(list,zip(*g[::-1]))]
p=lambda g:((g:=[a+b for a,b in zip(g,z(g))]),g+z(z(g)))[1]


example_num = 106
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(j,A=range):c,E=len(j),len(j[0]);k=[[max(j[y][x],j[y][x+1],j[y+1][x],j[y+1][x+1])for x in A(0,E,2)]for y in A(0,c,2)];return[[k[y//4][x//4]for x in A(2*E)]for y in A(2*c)]


example_num = 108
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
R=range;L=len
def p(g):
 h,w=L(g),L(g[0]);C=g[0][w//2];X=[[0]*(w-1)for _ in R(h-1)]
 for r in R(h//2):
  s=~r
  for c in R(w//2):
   t=~c;v=C*(g[r][c]>0)
   X[r][c]=X[s][c]=X[s][t]=X[r][t]=v
 return X


example_num = 109
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(j,u=enumerate):
 A=range;c=len(j);E=len(j[0]);k=lambda W,l:W==l or W*l<1;J=next((K for K in A(1,E)if all(k(L,e)for w in j for(L,e)in zip(w,w[K:]))),E);a=next((K for K in A(1,c)if all(k(L,e)for(K,w)in zip(j,j[K:])for(L,e)in zip(K,w))),c);C={}
 for(e,K)in u(j):
  for(w,L)in u(K):
   if L:C[e%a,w%J]=L
 for(e,K)in u(j):
  for(w,L)in u(K):
   if L<1:K[w]=C[e%a,w%J]
 return j


example_num = 110
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
e=enumerate
def p(j):
 A=c=0
 for i,r in e(j):
  for k,v in e(r):A+=i*(v==3);c+=k*(v==3)
 A>>=1;c>>=1
 for i,r in e(j):
  for k,v in e(r):
   if v==2:r[k]=j[A-i][k]=j[i][c-k]=j[A-i][c-k]=2
 return j


example_num = 112
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda j:j[:5]+j[:5][::-1]


example_num = 113
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(g):
 g=[[r[0]]+r+[r[-1]]for r in[g[0]]+g+[g[-1]]]
 g[0][0]=g[0][-1]=g[-1][0]=g[-1][-1]=0
 return g


example_num = 114
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(j):
 u=lambda a:list(dict.fromkeys(a))
 k=[*map(u,j)]
 return [k[0]] if k.count(k[0])==len(k) else [[e]for e in u(sum(j,[]))]


example_num = 115
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda j:j[::-1]+j


example_num = 116
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(g):
 a=[[*row]for row in zip(*g[::-1])]
 for i in range(len(a)):
  if a[i][0]:q=sum(a[i])//a[i][0];w=a[i][0];a[i]=[0]*15;a[i][q:q+q]=[w]*q
 return [[*row]for row in zip(*a)][::-1]


example_num = 128
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda j:[[max((l:=sum(j,[])),key=l.count)]*3]*3


example_num = 129
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(g,r=range):
 a=[[0]*3for i in r(3)]
 for i in r(9):
  for j in r(9):
   if g[i][j]!=5:a[i//3][j//3]=g[i][j]
 return a


example_num = 130
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda g:[r[-3:]for r in g[:3]]


example_num = 135
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda g:[r[::-1]for r in g[::-1]]


example_num = 140
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda j:(a:=[r+r[::-1]for r in j])+a[::-1]


example_num = 142
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda g,r=range:[[3*(g[i][j]+g[i+5][j]==0)for j in r(4)]for i in r(4)]


example_num = 144
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(g):
 for i in 0,3,6:
  a=g[i:i+3];A=tuple(map(tuple,a))
  if(A==tuple(x[::-1]for x in zip(*a[::-1])))+(A==tuple(zip(*a)))<1:return a


example_num = 146
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda j:[r[::-1]for r in j]


example_num = 150
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda j:(a:=[r+r[::-1]for r in j])+a[::-1]


example_num = 152
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda j:j[::-1]


example_num = 155
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda j:[R+R[::-1]for R in j]


example_num = 164
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda j:j+j[::-1]


example_num = 172
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda g:[*map(list,zip(*g))]


example_num = 179
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(g):
 r=lambda x:[*map(list,zip(*x[::-1]))];a=r(g);b=r(a);return[*map(list.__add__,g,a),*map(list.__add__,r(b),b)]


example_num = 194
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(g):
 for i,a in enumerate(g[-1]):
  if a:
   for r in g:r[i:10:2]=[a]*len(r[i:10:2])
   s=g[0];s[i+1:10:4]=[5]*len(s[i+1:10:4])
   s=g[-1];s[i+3:10:4]=[5]*len(s[i+3:10:4]);break
 return g


example_num = 200
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda j:j+j[::-1]


example_num = 210
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda j:(j:=[r[::-1]+r for r in j],a:=j[:3][::-1],a+j+a)[-1]


example_num = 211
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda j:[[r for j,r in enumerate(j)if sum(r)and j%3==i%3][0]for i in range(len(j))]


example_num = 215
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda g,r=range:[[2*(g[i][j]+g[i+4][j]<1)for j in r(4)]for i in r(4)]


example_num = 227
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(g):f=sum(g,[]);c=f.count;m=max(map(c,f));return [[x if c(x)==m else 5for x in r]for r in g]


example_num = 229
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda g,r=range:[[g[i%5][j%6]for j in r(len(g[0])*2)]for i in r(len(g))]


example_num = 231
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda j:[*map(list,zip(*j))]


example_num = 241
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda j:[E*2for E in j]


example_num = 249
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda g:[[2*(x==6)or x for x in r]for r in g]


example_num = 276
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(g):
 r=range(len(g))
 def l():
  for i in r:
   for j in r:
    if g[i][j]==4:g[i][j]=a[i][j]
 a=g[::-1];l()
 a=[i[::-1]for i in g];l()
 return g


example_num = 287
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(g,a=3):
 l=len(g)//2
 for i in range(l+1):
  if g[-2][l-i]<1:g[-a][l-i]=(b:=g[-1][l]);g[-a][l+i]=b;a+=1
 return g


example_num = 288
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(g):c=len({*sum(g,[])})-1;return [[g[i//c][j//c]for j in range(c*3)]for i in range(c*3)]


example_num = 289
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(g):a=[[j for j in i if j]for i in g if sum(i)];b=[*{*sum(a,[])}];return[[b[j==b[0]]for j in i]for i in a]


example_num = 290
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(g,e=enumerate):
 for i,r in e(g):
  for j,x in e(r):
   if x<1 and (a:=g[i-1][j])and a==r[j-1]:return[[a]]


example_num = 291
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(j):
 for A in j:A[::3]=[6 if v else v for v in A[::3]]
 return j


example_num = 292
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(e):f=lambda a:a[0]and a[:1]*len(a)or a;k=[*map(f,e)];return[k,[*map(list,zip(*map(f,zip(*e))))]][k==e]


example_num = 293
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(g):
 for i in range(1,9):
  for j in range(1,9):
   if g[i+1][j]*g[i][j+1]*g[i-1][j]*g[i][j-1]:g[i][j]=2
 return g


example_num = 294
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(g):r=g[0];l=sum(map(bool,r));g+=(r[:l]+r[:1]*i+[0]*(len(r)-l-i)for i in range(1,len(r)//2));return g


example_num = 295
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(g):return[[g[i][j]or g[i+2][j]or g[i][j+4]or g[i+2][j+4]for j in range(3)]for i in range(3)]


example_num = 296
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(g):(v:=len(g[0]));g[2:]=[[g[0][i]]*v for i in range(v)]*2;return g


example_num = 297
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(j):c=[j[i][i]for i in range(len(j)//2)];d=dict(zip(c,c[-1:]+c));return[[d[x]for x in r]for r in j]


example_num = 298
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(g,r=range(6)):
 for i in r:
  if g[i][0]|g[i][-1]:g[i]=[2]*6;a=i
  if g[0][i]|g[-1][i]:
   for R in g:R[i]=8;b=i
 g[a][b]=4;return g


example_num = 299
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(g,A=enumerate):m=max(range(1,10),key=sum(g,[]).count);X,Y=zip(*((i,j)for i,R in A(g)for j,v in A(R)if v==m));return[R[min(Y):max(Y)+1]for R in g[min(X):max(X)+1]]


example_num = 300
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda g:sorted(sorted(r)for r in g)


example_num = 301
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda j:[[x-2*(x==7)for x in r]for r in j]


example_num = 309
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda j:[R+R[::-1]for R in j]


example_num = 311
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda j:[r[:2]for r in j[:2]]


example_num = 326
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda j:[[x for x in sum(j,[])if x]]


example_num = 339
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
l=len;r=range
def p(g):
 d=[0]
 for _ in r(4):
  g=[*map(list,zip(*g[::-1]))]
  for i in r(1,l(g)-1):
   a=g[i];c=a[0];d+=[c];t=1
   for j in r(1,l(a)-1):
    if a[j]==c:a[t]=c;a[j]=0;t+=1
 return [[x*(x in d)for x in y]for y in g]


example_num = 340
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(g,e=enumerate):
 def f():
  a=[i for i,r in e(g)if len({*r})>2]
  for i in a[1:-1]:
   s=set()
   for j,x in e(g[i]):
    if x:s.add(x)
    if x<1&len(s)==1:g[i][j]=8
 f();g=[*map(list,zip(*g[::-1]))];f();g=[*map(list,zip(*g))][::-1];return g


example_num = 341
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(g,r=enumerate):
 def f():
  s=[0]*10
  for i,u in r(g):
   for j,v in r(u):
    if v!=8 and v:s[j]=v;g[i][j]=0
    if v==8:s=[k for k in s if k];g[i][j],g[i][j+1]=s[0],s[1];return
 for _ in 0,1:f();g=g[::-1]
 return g


example_num = 342
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(j,A=enumerate):
 for c,E in A(j):
  for k,W in A(E):
   for l,J in(c+1,k),(c-1,k),(c,k+1),(c,k-1):
    if W==2and 0<=l<len(j)and 0<=J<len(E)and j[l][J]==3:j[c][k]=0;j[l][J]=8
 return j


example_num = 344
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(j):
 for a in range(len(j[0])):
  if j[-1][a]==2:
   c=0
   for e in range(len(j)):
    if j[~e][a+c]==5:c+=1;j[-e][a+c]=2
    j[~e][a+c]=2
 return j


example_num = 345
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(g):
 for i in range(1,len(g)-1):
  for j in range(1,len(g[0])-1):
   if g[i][j]and len(s:={g[i+a][j+b]for a in(-1,0,1)for b in(-1,0,1)if a|b})<2and s.pop():return[[g[i][j]]]


example_num = 346
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
p=lambda j:[*map(list,zip(*j))][::-1]


example_num = 380
examples = load_examples(example_num)
verify_program(example_num, examples)


%%writefile task.py
def p(g):
 c=sum(g[i][j]==g[i][j+1]==g[i+1][j]==g[i+1][j+1]==2for i in range(len(g)-1) for j in range(len(g)-1));a=[[0]*3for _ in'***']
 for i in range(c):a[i*2//3][i*2%3]=1
 return a


example_num = 399
examples = load_examples(example_num)
verify_program(example_num, examples)

