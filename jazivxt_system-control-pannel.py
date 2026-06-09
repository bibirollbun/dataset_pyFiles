import sys
sys.path.append("/kaggle/input/google-code-golf-2025/code_golf_utils")
from code_golf_utils import *
show_legend()


%%writefile task001.py
p=lambda*g:[[*g,min,p][2](r,s)for r in g[0]for s in g[-1]]


%%writefile task002.py
p=lambda g,i=67:g*-i or p([[r.pop()%sum(r[-1:],4)or i>>4&4for r in g]for r in g],i-1)


%%writefile task003.py
p=lambda g:[[x*2for x in r]for r in g+g[g==2*g[:3]:][2:5]]


%%writefile task004.py
import re;p=lambda i:eval(re.sub(r"(.),(?=.*\1.*0, \1)",r"0,\1|",str(i)))


%%writefile task005.py
import re
p=lambda g:eval(re.sub(f"0(?=(?=(.)+{'(.{2%%d})+(?<=%s))'%max(str(29**14),key=f'{g}'.count)*2}|"*2%(*b"<<HH",),r"\1\4",f'{*zip(*g[147:]or p(g*2)),}'))[::-1]


%%writefile task006.py
p=lambda g:[eval('r.pop(0)*r[3]*2,'*3)for r in g]


%%writefile task007.py
p=lambda g:eval(f'[{"max(sum(g:=g[1:3]+g,[])[2::3]),"*7}],'*7)


%%writefile task008.py
p=lambda g,*n:sorted(zip(*n or p([],*g)),key=lambda r:(8in g.__iadd__(r))*3^any(r))


%%writefile task009.py
p=lambda g,*r,i=0:[x|max({*g[i::3]}&{*g[:(i:=i+1)]})for x in r]or[*map(p,g,*map(p,zip(*g),*g))]


%%writefile task010.py
p=lambda g:[g:=[(x*-5or y*sum(r))%6for x,y in zip(g,r)]for r in g]


%%writefile task011.py
p=eval("lambda a:max(a*(not'8'in'%s'%a)"+f"for*a,in[*map(zip,a,a,a{',a[3:]*9,*[a[%d:]]*3'*2%(1,2)})][::4]"*2+")")


%%writefile task012.py
import re
p=lambda g:eval([g:=re.sub(r"0(?=.{%d}([^0])..(.)..\1)"%x,f"\{895%x%5}",str(g)[::-1])for x in b'HHNB%'*2][-1])


%%writefile task013.py
p=lambda g,h=0,l=[]:[[(l:=[max(c+[j*(i>0)for i,j in zip(l,l[1::2])])]+l)[:1]*len(c),c][c[1:-1]>c]for*c,in zip(*h or p(g,g))]


%%writefile task014.py
p=lambda g:[p(zip(r,*g))or r[0]for r in[*g]if[*{*r}][2:]]


%%writefile task015.py
p=lambda g,i=-7:g*i or p([[r.pop()%9|36%(6^-[0,*r][i//7])%13for r in g]for*r,in g],i+1)


%%writefile task016.py
p=lambda g:[[x**10%95%18^4for x in g[0]]]*3


%%writefile task017.py
p=lambda g:[[*map(max,*[r*any(-i^-j>0for i,j in zip(r,s))+s for s in g])]for*r,in zip(*g)]


%%writefile task018.py
def p(a,T=enumerate):e={m*1j+f:a for m,a in T(a)for f,a in T(a)if a};[(l:={r},[l:={r}|l for r in[*e]*5for u in l if abs(r-u)<2],[*l][3:]and[5for i in[1,3,6,7]for n in e if all(sum(e[u]==e.get(f)for f in[*l,(u-r-i//4*(u-r).real*2)*1j**i+n])>1for u in l)for u in l if(a:=[[{u:0,(u-r-i//4*(u-r).real*2)*1j**i+n:e[u]}.get(m*1j+f,a)for f,a in T(a)]for m,a in T(a)])])for r in e];return a


%%writefile task019.py
p=lambda g,n=7:-n*g or p(-~(n>5)*[g:=[r.pop()or(x*-1or 0)%-8&8for x in[0]+g[:-1]]for*r,in zip(*g)],n-1)


%%writefile task020.py
def p(g):
 B,C=[[*map(any,A)].index(1)for A in(g,zip(*g))];A=75
 while A:A-=1;D=A//5%5;g[B+D][C+A%5]|=g[B+~A%5][C+D]
 return g


%%writefile task021.py
p=lambda i:i*-1*-1or-~min(map(i.count,i))*[p(i[0])]


%%writefile task022.py
r=-1,0,1
p=lambda g:[[dict(sorted(zip(o:=sum(g,[]),o[x*11+y:]+o)))[5]for y in r]for x in r]


%%writefile task023.py
import re;p=lambda i,w=2:s!=(r:=re.sub((w%2*"5, "+"5(.%s)??")%{w*3%-7%len(i[0]*3)+2}*(3-w%2)," 82,\81\ 12 \82, 82"[w::2],s,1))and p(eval(r))or w and p(i,w-1)if"5"in(s:=str(i))else i


%%writefile task024.py
p=lambda g:[[3%-~max(r)or(2in x)*2for x in zip(*g)]for r in g]


%%writefile task025.py
p=lambda g,N=0:[g:=[[any(min(g)+[I:=(x:=r.pop())in r,A:=all(c)])*x|N+(N:=A*I*x)for c in g[::-1]]for*r,in zip(*g)]for _ in g][3]


%%writefile task026.py
p=lambda g:[eval('8>>r.pop(0)+r[3],'*3)for r in g]


%%writefile task027.py
R=range(10);p=lambda g:[[g[i][j]or 2*g[~j+([*[*zip(*g)][5]]<g[5][::-1])][i]for j in R]for i in R]


%%writefile task028.py
p=lambda g:[[w:=max(g[y%15]),*[y&w]*8,w]for y in b'/ /  ppp']


%%writefile task029.py
p=lambda g,w=9:g*w and p(g,w-1)+[g:=[r[~x::-1]for*r,in zip(*g)if w in r]for x in[0]*4+[1]*5][7]*([]==g)


%%writefile task030.py
p=lambda g:[*zip(*map(lambda*r:r[(k:=(f:=sum(g,[]).index)(max(r))//10-f(1)//10):]+r[:k],*g))]


%%writefile task031.py
p=lambda g,*G:[*filter(any,zip(*G or p(*g)))]


%%writefile task032.py
p=lambda d:[*zip(*map(sorted,zip(*d)))]


%%writefile task033.py
p=lambda g,h=[],i=0:g*0!=0and[*map(p,g[:6]*3,h+g,g[5:6]*17)]or(g>h)*i|h


%%writefile task034.py
import re
p=lambda g,n=-35:n*g or eval(re.sub("([^02]), 2(.{28})0",r"*[\1]*3\2\1,2",f"{*zip(*p(g,n+1)),*g}"))[8::-1]


%%writefile task035.py
p=lambda g:g[150:]or[[g:=l[e!=8or g>0]or e for e in l]for*l,in zip(*p(g*2))][::-1]


%%writefile task036.py
p=lambda g,*G:[f for*f,in zip(*G or p(g,*g))if{*f}-{*sum(g[:5]+g[-4:],[])}]


%%writefile task037.py
import re
p=lambda g:g[:~99]or p(eval(re.sub('(?<=(.).{34})(?=(.{35})*\\1)0','\\1',str(g[9::-1])))+g)


%%writefile task038.py
p=lambda g:[(str(g).count("1, 1")*[1]+9*[0])[:9:2]]


%%writefile task039.py
p=lambda g:[*eval("zip(*[*filter(any,"*2+"g)]))][:3])")][:3]


%%writefile task040.py
p=lambda g,h=[]:g*0!=0and[*map(p,g[:1]*5+g[9:]*5,h+g)]or h%~h&g


%%writefile task041.py
p=lambda o,c=0:[[r|(c:=r^c)for r in r]for r in o]


%%writefile task042.py
import re
p=lambda g:exec("s=str(g);g[::-1]=zip(*eval(re.sub('0(?=.{%r}3.{%r}3)'%(*b'%Kq9V'[s.count('3')//8::3],),'8',s)));"*4)or g


%%writefile task043.py
p=lambda g:[[x+r[-1]&r.pop(0)+2for x in g[0]]for*r,in g]


%%writefile task044.py
from re import*
p=lambda o,n=X:-n*o or p(eval([o:="%s"%o,sub(x:=str(n%S),"0","|n%S".join(s:=split("(?<=5.{28}5..%s)(?=..5.{28}5)"%sub(x,"0)(",sub("[^%s]"%x,".",o)).strip("."),o)))][s.count("")%2]),n-1)


%%writefile task045.py
p=lambda i:[n[:n[0]==n[9]]*10or n for n in i]


%%writefile task046.py
def p(g,d=3):g=(5,),*zip(*g);return*zip(*[[sum({*x*(q+r+s)}-{5})for x in 3*r][d:3+d]for q,r,s in zip(g,g[1:],g[2:]+g)if any(r)or(d:=d-[*q,5].index(5)+s.index(5))*0]),


%%writefile task047.py
p=lambda g:[[sum({*r+c})%13for*c,in zip(*g)]for r in g]


%%writefile task048.py
p=lambda g:[[hash((*b'+]`dBPx <IaAacF#p3e7"kz0W}k&N%r'[sum(b'%r'%g)%39:]%g,))&8]]


%%writefile task049.py
p=lambda g:[r*[v]for l in g if(r:=l.count(v:=min(r:=sum(g,[0]*99),key=r.count)))]


%%writefile task050.py
p=lambda g,x=0:[[c|((x:=~sum(r)&c^x)>6>c)*3for c in r]for*r,in zip(*x*g or p(g,1))]


%%writefile task051.py
p=lambda i:[*eval("map(lambda*x,l=0,b=1,a=1:[[l:=l|(b!=y>a<1)*(a:=b),b:=y][y>0]for y in x][::-1],*"*4+"i))))")]


%%writefile task052.py
p=lambda g:[[len({*r})%2*5]*3for r in g]


%%writefile task053.py
p=lambda g:(g*2)[2:5]


%%writefile task054.py
def p(h,r=range(30)):
 n=sum(h,[]);s,d=sorted({*n},key=n.count)[-2:];t,x=[],[];[(t,x)[s in([],h[y][n+1])].append((y,n))for y in r for n in r if h[y][n]not in{d,s}];k=t[len(t)//2]
 for(y,n)in t:
  o=h[y][n];i=y-k[0];e=n-k[1];h[y][n]=d
  for(s,w)in x:s+=i;w+=e;h[s][w]-d and exec("h[s][w]=o;r,t=i>>1,e>>1;i*i+e*e==4==exec('while h[s][w]-d:h[s][w]=o;s+=r;w+=t')")
 return h


%%writefile task055.py
p=lambda i,z=0:i*0!=0and[p(y,3*(z:=z+([y]>i)))for y in i]or i or 2222096>>z&7


%%writefile task056.py
p=lambda g:[[2^(g<[[1]])+3*0**g[0][2]]]


%%writefile task057.py
p=lambda g:[*filter(any,zip(*g[8:]or p(g*2)*2))]


%%writefile task058.py
p=lambda a:a and[len(a)*[3],[*a[0],3>>2%len(a),3][2:],*zip(*p([*zip(*a[2:])])[::~0])]


%%writefile task059.py
R=range(11);p=lambda g,n=36,o=0:[[(g[i][j]==5)*5or(sum(S:=[g[i&12|k%3][j&12|k//3]for k in R[:9]])>n!=[o:=1])*max(S)for j in R]for i in R]*o or p(g,n-1)


%%writefile task060.py
p=lambda g:[r[:1]*5+[i%~i%6]+[i]*5for*r,i in g]


%%writefile task061.py
p=lambda g,q=range(18):[[i*j%max(g)[1]+1for j in q]for i in q]


%%writefile task062.py
p=lambda g:exec("q=[]\nfor r in zip(*g):q+=q[::~0]*({*r}^{2}=={3}<{*q[-1]})or[[v or 3for v in r]]\ng[:]=q[9::~0];"*8)or g


%%writefile task063.py
p=lambda g:[[x|3>>x+sum(r[1:-any(c)])for _,*c,_,x in zip(*g,r)]for r in g]


%%writefile task064.py
p=lambda g,n=-7:n*g or p([[P:=[x:=r.pop(),P][r.count(max({*r*P*(g[0].count(x)>3),n}-{x,P}))>2]for _ in g]for*r,in zip(*g)if[P:=0]],n+2)


%%writefile task065.py
p=lambda*g:min(g,key=g.count)if g[3:]else[*map(p,*g,*[h[len(h)//2+1:]for h in g])]


%%writefile task066.py
def p(e,T=range):
 def A(e,o,n,i,r,l=5):
  e=[A*1for A in e]
  while 0<o<len(e)-1>n>0==(B:=e[o+i][n+r]):o+=i;n+=r;e[o][n]=3
  return A(e,o,n,r,i,l+1)or A(e,o,n,-r,-i,l+1)if B>7>l else(B==2)*e
 return max(A(e,B,C,E,D)for B in T(len(e))for C in T(len(e))for E in T(-1,2)for D in T(-1,2)if 3==e[B][C]==e[B][C-D])


%%writefile task067.py
p=lambda i:[n[:len(i)]for n in i]


%%writefile task068.py
p=lambda g,x=0:[[(x:=x*2|2>>sum(g,g).count(y))%2*y|(x>>89&7345159>0)*2for y in r]for*r,in g*2][10:]


%%writefile task069.py
def p(g):
 *b,=a=sum(g,[]);i=j=100
 while i:i-=1;a[i]==8==exec("if b[j:=j-1]%8:g+=j,;a[i-g[10]+j]=b[j];a[j]=0\n"*j)
 return*zip(*[iter(a)]*10),


%%writefile task070.py
p=lambda g:[[max({*i,hash(i)%1070}&{*r})%5**i[0]for i in zip(r,*g)]for r in g]


%%writefile task071.py
p=lambda g,a=0:[[a*(v>0<r[~i+t])for i,v in enumerate(r)if a or{t:=i-r[::-1].index(a:=v)}]for r in g]


%%writefile task072.py
p=lambda g,u=[]:g*0!=0and[*map(p,g,u+g[7:])]or(g!=u)*3


%%writefile task073.py
p=lambda i:i[:1]*3+[i[3],[5-n*4for n in i[2]]]


%%writefile task074.py
p=lambda g:g[90:]or p(g+[*zip(*map(map,[min]*30,p(g[:30]+g),g[:2]+g[::-1]))])


%%writefile task075.py
R=range(9);p=lambda g:[g[i][:4]+[g[i-i%3+1][k-k%3+5]*g[i%3][k%3]for k in R]for i in R]


%%writefile task076.py
def p(g):
 A=enumerate;B={D+1j*B:C for(D,B)in A(g)for(B,C)in A(B)if C};A=[min(B,key=B.get)];D=0
 for C in A:A+=[B for B in B if(abs(B-C)<2)>(B in A)];D+=C*(B[C]==2)
 for F in B:
  for E in range(8):
   for(G,C)in(H:=[(B[A],F+(A-D-(E>>2)*2j*(A-D).imag)*1j**E)for A in A])*all(B.get(C,A|1)==A for(A,C)in H):g[int(C.real)][int(C.imag)]=G
 return g


%%writefile task077.py
p=lambda i,k=7,*w:k and p([*map(p,i,[k>1]*99,[i*2]+i,i[1:]+[i*2],*w)],k-1)or((c:=w.count)(2)+c(4)>=2!=i)*4or i


%%writefile task078.py
p=lambda i,*n:sorted(n,key=0 .__eq__)or[*zip(*map(p,i,*i))]


%%writefile task079.py
p=eval(f"lambda a:max(b:=[a {'for*a,in map(zip,a,a[1:],a[2:])'*2}if any(min(*a,*zip(*a)))],key=b.count)")


%%writefile task080.py
import re
def p(g):l=sum(g,[]);w=len(g);a,b,c,d=max(({*(q:=l[i:i+3]+l[i-2*~w::w*w]),0},i,q)for i in range(w*w))[2];return[g:=eval(re.sub(f"(?={-~l.index(b)*x*'.'}{a})0","y",f"{*zip(*g[::-1]),}"))for x,y in[(3,c)]*4+[(3*w+5,d)]*4][7]


%%writefile task081.py
p=lambda g:exec(f"g[:]={'(q:=1)*[q:=r.pop()or[1]<r[-1:q]for r in g],'*7};"*4)or g


%%writefile task082.py
p=lambda g:[x:=g[0],[*map(max,[0]+x,x[1:]+[0])]]*3


%%writefile task083.py
p=lambda D:[D+D[::-1]for D in D+D[::-1]]


%%writefile task084.py
def p(i,x=1):i[-1][x]=4;i[~x][x]=2;i[:~x]and p(i,x+1);return i


%%writefile task085.py
p=lambda g:g*0!=0and[g:=(v,p(v))[v==g]for v in g]


%%writefile task086.py
p=lambda i,k=7,s=0:-k*i or[[[-((s:=[abs(s)or 1,s&s//4][y>0]-1)>1)|y,*[x for x in sum(i,x)if 0<x!=y!=0],0][k>6]for y in x]for*x,in zip(*p(i,k-1)[::~0])]


%%writefile task087.py
p=lambda o:[o[::-1]for o in o[::-1]]


%%writefile task088.py
exec(f"def p(g):{'g[:]=zip(*g[any(h:=g[-1])-2::-1]);'*48}return[[h[g<1]\nfor g in g][1:-1]#"*2)


%%writefile task089.py
def p(i,T=enumerate):
 for D in(A:={B*1j+C:A for(B,A)in T(i)for(C,A)in T(A)if A}):
  for B in A:
   E={B}
   for C in[*A]*3:
    if A[D]==A[B]!=any(0<abs(D-A)<2for A in A)<any(abs(C-A)<2for A in E):E|={C};i[int((C-B+D).imag)][int((-(-1)**A[B]*(C-B)+D).real)]=A[C]
 return i


%%writefile task090.py
import re
p=lambda g:eval('6'.join(max([re.split((f"(..{{{len(g[0])*3-i%8*3}}})0{i%8*'(, )0'}"*(i%5+2))[8:],str(g),1)for i in range(40)],key=len)))


%%writefile task091.py
p=lambda g,k=46:~k*g or p([*zip(*g[(5in g[k|-2])-2::-1])],k-1)


%%writefile task092.py
p=lambda g:[*map(F:=lambda*l,c=0:[l.count(e)>1and(c:=c^e)or e for e in l],*map(F,*g))]


%%writefile task093.py
import re
p=lambda g:g[57330:]or eval(re.sub("[^05],([0, ]*5)",r"\1,5",f"{*zip(*p(2*g)[::-1]),}"))


%%writefile task094.py
import re
p=lambda g,x=0:eval(re.sub("8(?=[^(]*+[^)]*1.{46}1, 1)","6",f'{*zip(*x or p(g,g)),}'))


%%writefile task095.py
p=lambda g:g[99:]or[eval("r.pop()|r[-1]%4,"*9)for*r,in zip(*p(g*2)*2)]


%%writefile task096.py
import re
def p(i,T=range):
 A=re.sub(', ','',str(i+[*zip(*i)]));D=A+A[::-1];i=int(max(A,key=A.count));B={0:(0,i)}
 for A in T(10):
  if(A^i)*(E:=re.findall(f"{A}+",D)):C=len(re.findall(f"{A}{A}([^]){A}]+){A}|$",D)[0]);B[len(max(E))*(1+(C>0))+C>>1]=-~C>>1,A
 A=max(B);return[[i*((F:=B[max(abs(A),abs(C))])[0]>min(abs(A),abs(C)))or F[1]for A in T(-A,A+1)]for C in T(-A,A+1)]


%%writefile task097.py
p=lambda i,*w:i*0!=0and[*map(p,*sum([[x,x[1:]+[i*4],[i*4]+x]for x in[i,*w]],[]))]or(i in w)*i


%%writefile task098.py
p=lambda i,*w:i*0!=0and[*map(p,i,i[:1]+i,i[1:]+i,*w)]or i^min(w)


%%writefile task099.py
p=lambda g,n=0:[g:=[r.pop()or 1in g!=x>0and~-sum({*g})for x in g]for r in(n or p(g,g))[::-1]]


%%writefile task100.py
p=lambda g:[max(y*[r.count(y)*c.count(y),y]for r in g for*c,y in zip(*g,r))[-1:]*2]*2


%%writefile task101.py
def p(n):
 E=enumerate;D={C+A*1j:D for(A,B)in E(n)for(C,D)in E(B)};C,A=({A for A in D if D[A]&B}for B in(1,2))
 for B in A:C|=A&{A+B for A in C for B in(1,1j,-1,-1j)}
 H,*B=C&A;F=lambda t:{I+(C-H)*B+A*1for C in t for A in range(B)}
 for B in(3,2,1):
  for I in A-C:
   for G in(J:=F(C&A))<A and F(C-A)&{*D}or():A-=J;n[int(G.imag)][int(G.real)]=1
 return n


%%writefile task102.py
p=lambda g,k=-11,i=1,q=0:k*g or[[q:=[i:=i<<9,2&132132>>c%511,5,c|q,0][k*6%64%6+c%~c]for c in g]for g[::-1]in zip(*p(g,k+1))]


%%writefile task103.py
p=lambda g:[[g==g[::-1]or 7]]


%%writefile task104.py
p=lambda g:[[*[u%4]*4+[u%5]*4,0][::1|1-g[1][2]]for u in b"####0000("][::1|1-g[2][1]]


%%writefile task105.py
p=lambda g,i=11,k=0:-i*g or[[c*(k:=b|(1in r))or(sum(map(bool,r))-b>2>i%6)*2for c in r]for r in zip(*p(g,i-1)[::-1])if[b:=k]]


%%writefile task106.py
p=lambda i,s=[],k=3:-k*i or p([*zip(*i+s)],i[::~0],k-1)


%%writefile task107.py
def p(i):z=len({*str(i)})-5;r=range(5*z);return[[i[x//z][y//z]or(x-z*0**i[0][1]in[u:=y-z*0**i[1][0],z*2+~u])*2for y in r]for x in r]


%%writefile task108.py
p=lambda a:a>a*0!=0and[p(a[1])]*4+p(a[2:])or a


%%writefile task109.py
p=lambda a,s=0:a*0!=0and(b:=[*map(p,a,[a[l:=len(a)//2]]*l)])+b[::~0]or a%~a&s


%%writefile task110.py
p=lambda g:[[*map(max,*(r*max(-a^-b for a,b in zip(r,t))+t for t in g))]for r in g]


%%writefile task111.py
p=lambda g:[(l:=sum(g,g))[l.index(5)+n:][:3]for n in b"	"]


%%writefile task112.py
p=lambda g:exec('g[:]=zip(*map(M:=max,g,(g*3)[2*g.index(M(g,key=M))-~len(g)::-1]));'*2)or g


%%writefile task113.py
p=lambda g:g[:5]+g[4::-1]


%%writefile task114.py
*Z,p=0,lambda g:[Z+g[0]+Z,*[r[:1]+r+r[-1:]for r in g],Z+g[-1]+Z]


%%writefile task115.py
p=lambda g,F={}.fromkeys:[*F(zip(*zip(*map(F,g))))]


%%writefile task116.py
p=lambda o:o[::-1]+o


%%writefile task117.py
p=lambda g,n=-79:n*g or p([*zip(*[g,h:=[*map(max,g,i:=(g*3)[n%-21::-1])]][40in map(str(i+6*h).count,str(h))])],n+1)


%%writefile task118.py
import re
p=lambda o,n=23,d=2:-n*o or p(eval(re.sub(["(?<=[^05], )5(?=[^)]{,8}2)","5(?=.{,%d}[^05], [^05].{%d}[^)05]|..{%d}[^)05]{%d}|[^05]{%d}..( |0|.{15}8, 8))"%(3*d,3*len(o)-2,3*len(o)-2-3*d,3+6*d,6*d)][n<12],"8",f"{*zip(*o[::-1]),}")),n-1,d|("p"in re.sub(20*"[^05]","p",f"{o}")))


%%writefile task119.py
import re
p=lambda g:exec('g[::-1]=zip(*eval(re.sub("0(?=.{40}[38].{40}[238])","3",str(g))));'*40)or g


%%writefile task120.py
p=lambda i,*w:i*0!=0and[*map(p,i,[w]+i,i[1:]+[w],*w)]or[8,i]["0"in str(w)]


%%writefile task121.py
p=lambda g:(a:=(g:=(g:=sum(g,[]))[g.index(8)-14:])[:3],(g[13],max(a),g[15]),g[26:29])


%%writefile task122.py
p=lambda g:'3, 0'in'%s'%max(g)and[*map(p,g)]or min(g,g[4:])[:2]+g[:-2]


%%writefile task123.py
p=lambda g:[[(B:=g[0][:4+all(g[0])]*3)[A]]*A+B[A:10]for A in range(10)]


%%writefile task124.py
p=lambda i:i[9:]and i or p(i+[((w:=i[4]!=i[1])*[0for k in(1,2)if[0]*k+i[0]>i[2]]+i[w-3])[:10]])


%%writefile task125.py
p=lambda g,i=87,q=8:g*-i or[[[12%c,q//c*c,-(q&(q:=c)%3)%5][i//42]or c for c in g]for g[::-1]in zip(*p(g,i-1))]


%%writefile task126.py
p=lambda i:i[:-1]+[[4*(0<sum(a)in a)for a in zip(*i)]]


%%writefile task127.py
p=lambda g:g*-1and g+5or g and[p(g[1])]*3+g[3:4]+p(g[4:])


%%writefile task128.py
p=lambda g:[*zip(*map(lambda*c:c[c.count(c[-1]):]+c,*g))]


%%writefile task129.py
p=lambda _:3*[[max(A:=sum(_,_),key=A.count)]*3]


%%writefile task130.py
p=lambda g:g*(g!=5)and(g*-1*-1or[max(map(p,g[:3]))]+p(g[3:]))


%%writefile task131.py
p=lambda g:exec("c=2;g[:]=zip(*([j for*j,in g if(c:=c-max(j)*(c>0))|max(j)]+[[8]*9]+g[:1]*99)[len(g)-1::-1]);"*4)or g


%%writefile task132.py
p=lambda g,k=0:[g:=[[k|(k:=k^max({*h}&{*r}))for h in g]for r in zip(*g)]for _ in g][1]


%%writefile task133.py
def p(a):
 *E,A={B*66+C:A for(B,A)in enumerate(a)for(C,A)in enumerate(A)if A},
 for B in A:C={B};E=[A for A in E if A==A-{B-66,B-1}or C.update(A)]+[C]
 for F in E:
  for B in F:
   for C in F:
    D=F
    for D in 1//len([C for C in D if A[B]==A[C]])*E:
     for H in F-{C}:
      for G in[D for D in D if A[B]==A[D]==A[C]]:G+=(len([D for D in D if A[D]==A[C]])^6)%6*(H-C);a[G//66][G%66],={A[B]for B in D}-{A[B]}
 return a
 return a


%%writefile task134.py
p=lambda g,D=10:[[D*(D!=v>0)for v in r]for r in('0, %g, 0'%D in'%s'%g)*g]or(h:=[*filter(any,zip(*p(g,D-.5)))])[::len(h)//3]


%%writefile task135.py
p=lambda i:[a[6:]for a in i[:3]]


%%writefile task136.py
import re
p=lambda g:eval([g:=re.sub('0(?=(.{34}%s){2})'%x,x,str(g)[::-1])for x in"21"*8][-1])


%%writefile task137.py
def p(g):*_,(a,A),b=sorted(zip(g,R:=range(len(g))));C=max(a);return[[C*(max(y-A,A-y,abs(x-a.index(C)))%(A-b[1])==0)for x in R]for y in R]


%%writefile task138.py
p=lambda g,k=35:-k*g or p([[g:=[e,g][l[0]==g>e<1<9>k]for e in l[::-1]]for*l,in zip(*g[0in g[0]:])],k-1)


%%writefile task139.py
p=lambda g:[[L[0]or(r>g[0]<L[-max(r[5:])%9:][:5])*7for*L,in zip(r,*g)]for r in g]


%%writefile task140.py
p=lambda s:[s[::-1]for s in s[::-1]]


%%writefile task141.py
p=lambda g,i=1:[[*map(max,r[:(a:=abs(g.index(m:=max(g))+(i:=i-1)))]+m,m[a:]+r,r)]for r in g]


%%writefile task142.py
p=lambda r:[r+r[::-1]for r in r+r[::-1]]


%%writefile task143.py
def p(g,i=1):
	*R,=G=b'%r'%g
	for k in b'"%(BEH':R[k+i]=(48%G[k]or-5)%G[k+i]+5
	return{*G}>{*R}and eval(bytes(R))or p(g,i+1)


%%writefile task144.py
p=lambda g,u=[]:g*0!=0and[*map(p,g,u+g[5:])]or 3>>g+u


%%writefile task145.py
p=lambda x,k=7,v=1:-k*x or p([[a:=b&2or[b//max(f:=sum(x,r))+(min({*f}-{2})==b)*8,b%511*4,*[a&~2|b]*5,v:=v*512][k]for b in[2]+r][:0:-1]for*r,in zip(*x)],k-1)


%%writefile task146.py
p=lambda l:(s:=l[:3])*(s!=[*map(list,zip(*s))])or p(l[3:])


%%writefile task147.py
p=lambda i,*w:i*0!=0and[*map(p,i,[i*2]+i,i[1:]+[i*2],*w)]or(3in w)+7&i*9


%%writefile task148.py
p=lambda g,X=0:[[s:=0,X:=-any(r)&1+X%6,[332%(2+sum(r,-s)*(s:=s+v))or(X in g)*8for v in r],g:=g+s//8*[X]][2]for r in g]


%%writefile task149.py
p=lambda g:g[3:]and[p([*zip(*g[i:i+3])])for i in[0,4,8]]or sum(b"%r/"%g)%5


%%writefile task150.py
p=lambda i:[a[::-1]for a in i]


%%writefile task151.py
def p(j):e=j.index(s:=max(j));u=j[e+1]=j[e-1];s[k:=u.index(max(u))-1]=s[k+2]=4;u[k:k+3]=4,4,4;return j


%%writefile task152.py
p=lambda g:g*-1*-1or[*map(p,g+g[::-1])]


%%writefile task153.py
T=0,1,2;p=lambda g:max(all(sum(G:=[[g[x+i%7][y+i%8]^g[x-i%9][y-i%11]for y in T]for x in T],G))*G for i in range(5544))


%%writefile task154.py
p=lambda g:g[225:]or[(r,r[8-r[3]::-1]+r[9-r[3]:])[5in r[:3]]for r in zip(*p(g*2)[::-1])]


%%writefile task155.py
p=lambda j:j[::-1]


%%writefile task156.py
import re
p=lambda g:eval(g:=re.sub("(?<=4.{34}4)(?=.{34}4(.*0.{31}(4))?)",r"*(X:=g.count)('X\2X')//X('+')+1",str(g)))


%%writefile task157.py
def p(n):p,i,*r=range,15;m=sum(n,[]);n=[[]];[(n:=[n+[(f,r)]for n in n for f in p(45)if m[f]<1],r:=[])for f in p(16)if r==(r:=r+[p for p in p(f,150,i)if(i>f)&m[p]])>[]];return max([*zip(*[((any(p-f+min(n)in n*(f%i-p%i<6)for f,n in n)|m[p]%5)%3for p in p(150))]*i)]for n in n)


%%writefile task158.py
def p(e):F=range;N,i=len(e),len(e[0]);o=max(e[0],key=e[0].count);g=e,e[::-1];f,=[t for a in g for l in F(N-2)for h in F(i-2)if 3<len(set(t:=[a[l+k//3][h+k%3]for k in F(9)]))and t[0]!=t[8]];[(t:=lambda a=o,e=o,c=o,*_:[a]*n+[e]*n+[c]*n)(t(f[0]),t(),t(c=f[8]))!=[a[l+k][e]for k in F(3*n)]or exec(f'for k in F(3*n):a[l+k][{e}]=t(*f[k//n*3:])')for n in F(4)for s in(1,-1)for l in F(N-3*n+1)for h in F(i)for a in g for e in[slice(h,[h+3*n*s,None][h+3*n*s<0],s)]];return e


%%writefile task159.py
p=lambda g,*G:[*zip(a:=[2]*99,*[r for*r,in G or p(g,*g)for c in g[::3]if c.count(2)==2if{*r}-{0,2}],a)]


%%writefile task160.py
import re;p=lambda i,*n:eval(re.sub("1.{5}1(.{25})??"*3,r"0,2,0\1 2,2,2\2 0,2,0",str(n or p(i,*i))))


%%writefile task161.py
p=lambda m:[[4//(C:=sum(m,m).count)(x)*x|4//C(i)*i for i in m[0]]for x,*_ in m]


%%writefile task162.py
import re;p=lambda g,k=0:eval('1,1,1'.join(re.split(("(.{55})0, 0, 0"*3)[7:],str(k or p(g,g)))))


%%writefile task163.py
R=range(11);p=lambda g:[[max(5*(g[y][x]==5)or(g[p%3*4+y//4][p&-4|x//4]==4)*g[p%3*4+y%4][p&-4|x%4]for p in R)for x in R]for y in R]


%%writefile task164.py
p=lambda i:[n+n[::-1]for n in i]


%%writefile task165.py
p=lambda g:[*zip(*map(lambda*r:r[:-(k:=r[::-1].index(max(r,key=sum(g[::-1],g).index)))]+k*(max(r[-k:]),)+r,*g))]


%%writefile task166.py
p=lambda g:[[e+(c>[e]*99<l)*2for*c,e in zip(*g,l)]for l in g]


%%writefile task167.py
p=lambda i:[[5*(y==x%len({*str(i)})%3)for x in b'']for y in(0,1,2)]


%%writefile task168.py
import re
p=lambda g:eval(re.sub(r'0(?=(.{35})+.(.[^0]).{27}\2,\2)',r'\2',f"{*zip(*g[70:]or p(g*2)),}"))[::-1]


%%writefile task169.py
p=lambda g,k=11,l=6:-k*g or p([(a:=1)*[a:=[c%7,a|c|(l:=l*8)][k>0<c]for c in r][::-1]for*r,in zip(*g)],k-1,0)


%%writefile task170.py
f=lambda t:t[exec("t[:]=zip(*t[sum(t[0])in t[0]:][::-1]);"*82):]
p=lambda g:eval(f"[[r&n%~n\nfor r,n in zip(r,n[::{-~len(n:=f(f(g)[4:]))//len(r:=f(g[:4]))}])]#"*2)


%%writefile task171.py
p=lambda g:g*all(g[0])or p([*zip(*g[:0:-1],[8]*9)])


%%writefile task172.py
p=lambda s:s+s[::-1]


%%writefile task173.py
def p(g):i=range;h,w,a=i(len(g)-2),i(len(g[0])-2),i(9);[exec('for a in a:g[i+a//3][x+a%3]=r[a]')for i in h for x in w for d in h for t in w if((r:=[g[d+a//3][t+a%3]for a in a])==r[::-1])*r[4]*sum(r[:4])*(r[4]==g[i+1][x+1]or sum(g[i+a//3][x+a%3]==r[a]for a in a)>7)];return g


%%writefile task174.py
p=lambda g,i=1:eval(f'(g==g[::-1])*(g:=[x for x in zip(*g)if {i}in x]),'*4)[3]or p(g,i+1)


%%writefile task175.py
p=lambda g:[g:=[y|z or x for x,y,z in zip([0]+g,r,c)]for*c,r in zip(*g,g)]


%%writefile task176.py
p=lambda g,r=5:[[x|~r*(r:=r-x)%6for x in s]*(r:=1)for s in g]


%%writefile task177.py
p=lambda g,*G:[*filter(any,zip(*G or p(*g)[::-1]))]


%%writefile task178.py
p=lambda g:g*-1*-1or[p(g:=r)for r in g if g!=r]


%%writefile task179.py
p=lambda g:[*zip(*g)]


%%writefile task180.py
p=lambda a:[p(b)for*b,in map(zip,a,a[4:])]or max(sum(a+a,())[1:],key=bool)


%%writefile task181.py
def p(g):
	for l in g[:3]:x=6>>g[3][3];l[x:x+3]=l[5:2:-1]
	return g


%%writefile task182.py
from re import*
p=lambda g,h=0:eval(sub(sub("2|3","1","(.{47})".join(a:=findall(".*5, "+"5.{46}(.{15})"*5,s:=str(h or p(g,g)))[0])),"\%d ".join(a)%(1,2,3,4),s))


%%writefile task183.py
p=lambda g,h=0:g*0!=0and[*map(p,len(g)//2*g[:1]+g[-1:]*9,h or g)][2:-2]or h%7*g


%%writefile task184.py
p=lambda g,*h,q=[]:[q+0*(q:=r)for*r,in zip(*h or p(*g))if[0,*r,q:=[*map(max,q+r,r)]]>r]+[q]


%%writefile task185.py
p=lambda g,*P:[g:=[[x*(P==(P:=x)!=max(Z))for c,x in zip(g,r)if{*c}-{*Z}][1:]for r in zip(*g)]for Z in g][1]


%%writefile task186.py
p=lambda g:[[2,1%(n:=sum(sum(g,[])))*2,2%n],[0,6%n,0],3*[0]]


%%writefile task187.py
p=lambda g,i=7:g*-i or[[i:=c|2>>-i*c%7for c in[3]+r][:0:-1]for*r,in zip(*p(g,i-1))]


%%writefile task188.py
p=lambda g:(X:=g[:53%~-len(g)])*(g==X+X)or[*map(p,g)]


%%writefile task189.py
p=lambda g,h=[]:g*0!=0and[*map(p,3*[g[i:=('0'in'%r'%g[2])*7]]+3*[g[i+1]],(h+g)[3>>i:])]or h%2*g


%%writefile task190.py
import re;p=lambda i:exec(r"i[::-1]=zip(*eval(re.sub('.{31}0, ([^0])'*2,r'|\1\g<0>',str(i))));"*20)or i


%%writefile task191.py
from re import*
def p(l):
 X=[0]*U;r=*l,=X,*zip(X,*l,X),X
 for e in[1,1,-1]*8:l[::e]=zip(*eval('1'.join(split(sub('1',')0(',sub('[^1]+'*S,lambda l:'.'*len(l[0]),str(r))).strip('(.)'),'%s'%l))))
 return[*zip(*l[1:e])][1:e]


%%writefile task192.py
p=lambda g,k=3:g*-k or[[k:=[c:=r.pop(),k][[0,*r][-1]in[k,0]*c]for _ in r*1]for*r,in zip(*p(g,k-1))]


%%writefile task193.py
p=lambda i,*w:i*0!=0and[*map(p,i,[i]+i,i[1:]+[i],*w)]or(w.count(i)>1)*i


%%writefile task194.py
p=lambda i,s=[],k=3:-k*i or p([*zip(*i+s)],i[::~0],k-1)


%%writefile task195.py
p=lambda a,n=1:[[y&sum(a*3,())[n:=n+3]for y in x]for x in-n*a]or p([*filter(any,zip(*a))],n-1)


%%writefile task196.py
p=lambda g,i=19:g*-i or[*map(lambda*r,q=8:[q:=[c|8-c&q,c|-c&~q%3,c%4][i//8]for c in r],*p(g,i-1)[::-1])]


%%writefile task197.py
p=lambda g:[[*map({}.setdefault,g[1],A)]for A in g]


%%writefile task198.py
import re;p=lambda g:exec("g[::-1]=zip(*eval(re.sub('(?=0|3, 4|3[^)]*[^)34]{6})','6^5-',str(g))));"*24)or g


%%writefile task199.py
p=lambda a:-~(y:=a.index(r:=max(a)))*[([4,0]*8)[r<r[1::2]:][:len(r)]]+a[y:-1]


%%writefile task200.py
p=lambda g,c=5:[(g[9].index(m:=max(g[9]))*[0]+[m,c,m,c:=any(r)*5]*9)[:10]for r in g]


%%writefile task201.py
def p(g):f=1;m=[R for r in zip(*g)if any(R:=[x*f*(f:=f^(x==4)or(g:=g+[x])>g)for x in r])];return[z:=[4,*[0]*len(m),4],*[[a:=g[15],*r[::a in m[0]or-1],g[-2]]for*r,in zip(*m)if any(r)],z]


%%writefile task202.py
p=lambda g:exec("v=p,;g[:]=zip(*[map(min,[i,v][len({*v,*i,0})<3],v:=i)for*i,in g[::-1]]);"*36)or g


%%writefile task203.py
p=lambda i:[[i[l:=len(i)//2][j.index(a)-l]for a in j]for j in i]


%%writefile task204.py
import re;p=lambda i:eval(re.sub("(?<!1, )1,(.+?)1",r"1,*[(s:=len([\1]))%2*5+2]*s,1",str(i)))


%%writefile task205.py
p=lambda g,k=87:-k*[[min(e+u,key=e.count)for*u,in zip(*g)]for*e,in g]or p([*zip(*g)][sum(u[:6].count(u[0])>4for u in g)<6:][::-1],k-1)


%%writefile task206.py
def p(g):
 A=sum(g,[]).index(5);B=len(g[0]);C=lambda h:[A for A in zip(*h)if{*A}-{0,5}]
 for g[A//B-1][A%B-1:A%B+2]in C(C(g)):A+=B
 return g


%%writefile task207.py
p=lambda a:[p(b)for*b,in map(zip,a,a[3:])]or min(b:=sum(a,()),key=b.count)


%%writefile task208.py
from re import*
p=lambda g:eval((i:=min(k:=str(g)+'#[]'*X,key=k.count)).join(split(sub(i,f')[^{i}](',sub("[^%s]+"%i*18,lambda x:'.'*len(x[0]),k)).strip(".()"),k)))


%%writefile task209.py
def p(i,T=enumerate):
 B=i
 for A in B*4:B=[[*A]for A in zip(*B[~4:])if{*A}-{0,4}];i=[[*A]for A in zip(*i[(4in i[-1])-2::-1])]
 for(A,C)in T(i):
  if[0for(D,C)in T(i)for(E,C)in T(A*C)for(F,C)in T(A*all(C in[0,((B+20*[[]])[(F-D)//A]+20*[4])[(G-E)//A]]for(F,C)in T(i)for(G,C)in T(C))*B)for(G,C)in T(A*C)for i[F+D][G+E]in[B[F//A][G//A]]]:return i


%%writefile task210.py
p=lambda e:e+e[::-1]


%%writefile task211.py
p=lambda g:[r[::-1]+r for r in(g[::-1]+g)*2][:9]


%%writefile task212.py
p=lambda i,k=18:~k*i or[[x.pop()or[0,*x][k%-2]%5&6-(5in x)for _ in i]for*x,in zip(*p(i,k-1))]


%%writefile task213.py
p=lambda g:[l[o:]for r in g if(l:=[e for e in r if e%5])[~(o:=6-len({*"%s"%g})):]][o:]


%%writefile task214.py
p=lambda g:[r+g.pop()[3::-1]for r,*r[4:]in zip(g*1,*g[::-1])]


%%writefile task215.py
p=lambda g:[*map(max,g,g[3:6]*9,g[6:9]*9)]


%%writefile task216.py
p=lambda g:max([sorted(sum(v:=[k[i%19:i%23]for k in g[i%21:i%22]],[]),key=1 .__eq__),v]for i in range(4**9))[1]


%%writefile task217.py
p=lambda g:[*filter(any,zip(*g[81:]or p([[c&d for c in a for d in b]for a in g*2for b in g])))]


%%writefile task218.py
p=lambda g,*h:[*{r:0for r in zip(*h or p(*g))if any(r)}]


%%writefile task219.py
def p(o):
 C=B=A=0
 while o[A:]:
  if o[A]>o[0]:C=C or A;B=B or o[A:].index(o[0]);o[A-1:A+B]=[E for D in(0,-1,1,2)for F in(C-1,C)if 7not in sum((E:=[[B%7-2%~A for(A,B)in zip(B,A[:D%4%3]+A[D<0:]+[0])]for(B,A)in zip(o[A-1:A+B],o[F:])]),[])][0];A+=B
  A+=1
 return o


%%writefile task220.py
p=lambda g,i=3:g*-i or p([[r.pop()or[0,*r][-1]**4%84%15for r in g]for r in g],i-1)


%%writefile task221.py
def p(g):w=sum(g,g).count(i:=0);return[(r*(9+(i:=i-1)//3*w)+[0]*99)[:3*w]for r in g*w]


%%writefile task222.py
p=lambda g:[g:=[[c*(str(r*7+g).count(2*f"{c}, ")>9)for c in r]for*r,in zip(*g)]for _ in g][5]


%%writefile task223.py
p=lambda g:g*-1*-1or g and[p(g[0])]*3+p(g[1:])


%%writefile task224.py
p=lambda g,i=21:-i*g or[g[:(b:=any(g[0])*i<8)],[[max(max(g))]*99]][i<4]+[*zip(*p([*zip(*g[b:][::-1])],i-1))][::-1]


%%writefile task225.py
p=lambda a,n=0,R=range(6):a[i:=n//6][j:=n%6]and[[a[y][x]+(x-j&y-i&2>0)*a[i+(y<i)][j+(x<j)]for x in R]for y in R]or p(a,n+1)


%%writefile task226.py
p=lambda g,k=7,r=range(10):g*-k or p([(q:=0)or[q:=g[i][~j]or[~i//4*-(j^9-i<2&~j),q%5][k<7]for i in r]for j in r],k-1)


%%writefile task227.py
p=lambda g,u=[]:g*0!=0and[*map(p,g,u+g[4:])]or~g+u&2


%%writefile task228.py
import re
p=lambda g:eval(re.sub(r'([^0])((, (?!\1|0).).*0\3.{28})0',r'0\2\1',f'{*zip(*g[70:]or p(g*2)),}'))[::-1]


%%writefile task229.py
p=lambda g:[[(5,v)[v==max(l:=sum(g,g),key=l.count)]for v in r]for r in g]


%%writefile task230.py
p=lambda g,i=6:g*(i<3)or p([[(d:=r.pop()or[0,*r][-1]*i)>>5*(d>5>4>i)for r in g]for _ in g],i-1)


%%writefile task231.py
p=lambda g:[(r[:6]*2+r*2)[:-12]for r in g]


%%writefile task232.py
p=lambda i,e=0:i*0!=0and[p(y)or[e:=y-e,5][e<0]for y in i]


%%writefile task233.py
def p(r,T=enumerate):
 for A in 92*[r]:r=[[*A]for A in zip(*r[('2, '*4in str(r[-1]))-2::-1])]
 for A in 9*[[A[D:3+D]for A in A[B:3+B]]for(B,C)in T(A[2:])for(D,C)in T(C[2:])][::-1]:
  for(D,B)in T(r*({*sum(A,[])}^{0}>{2,0})):
   for(C,B)in T(B):
    for(E,B)in T(A*all((2*(2*r)[B+D])[E+C]==2*(2!=A)for(B,A)in T(A)for(E,A)in T(A))):r[E+D][C:3+C],*A=B,
  A[:]=[[*A]for A in zip(*A[::-1])]
 return r


%%writefile task234.py
p=lambda g:exec('c={0};g[:]=zip(*(g[:1]*99+[j for*j,in g if(c:=c|{*j})-{sum(j),0}])[:~len(g):-1]);'*4)or g


%%writefile task235.py
p=lambda g:[[g[1][x]*sum(g[2][x:x+3])%13^8]*3for x in b'']


%%writefile task236.py
p=lambda g,u=[]:g*0!=0and[*map(p,g,u+g[5:])]or-u%5^g*3


%%writefile task237.py
p=lambda g,P=0:[[*[(P:=P or x for x in r+[P])][P:=0]]for*r,_ in g]


%%writefile task238.py
def p(g,s='[[8,*r,8]for r in zip(*%s)if 8in r]'):R=range(l:=len(B:=eval(s%s%g)));return[[B[i][j]*[[*{c/8:0for c in sum(g,[])if c&7}][(i>j)+(~i+l<j)*2],0<i<l-1][i in(j,~j+l)]for j in R]for i in R]


%%writefile task239.py
p=lambda g:[s:=sum(g,[]),*filter(any,zip(*sorted([c:=-s.count(e)]+[e]*-c+[0]*99for e in{*s})))][2:]


%%writefile task240.py
k=1;p=lambda g:exec("k+=2;g[i:=k%19][j:=k%18]|=g[i][j-2]*(i<j-2<16-i)|g[j][~i];"*746)or g


%%writefile task241.py
p=lambda g:[*zip(*g)]


%%writefile task242.py
p=lambda r:[r[~r.index(0)::-1][:3]for r in r if 0in r]


%%writefile task243.py
p=lambda g:exec('g[::-1]=zip(*eval(str(g).replace("1, 0","1,1")));'*80)or g


%%writefile task244.py
p=lambda g,w=2:[[0,max,p][w](g:=r,-2)for r in g if g!=r][::w]


%%writefile task245.py
p=lambda i,k=7:-k*i or p([*map(lambda*r,w=0:[[w%3,w:=v][v%2+5>sum({*max(i)})]for v in r],*i)],k-1)


%%writefile task246.py
p=lambda g,n=3:-n*g or p([[r.pop()|(n|2in r!=3-n%2in(g:=g.pop()+g))*8for _ in r*1]for*r,in zip(*g)],n-1)


%%writefile task247.py
p=lambda a,m=9:[*zip(*{(d,)*m:0for d in sum(zip(*a),())if sum(a,a).count(d)==m})]or p(a,m-1)


%%writefile task248.py
def p(g):
 A=B=0
 for C in g[::-1]:C[B]=1;A^=-C[A];B-=A|1
 return g


%%writefile task249.py
p=lambda i:[n*2for n in i]


%%writefile task250.py
p=lambda a:[a:=[sorted(b[:(i:=str(a).find('2')>>5)])+b[i:]for*b,in zip(*a)][::~0]for _ in a][3]


%%writefile task251.py
p=lambda g,i=31:g*-i or p([[r.pop()&~4**[0,*r][-1]or i>30for r in g]for _ in g],i-1)


%%writefile task252.py
p=lambda g,v=0:g*0!=0and[*map(p,g,[-1,4]*9)]or-g//v&v


%%writefile task253.py
p=lambda g:[[(i:=a&b)*0+max(v*(sum(g,g)[i-1:(i:=i+1)]==[v,v])for v in sum(g,[]))for a in b'[Z']for b in b'A[']


%%writefile task254.py
p=lambda g:[[(sum(g[-sum(c)//5])|5)%sum(g[8])*x%3for*c,x in zip(*g,r)]for r in g]


%%writefile task255.py
import re;p=lambda g,k=9:eval(re.sub(*["\((%s|(0, )+3.{5}3),?"%re.search(r"([ ,03]{61,})(.*\1){3}|$",g:=f'{*zip(*~k*g or p(g,k-1)),}')[1],"(?=0((,[^,]*){31}|, )[1-9])",r"(*[3]*len([\1]),","1<"][k<2::2],g))[::-1]


%%writefile task256.py
def p(g):T=sum(V:=max(g))//2;S=T-~g.index(V);return[[((S:=0-2%~S)-T|8)//9+2]*S+r[S:]for r in g]


%%writefile task257.py
p=lambda a:[p(b)for*b,in map(zip,a,a[5:])]or max(sum(a,()),key=bool)


%%writefile task258.py
import re;p=lambda g:eval(re.sub('1, 0(?=, 1)','1,2',str(g)))


%%writefile task259.py
p=lambda g:exec('g[:]=zip(*eval(str(g).replace(*"10"))[any(g[-1])-2::-1]);'*24)or g


%%writefile task260.py
exec("p=lambda a:[[max({*max(a)}-{5})*any(a[i][j]%5or 2==sum(m-n-i+j+k%5==2<a[m][n]"+'for %s in range(10)%s'*6%(*'m n k)_)j]i]',))


%%writefile task261.py
p=lambda g:[[r%3for r in r]for r in[g.pop()]+g]


%%writefile task262.py
p=lambda g:[3*[c^6&b+~a]for a,b,c in g]


%%writefile task263.py
p=lambda g,h=0:-(M:=bytes(map(bool,sum(g:=[*zip(*h or p(g,g))],())))).find(M[:9],9)*g[:3]or p(g[3:]+g[:3])


%%writefile task264.py
T=3,2,1;p=lambda g:[sum([eval(f"sorted([0in(S:=sum(g,T)),[i!=5for i in S],*g]{'for*g,in map(zip,g,g[1:],g[2:])'*2})#{g}")[42>>Y&7*~-X^Y][-x]for Y in T],())for X in T for x in T]


%%writefile task265.py
import re
p=lambda g:eval(re.sub('0(?=.{949,952}(.{56})?0(?!.{37}0.{485}]), 0.{52}0, 0)','2','%r#'%g*2))


%%writefile task266.py
p=lambda g:g[9:]or[[(8|9>>c|a*6)%9for a,c in zip([0]+r,r[1:]+[0])]for*r,in zip(*p(g*2))]


%%writefile task267.py
p=lambda g:[[g[6][r>[x]]for x in r]for r in g]


%%writefile task268.py
import re
p=lambda g,i=7:-i*g or p(eval(re.sub("0(?=%s)"%["(.%r0)*, [^0].%%r[^0], 4"%{o:=len(g)*3+4}%{o-6},r"[0, ]++(.).{,3}\).*#.*\1, \1, [0, ]+\1"][i>3],"4",f'{*zip(*g),}#{g}'))[::-1],i-1)


%%writefile task269.py
p=lambda g:eval("[[g\nfor g in g for _ in[*{*'%r'}][5:]]#"%g*2)


%%writefile task270.py
import re
p=lambda g:[g:=eval(re.sub(f'{n}([^(]*)0(?=, {8%n})',r'0\1n',f"{*zip(*g[::-1]),}"))for n in(7,3,3)*8][-1]


%%writefile task271.py
exec(f"p=lambda g:max([str(g).count('1'),g]{'for*g,in map(zip,g,g[1:],g[2:])'*2})[1]")


%%writefile task272.py
p=lambda i,*w:i*0!=0and[*map(p,i,[i*2]+i,i[1:]+[i*2],*w)]or~(2in w)*i%3


%%writefile task273.py
p=lambda g,*r,k=0:[c+(k:=k^c)%6and c^2for c in r]or[*map(p,g,*map(p,g,*g))]


%%writefile task274.py
p=lambda g:[[8,(s:=sum(map(max,g))*6)+4&8,-s&8],[0,0,~s&8],[0]*3]


%%writefile task275.py
def p(g):R=len(g+g[0])//3;r=range(-R*R,0);return[[sum(g[u%R-t][v%R-t]//8*g[u//R+t][v//R+t]for t in(0,R))for v in r]for u in r]


%%writefile task276.py
p=lambda g:g*-1and-g%6|2or[*map(p,g)]


%%writefile task277.py
z=[0];p=lambda g,k=38,h=2,q=z*9:~k*g or p([q:=[v and[v%63,P|p|v,h:=h*64,v//sum(g,z).count(v)][k>>4]for P,p,v in zip(z+q,z+r,r)]for*r,in zip(*g[::~0])],k-1)


%%writefile task278.py
p=lambda m,k=11:-k*m or p([[[v%9,v|3%-~u,v<<u][k>>2]for u,v in zip([0]+r,r)]for*r,in zip(*m[::-1])],k-1)


%%writefile task279.py
p=lambda g,i=94:g*~i or p([[9&r.pop()%[q+9,9|3-q][i<9]or(i<0)*9for q in[0]+r[:0:-1]]for*r,in zip(*g)],i-1)


%%writefile task280.py
p=lambda g,n=11:-n*g or[[P:=[((a:=(8%~P<x)*a+x%2*8|2)>2>x)*a,-6%(P-2|1-x|x),x&3][n//4]or x for x in r]for r in zip(*p(g,n-1))if[a:=0,P:=0]][::-1]


%%writefile task281.py
p=lambda a,n=47,*P:-n*a or p([*zip(*[max(P*({0,8}in map(set,a)),P:=a.pop(),key=set)for _ in a*1])],n-1)


%%writefile task282.py
p=lambda g:g[99:]or[eval(8*"+(x:=r.pop()),x//5|x^0")for*r,in zip(*p(g*2))]


%%writefile task283.py
p=lambda i,*w:i*0!=0and[*map(p,i,[i]+i,i[1:]+[i],*w)]or-i%8*w.count(5)%5


%%writefile task284.py
p=lambda g:exec("m=max(g)\nfor x in-2%len({*m})*b'osqmnr':(l,c),*_,r=filter(min,enumerate(m,1));s=l-2+r[0]>>1;m[l:s]=-l%s*[c];g[g.index(m)+x%5-2][s-x%2]=c\ng[:]=[y[::-1]for*y,in zip(*g)];"*4)or g


%%writefile task285.py
def p(h,T=range):
 for C in T(8):
  h[::-1]=zip(*h)
  for C in T(len(h)):
   for A in T(len(h)):
    for(B,D)in(E:=[(C,A)]):*h[B],=h[B];h[B][A+A-D-1]=h[C][A-1];E+=[(E+B,D+F)for E in T(-1,2)for F in T(-1,2)if 0<h[C][A-1]!=(2*(2*h)[E+B])[D+F]==h[C][A]>0==h[E+B][A+A-D-F-1]]
 return h


%%writefile task286.py
p=lambda i,k=39:-k*i or[[t:=y or sum({*t%8*sum(i,x)}-{t,8})for y in[8]+x][:0:-1]for*x,in zip(*p(i,k-1))]


%%writefile task287.py
p=lambda*g:g[g[0]==4]*-1*-1or[*map(p,g[-1][::-1],*g)]


%%writefile task288.py
def p(g):x=i=g[-2].count(0)>>1;exec('i-=1;r=g[i-x-2];r[i]=r[~i]=g[-1][x];'*x);return g


%%writefile task289.py
p=lambda g:eval("[[g\nfor g in g for _ in[*{*'%r'}][5:]]#"%g*2)


%%writefile task290.py
p=lambda g,*u:[sum({*u},r*-1)or p(r,*sum(g,r))for r in g if[r]>g]


%%writefile task291.py
p=lambda m,k=1:[*{r.count(k)for r in m},[k]][3:]or p(m,k+1)


%%writefile task292.py
p=lambda g,v=0:g*0!=0and[*map(p,g,b'\n'*7)]or-g%v


%%writefile task293.py
p=lambda g:[[(c in r)**c*r[0]or c for c in g[0]]for r in g]


%%writefile task294.py
import re;p=lambda g:eval(re.sub("(?<=5.{34})5(?=.{34}5)","2",str(g)))


%%writefile task295.py
p=lambda g:[P:=g[0]]+[P:=P[:1]+P[:-1]for x in P[2::2]]


%%writefile task296.py
p=lambda*g:[*map([*g,max,p][2],*[r[-3:]for r in g],*g)]


%%writefile task297.py
p=lambda g:g[:2]+[*zip(*g[:1]*len(g[0]))]*2


%%writefile task298.py
p=lambda g:[[g[2][-r.index(v)|2]for v in r]for r in g]


%%writefile task299.py
p=lambda g:[[(c,2+c/4)[2in r]for c in g[0]]for r in g]


%%writefile task300.py
p=lambda g,*G:[r for*r,in zip(*G or p(g,*g))if max(range(1,10),key=sum(g,g).count)in r]


%%writefile task301.py
p=lambda T,E=sorted:E(map(E,T))


%%writefile task302.py
import re
p=lambda g:eval(re.sub('([^5]..5,)([^5]+)',r'\1*(k:=len([\2]))*[k+5],',str(g)))


%%writefile task303.py
p=lambda g:[[[2,e][c>[0]*99<l]for*c,e in zip(*g,l)]for l in g]


%%writefile task304.py
p=lambda g:[[c*(w==max(p:=sum(g,g),key=p.count))for w in q for c in r]for q in g for r in g]


%%writefile task305.py
p=lambda g:[(9*[*{*r}-{0}])[g.index(r):][:16]for r in g]


%%writefile task306.py
p=lambda g:[g:=[*zip(*map(max,g,g[:10]+g))][::-1]for _ in g][7]


%%writefile task307.py
p=lambda g:g*-1*-1or g and[p(g[0])]*2+p(g[1:])


%%writefile task308.py
def p(e):s,u=len(e[0]),enumerate;e=sum(e,[]);m=max(e:={e.count(r)<9and f-sum(e*(u==r)for e,u in u(e))//4:r for f,r in u(e)})//s;p=range(-m,m+1);return[[e.get(u*s+r,e[0])for r in p]for u in p]


%%writefile task309.py
p=lambda g:g*-1and g&-3or[*map(p,g)]


%%writefile task310.py
p=lambda a,*n:[b for b in zip(*n or p(a,*a))if{*b}-({*a[1]}&{*a[12]})]


%%writefile task311.py
p=lambda i:[n+n[::-1]for n in i]


%%writefile task312.py
p=lambda g:[[x%~x&r[0]for x in r]for r in g]


%%writefile task313.py
p=lambda g,u=[]:g*-1*-1or[*map(p,g[u>[]:2--len(u)//11]*10,g)]


%%writefile task314.py
p=lambda i,*w:i*0!=0and[*map(p,i,i[:3]+i,i[3:]+i,*w)]or max(w[0]&w[1],w[2]&w[3],i)


%%writefile task315.py
p=lambda g:[[y&-x%5for x in r for y in s]for r in g for s in g]


%%writefile task316.py
p=lambda g:[(S:=[*filter(int,map(max,*g))]+[0]*9)[:3],S[5:2:-1],S[6:9]]


%%writefile task317.py
p=lambda g:g==5or g and[p(g[1])]*3+p(g[3:])


%%writefile task318.py
p=lambda g,u=[]:g*0!=0and[*map(p,g,u+g[5:])]or(g!=u)*3


%%writefile task319.py
p=lambda g,x=0:[[[v[3],c][a==c]for c in r]for r in zip(*x or p(g,g))if(a:=(v:=sorted({*(q:=sum(g,[]))},key=q.count))[hash((*b'Q9n7l$hj(6ytfgHBq|^^KH^m"%r'[sum(q)%37:]%q,))%2])in r]


%%writefile task320.py
p=lambda g:[*zip(*map(lambda*r:r[:-(k:=sum(r)//4)]+k*(8,)+r,*g))]


%%writefile task321.py
p=lambda g:[eval("r.pop(0)or r[4]|r[9],"*4)for r in g]


%%writefile task322.py
p=lambda g:g and p(g[:-1])+[[*map(max,*g*2)]]


%%writefile task323.py
import re
p=lambda g:[p,eval][''in g](re.sub('0(?=(.{76})*(.{40}|(...){25,27})8)','5',str(g)[::-1]))


%%writefile task324.py
def p(r,T=enumerate):l,t,i,n=a=sorted({*sum(r,[])},key=sum(r,[]).count);return[[([a[l!=e!=a[~all({l,n}!={*s}!={t,i}for s in[*zip(*r)]+r)]]for z,s in T(r)for u,o in T(s)if o in(l,t)!=abs(u-m)==abs(z-f)]+[e])[0]for m,e in T(s)]for f,s in T(r)]


%%writefile task325.py
p=lambda i,z=8,*w:i[22:]and[[8*(a==b)for b in w]for a in w]or p([[s:=h and(z:=z*2)|s|h for h in[0]+x]for*x,in zip(*i[::~0])],*{*sum(i,[0])})


%%writefile task326.py
p=lambda g:[g[0][:2],g[1][:2]]


%%writefile task327.py
p=lambda g,l=[0]*3:[l:=[*map(max,[0]+l*2,r+[0]*3)]for r in g+[l]*3]


%%writefile task328.py
exec("p=lambda g:[[(D:=sorted((sum(T:=[abs(x-r),abs(y-c)]),~max(T)%2*f[y])"+'for %s,f in enumerate(g)%s'*4%(*'y x','if f[y]))[0][1]*(D[0]<D[1][:1])',*'c]r]'))


%%writefile task329.py
p=lambda i:[(r:=len(i)//2)*[0]+[a[r]]+r*[0]for a in i]


%%writefile task330.py
p=lambda i,k=-19,z=1:k*i or p([[e:=y and[1+y%7//6,z:=z*8,e|y][k>>4]for y in[0]+i][:0:-1]for*i,in zip(*i)],k+1)


%%writefile task331.py
p=lambda g:[g:=eval(f"{*zip(*g[::-1]),}".replace('1, 0','1,'+k))for k in'2786'][3]


%%writefile task332.py
p=lambda g:[[r.pop(0)*-7**len(r)&7for x in r*1]for r in g]


%%writefile task333.py
p=lambda g:[eval("P"+9*",(P:=r.pop()or(3in r)*P)")for*r,P in zip(*g[70:]or p(g*2))]


%%writefile task334.py
p=lambda g:[[i%7,i%6,i%11]for i in b" M~M~MM"[max(max(g))::3]]


%%writefile task335.py
p=lambda g:[g:=[[r.pop()|(n in r!=n^10in(g:=g.pop()+g))*4for _ in r*1]for*r,in zip(*g)]for n in[8,2]*2][3]


%%writefile task336.py
p=lambda g:exec(f"g[:]={'[r.pop()or(5in{*r}-{*r[4:]})*8for r in g],'*10};"*4)or g


%%writefile task337.py
p=lambda g:g*-1and g^84%g%3*13or[*map(p,g)]


%%writefile task338.py
p=lambda g:[(c:=1)*[((c:=-j^-c%7%3)>1)*3for j in i]for i in g]


%%writefile task339.py
p=lambda g:[[*filter(int,sum(g,[]))]]


%%writefile task340.py
p=lambda g:[g:=[[a]*-~(C:=a in r*a)+[c*(c in g[-1]+g[1])for c in r[C:]]for a,*r in zip(*g)][::-1]for _ in g][3]


%%writefile task341.py
p=lambda g:[g:=[[g[i][j]or(9>j>=2<len({*min(g[i-1:][:3])}))*8for i in R]for j in R]for R in[range(10)]*2][1]


%%writefile task342.py
import re
p=lambda g:g[150:]or eval(re.sub("([1-9])((.{32})+?[^)]+?)8",r"0\2\1",f"{*zip(*p(g*2)[::-1]),}"))


%%writefile task343.py
p=lambda g:[(r[:8^-(r[8:12]!=r[:4]!=r[4:8])]*3)[:15]for r in g]


%%writefile task344.py
p=lambda i,*w:i*0!=0and[*map(p,i,[i*4]+i,i[1:]+[i*4],*w)]or(i^1in w)+7&i*9


%%writefile task345.py
p=lambda i,a=0,*h:i[1:]and[[c[1]|2&6%~sum(h)+max((h:=c)[1:])for c in zip(*i)]]+p(i[a:],1)


%%writefile task346.py
p=lambda a:[[min(b:=sum(a[1:-1],a[3]),key=b.count)]]


%%writefile task347.py
p=lambda g:[[6^6>>x+r.pop(3)for x in r]for r in g]


%%writefile task348.py
p=lambda g,G=0,*s:[s:=[r.pop()|-y%15for y in s[1:]]+r for r in(G or p(g,g))[::-1]][::-1]


%%writefile task349.py
p=lambda g,n=15:-n*g or[[P:=[max(x:=r.pop(),x%~x&P,a:=x and a+1198080|9),x or P&8**n*7and~8&P-8**n|3,x&15,x|(n-3in r)][n//4]for _ in g]for*r,in zip(*p(g,n-1))if[P:=0,a:=0]]


%%writefile task350.py
p=lambda i:[*map(f:=lambda*x,s=0:[y|(x.count(1)>(s:=s+y%8)>y<1)*8for y in x],*map(f,*i))]


%%writefile task351.py
p=lambda i:[r[:5]for x in[*i]if(r:=i.pop()[~[*x,3].index(3)::~0])]


%%writefile task352.py
p=lambda g:[g:=[[r.pop()or[0]<r[-1:]<[3]for _ in g]for*r,in zip(*g)]for _ in g][3]


%%writefile task353.py
p=lambda a,n=-3:n*a or p([*zip(*[a.pop(sum(sum(a,r))==10)for r in a*1][::-1])],n+1)


%%writefile task354.py
R=range(10)
p=lambda g:[[max(g[0][A]*all(B[C:A+1]+B[A:C+1])for A in R)for C in R]for B in g]


%%writefile task355.py
p=lambda g:[sorted({*(A:=[(*{*j}&{*a}^{b},)for j in g for*a,b in zip(*g,j)])},key=A.count)[-3]]


%%writefile task356.py
p=lambda g,*r,i=0:[x|max(g[i:])&max(g[:(i:=i+1)])for x in r]or[*map(p,g,*map(p,zip(*g),*g))]


%%writefile task357.py
def p(g,i=9):
 for A in g:B=~-len(A);A[:]=[8]*-~B;A[i%B^i//B%-2]=1;i-=1
 return g


%%writefile task358.py
p=lambda g,*r:[max(r[::6^83>>len({*(r:=(0,)*35+r)})])for _ in r]or[*map(p,g,*map(p,g,*g))]


%%writefile task359.py
p=lambda d:[[max(t:=r+X,key=t.count)for*X,in zip(*d)]for r in d]


%%writefile task360.py
p=lambda i:[[*map(max,n,n[:4:-1])]for n in i]


%%writefile task361.py
def p(g,x=0,i=2):r,c,e=x%9,x%8,3-x//72;h=*zip(*g[::-1]),;return-i*g or any(0in a[c:c+e]for a in g[r:r+e])and p(g,x+1)or p([[*map(max,x,y[10-c-r-e:]+y)]for x,y in zip(g,h[c-r:]+h)],x,i-1)


%%writefile task362.py
p=lambda g:[r[(k:=g.count(g[0])):9]+-~k*r[:1]for r in g*2][~k-9:-k]


%%writefile task363.py
from re import*
def p(g):h=hash((*g[3],));g[~h%7][3]|=h%149<1;return eval(sub(*'10',eval("'2'.join(split(sub('2',')0(',sub('[^2]','.',K:=str(g))).strip('.()'),"*3+'K))))))')))


%%writefile task364.py
p=lambda g,k=95:-k*g or p([g:=[[7&88>>c%7,c|a|all([a,b*-1,k>83])<<k*3][k>0<c]for c,a,b in zip(r,[0]+r,g)]for*r,in zip(*g[::-1])],k-1)


%%writefile task365.py
p=lambda g:max((-(y:=sum(s:=[r[x%9:x%13]for r in g[x%8:x%11]],g).count)(0),y(2),y(1),s)for x in range(5**6))[3]


%%writefile task366.py
def p(e):
 *A,E,D,C=sorted({*sum(e,[])},key=sum(e,[]).count);*A,=e,
 for B in A*6:
  for B in A+(A:=[]):
   A+=[],
   for B in zip(*B):A+=A.pop()+[B]if{D,C}-{*B}>{D}or{*B}>{D}else[],
 [6for(D,F)in sorted((-sum(E^A for A in A for A in A),A)for A in A)for(D,A)in enumerate(e)for(B,A)in enumerate(A)for(G,A)in zip(e[D:],F*all((D==A)!=(E==A)==(D==C)for(F,A)in zip(e[D:]+e,F)for(D,A)in zip(F[B:B+len(A)]+e,A)))for G[B:B+len(A)]in[A]];return[A for A in zip(*[A for A in zip(*e)if C in A])if C in A]


%%writefile task367.py
import re
p=lambda g,k=-19:k*g or p(eval(re.sub(f"0(?=, 4|.{ {N:=len(g)*3+4}}5, 0.{ {N-6}}0)","4",str([*zip(*g)][::-1]))),k+1)


%%writefile task368.py
import re
p=lambda i:eval(re.sub("5(, 5)+",lambda m:re.findall("[^50](?:, [1-9])+",s*2)[-s[m.end()-1::32].count('5')],s:=str(i)))


%%writefile task369.py
p=lambda g,n=7:g*-n or[*map(lambda*r,b=5:[b:=a*(a>4)or[a+b//4-.25,min(a,b)][n>3]for a in r],*p(g,n-1)[::-1])]


%%writefile task370.py
import re
p=lambda g:exec('s=f"{*zip(*g[::-1]),}";i=s.rfind;d=i("0")-i(j:=min(s,key=i));g[:]=eval(re.sub("(?=(.{%d})+0)\d"%d,j,s,d%8*d%-(len(g)*3+5)));'*4)or g


%%writefile task371.py
p=lambda g,u=0:eval((G:=f"{*zip(*u or p(g,g)),}")[:(w:=G.rfind("1")+G.find("1")>>1)-3]+"3,"*3+G[w+5:])


%%writefile task372.py
p=lambda g:eval("[*map(max,g.pop(0),g[5])],"*5)


%%writefile task373.py
p=lambda g:[T:=max(zip(*g))*3,T[::-1]]


%%writefile task374.py
p=lambda g,i=51:-i*g or p(eval(x:=f"{*zip(*g),}".replace(i//5*", 5",i//5*",7*~len({*x})%5"))[::-1],i-1)


%%writefile task375.py
def p(g,i=0):
	for r in g:r[i]=r[~i]=0;i+=1
	return g


%%writefile task376.py
p=lambda D:(D+D[1:-1])*2+D[:1]


%%writefile task377.py
p=lambda g,h=0:[h:=r for*r,in zip(*h or p(g,g))if r!=h]


%%writefile task378.py
import re
p=lambda g:exec("x='...'*len(g)+'.0';g[::-1]=zip(*eval(re.sub(f'0(?=({x}, .)*, [^0]{x*2}, (.))',r'\\2',str(g))));"*4)or g


%%writefile task379.py
p=lambda g,i=47:g*-i or[[[3&q%~(c:=r.pop())%5|2&q*-(8in r),~c*q%2*9,c&10][8>>13-i//4]or c for q in[0]+r[:0:-1]]for*r,in zip(*p(g,i-1))]


%%writefile task380.py
p=lambda g:[*zip(*g)][::-1]


%%writefile task381.py
p=lambda g:[r*(P:=r in g[::9])or[P:=r.pop(0)or any(r*P)*9for _ in g]for r in g]


%%writefile task382.py
p=lambda i,k=-3:k*i or p([*zip(w:=i.pop(),*[[*map(max,r,w:=r*('8'in'%s'%i)+[0,*w,0][r[-1]or[1]>r:])]for*r,in i[::~0]])],k+1)


%%writefile task383.py
p=lambda g:[[(a:=[v,*{}.fromkeys(sum(g,r))])[any(0<i.count(a[2])<4for i in[r,c])*2+0**v]for*c,v in zip(*g,r)]for r in g]


%%writefile task384.py
p=lambda g,*a:sum([any(x)*2*[x]for*x,in zip(*a or p(*g))],[])


%%writefile task385.py
p=lambda g:g[:4:-1]+g[5:]


%%writefile task386.py
p=lambda g:[eval('3>>r.pop(0)+r[3],'*3)for r in g]


%%writefile task387.py
p=lambda g,n=11:-n*g or[[[(i:=((x:=r.pop())+i>0)*-~i)%2*sum(r[:2-i])**4%5*5|x,x or-P**4%5*P*4,sum({*sum(g*(x>9),[])}-{x%15})or x][n//4]for P in[0]+r[:0:-1]]for*r,in zip(*p(g,n-1))if[i:=0]]


%%writefile task388.py
p=lambda g:[[x|8&x-any(c)for*c,x in zip(*g,r)]*2for r in g]*2


%%writefile task389.py
p=lambda g:[[sum({*sum(g,r)}-{5,x})for x in r]for r in g]


%%writefile task390.py
import re
p=lambda g,k=0:eval(re.sub("[^(2]{9}2"*2+"?","*[\g<0>][::-1]",f"{*zip(*k or p(g,g)),}"))


%%writefile task391.py
p=lambda n:[*zip(sorted({*(i:=sum(n,[]))},key=i.count)[2::-1])]


%%writefile task392.py
p=lambda g,n=23:-n*g or[eval(f"r.pop()or[5,w:=max(max(g))][{n}//4%(('0, %d, '%w*2in'{g}')-3)]*any(r[-1:]),"*10)for*r,in zip(*p(g,n-1))]


%%writefile task393.py
p=lambda g:[*zip(sorted({*(s:=sum(g,[]))},key=s.count)[2::-1])]


%%writefile task394.py
p=lambda i:[[v for y,v in zip(x,x[n:n*2]+x)if 1>y]for x in i if(n:=122%len(i))>0in x]


%%writefile task395.py
p=lambda g,u=[]:g*0!=0and[*map(p,g,u+g[3:])]or~g+~u&2


%%writefile task396.py
p=lambda m,n=266,f=0:[[sum({*e*sum(m,[-f])})for e in s]for r in m if(f:=((X:=n>>5)*[a:=(r*3)[x:=n%32]]in(s:=r[x:x+X],[f]*X))*a)]or p(m,n-1)


%%writefile task397.py
p=lambda g,i=81:exec("i-=1;c=i//9-1\nfor _ in{*all(b:=%s+%s)*b}:%s=3,3\n"%(('g[c:=c+1][i%9:i%9+2]',)*3)*i)or g


%%writefile task398.py
def p(g):g,=g;q=len({*g}-{0})*[0]*5;return[q:=q[1:]+[v]for v in g+q[5:]]


%%writefile task399.py
p=lambda g:[*zip(*[iter(sum(sum(g,[]))%7*[1,0]+[0]*9)]*3)][:3]


%%writefile task400.py
p=lambda g:[h[:5]for r in[*g]if(h:=g.pop()[~[*r,1].index(1)::~0])]


import zipfile, zlib, sys, os
import warnings, json, copy
import signal, time, functools
from zipfile import ZipFile
from zlib import compress
from tqdm import tqdm
import numpy as np

warnings.filterwarnings("ignore")

def check(solution, task_num, valall=False):
    task_data = load_examples(task_num)
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
                if not np.array_equal(np.array(actual),np.array(expected)):
                    return False
            except:
                return False
        return True
    except Exception as e:
        return False

score = 0
with ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
    for f in tqdm(range(1,401)):
        solution=open('task' + str(f).zfill(3) + '.py','rb').read()
        open('task' + str(f).zfill(3) + '.py','wb').write(solution.strip())
        if check(solution, f, valall=False):
            s = max([0.1,2500-len(solution)])
            score += s + 1
        else: print('ERROR...', f)
        zipf.write('task' + str(f).zfill(3) + '.py')
print('Score:', score)

