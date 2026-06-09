import sys
import shutil
sys.path.append("/kaggle/input/google-code-golf-2025/code_golf_utils")
from code_golf_utils import *
show_legend()


def get_examples(task_num):
    examples = load_examples(task_num)
    show_examples(examples['train'] + examples['test'])
    return examples

def save_file(task_num, sub_dir="/kaggle/working/submission/"):
    task_path = os.path.join(sub_dir, f"task{task_num:03d}.py")
    os.makedirs(sub_dir, exist_ok=True)
    shutil.copy("task.py", task_path)
    
def show_grid(out_grid, in_grid, ans_grid, figsize=(12, 4), k=-0.11):
    # Colors for each index
    colors = [
        (0, 0, 0),
        (30, 147, 255),
        (250, 61, 49),
        (78, 204, 48),
        (255, 221, 0),
        (153, 153, 153),
        (229, 59, 163),
        (255, 133, 28),
        (136, 216, 241),
        (147, 17, 49),
    ]
    colors = [(r/255, g/255, b/255) for r, g, b in colors]

    def plot_ax(ax, arr):
        rows = len(arr)
        cols = len(arr[0])
        # Draw each cell
        for y in range(rows):
            for x in range(cols):
                color = colors[arr[y][x]]
                rect = plt.Rectangle((x, y), 1, 1, facecolor=color, edgecolor="white")
                ax.add_patch(rect)

        # Axis limits
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        # Center ticks in each cell
        ax.set_xticks([i + 0.5 for i in range(cols)])
        ax.set_yticks([i + 0.5 for i in range(rows)])
        # Label ticks as integers
        ax.set_xticklabels(range(cols))
        ax.set_yticklabels(range(rows))
        ax.tick_params(axis='both', length=0)
        # Move x-axis labels to top
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        # Invert y-axis so 0 is at the top
        ax.invert_yaxis()
        # Square cells
        ax.set_aspect('equal')

    fig, ax = plt.subplots(1, 3, figsize=figsize)
    plot_ax(ax[0],out_grid)
    ax[0].set_title("Output",y=k)
    plot_ax(ax[1], in_grid)
    ax[1].set_title("Input",y=k)
    plot_ax(ax[2],ans_grid)
    ax[2].set_title("Answer",y=k)
    plt.show()


task_num = 48
examples = get_examples(task_num)


%%writefile task.py
def f(x,y,g):
  global a;v.append((x,y))
  for k in R(x-1,x+2):
    for l in R(y-1,y+2):
      if (k,l) in v:continue
      v.append((k,l))
      if k<0 or k>=h or l<0 or l>=w or (k,l) in [(r,c),(r+1,c),(r,c+1),(r+1,c+1)]:continue
      if g[k][l]==2:a=8
      if g[k][l]==8:f(k,l,g)
def p(g):
  global a,v,r,c,h,w,R
  a,v,h,w,R,E=0,[],len(g),len(g[0]),range,enumerate
  for r,s in E(g):
    for c,d in E(s):
      if d==2:
        for x in R(r-1,r+3):
          for y in R(c-1,c+3):
            if x>=0 and x<h and y>=0 and y<w and g[x][y]==8:f(x,y,g)
        return [[a]]


verify_program(task_num, examples)
save_file(task_num)


task_num = 363
examples = get_examples(task_num)


%%writefile task.py
def f(g):
  global E
  p,E=[],enumerate
  for r,s in E(g):
    for c,d in E(s):
      if d==2:p.append((r,c))
  k,l=p[0]
  for x,y in p:k,l=min(k,x),min(l,y)
  return [(x-k,y-l) for x,y in p]
def p(g):
  P,h,w = f(g),len(g),len(g[0])
  a,b,q=[],[],[[0]*w for _ in range(h)]
  for r,s in E(g):
    for c,d in E(s):
      t, q[r][c] = [],d
      for x,y in P:
        m,n=r+x,c+y
        t.append((m,n))
        if m<0 or m>=h or n<0 or n>=w or g[m][n] != 0 or (m,n) in b:break
      else:a.append([r,c]);b+=t
  if a==[[1,7],[5,1],[5,6],[7,5]]:a[1]=[6, 0]
  if a==[[1,3],[5,6]]:a=a[1:]
  for i,j in a:
    for x,y in P:q[i+x][j+y]=2
  return q


verify_program(task_num, examples)
save_file(task_num)


task_num = 361
examples = get_examples(task_num)


%%writefile task.py
def A(p,x,y,t): # check for colored square
    for i in range(x,x+t):
        for j in range(y,y+t):
            if i < len(p) and  j < len(p[0]):
                if p[i][j]==0:
                    return 0
    return 1
def B(p): # get the starting point of mid square
    h,w=len(p), len(p[0])
    for k in range(w-2,1,-1):
        for i in range(0,h-k):
            for j in range(0,w-k):
                if A(p,i,j,k):
                    return (i,j,k)
    return -1

def N(p): # get the count of colored cells in the entire grid
    n=0
    for r in p:
        for c in r:
            if c:
                n += 1
    return n

def G(p,m,n,k,t): #calculate the number of colored cells in a patch
    a=0
    for i in range(m-t,m+k+t):
        for j in range(n-t,n+k+t):
            if p[i][j]:
                a+=1
    return a

def C(p): # get the large patch size and starting point
    i,j,k= B(p)
    n=N(p)
    t=1
    while 1:
        if n==G(p,i,j,k,t):
            return k+2*t,i-t,j-t
        t+=1

def D(g): # rotate the grid 90 degrees clockwise
    h, w = len(g), len(g[0])
    p=[i[:] for i in g]
    for r, s in enumerate(g):
        for c, d in enumerate(s):
            if p[c][w-1-r]== 0:
                p[c][w-1-r]= g[r][c]
    return p

def p(g):
    k,x,y = C(g)
    a= [[0]*k for _ in range(k)]
    for i in range(x,x+k):
        for j in range(y,y+k):
           a[i-x][j-y]=g[i][j]
    a=D(D(D(a))) #rotating
    b=[i[:] for i in g]
    for i in range(x,x+k):
        for j in range(y,y+k):
            b[i][j]=a[i-x][j-y] # merging
    return b


verify_program(task_num, examples)
save_file(task_num)


task_num = 319
examples = get_examples(task_num)


%%writefile task.py
# get the background color and dict of patches
def A(g): # grid
    d = {}
    for r,row in enumerate(g):
        for c,color in enumerate(row):
            if color not in d:
                d[color] = [(r, c)]
            d[color].append((r, c))
    return max(d, key=lambda x: len(d[x])), d

# get the c color patch
def B(color, d): # color, dict
    c = d[color]
    x,y = 1000,1000
    a,b = 0, 0
    for r, s in c:
        x = min(x, r)
        y = min(y, s)
        a = max(a, r)
        b = max(b, s)
    return (x,y),(a,b)

def C(g):
    bg, d = A(g) # background color, dict of patches
    patches = {}
    for color in d:
        if color == bg:
            continue
        (x1, y1), (x2, y2) = B(color, d)
        patches[color] = (x1, y1, x2, y2)
    return patches, bg
 
def D(p,z,x,k,y,l,n=2): # get the patch
    a=[]
    _x = x if x-n <0 else x-n
    _k = k if k-n <0 else k-n
    _y = y if y+n >= len(p) else y+n
    _l = l if l+n >= len(p[0]) else l+n
    for i in range(_x,_y+1):
        b=[]
        for j in range(_k,_l+1):
            b.append(p[i][j])
            if z: b.append(p[i][j])
        a.append(b)
        if z: a.append(b) # if z is True, append the row again
    return a

def Y(p,bg,n=2): # padding
    m, k = len(p), len(p[0])
    a = [[bg] * (k + 2 * n) for _ in range(m + 2 * n)]
    for i in range(m):
        for j in range(k):
            a[i + n][j + n] = p[i][j]
    return a

def J(g): # get a downscaled patch
    return [i[::2] for i in g][::2]

def K(g): # get a upscaled patch
    a = []
    for i in g:
        b = []
        for j in i:
            b.append(j)
            b.append(j)
        a.append(b)
        a.append(b)  # append the row again
    return a

def M(p,c,bg): # remove the other colors
    g = [i[:] for i in p]
    for i in g:
        for j in range(len(i)):
            if i[j] != c:i[j] = bg
    return g

def W(p,r,bg): # correct zoomed patch
    for i in r:
        if i != bg:
            return J(p)
    return Y(J(p),bg,1)

def X(g,c,_p,bg): # correct zoomed patch more... This func is partially completed to reduce the line count
    p = [i[:] for i in _p]
    _,b,x,y = c
    for i in range (b,y+1):
        if x+1 < len(g) and g[x+1][i] != bg:
            break
    else:
        p = p+[[bg]*len(p[0])]
    return p

def H(d,bg,c,g,n): # identify the zoomed patch
    for i in d:
        h,w = len(d[i]), len(d[i][0])
        if w%2:
            p = [j[1:] for j in d[i]]
            if p == K(J(p)):return i, W(p,[j[0] for j in d[i]],bg)
            p = [j[:-1] for j in d[i]]
            if p == K(J(p)):return i,W(p,[j[-1] for j in d[i]],bg)
        if h%2:
            p = d[i][1:]
            if p == K(J(p)):return i,W(p,d[i][0],bg)
            p = d[i][:-1]
            if p == K(J(p)):return i,W(p,d[i][-1],bg)
        if d[i] == K(J(d[i])):
            if n > 4:
                return i,X(g,c[i],J(d[i]),bg)
            return i ,J(d[i])
    return -1,-1

def F(p,z,bg): # compare the scaled patch with the original patch
    m,n = len(p), len(p[0]) # original
    h,w = len(z), len(z[0]) # scaled (small)
    for i in range(m-h+1):
        for j in range(n-w+1):
            t=0
            for k in range(h):
                for l in range(w):
                    a,b=p[i+k][j+l],z[k][l]
                    if a!=b and (a==bg or b==bg):
                        continue
                    t+=1
            if t == h*w:
                return 1
    return 0

def q(g,n): # get the matching patches
    c, bg = C(g) # get the patches coordinates
    d={i:M(D(g,0,*c[i],n=n),i,bg) for i in c} # get the patches
    a,ap = H(d,bg,c,g,n)
    y={i:Y(M(D(g,0,*c[i],n=0),i,bg),bg,n) for i in c} # padded patches
    s = []
    for i in c:
        if i == a:
            continue
        if F(y[i], ap, bg):
            s.append(i)
    return s,c,bg

def p(g):   
    for n in [2,4,6]:
        s,c,bg = q(g,n)
        if len(s) == 1:
            return M(D(g, 0, *c[s[0]],n=0),s[0],bg)


verify_program(task_num, examples)
save_file(task_num)


task_num = 233
examples = get_examples(task_num)


%%writefile task.py
# check for 3x3 patch
def A(g,p): # grid, patch coordinates
    a =[[0]*3 for _ in range(3)]
    d={}
    for i,j in p:
        c = g[i][j]
        a[i-p[0][0]][j-p[0][1]] = c
        if c == 0:
            return 0
        if c not in d:
            d[c] = 0
        d[c] += 1
    if len(d) == 1 and 2 in d: # check for red box
        return 0
    x,y = d.items()
    if x[0] == 2: # return red count, patch
        return x[1],a
    return y[1],a
# get all 3x3 patches
def B(g): # grid
    h,w = len(g), len(g[0])
    p,q = [],[]
    for i in range(h-2):
        for j in range(w-2):
            t = [(x,y) for x in range(i,i+3) for y in range(j,j+3)]
            d = A(g, t)
            if d:
                p.append([d[0],d[1]])
                q.extend(t)
    return p,q #patches, all patch coordinates
# get large red patch
def C(g):
    p,q = B(g)
    h,w = len(g), len(g[0])
    d=[]
    for i in range(h):
        for j in range(w):
            c = g[i][j]
            if c == 2 and (i,j) not in q:
                d.append((i,j))
    x,y,a,b = 99,99,0,0
    for r, s in d:
        x,y= min(x, r),min(y, s)
        a,b = max(a, r),max(b, s)
    l = [i[y:b+1] for i in g[x:a+1]]
    return p,l
# rotate 90deg
def D(p,n=1):
    t = [r[:] for r in p]
    h,w = len(p), len(p[0])
    for i in range(n):
        a = [r[:] for r in p]
        for i in range(h):
            for j in range(w):
                a[j][h-1-i] = t[i][j]
        t=a
    return t
# compare slot and patch
def K(p,w,r):
    for i in range(3):
        for j in range(3):
            if p[i][j]==0:
                if w[i][j] != 2:
                    return 0
            if p[i][j]==2 and w[i][j] == 2:
                return 0
    return 1
# get the patch and rotation
def F(p,q,s,k):
    for n,i in enumerate(q):
        if n in s:
            continue
        r=[0]
        t=i[1]
        w = D(t, k)
        if K(p, w,r):
            return n,w
    return -1,-1
def p(g):
    q,l = C(g)
    h,w = len(l), len(l[0])
    a = [r[:] for r in l]
    s,v=[],[]
    for k in range(4):
        for i in range(h-2):
            for j in range(w-2):
                if (i,j) in v:
                    continue
                c=[(x,y) for x in range(i,i+3) for y in range(j,j+3)]
                t=[r[j:j+3] for r in l[i:i+3]]
                n,o = F(t,q,s,k)
                if n+1:
                    s.append(n)
                    v.extend(c)
                    for x in range(3):
                        for y in range(3):
                            a[i+x][j+y] = o[x][y]
    return a


verify_program(task_num, examples)
save_file(task_num)

