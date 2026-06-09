import os, json, zipfile
from tqdm import tqdm
from collections import Counter

# -------------------- util --------------------
def _len(s:str)->int: return len(s.encode("utf-8"))
def _ok(code:str,task)->bool:
    try:
        ns={};exec(code,{},ns);f=ns["p"]
        for ex in task.get("train",[])+task.get("test",[])+task.get("arc-gen",[]):
            if f(ex["input"])!=ex["output"]: return False
        return True
    except: return False

def dims(g): return (len(g),len(g[0]) if g else 0)
def flat(g):
    a=[]
    for r in g:a+=r
    return a

# -------------------- tiny geom exprs --------------------
def _id(X):   return X
def _vv(X):   return f"{X}[::-1]"
def _hh(X):   return f"[r[::-1]for r in{X}]"
def _tp(X):   return f"list(map(list,zip(*{X})))"
def _r90(X):  return f"list(map(list,zip(*{X}[::-1])))"
def _r180(X): return f"[r[::-1]for r in{X}[::-1]]"
def _r270(X): return f"{_tp(X)}[::-1]"
GEOMS=[_id,_vv,_hh,_tp,_r90,_r180,_r270]

def geom_lambdas():  # базовые геомы (самые короткие варианты)
    return [
        "p=lambda g:g",
        "p=lambda g:g[::-1]",
        "p=lambda g:[r[::-1]for r in g]",
        "p=lambda g:list(map(list,zip(*g)))",
        "p=lambda g:list(map(list,zip(*g[::-1])))",
        "p=lambda g:[r[::-1]for r in g[::-1]]",
        "p=lambda g:list(map(list,zip(*g)))[::-1]",
    ]

# -------------------- MDL-модели → код --------------------
def mdl_constant(task):
    outs=[e["output"] for e in task.get("train",[])+task.get("test",[])+task.get("arc-gen",[])]
    if outs and all(o==outs[0] for o in outs):
        return [f"p=lambda g:{repr(outs[0])}"]
    return []

def mdl_palette_map(task):
    exs=task.get("train",[])+task.get("test",[])+task.get("arc-gen",[])
    mp={}
    for e in exs:
        a,b=e["input"],e["output"]
        if dims(a)!=dims(b): continue
        for x,y in zip(flat(a),flat(b)):
            if x in mp and mp[x]!=y: return []
            mp[x]=y
    if not mp or all(k==v for k,v in mp.items()): return []
    d="{"+",".join(f"{k}:{v}" for k,v in sorted(mp.items()))+"}"
    return [f"p=lambda g:[[({d}).get(x,x)for x in r]for r in g]"]

def mdl_bbox_crop(task):
    # проверяем, что output == bbox(g!=0)
    def bbox(g):
        H=len(g);W=len(g[0]) if H else 0
        a,b,c,d=H,W,-1,-1
        for i in range(H):
            for j,x in enumerate(g[i]):
                if x:
                    if i<a:a=i
                    if j<b:b=j
                    if i>c:c=i
                    if j>d:d=j
        return None if c<0 else (a,b,c,d)
    ok=True
    for e in task.get("train",[])+task.get("test",[])+task.get("arc-gen",[]):
        a,b=e["input"],e["output"]; bb=bbox(a)
        if not bb: ok=False;break
        i0,j0,i1,j1=bb
        cut=[row[j0:j1+1]for row in a[i0:i1+1]]
        if cut!=b: ok=False;break
    if not ok: return []
    # короткий def (lambda тут получается длиннее)
    return [("def p(g):H=len(g);W=len(g[0]);a,b,c,d=H,W,-1,-1\n"
             "for i in range(H):\n for j,x in enumerate(g[i]):\n  "
             "a=i if x and i<a else a; b=j if x and j<b else b; c=i if x and i>c else c; d=j if x and j>d else d\n"
             "return[g[i][b:d+1]for i in range(a,c+1)]").replace("\n","")]

def mdl_border_pad_or_strip(task):
    exs=task.get("train",[])+task.get("test",[])+task.get("arc-gen",[])
    def infer_pad(a,b):
        Ha,Wa=dims(a);Hb,Wb=dims(b)
        if Hb<Ha or Wb<Wa:return None
        k=(Hb-Ha)//2
        if Hb!=Ha+2*k or Wb!=Wa+2*k:return None
        for i in range(Ha):
            if b[i+k][k:Wb-k]!=a[i]:return None
        cols=set()
        for i in range(Hb):
            for j in range(Wb):
                if not(k<=i<Hb-k and k<=j<Wb-k): cols.add(b[i][j])
        return (k,next(iter(cols))) if len(cols)==1 else None
    def infer_strip(a,b):
        Ha,Wa=dims(a);Hb,Wb=dims(b)
        if Ha<=Hb or Wa<=Wb:return None
        k=(Ha-Hb)//2
        if Hb!=Ha-2*k or Wb!=Wa-2*k:return None
        for i in range(Hb):
            if a[i+k][k:Wa-k]!=b[i]:return None
        return k
    pads=[infer_pad(e["input"],e["output"]) for e in exs]
    strips=[infer_strip(e["input"],e["output"]) for e in exs]
    if all(p is not None for p in pads):
        k,c=pads[0]
        if all(p==(k,c) for p in pads):
            # максимально кратко: верх/низ с умножением списка ОК (мутация нам не грозит)
            return [f"p=lambda g:[[{c}]*(len(g[0])+2*{k})]*{k}+[[{c}]*{k}+r+[{c}]*{k}for r in g]+[[{c}]*(len(g[0])+2*{k})]*{k}"]
    if all(s is not None for s in strips):
        k=strips[0]
        if all(s==k for s in strips):
            return [f"p=lambda g:[r[{k}:len(g[0])-{k}]for r in g[{k}:len(g)-{k}]] if len(g)>2*{k} and len(g[0])>2*{k} else [[0]]"]
    return []

def mdl_shift(task):
    # единственный (di,dj) для всех примеров; нулевое заполнение, без wrap
    def infer(a,b):
        Ha,Wa=dims(a);Hb,Wb=dims(b)
        if (Ha,Wa)!=(Hb,Wb):return None
        def C(g):
            pts=[(i,j)for i,row in enumerate(g) for j,x in enumerate(row) if x]
            if not pts:return(0,0)
            n=len(pts);si=sum(i for i,_ in pts);sj=sum(j for _,j in pts)
            return(round(si/n),round(sj/n))
        ai,aj=C(a);bi,bj=C(b);di,dj=bi-ai,bj-aj
        H,W=Ha,Wa
        for i in range(H):
            for j in range(W):
                I=i-di;J=j-dj
                x=a[I][J] if 0<=I<H and 0<=J<W else 0
                if x!=b[i][j]: return None
        return(di,dj)
    pars=[]
    for e in task.get("train",[])+task.get("test",[])+task.get("arc-gen",[]):
        p=infer(e["input"],e["output"])
        if p is None:return []
        pars.append(p)
    di,dj=pars[0]
    if not all(p==(di,dj) for p in pars): return []
    # однострочник без H,W переменных
    return [f"p=lambda g:[[g[i-{di}][j-{dj}] if 0<=i-{di}<len(g) and 0<=j-{dj}<len(g[0]) else 0 for j in range(len(g[0]))] for i in range(len(g))]"]

def mdl_color_basics():
    C=[]
    for k in range(10):
        C.append(f"p=lambda g:[[{k} if x==0 else x for x in r]for r in g]")
        C.append(f"p=lambda g:[[{k} if x else 0 for x in r]for r in g]")
        C.append(f"p=lambda g:[[x if x=={k} else 0 for x in r]for r in g]")
        C.append(f"p=lambda g:[[{k}]*len(g[0])for _ in g]")
        # точечная замена часто нужна и короткая
        for a in range(10):
            if a!=k:C.append(f"p=lambda g:[[{k} if x=={a} else x for x in r]for r in g]")
    return list(dict.fromkeys(C))

# -------- инлайн-композиции: палитра ∘ геом и геом ∘ палитра --------
def mdl_geom_palette_inline(task):
    P=mdl_palette_map(task)
    if not P: return []
    # извлечём словарь из уже сгенерированного кода
    # формат: p=lambda g:[[({d}).get(x,x)for x in r]for r in g]
    dstr=P[0].split("((")[1].split(")).get")[0] if "((" in P[0] else P[0].split("((")[0]
    if "{" not in P[0]: return []
    d=P[0][P[0].find("{"):P[0].find("}")+1]
    C=[]
    # G after P: G( [[d.get(..) for x in r] for r in g] )
    for op in GEOMS[:4]: # берём короткие
        X=f"[[({d}).get(x,x)for x in r]for r in g]"
        C.append(f"p=lambda g:{op(X)}")
    # P after G: [[d.get(..) for x in r] for r in G(g)]
    for op in GEOMS[:4]:
        X=op("g")
        C.append(f"p=lambda g:[[({d}).get(x,x)for x in r]for r in {X}]")
    return C

# -------------------- генерация кандидатов --------------------
def candidates(task):
    C=[]
    C+=mdl_constant(task)
    C+=mdl_palette_map(task)
    C+=geom_lambdas()
    C+=mdl_bbox_crop(task)
    C+=mdl_border_pad_or_strip(task)
    C+=mdl_shift(task)
    C+=mdl_color_basics()
    C+=mdl_geom_palette_inline(task)
    return list(dict.fromkeys(C))

# -------------------- solver (MDL=кратчайший корректный) --------------------
def solve_task(task):
    val=[]
    for code in candidates(task):
        if _ok(code,task): val.append((_len(code),code))
    if val:
        val.sort(key=lambda x:x[0])
        best=val[0][1]
        return best, max(1,2500-_len(best)), True
    return "p=lambda g:g", 0.001, False

# -------------------- main --------------------
def main():
    IN="/kaggle/input/google-code-golf-2025"; OUT="/kaggle/working/submission"
    os.makedirs(OUT,exist_ok=True)
    solved=0; score=0.0; ids=[]
    for n in tqdm(range(1,401),desc="Processing tasks"):
        tid=f"{n:03d}"; path=os.path.join(IN,f"task{tid}.json")
        try:
            with open(path) as f: task=json.load(f)
            code,s,ok=solve_task(task)
            if ok: solved+=1; ids.append(n)
            score+=s
            with open(os.path.join(OUT,f"task{tid}.py"),"w") as w:w.write(code)
        except:
            with open(os.path.join(OUT,f"task{tid}.py"),"w") as w:w.write("p=lambda g:g")
            score+=0.001
    zp="/kaggle/working/submission.zip"
    with zipfile.ZipFile(zp,"w") as z:
        for i in range(1,401):
            fn=os.path.join(OUT,f"task{i:03d}.py")
            if os.path.exists(fn): z.write(fn,arcname=f"task{i:03d}.py")
    print(f"Tasks solved: {solved}/400")
    print(f"Success rate: {solved/400*100:.1f}%")
    print(f"Total score: {score:.3f}")
    print(f"Average score per task: {score/400:.3f}")
    print(f"Solved task numbers: {ids[:20]}{'...' if len(ids)>20 else ''}")
    print("Submission file 'submission.zip' created successfully!")

if __name__=="__main__": main()



import os, json, zipfile
from tqdm import tqdm
from collections import Counter

# ---------------- util ----------------
def _len(s:str)->int: return len(s.encode("utf-8"))
def _ok(code:str,task)->bool:
    try:
        ns={}; exec(code,{},ns); f=ns["p"]
        for ex in task.get("train",[])+task.get("test",[])+task.get("arc-gen",[]):
            if f(ex["input"])!=ex["output"]: return False
        return True
    except:
        return False

def dims(g): return (len(g), len(g[0]) if g else 0)
def flat(g):
    a=[]
    for r in g: a+=r
    return a

# ---------------- basic geom lambdas ----------------
def geom_lambdas():
    return [
        "p=lambda g:g",
        "p=lambda g:g[::-1]",
        "p=lambda g:[r[::-1]for r in g]",
        "p=lambda g:list(map(list,zip(*g)))",
        "p=lambda g:list(map(list,zip(*g[::-1])))",
        "p=lambda g:[r[::-1]for r in g[::-1]]",
        "p=lambda g:list(map(list,zip(*g)))[::-1]",
    ]

# ---------------- atomic code generators ----------------
def const_fill(k): return f"p=lambda g:[[{k}]*len(g[0])for _ in g]"
def replace_zero(k): return f"p=lambda g:[[{k} if x==0 else x for x in r]for r in g]"
def replace_nz(k): return f"p=lambda g:[[{k} if x else 0 for x in r]for r in g]"
def keep_only(k): return f"p=lambda g:[[x if x=={k} else 0 for x in r]for r in g]"
def replace_eq(a,b): return f"p=lambda g:[[{b} if x=={a} else x for x in r]for r in g]"
def map_binary(): return "p=lambda g:[[1 if x else 0 for x in r]for r in g]"
def inv_binary(): return "p=lambda g:[[0 if x==1 else(1 if x==0 else x)for x in r]for r in g]"
def crop_ul_quarter(): return "p=lambda g:[r[:len(g[0])//2]for r in g[:len(g)//2]]"
def bbox_crop_def():
    return ("def p(g):H=len(g);W=len(g[0]);a,b,c,d=H,W,-1,-1;"
            "for i in range(H):"
            " for j,x in enumerate(g[i]):"
            "  if x:"
            "   a=i if i<a else a; b=j if j<b else b; c=i if i>c else c; d=j if j>d else d;"
            "return[g[i][b:d+1]for i in range(a,c+1)]")
def border_pad(k,c):
    return f"p=lambda g:[[{c}]*(len(g[0])+2*{k})]*{k}+[[{c}]*{k}+r+[{c}]*{k}for r in g]+[[{c}]*(len(g[0])+2*{k})]*{k}"
def strip_border(k):
    return f"p=lambda g:[r[{k}:len(g[0])-{k}]for r in g[{k}:len(g)-{k}]] if len(g)>2*{k} and len(g[0])>2*{k} else [[0]]"
def roll_shift(di,dj):
    # zero-filled shift; note that di,dj ints can be negative
    return (f"p=lambda g:[[g[i-{di}][j-{dj}] if 0<=i-{di}<len(g) and 0<=j-{dj}<len(g[0]) else 0 "
            f"for j in range(len(g[0]))] for i in range(len(g))]")

# ---------------- pattern detectors that emit candidate code lists ----------------
def mdl_constant(task):
    outs=[e["output"] for e in task.get("train",[])+task.get("test",[])+task.get("arc-gen",[])]
    if outs and all(o==outs[0] for o in outs):
        return [f"p=lambda g:{repr(outs[0])}"]
    return []

def mdl_palette_map(task):
    exs=task.get("train",[])+task.get("test",[])+task.get("arc-gen",[])
    mp={}
    for e in exs:
        a,b=e["input"],e["output"]
        if dims(a)!=dims(b): continue
        for x,y in zip(flat(a),flat(b)):
            if x in mp and mp[x]!=y: return []
            mp[x]=y
    if not mp or all(k==v for k,v in mp.items()): return []
    d="{"+",".join(f"{k}:{v}" for k,v in sorted(mp.items()))+"}"
    return [f"p=lambda g:[[({d}).get(x,x)for x in r]for r in g]"]

def mdl_bbox_crop(task):
    # detect if output equals bbox of nonzero of input for all examples
    def bbox(g):
        H=len(g);W=len(g[0]) if H else 0
        a,b,c,d=H,W,-1,-1
        for i in range(H):
            for j,x in enumerate(g[i]):
                if x:
                    if i<a:a=i
                    if j<b:b=j
                    if i>c:c=i
                    if j>d:d=j
        return None if c<0 else (a,b,c,d)
    for e in task.get("train",[])+task.get("test",[])+task.get("arc-gen",[]):
        a,b=e["input"],e["output"]; bb=bbox(a)
        if not bb: return []
        i0,j0,i1,j1=bb
        cut=[row[j0:j1+1] for row in a[i0:i1+1]]
        if cut!=b: return []
    return [bbox_crop_def()]

def mdl_border_pad_strip(task):
    exs=task.get("train",[])+task.get("test",[])+task.get("arc-gen",[])
    def infer_pad(a,b):
        Ha,Wa=dims(a);Hb,Wb=dims(b)
        if Hb<Ha or Wb<Wa:return None
        k=(Hb-Ha)//2
        if Hb!=Ha+2*k or Wb!=Wa+2*k:return None
        for i in range(Ha):
            if b[i+k][k:Wb-k]!=a[i]: return None
        cols=set()
        for i in range(Hb):
            for j in range(Wb):
                if not (k<=i<Hb-k and k<=j<Wb-k): cols.add(b[i][j])
        if len(cols)==1: return (k,next(iter(cols)))
        return None
    def infer_strip(a,b):
        Ha,Wa=dims(a);Hb,Wb=dims(b)
        if Ha<=Hb or Wa<=Wb:return None
        k=(Ha-Hb)//2
        if Hb!=Ha-2*k or Wb!=Wa-2*k:return None
        for i in range(Hb):
            if a[i+k][k:Wa-k]!=b[i]: return None
        return k
    pads=[infer_pad(e["input"],e["output"]) for e in exs]
    strips=[infer_strip(e["input"],e["output"]) for e in exs]
    if all(p is not None for p in pads):
        k,c=pads[0]
        if all(p==(k,c) for p in pads): return [border_pad(k,c)]
    if all(s is not None for s in strips):
        k=strips[0]
        if all(s==k for s in strips): return [strip_border(k)]
    return []

def mdl_shift(task):
    # detect constant (di,dj) by centroids and verify zero-filled shift
    def centroid(g):
        pts=[(i,j) for i,row in enumerate(g) for j,x in enumerate(row) if x]
        if not pts: return (0,0)
        si=sum(i for i,_ in pts); sj=sum(j for _,j in pts); n=len(pts)
        return (round(si/n), round(sj/n))
    for e in task.get("train",[])+task.get("test",[])+task.get("arc-gen",[]):
        pass
    pars=[]
    for e in task.get("train",[])+task.get("test",[])+task.get("arc-gen",[]):
        a,b=e["input"],e["output"]
        if dims(a)!=dims(b): return []
        ai,aj=centroid(a); bi,bj=centroid(b); di,dj=bi-ai,bj-aj
        H,W=dims(a)
        ok=True
        for i in range(H):
            for j in range(W):
                I=i-di;J=j-dj
                x=a[I][J] if 0<=I<H and 0<=J<W else 0
                if x!=b[i][j]: ok=False; break
            if not ok: break
        if not ok: return []
        pars.append((di,dj))
    di,dj=pars[0]
    if all(p==(di,dj) for p in pars):
        return [roll_shift(di,dj)]
    return []

# ---------------- Phase A candidates (compact) ----------------
def candidates_phaseA(task):
    C=[]
    C+=mdl_constant(task)
    C+=mdl_palette_map(task)
    C+=geom_lambdas()
    C+=mdl_bbox_crop(task)
    C+=mdl_border_pad_strip(task)
    C+=mdl_shift(task)
    # some tiny color ops
    C+=["p=lambda g:[[1 if x else 0 for x in r]for r in g]","p=lambda g:[[0 if x==1 else(1 if x==0 else x)for x in r]for r in g]"]
    for k in range(3): # small constants 0..2 are common and short
        C.append(const_fill(k))
        C.append(replace_zero(k))
        C.append(replace_nz(k))
        C.append(keep_only(k))
    return list(dict.fromkeys(C))

# ---------------- Phase B candidates (expanded search) ----------------
def candidates_phaseB(task):
    C=[]
    # many single color replacements/swaps
    for a in range(10):
        for b in range(10):
            if a!=b:
                C.append(replace_eq(a,b))
    # keep-only & fills for all colors
    for k in range(10):
        C.append(keep_only(k)); C.append(const_fill(k))
        C.append(replace_zero(k)); C.append(replace_nz(k))
    # small shifts [-2..2]
    for di in range(-2,3):
        for dj in range(-2,3):
            if di==0 and dj==0: continue
            C.append(roll_shift(di,dj))
    # crops: UL, UR, LL, LR quarter candidates (by slicing)
    C.append("p=lambda g:[r[:len(g[0])//2]for r in g[:len(g)//2]]")
    C.append("p=lambda g:[r[len(g[0])//2:]for r in g[:len(g)//2]]")
    C.append("p=lambda g:[r[:len(g[0])//2]for r in g[len(g)//2:]]")
    C.append("p=lambda g:[r[len(g[0])//2:]for r in g[len(g)//2:]]")
    # compose palette<->geom inline (apply palette after geom and geom after palette)
    P=mdl_palette_map(task)
    G=geom_lambdas()
    if P:
        d=P[0][P[0].find("{"):P[0].find("}")+1]
        for g in G[:6]:
            X=f"[[({d}).get(x,x)for x in r]for r in {g.split('=',1)[1]}]"
            C.append(f"p=lambda g:{X}")
            X2=f"{g.split('=',1)[1].replace('g','[[({d}).get(x,x)for x in r]for r in g')}"
            C.append(f"p=lambda g:{X2}")
    # add some bbox/def forms as fallback
    C.append(bbox_crop_def())
    # dedup preserve order
    return list(dict.fromkeys(C))

# ---------------- solver with two phases ----------------
def solve_task(task):
    found=[]
    # Phase A: quick compact search
    for code in candidates_phaseA(task):
        if _ok(code,task):
            found.append(( _len(code), code ))
    # If none found — Phase B (wider)
    if not found:
        for code in candidates_phaseB(task):
            if _ok(code,task):
                found.append(( _len(code), code ))
    if found:
        found.sort(key=lambda x:x[0])
        best=found[0][1]
        score=max(1,2500-_len(best))
        return best, score, True
    return "p=lambda g:g", 0.001, False

# ---------------- main ----------------
def main():
    IN="/kaggle/input/google-code-golf-2025"; OUT="/kaggle/working/submission"
    os.makedirs(OUT,exist_ok=True)
    solved=0; score=0.0; ids=[]
    for n in tqdm(range(1,401),desc="Processing tasks"):
        tid=f"{n:03d}"; path=os.path.join(IN,f"task{tid}.json")
        try:
            with open(path) as f: task=json.load(f)
            code,s,ok=solve_task(task)
            if ok: solved+=1; ids.append(n)
            score+=s
            with open(os.path.join(OUT,f"task{tid}.py"),"w") as w: w.write(code)
        except Exception:
            with open(os.path.join(OUT,f"task{tid}.py"),"w") as w: w.write("p=lambda g:g")
            score+=0.001
    zp="/kaggle/working/submission.zip"
    with zipfile.ZipFile(zp,"w") as z:
        for i in range(1,401):
            fn=os.path.join(OUT,f"task{i:03d}.py")
            if os.path.exists(fn): z.write(fn,arcname=f"task{i:03d}.py")
    print(f"Tasks solved: {solved}/400")
    print(f"Success rate: {solved/400*100:.1f}%")
    print(f"Total score: {score:.3f}")
    print(f"Average score per task: {score/400:.3f}")
    print(f"Solved task numbers: {ids[:20]}{'...' if len(ids)>20 else ''}")
    print("Submission file 'submission.zip' created successfully!")

if __name__=="__main__": main()



import os, json, zipfile
from tqdm import tqdm
from collections import Counter

# ---------------- util ----------------
def _len(s:str)->int: return len(s.encode("utf-8"))
def _ok(code:str,task)->bool:
    try:
        ns={}; exec(code,{},ns); f=ns["p"]
        for ex in task.get("train",[])+task.get("test",[])+task.get("arc-gen",[]):
            if f(ex["input"])!=ex["output"]: return False
        return True
    except:
        return False

def dims(g): return (len(g), len(g[0]) if g else 0)
def flat(g):
    a=[]
    for r in g: a+=r
    return a

# ---------------- basic geom lambdas ----------------
def geom_lambdas():
    return [
        "p=lambda g:g",
        "p=lambda g:g[::-1]",
        "p=lambda g:[r[::-1]for r in g]",
        "p=lambda g:list(map(list,zip(*g)))",
        "p=lambda g:list(map(list,zip(*g[::-1])))",
        "p=lambda g:[r[::-1]for r in g[::-1]]",
        "p=lambda g:list(map(list,zip(*g)))[::-1]",
    ]

# ---------------- atomic code generators ----------------
def const_fill(k): return f"p=lambda g:[[{k}]*len(g[0])for _ in g]"
def replace_zero(k): return f"p=lambda g:[[{k} if x==0 else x for x in r]for r in g]"
def replace_nz(k): return f"p=lambda g:[[{k} if x else 0 for x in r]for r in g]"
def keep_only(k): return f"p=lambda g:[[x if x=={k} else 0 for x in r]for r in g]"
def replace_eq(a,b): return f"p=lambda g:[[{b} if x=={a} else x for x in r]for r in g]"
def map_binary(): return "p=lambda g:[[1 if x else 0 for x in r]for r in g]"
def inv_binary(): return "p=lambda g:[[0 if x==1 else(1 if x==0 else x)for x in r]for r in g]"
def crop_ul_quarter(): return "p=lambda g:[r[:len(g[0])//2]for r in g[:len(g)//2]]"
def bbox_crop_def():
    return ("def p(g):H=len(g);W=len(g[0]);a,b,c,d=H,W,-1,-1;"
            "for i in range(H):"
            " for j,x in enumerate(g[i]):"
            "  if x:"
            "   a=i if i<a else a; b=j if j<b else b; c=i if i>c else c; d=j if j>d else d;"
            "return[g[i][b:d+1]for i in range(a,c+1)]")
def border_pad(k,c):
    return f"p=lambda g:[[{c}]*(len(g[0])+2*{k})]*{k}+[[{c}]*{k}+r+[{c}]*{k}for r in g]+[[{c}]*(len(g[0])+2*{k})]*{k}"
def strip_border(k):
    return f"p=lambda g:[r[{k}:len(g[0])-{k}]for r in g[{k}:len(g)-{k}]] if len(g)>2*{k} and len(g[0])>2*{k} else [[0]]"
def roll_shift(di,dj):
    return (f"p=lambda g:[[g[i-{di}][j-{dj}] if 0<=i-{di}<len(g) and 0<=j-{dj}<len(g[0]) else 0 "
            f"for j in range(len(g[0]))] for i in range(len(g))]")

# ---------------- segment & swap code generators ----------------
def swap_code(a,b):
    # swap color a and b
    return f"p=lambda g:[[ {b} if x=={a} else ({a} if x=={b} else x) for x in r]for r in g]"

def segment_code():
    # keep full grid size, replace background detected by mode, zero everything except bbox of non-bg object (bg preserved)
    return ("def p(g):"
            "H=len(g);W=len(g[0]);cnt={};"
            "for r in g:"
            " for x in r: cnt[x]=cnt.get(x,0)+1;"
            "bg=max(cnt,key=cnt.get);"
            "a,b,c,d=H,W,-1,-1;"
            "for i in range(H):"
            " for j,x in enumerate(g[i]):"
            "  if x!=bg:"
            "   if i<a:a=i"
            "   if j<b:b=j"
            "   if i>c:c=i"
            "   if j>d:d=j"
            "if c==-1: return [[bg]*W for _ in range(H)];"
            "res=[[bg]*W for _ in range(H)];"
            "for i in range(a,c+1):"
            " res[i][b:d+1]=g[i][b:d+1];"
            "return res")

def segment_shift_code(di,dj):
    # extract object bbox and place it shifted by (di,dj) with bg fill
    return ("def p(g):"
            "H=len(g);W=len(g[0]);cnt={};"
            "for r in g:"
            " for x in r: cnt[x]=cnt.get(x,0)+1;"
            "bg=max(cnt,key=cnt.get);"
            "a,b,c,d=H,W,-1,-1;"
            "for i in range(H):"
            " for j,x in enumerate(g[i]):"
            "  if x!=bg:"
            "   if i<a:a=i"
            "   if j<b:b=j"
            "   if i>c:c=i"
            "   if j>d:d=j"
            "if c==-1: return [[bg]*W for _ in range(H)];"
            "obj=[row[b:d+1] for row in g[a:c+1]];"
            "o=[[bg]*W for _ in range(H)];"
            f"di={di};dj={dj};"
            "for i,row in enumerate(obj):"
            " for j,x in enumerate(row):"
            "  I=i+a+di;J=j+b+dj;"
            "  if 0<=I<H and 0<=J<W: o[I][J]=x"
            "return o")

# ---------------- pattern detectors that emit candidate code lists ----------------
def mdl_constant(task):
    outs=[e["output"] for e in task.get("train",[])+task.get("test",[])+task.get("arc-gen",[])]
    if outs and all(o==outs[0] for o in outs):
        return [f"p=lambda g:{repr(outs[0])}"]
    return []

def mdl_palette_map(task):
    exs=task.get("train",[])+task.get("test",[])+task.get("arc-gen",[])
    mp={}
    for e in exs:
        a,b=e["input"],e["output"]
        if dims(a)!=dims(b): continue
        for x,y in zip(flat(a),flat(b)):
            if x in mp and mp[x]!=y: return []
            mp[x]=y
    if not mp or all(k==v for k,v in mp.items()): return []
    d="{"+",".join(f"{k}:{v}" for k,v in sorted(mp.items()))+"}"
    return [f"p=lambda g:[[({d}).get(x,x)for x in r]for r in g]"]

def mdl_bbox_crop(task):
    def bbox(g):
        H=len(g);W=len(g[0]) if H else 0
        a,b,c,d=H,W,-1,-1
        for i in range(H):
            for j,x in enumerate(g[i]):
                if x:
                    if i<a:a=i
                    if j<b:b=j
                    if i>c:c=i
                    if j>d:d=j
        return None if c<0 else (a,b,c,d)
    for e in task.get("train",[])+task.get("test",[])+task.get("arc-gen",[]):
        a,b=e["input"],e["output"]; bb=bbox(a)
        if not bb: return []
        i0,j0,i1,j1=bb
        cut=[row[j0:j1+1] for row in a[i0:i1+1]]
        if cut!=b: return []
    return [bbox_crop_def()]

def mdl_border_pad_strip(task):
    exs=task.get("train",[])+task.get("test",[])+task.get("arc-gen",[])
    def infer_pad(a,b):
        Ha,Wa=dims(a);Hb,Wb=dims(b)
        if Hb<Ha or Wb<Wa:return None
        k=(Hb-Ha)//2
        if Hb!=Ha+2*k or Wb!=Wa+2*k:return None
        for i in range(Ha):
            if b[i+k][k:Wb-k]!=a[i]: return None
        cols=set()
        for i in range(Hb):
            for j in range(Wb):
                if not (k<=i<Hb-k and k<=j<Wb-k): cols.add(b[i][j])
        if len(cols)==1: return (k,next(iter(cols)))
        return None
    def infer_strip(a,b):
        Ha,Wa=dims(a);Hb,Wb=dims(b)
        if Ha<=Hb or Wa<=Wb:return None
        k=(Ha-Hb)//2
        if Hb!=Ha-2*k or Wb!=Wa-2*k:return None
        for i in range(Hb):
            if a[i+k][k:Wa-k]!=b[i]: return None
        return k
    pads=[infer_pad(e["input"],e["output"]) for e in exs]
    strips=[infer_strip(e["input"],e["output"]) for e in exs]
    if all(p is not None for p in pads):
        k,c=pads[0]
        if all(p==(k,c) for p in pads): return [border_pad(k,c)]
    if all(s is not None for s in strips):
        k=strips[0]
        if all(s==k for s in strips): return [strip_border(k)]
    return []

def mdl_shift(task):
    def centroid(g):
        pts=[(i,j) for i,row in enumerate(g) for j,x in enumerate(row) if x]
        if not pts: return (0,0)
        si=sum(i for i,_ in pts); sj=sum(j for _,j in pts); n=len(pts)
        return (round(si/n), round(sj/n))
    pars=[]
    for e in task.get("train",[])+task.get("test",[])+task.get("arc-gen",[]):
        a,b=e["input"],e["output"]
        if dims(a)!=dims(b): return []
        ai,aj=centroid(a); bi,bj=centroid(b); di,dj=bi-ai,bj-aj
        H,W=dims(a)
        ok=True
        for i in range(H):
            for j in range(W):
                I=i-di;J=j-dj
                x=a[I][J] if 0<=I<H and 0<=J<W else 0
                if x!=b[i][j]: ok=False; break
            if not ok: break
        if not ok: return []
        pars.append((di,dj))
    di,dj=pars[0]
    if all(p==(di,dj) for p in pars):
        return [roll_shift(di,dj)]
    return []

# ---------------- Phase A candidates (compact) ----------------
def candidates_phaseA(task):
    C=[]
    C+=mdl_constant(task)
    C+=mdl_palette_map(task)
    C+=geom_lambdas()
    C+=mdl_bbox_crop(task)
    C+=mdl_border_pad_strip(task)
    C+=mdl_shift(task)
    # some tiny color ops
    C+=["p=lambda g:[[1 if x else 0 for x in r]for r in g]","p=lambda g:[[0 if x==1 else(1 if x==0 else x)for x in r]for r in g]"]
    for k in range(3): # small constants 0..2 are common and short
        C.append(const_fill(k))
        C.append(replace_zero(k))
        C.append(replace_nz(k))
        C.append(keep_only(k))
    return list(dict.fromkeys(C))

# ---------------- Phase B candidates (expanded search) ----------------
def candidates_phaseB(task):
    C=[]
    # many single color replacements/swaps
    for a in range(10):
        for b in range(10):
            if a!=b:
                C.append(replace_eq(a,b))
                C.append(swap_code(a,b))
    # include segment and segment+shift candidates
    C.append(segment_code())
    for di in range(-2,3):
        for dj in range(-2,3):
            if di==0 and dj==0: continue
            C.append(segment_shift_code(di,dj))
    # keep-only & fills for all colors
    for k in range(10):
        C.append(keep_only(k)); C.append(const_fill(k))
        C.append(replace_zero(k)); C.append(replace_nz(k))
    # small shifts [-2..2]
    for di in range(-2,3):
        for dj in range(-2,3):
            if di==0 and dj==0: continue
            C.append(roll_shift(di,dj))
    # crops: UL, UR, LL, LR quarter candidates (by slicing)
    C.append("p=lambda g:[r[:len(g[0])//2]for r in g[:len(g)//2]]")
    C.append("p=lambda g:[r[len(g[0])//2:]for r in g[:len(g)//2]]")
    C.append("p=lambda g:[r[:len(g[0])//2]for r in g[len(g)//2:]]")
    C.append("p=lambda g:[r[len(g[0])//2:]for r in g[len(g)//2:]]")
    # compose palette<->geom inline (apply palette after geom and geom after palette)
    P=mdl_palette_map(task)
    G=geom_lambdas()
    if P:
        d=P[0][P[0].find("{"):P[0].find("}")+1]
        for g in G[:6]:
            X=f"[[({d}).get(x,x)for x in r]for r in {g.split('=',1)[1]}]"
            C.append(f"p=lambda g:{X}")
            X2=f"{g.split('=',1)[1].replace('g','[[({d}).get(x,x)for x in r]for r in g')}"
            C.append(f"p=lambda g:{X2}")
    # add some bbox/def forms as fallback
    C.append(bbox_crop_def())
    # dedup preserve order
    return list(dict.fromkeys(C))

# ---------------- solver with two phases ----------------
def solve_task(task):
    found=[]
    # Phase A: quick compact search
    for code in candidates_phaseA(task):
        if _ok(code,task):
            found.append(( _len(code), code ))
    # If none found — Phase B (wider)
    if not found:
        for code in candidates_phaseB(task):
            if _ok(code,task):
                found.append(( _len(code), code ))
    if found:
        found.sort(key=lambda x:x[0])
        best=found[0][1]
        score=max(1,2500-_len(best))
        return best, score, True
    return "p=lambda g:g", 0.001, False

# ---------------- main ----------------
def main():
    IN="/kaggle/input/google-code-golf-2025"; OUT="/kaggle/working/submission"
    os.makedirs(OUT,exist_ok=True)
    solved=0; score=0.0; ids=[]
    for n in tqdm(range(1,401),desc="Processing tasks"):
        tid=f"{n:03d}"; path=os.path.join(IN,f"task{tid}.json")
        try:
            with open(path) as f: task=json.load(f)
            code,s,ok=solve_task(task)
            if ok: solved+=1; ids.append(n)
            score+=s
            with open(os.path.join(OUT,f"task{tid}.py"),"w") as w: w.write(code)
        except Exception:
            with open(os.path.join(OUT,f"task{tid}.py"),"w") as w: w.write("p=lambda g:g")
            score+=0.001
    zp="/kaggle/working/submission.zip"
    with zipfile.ZipFile(zp,"w") as z:
        for i in range(1,401):
            fn=os.path.join(OUT,f"task{i:03d}.py")
            if os.path.exists(fn): z.write(fn,arcname=f"task{i:03d}.py")
    print(f"Tasks solved: {solved}/400")
    print(f"Success rate: {solved/400*100:.1f}%")
    print(f"Total score: {score:.3f}")
    print(f"Average score per task: {score/400:.3f}")
    print(f"Solved task numbers: {ids[:20]}{'...' if len(ids)>20 else ''}")
    print("Submission file 'submission.zip' created successfully!")

if __name__=="__main__": main()



import os, json, zipfile
from tqdm import tqdm
from collections import Counter

# ---------------- util ----------------
def _len(s:str)->int: return len(s.encode("utf-8"))
def _ok(code:str,task)->bool:
    try:
        ns={}; exec(code,{},ns); f=ns["p"]
        for ex in task.get("train",[])+task.get("test",[])+task.get("arc-gen",[]):
            if f(ex["input"])!=ex["output"]: return False
        return True
    except:
        return False

def dims(g): return (len(g), len(g[0]) if g else 0)
def flat(g):
    a=[]
    for r in g: a+=r
    return a

def train_colors(task):
    cols=set()
    for e in task.get("train",[]):
        for r in e["input"]:
            cols.update(r)
    return sorted(cols)

# ---------------- basic geom lambdas ----------------
def geom_lambdas():
    return [
        ("geom:id","p=lambda g:g"),
        ("geom:flip_v","p=lambda g:g[::-1]"),
        ("geom:flip_h","p=lambda g:[r[::-1]for r in g]"),
        ("geom:trans","p=lambda g:list(map(list,zip(*g)))"),
        ("geom:rot90","p=lambda g:list(map(list,zip(*g[::-1])))"),
        ("geom:rot180","p=lambda g:[r[::-1]for r in g[::-1]]"),
        ("geom:rot270","p=lambda g:list(map(list,zip(*g)))[::-1]"),
    ]

# ---------------- atomic code generators (name,code) ----------------
def const_fill(k): return (f"fill:{k}", f"p=lambda g:[[{k}]*len(g[0])for _ in g]")
def replace_zero(k): return (f"rep0:{k}", f"p=lambda g:[[{k} if x==0 else x for x in r]for r in g]")
def replace_nz(k): return (f"repnz:{k}", f"p=lambda g:[[{k} if x else 0 for x in r]for r in g]")
def keep_only(k): return (f"keep:{k}", f"p=lambda g:[[x if x=={k} else 0 for x in r]for r in g]")
def replace_eq(a,b): return (f"repeq:{a}->{b}", f"p=lambda g:[[{b} if x=={a} else x for x in r]for r in g]")
def map_binary(): return ("bin","p=lambda g:[[1 if x else 0 for x in r]for r in g]")
def inv_binary(): return ("ibin","p=lambda g:[[0 if x==1 else(1 if x==0 else x)for x in r]for r in g]")
def crop_ul_quarter(): return ("crop:ul","p=lambda g:[r[:len(g[0])//2]for r in g[:len(g)//2]]")
def bbox_crop_def():
    return ("bbox_crop",
            "def p(g):H=len(g);W=len(g[0]);a,b,c,d=H,W,-1,-1;"
            "for i in range(H):"
            " for j,x in enumerate(g[i]):"
            "  if x:"
            "   a=i if i<a else a; b=j if j<b else b; c=i if i>c else c; d=j if j>d else d;"
            "return[g[i][b:d+1]for i in range(a,c+1)]")
def border_pad(k,c):
    return (f"pad:{k},{c}", f"p=lambda g:[[{c}]*(len(g[0])+2*{k})]*{k}+[[{c}]*{k}+r+[{c}]*{k}for r in g]+[[{c}]*(len(g[0])+2*{k})]*{k}")
def strip_border(k):
    return (f"strip:{k}", f"p=lambda g:[r[{k}:len(g[0])-{k}]for r in g[{k}:len(g)-{k}]] if len(g)>2*{k} and len(g[0])>2*{k} else [[0]]")
def roll_shift(di,dj):
    return (f"shift:{di},{dj}",
            f"p=lambda g:[[g[i-{di}][j-{dj}] if 0<=i-{di}<len(g) and 0<=j-{dj}<len(g[0]) else 0 for j in range(len(g[0]))] for i in range(len(g))]")

# ---------------- segment & swap code generators (strings) ----------------
def swap_code(a,b):
    return (f"swap:{a},{b}",
            f"p=lambda g:[[ {b} if x=={a} else ({a} if x=={b} else x) for x in r]for r in g]")

def segment_code():
    return ("segment",
            "def p(g):"
            "H=len(g);W=len(g[0]);cnt={};"
            "for r in g:"
            " for x in r: cnt[x]=cnt.get(x,0)+1;"
            "bg=max(cnt,key=cnt.get);"
            "a,b,c,d=H,W,-1,-1;"
            "for i in range(H):"
            " for j,x in enumerate(g[i]):"
            "  if x!=bg:"
            "   if i<a:a=i"
            "   if j<b:b=j"
            "   if i>c:c=i"
            "   if j>d:d=j"
            "if c==-1: return [[bg]*W for _ in range(H)];"
            "res=[[bg]*W for _ in range(H)];"
            "for i in range(a,c+1):"
            " res[i][b:d+1]=g[i][b:d+1];"
            "return res")

def segment_shift_code(di,dj):
    return (f"segshift:{di},{dj}",
            "def p(g):"
            "H=len(g);W=len(g[0]);cnt={};"
            "for r in g:"
            " for x in r: cnt[x]=cnt.get(x,0)+1;"
            "bg=max(cnt,key=cnt.get);"
            "a,b,c,d=H,W,-1,-1;"
            "for i in range(H):"
            " for j,x in enumerate(g[i]):"
            "  if x!=bg:"
            "   if i<a:a=i"
            "   if j<b:b=j"
            "   if i>c:c=i"
            "   if j>d:d=j"
            "if c==-1: return [[bg]*W for _ in range(H)];"
            "obj=[row[b:d+1] for row in g[a:c+1]];"
            "o=[[bg]*W for _ in range(H)];"
            f"di={di};dj={dj};"
            "for i,row in enumerate(obj):"
            " for j,x in enumerate(row):"
            "  I=i+a+di;J=j+b+dj;"
            "  if 0<=I<H and 0<=J<W: o[I][J]=x"
            "return o")

# ---------------- pattern detectors that emit candidate (name,code) lists ----------------
def mdl_constant(task):
    outs=[e["output"] for e in task.get("train",[])+task.get("test",[])+task.get("arc-gen",[])]
    if outs and all(o==outs[0] for o in outs):
        return [("const_out", f"p=lambda g:{repr(outs[0])}")]
    return []

def mdl_palette_map(task):
    exs=task.get("train",[])+task.get("test",[])+task.get("arc-gen",[])
    mp={}
    for e in exs:
        a,b=e["input"],e["output"]
        if dims(a)!=dims(b): continue
        for x,y in zip(flat(a),flat(b)):
            if x in mp and mp[x]!=y: return []
            mp[x]=y
    if not mp or all(k==v for k,v in mp.items()): return []
    d="{"+",".join(f"{k}:{v}" for k,v in sorted(mp.items()))+"}"
    return [("palette", f"p=lambda g:[[({d}).get(x,x)for x in r]for r in g]")]

def mdl_bbox_crop(task):
    def bbox(g):
        H=len(g);W=len(g[0]) if H else 0
        a,b,c,d=H,W,-1,-1
        for i in range(H):
            for j,x in enumerate(g[i]):
                if x:
                    if i<a:a=i
                    if j<b:b=j
                    if i>c:c=i
                    if j>d:d=j
        return None if c<0 else (a,b,c,d)
    for e in task.get("train",[])+task.get("test",[])+task.get("arc-gen",[]):
        a,b=e["input"],e["output"]; bb=bbox(a)
        if not bb: return []
        i0,j0,i1,j1=bb
        cut=[row[j0:j1+1] for row in a[i0:i1+1]]
        if cut!=b: return []
    return [bbox_crop_def()]

def mdl_border_pad_strip(task):
    exs=task.get("train",[])+task.get("test",[])+task.get("arc-gen",[])
    def infer_pad(a,b):
        Ha,Wa=dims(a);Hb,Wb=dims(b)
        if Hb<Ha or Wb<Wa:return None
        k=(Hb-Ha)//2
        if Hb!=Ha+2*k or Wb!=Wa+2*k:return None
        for i in range(Ha):
            if b[i+k][k:Wb-k]!=a[i]: return None
        cols=set()
        for i in range(Hb):
            for j in range(Wb):
                if not (k<=i<Hb-k and k<=j<Wb-k): cols.add(b[i][j])
        if len(cols)==1: return (k,next(iter(cols)))
        return None
    def infer_strip(a,b):
        Ha,Wa=dims(a);Hb,Wb=dims(b)
        if Ha<=Hb or Wa<=Wb:return None
        k=(Ha-Hb)//2
        if Hb!=Ha-2*k or Wb!=Wa-2*k:return None
        for i in range(Hb):
            if a[i+k][k:Wa-k]!=b[i]: return None
        return k
    pads=[infer_pad(e["input"],e["output"]) for e in exs]
    strips=[infer_strip(e["input"],e["output"]) for e in exs]
    if all(p is not None for p in pads):
        k,c=pads[0]
        if all(p==(k,c) for p in pads): return [border_pad(k,c)]
    if all(s is not None for s in strips):
        k=strips[0]
        if all(s==k for s in strips): return [strip_border(k)]
    return []

def mdl_shift(task):
    def centroid(g):
        pts=[(i,j) for i,row in enumerate(g) for j,x in enumerate(row) if x]
        if not pts: return (0,0)
        si=sum(i for i,_ in pts); sj=sum(j for _,j in pts); n=len(pts)
        return (round(si/n), round(sj/n))
    pars=[]
    for e in task.get("train",[])+task.get("test",[])+task.get("arc-gen",[]):
        a,b=e["input"],e["output"]
        if dims(a)!=dims(b): return []
        ai,aj=centroid(a); bi,bj=centroid(b); di,dj=bi-ai,bj-aj
        H,W=dims(a)
        ok=True
        for i in range(H):
            for j in range(W):
                I=i-di;J=j-dj
                x=a[I][J] if 0<=I<H and 0<=J<W else 0
                if x!=b[i][j]: ok=False; break
            if not ok: break
        if not ok: return []
        pars.append((di,dj))
    di,dj=pars[0]
    if all(p==(di,dj) for p in pars):
        return [roll_shift(di,dj)]
    return []

# ---------------- Phase A candidates (compact) ----------------
def candidates_phaseA(task):
    C=[]
    C+=mdl_constant(task)
    C+=mdl_palette_map(task)
    C+=geom_lambdas()
    C+=mdl_bbox_crop(task)
    C+=mdl_border_pad_strip(task)
    C+=mdl_shift(task)
    # tiny color ops
    C+= [("bin","p=lambda g:[[1 if x else 0 for x in r]for r in g]"),
         ("ibin","p=lambda g:[[0 if x==1 else(1 if x==0 else x)for x in r]for r in g]")]
    for k in range(3):
        C.append(const_fill(k)); C.append(replace_zero(k)); C.append(replace_nz(k)); C.append(keep_only(k))
    # dedup by name preserve order
    seen=set(); out=[]
    for name,code in C:
        if name not in seen:
            seen.add(name); out.append((name,code))
    return out

# ---------------- Phase B candidates (expanded search, uses train colors) ----------------
def candidates_phaseB(task):
    C=[]
    cols=train_colors(task)
    # swaps only for colors seen in training set (reduces перебор)
    for i in range(len(cols)):
        for j in range(i+1,len(cols)):
            a=cols[i]; b=cols[j]
            C.append(swap_code(a,b))
    # single-color replacements for all colors (keeps coverage)
    for k in cols:
        C.append(keep_only(k)); C.append(const_fill(k)); C.append(replace_zero(k)); C.append(replace_nz(k))
    # include segment and segment+shift candidates (limited shifts)
    C.append(segment_code())
    for di in range(-2,3):
        for dj in range(-2,3):
            if di==0 and dj==0: continue
            C.append(segment_shift_code(di,dj))
    # small shifts [-2..2]
    for di in range(-2,3):
        for dj in range(-2,3):
            if di==0 and dj==0: continue
            C.append(roll_shift(di,dj))
    # quarter crops
    C.append(("crop:ul","p=lambda g:[r[:len(g[0])//2]for r in g[:len(g)//2]]"))
    C.append(("crop:ur","p=lambda g:[r[len(g[0])//2:]for r in g[:len(g)//2]]"))
    C.append(("crop:ll","p=lambda g:[r[:len(g[0])//2]for r in g[len(g)//2:]]"))
    C.append(("crop:lr","p=lambda g:[r[len(g[0])//2:]for r in g[len(g)//2:]]"))
    # compose palette<->geom inline if palette detected
    pal=mdl_palette_map(task)
    G=geom_lambdas()
    if pal:
        d=pal[0][1][pal[0][1].find("{"):pal[0][1].find("}")+1]
        for name,g in G[:6]:
            expr=g.split('=',1)[1]
            X=f"[[({d}).get(x,x)for x in r]for r in {expr}]"
            C.append((f"pal_after_{name}", f"p=lambda g:{X}"))
            X2=expr.replace('g','[[({d}).get(x,x)for x in r]for r in g')
            C.append((f"pal_before_{name}", f"p=lambda g:{X2}"))
    # bbox fallback
    C.append(bbox_crop_def())
    # dedup by name preserve order
    seen=set(); out=[]
    for entry in C:
        name,code = entry if isinstance(entry,tuple) else entry
        if name not in seen:
            seen.add(name); out.append((name,code))
    return out

# ---------------- solver with two phases, now returns model info ----------------
def solve_task(task):
    found=[]
    model_used=None
    # Phase A
    for name,code in candidates_phaseA(task):
        if _ok(code,task):
            found.append((_len(code),name,code))
    # Phase B if nothing
    if not found:
        for name,code in candidates_phaseB(task):
            if _ok(code,task):
                found.append((_len(code),name,code))
    if found:
        found.sort(key=lambda x:x[0])
        length,name,best=found[0]
        score=max(1,2500-length)
        return best, score, True, name
    return "p=lambda g:g", 0.001, False, "id"

# ---------------- main ----------------
def main():
    IN="/kaggle/input/google-code-golf-2025"; OUT="/kaggle/working/submission"
    os.makedirs(OUT,exist_ok=True)
    solved=0; score=0.0; ids=[]; model_counts=Counter()
    for n in tqdm(range(1,401),desc="Processing tasks"):
        tid=f"{n:03d}"; path=os.path.join(IN,f"task{tid}.json")
        try:
            with open(path) as f: task=json.load(f)
            code,s,ok,model=solve_task(task)
            if ok: solved+=1; ids.append(n); model_counts[model]+=1
            score+=s
            with open(os.path.join(OUT,f"task{tid}.py"),"w") as w: w.write(code)
        except Exception:
            with open(os.path.join(OUT,f"task{tid}.py"),"w") as w: w.write("p=lambda g:g")
            score+=0.001
    zp="/kaggle/working/submission.zip"
    with zipfile.ZipFile(zp,"w") as z:
        for i in range(1,401):
            fn=os.path.join(OUT,f"task{i:03d}.py")
            if os.path.exists(fn): z.write(fn,arcname=f"task{i:03d}.py")
    print(f"Tasks solved: {solved}/400")
    print(f"Success rate: {solved/400*100:.1f}%")
    print(f"Total score: {score:.3f}")
    print(f"Average score per task: {score/400:.3f}")
    print(f"Solved task numbers: {ids[:20]}{'...' if len(ids)>20 else ''}")
    print("Top models used:", model_counts.most_common(10))
    print("Submission file 'submission.zip' created successfully!")

if __name__=="__main__": main()



import os, json, zipfile
from tqdm import tqdm
from collections import Counter

def _len(s:str)->int: return len(s.encode("utf-8"))
def _ok(code:str,task)->bool:
    try:
        ns={}; exec(code,{},ns); f=ns["p"]
        for ex in task.get("train",[])+task.get("test",[])+task.get("arc-gen",[]):
            if f(ex["input"])!=ex["output"]: return False
        return True
    except:
        return False

def dims(g): return (len(g), len(g[0]) if g else 0)
def flat(g):
    a=[]
    for r in g: a+=r
    return a

def train_colors(task):
    cols=set()
    for e in task.get("train",[]):
        for r in e["input"]:
            cols.update(r)
    return sorted(cols)

def geom_lambdas():
    return [
        ("geom:id","p=lambda g:g"),
        ("geom:flip_v","p=lambda g:g[::-1]"),
        ("geom:flip_h","p=lambda g:[r[::-1]for r in g]"),
        ("geom:trans","p=lambda g:list(map(list,zip(*g)))"),
        ("geom:rot90","p=lambda g:list(map(list,zip(*g[::-1])))"),
        ("geom:rot180","p=lambda g:[r[::-1]for r in g[::-1]]"),
        ("geom:rot270","p=lambda g:list(map(list,zip(*g)))[::-1]"),
    ]

# Enhanced geometry operations
def advanced_geom():
    return [
        ("diag_main","p=lambda g:[[g[i][i] if i<len(g) and i<len(g[0]) else 0 for i in range(max(len(g),len(g[0]) if g else 0))] for _ in range(max(len(g),len(g[0]) if g else 0))]"),
        ("diag_anti","p=lambda g:[[g[i][len(g[0])-1-i] if i<len(g) and len(g[0])-1-i>=0 else 0 for i in range(max(len(g),len(g[0]) if g else 0))] for _ in range(max(len(g),len(g[0]) if g else 0))]"),
        ("top_half","p=lambda g:g[:len(g)//2]"),
        ("bottom_half","p=lambda g:g[len(g)//2:]"),
        ("left_half","p=lambda g:[r[:len(r)//2] for r in g]"),
        ("right_half","p=lambda g:[r[len(r)//2:] for r in g]"),
    ]

# Pattern detection functions
def detect_scaling(task):
    """Detect if output is scaled version of input"""
    results = []
    for scale in [2, 3, 4]:
        valid = True
        for e in task.get("train", []):
            inp, out = e["input"], e["output"]
            if len(out) != len(inp) * scale or (out and len(out[0]) != len(inp[0]) * scale):
                valid = False
                break
            for i in range(len(inp)):
                for j in range(len(inp[0])):
                    val = inp[i][j]
                    for di in range(scale):
                        for dj in range(scale):
                            if out[i*scale + di][j*scale + dj] != val:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if not valid: break
        if valid:
            results.append((f"scale_{scale}", f"p=lambda g:sum([[[x]*{scale} for x in r]*{scale} for r in g],[])"))
    return results

def detect_pattern_fill(task):
    """Detect patterns where specific shapes are filled"""
    results = []
    exs = task.get("train", [])
    
    # Check for rectangle fill patterns
    for color in range(10):
        valid = True
        for e in exs:
            inp, out = e["input"], e["output"]
            if dims(inp) != dims(out):
                valid = False
                break
                
            # Look for rectangular regions of zeros that get filled
            found_rect = False
            for i in range(len(inp)):
                for j in range(len(inp[0])):
                    if inp[i][j] == 0 and out[i][j] == color:
                        # Check if this starts a rectangle
                        w = h = 1
                        while j+w < len(inp[0]) and inp[i][j+w] == 0:
                            w += 1
                        while i+h < len(inp) and all(inp[i+h][j+k] == 0 for k in range(w)):
                            h += 1
                        
                        # Verify the rectangle is filled in output
                        rect_filled = all(out[i+di][j+dj] == color 
                                        for di in range(h) for dj in range(w))
                        if rect_filled:
                            found_rect = True
                            break
                if found_rect: break
            
            if not found_rect:
                valid = False
                break
        
        if valid:
            results.append((f"fill_rect_{color}", 
                          f"p=lambda g:[[{color} if x==0 else x for x in r] for r in g]"))
    
    return results

def detect_majority_color(task):
    """Detect patterns based on majority/minority colors"""
    results = []
    
    for e in task.get("train", []):
        inp, out = e["input"], e["output"]
        if dims(inp) != dims(out):
            continue
            
        # Count colors in input
        color_count = Counter(flat(inp))
        if len(color_count) < 2:
            continue
            
        majority = max(color_count, key=color_count.get)
        minority = min(color_count, key=color_count.get)
        
        # Check if output swaps majority/minority
        swap_works = True
        for i in range(len(inp)):
            for j in range(len(inp[0])):
                inp_val = inp[i][j]
                expected = minority if inp_val == majority else (majority if inp_val == minority else inp_val)
                if out[i][j] != expected:
                    swap_works = False
                    break
            if not swap_works: break
                
        if swap_works:
            results.append((f"swap_maj_min_{majority}_{minority}", 
                          f"p=lambda g:[[{minority} if x=={majority} else ({majority} if x=={minority} else x) for x in r] for r in g]"))
            break
    
    return results

def detect_line_patterns(task):
    """Detect horizontal/vertical line patterns"""
    results = []
    
    # Check for row/column operations
    for e in task.get("train", []):
        inp, out = e["input"], e["output"]
        if dims(inp) != dims(out):
            continue
            
        # Check if each row is replaced by its most common element
        row_mode_works = True
        for i in range(len(inp)):
            row_count = Counter(inp[i])
            if not row_count:
                continue
            mode_val = max(row_count, key=row_count.get)
            if not all(out[i][j] == mode_val for j in range(len(inp[0]))):
                row_mode_works = False
                break
                
        if row_mode_works:
            results.append(("row_mode", 
                          "p=lambda g:[[max(set(r),key=r.count)]*len(r) for r in g]"))
            break
            
        # Check column mode
        col_mode_works = True
        for j in range(len(inp[0])):
            col = [inp[i][j] for i in range(len(inp))]
            col_count = Counter(col)
            if not col_count:
                continue
            mode_val = max(col_count, key=col_count.get)
            if not all(out[i][j] == mode_val for i in range(len(inp))):
                col_mode_works = False
                break
                
        if col_mode_works:
            results.append(("col_mode",
                          "p=lambda g:[[max([g[i][j] for i in range(len(g))],key=[g[i][j] for i in range(len(g))].count) for j in range(len(g[0]))] for _ in range(len(g))]"))
            break
    
    return results

def detect_connected_components(task):
    """Detect operations on connected components"""
    results = []
    
    # Simple connected component coloring
    code = """def p(g):
    H,W=len(g),len(g[0]) if g else 0
    vis=[[False]*W for _ in range(H)]
    comp=[[0]*W for _ in range(H)]
    color=1
    
    def dfs(i,j,val):
        if i<0 or i>=H or j<0 or j>=W or vis[i][j] or g[i][j]!=val: return
        vis[i][j]=True
        comp[i][j]=color
        for di,dj in [(0,1),(1,0),(0,-1),(-1,0)]:
            dfs(i+di,j+dj,val)
    
    for i in range(H):
        for j in range(W):
            if not vis[i][j] and g[i][j]!=0:
                dfs(i,j,g[i][j])
                color+=1
    
    return comp"""
    
    results.append(("connected_comp", code))
    return results

def const_fill(k): return (f"fill:{k}", f"p=lambda g:[[{k}]*len(g[0])for _ in g]")
def replace_zero(k): return (f"rep0:{k}", f"p=lambda g:[[{k} if x==0 else x for x in r]for r in g]")
def replace_nz(k): return (f"repnz:{k}", f"p=lambda g:[[{k} if x else 0 for x in r]for r in g]")
def keep_only(k): return (f"keep:{k}", f"p=lambda g:[[x if x=={k} else 0 for x in r]for r in g]")
def replace_eq(a,b): return (f"repeq:{a}->{b}", f"p=lambda g:[[{b} if x=={a} else x for x in r]for r in g]")

def bbox_crop_def():
    return ("bbox_crop",
            "def p(g):H=len(g);W=len(g[0]);a,b,c,d=H,W,-1,-1;"
            "for i in range(H):"
            " for j,x in enumerate(g[i]):"
            "  if x:"
            "   a=i if i<a else a; b=j if j<b else b; c=i if i>c else c; d=j if j>d else d;"
            "return[g[i][b:d+1]for i in range(a,c+1)]")

def border_pad(k,c):
    return (f"pad:{k},{c}", f"p=lambda g:[[{c}]*(len(g[0])+2*{k})]*{k}+[[{c}]*{k}+r+[{c}]*{k}for r in g]+[[{c}]*(len(g[0])+2*{k})]*{k}")

def strip_border(k):
    return (f"strip:{k}", f"p=lambda g:[r[{k}:len(g[0])-{k}]for r in g[{k}:len(g)-{k}]] if len(g)>2*{k} and len(g[0])>2*{k} else [[0]]")

def roll_shift(di,dj):
    return (f"shift:{di},{dj}",
            f"p=lambda g:[[g[i-{di}][j-{dj}] if 0<=i-{di}<len(g) and 0<=j-{dj}<len(g[0]) else 0 for j in range(len(g[0]))] for i in range(len(g))]")

def swap_code(a,b):
    return (f"swap:{a},{b}",
            f"p=lambda g:[[ {b} if x=={a} else ({a} if x=={b} else x) for x in r]for r in g]")

def segment_code():
    return ("segment",
            "def p(g):"
            "H=len(g);W=len(g[0]);cnt={};"
            "for r in g:"
            " for x in r: cnt[x]=cnt.get(x,0)+1;"
            "bg=max(cnt,key=cnt.get);"
            "a,b,c,d=H,W,-1,-1;"
            "for i in range(H):"
            " for j,x in enumerate(g[i]):"
            "  if x!=bg:"
            "   if i<a:a=i"
            "   if j<b:b=j"
            "   if i>c:c=i"
            "   if j>d:d=j"
            "if c==-1: return [[bg]*W for _ in range(H)];"
            "res=[[bg]*W for _ in range(H)];"
            "for i in range(a,c+1):"
            " res[i][b:d+1]=g[i][b:d+1];"
            "return res")

def segment_shift_code(di,dj):
    return (f"segshift:{di},{dj}",
            "def p(g):"
            "H=len(g);W=len(g[0]);cnt={};"
            "for r in g:"
            " for x in r: cnt[x]=cnt.get(x,0)+1;"
            "bg=max(cnt,key=cnt.get);"
            "a,b,c,d=H,W,-1,-1;"
            "for i in range(H):"
            " for j,x in enumerate(g[i]):"
            "  if x!=bg:"
            "   if i<a:a=i"
            "   if j<b:b=j"
            "   if i>c:c=i"
            "   if j>d:d=j"
            "if c==-1: return [[bg]*W for _ in range(H)];"
            "obj=[row[b:d+1] for row in g[a:c+1]];"
            "o=[[bg]*W for _ in range(H)];"
            f"di={di};dj={dj};"
            "for i,row in enumerate(obj):"
            " for j,x in enumerate(row):"
            "  I=i+a+di;J=j+b+dj;"
            "  if 0<=I<H and 0<=J<W: o[I][J]=x"
            "return o")

def mdl_constant(task):
    outs=[e["output"] for e in task.get("train",[])+task.get("test",[])+task.get("arc-gen",[])]
    if outs and all(o==outs[0] for o in outs):
        return [("const_out", f"p=lambda g:{repr(outs[0])}")]
    return []

def mdl_palette_map(task):
    exs=task.get("train",[])+task.get("test",[])+task.get("arc-gen",[])
    mp={}
    for e in exs:
        a,b=e["input"],e["output"]
        if dims(a)!=dims(b): continue
        for x,y in zip(flat(a),flat(b)):
            if x in mp and mp[x]!=y: return []
            mp[x]=y
    if not mp or all(k==v for k,v in mp.items()): return []
    d="{"+",".join(f"{k}:{v}" for k,v in sorted(mp.items()))+"}"
    return [("palette", f"p=lambda g:[[({d}).get(x,x)for x in r]for r in g]")]

def mdl_bbox_crop(task):
    def bbox(g):
        H=len(g);W=len(g[0]) if H else 0
        a,b,c,d=H,W,-1,-1
        for i in range(H):
            for j,x in enumerate(g[i]):
                if x:
                    if i<a:a=i
                    if j<b:b=j
                    if i>c:c=i
                    if j>d:d=j
        return None if c<0 else (a,b,c,d)
    for e in task.get("train",[])+task.get("test",[])+task.get("arc-gen",[]):
        a,b=e["input"],e["output"]; bb=bbox(a)
        if not bb: return []
        i0,j0,i1,j1=bb
        cut=[row[j0:j1+1] for row in a[i0:i1+1]]
        if cut!=b: return []
    return [bbox_crop_def()]

def mdl_border_pad_strip(task):
    exs=task.get("train",[])+task.get("test",[])+task.get("arc-gen",[])
    def infer_pad(a,b):
        Ha,Wa=dims(a);Hb,Wb=dims(b)
        if Hb<Ha or Wb<Wa:return None
        k=(Hb-Ha)//2
        if Hb!=Ha+2*k or Wb!=Wa+2*k:return None
        for i in range(Ha):
            if b[i+k][k:Wb-k]!=a[i]: return None
        cols=set()
        for i in range(Hb):
            for j in range(Wb):
                if not (k<=i<Hb-k and k<=j<Wb-k): cols.add(b[i][j])
        if len(cols)==1: return (k,next(iter(cols)))
        return None
    def infer_strip(a,b):
        Ha,Wa=dims(a);Hb,Wb=dims(b)
        if Ha<=Hb or Wa<=Wb:return None
        k=(Ha-Hb)//2
        if Hb!=Ha-2*k or Wb!=Wa-2*k:return None
        for i in range(Hb):
            if a[i+k][k:Wa-k]!=b[i]: return None
        return k
    pads=[infer_pad(e["input"],e["output"]) for e in exs]
    strips=[infer_strip(e["input"],e["output"]) for e in exs]
    if all(p is not None for p in pads):
        k,c=pads[0]
        if all(p==(k,c) for p in pads): return [border_pad(k,c)]
    if all(s is not None for s in strips):
        k=strips[0]
        if all(s==k for s in strips): return [strip_border(k)]
    return []

def mdl_shift(task):
    def centroid(g):
        pts=[(i,j) for i,row in enumerate(g) for j,x in enumerate(row) if x]
        if not pts: return (0,0)
        si=sum(i for i,_ in pts); sj=sum(j for _,j in pts); n=len(pts)
        return (round(si/n), round(sj/n))
    pars=[]
    for e in task.get("train",[])+task.get("test",[])+task.get("arc-gen",[]):
        a,b=e["input"],e["output"]
        if dims(a)!=dims(b): return []
        ai,aj=centroid(a); bi,bj=centroid(b); di,dj=bi-ai,bj-aj
        H,W=dims(a)
        ok=True
        for i in range(H):
            for j in range(W):
                I=i-di;J=j-dj
                x=a[I][J] if 0<=I<H and 0<=J<W else 0
                if x!=b[i][j]: ok=False; break
            if not ok: break
        if not ok: return []
        pars.append((di,dj))
    di,dj=pars[0]
    if all(p==(di,dj) for p in pars):
        return [roll_shift(di,dj)]
    return []

def candidates_phaseA(task):
    C=[]
    C+=mdl_constant(task)
    C+=mdl_palette_map(task)
    C+=geom_lambdas()
    C+=advanced_geom()  # NEW
    C+=mdl_bbox_crop(task)
    C+=mdl_border_pad_strip(task)
    C+=mdl_shift(task)
    C+=detect_scaling(task)  # NEW
    C+=detect_pattern_fill(task)  # NEW
    C+=detect_majority_color(task)  # NEW
    C+=detect_line_patterns(task)  # NEW
    
    # Enhanced basic operations
    C+= [("bin","p=lambda g:[[1 if x else 0 for x in r]for r in g]"),
         ("ibin","p=lambda g:[[0 if x==1 else(1 if x==0 else x)for x in r]for r in g]"),
         ("outline","p=lambda g:[[x if any(g[i+di][j+dj]==0 for di,dj in [(0,1),(1,0),(0,-1),(-1,0)] if 0<=i+di<len(g) and 0<=j+dj<len(g[0])) and x!=0 else 0 for j,x in enumerate(r)] for i,r in enumerate(g)]")]
    
    # More color operations
    for k in range(10):  # Expanded to all 10 colors
        C.append(const_fill(k)); C.append(replace_zero(k)); C.append(replace_nz(k)); C.append(keep_only(k))
    
    # Size-based transformations
    C+= [("double_h","p=lambda g:[r+r for r in g]"),
         ("double_v","p=lambda g:g+g"),
         ("half_h","p=lambda g:[r[::2] for r in g]"),
         ("half_v","p=lambda g:g[::2]")]
    
    # dedup by name preserve order
    seen=set(); out=[]
    for name,code in C:
        if name not in seen:
            seen.add(name); out.append((name,code))
    return out

def candidates_phaseB(task):
    C=[]
    cols=train_colors(task)
    
    # More comprehensive swaps
    for i in range(len(cols)):
        for j in range(i+1,len(cols)):
            a=cols[i]; b=cols[j]
            C.append(swap_code(a,b))
    
    # Extended color operations
    for k in cols + [x for x in range(10) if x not in cols]:  # Include all colors
        C.append(keep_only(k)); C.append(const_fill(k)); C.append(replace_zero(k)); C.append(replace_nz(k))
        # Add color increment/decrement
        if k < 9:
            C.append((f"inc_{k}", f"p=lambda g:[[{k+1} if x=={k} else x for x in r] for r in g]"))
        if k > 0:
            C.append((f"dec_{k}", f"p=lambda g:[[{k-1} if x=={k} else x for x in r] for r in g]"))
    
    # Advanced pattern detection
    C += detect_connected_components(task)
    
    # Extended segments and shifts
    C.append(segment_code())
    for di in range(-3,4):  # Extended range
        for dj in range(-3,4):
            if di==0 and dj==0: continue
            C.append(segment_shift_code(di,dj))
            C.append(roll_shift(di,dj))
    
    # More crop operations
    C.append(("crop:ul","p=lambda g:[r[:len(g[0])//2]for r in g[:len(g)//2]]"))
    C.append(("crop:ur","p=lambda g:[r[len(g[0])//2:]for r in g[:len(g)//2]]"))
    C.append(("crop:ll","p=lambda g:[r[:len(g[0])//2]for r in g[len(g)//2:]]"))
    C.append(("crop:lr","p=lambda g:[r[len(g[0])//2:]for r in g[len(g)//2:]]"))
    C.append(("crop:center","p=lambda g:[r[len(r)//4:3*len(r)//4] for r in g[len(g)//4:3*len(g)//4]]"))
    
    # Morphological operations
    C.append(("dilate", """def p(g):
h,w=len(g),len(g[0]) if g else 0
r=[[g[i][j] for j in range(w)] for i in range(h)]
for i in range(h):
 for j in range(w):
  if g[i][j]:
   for di in [-1,0,1]:
    for dj in [-1,0,1]:
     if 0<=i+di<h and 0<=j+dj<w: r[i+di][j+dj]=g[i][j]
return r"""))
    
    # Compose palette with more operations
    pal=mdl_palette_map(task)
    G=geom_lambdas() + advanced_geom()
    if pal:
        d=pal[0][1][pal[0][1].find("{"):pal[0][1].find("}")+1]
        for name,g in G:
            if 'lambda' in g:
                expr=g.split('=',1)[1]
                X=f"[[({d}).get(x,x)for x in r]for r in {expr}]"
                C.append((f"pal_after_{name}", f"p=lambda g:{X}"))
                X2=expr.replace('g',f'[[({d}).get(x,x)for x in r]for r in g]')
                C.append((f"pal_before_{name}", f"p=lambda g:{X2}"))
    
    # Fallbacks
    C.append(bbox_crop_def())
    
    # dedup by name preserve order
    seen=set(); out=[]
    for entry in C:
        name,code = entry if isinstance(entry,tuple) else (entry[0], entry[1])
        if name not in seen:
            seen.add(name); out.append((name,code))
    return out

def solve_task(task):
    found=[]
    model_used=None
    
    # Phase A - Quick common patterns
    for name,code in candidates_phaseA(task):
        if _ok(code,task):
            found.append((_len(code),name,code))
    
    # Phase B - Exhaustive search if nothing found
    if not found:
        for name,code in candidates_phaseB(task):
            if _ok(code,task):
                found.append((_len(code),name,code))
    
    # Phase C - Desperate measures (combinations)
    if not found:
        geoms = geom_lambdas()
        basic_ops = [(f"rep0:{k}", f"p=lambda g:[[{k} if x==0 else x for x in r]for r in g]") for k in range(3)]
        
        # Try geometry + basic operation combinations
        for gname, gcode in geoms[:4]:  # Limit to avoid timeout
            for bname, bcode in basic_ops:
                # Apply geometry then basic op
                combined_name = f"{gname}+{bname}"
                if 'lambda' in gcode and 'lambda' in bcode:
                    gexpr = gcode.split('=',1)[1]
                    bexpr = bcode.split('=',1)[1]
                    combined_code = f"p=lambda g:{bexpr.replace('g', gexpr)}"
                    if _ok(combined_code, task):
                        found.append((_len(combined_code), combined_name, combined_code))
                        break
            if found: break
    
    if found:
        found.sort(key=lambda x:x[0])
        length,name,best=found[0]
        score=max(1,2500-length)
        return best, score, True, name
    return "p=lambda g:g", 0.001, False, "id"

def main():
    IN="/kaggle/input/google-code-golf-2025"; OUT="/kaggle/working/submission"
    os.makedirs(OUT,exist_ok=True)
    solved=0; score=0.0; ids=[]; model_counts=Counter()
    
    for n in tqdm(range(1,401),desc="Processing tasks"):
        tid=f"{n:03d}"; path=os.path.join(IN,f"task{tid}.json")
        try:
            with open(path) as f: task=json.load(f)
            code,s,ok,model=solve_task(task)
            if ok: solved+=1; ids.append(n); model_counts[model]+=1
            score+=s
            with open(os.path.join(OUT,f"task{tid}.py"),"w") as w: w.write(code)
        except Exception as e:
            print(f"Error processing task {tid}: {e}")
            with open(os.path.join(OUT,f"task{tid}.py"),"w") as w: w.write("p=lambda g:g")
            score+=0.001
    
    zp="/kaggle/working/submission.zip"
    with zipfile.ZipFile(zp,"w") as z:
        for i in range(1,401):
            fn=os.path.join(OUT,f"task{i:03d}.py")
            if os.path.exists(fn): 
                z.write(fn,arcname=f"task{i:03d}.py")
    
    print(f"Tasks solved: {solved}/400")
    print(f"Success rate: {solved/400*100:.1f}%")
    print(f"Total score: {score:.3f}")
    print(f"Average score per task: {score/400:.3f}")
    print(f"Solved task numbers: {ids[:20]}{'...' if len(ids)>20 else ''}")
    print("Top models used:", model_counts.most_common(15))
    print("Submission file 'submission.zip' created successfully!")

if __name__=="__main__": main()


import os, json, zipfile
from tqdm import tqdm
from collections import Counter
import functools

def _len(s: str) -> int: 
    return len(s.encode("utf-8"))

def _ok(code: str, task) -> bool:
    try:
        ns = {}
        exec(code, {}, ns)
        f = ns["p"]
        for ex in task.get("train", []) + task.get("test", []) + task.get("arc-gen", []):
            if f(ex["input"]) != ex["output"]: 
                return False
        return True
    except:
        return False

def dims(g): 
    return (len(g), len(g[0]) if g else 0)

def flat(g):
    result = []
    for r in g: 
        result.extend(r)
    return result

def train_colors(task):
    cols = set()
    for e in task.get("train", []):
        for r in e["input"]:
            cols.update(r)
    return sorted(cols)

def geom_lambdas():
    return [
        ("geom:id", "p=lambda g:g"),
        ("geom:flip_v", "p=lambda g:g[::-1]"),
        ("geom:flip_h", "p=lambda g:[r[::-1]for r in g]"),
        ("geom:trans", "p=lambda g:list(map(list,zip(*g)))"),
        ("geom:rot90", "p=lambda g:list(map(list,zip(*g[::-1])))"),
        ("geom:rot180", "p=lambda g:[r[::-1]for r in g[::-1]]"),
        ("geom:rot270", "p=lambda g:list(map(list,zip(*g)))[::-1]"),
    ]

def advanced_geom():
    return [
        ("diag_main", "p=lambda g:[[g[i][i] if i<len(g) and i<len(g[0]) else 0 for i in range(max(len(g),len(g[0]) if g else 0))] for _ in range(max(len(g),len(g[0]) if g else 0))]"),
        ("diag_anti", "p=lambda g:[[g[i][len(g[0])-1-i] if i<len(g) and len(g[0])-1-i>=0 else 0 for i in range(max(len(g),len(g[0]) if g else 0))] for _ in range(max(len(g),len(g[0]) if g else 0))]"),
        ("top_half", "p=lambda g:g[:len(g)//2]"),
        ("bottom_half", "p=lambda g:g[len(g)//2:]"),
        ("left_half", "p=lambda g:[r[:len(r)//2] for r in g]"),
        ("right_half", "p=lambda g:[r[len(r)//2:] for r in g]"),
        ("mirror_h", "p=lambda g:[r+r[::-1] for r in g]"),
        ("mirror_v", "p=lambda g:g+g[::-1]"),
        ("shrink_2x", "p=lambda g:[r[::2] for r in g[::2]]"),
        ("expand_2x", "p=lambda g:sum([[[x]*2 for x in r]*2 for r in g],[])"),
    ]

def detect_scaling(task):
    """Detect if output is scaled version of input"""
    results = []
    for scale in [2, 3, 4, 5]:  # Extended range
        valid = True
        for e in task.get("train", []):
            inp, out = e["input"], e["output"]
            if len(out) != len(inp) * scale or (out and len(out[0]) != len(inp[0]) * scale):
                valid = False
                break
            
            # More efficient scaling check
            for i in range(len(inp)):
                for j in range(len(inp[0])):
                    val = inp[i][j]
                    # Check entire scaled block at once
                    block_valid = all(
                        out[i*scale + di][j*scale + dj] == val 
                        for di in range(scale) 
                        for dj in range(scale)
                    )
                    if not block_valid:
                        valid = False
                        break
                if not valid: break
            if not valid: break
            
        if valid:
            results.append((f"scale_{scale}", f"p=lambda g:sum([[[x]*{scale} for x in r]*{scale} for r in g],[])"))
    return results

def detect_color_counting(task):
    """Detect patterns based on color counting"""
    results = []
    
    for e in task.get("train", []):
        inp, out = e["input"], e["output"]
        
        # Check if output encodes color counts
        inp_colors = Counter(flat(inp))
        out_colors = Counter(flat(out))
        
        # Pattern: replace each color with its count modulo 10
        count_pattern_works = True
        for i in range(len(inp)):
            for j in range(len(inp[0])):
                expected = inp_colors[inp[i][j]] % 10
                if out[i][j] != expected:
                    count_pattern_works = False
                    break
            if not count_pattern_works: break
                
        if count_pattern_works:
            results.append(("color_count_mod", 
                          "p=lambda g:[[sum(sum(r.count(g[i][j]) for r in g) for _ in [1])%10 for j in range(len(g[0]))] for i in range(len(g))]"))
            break
    
    return results

def detect_symmetry_patterns(task):
    """Detect symmetry-based transformations"""
    results = []
    
    for e in task.get("train", []):
        inp, out = e["input"], e["output"]
        if dims(inp) != dims(out):
            continue
            
        # Check for symmetry completion
        h, w = dims(inp)
        
        # Horizontal symmetry completion
        h_sym_works = True
        for i in range(h):
            for j in range(w//2):
                left_val = inp[i][j]
                right_val = inp[i][w-1-j]
                # If one side is 0, fill with the other
                if left_val == 0 and right_val != 0:
                    expected = right_val
                elif right_val == 0 and left_val != 0:
                    expected = left_val
                else:
                    expected = left_val
                    
                if out[i][j] != expected or out[i][w-1-j] != expected:
                    h_sym_works = False
                    break
            if not h_sym_works: break
                
        if h_sym_works:
            results.append(("h_symmetry", 
                          "p=lambda g:[[g[i][j] if g[i][j]!=0 else g[i][len(g[0])-1-j] for j in range(len(g[0]))] for i in range(len(g))]"))
            break
    
    return results

def detect_edge_patterns(task):
    """Detect edge and boundary patterns"""
    results = []
    
    for e in task.get("train", []):
        inp, out = e["input"], e["output"]
        if dims(inp) != dims(out):
            continue
            
        h, w = dims(inp)
        
        # Check for edge highlighting
        edge_works = True
        for i in range(h):
            for j in range(w):
                is_edge = (i == 0 or i == h-1 or j == 0 or j == w-1)
                if is_edge and inp[i][j] != 0:
                    if out[i][j] != 9:  # Highlight edges with color 9
                        edge_works = False
                        break
                else:
                    if out[i][j] != inp[i][j]:
                        edge_works = False
                        break
            if not edge_works: break
                
        if edge_works:
            results.append(("highlight_edges", 
                          "p=lambda g:[[9 if (i==0 or i==len(g)-1 or j==0 or j==len(g[0])-1) and g[i][j]!=0 else g[i][j] for j in range(len(g[0]))] for i in range(len(g))]"))
            break
    
    return results

def detect_pattern_fill(task):
    """Enhanced pattern fill detection"""
    results = []
    exs = task.get("train", [])
    
    # Check for various fill patterns
    for color in range(10):
        # Rectangle fill
        rect_fill_works = True
        for e in exs:
            inp, out = e["input"], e["output"]
            if dims(inp) != dims(out):
                rect_fill_works = False
                break
                
            # Enhanced rectangle detection
            filled_any_rect = False
            for i in range(len(inp)):
                for j in range(len(inp[0])):
                    if inp[i][j] == 0 and out[i][j] == color:
                        # Find largest rectangle starting here
                        max_w = max_h = 0
                        for w in range(1, len(inp[0]) - j + 1):
                            if inp[i][j + w - 1] != 0:
                                break
                            max_w = w
                            
                        for h in range(1, len(inp) - i + 1):
                            if any(inp[i + h - 1][j + k] != 0 for k in range(max_w)):
                                break
                            max_h = h
                            
                        if max_w > 0 and max_h > 0:
                            # Check if this rectangle is filled
                            rect_filled = all(
                                out[i + di][j + dj] == color 
                                for di in range(max_h) 
                                for dj in range(max_w)
                            )
                            if rect_filled:
                                filled_any_rect = True
                                break
                if filled_any_rect: break
            
            if not filled_any_rect:
                rect_fill_works = False
                break
        
        if rect_fill_works:
            results.append((f"fill_rect_{color}", 
                          f"p=lambda g:[[{color} if x==0 else x for x in r] for r in g]"))
            break
    
    return results

def detect_grid_patterns(task):
    """Detect grid-based patterns"""
    results = []
    
    for e in task.get("train", []):
        inp, out = e["input"], e["output"]
        h, w = dims(inp)
        
        # Check for checkerboard pattern
        checkerboard_works = True
        for i in range(h):
            for j in range(w):
                expected = (i + j) % 2
                if out[i][j] != expected:
                    checkerboard_works = False
                    break
            if not checkerboard_works: break
                
        if checkerboard_works:
            results.append(("checkerboard", 
                          "p=lambda g:[[(i+j)%2 for j in range(len(g[0]))] for i in range(len(g))]"))
            break
            
        # Check for grid lines
        grid_works = True
        for i in range(h):
            for j in range(w):
                is_grid_line = (i % 3 == 0) or (j % 3 == 0)  # Every 3rd line
                if is_grid_line:
                    if out[i][j] != 1:  # Grid lines are color 1
                        grid_works = False
                        break
                else:
                    if out[i][j] != inp[i][j]:
                        grid_works = False
                        break
            if not grid_works: break
                
        if grid_works:
            results.append(("grid_lines_3", 
                          "p=lambda g:[[1 if i%3==0 or j%3==0 else g[i][j] for j in range(len(g[0]))] for i in range(len(g))]"))
            break
    
    return results

# ... (keeping all the existing helper functions like const_fill, bbox_crop_def, etc.)

def const_fill(k): 
    return (f"fill:{k}", f"p=lambda g:[[{k}]*len(g[0])for _ in g]")

def replace_zero(k): 
    return (f"rep0:{k}", f"p=lambda g:[[{k} if x==0 else x for x in r]for r in g]")

def replace_nz(k): 
    return (f"repnz:{k}", f"p=lambda g:[[{k} if x else 0 for x in r]for r in g]")

def keep_only(k): 
    return (f"keep:{k}", f"p=lambda g:[[x if x=={k} else 0 for x in r]for r in g]")

def replace_eq(a, b): 
    return (f"repeq:{a}->{b}", f"p=lambda g:[[{b} if x=={a} else x for x in r]for r in g]")

def bbox_crop_def():
    return ("bbox_crop",
            "def p(g):H=len(g);W=len(g[0]);a,b,c,d=H,W,-1,-1;"
            "for i in range(H):"
            " for j,x in enumerate(g[i]):"
            "  if x:"
            "   a=i if i<a else a; b=j if j<b else b; c=i if i>c else c; d=j if j>d else d;"
            "return[g[i][b:d+1]for i in range(a,c+1)]")

def border_pad(k, c):
    return (f"pad:{k},{c}", f"p=lambda g:[[{c}]*(len(g[0])+2*{k})]*{k}+[[{c}]*{k}+r+[{c}]*{k}for r in g]+[[{c}]*(len(g[0])+2*{k})]*{k}")

def strip_border(k):
    return (f"strip:{k}", f"p=lambda g:[r[{k}:len(g[0])-{k}]for r in g[{k}:len(g)-{k}]] if len(g)>2*{k} and len(g[0])>2*{k} else [[0]]")

def roll_shift(di, dj):
    return (f"shift:{di},{dj}",
            f"p=lambda g:[[g[i-{di}][j-{dj}] if 0<=i-{di}<len(g) and 0<=j-{dj}<len(g[0]) else 0 for j in range(len(g[0]))] for i in range(len(g))]")

def swap_code(a, b):
    return (f"swap:{a},{b}",
            f"p=lambda g:[[ {b} if x=={a} else ({a} if x=={b} else x) for x in r]for r in g]")

# ... (keeping other helper functions)

@functools.lru_cache(maxsize=128)
def get_cached_candidates_a():
    """Cache expensive candidate generation"""
    base_candidates = []
    base_candidates += geom_lambdas()
    base_candidates += advanced_geom()
    
    # Enhanced basic operations with better coverage
    base_candidates += [
        ("bin", "p=lambda g:[[1 if x else 0 for x in r]for r in g]"),
        ("ibin", "p=lambda g:[[0 if x==1 else(1 if x==0 else x)for x in r]for r in g]"),
        ("outline", "p=lambda g:[[x if any(g[i+di][j+dj]==0 for di,dj in [(0,1),(1,0),(0,-1),(-1,0)] if 0<=i+di<len(g) and 0<=j+dj<len(g[0])) and x!=0 else 0 for j,x in enumerate(r)] for i,r in enumerate(g)]"),
        ("corners", "p=lambda g:[[g[i][j] if (i in [0,len(g)-1] and j in [0,len(g[0])-1]) else 0 for j in range(len(g[0]))] for i in range(len(g))]"),
        ("center_cross", "p=lambda g:[[g[i][j] if i==len(g)//2 or j==len(g[0])//2 else 0 for j in range(len(g[0]))] for i in range(len(g))]"),
    ]
    
    # More size transformations
    base_candidates += [
        ("double_h", "p=lambda g:[r+r for r in g]"),
        ("double_v", "p=lambda g:g+g"),
        ("triple_h", "p=lambda g:[r+r+r for r in g]"),
        ("quad_corners", "p=lambda g:[[g[i%len(g)][j%len(g[0])] for j in range(len(g[0])*2)] for i in range(len(g)*2)]"),
    ]
    
    return base_candidates

def candidates_phaseA(task):
    C = []
    C += mdl_constant(task)
    C += mdl_palette_map(task) 
    C += get_cached_candidates_a()
    C += mdl_bbox_crop(task)
    C += mdl_border_pad_strip(task)
    C += mdl_shift(task)
    
    # Enhanced pattern detection
    C += detect_scaling(task)
    C += detect_pattern_fill(task)
    C += detect_majority_color(task)
    C += detect_line_patterns(task)
    C += detect_color_counting(task)  # NEW
    C += detect_symmetry_patterns(task)  # NEW
    C += detect_edge_patterns(task)  # NEW
    C += detect_grid_patterns(task)  # NEW
    
    # More color operations with all 10 colors
    for k in range(10):
        C.append(const_fill(k))
        C.append(replace_zero(k))
        C.append(replace_nz(k))
        C.append(keep_only(k))
    
    # Remove duplicates while preserving order
    seen = set()
    result = []
    for name, code in C:
        if name not in seen:
            seen.add(name)
            result.append((name, code))
    return result

def candidates_phaseB(task):
    C = []
    cols = train_colors(task)
    
    # Comprehensive color operations
    all_colors = set(range(10))
    for k in all_colors:
        C.append(keep_only(k))
        C.append(const_fill(k))
        C.append(replace_zero(k))
        C.append(replace_nz(k))
        
        # Enhanced color transformations
        if k < 9:
            C.append((f"inc_{k}", f"p=lambda g:[[{k+1} if x=={k} else x for x in r] for r in g]"))
        if k > 0:
            C.append((f"dec_{k}", f"p=lambda g:[[{k-1} if x=={k} else x for x in r] for r in g]"))
        
        # Conditional replacements
        C.append((f"rep_even_{k}", f"p=lambda g:[[{k} if x%2==0 and x!=0 else x for x in r] for r in g]"))
        C.append((f"rep_odd_{k}", f"p=lambda g:[[{k} if x%2==1 else x for x in r] for r in g]"))
    
    # Extended swaps
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            C.append(swap_code(a, b))
    
    # More comprehensive segments and shifts
    C.append(segment_code())
    for di in range(-4, 5):  # Extended range
        for dj in range(-4, 5):
            if di == 0 and dj == 0: 
                continue
            if abs(di) + abs(dj) <= 6:  # Limit combinations to avoid timeout
                C.append(segment_shift_code(di, dj))
                C.append(roll_shift(di, dj))
    
    # Enhanced crop operations
    crop_ops = [
        ("crop:ul", "p=lambda g:[r[:len(g[0])//2]for r in g[:len(g)//2]]"),
        ("crop:ur", "p=lambda g:[r[len(g[0])//2:]for r in g[:len(g)//2]]"),
        ("crop:ll", "p=lambda g:[r[:len(g[0])//2]for r in g[len(g)//2:]]"),
        ("crop:lr", "p=lambda g:[r[len(g[0])//2:]for r in g[len(g)//2:]]"),
        ("crop:center", "p=lambda g:[r[len(r)//4:3*len(r)//4] for r in g[len(g)//4:3*len(g)//4]]"),
        ("crop:inner", "p=lambda g:[r[1:-1] for r in g[1:-1]] if len(g)>2 and len(g[0])>2 else [[0]]"),
    ]
    C.extend(crop_ops)
    
    # Advanced morphological operations  
    morph_ops = [
        ("dilate", """def p(g):
h,w=len(g),len(g[0]) if g else 0
r=[[g[i][j] for j in range(w)] for i in range(h)]
for i in range(h):
 for j in range(w):
  if g[i][j]:
   for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
    if 0<=i+di<h and 0<=j+dj<w: r[i+di][j+dj]=g[i][j]
return r"""),
        ("erode", """def p(g):
h,w=len(g),len(g[0]) if g else 0
r=[[0]*w for _ in range(h)]
for i in range(h):
 for j in range(w):
  if g[i][j] and all(g[i+di][j+dj] for di,dj in [(-1,0),(1,0),(0,-1),(0,1)] if 0<=i+di<h and 0<=j+dj<w):
   r[i][j]=g[i][j]
return r"""),
    ]
    C.extend(morph_ops)
    
    # Keep existing advanced operations
    C += detect_connected_components(task)
    
    # Enhanced palette compositions
    pal = mdl_palette_map(task)
    G = geom_lambdas() + advanced_geom()
    if pal:
        d = pal[0][1][pal[0][1].find("{"):pal[0][1].find("}") + 1]
        for name, g in G[:8]:  # Limit to avoid timeout
            if 'lambda' in g:
                expr = g.split('=', 1)[1]
                X = f"[[({d}).get(x,x)for x in r]for r in {expr}]"
                C.append((f"pal_after_{name}", f"p=lambda g:{X}"))
                X2 = expr.replace('g', f'[[({d}).get(x,x)for x in r]for r in g]')
                C.append((f"pal_before_{name}", f"p=lambda g:{X2}"))
    
    # Fallbacks
    C.append(bbox_crop_def())
    
    # Remove duplicates while preserving order
    seen = set()
    result = []
    for entry in C:
        name, code = entry if isinstance(entry, tuple) else (entry[0], entry[1])
        if name not in seen:
            seen.add(name)
            result.append((name, code))
    return result

# ... (keeping remaining helper functions like mdl_constant, segment_code, etc.)

def mdl_constant(task):
    outs = [e["output"] for e in task.get("train", []) + task.get("test", []) + task.get("arc-gen", [])]
    if outs and all(o == outs[0] for o in outs):
        return [("const_out", f"p=lambda g:{repr(outs[0])}")]
    return []

def mdl_palette_map(task):
    exs = task.get("train", []) + task.get("test", []) + task.get("arc-gen", [])
    mp = {}
    for e in exs:
        a, b = e["input"], e["output"]
        if dims(a) != dims(b): 
            continue
        for x, y in zip(flat(a), flat(b)):
            if x in mp and mp[x] != y: 
                return []
            mp[x] = y
    if not mp or all(k == v for k, v in mp.items()): 
        return []
    d = "{" + ",".join(f"{k}:{v}" for k, v in sorted(mp.items())) + "}"
    return [("palette", f"p=lambda g:[[({d}).get(x,x)for x in r]for r in g]")]

def detect_majority_color(task):
    """Enhanced majority/minority color detection"""
    results = []
    
    for e in task.get("train", []):
        inp, out = e["input"], e["output"]
        if dims(inp) != dims(out):
            continue
            
        color_count = Counter(flat(inp))
        if len(color_count) < 2:
            continue
            
        majority = max(color_count, key=color_count.get)
        minority = min(color_count, key=color_count.get)
        
        # Check various majority/minority operations
        swap_works = all(
            out[i][j] == (minority if inp[i][j] == majority else (majority if inp[i][j] == minority else inp[i][j]))
            for i in range(len(inp)) for j in range(len(inp[0]))
        )
        
        if swap_works:
            results.append((f"swap_maj_min_{majority}_{minority}", 
                          f"p=lambda g:[[{minority} if x=={majority} else ({majority} if x=={minority} else x) for x in r] for r in g]"))
            break
            
        # Check if majority color is removed
        remove_maj_works = all(
            out[i][j] == (0 if inp[i][j] == majority else inp[i][j])
            for i in range(len(inp)) for j in range(len(inp[0]))
        )
        
        if remove_maj_works:
            results.append((f"remove_majority_{majority}", 
                          f"p=lambda g:[[0 if x=={majority} else x for x in r] for r in g]"))
            break
    
    return results

def detect_line_patterns(task):
    """Enhanced line pattern detection"""
    results = []
    
    for e in task.get("train", []):
        inp, out = e["input"], e["output"]
        if dims(inp) != dims(out):
            continue
            
        h, w = dims(inp)
        
        # Row mode (most common element in each row)
        row_mode_works = True
        for i in range(h):
            if not inp[i]:  # Empty row
                continue
            row_count = Counter(inp[i])
            mode_val = max(row_count, key=row_count.get)
            if not all(out[i][j] == mode_val for j in range(w)):
                row_mode_works = False
                break
                
        if row_mode_works:
            results.append(("row_mode", 
                          "p=lambda g:[[max(set(r),key=r.count) if r else 0]*len(r) for r in g]"))
            break
            
        # Column mode
        col_mode_works = True
        for j in range(w):
            col = [inp[i][j] for i in range(h)]
            if not col:
                continue
            col_count = Counter(col)
            mode_val = max(col_count, key=col_count.get)
            if not all(out[i][j] == mode_val for i in range(h)):
                col_mode_works = False
                break
                
        if col_mode_works:
            results.append(("col_mode",
                          "p=lambda g:[[max([g[i][j] for i in range(len(g))],key=[g[i][j] for i in range(len(g))].count) for j in range(len(g[0]))] for _ in range(len(g))]"))
            break
            
        # Line drawing patterns
        h_lines_work = all(
            out[i][j] == (1 if i % 2 == 0 else inp[i][j])
            for i in range(h) for j in range(w)
        )
        
        if h_lines_work:
            results.append(("h_lines_even", 
                          "p=lambda g:[[1 if i%2==0 else g[i][j] for j in range(len(g[0]))] for i in range(len(g))]"))
            break
    
    return results

def detect_connected_components(task):
    """Enhanced connected component operations"""
    results = []
    
    # Basic connected component labeling
    code = """def p(g):
    H,W=len(g),len(g[0]) if g else 0
    vis=[[False]*W for _ in range(H)]
    comp=[[0]*W for _ in range(H)]
    color=1
    
    def dfs(i,j,val):
        if i<0 or i>=H or j<0 or j>=W or vis[i][j] or g[i][j]!=val: return
        vis[i][j]=True
        comp[i][j]=color
        for di,dj in [(0,1),(1,0),(0,-1),(-1,0)]:
            dfs(i+di,j+dj,val)
    
    for i in range(H):
        for j in range(W):
            if not vis[i][j] and g[i][j]!=0:
                dfs(i,j,g[i][j])
                color+=1
    
    return comp"""
    
    # Component size-based operations
    size_code = """def p(g):
    H,W=len(g),len(g[0]) if g else 0
    vis=[[False]*W for _ in range(H)]
    result=[[0]*W for _ in range(H)]
    
    def dfs(i,j,val):
        if i<0 or i>=H or j<0 or j>=W or vis[i][j] or g[i][j]!=val: return []
        vis[i][j]=True
        coords=[(i,j)]
        for di,dj in [(0,1),(1,0),(0,-1),(-1,0)]:
            coords.extend(dfs(i+di,j+dj,val))
        return coords
    
    for i in range(H):
        for j in range(W):
            if not vis[i][j] and g[i][j]!=0:
                coords=dfs(i,j,g[i][j])
                size=len(coords)
                for ci,cj in coords:
                    result[ci][cj]=size%10
    
    return result"""
    
    results.append(("connected_comp", code))
    results.append(("comp_size", size_code))
    return results

def segment_code():
    return ("segment",
            "def p(g):"
            "H=len(g);W=len(g[0]);cnt={};"
            "for r in g:"
            " for x in r: cnt[x]=cnt.get(x,0)+1;"
            "bg=max(cnt,key=cnt.get);"
            "a,b,c,d=H,W,-1,-1;"
            "for i in range(H):"
            " for j,x in enumerate(g[i]):"
            "  if x!=bg:"
            "   if i<a:a=i"
            "   if j<b:b=j"
            "   if i>c:c=i"
            "   if j>d:d=j"
            "if c==-1: return [[bg]*W for _ in range(H)];"
            "res=[[bg]*W for _ in range(H)];"
            "for i in range(a,c+1):"
            " res[i][b:d+1]=g[i][b:d+1];"
            "return res")

def segment_shift_code(di, dj):
    return (f"segshift:{di},{dj}",
            "def p(g):"
            "H=len(g);W=len(g[0]);cnt={};"
            "for r in g:"
            " for x in r: cnt[x]=cnt.get(x,0)+1;"
            "bg=max(cnt,key=cnt.get);"
            "a,b,c,d=H,W,-1,-1;"
            "for i in range(H):"
            " for j,x in enumerate(g[i]):"
            "  if x!=bg:"
            "   if i<a:a=i"
            "   if j<b:b=j"
            "   if i>c:c=i"
            "   if j>d:d=j"
            "if c==-1: return [[bg]*W for _ in range(H)];"
            "obj=[row[b:d+1] for row in g[a:c+1]];"
            "o=[[bg]*W for _ in range(H)];"
            f"di={di};dj={dj};"
            "for i,row in enumerate(obj):"
            " for j,x in enumerate(row):"
            "  I=i+a+di;J=j+b+dj;"
            "  if 0<=I<H and 0<=J<W: o[I][J]=x"
            "return o")

def mdl_bbox_crop(task):
    def bbox(g):
        H = len(g)
        W = len(g[0]) if H else 0
        a, b, c, d = H, W, -1, -1
        for i in range(H):
            for j, x in enumerate(g[i]):
                if x:
                    if i < a: a = i
                    if j < b: b = j  
                    if i > c: c = i
                    if j > d: d = j
        return None if c < 0 else (a, b, c, d)
        
    for e in task.get("train", []) + task.get("test", []) + task.get("arc-gen", []):
        a, b = e["input"], e["output"]
        bb = bbox(a)
        if not bb: 
            return []
        i0, j0, i1, j1 = bb
        cut = [row[j0:j1+1] for row in a[i0:i1+1]]
        if cut != b: 
            return []
    return [bbox_crop_def()]

def mdl_border_pad_strip(task):
    exs = task.get("train", []) + task.get("test", []) + task.get("arc-gen", [])
    
    def infer_pad(a, b):
        Ha, Wa = dims(a)
        Hb, Wb = dims(b)
        if Hb < Ha or Wb < Wa: 
            return None
        k = (Hb - Ha) // 2
        if Hb != Ha + 2*k or Wb != Wa + 2*k: 
            return None
        for i in range(Ha):
            if b[i+k][k:Wb-k] != a[i]: 
                return None
        cols = set()
        for i in range(Hb):
            for j in range(Wb):
                if not (k <= i < Hb-k and k <= j < Wb-k): 
                    cols.add(b[i][j])
        if len(cols) == 1: 
            return (k, next(iter(cols)))
        return None
    
    def infer_strip(a, b):
        Ha, Wa = dims(a)
        Hb, Wb = dims(b)
        if Ha <= Hb or Wa <= Wb: 
            return None
        k = (Ha - Hb) // 2
        if Hb != Ha - 2*k or Wb != Wa - 2*k: 
            return None
        for i in range(Hb):
            if a[i+k][k:Wa-k] != b[i]: 
                return None
        return k
    
    pads = [infer_pad(e["input"], e["output"]) for e in exs]
    strips = [infer_strip(e["input"], e["output"]) for e in exs]
    
    if all(p is not None for p in pads):
        k, c = pads[0]
        if all(p == (k, c) for p in pads): 
            return [border_pad(k, c)]
            
    if all(s is not None for s in strips):
        k = strips[0]
        if all(s == k for s in strips): 
            return [strip_border(k)]
    return []

def mdl_shift(task):
    def centroid(g):
        pts = [(i, j) for i, row in enumerate(g) for j, x in enumerate(row) if x]
        if not pts: 
            return (0, 0)
        si = sum(i for i, _ in pts)
        sj = sum(j for _, j in pts)
        n = len(pts)
        return (round(si/n), round(sj/n))
    
    pars = []
    for e in task.get("train", []) + task.get("test", []) + task.get("arc-gen", []):
        a, b = e["input"], e["output"]
        if dims(a) != dims(b): 
            return []
        ai, aj = centroid(a)
        bi, bj = centroid(b)
        di, dj = bi - ai, bj - aj
        H, W = dims(a)
        ok = True
        for i in range(H):
            for j in range(W):
                I = i - di
                J = j - dj
                x = a[I][J] if 0 <= I < H and 0 <= J < W else 0
                if x != b[i][j]:
                    ok = False
                    break
            if not ok: 
                break
        if not ok: 
            return []
        pars.append((di, dj))
        
    if pars:
        di, dj = pars[0]
        if all(p == (di, dj) for p in pars):
            return [roll_shift(di, dj)]
    return []

def solve_task(task):
    """Enhanced solver with better phase management"""
    found = []
    
    # Phase A - Fast common patterns
    candidates_a = candidates_phaseA(task)
    for name, code in candidates_a:
        if _ok(code, task):
            found.append((_len(code), name, code))
            if len(found) >= 5:  # Early termination for efficiency
                break
    
    # Phase B - More comprehensive search if needed
    if not found:
        candidates_b = candidates_phaseB(task)
        for name, code in candidates_b:
            if _ok(code, task):
                found.append((_len(code), name, code))
                if len(found) >= 3:  # Limit to avoid timeout
                    break
    
    # Phase C - Smart combinations (improved)
    if not found:
        geoms = geom_lambdas()[:6]  # Limit geometry operations
        basic_ops = [
            (f"rep0:{k}", f"p=lambda g:[[{k} if x==0 else x for x in r]for r in g]") 
            for k in [1, 2, 3, 7, 8, 9]  # Most common replacement colors
        ]
        
        # Try high-probability combinations
        for gname, gcode in geoms:
            if found: 
                break
            for bname, bcode in basic_ops:
                combined_name = f"{gname}+{bname}"
                if 'lambda' in gcode and 'lambda' in bcode:
                    try:
                        gexpr = gcode.split('=', 1)[1]
                        bexpr = bcode.split('=', 1)[1] 
                        combined_code = f"p=lambda g:{bexpr.replace('g', gexpr)}"
                        if _ok(combined_code, task):
                            found.append((_len(combined_code), combined_name, combined_code))
                            break
                    except:
                        continue
        
        # Try palette + geometry combinations
        if not found:
            pal = mdl_palette_map(task)
            if pal:
                d = pal[0][1][pal[0][1].find("{"):pal[0][1].find("}")+1]
                for gname, gcode in geoms[:4]:
                    if 'lambda' in gcode:
                        try:
                            expr = gcode.split('=', 1)[1]
                            combined_code = f"p=lambda g:[[({d}).get(x,x)for x in r]for r in {expr}]"
                            if _ok(combined_code, task):
                                found.append((_len(combined_code), f"pal+{gname}", combined_code))
                                break
                        except:
                            continue
    
    # Return best solution
    if found:
        found.sort(key=lambda x: (x[0], x[1]))  # Sort by length, then name
        length, name, best = found[0]
        score = max(1, 2500 - length)
        return best, score, True, name
    
    return "p=lambda g:g", 0.001, False, "id"

def main():
    """Enhanced main function with better error handling and reporting"""
    IN = "/kaggle/input/google-code-golf-2025"
    OUT = "/kaggle/working/submission"
    os.makedirs(OUT, exist_ok=True)
    
    solved = 0
    score = 0.0
    ids = []
    model_counts = Counter()
    errors = []
    
    for n in tqdm(range(1, 401), desc="Processing tasks", ncols=100):
        tid = f"{n:03d}"
        path = os.path.join(IN, f"task{tid}.json")
        
        try:
            with open(path) as f: 
                task = json.load(f)
            
            code, s, ok, model = solve_task(task)
            
            if ok: 
                solved += 1
                ids.append(n)
                model_counts[model] += 1
            
            score += s
            
            with open(os.path.join(OUT, f"task{tid}.py"), "w") as w: 
                w.write(code)
                
        except Exception as e:
            error_msg = f"Task {tid}: {str(e)[:100]}"
            errors.append(error_msg)
            print(f"Error processing task {tid}: {e}")
            
            # Write fallback solution
            with open(os.path.join(OUT, f"task{tid}.py"), "w") as w: 
                w.write("p=lambda g:g")
            score += 0.001
    
    # Create submission zip
    zp = "/kaggle/working/submission.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for i in range(1, 401):
            fn = os.path.join(OUT, f"task{i:03d}.py")
            if os.path.exists(fn): 
                z.write(fn, arcname=f"task{i:03d}.py")
    
    # Enhanced reporting
    print("=" * 60)
    print("SOLUTION SUMMARY")
    print("=" * 60)
    print(f"Tasks solved: {solved}/400 ({solved/400*100:.1f}%)")
    print(f"Total score: {score:.3f}")
    print(f"Average score per task: {score/400:.3f}")
    print(f"Average score per solved task: {score/max(1,solved):.3f}")
    
    print(f"\nSolved task IDs (first 30): {ids[:30]}")
    if len(ids) > 30:
        print(f"... and {len(ids)-30} more")
    
    print(f"\nTop 15 models used:")
    for model, count in model_counts.most_common(15):
        print(f"  {model}: {count} ({count/solved*100:.1f}%)" if solved > 0 else f"  {model}: {count}")
    
    if errors:
        print(f"\nErrors encountered: {len(errors)}")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors)-5} more errors")
    
    print(f"\nSubmission file 'submission.zip' created successfully!")
    print("=" * 60)

if __name__ == "__main__": 
    main()

