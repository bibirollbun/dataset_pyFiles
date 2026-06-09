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


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ==== ARC DEBUG DIAGNOSTICS (run this first) ====
import os, json
from pathlib import Path

EVAL_CANDIDATES = [
    Path("/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json"),
    Path("/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json"),
    Path("./arc-agi-2/data/evaluation"),
    Path("./evaluation"),
    Path("./test_data"),
    Path("./")
]

print("PWD:", os.getcwd())
print("Dir listing (root):", os.listdir("/kaggle") if os.path.exists("/kaggle") else "no /kaggle")
print("Dir listing (input):", os.listdir("/kaggle/input") if os.path.exists("/kaggle/input") else "no /kaggle/input")
print("Dir listing (working):", os.listdir("/kaggle/working") if os.path.exists("/kaggle/working") else "no /kaggle/working")

found = None
for p in EVAL_CANDIDATES:
    if p.exists() and p.suffix == ".json":
        print("✅ Found JSON:", p); found = ("file", p); break
for p in EVAL_CANDIDATES:
    if found: break
    if p.exists() and p.is_dir():
        js = list(p.glob("*.json"))
        if js:
            print("✅ Found folder:", p, "with", len(js), "jsons"); found = ("folder", p); break

if not found:
    print("⚠️ No evaluation data found. You must add the dataset:")
    print("  -> Add Data ➜ 'arc-prize-2025' (official competition dataset)")
else:
    kind, path = found
    if kind == "file":
        with open(path, "r") as f:
            data = json.load(f)
        print(f"Loaded JSON: {path}  | type: {type(data).__name__} | entries:",
              len(data) if hasattr(data, "__len__") else "n/a")
    else:
        js = list(Path(path).glob("*.json"))
        print(f"Loaded folder: {path} | json files:", len(js))


"""
ARC — Controlled Learning (VRPT Nexus Horizon, v4 AUTODETECT)
- Auto-load priors / accepts / codebooks from /kaggle/input/*
- Thinker Profiles + two gates (inset_border, centerline_mirror)
- CC-first rare-color anchor, tiny Stage-A, stripe probe, RLE prefs, CAG guard
- No brute force; verbose logging
"""

import json, os, re, csv
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

# -----------------------------
# Paths
# -----------------------------
EVAL_CANDIDATES = [
    Path("/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json"),
    Path("/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json"),
    Path("./arc-agi-2/data/evaluation"),
    Path("./evaluation"),
    Path("./test_data"),
    Path("./"),
]
OUT_PATH = Path("./submission.json")

# -----------------------------
# Defaults (safe if nothing uploaded)
# -----------------------------
DEFAULT_AXIS_PRIOR  = {"col": 0.66, "row": 0.34}
DEFAULT_OFFSET_PRIOR = {0: 0.33, 1: 0.33, -1: 0.22, 2: 0.12}
DEFAULT_PROFILES = {
  "mirror_maker":       {"trigger":"symmetry_detected","axis_bias":{"col":0.7,"row":0.3},"stageA_order":["mirror","identity","center_shift+1","center_shift-1"],"gates":["centerline_mirror"]},
  "stripe_painter":     {"trigger":"periodicity_detected","axis_bias":{"col":0.75,"row":0.25},"stageA_order":["identity","center_shift+1","center_shift-1","mirror"],"gates":["inset_border"]},
  "border_cleaner":     {"trigger":"border_salient","axis_bias":{"col":0.6,"row":0.4},"stageA_order":["identity","mirror","center_shift+1","center_shift-1"],"gates":["inset_border"]},
  "default_object_first":{"trigger":"fallback","axis_bias":{"col":0.66,"row":0.34},"stageA_order":["identity","mirror","center_shift+1","center_shift-1"],"gates":[]}
}

# -----------------------------
# Utils
# -----------------------------
def to_np(x): return np.array(x, dtype=int)

def diff_mask_aligned(inp,out):
    A=to_np(inp); B=to_np(out)
    H,W=A.shape
    h=min(H,B.shape[0]); w=min(W,B.shape[1])
    dm=np.zeros((H,W), dtype=bool)
    dm[:h,:w] = (A[:h,:w]!=B[:h,:w])
    return dm

def color_hist(a):
    vals, cnts = np.unique(a, return_counts=True)
    return dict(zip([int(v) for v in vals], [int(c) for c in cnts]))

# -----------------------------
# Find evaluation data
# -----------------------------
def find_eval() -> Tuple[Optional[str], Optional[Path]]:
    for p in EVAL_CANDIDATES:
        if p.exists() and p.suffix == ".json":
            return "file", p
    for p in EVAL_CANDIDATES:
        if p.exists() and p.is_dir():
            files = list(p.glob("*.json"))
            if files: return "folder", p
    for p in Path(".").glob("*.json"):
        try:
            with open(p, "r") as f: json.load(f)
            return "file", p
        except: pass
    return None, None

# -----------------------------
# Autoload helper: scan /kaggle/input/*
# -----------------------------
def scan_inputs() -> Dict[str, List[Path]]:
    base = Path("/kaggle/input")
    found = {"json":[], "csv":[]}
    if not base.exists(): return found
    for ds in base.iterdir():
        try:
            for p in ds.rglob("*"):
                if p.is_file():
                    if p.suffix.lower()==".json": found["json"].append(p)
                    if p.suffix.lower()==".csv":  found["csv"].append(p)
        except Exception:
            continue
    return found

def lowercase(s): return s.lower()

def autoload_priors_and_profiles() -> Tuple[Dict[str,float], Dict[int,float], Dict[str,dict], Dict[str,float], Dict[int,int]]:
    """
    Returns:
      axis_prior, offset_prior, profiles, profile_prior_weights, rle_bias_runs
    """
    axis_prior  = DEFAULT_AXIS_PRIOR.copy()
    offset_prior= DEFAULT_OFFSET_PRIOR.copy()
    profiles    = DEFAULT_PROFILES.copy()
    profile_prior_weights = {}   # e.g., {'mirror_maker': 12, ...}
    rle_bias_runs = {}           # per-color mean run length if provided

    hits = scan_inputs()
    print(f"[AUTO] Found {len(hits['json'])} JSON and {len(hits['csv'])} CSV under /kaggle/input")

    # JSON files
    for jpath in hits["json"]:
        name = lowercase(jpath.name)
        try:
            if "thinker_profiles_priors" in name:
                with open(jpath, "r") as f: profiles = json.load(f)
                print("[AUTO] Loaded Thinker Profiles from:", jpath)
            if "vrpt_nexus_horizon_archetype_priors_updated" in name or "archetype_priors" in name:
                with open(jpath, "r") as f: j = json.load(f)
                if isinstance(j.get("axis_prior"), dict):
                    axis_prior = {k: float(v) for k,v in j["axis_prior"].items()}
                if isinstance(j.get("offset_prior"), dict):
                    offset_prior = {int(k): float(v) for k,v in j["offset_prior"].items()}
                print("[AUTO] Loaded archetype priors from:", jpath)
        except Exception as e:
            print("[AUTO][JSON] skip", jpath, "err:", e)

    # CSV files — soft parse by column names
    axis_counts = {"row":0, "col":0}
    offset_counts = {}
    profile_counts= {}
    color_run_sum = {}
    color_run_n   = {}

    def upd_off(v):
        try:
            vi = int(float(v))
            offset_counts[vi] = offset_counts.get(vi, 0) + 1
        except: pass

    def upd_axis(v):
        s = lowercase(str(v))
        if "row" in s: axis_counts["row"] += 1
        if "col" in s: axis_counts["col"] += 1

    for cpath in hits["csv"]:
        name = lowercase(cpath.name)
        # We only bother if filename hints relevance
        if not re.search(r"(accept|archetype|prior|controlled|codebook|anchor|delta)", name):
            continue
        try:
            # light CSV scan
            with open(cpath, newline="") as f:
                reader = csv.DictReader(f)
                cols = [lowercase(c) for c in reader.fieldnames or []]
                used=False
                for row in reader:
                    used=True
                    # axis / offset / profile priors
                    for key in ["axis","axis_mode","axis_guess"]:
                        if key in cols and row.get(key) not in (None, ""):
                            upd_axis(row.get(key))
                            break
                    for key in ["offset","offsets","band_offset","delta_offset"]:
                        if key in cols and row.get(key) not in (None, ""):
                            # may be list-like; split on , or space
                            val = row.get(key)
                            parts = re.split(r"[ ,;]+", str(val))
                            for p in parts:
                                upd_off(p)
                            break
                    for key in ["profile","thinker","archetype"]:
                        if key in cols and row.get(key) not in (None, ""):
                            profile = lowercase(str(row.get(key))).strip()
                            profile_counts[profile] = profile_counts.get(profile,0)+1
                            break
                    # codebook-style color runs
                    if "color" in cols and "run_len" in cols:
                        try:
                            c = int(row.get("color"))
                            r = float(row.get("run_len"))
                            color_run_sum[c] = color_run_sum.get(c,0.0)+r
                            color_run_n[c]   = color_run_n.get(c,0)+1
                        except: pass
                if used:
                    print("[AUTO] Parsed:", cpath)
        except Exception as e:
            print("[AUTO][CSV] skip", cpath, "err:", e)

    # Blend axis prior (normalize with small epsilon to keep defaults)
    tot = axis_counts["row"] + axis_counts["col"]
    if tot > 0:
        axis_prior = {
            "row": round((axis_counts["row"] + 0.5)/(tot + 1.0), 4),
            "col": round((axis_counts["col"] + 0.5)/(tot + 1.0), 4),
        }
        print("[AUTO] axis_prior from CSV:", axis_prior)

    # Blend offset prior
    if offset_counts:
        s = sum(offset_counts.values())
        offset_prior = {int(k): round(v/s, 4) for k,v in sorted(offset_counts.items(), key=lambda kv: -kv[1])}
        print("[AUTO] offset_prior from CSV (top 8):", dict(list(offset_prior.items())[:8]))

    # Profile prior weights (used to order selection)
    if profile_counts:
        profile_prior_weights = {k: int(v) for k,v in sorted(profile_counts.items(), key=lambda kv: -kv[1])}
        print("[AUTO] profile prior hints:", profile_prior_weights)

    # RLE run-length bias per color
    for c in list(color_run_sum.keys()):
        rle_bias_runs[int(c)] = int(round(color_run_sum[c]/max(1, color_run_n[c])))
    if rle_bias_runs:
        print("[AUTO] RLE color run bias (means):", rle_bias_runs)

    return axis_prior, offset_prior, profiles, profile_prior_weights, rle_bias_runs

# -----------------------------
# CC + rare-color anchor
# -----------------------------
def cc_by_color(a, target_color):
    H,W=a.shape
    seen=np.zeros_like(a, dtype=bool)
    comps=[]
    for y in range(H):
        for x in range(W):
            if seen[y,x] or a[y,x]!=target_color: continue
            stack=[(y,x)]; seen[y,x]=True; comp=[]
            while stack:
                cy,cx=stack.pop(); comp.append((cy,cx))
                for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                    ny,nx=cy+dy,cx+dx
                    if 0<=ny<H and 0<=nx<W and not seen[ny,nx] and a[ny,nx]==target_color:
                        seen[ny,nx]=True; stack.append((ny,nx))
            comps.append(comp)
    return comps

def choose_anchor_via_rarity(inp, dm):
    a=to_np(inp)
    region = a[dm] if dm.any() else a.reshape(-1)
    h = color_hist(region) or color_hist(a)
    for col,_ in sorted(h.items(), key=lambda kv: (kv[1], kv[0])):  # rarest first
        comps = cc_by_color(a, col)
        if comps:
            comp = max(comps, key=len)
            ys=[p[0] for p in comp]; xs=[p[1] for p in comp]
            cy=int(np.mean(ys)); cx=int(np.mean(xs))
            axis = ("col", cx) if (max(xs)-min(xs)) >= (max(ys)-min(ys)) else ("row", cy)
            return {"pixels": comp, "centroid": (cy,cx), "axis": axis}, col
    return None, None

# -----------------------------
# Learning & gates (as before)
# -----------------------------
def learn_anchor_locked_mask_with_priors(train_pairs, axis_prior, offset_prior, axis_bias=None):
    from collections import Counter as C
    offset_counts = C(); axis_mode_counts = {"row":0,"col":0}
    for pr in train_pairs:
        inp=to_np(pr["input"]); out=to_np(pr["output"]); dm=diff_mask_aligned(inp,out)
        anchor,_ = choose_anchor_via_rarity(inp, dm)
        if anchor is None: continue
        axis, idx = anchor["axis"]; axis_mode_counts[axis]+=1
        H,W=inp.shape
        if axis=="row":
            for y in range(H):
                if dm[y,:].any(): offset_counts[("row", y-idx)] += 1
        else:
            for x in range(W):
                if dm[:,x].any(): offset_counts[("col", x-idx)] += 1
    # combine dataset priors + thinker bias
    row_c = axis_mode_counts["row"] + (axis_bias.get("row",0) if axis_bias else 0)
    col_c = axis_mode_counts["col"] + (axis_bias.get("col",0) if axis_bias else 0)
    col_bias = axis_prior.get("col", 0.5); row_bias = axis_prior.get("row", 0.5)
    axis = "col" if (col_c + col_bias) >= (row_c + row_bias) else "row"
    offs = [o for (ax,o),_ in offset_counts.items() if ax==axis]
    prior_favs = sorted(offset_prior.keys(), key=lambda k: -offset_prior[k])[:2]
    cand = {o: offset_counts[(axis,o)] for o in offs}
    for po in prior_favs: cand.setdefault(po, 0.1*offset_prior[po])
    chosen_offsets = [o for o,_ in sorted(cand.items(), key=lambda kv:-kv[1])[:3]] or [0]
    return {"axis": axis, "offsets": chosen_offsets}

def apply_inset_border(mask, inset=1):
    H,W = mask.shape
    m = np.zeros_like(mask, dtype=bool)
    y0,y1 = inset, max(inset, H-inset)
    x0,x1 = inset, max(inset, W-inset)
    if y1>y0 and x1>x0:
        m[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return m

def apply_centerline_mirror(mask, axis, center_idx):
    m = np.zeros_like(mask, dtype=bool)
    H,W = mask.shape
    if axis=="row":
        for y in range(H):
            my = center_idx - (y - center_idx)
            if 0<=y<H and 0<=my<H:
                m[y,:] = mask[y,:] | mask[my,:]
    else:
        for x in range(W):
            mx = center_idx - (x - center_idx)
            if 0<=x<W and 0<=mx<W:
                m[:,x] = mask[:,x] | mask[:,mx]
    return m

def detect_periodicity_along_axis(out, axis, mask):
    out = to_np(out); H,W=out.shape
    periods=[2,3,4]; best=None; best_score=-1
    if axis=="row":
        for p in periods:
            if H - p <= 0: continue
            score=0; den=0
            for y in range(0,H-p):
                mm = (mask[y,:] & mask[y+p,:]) if y<mask.shape[0] and (y+p)<mask.shape[0] else None
                if mm is not None and mm.any():
                    score += (out[y, mm] == out[y+p, mm]).mean(); den+=1
            if den>0 and score/den>best_score: best_score=score/den; best=p
    else:
        for p in periods:
            if W - p <= 0: continue
            score=0; den=0
            for x in range(0,W-p):
                mm = (mask[:,x] & mask[:,x+p]) if x<mask.shape[1] and (x+p)<mask.shape[1] else None
                if mm is not None and mm.any():
                    score += (out[mm, x] == out[mm, x+p]).mean(); den+=1
            if den>0 and score/den>best_score: best_score=score/den; best=p
    return best if best_score>=0.7 else None

def build_anchor_mask(model, grid):
    a=to_np(grid); H,W=a.shape
    m=np.zeros_like(a, dtype=bool)
    axis=model["axis"]; offs=model["offsets"]
    anchor,_=choose_anchor_via_rarity(a, np.zeros_like(a, dtype=bool))
    if anchor is None: return m, ("row",0)
    ax, idx = anchor["axis"]
    if axis=="row":
        for off in offs:
            y = idx + off
            if 0<=y<H: m[y,:]=True
    else:
        for off in offs:
            x = idx + off
            if 0<=x<W: m[:,x]=True
    return m, (ax, idx)

def learn_band_templates(train_pairs, anchor_mask_model, gates):
    templates = {}; stripe=None
    for pr in train_pairs:
        inp=to_np(pr["input"]); out=to_np(pr["output"])
        m, (ax,idx) = build_anchor_mask(anchor_mask_model, inp)
        if "inset_border" in gates: m = apply_inset_border(m, inset=1)
        if "centerline_mirror" in gates: m = apply_centerline_mirror(m, ax, idx)
        H,W=inp.shape; Ho,Wo=out.shape
        mh,mw = min(H,Ho), min(W,Wo)
        outc=out[:mh,:mw]; mc=m[:mh,:mw]
        if stripe is None:
            stripe = detect_periodicity_along_axis(outc, anchor_mask_model["axis"], mc)
        if anchor_mask_model["axis"]=="row":
            for y in range(mh):
                if mc[y,:].any():
                    idxs=np.where(mc[y,:])[0]
                    if stripe:
                        seq=outc[y, idxs].tolist()
                        templates[("row",y)] = ("stripe", stripe, seq[:stripe] if len(seq)>=stripe else seq)
                    else:
                        vals=outc[y, idxs]
                        templates[("row",y)] = ("flat", int(np.bincount(vals).argmax()))
        else:
            for x in range(mw):
                if mc[:,x].any():
                    idxs=np.where(mc[:,x])[0]
                    if stripe:
                        seq=outc[idxs, x].tolist()
                        templates[("col",x)] = ("stripe", stripe, seq[:stripe] if len(seq)>=stripe else seq)
                    else:
                        vals=outc[idxs, x]
                        templates[("col",x)] = ("flat", int(np.bincount(vals).argmax()))
    return templates

def rle(line):
    runs=[]; cur=line[0]; k=1
    for z in line[1:]:
        if z==cur: k+=1
        else: runs.append((int(cur),k)); cur=z; k=1
    runs.append((int(cur),k)); return runs

def learn_rle_prefs(train_pairs, rle_bias_runs=None):
    from collections import Counter
    row_runs=Counter(); col_runs=Counter()
    row_len=Counter(); col_len=Counter()
    for pr in train_pairs:
        out=to_np(pr["output"])
        for y in range(out.shape[0]):
            for c,k in rle(out[y,:]): row_runs[c]+=k; row_len[c]+=1
        for x in range(out.shape[1]):
            for c,k in rle(out[:,x]): col_runs[c]+=k; col_len[c]+=1
    row_pref={c:int(row_runs[c]/max(1,row_len[c])) for c in row_runs}
    col_pref={c:int(col_runs[c]/max(1,col_len[c])) for c in col_runs}
    # blend uploaded bias (per color mean runs)
    if rle_bias_runs:
        for c,mean_run in rle_bias_runs.items():
            row_pref[c] = max(row_pref.get(c,0), int(mean_run))
            col_pref[c] = max(col_pref.get(c,0), int(mean_run))
    return {"row_pref":row_pref, "col_pref":col_pref}

def compute_CAG(arr):
    a=to_np(arr); H,W=a.shape
    from collections import Counter
    E=Counter()
    for y in range(H):
        for x in range(W):
            c=a[y,x]
            if y+1<H: E[(int(c), int(a[y+1,x]))]+=1
            if x+1<W: E[(int(c), int(a[y,x+1]))]+=1
    return E

def symmetry_candidates(out):
    A=to_np(out); ops=set()
    if np.array_equal(np.fliplr(A), A): ops.add("fliplr")
    if np.array_equal(np.flipud(A), A): ops.add("flipud")
    if np.array_equal(A.T, A): ops.add("transpose")
    if np.array_equal(np.rot90(A,1), A) or np.array_equal(np.rot90(A,3), A): ops.add("rot90_sym")
    return ops

def stageA_variants(inp, anchor):
    a=to_np(inp); variants={"identity": a.copy()}
    if anchor is not None:
        ax, idx = anchor["axis"]
        if ax=="row":
            mm=a.copy(); mm[idx+1:,:]=mm[idx+1:,:][::-1,:]; variants["mirror"]=mm
            s1=a.copy(); s1[:idx,:]=np.roll(s1[:idx,:], 1, axis=0); s1[idx+1:,:]=np.roll(s1[idx+1:,:], 1, axis=0)
            s2=a.copy(); s2[:idx,:]=np.roll(s2[:idx,:],-1, axis=0); s2[idx+1:,:]=np.roll(s2[idx+1:,:],-1, axis=0)
            variants["center_shift+1"]=s1; variants["center_shift-1"]=s2
        else:
            mm=a.copy(); mm[:,idx+1:]=mm[:,idx+1:][:,::-1]; variants["mirror"]=mm
            s1=a.copy(); s1[:,:idx]=np.roll(s1[:,:idx], 1, axis=1); s1[:,idx+1:]=np.roll(s1[:,idx+1:], 1, axis=1)
            s2=a.copy(); s2[:,:idx]=np.roll(s2[:,:idx],-1, axis=1); s2[:,idx+1:]=np.roll(s2[:,idx+1:],-1, axis=1)
            variants["center_shift+1"]=s1; variants["center_shift-1"]=s2
    return variants

# -----------------------------
# Thinker profile selection
# -----------------------------
def pick_profile_order(train_pairs, profiles: Dict[str,dict], profile_prior_weights: Dict[str,float]):
    cues={"symmetry_detected": False, "periodicity_detected": False, "border_salient": False}
    # symmetry cue
    for pr in train_pairs:
        if symmetry_candidates(pr["output"]):
            cues["symmetry_detected"]=True; break
    # border cue
    for pr in train_pairs:
        out=to_np(pr["output"]); H,W=out.shape
        border=np.concatenate([out[0,:], out[-1,:], out[:,0], out[:,-1]])
        bh = color_hist(border); oh = color_hist(out.reshape(-1))
        if bh and max(bh.values())/max(1,sum(bh.values())) > 0.4 and max(bh, key=bh.get)==max(oh, key=oh.get):
            cues["border_salient"]=True; break
    # periodicity cue (rough)
    for pr in train_pairs:
        out=to_np(pr["output"])
        for p in (2,3,4):
            if (out.shape[0]>p and np.all(out[:-p,:]==out[p:,:])) or (out.shape[1]>p and np.all(out[:,:-p]==out[:,p:])):
                cues["periodicity_detected"]=True; break
        if cues["periodicity_detected"]: break

    # candidate order from cues
    order=[]
    if cues["symmetry_detected"]: order.append("mirror_maker")
    if cues["periodicity_detected"]: order.append("stripe_painter")
    if cues["border_salient"]: order.append("border_cleaner")
    order.append("default_object_first")

    # blend with prior weights (move heavier profiles earlier if not already)
    if profile_prior_weights:
        rest = [p for p in profiles.keys() if p not in order]
        rest.sort(key=lambda k: -profile_prior_weights.get(k,0))
        order.extend(rest)
    return order, cues

# -----------------------------
# Learn & Predict
# -----------------------------
def learn_model(train_data, axis_prior, offset_prior, profiles, chosen_profile, rle_bias_runs):
    prof = profiles.get(chosen_profile, profiles["default_object_first"])
    axis_bias = prof.get("axis_bias", {})
    aml = learn_anchor_locked_mask_with_priors(train_data, axis_prior, offset_prior, axis_bias=axis_bias)
    templates = learn_band_templates(train_data, aml, prof.get("gates", []))
    rle_prefs = learn_rle_prefs(train_data, rle_bias_runs)
    sym_ops=set()
    for ex in train_data: sym_ops |= symmetry_candidates(ex["output"])
    return {"aml": aml, "templates": templates, "rle": rle_prefs, "sym_ops": sym_ops, "profile": prof}

def predict_for_task(task, model):
    aml=model["aml"]; templates=model["templates"]; rle=model["rle"]; prof=model["profile"]
    order = prof.get("stageA_order", ["identity","mirror","center_shift+1","center_shift-1"])
    preds=[]
    for ex in task.get("test", []):
        inp = to_np(ex["input"] if isinstance(ex, dict) else ex)
        anchor,_=choose_anchor_via_rarity(inp, np.zeros_like(inp, dtype=bool))
        Avars = stageA_variants(inp, anchor)
        best=None; best_score=1e9
        for v in order:
            aA = Avars.get(v, inp.copy())
            pred = fill_with_anchor_model(aA, aml, templates, rle, iters=6)
            score = len(compute_CAG(pred))  # small adjacency regularizer
            if score < best_score:
                best_score = score; best = pred
        preds.append(best.astype(int).tolist())
    return preds

def fill_with_anchor_model(inp, anchor_model, templates, rle_prefs, iters=6):
    a=to_np(inp).copy(); H,W=a.shape
    m,(ax,idx) = build_anchor_mask(anchor_model, inp)
    # lay templates
    if anchor_model["axis"]=="row":
        for y in range(H):
            if m[y,:].any():
                kind = templates.get(("row", y), None)
                if kind is None: continue
                if kind[0]=="flat":
                    a[y, m[y,:]] = kind[1]
                else:
                    _, period, seq = kind
                    idxs = np.where(m[y,:])[0]
                    if seq:
                        for k,xx in enumerate(idxs):
                            a[y, xx] = seq[k % len(seq)]
    else:
        for x in range(W):
            if m[:,x].any():
                kind = templates.get(("col", x), None)
                if kind is None: continue
                if kind[0]=="flat":
                    a[m[:,x], x] = kind[1]
                else:
                    _, period, seq = kind
                    idxs = np.where(m[:,x])[0]
                    if seq:
                        for k,yy in enumerate(idxs):
                            a[yy, x] = seq[k % len(seq)]
    # light inpaint
    for _ in range(iters):
        changed=False
        for y in range(H):
            for x in range(W):
                if not m[y,x]: continue
                props=set([int(a[y,x])])
                if anchor_model["axis"]=="row" and ("row", y) in templates:
                    t=templates[("row", y)]; props.add(int(t[1]) if t[0]=="flat" else int(np.bincount(np.array(t[2], dtype=int)).argmax() if t[2] else a[y,x]))
                if anchor_model["axis"]=="col" and ("col", x) in templates:
                    t=templates[("col", x)]; props.add(int(t[1]) if t[0]=="flat" else int(np.bincount(np.array(t[2], dtype=int)).argmax() if t[2] else a[y,x]))
                for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                    ny,nx=y+dy,x+dx
                    if 0<=ny<H and 0<=nx<W: props.add(int(a[ny,nx]))
                best=int(a[y,x]); best_s=-1e9
                for c in props:
                    s=0.0; like=0; tot=0
                    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny,nx=y+dy,x+dx
                        if 0<=ny<H and 0<=nx<W:
                            tot+=1; like += (a[ny,nx]==c)
                    s += like/max(1,tot)
                    s += 0.05 * rle_prefs["row_pref"].get(c,1)
                    s += 0.05 * rle_prefs["col_pref"].get(c,1)
                    if s>best_s: best_s=s; best=c
                if best!=a[y,x]: a[y,x]=best; changed=True
        if not changed: break
    return a

# -----------------------------
# I/O helpers
# -----------------------------
def load_tasks_from_file(p: Path) -> List[Tuple[str, Dict]]:
    with open(p, "r") as f: data=json.load(f)
    if isinstance(data, dict): return list(data.items())
    if isinstance(data, list): return [(f"task_{i}", t) for i,t in enumerate(data)]
    return []

def load_tasks_from_folder(p: Path) -> List[Tuple[str, Dict]]:
    tasks=[]
    for fp in sorted(p.glob("*.json")):
        try:
            with open(fp,"r") as f: tasks.append((fp.stem, json.load(f)))
        except: pass
    return tasks

# -----------------------------
# Main
# -----------------------------
def main():
    print("PWD:", os.getcwd())
    print("Inputs:", os.listdir("/kaggle/input") if os.path.exists("/kaggle/input") else "n/a")

    # Autoload uploads → priors
    axis_prior, offset_prior, profiles, profile_prior_weights, rle_bias_runs = autoload_priors_and_profiles()
    print("[INFO] axis_prior:", axis_prior)
    print("[INFO] offset_prior (top):", dict(list(sorted(offset_prior.items(), key=lambda kv:-kv[1])[:5])))
    print("[INFO] profiles:", list(profiles.keys()))

    # Find eval
    dtype, dpath = find_eval()
    print("[INFO] eval detection:", dtype, dpath)
    if dtype is None:
        with open(OUT_PATH,"w") as f: json.dump({"placeholder":[{"attempt_1":[[0]],"attempt_2":[[0]]}]}, f, indent=2)
        print("[WARN] No eval data. Wrote placeholder to", OUT_PATH); return

    tasks = load_tasks_from_folder(dpath) if dtype=="folder" else load_tasks_from_file(dpath)
    print(f"[INFO] Loaded {len(tasks)} tasks")

    submission={}
    for idx,(tid,task) in enumerate(tasks):
        if idx%10==0: print(f"[PROGRESS] {idx}/{len(tasks)} … {tid}")
        train_pairs=[{"input":ex["input"],"output":ex["output"]} for ex in task.get("train", []) if "output" in ex]

        # Decide profile order using cues + prior weights
        order, cues = pick_profile_order(train_pairs, profiles, profile_prior_weights)
        # Learn using first profile with non-empty templates; else fall through
        model=None
        for prof_name in order:
            model = learn_model(train_pairs, axis_prior, offset_prior, profiles, prof_name, rle_bias_runs)
            if model["templates"]: break

        preds = predict_for_task(task, model)
        attempts=[{"attempt_1":p, "attempt_2":np.rot90(np.array(p),1).astype(int).tolist()} for p in preds] or [{"attempt_1":[[0]],"attempt_2":[[0]]}]
        submission[tid]=attempts

    with open(OUT_PATH,"w") as f: json.dump(submission, f, indent=2)
    print("[DONE] Wrote submission:", OUT_PATH.resolve())

    # quick sample
    try:
        k = list(submission.keys())[0]
        a1 = np.array(submission[k][0]["attempt_1"]); a2 = np.array(submission[k][0]["attempt_2"])
        print(f"[SAMPLE] task={k}  attempt_1={a1.shape}  attempt_2={a2.shape}")
    except Exception as e:
        print("[NOTE] sample print failed:", e)

if __name__ == "__main__":
    main()

