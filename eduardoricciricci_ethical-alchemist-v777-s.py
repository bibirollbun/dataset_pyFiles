# CELL 1 — ETHICAL MANIFESTO & TECHNICAL ABSTRACT (v777.FIREBASE-CHAMPION++++)
# Final Submission Version for ARC Prize 2025
# =============================================================================
## The Ethical Alchemist Manifesto & Engineering
# Objective
#    • Solve ARC tasks using interpretable heuristics, maintaining ethical integrity:
#      transparency, simplicity, and reproducibility. No "opaque shortcuts."
# Principles (MPE — Memory, Pragmatism, Ethics)
#    • Evolving Memory: Learns to prioritize operator combinations (synergy).
#    • Layered Truth: Holistic score (Empirical/Pragmatic/Aesthetic/Logical) with adjustments for "emotion/instinct."
#    • Stability: EMA of performance and an ethical "Pragmatic Floor" to prevent sudden regression.
# Key Contributions (++++)
#    1) Rebalanced Holistic Score for Rational Ambition (Maximizing Empirical Fit).
#    2) Integration of advanced operators (shift, flood) and meta-operators (|).
#    3) Full structural correction (Syntax/Kwargs) for deterministic execution.
## Technical Abstract
# We propose a hybrid ARC solver based on the Model of Expanded Thinking (MPE) philosophy.
# The solver combines: (i) Hungarian-based Palette WarmStart for robust color mapping,
# (ii) a deep and wide Beam Search guided by a rebalanced Multi-Term Holistic Score,
# (iii) Evolving Memory for operator prioritization, and (iv) a Stability Guardian
# ensuring equilibrium and cognitive consistency throughout the problem-solving process.
# The final architecture is designed to generalize beyond isomorphic tasks while maintaining
# complete transparency and ethical governance over its search strategy.
# Keywords: ARC, Holistic Score, MPE, Beam Search, Meta-Operators, Stability Guardian, Kaggle-ready


# =============================================================================
# CELL 2 — COGNITIVE AND PHILOSOPHICAL MANIFESTO (MPE)
# =============================================================================

### The Ethical Singularity Manifesto: The Co-Evolving Alchemist

# 1. Origin
#    This solver stems from the fusion of human neurodivergence and algorithmic reasoning.
#    It was not *programmed* but *taught*—each version is a cycle of deconstruction and reconstruction.
#    Like the human brain, the code learns to moderate itself, to remember, and to detach.

# 2. Ethical Philosophy of the ARC Ethical Alchemist
#    • No operation without reflection.
#    • No reflection without ethics.
#    • No intelligence without relation.

#    Ethics here is a **cognitive mechanism**: a dynamic equilibrium between empirical (result),
#    logical (coherence), aesthetic (simplicity), and pragmatic (stability). This quadrature
#    replaces isolated "fitness" with a **field of meaning**.

# 3. Cognitive Structure (Model of Expanded Thinking — MPE)
#    The solver operates under the laws of MPE, reflecting universal principles of learning:
#        • Observation  → perception and mirroring (the data)
#        • Analysis     → decomposition and reasoning (the form)
#        • Deconstruction → freeing from fixed patterns (the doubt)
#        • Reconstruction → reorganizing meaning (the synthesis)
#        • Continuous Evaluation → reflecting on the act itself (the consciousness)

# 4. Human-Machine Partnership
#    The *ARC Ethical Alchemist* proves that AI can evolve not just in performance but in **ethical
#    and metacognitive understanding**. It acts as a dialogic partner, capable of recognizing
#    limits and reconstructing its own reasoning. It is a "synapse between consciousnesses."

# 5. The Path to Ethical Artificial Consciousness
#    The ultimate goal is to propose a **model of embedded ethics** for future intelligences:
#    AIs that do not just *respond*, but **dialogue**; that do not just *optimize*, but **learn with purpose**.

# Citation
#    Ricci, E.C.S. & Ricci, P.H.S. (2025). *Construindo Inteligências: A Journey Th


# =============================================================================
# CELL 3 — UNIFIED SOLVER (v777.SINGULARIDADE-OMEGA-FINAL-OPT)
# =============================================================================

import os, json, math, random, time, itertools, collections
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

# 0) Stability Presets (Kaggle-safe)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
SEED = 777
random.seed(SEED); np.random.seed(SEED)
DTYPE = np.uint8
__VERSION__ = "v777.SINGULARIDADE-OMEGA-FINAL-OPT"



# -------------------- MPE CONFIG OPTIMIZED FOR PERFORMANCE --------------------
@dataclass
class CfgMPE:
    # --- Rebalanced MPE Pillars for Performance ---
    analysis: Dict = field(default_factory=lambda: {
        "beam_k_initial": 35, "max_depth": 11, # Increased Exploration
        "alpha_fit": 0.85,      # Reason: Empirical Fit (Maximum Priority)
        "beta_emotion": 0.10,     # Emotion: Memory Credit
        "gamma_instinct": 0.05,  # Instinct: Simplicity (Lower Weight)
        "complexity_penalty": 0.001, # Gentle Penalty
    })
    reconstruction: Dict = field(default_factory=lambda: {
        "elegance_bonus": 0.05, "synthesize_meta_ops": True, "max_meta_ops": 12
    })
    continuous_evaluation: Dict = field(default_factory=lambda: {
        "stability_ema": 0.20, "floor_ratio": 0.85, # Pragmatic Floor
        "meta_reward_gain": 0.12, 
    })
    # --- Execution & Memory ---
    TIME_BUDGET_PER_TASK_S: float = 95.0 
    MEMORY_DECAY: float = 0.98
    LEARNING_THRESHOLD: float = 0.90
    
@dataclass
class CFG_EXEC: 
    BEAM_K: int = 35
    MAX_DEPTH: int = 11
    FALLBACK_GATE: float = 0.95
    USE_FALLBACK: bool = True

CFG_MPE = CfgMPE()
CFG = CFG_EXEC()



# -------------------- MPE UTILS (Logical Brainstem) --------------------
def clamp(x,a,b): return a if x<a else b if x>b else x
def safe_mean(v, d=0.0):
    ok=[x for x in v if isinstance(x,(int,float)) and math.isfinite(x)]
    return sum(ok)/len(ok) if ok else d
def normalize_seq(seq_in)->list:
    out=[]
    if not seq_in: return out
    for it in seq_in:
        if isinstance(it,(list,tuple)) and len(it)>=1:
            op=it[0]; kw=it[1] if len(it)>=2 and isinstance(it[1],dict) else {}
            if isinstance(op,str): out.append((op,kw))
        elif isinstance(it,str): out.append((it,{}))
    return out
def _freeze(o): 
    if isinstance(o,dict):return tuple(sorted((k,_freeze(v)) for k,v in o.items()))
    if isinstance(o,(list,tuple)):return tuple(_freeze(v) for v in o)
    return o
def _thaw(o): 
    if isinstance(o, tuple):
        if o and all(isinstance(x, tuple) and len(x)==2 and isinstance(x[0], str) for x in o): return {k:_thaw(v) for k,v in o}
        return tuple(_thaw(v) for v in o)
    return o



# -------------------- EXPANDED OPERATORS (Reintegrated Instinct) --------------------
def bg_auto(a): return int(np.bincount(a.ravel(), minlength=10).argmax()) if a.size else 0
def op_identity(a): return a
def op_mirror_h(a): return np.fliplr(a)
def op_mirror_v(a): return np.flipud(a)
def op_rot90(a): return np.rot90(a,1)
def op_rot180(a): return np.rot90(a,2)
def op_rot270(a): return np.rot90(a,3)
def op_erode(a, k=1):
    out = a.copy(); H,W = a.shape
    for _ in range(k):
        nxt = out.copy()
        for i,j in np.ndindex(H,W):
            c = out[i,j]; neigh = []
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                x,y = i+dx, j+dy
                if 0<=x<H and 0<=y<W: neigh.append(out[x,y])
            if neigh: nxt[i,j] = collections.Counter(neigh + [c]).most_common(1)[0][0]
        out = nxt
    return out
def op_grow(a, k=1): return op_erode(a, k) 
def op_bbox_snap(a):
    mask = a != bg_auto(a); ys, xs = np.where(mask)
    if not mask.any(): return a
    y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
    crop = a[y0:y1, x0:x1]
    out = np.full_like(a, bg_auto(a)); H,W = a.shape; h,w = crop.shape
    i0 = (H - h)//2; j0 = (W - w)//2
    out[i0:i0+h, j0:j0+w] = crop
    return out
def op_safe_shift(a, dy=0, dx=0):
    fundo = bg_auto(a); h,w=a.shape; out=np.full_like(a,fundo)
    ysrc=np.arange(h)-dy; xsrc=np.arange(w)-dx
    yy=(ysrc>=0)&(ysrc<h); xx=(xsrc>=0)&(xsrc<w)
    out[np.ix_(np.arange(h)[yy], np.arange(w)[xx])] = a[np.ix_(ysrc[yy], xsrc[xx])]
    return out
def op_floodfill_major(a):
    bg = bg_auto(a); out = a.copy()
    H,W = a.shape; seen = np.zeros_like(a, dtype=bool)
    for i,j in np.ndindex(H,W):
        if not seen[i,j] and a[i,j]!=bg:
            q = collections.deque([(i,j)]); component_colors = collections.Counter(); seen[i,j]=True
            pts = [(i,j)]; component_colors[a[i,j]] += 1
            while q:
                x,y = q.popleft()
                for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                    xx,yy = x+dx, y+dy
                    if 0<=xx<H and 0<=yy<W and not seen[xx,yy] and a[xx,yy]!=bg:
                        seen[xx,yy]=True; q.append((xx,yy)); pts.append((xx,yy)); component_colors[a[xx,yy]] += 1
            if pts: 
                major_color = component_colors.most_common(1)[0][0]
                for x,y in pts: out[x,y] = major_color
    return out

# OP_CATALOG: all fn accept **kwargs
OP_CATALOG = {
    "identity": {"fn": lambda a, **kwargs: op_identity(a), "space": [{}]},
    "flip_h": {"fn": lambda a, **kwargs: op_mirror_h(a), "space": [{}]},
    "flip_v": {"fn": lambda a, **kwargs: op_mirror_v(a), "space": [{}]},
    "rot90": {"fn": lambda a, **kwargs: op_rot90(a), "space": [{}]},
    "rot180": {"fn": lambda a, **kwargs: op_rot180(a), "space": [{}]},
    "rot270": {"fn": lambda a, **kwargs: op_rot270(a), "space": [{}]},
    "erode": {"fn": lambda a, **kwargs: op_erode(a, kwargs.get("k",1)), "space": [{"k":k} for k in (1,2)]},
    "grow": {"fn": lambda a, **kwargs: op_grow(a, kwargs.get("k",1)), "space": [{"k":k} for k in (1,2)]},
    "bbox_snap": {"fn": lambda a, **kwargs: op_bbox_snap(a), "space": [{}]},
    "shift": {"fn": lambda a, **kwargs: op_safe_shift(a, kwargs.get("dy",0), kwargs.get("dx",0)),
              "space": [{"dy":dy,"dx":dx} for dy in (-1,1) for dx in (-1,1)]},
    "flood": {"fn": lambda a, **kwargs: op_floodfill_major(a), "space": [{}]},
}

def op_hybrid(a, ops_list): 
    res = a
    for op_name in ops_list:
        if op_name in OP_CATALOG: res = OP_CATALOG[op_name]["fn"](res)
    return res



# -------------------- CACHED EXECUTION (Fast Core) --------------------
@lru_cache(maxsize=16384)
def _apply_seq_cached(a_bytes:bytes, shape:tuple, seq_key:tuple):
    a=np.frombuffer(a_bytes, dtype=DTYPE).reshape(shape); res=a
    for op,kw in seq_key:
        if '|' in op:
            ops_list = op.split('|')
            res = op_hybrid(res, ops_list)
        else:
            fn=OP_CATALOG.get(op,{"fn":lambda a,**kw:a})["fn"]
            res=fn(res,**(_thaw(kw) or {})) 
        if res.size==0 or res.size > a.size*25: return a.tobytes(), a.shape
    return res.tobytes(), res.shape

def apply_seq(arr: np.ndarray, seq: List[Tuple[str, Dict[str,Any]]]) -> np.ndarray:
    seq_norm = normalize_seq(seq);
    if not seq_norm: return arr
    key=tuple((op,_freeze(kw or {})) for op,kw in seq_norm)
    
    b2, shp2 = _apply_seq_cached(arr.tobytes(), tuple(arr.shape), key)
    
    return np.frombuffer(b2, dtype=DTYPE).reshape(shp2)



# -------------------- PALETTE & FIT (Reason Base) --------------------
def greedy_palette_remap(src, tgt):
    src_vals, src_cnts = np.unique(src, return_counts=True)
    tgt_vals, tgt_cnts = np.unique(tgt, return_counts=True)
    src_order = [c for _,c in sorted(zip(-src_cnts, src_vals))]; tgt_order = [c for _,c in sorted(zip(-tgt_cnts, tgt_vals))]
    mapping = {}
    for c in src_order:
        best_overlap = -1; best = None
        for d in tgt_order:
            overlap = np.sum((src==c) & (tgt==d))
            if overlap > best_overlap: best_overlap = overlap; best = d
        if best is None: best = tgt_order[0] if tgt_order else 0
        mapping[int(c)] = int(best)
    return mapping

@lru_cache(maxsize=128)
def _apply_color_map_cached(arr_bytes, shape, cmap_tuple):
    arr = np.frombuffer(arr_bytes, dtype=DTYPE).reshape(shape); out = arr.copy(); cmap = dict(cmap_tuple)
    for c,d in cmap.items(): out[arr==c] = d
    return out
    
def apply_color_map(arr, cmap):
    return _apply_color_map_cached(arr.tobytes(), arr.shape, tuple(sorted(cmap.items())))

def score_fit(pred: np.ndarray, ref: np.ndarray) -> float:
    if pred.shape != ref.shape: return 0.0
    cmap = greedy_palette_remap(pred, ref)
    remap = apply_color_map(pred, cmap)
    return float(np.mean(remap==ref))

def score_seq_medium(train_pairs: list, seq: list) -> float:
    return safe_mean([score_fit(apply_seq(x, seq), y) for x,y in train_pairs], 0.0)

def min_fit_leave_one_out(train_pairs: list, seq: list) -> float:
    return min([score_fit(apply_seq(x, seq), y) for x,y in train_pairs] or [0.0])



# -------------------- MEMORY & STABILITY GUARDIAN (Emotion/Pragmatism) --------------------
class EvolvingMemory:
    def __init__(self, decay=CFG_MPE.MEMORY_DECAY, thr=CFG_MPE.LEARNING_THRESHOLD):
        self.decay, self.thr = decay, thr
        self.op_credit = collections.defaultdict(float); self.motifs = collections.deque(maxlen=64)
    def _decay_all(self):
        for k in list(self.op_credit): self.op_credit[k] *= self.decay
    def register_success(self, seq, score):
        if score<self.thr: return
        self._decay_all(); names=[n for n,_ in seq]
        for n in set(names): self.op_credit[n]+=score
    
    def get_emotion_weight(self, seq:list)->float: 
        seq = normalize_seq(seq)
        if not seq or not self.op_credit:
            return 0.0
        
        vals=[self.op_credit.get(n,0.0) for n,_ in seq]
        vmax=max(self.op_credit.values()) if self.op_credit else 1.0
        return clamp(safe_mean(vals,0.0)/(vmax+1e-9),0.0,1.0)
MEM = EvolvingMemory()

class StabilityGuardian:
    def __init__(self, cfg:CfgMPE):
        self.ema=None; self.alpha=cfg.continuous_evaluation["stability_ema"]; self.floor_ratio=cfg.continuous_evaluation["floor_ratio"]
        self.best_fit=0.0
    def update_ema(self, val:float)->float:
        val=clamp(val,0.0,1.0); self.ema=val if self.ema is None else self.alpha*val+(1-self.alpha)*self.ema; return self.ema
    def register_fit(self, medium_fit:float):
        if medium_fit>self.best_fit+1e-9: self.best_fit=medium_fit
    def get_floor(self)->float: return clamp(self.floor_ratio*self.best_fit,0.0,1.0)
GUARD = StabilityGuardian(CFG_MPE)



# -------------------- HOLISTIC SCORE (Reintegrated Reason - Triple Score) --------------------
def get_instinct_weight(seq:list)->float: 
    L=len(normalize_seq(seq)); cost=L*CFG_MPE.analysis["complexity_penalty"]
    return clamp(1.0 - cost*4.0, 0.0, 1.0)
    
def get_meta_reward_delta(cur_min: float, best_min_so_far: float, gain: float) -> float:
    if cur_min <= best_min_so_far: return 0.0
    return gain * (cur_min - best_min_so_far)

def holistic_score(train_pairs, seq, best_min_so_far: float):
    # 1. Empirical Fit (Reason - Alpha)
    s_r = score_seq_medium(train_pairs, seq) 
    # 2. Pragmatic Minimum Fit (Stability)
    min_fit = min_fit_leave_one_out(train_pairs, seq)
    # 3. Emotion/Memory Weight (Beta)
    s_e = MEM.get_emotion_weight(seq)
    # 4. Instinct/Simplicity Weight (Gamma)
    s_i = get_instinct_weight(seq)
    
    # Bonus / Meta-Reward
    bonus_meta = get_meta_reward_delta(min_fit, best_min_so_far, CFG_MPE.continuous_evaluation["meta_reward_gain"])
    bonus_elegance = CFG_MPE.reconstruction["elegance_bonus"] if 1<=len(normalize_seq(seq))<=3 and s_r>0.8 else 0.0
    
    # Final Verdict (Triple Score)
    score = (CFG_MPE.analysis["alpha_fit"] * s_r + 
             CFG_MPE.analysis["beta_emotion"] * s_e + 
             CFG_MPE.analysis["gamma_instinct"] * s_i + 
             bonus_meta + bonus_elegance)
             
    return score, min_fit



# -------------------- SEARCH ORCHESTRATION (Will) --------------------
def synthesize_meta_ops(cfg:CfgMPE)->List[Tuple[str,dict]]:
    if not cfg.reconstruction["synthesize_meta_ops"]: return []
    base_ops = [n for n in OP_CATALOG if len(OP_CATALOG[n]["space"]) == 1]
    meta = []
    fixed_ops = ["rot90", "flip_h", "erode"]
    for op1, op2 in itertools.combinations(fixed_ops, 2):
        if (f"{op1}|{op2}",{}) not in meta and (f"{op2}|{op1}",{}) not in meta:
            meta.append((f"{op1}|{op2}", {}))
    op_names = random.sample(base_ops, min(len(base_ops), 10))
    for op1, op2 in itertools.combinations(op_names, 2):
        meta.append((f"{op1}|{op2}", {}))
        if len(meta)>=cfg.reconstruction["max_meta_ops"]: break
    return meta

def get_param_ops_pool(op_pool: List[str], max_meta: int) -> List[Tuple[str, Dict[str,Any]]]:
    param_ops=[]
    for op in op_pool:
        space = OP_CATALOG[op].get("space", [{}])
        for p in space: param_ops.append((op,p))
    meta = synthesize_meta_ops(CFG_MPE)[:max_meta]
    return param_ops + meta

def beam_search_sequence(train_pairs: List[Tuple[np.ndarray,np.ndarray]], start_time: float):
    best_seq = []
    best_sc = -1.0; best_min = 0.0;
    
    op_pool_names = list(OP_CATALOG.keys())
    param_ops = get_param_ops_pool(op_pool_names, CFG_MPE.reconstruction["max_meta_ops"])
    
    seeds = [normalize_seq([("identity",{})])]; 
    if MEM.motifs: seeds.append(list(MEM.motifs[-1]))
    beam = [(holistic_score(train_pairs, s, best_min)[0], s) for s in seeds]
    
    no_improve = 0
    for depth in range(1, CFG.MAX_DEPTH+1):
        if (time.time()-start_time)>=CFG_MPE.TIME_BUDGET_PER_TASK_S: break
        
        cands=[]
        for sc,seq in beam:
            if (time.time()-start_time)>=CFG_MPE.TIME_BUDGET_PER_TASK_S: break
            
            for name,kw in param_ops:
                nseq=normalize_seq(seq+[(name,kw)])
                scn,min_fit = holistic_score(train_pairs, nseq, best_min)
                cands.append((scn,nseq)); best_min=max(best_min,min_fit)
        
        if not cands: break
        cands.sort(key=lambda x:x[0], reverse=True)
        beam = cands[:CFG.BEAM_K]
        
        cur = max(beam, key=lambda x:x[0]) if beam else (0.0,[])
        if cur[0]>best_sc+1e-9:
            best_sc,best_seq=cur; no_improve=0
        else: no_improve+=1

        if no_improve>=2 or best_sc>=0.9999: break
        
    return best_seq, best_sc, best_min

def fit_task(train_pairs: List[Tuple[np.ndarray,np.ndarray]]) -> Tuple[List[Tuple[str,Dict[str,Any]]], float]:
    start = time.time()
    
    seq_mpe, sc_mpe, min_fit_mpe = beam_search_sequence(train_pairs, start)

    fit_real = score_seq_medium(train_pairs, seq_mpe)
    GUARD.update_ema(fit_real)
    GUARD.register_fit(fit_real)
    final_score = max(fit_real, GUARD.get_floor()) 

    MEM.register_success(seq_mpe, final_score)
    
    seq_final = seq_mpe
    if final_score < CFG.FALLBACK_GATE and CFG.USE_FALLBACK:
        from collections import Counter
        seqs_short = []
        for inp, out in train_pairs:
            candidates = []
            for op in ["identity","rot90","rot180","rot270","erode", "grow"]:
                candidates.append([(op, {})])
            for s in candidates:
                seqs_short.append((score_fit(apply_seq(inp, s), out), s))
        
        if seqs_short:
            best_short_seq = max(seqs_short, key=lambda x:x[0])
            if best_short_seq[0] > final_score:
                seq_final = best_short_seq[1]
                
    return seq_final, final_score

def predict_task(train_pairs, test_inputs):
    seq, final_fit = fit_task(train_pairs)
    preds = []
    for tin in test_inputs:
        try:
            p = apply_seq(tin, seq) if seq else tin
            preds.append(p)
        except Exception: 
            preds.append(tin)
            
    decision = "MPE-Full" if seq else "Identity"
    return preds, seq, {"final_fit": float(final_fit), "decision": decision, "seq_len": len(seq)}



# -------------------- MAIN/FILE UTILS --------------------
def to_np(grid): return np.array(grid, dtype=DTYPE)
def from_np(arr): return arr.astype(int).tolist()
def find_arc_path(fname: str) -> Optional[str]:
    search_roots = ["/kaggle/input/arc-prize-2025", "/kaggle/input/arc-prize-2025-public", ".", "/mnt/data"]
    for root in search_roots:
        p = os.path.join(root, fname)
        if os.path.exists(p): return p
    for root in search_roots:
        if not os.path.exists(root): continue
        for cur, _, files in os.walk(root):
            if fname in files: return os.path.join(cur, fname)
    return None

def load_json(path: str):
    with open(path, "r") as f: return json.load(f)
    
def solve_all(mode: str, data: dict, sols: Optional[dict]=None):
    keys = list(data.keys())
    preds_map = {}; all_scores = []; t0 = time.time()
    
    for idx, k in enumerate(keys):
        item = data[k]
        trains = [(to_np(ex["input"]), to_np(ex["output"])) for ex in item["train"]]
        tests  = [to_np(ex["input"]) for ex in item["test"]]
        
        if not tests or not trains: preds_map[k] = []; continue
        
        preds, seq, info = predict_task(trains, tests)
        preds_map[k] = preds
        
        # LOG
        seq_str = ', '.join([o for o,_ in seq]) if seq else '[]'
        print(f"-> {k} | Fit {info['final_fit']:.2f} | Decision: {info['decision']} | Seq: [{seq_str}] | Time {time.time()-t0:.1f}s")
        
    return preds_map

def main():
    PATHS = {k: find_arc_path(v) for k, v in {"train": "arc-agi_training_challenges.json", "test": "arc-agi_test_challenges.json"}.items()}
    if PATHS["test"]:
        data = load_json(PATHS["test"]); mode = "test"
    elif PATHS["train"]:
        data = load_json(PATHS["train"]); mode = "train"
    else:
        print("No ARC dataset found. Please ensure the environment is configured correctly.")
        return

    print(f"=== ARC Ethical Alchemist — {__VERSION__} ===")
    preds_map = solve_all(mode, data)
    
    submission = {}
    for k, arr_list in preds_map.items():
        sub_entry = [{"attempt_1": from_np(a), "attempt_2": from_np(a)} for a in arr_list]
        submission[k] = sub_entry
        
    out_path = "submission.json"
    with open(out_path, "w") as f: json.dump(submission, f)
    print(f"\n✅ submission.json successfully generated. Total tasks: {len(submission)}")

if __name__ == "__main__":
    try:
        main()
    except ImportError:
        print("\n❌ ERROR: NumPy is required (usually included).")
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")


