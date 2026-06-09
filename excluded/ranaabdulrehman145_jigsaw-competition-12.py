# ---------------------------
# Robust JIGSAW AGILE runner
# - tries to load local ST models from provided paths (searches nested folders)
# - falls back to TF-IDF if models fail to load
# - builds centroids, scores, calibrates, writes submission.csv
# ---------------------------
import os, gc, re, random
import numpy as np, pandas as pd
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.special import expit

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Try import sentence-transformers; if not present, we'll fallback to TF-IDF
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except Exception:
    ST_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------- CONFIG ----------------
CFG = {
    'input_csv': '/kaggle/input/jigsaw-agile-community-rules/test.csv',
    'output_submission': 'submission.csv',
    # Put the top-level paths you uploaded to Kaggle; they may be nested inside these folders
    'bge_model_hint': '/kaggle/input/baai/transformers/bge-large-en-v1.5/1',
    'gte_model_hint': '/kaggle/input/gte-large-en-v1.5-ft-new/pytorch/default/1',
    'blend_weight': 0.55,   # weight for first model if dims match; otherwise concat
    'tfidf_max_features': 4096,
    'embed_batch': 64
}

# ---------------- helper cleaners ----------------
def clean_text(t):
    if not isinstance(t, str):
        return ""
    t = t.strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"http\S+", "<URL>", t)
    return t.lower()

# ---------------- load df ----------------
if not os.path.exists(CFG['input_csv']):
    raise FileNotFoundError(f"Test CSV not found at {CFG['input_csv']}. Upload dataset or change CFG['input_csv'].")
df = pd.read_csv(CFG['input_csv'])
print(f"Loaded {len(df)} rows; columns: {list(df.columns)}")

# ---------------- find candidate model roots ----------------
def collect_candidate_roots(hint_path):
    """
    Given a hint (top-level path the user uploaded), search it for subfolders that
    might be valid model roots (contain config + weights). Return a list of candidates.
    """
    candidates = []
    if hint_path and os.path.exists(hint_path):
        # If the hint itself looks like a candidate
        candidates.append(hint_path)
        # search depth-first up to 4 levels deep
        for root, dirs, files in os.walk(hint_path):
            # Add any folder that has config or model weights
            if any(f in files for f in ('config.json','pytorch_model.bin','model.safetensors','tf_model.h5','adapter_config.json')):
                candidates.append(root)
            # also add immediate subdirs (some Kaggle zips keep models nested)
            for d in dirs:
                candidates.append(os.path.join(root, d))
    return list(dict.fromkeys(candidates))  # unique, preserve order

def try_load_sentence_transformer(path):
    """
    Try to construct a SentenceTransformer from path. Return model or None.
    Uses a few fallbacks and prints helpful messages.
    """
    if not ST_AVAILABLE:
        print("sentence-transformers package not available in this runtime -> cannot load ST models.")
        return None
    tried = []
    candidates = []
    if path:
        candidates = collect_candidate_roots(path)
    # also try path as-is if it wasn't captured
    if path and path not in candidates and os.path.exists(path):
        candidates.append(path)
    # remove duplicates and non-existent
    candidates = [c for c in candidates if os.path.exists(c)]
    if not candidates:
        return None
    for c in candidates:
        if c in tried: 
            continue
        tried.append(c)
        try:
            print(f"Attempting to load SentenceTransformer from: {c}")
            # trust_remote_code may be required for some custom HF repos
            m = SentenceTransformer(c, device='cuda' if (('CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES']!='') or (os.environ.get('NVIDIA_VISIBLE_DEVICES')!='')) else 'cpu', trust_remote_code=True)
            print("Loaded model from:", c)
            return m
        except Exception as e:
            # not fatal; try next candidate
            print(f"Failed to load from {c}: {str(e)[:260]}")
            continue
    return None

# ---------------- try to load models ----------------
bge = try_load_sentence_transformer(CFG['bge_model_hint'])
gte = try_load_sentence_transformer(CFG['gte_model_hint'])

USE_TFIDF_FALLBACK = (bge is None and gte is None)
if USE_TFIDF_FALLBACK:
    print("No sentence-transformer models loaded. Proceeding with TF-IDF fallback (offline-safe).")
else:
    loaded_names = [n for n in ('BGE' if bge else None, 'GTE' if gte else None) if n]
    print("Loaded ST models:", loaded_names)

# ---------------- collect unique texts ----------------
uniq_texts = set()
for _, r in df.iterrows():
    uniq_texts.add(clean_text(r['rule']))
    uniq_texts.add(clean_text(r['body']))
    for c in ['positive_example_1','positive_example_2','negative_example_1','negative_example_2']:
        v = r.get(c)
        if pd.notna(v):
            uniq_texts.add(clean_text(v))
uniq_texts = list(uniq_texts)
print("Unique texts to embed:", len(uniq_texts))

# ---------------- compute embeddings ----------------
text2emb = {}  # will hold final embeddings per text (np.array)
emb_sources = {}  # keep raw embeddings per model if needed

if USE_TFIDF_FALLBACK:
    vectorizer = TfidfVectorizer(max_features=CFG['tfidf_max_features'], ngram_range=(1,2), stop_words='english')
    X = vectorizer.fit_transform(uniq_texts)
    for i, t in enumerate(uniq_texts):
        arr = X[i].toarray().reshape(-1)
        nrm = np.linalg.norm(arr)
        if nrm > 0:
            arr = arr / nrm
        text2emb[t] = arr.astype(np.float32)
    print("TF-IDF embeddings created (dim=%d)." % (len(next(iter(text2emb.values()))),))
else:
    # encode with each available model and combine
    model_embs = {}
    if bge is not None:
        print("Encoding with BGE...")
        embs = bge.encode(uniq_texts, batch_size=CFG['embed_batch'], show_progress_bar=True, normalize_embeddings=True)
        model_embs['bge'] = np.array(embs)
    if gte is not None:
        print("Encoding with GTE...")
        embs = gte.encode(uniq_texts, batch_size=CFG['embed_batch'], show_progress_bar=True, normalize_embeddings=True)
        model_embs['gte'] = np.array(embs)
    # unify into text2emb: prefer weighted average if dims match, else concat
    dims = [v.shape[1] for v in model_embs.values()]
    if len(set(dims)) == 1:
        # dims match -> weighted average
        keys = list(model_embs.keys())
        if len(keys) == 2:
            w = CFG['blend_weight']
            arr = w * model_embs[keys[0]] + (1 - w) * model_embs[keys[1]]
        else:
            # single model
            arr = list(model_embs.values())[0]
        # normalize just in case
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        arr = arr / norms
        for t, e in zip(uniq_texts, arr):
            text2emb[t] = e.astype(np.float32)
    else:
        # dims differ -> concatenate normalized vectors (safe)
        parts = []
        for k in model_embs:
            m = model_embs[k]
            m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)
            parts.append(m)
        cat = np.concatenate(parts, axis=1)
        # final normalize
        cat = cat / (np.linalg.norm(cat, axis=1, keepdims=True) + 1e-12)
        for t, e in zip(uniq_texts, cat):
            text2emb[t] = e.astype(np.float32)
    print("Combined embeddings ready (dim=%d)." % (len(next(iter(text2emb.values()))),))
    del model_embs
    gc.collect()

# ---------------- build rule centroids (weighted) ----------------
def weighted_centroid(arrs, rule_emb):
    """arrs: np.array (n_examples x dim); rule_emb: vector or None"""
    if arrs is None or len(arrs) == 0:
        return None
    A = np.vstack(arrs)
    if rule_emb is None:
        cent = A.mean(axis=0)
        cent = cent / (np.linalg.norm(cent) + 1e-12)
        return cent
    # weight by similarity to rule
    sims = (A @ rule_emb)
    # rescale sims to 0..1
    smin, smax = sims.min(), sims.max()
    if smax - smin < 1e-6:
        weights = np.ones_like(sims)
    else:
        weights = (sims - smin) / (smax - smin)
    wsum = (A * weights[:,None]).sum(axis=0)
    if np.linalg.norm(wsum) < 1e-12:
        cent = A.mean(axis=0)
    else:
        cent = wsum / (np.linalg.norm(wsum) + 1e-12)
    return cent

rule_centroids = {}
for rule, g in df.groupby('rule'):
    ru_clean = clean_text(rule)
    rule_emb = text2emb.get(ru_clean)
    pos_list, neg_list = [], []
    for _, r in g.iterrows():
        for c in ['negative_example_1','negative_example_2']:
            v = r.get(c)
            if pd.notna(v):
                e = text2emb.get(clean_text(v))
                if e is not None: pos_list.append(e)
        for c in ['positive_example_1','positive_example_2']:
            v = r.get(c)
            if pd.notna(v):
                e = text2emb.get(clean_text(v))
                if e is not None: neg_list.append(e)
    if len(pos_list) > 0 and len(neg_list) > 0:
        pos_cent = weighted_centroid(pos_list, rule_emb)
        neg_cent = weighted_centroid(neg_list, rule_emb)
        rule_centroids[rule] = {'pos': pos_cent, 'neg': neg_cent, 'rule_emb': rule_emb}
print("Constructed centroids for %d rules." % len(rule_centroids))

# ---------------- scoring ----------------
row_ids = []
scores = []
for _, r in tqdm(df.iterrows(), total=len(df)):
    rule = r['rule']
    rid = r['row_id']
    if rule not in rule_centroids:
        row_ids.append(rid); scores.append(0.0); continue
    c = rule_centroids[rule]
    posc = c['pos']; negc = c['neg']; rule_emb = c['rule_emb']
    body_clean = clean_text(r.get('body',''))
    b_emb = text2emb.get(body_clean)
    if b_emb is None:
        # on-the-fly compute for TF-IDF fallback (or missing)
        if USE_TFIDF_FALLBACK:
            v = vectorizer.transform([body_clean]).toarray().reshape(-1)
            n = np.linalg.norm(v)
            if n>0: v = v / n
            b_emb = v.astype(np.float32)
        else:
            # try encode with whichever ST model loaded
            try:
                if bge is not None:
                    b_emb = bge.encode([body_clean], normalize_embeddings=True)[0]
                elif gte is not None:
                    b_emb = gte.encode([body_clean], normalize_embeddings=True)[0]
                else:
                    b_emb = np.zeros_like(posc)
            except Exception:
                b_emb = np.zeros_like(posc)
    # compute similarities
    pos_sim = float(np.dot(b_emb, posc)) if posc is not None else 0.0
    neg_sim = float(np.dot(b_emb, negc)) if negc is not None else 0.0
    rule_sim = float(np.dot(b_emb, rule_emb)) if rule_emb is not None else 0.0
    # fused score: main signal is neg - pos; blend with rule similarity
    score = (neg_sim - pos_sim) + 0.25 * rule_sim
    row_ids.append(rid); scores.append(score)

# ---------------- calibration (sigmoid) ----------------
arr = np.array(scores)
if len(arr) > 0:
    # robust scaling then sigmoid
    mean, std = arr.mean(), arr.std() if arr.std() > 0 else 1.0
    calibrated = expit((arr - mean) / (std + 1e-12))
else:
    calibrated = arr

sub = pd.DataFrame({'row_id': df['row_id'].values})
sub_map = dict(zip(row_ids, calibrated))
sub['rule_violation'] = sub['row_id'].map(sub_map).fillna(0.0).astype(float)
sub.to_csv(CFG['output_submission'], index=False)
print("Saved", CFG['output_submission'])
print("rule_violation distribution:")
print(sub['rule_violation'].describe())

# ---------------- final notes ----------------
if USE_TFIDF_FALLBACK:
    print("WARNING: Running TF-IDF fallback (no ST models loaded). For best LB, upload BGE/GTE model folders and re-run.")
else:
    print("Success: embeddings used. If you still see errors when loading your model paths, check that the folder you uploaded contains a HuggingFace-style model (config.json + weights).")


