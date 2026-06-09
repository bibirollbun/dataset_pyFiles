# ==== Step 0: setup & utils ====
import os, gc, sys, time, math, random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GroupShuffleSplit

DATA_DIR = "/kaggle/input/shopee-product-matching"
WORK_DIR = "/kaggle/working"
os.makedirs(WORK_DIR, exist_ok=True)

def timer(msg=""):
    t0 = time.time()
    print(f"[start] {msg}")
    def done():
        dt = time.time() - t0
        print(f"[done ] {msg} in {dt:.1f}s")
    return done

def clean_text(s: str) -> str:
    s = str(s).lower().replace("\n"," ").replace("\t"," ")
    return " ".join(s.split())

def phash_hex_to_int64(h: str) -> np.uint64:
    try: return np.uint64(int(h,16))
    except: return np.uint64(0)

def hamming_u64(a: np.uint64, b: np.uint64) -> int:
    return int((a ^ b).bit_count())

def phash_bucket(h: str, n_prefix=6):
    return h[:n_prefix] if isinstance(h,str) else ""

def token_set(title: str):
    return set(title.split())

def extract_numbers(s: str):
    return set([tok for tok in s.split() if any(ch.isdigit() for ch in tok)])

def numeric_overlap(a_set, b_set):
    if not a_set or not b_set: return 0.0
    inter = len(a_set & b_set); union = len(a_set | b_set)
    return inter/union if union else 0.0

def prefix_suffix_match(a: str, b: str):
    a_tok, b_tok = a.split(), b.split()
    if not a_tok or not b_tok: return 0.0
    first = 1.0 if a_tok[0]==b_tok[0] else 0.0
    last  = 1.0 if a_tok[-1]==b_tok[-1] else 0.0
    return max(first, last)

print("Setup OK")



# ==== Step 1: load & clean ====
done = timer("load csv & clean")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

for df in (train, test):
    for col in ["posting_id","image","image_phash","title"]:
        assert col in df.columns, f"Missing column {col}"

train["title_c"] = train["title"].map(clean_text)
test["title_c"]  = test["title"].map(clean_text)

for df in (train, test):
    df["ph_int"]     = df["image_phash"].astype(str).map(phash_hex_to_int64)
    df["ph_bucket"]  = df["image_phash"].astype(str).map(lambda s: phash_bucket(s, 6))

train.to_pickle(os.path.join(WORK_DIR, "train_clean.pkl"))
test.to_pickle(os.path.join(WORK_DIR,  "test_clean.pkl"))
done()
print("Step 1 saved: train_clean.pkl, test_clean.pkl")



# ==== Step 2: BERT embeddings (CLS) ====
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# авто-поиск папки с моделью в /kaggle/input
MODEL_DIR = None
CANDIDATE_KEYS = ("minilm", "mpnet", "bert", "sentence", "all-minilm")
for d in sorted(os.listdir("/kaggle/input")):
    low = d.lower()
    if any(k in low for k in CANDIDATE_KEYS):
        p = os.path.join("/kaggle/input", d)
        files = set(os.listdir(p))
        if ("config.json" in files) and (("pytorch_model.bin" in files) or ("model.safetensors" in files)):
            MODEL_DIR = p
            break
assert MODEL_DIR is not None, "❌ Нет модели в /kaggle/input. Добавь датасет с весами (HF) через Add data."
print("✅ Using model from:", MODEL_DIR)

# твой класс, слегка обобщённый под батчи
class Bert(nn.Module):
    def __init__(self, model_dir, trainable=False):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.bert = AutoModel.from_pretrained(model_dir).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        if not trainable:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.CLS = 0  # индекс CLS

    def forward(self, input_texts):
        enc = self.tokenizer(
            input_texts, padding=True, truncation=True, max_length=64, return_tensors='pt'
        ).to(self.device)
        with torch.no_grad():
            out = self.bert(**enc)
        cls = out.last_hidden_state[:, self.CLS, :]  # [B, hidden]
        # нормализуем для косинуса
        cls = torch.nn.functional.normalize(cls, p=2, dim=1)
        return cls

def get_bert_embeddings(texts: pd.Series, model: nn.Module, batch_size=128):
    embs = []
    model.eval()
    for i in tqdm(range(0, len(texts), batch_size), desc="BERT embeddings"):
        batch = texts.iloc[i:i+batch_size].tolist()
        with torch.no_grad():
            v = model(batch)  # [B, hidden]
        embs.append(v.cpu().numpy())
    return np.vstack(embs)

done = timer("BERT encode train+test")
bert_model = Bert(MODEL_DIR, trainable=False)
train = pd.read_pickle(os.path.join(WORK_DIR, "train_clean.pkl"))
test  = pd.read_pickle(os.path.join(WORK_DIR,  "test_clean.pkl"))

X_tr = get_bert_embeddings(train["title_c"], bert_model, batch_size=128)
X_te = get_bert_embeddings(test["title_c"],  bert_model, batch_size=128)

np.save(os.path.join(WORK_DIR, "X_tr_bert.npy"), X_tr)
np.save(os.path.join(WORK_DIR, "X_te_bert.npy"), X_te)
done()
print("Step 2 saved: X_tr_bert.npy, X_te_bert.npy; shapes:", X_tr.shape, X_te.shape)



# ==== Step 2b: TF-IDF (char_wb) -> normalized for cosine ====
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

done = timer("TF-IDF build")
train = pd.read_pickle(os.path.join(WORK_DIR, "train_clean.pkl"))
test  = pd.read_pickle(os.path.join(WORK_DIR, "test_clean.pkl"))

all_text = pd.concat([train["title_c"], test["title_c"]], axis=0)
tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=2, max_features=200_000)
X_all = tfidf.fit_transform(all_text)
X_tr_tfidf = normalize(X_all[:len(train)])
X_te_tfidf = normalize(X_all[len(train):])
import joblib
joblib.dump(tfidf, os.path.join(WORK_DIR, "tfidf_vectorizer.joblib"))
# сохраним в .npz (sparse)
from scipy.sparse import save_npz
save_npz(os.path.join(WORK_DIR, "X_tr_tfidf.npz"), X_tr_tfidf)
save_npz(os.path.join(WORK_DIR, "X_te_tfidf.npz"), X_te_tfidf)
done()
print("Step 2b saved: X_tr_tfidf.npz, X_te_tfidf.npz")



# ==== Step 2b-bis: KNN by TF-IDF for train/test ====
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import load_npz
import numpy as np, os

TOPK_TFIDF = 80  # как для BERT TOPK=80

X_tr_tfidf = load_npz(os.path.join(WORK_DIR, "X_tr_tfidf.npz")).tocsr()
X_te_tfidf = load_npz(os.path.join(WORK_DIR, "X_te_tfidf.npz")).tocsr()

# train
nn_tfidf_tr = NearestNeighbors(n_neighbors=min(TOPK_TFIDF, X_tr_tfidf.shape[0]), metric="cosine", n_jobs=-1)
nn_tfidf_tr.fit(X_tr_tfidf)
D_tr_tf, I_tr_tf = nn_tfidf_tr.kneighbors(X_tr_tfidf, return_distance=True)
S_tr_tf = 1.0 - D_tr_tf
np.save(os.path.join(WORK_DIR, "train_nn_indices_tfidf.npy"), I_tr_tf)
np.save(os.path.join(WORK_DIR, "train_nn_sims_tfidf.npy"),    S_tr_tf)

# test
nn_tfidf_te = NearestNeighbors(n_neighbors=min(TOPK_TFIDF, X_te_tfidf.shape[0]), metric="cosine", n_jobs=-1)
nn_tfidf_te.fit(X_te_tfidf)
D_te_tf, I_te_tf = nn_tfidf_te.kneighbors(X_te_tfidf, return_distance=True)
S_te_tf = 1.0 - D_te_tf
np.save(os.path.join(WORK_DIR, "test_nn_indices_tfidf.npy"), I_te_tf)
np.save(os.path.join(WORK_DIR, "test_nn_sims_tfidf.npy"),    S_te_tf)

print("Saved TF-IDF KNN for train/test")



# ==== Step 2c: precompute image-lite features for all images (fast, once) ====
import os, cv2, numpy as np, pandas as pd
from tqdm.auto import tqdm

def _load_img_quick(path):
    # быстрое чтение + уменьшение
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None: return None
    h, w = img.shape[:2]
    if max(h,w) > 128:
        r = 128 / max(h,w)
        img = cv2.resize(img, (int(w*r), int(h*r)), interpolation=cv2.INTER_AREA)
    return img

def _img_feats_quick(img):
    # более дешёвая гистограмма HSV: 8x4x4
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2], None, [8,4,4], [0,180,0,256,0,256]).flatten().astype(np.float32)
    s = hist.sum()
    if s>0: hist /= s
    mean = img.mean(axis=(0,1)).astype(np.float32)  # B,G,R
    h, w = img.shape[:2]
    ar = float(w)/max(float(h),1.0)
    return hist, mean, ar

def compute_pack(df, img_dir):
    H = np.zeros((len(df), 8*4*4), dtype=np.float32)
    M = np.zeros((len(df), 3), dtype=np.float32)
    AR = np.ones(len(df), dtype=np.float32)
    for i in tqdm(range(len(df)), desc=f"image feats @ {os.path.basename(img_dir)}"):
        path = os.path.join(img_dir, df.at[i, "image"])
        img = _load_img_quick(path)
        if img is None: continue
        h, m, ar = _img_feats_quick(img)
        H[i] = h; M[i] = m; AR[i] = ar
    return H, M, AR

done = timer("precompute image feats")
train = pd.read_pickle(os.path.join(WORK_DIR, "train_clean.pkl"))
test  = pd.read_pickle(os.path.join(WORK_DIR,  "test_clean.pkl"))

H_tr, M_tr, AR_tr = compute_pack(train, os.path.join(DATA_DIR, "train_images"))
H_te, M_te, AR_te = compute_pack(test,  os.path.join(DATA_DIR, "test_images"))

np.savez_compressed(os.path.join(WORK_DIR, "img_feats_train.npz"), H=H_tr, M=M_tr, AR=AR_tr)
np.savez_compressed(os.path.join(WORK_DIR, "img_feats_test.npz"),  H=H_te, M=M_te, AR=AR_te)
done()
print("Step 2c saved: img_feats_train.npz, img_feats_test.npz")



# ==== Step 2d: precompute ORB descriptors (train/test) ====
import cv2, os, numpy as np, pandas as pd
from tqdm.auto import tqdm

def compute_orb_pack(df, img_dir, max_kp=256):
    orb = cv2.ORB_create(nfeatures=max_kp, fastThreshold=5, edgeThreshold=15)
    K = []
    D = []
    for i in tqdm(range(len(df)), desc=f"ORB @ {os.path.basename(img_dir)}"):
        path = os.path.join(img_dir, df.at[i,"image"])
        img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            K.append(np.empty((0,2), np.float32))
            D.append(np.empty((0,32), np.uint8))
            continue
        h,w = img.shape[:2]
        if max(h,w) > 256:  # поменьше, быстрее
            r = 256 / max(h,w)
            img = cv2.resize(img, (int(w*r), int(h*r)), interpolation=cv2.INTER_AREA)
        kps, des = orb.detectAndCompute(img, None)
        if des is None:
            K.append(np.empty((0,2), np.float32))
            D.append(np.empty((0,32), np.uint8))
        else:
            K.append(np.array([k.pt for k in kps], np.float32))
            D.append(des.astype(np.uint8))
    return K, D

done = timer("precompute ORB")
train = pd.read_pickle(os.path.join(WORK_DIR, "train_clean.pkl"))
test  = pd.read_pickle(os.path.join(WORK_DIR,  "test_clean.pkl"))

K_tr, D_tr = compute_orb_pack(train, os.path.join(DATA_DIR, "train_images"))
K_te, D_te = compute_orb_pack(test,  os.path.join(DATA_DIR, "test_images"))

# сохраним как object npy (каждый элемент — массив переменной длины)
np.save(os.path.join(WORK_DIR, "orb_kp_train.npy"), np.array(K_tr, dtype=object))
np.save(os.path.join(WORK_DIR, "orb_desc_train.npy"), np.array(D_tr, dtype=object))
np.save(os.path.join(WORK_DIR, "orb_kp_test.npy"),  np.array(K_te, dtype=object))
np.save(os.path.join(WORK_DIR, "orb_desc_test.npy"),np.array(D_te, dtype=object))
done()
print("Step 2d saved: ORB kp/desc for train/test")



# ==== Image-lite features (HSV hist, mean color, aspect ratio, same_image) ====
import cv2
from functools import lru_cache

TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR  = os.path.join(DATA_DIR, "test_images")

def _load_img(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    # ускоряем и стабилизируем
    h, w = img.shape[:2]
    scale_to = 128
    if max(h,w) > scale_to:
        r = scale_to / max(h,w)
        img = cv2.resize(img, (int(w*r), int(h*r)), interpolation=cv2.INTER_AREA)
    return img

def _img_feats(img):
    # hsv hist (16x8x8), mean BGR, aspect ratio
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2], None, [16,8,8], [0,180,0,256,0,256]).flatten().astype(np.float32)
    s = hist.sum()
    if s > 0: hist /= s
    mean = img.mean(axis=(0,1)).astype(np.float32)  # B,G,R
    h, w = img.shape[:2]
    ar = float(w)/max(float(h),1.0)
    return hist, mean, ar

# кэшируем фичи по индексам, чтобы не читать повторно
_img_cache_train = {}
_img_cache_test  = {}

def get_train_img_feats(i, train_df):
    x = _img_cache_train.get(i)
    if x is not None: return x
    path = os.path.join(TRAIN_IMG_DIR, train_df.at[i, "image"])
    img = _load_img(path)
    feats = _img_feats(img) if img is not None else (
        np.zeros((16*8*8,), np.float32), np.zeros(3, np.float32), 1.0
    )
    _img_cache_train[i] = feats
    return feats

def get_test_img_feats(i, test_df):
    x = _img_cache_test.get(i)
    if x is not None: return x
    path = os.path.join(TEST_IMG_DIR, test_df.at[i, "image"])
    img = _load_img(path)
    feats = _img_feats(img) if img is not None else (
        np.zeros((16*8*8,), np.float32), np.zeros(3, np.float32), 1.0
    )
    _img_cache_test[i] = feats
    return feats

def hist_intersection(h1, h2):
    return float(np.minimum(h1, h2).sum())

def mean_color_dist(m1, m2):
    # нормированная евклид. дистанция (меньше — ближе)
    denom = np.linalg.norm(m1) + np.linalg.norm(m2) + 1e-9
    return float(np.linalg.norm(m1 - m2) / denom)



# ==== Step 3: train-side KNN (BERT) + pHash buckets ====
from sklearn.neighbors import NearestNeighbors
import joblib

TOPK = 50

done = timer("KNN train + buckets")
train = pd.read_pickle(os.path.join(WORK_DIR, "train_clean.pkl"))
X_tr  = np.load(os.path.join(WORK_DIR, "X_tr_bert.npy")).astype(np.float32)

nn_text_tr = NearestNeighbors(n_neighbors=min(TOPK, len(train)), metric="cosine", n_jobs=-1)
nn_text_tr.fit(X_tr)
dist_tr, ind_tr = nn_text_tr.kneighbors(X_tr, return_distance=True)
sim_tr = 1.0 - dist_tr
np.save(os.path.join(WORK_DIR, "train_nn_indices_bert.npy"), ind_tr)
np.save(os.path.join(WORK_DIR, "train_nn_sims_bert.npy"),    sim_tr)

bucket_to_indices_tr = train.groupby("ph_bucket").indices
joblib.dump(bucket_to_indices_tr, os.path.join(WORK_DIR, "bucket_to_indices_tr.joblib"))
done()
print("Step 3 saved: train_nn_* + bucket_to_indices_tr.joblib")



# ==== Step 4 (FAST): build pairwise train data with 14 features and union candidates ====
import os, numpy as np, pandas as pd, joblib
from tqdm.auto import tqdm
from scipy.sparse import load_npz
from joblib import Parallel, delayed

done = timer("build pairs FAST (14 feats, BERT∪TF-IDF∪pHash)")

# --- data & artifacts ---
train = pd.read_pickle(os.path.join(WORK_DIR, "train_clean.pkl"))

# BERT KNN (train)
ind_tr = np.load(os.path.join(WORK_DIR, "train_nn_indices_bert.npy"))
sim_tr = np.load(os.path.join(WORK_DIR, "train_nn_sims_bert.npy"))

# TF-IDF KNN (train)
I_tr_tf = np.load(os.path.join(WORK_DIR, "train_nn_indices_tfidf.npy"))
# S_tr_tf = np.load(os.path.join(WORK_DIR, "train_nn_sims_tfidf.npy"))  # не обязателен

# pHash buckets
bucket_to_indices_tr = joblib.load(os.path.join(WORK_DIR, "bucket_to_indices_tr.joblib"))

# TF-IDF sparse (нормализованный)
X_tr_tfidf = load_npz(os.path.join(WORK_DIR, "X_tr_tfidf.npz")).tocsr()

# precomputed image-lite feats
img_tr = np.load(os.path.join(WORK_DIR, "img_feats_train.npz"))
H_tr, M_tr, AR_tr = img_tr["H"], img_tr["M"], img_tr["AR"]

# precomputed ORB desc (object arrays)
D_tr_orb = np.load(os.path.join(WORK_DIR, "orb_desc_train.npy"), allow_pickle=True)

# --- small helpers ---
def hist_intersection_arr(h1, h2):
    return float(np.minimum(h1, h2).sum())

def mean_color_dist_arr(m1, m2):
    denom = np.linalg.norm(m1) + np.linalg.norm(m2) + 1e-9
    return float(np.linalg.norm(m1 - m2) / denom)

# ORB matcher (BFMatcher Hamming + ratio test)
import cv2
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
def orb_match_ratio(desc1, desc2, ratio=0.75, max_pairs=128):
    if desc1 is None or desc2 is None or len(desc1)==0 or len(desc2)==0:
        return 0.0
    if len(desc1) > max_pairs: desc1 = desc1[:max_pairs]
    if len(desc2) > max_pairs: desc2 = desc2[:max_pairs]
    matches = bf.knnMatch(desc1, desc2, k=2)
    good = 0
    for m in matches:
        if len(m)==2 and m[0].distance < ratio * m[1].distance:
            good += 1
    denom = min(len(desc1), len(desc2)) + 1e-9
    return float(good / denom)

# --- text caches ---
titles_tr = train["title_c"].tolist()
tokens_tr = [token_set(t) for t in titles_tr]
lens_tr   = np.array([len(t) for t in titles_tr], dtype=np.float32)
nums_tr   = [extract_numbers(t) for t in titles_tr]
label_group = train["label_group"].values

# caps & filters
POS_CAP, NEG_CAP = 15, 20
SIM_PREFILT_T = 0.30   # пропускаем пары, где и BERT_sim низкий, и Hamming слишком большой
HAM_PREFILT_T = 20

def build_for_index(i):
    rng = np.random.default_rng(123 + i)
    # --- candidates = BERT ∪ TF-IDF ∪ pHash ---
    cand = set(ind_tr[i].tolist()) | set(I_tr_tf[i].tolist()) | set(bucket_to_indices_tr.get(train.at[i,"ph_bucket"], []))
    cand.discard(i)
    # префильтр (дешёвый)
    pruned = []
    for j in cand:
        sc = sim_tr[i, np.where(ind_tr[i]==j)[0][0]] if j in ind_tr[i] else 0.0
        ham = hamming_u64(train.at[i,"ph_int"], train.at[j,"ph_int"])
        if (sc >= SIM_PREFILT_T) or (ham <= HAM_PREFILT_T):
            pruned.append(j)
    cand = pruned

    pos = [j for j in cand if label_group[j]==label_group[i]]
    neg = [j for j in cand if label_group[j]!=label_group[i]]
    if len(pos) > POS_CAP: pos = rng.choice(pos, POS_CAP, replace=False).tolist()
    if len(neg) > NEG_CAP: neg = rng.choice(neg, NEG_CAP, replace=False).tolist()

    ti, li, ni, si = tokens_tr[i], lens_tr[i], nums_tr[i], titles_tr[i]
    h1, m1, ar1 = H_tr[i], M_tr[i], AR_tr[i]
    d1 = D_tr_orb[i]

    feats_local, y_local = [], []

    for j in pos + neg:
        # base 7
        sc   = sim_tr[i, np.where(ind_tr[i]==j)[0][0]] if j in ind_tr[i] else 0.0
        ham  = hamming_u64(train.at[i,"ph_int"], train.at[j,"ph_int"])
        jac  = (len(ti & tokens_tr[j]) / len(ti | tokens_tr[j])) if (ti or tokens_tr[j]) else 0.0
        lenr = float(min(li, lens_tr[j]) / max(li, lens_tr[j])) if max(li, lens_tr[j])>0 else 1.0
        nover= numeric_overlap(ni, nums_tr[j])
        pref = prefix_suffix_match(si, titles_tr[j])
        strict_num = 1.0 if (ni & nums_tr[j]) else 0.0

        # tf-idf cosine
        tfidf_sim = float(X_tr_tfidf[i].multiply(X_tr_tfidf[j]).sum())

        # image-lite 4
        h2, m2, ar2 = H_tr[j], M_tr[j], AR_tr[j]
        hist_int  = hist_intersection_arr(h1, h2)
        mean_dist = mean_color_dist_arr(m1, m2)
        ar_diff   = float(abs(ar1 - ar2) / max(ar1, ar2))
        same_img  = 1.0 if train.at[i,"image"] == train.at[j,"image"] else 0.0

        # ORB + SSIM
        d2 = D_tr_orb[j]
        orb_ratio = orb_match_ratio(d1, d2)

        # (SSIM по train нам не нужен в обучении — дороговато; оставим на тесте)
        ssim_val = 0.0  # placeholder, чтобы размер был одинаковым

        y = int(label_group[i]==label_group[j])

        feats_local.append([sc, ham, jac, lenr, nover, pref, strict_num,
                            tfidf_sim, hist_int, mean_dist, ar_diff, same_img,
                            orb_ratio, ssim_val])  # 14 фич
        y_local.append(y)

    if feats_local:
        return np.array(feats_local, dtype=np.float32), np.array(y_local, dtype=np.int8)
    else:
        return np.empty((0,14), np.float32), np.empty((0,), np.int8)

results = Parallel(n_jobs=-1, prefer="threads")(
    delayed(build_for_index)(i) for i in tqdm(range(len(train)), desc="pairs fast (14)")
)

pairs_feat = np.concatenate([r[0] for r in results if r[0].size], axis=0) if results else np.empty((0,14), np.float32)
pairs_y    = np.concatenate([r[1] for r in results if r[1].size], axis=0) if results else np.empty((0,), np.int8)

np.save(os.path.join(WORK_DIR, "pairs_feat.npy"), pairs_feat)
np.save(os.path.join(WORK_DIR, "pairs_y.npy"),    pairs_y)
done()
print("Step 4 FAST saved:", pairs_feat.shape, "positives:", int(pairs_y.sum()), "negatives:", int((pairs_y==0).sum()))



# ==== Step 5: CatBoost reranker (1000 iters) + thresholds ====
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
import joblib

done = timer("train CatBoost reranker")
pairs_feat = np.load(os.path.join(WORK_DIR, "pairs_feat.npy"))
pairs_y    = np.load(os.path.join(WORK_DIR, "pairs_y.npy"))

scaler = StandardScaler()
Xr = scaler.fit_transform(pairs_feat)
clf = CatBoostClassifier(
    iterations=1000,
    depth=6,
    learning_rate=0.1,
    loss_function="Logloss",
    random_seed=42,
    verbose=False
)
clf.fit(Xr, pairs_y)

joblib.dump(scaler, os.path.join(WORK_DIR, "scaler.joblib"))
joblib.dump(clf,    os.path.join(WORK_DIR, "reranker_catboost.joblib"))

# стартовые пороги (можно потом подвинуть, без грид-поиска)
SIM_T = 0.55   # bert cosine threshold
HAM_T = 12.0   # max Hamming
P_T   = 0.50   # min proba from CatBoost
with open(os.path.join(WORK_DIR, "thresholds.txt"), "w") as f:
    f.write(f"{SIM_T},{HAM_T},{P_T}\n")

done()
print("Step 5 saved: scaler, reranker, thresholds")



# ==== Fast grid search for thresholds (SIM_T, HAM_T, P_T) [12-feature safe] ====
import os, numpy as np, joblib
from tqdm.auto import tqdm
from sklearn.model_selection import GroupShuffleSplit
from scipy.sparse import load_npz

# --- load artifacts ---
train = pd.read_pickle(os.path.join(WORK_DIR, "train_clean.pkl"))
ind_tr = np.load(os.path.join(WORK_DIR, "train_nn_indices_bert.npy"))
sim_tr = np.load(os.path.join(WORK_DIR, "train_nn_sims_bert.npy"))
bucket_to_indices_tr = joblib.load(os.path.join(WORK_DIR, "bucket_to_indices_tr.joblib"))
scaler = joblib.load(os.path.join(WORK_DIR, "scaler.joblib"))
clf    = joblib.load(os.path.join(WORK_DIR, "reranker_catboost.joblib"))

# TF-IDF (train)
X_tr_tfidf = load_npz(os.path.join(WORK_DIR, "X_tr_tfidf.npz")).tocsr()

# image feats (train)
img_tr = np.load(os.path.join(WORK_DIR, "img_feats_train.npz"))
H_tr, M_tr, AR_tr = img_tr["H"], img_tr["M"], img_tr["AR"]

def hist_intersection_arr(h1, h2):
    return float(np.minimum(h1, h2).sum())
def mean_color_dist_arr(m1, m2):
    denom = np.linalg.norm(m1) + np.linalg.norm(m2) + 1e-9
    return float(np.linalg.norm(m1 - m2) / denom)

# text caches
titles = train["title_c"].tolist()
tokens = [token_set(t) for t in titles]
lens   = np.array([len(t) for t in titles], dtype=np.float32)
nums   = [extract_numbers(t) for t in titles]
pid    = train["posting_id"].values
lg     = train["label_group"].values

# --- validation subset ---
VAL_SIZE = 2000
gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
_, val_idx = next(gss.split(train, groups=lg))
VAL = val_idx[:VAL_SIZE]

# expected feature length from scaler (should be 12)
N_EXP = getattr(scaler, "n_features_in_", 12)

def fit_len(feat_vec, n=N_EXP):
    """Pad/truncate feature vector to expected length to avoid crashes."""
    if len(feat_vec) == n:
        return feat_vec
    if len(feat_vec) > n:
        return feat_vec[:n]
    # pad with zeros if fewer
    return feat_vec + [0.0] * (n - len(feat_vec))

# --- precompute features & proba for VAL once ---
# store: pid[i] -> list of (pid_j, sc, ham, proba)
precomp = {}
for i in tqdm(VAL, desc="precompute 12-feature once"):
    cand = set(ind_tr[i].tolist())
    b = train.at[i, "ph_bucket"]
    cand |= set(bucket_to_indices_tr.get(b, []))
    cand.discard(i)

    ti, li, ni, si = tokens[i], lens[i], nums[i], titles[i]
    h1, m1, ar1 = H_tr[i], M_tr[i], AR_tr[i]

    rows = []
    for j in cand:
        # базовые
        sc  = sim_tr[i, np.where(ind_tr[i]==j)[0][0]] if j in ind_tr[i] else 0.0
        ham = hamming_u64(train.at[i,"ph_int"], train.at[j,"ph_int"])
        jac  = (len(ti & tokens[j]) / len(ti | tokens[j])) if (ti or tokens[j]) else 0.0
        lenr = float(min(li, lens[j]) / max(li, lens[j])) if max(li, lens[j])>0 else 1.0
        nover= numeric_overlap(ni, nums[j])
        pref = prefix_suffix_match(si, titles[j])
        strict_num = 1.0 if (ni & nums[j]) else 0.0

        # TF-IDF cosine
        tfidf_sim = float(X_tr_tfidf[i].multiply(X_tr_tfidf[j]).sum())

        # image-lite
        h2, m2, ar2 = H_tr[j], M_tr[j], AR_tr[j]
        hist_int = hist_intersection_arr(h1, h2)
        mean_dist = mean_color_dist_arr(m1, m2)
        ar_diff   = float(abs(ar1 - ar2) / max(ar1, ar2))
        same_img  = 1.0 if train.at[i,"image"] == train.at[j,"image"] else 0.0

        # порядок фич должен совпадать со Step 4 (12 штук)
        feat_vec = [sc, ham, jac, lenr, nover, pref, strict_num,
                    tfidf_sim, hist_int, mean_dist, ar_diff, same_img]
        feat_vec = fit_len(feat_vec, N_EXP)

        x = scaler.transform(np.array([feat_vec], dtype=np.float32))
        proba = clf.predict_proba(x)[0,1] if hasattr(clf, "predict_proba") else float(clf.decision_function(x))
        rows.append((pid[j], sc, ham, proba))
    precomp[pid[i]] = rows

# ground-truth map
group_map = {}
for g in np.unique(lg[VAL]):
    idx = np.where(lg==g)[0]
    ids = set(pid[idx])
    for k in idx:
        group_map[pid[k]] = ids

def f1_from_thresholds(SIM_T, HAM_T, P_T):
    f1s=[]
    for i in VAL:
        pset = {pid[i]}
        for pid_j, sc, ham, proba in precomp[pid[i]]:
            if (sc >= SIM_T) or (ham <= HAM_T):
                if proba >= P_T:
                    pset.add(pid_j)
        tset = group_map[pid[i]]
        tp = len(pset & tset)
        prec = tp / len(pset) if pset else 0.0
        rec  = tp / len(tset) if tset else 0.0
        f1s.append(2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0)
    return float(np.mean(f1s))

# --- grids (можно сузить/расширить) ---
SIM_GRID = [0.53, 0.55, 0.57]
HAM_GRID = [8, 9, 10]
P_GRID   = [0.50, 0.55, 0.60]
# оставить тот же грид-блок — он быстро доберёт локальный максимум


best = (-1, None, None, None)
for s in SIM_GRID:
    for h in HAM_GRID:
        for pthr in P_GRID:
            score = f1_from_thresholds(s, h, pthr)
            print(f"F1={score:.4f} @ sim≥{s} ham≤{h} p≥{pthr}")
            if score > best[0]:
                best = (score, s, h, pthr)

print("BEST:", best)
# save for inference
_, SIM_T_BEST, HAM_T_BEST, P_T_BEST = best
with open(os.path.join(WORK_DIR, "thresholds.txt"), "w") as f:
    f.write(f"{SIM_T_BEST},{HAM_T_BEST},{P_T_BEST}\n")
print(f"Saved thresholds.txt: sim≥{SIM_T_BEST}, ham≤{HAM_T_BEST}, p≥{P_T_BEST} (expected {N_EXP} feats)")



# ==== Step 6: test-side KNN (BERT) & buckets ====
import joblib

done = timer("KNN test + buckets")
test = pd.read_pickle(os.path.join(WORK_DIR, "test_clean.pkl"))
X_te = np.load(os.path.join(WORK_DIR, "X_te_bert.npy")).astype(np.float32)

nn_text_te = NearestNeighbors(n_neighbors=min(50, len(test)), metric="cosine", n_jobs=-1)
nn_text_te.fit(X_te)
D_te, I_te = nn_text_te.kneighbors(X_te, return_distance=True)
S_te = 1.0 - D_te
np.save(os.path.join(WORK_DIR, "test_nn_indices_bert.npy"), I_te)
np.save(os.path.join(WORK_DIR, "test_nn_sims_bert.npy"),    S_te)

bucket_to_indices_te = test.groupby("ph_bucket").indices
joblib.dump(bucket_to_indices_te, os.path.join(WORK_DIR, "bucket_to_indices_te.joblib"))
done()
print("Step 6 saved: test_nn_* + bucket_to_indices_te.joblib")



# ==== Step 7: build edges on TEST (14 feats, union candidates) & components ====
import os, sys, numpy as np, pandas as pd, joblib
from tqdm.auto import tqdm
from scipy.sparse import load_npz
import cv2

done = timer("edges + components (14 feats, BERT∪TF-IDF∪pHash)")

# --- data & artifacts ---
test = pd.read_pickle(os.path.join(WORK_DIR, "test_clean.pkl"))

# BERT KNN (test)
I_te = np.load(os.path.join(WORK_DIR, "test_nn_indices_bert.npy"))
S_te = np.load(os.path.join(WORK_DIR, "test_nn_sims_bert.npy"))

# TF-IDF KNN (test)
I_te_tf = np.load(os.path.join(WORK_DIR, "test_nn_indices_tfidf.npy"))
# S_te_tf = np.load(os.path.join(WORK_DIR, "test_nn_sims_tfidf.npy"))

# pHash buckets
bucket_to_indices_te = joblib.load(os.path.join(WORK_DIR, "bucket_to_indices_te.joblib"))

# TF-IDF sparse (test)
X_te_tfidf = load_npz(os.path.join(WORK_DIR, "X_te_tfidf.npz")).tocsr()

# image-lite feats (test)
img_te = np.load(os.path.join(WORK_DIR, "img_feats_test.npz"))
H_te, M_te, AR_te = img_te["H"], img_te["M"], img_te["AR"]

# ORB desc (test)
D_te_orb = np.load(os.path.join(WORK_DIR, "orb_desc_test.npy"), allow_pickle=True)

# scaler & model
scaler = joblib.load(os.path.join(WORK_DIR, "scaler.joblib"))
clf    = joblib.load(os.path.join(WORK_DIR, "reranker_catboost.joblib"))

# thresholds
with open(os.path.join(WORK_DIR, "thresholds.txt")) as f:
    SIM_T, HAM_T, P_T = map(float, f.read().strip().split(","))

# text caches
tokens_te = [token_set(t) for t in test["title_c"]]
lens_te   = np.array([len(t) for t in test["title_c"]], dtype=np.float32)
nums_te   = [extract_numbers(t) for t in test["title_c"]]

# helpers
def hist_intersection_arr(h1, h2):
    return float(np.minimum(h1, h2).sum())

def mean_color_dist_arr(m1, m2):
    denom = np.linalg.norm(m1) + np.linalg.norm(m2) + 1e-9
    return float(np.linalg.norm(m1 - m2) / denom)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
def orb_match_ratio(desc1, desc2, ratio=0.75, max_pairs=128):
    if desc1 is None or desc2 is None or len(desc1)==0 or len(desc2)==0:
        return 0.0
    if len(desc1) > max_pairs: desc1 = desc1[:max_pairs]
    if len(desc2) > max_pairs: desc2 = desc2[:max_pairs]
    matches = bf.knnMatch(desc1, desc2, k=2)
    good = 0
    for m in matches:
        if len(m)==2 and m[0].distance < ratio * m[1].distance:
            good += 1
    denom = min(len(desc1), len(desc2)) + 1e-9
    return float(good / denom)

TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
def ssim64(img_path1, img_path2):
    g1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    g2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
    if g1 is None or g2 is None:
        return 0.0
    if g1.shape != (64,64):
        g1 = cv2.resize(g1, (64,64), interpolation=cv2.INTER_AREA)
    if g2.shape != (64,64):
        g2 = cv2.resize(g2, (64,64), interpolation=cv2.INTER_AREA)
    C1, C2 = (0.01*255)**2, (0.03*255)**2
    mu1, mu2 = g1.mean(), g2.mean()
    v1, v2   = g1.var(),  g2.var()
    cov12    = ((g1 - mu1)*(g2 - mu2)).mean()
    ssim = ((2*mu1*mu2 + C1)*(2*cov12 + C2))/((mu1**2 + mu2**2 + C1)*(v1 + v2 + C2) + 1e-9)
    return float(max(0.0, min(1.0, ssim)))

# edges
edges = [[] for _ in range(len(test))]

for i in tqdm(range(len(test)), desc="edges (14)", mininterval=2.0):
    # --- candidates = BERT ∪ TF-IDF ∪ pHash ---
    cand = set(I_te[i].tolist()) | set(I_te_tf[i].tolist()) | set(bucket_to_indices_te.get(test.at[i,"ph_bucket"], []))
    cand.discard(i)

    ti = tokens_te[i]; li = lens_te[i]; ni = nums_te[i]; si = test.at[i,"title_c"]
    h1, m1, ar1 = H_te[i], M_te[i], AR_te[i]
    d1 = D_te_orb[i]
    img_i_path = os.path.join(TEST_IMG_DIR, test.at[i,"image"])

    keep = []
    for j in cand:
        sc  = S_te[i, np.where(I_te[i]==j)[0][0]] if j in I_te[i] else 0.0
        ham = hamming_u64(test.at[i,"ph_int"], test.at[j,"ph_int"])

        if (sc >= SIM_T) or (ham <= HAM_T):
            jac  = (len(ti & tokens_te[j]) / len(ti | tokens_te[j])) if (tokens_te[j] or ti) else 0.0
            lenr = float(min(li, lens_te[j]) / max(li, lens_te[j])) if max(li, lens_te[j])>0 else 1.0
            nover= numeric_overlap(ni, nums_te[j])
            pref = prefix_suffix_match(si, test.at[j,"title_c"])
            strict_num = 1.0 if (ni & nums_te[j]) else 0.0

            tfidf_sim = float(X_te_tfidf[i].multiply(X_te_tfidf[j]).sum())

            h2, m2, ar2 = H_te[j], M_te[j], AR_te[j]
            hist_int  = hist_intersection_arr(h1, h2)
            mean_dist = mean_color_dist_arr(m1, m2)
            ar_diff   = float(abs(ar1 - ar2) / max(ar1, ar2))
            same_img  = 1.0 if test.at[i,"image"] == test.at[j,"image"] else 0.0

            d2 = D_te_orb[j]
            orb_ratio = orb_match_ratio(d1, d2)

            ssim_val = ssim64(img_i_path, os.path.join(TEST_IMG_DIR, test.at[j,"image"]))

            feat_vec = [sc, ham, jac, lenr, nover, pref, strict_num,
                        tfidf_sim, hist_int, mean_dist, ar_diff, same_img,
                        orb_ratio, ssim_val]

            # подгоним длину под scaler в случае рассинхрона
            n_exp = getattr(scaler, "n_features_in_", len(feat_vec))
            if len(feat_vec) != n_exp:
                if len(feat_vec) > n_exp:
                    feat_vec = feat_vec[:n_exp]
                else:
                    feat_vec = feat_vec + [0.0]*(n_exp - len(feat_vec))

            x = scaler.transform(np.array([feat_vec], dtype=np.float32))
            p = clf.predict_proba(x)[0,1]
            if p >= P_T:
                keep.append(j)

    edges[i] = keep

# --- connected components ---
sys.setrecursionlimit(10_000_000)
G = [set() for _ in range(len(test))]
for i in range(len(test)):
    for j in edges[i]:
        G[i].add(j); G[j].add(i)

visited = [False]*len(test)
components = []
for i in range(len(test)):
    if not visited[i]:
        stack=[i]; visited[i]=True; comp=[i]
        while stack:
            u = stack.pop()
            for v in G[u]:
                if not visited[v]:
                    visited[v]=True
                    stack.append(v)
                    comp.append(v)
        components.append(comp)

components = [c[:50] for c in components]
node2comp = np.empty(len(test), dtype=object)
for comp in components:
    for idx in comp:
        node2comp[idx] = comp

# save artifacts
np.save(os.path.join(WORK_DIR, "edges.npy"),      np.array([np.array(e, dtype=np.int32) for e in edges], dtype=object))
np.save(os.path.join(WORK_DIR, "components.npy"), np.array([np.array(c, dtype=np.int32) for c in components], dtype=object))
np.save(os.path.join(WORK_DIR, "node2comp.npy"),  node2comp)
done()
print("Step 7 saved (14 feats, union candidates).")



# ==== Step 8: submission ====
done = timer("build submission")
test = pd.read_pickle(os.path.join(WORK_DIR, "test_clean.pkl"))
node2comp = np.load(os.path.join(WORK_DIR, "node2comp.npy"), allow_pickle=True)
pid_te = test["posting_id"].values

matches = []
for i in range(len(test)):
    comp = node2comp[i]
    ids = [pid_te[i]] if comp is None else [pid_te[k] for k in comp]
    if pid_te[i] not in ids:
        ids = [pid_te[i]] + ids
    matches.append(" ".join(ids))

sub = pd.DataFrame({"posting_id": pid_te, "matches": matches})
sub.to_csv("submission.csv", index=False)
done()
print("Saved submission.csv with", len(sub), "rows")

# (опционально) очистим WORK_DIR, чтобы Output версии был лёгким
KEEP = {"submission.csv"}
for fname in os.listdir(WORK_DIR):
    if fname not in KEEP:
        try: os.remove(os.path.join(WORK_DIR, fname))
        except: pass
print("Cleaned working dir; kept only submission.csv")



import pandas as pd

sub = pd.read_csv("submission.csv")
print(sub.shape)
print(sub.head(10))





