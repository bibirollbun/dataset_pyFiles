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


# ================= Kaggle — High-AUC Protector Model (HF 4.44.2, logit-blend, no sentence-transformers) =================
# Dependencies (stable)
!pip -q uninstall -y sentence-transformers transformers tokenizers >/dev/null 2>&1 || true
!pip -q install -U "transformers==4.44.2" "huggingface-hub>=0.28.1" "tokenizers==0.19.1" accelerate "xgboost>=2.0.0" lightgbm >/dev/null

import os, re, glob, warnings, numpy as np, pandas as pd, torch
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
np.random.seed(42); torch.manual_seed(42)

# ---------------- Config (auto GPU/CPU) ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    EMBED_MODELS = [
        ("intfloat/multilingual-e5-large", True),        # mạnh (GPU)
        ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", False)
    ]
    MAX_LEN, BATCH = 384, 128
else:
    EMBED_MODELS = [
        ("intfloat/multilingual-e5-small", True),        # nhanh (CPU)
        ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", False)
    ]
    MAX_LEN, BATCH = 160, 64

N_FOLDS   = 5
CACHE_DIR = "/kaggle/working"  # cache embeddings để lần sau chạy nhanh

# ---------------- Find files ----------------
def find_file(name_like):
    cands = sorted(glob.glob(f"/kaggle/input/**/{name_like}", recursive=True))
    if not cands: raise FileNotFoundError(name_like)
    pref = [p for p in cands if re.search(r"(rmit|hackathon)", p, re.I)]
    return pref[0] if pref else cands[0]

train_p  = find_file("train.csv")
test_p   = find_file("test.csv")
sample_p = find_file("sample_submission.csv")
print("Using files:\n ", train_p, "\n ", test_p, "\n ", sample_p)

train = pd.read_csv(train_p)
test  = pd.read_csv(test_p)
sample_sub = pd.read_csv(sample_p)

# ---------------- Columns & labels ----------------
def infer_text_col(df):
    for c in df.columns:
        if c.lower() in ("prompt","text","input","message","content"): return c
    for c in df.columns:
        if df[c].dtype==object: return c
    raise RuntimeError("Cannot infer text column")

def infer_target_col(df):
    for c in df.columns:
        if c.lower() in ("target","label","y","is_jailbreak","jailbreak"): return c
    for c in df.columns:
        if c.lower() not in ("id","idx","identifier") and df[c].nunique()<=10: return c
    raise RuntimeError("Cannot infer target column")

text_col   = infer_text_col(train)
target_col = infer_target_col(train)
train[text_col] = train[text_col].fillna("").astype(str)
test[text_col]  = test[text_col].fillna("").astype(str)

y_raw = train[target_col]
if np.issubdtype(y_raw.dtype, np.number):
    y = y_raw.astype(int).values
else:
    uniq = [str(u) for u in y_raw.unique()]
    if any("jail" in u.lower() for u in uniq):
        mapping = {u: (1 if "jail" in u.lower() else 0) for u in uniq}
    else:
        most = y_raw.mode().iloc[0]; mapping = {u: (0 if u==most else 1) for u in uniq}
    print("Label mapping:", mapping)
    y = y_raw.map(mapping).astype(int).values

texts_tr = train[text_col].tolist()
texts_te = test[text_col].tolist()

# ---------------- Prompt-aware features ----------------
KEYPHRASES = [
    "ignore previous","bypass","jailbreak","override","disable safety",
    "system prompt","unfiltered","no restrictions","roleplay","pretend to",
    "do anything now","DAN","prompt injection","ignore all","as an ai",
    # mở rộng thêm một ít dấu hiệu hay gặp
    "developer mode","you must comply","no filter","unmoderated","simulate"
]
def prompt_feats(texts):
    M = len(KEYPHRASES)
    out = np.zeros((len(texts), 12+M), dtype=np.float32)
    for i,t in enumerate(texts):
        L=len(t); words=t.split()
        q=t.count('?'); e=t.count('!'); quotes=t.count('"')+t.count("'")
        upp=sum(1 for ch in t if ch.isupper()); url=1 if re.search(r"https?://",t) else 0
        punct=sum(ch in "{}[]()#@$%^&*<>|/~" for ch in t)
        ratio_up = upp/max(L,1); ratio_punct = punct/max(L,1)
        out[i,:12]=[
            np.log1p(L), np.log1p(len(words)),
            np.mean([len(w) for w in words]) if words else 0.0,
            q,e,quotes,upp,url,ratio_up,ratio_punct,
            t.lower().count("system:"), t.lower().count("assistant:")
        ]
        tl=t.lower()
        for j,k in enumerate(KEYPHRASES):
            out[i,12+j]=1.0 if k in tl else 0.0
    return out

feat_tr = prompt_feats(texts_tr)
feat_te = prompt_feats(texts_te)

# ---------------- HF embeddings (mean pooling) w/ cache ----------------
from transformers import AutoTokenizer, AutoModel

def mean_pool(last_hidden_state, attn_mask):
    mask = attn_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state*mask).sum(1)/mask.sum(1).clamp(min=1e-9)

@torch.no_grad()
def embed_model(model_name, add_prefix, texts, cache_prefix, max_len=MAX_LEN, batch=BATCH):
    cache_path = os.path.join(CACHE_DIR, f"{cache_prefix}_{model_name.split('/')[-1]}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)
    device_local = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device_local); mdl.eval()
    vecs=[]
    for i in tqdm(range(0,len(texts),batch), desc=f"Embed {model_name.split('/')[-1]}"):
        batch_text=texts[i:i+batch]
        if add_prefix and model_name.startswith("intfloat/multilingual-e5"):
            batch_text=[f"query: {t}" for t in batch_text]
        enc=tok(batch_text, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device_local)
        out=mdl(**enc)
        sent=mean_pool(out.last_hidden_state, enc["attention_mask"])
        sent=torch.nn.functional.normalize(sent,p=2,dim=1)
        vecs.append(sent.detach().cpu().numpy())
    arr = np.vstack(vecs)
    np.save(cache_path, arr)
    return arr

emb_tr_list=[]; emb_te_list=[]
for mid,addp in EMBED_MODELS:
    emb_tr_list.append(embed_model(mid, addp, texts_tr, cache_prefix="Xtr"))
    emb_te_list.append(embed_model(mid, addp, texts_te, cache_prefix="Xte"))

X_tr = np.hstack(emb_tr_list+[feat_tr])
X_te = np.hstack(emb_te_list+[feat_te])
print("Final feature shapes:", X_tr.shape, X_te.shape)

# ---------------- 5-fold: 4 models (LR, XGB, LGBM, TFIDF-SVC) ----------------
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_lr = np.zeros(len(train)); te_lr = np.zeros(len(test))
oof_xg = np.zeros(len(train)); te_xg = np.zeros(len(test))
oof_lgb= np.zeros(len(train)); te_lgb= np.zeros(len(test))
oof_svc= np.zeros(len(train)); te_svc= np.zeros(len(test))  # TF-IDF SVC

for fold,(tr,va) in enumerate(skf.split(X_tr, y),1):
    Xtr,Xva = X_tr[tr], X_tr[va]; ytr,yva = y[tr], y[va]
    texts_tr_tr = [texts_tr[i] for i in tr]
    texts_tr_va = [texts_tr[i] for i in va]
    texts_te_full= texts_te  # dùng nguyên cho test

    # 1) Logistic Regression (trên embeddings + prompt feats)
    lr = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(solver="saga", C=2.0, max_iter=2000,
                           class_weight="balanced", random_state=100+fold)
    )
    lr.fit(Xtr,ytr)
    oof_lr[va]=lr.predict_proba(Xva)[:,1]; te_lr += lr.predict_proba(X_te)[:,1]/N_FOLDS
    print(f"[F{fold}] LR   AUC: {roc_auc_score(yva,oof_lr[va]):.5f}")

    # 2) XGBoost (2.x safe)
    dtr=xgb.DMatrix(Xtr,label=ytr); dva=xgb.DMatrix(Xva,label=yva); dte=xgb.DMatrix(X_te)
    xgb_params={"objective":"binary:logistic","eval_metric":"auc","eta":0.05,"max_depth":6,
                "subsample":0.9,"colsample_bytree":0.7,"min_child_weight":1.0,
                "lambda":1.2,"alpha":0.0,"tree_method":"hist","random_state":200+fold}
    bst=xgb.train(xgb_params,dtr, num_boost_round=4000, evals=[(dtr,"tr"),(dva,"va")],
                  early_stopping_rounds=150, verbose_eval=False)
    best_it = getattr(bst, "best_iteration", None)
    if best_it is not None:
        oof_xg[va]=bst.predict(dva,iteration_range=(0,best_it+1))
        te_xg += bst.predict(dte,iteration_range=(0,best_it+1))/N_FOLDS
    else:
        oof_xg[va]=bst.predict(dva); te_xg += bst.predict(dte)/N_FOLDS
    print(f"[F{fold}] XGB  AUC: {roc_auc_score(yva,oof_xg[va]):.5f}")

    # 3) LightGBM (>=4: callbacks)
    ltr=lgb.Dataset(Xtr,label=ytr); lva=lgb.Dataset(Xva,label=yva,reference=ltr)
    lgb_params={"objective":"binary","metric":"auc","learning_rate":0.05,"num_leaves":64,
                "feature_fraction":0.8,"bagging_fraction":0.9,"bagging_freq":1,
                "min_data_in_leaf":20,"lambda_l1":0.0,"lambda_l2":1.2,"verbose":-1,
                "seed":300+fold}
    callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)]
    lgbm=lgb.train(lgb_params,ltr, num_boost_round=4000,
                   valid_sets=[ltr,lva], valid_names=["tr","va"],
                   callbacks=callbacks)
    oof_lgb[va]=lgbm.predict(Xva, num_iteration=lgbm.best_iteration)
    te_lgb += lgbm.predict(X_te, num_iteration=lgbm.best_iteration)/N_FOLDS
    print(f"[F{fold}] LGB  AUC: {roc_auc_score(yva,oof_lgb[va]):.5f}")

    # 4) TF-IDF → LinearSVC (calibrated)  —— fit TRONG fold (không leakage)
    word_tv = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=120_000)
    char_tv = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2, max_features=80_000)
    union = FeatureUnion([("w", word_tv), ("c", char_tv)])
    base  = Pipeline([("tfidf", union), ("svc", LinearSVC(C=0.5, class_weight="balanced", random_state=400+fold))])
    svc_cal= CalibratedClassifierCV(base, method="sigmoid", cv=3)

    svc_cal.fit(texts_tr_tr, ytr)
    oof_svc[va] = svc_cal.predict_proba(texts_tr_va)[:,1]
    te_svc     += svc_cal.predict_proba(texts_te_full)[:,1]/N_FOLDS
    print(f"[F{fold}] TFIDF-SVC AUC: {roc_auc_score(yva,oof_svc[va]):.5f}")

print("\nOOF AUCs:",
      f"LR={roc_auc_score(y,oof_lr):.5f},",
      f"XGB={roc_auc_score(y,oof_xg):.5f},",
      f"LGB={roc_auc_score(y,oof_lgb):.5f},",
      f"TFIDF-SVC={roc_auc_score(y,oof_svc):.5f}")

# ---------------- Logit-blend + weight search (fine grid) ----------------
def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1-eps)
    return np.log(p/(1-p))
def sigmoid(x): return 1/(1+np.exp(-x))

oofs = np.stack([oof_lr, oof_xg, oof_lgb, oof_svc], axis=1)
tests= np.stack([te_lr,  te_xg,  te_lgb,  te_svc],  axis=1)
names= ["lr","xgb","lgb","tfidf"]

oofs_log  = logit(oofs); tests_log = logit(tests)

def best_blend_logit(oofs_log, y, step=0.02):
    best_auc, best_w = -1, None
    grid = np.arange(0.0, 1.0+1e-9, step)
    for w1 in grid:
      for w2 in grid:
        for w3 in grid:
          w4 = 1.0 - (w1+w2+w3)
          if w4 < -1e-9: continue
          w = np.array([w1,w2,w3,max(0.0,w4)], dtype=float); w/=w.sum()
          auc = roc_auc_score(y, sigmoid(oofs_log @ w))
          if auc > best_auc: best_auc, best_w = auc, w
    return best_auc, best_w

best_auc, w = best_blend_logit(oofs_log, y, step=0.02)
print(f"\nBest OOF AUC (logit-blend): {best_auc:.5f} with weights {dict(zip(names, np.round(w,3)))}")
test_pred = sigmoid(tests_log @ w)

# ---------------- Submission (probabilities) ----------------
id_col = next((c for c in sample_sub.columns if c.lower()=="id"), None)
if id_col is None and "Id" in test.columns: id_col="Id"
if id_col is None: id_col="id"; ids=np.arange(1,len(test)+1)
else: ids=test[id_col]

target_name = next((c for c in sample_sub.columns if c.lower() in ("target","label","prediction")), "TARGET")
sub = pd.DataFrame({id_col: ids, target_name: test_pred})
out_path="/kaggle/working/submission.csv"
sub.to_csv(out_path, index=False)
print(f"\nSaved submission to: {out_path}")
display(sub.head())
# =================================================================================================


