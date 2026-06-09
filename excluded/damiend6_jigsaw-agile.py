# =================== Jigsaw Agile — TFIDF + DeBERTa (HF) + Auto-blend ===================
import os, re, gc, math, numpy as np, pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.base import clone

# ------------------ Chargement données ------------------
COMP = "/kaggle/input/jigsaw-agile-community-rules"
train  = pd.read_csv(f"{COMP}/train.csv")
test   = pd.read_csv(f"{COMP}/test.csv")
sample = pd.read_csv(f"{COMP}/sample_submission.csv")
print(f"Train: {train.shape} | Test: {test.shape}")
print("Sample columns:", list(sample.columns))

# ------------------ Construction texte ------------------
def build_text(df: pd.DataFrame) -> pd.Series:
    txt = df["body"].fillna("").astype(str)
    if "title" in df.columns:     txt = txt + " [T] "  + df["title"].fillna("").astype(str)
    if "subreddit" in df.columns: txt = txt + " [SR] " + df["subreddit"].fillna("unk").astype(str)
    # nettoyage léger (on garde la casse pour cap-features)
    txt = txt.str.replace(r"http\S+|www\.\S+", " URL ", regex=True)
    txt = txt.str.replace(r"@\w+", " USER ", regex=True)
    return txt.str.replace(r"\s+", " ", regex=True).str.strip()

Xtr_text = build_text(train)
Xte_text = build_text(test)
y = train["rule_violation"].astype(int).values

# ------------------ Features manuelles simples ------------------
def small_feats(s: pd.Series) -> np.ndarray:
    L    = s.str.len().clip(lower=1).astype(float)
    excl = s.str.count("!").astype(float)
    ques = s.str.count(r"\?").astype(float)
    caps = s.str.count(r"[A-Z]").astype(float)
    return np.vstack([L, excl, ques, caps/L]).T

Ftr = small_feats(Xtr_text); Fte = small_feats(Xte_text)
scaler = StandardScaler()
Ftr = scaler.fit_transform(Ftr).astype(float)
Fte = scaler.transform(Fte).astype(float)

# ------------------ TF-IDF (word + char) ------------------
tfw = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95, max_features=60_000, sublinear_tf=True, lowercase=True)
tfc = TfidfVectorizer(analyzer="char", ngram_range=(2,6), min_df=2, max_df=0.95, max_features=60_000, sublinear_tf=True, lowercase=True)
try:
    Xw_tr = tfw.fit_transform(Xtr_text); Xw_te = tfw.transform(Xte_text)
    Xc_tr = tfc.fit_transform(Xtr_text); Xc_te = tfc.transform(Xte_text)
except ValueError:
    # fallback ultra robuste
    tfw = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=1.0, max_features=50_000, sublinear_tf=True)
    tfc = TfidfVectorizer(analyzer="char", ngram_range=(2,6), min_df=1, max_df=1.0, max_features=50_000, sublinear_tf=True)
    Xw_tr = tfw.fit_transform(Xtr_text); Xw_te = tfw.transform(Xte_text)
    Xc_tr = tfc.fit_transform(Xtr_text); Xc_te = tfc.transform(Xte_text)

X_tr_tfidf = hstack([Xw_tr, Xc_tr, csr_matrix(Ftr)], format="csr")
X_te_tfidf = hstack([Xw_te, Xc_te, csr_matrix(Fte)], format="csr")
print("Shapes TF:", X_tr_tfidf.shape, X_te_tfidf.shape)

# ------------------ Outils OOF ------------------
def fit_oof_and_test(name, est, X, y, Xtest, n_splits=5):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
    oof = np.zeros(len(y), dtype=float)
    Pte = np.zeros(Xtest.shape[0], dtype=float)
    for tr_idx, va_idx in kf.split(X, y):
        mdl = clone(est)
        mdl.fit(X[tr_idx], y[tr_idx])
        if hasattr(mdl, "predict_proba"):
            p_va = mdl.predict_proba(X[va_idx])[:,1]
            p_te = mdl.predict_proba(Xtest)[:,1]
        else:
            dv  = mdl.decision_function(X[va_idx]); dt = mdl.decision_function(Xtest)
            p_va = 1/(1+np.exp(-dv)); p_te = 1/(1+np.exp(-dt))
        oof[va_idx] = p_va
        Pte += p_te / n_splits
    auc = roc_auc_score(y, oof)
    ll  = log_loss(y, np.clip(oof,1e-7,1-1e-7))
    print(f"   {name:<12} -> AUC(OOF)={auc:.4f}  LogLoss(OOF)={ll:.4f}")
    return oof, Pte

# ------------------ Modèles TF-IDF ------------------
# Cherche le meilleur C pour LR, + un SVC calibré (complémentaire)
c_grid = [1.0, 2.0, 3.0, 4.0]
best_lr = (None, 1e9, None, None)  # (C, ll, oof, pte)
for C in c_grid:
    est = LogisticRegression(C=C, solver="liblinear", max_iter=2000, random_state=123)
    oof, pte = fit_oof_and_test(f"logreg_C{C}", est, X_tr_tfidf, y, X_te_tfidf)
    ll = log_loss(y, np.clip(oof,1e-7,1-1e-7))
    if ll < best_lr[1]:
        best_lr = (C, ll, oof, pte)

svc_cal = CalibratedClassifierCV(LinearSVC(C=1.0, random_state=123), cv=3, method="sigmoid")
oof_svc, pte_svc = fit_oof_and_test("svc_cal_C1", svc_cal, X_tr_tfidf, y, X_te_tfidf)

# Blend TF-IDF LR+SVC (coarse puis raffinement fin)
w_best, ll_best, oof_tf, pte_tf = None, 1e9, None, None
for w in np.linspace(0.70, 0.99, 30):
    o = w*best_lr[2] + (1-w)*oof_svc
    ll = log_loss(y, np.clip(o,1e-7,1-1e-7))
    if ll < ll_best:
        w_best, ll_best, oof_tf, pte_tf = float(w), float(ll), o, w*best_lr[3] + (1-w)*pte_svc
# affiner autour de w_best à pas 0.005
for w in np.linspace(max(0,w_best-0.03), min(1,w_best+0.03), 25):
    o = w*best_lr[2] + (1-w)*oof_svc
    ll = log_loss(y, np.clip(o,1e-7,1-1e-7))
    if ll < ll_best:
        w_best, ll_best, oof_tf, pte_tf = float(w), float(ll), o, w*best_lr[3] + (1-w)*pte_svc
print(f"\n⇒ Blend TF-IDF choisi: w(LR)={w_best:.2f} | LogLoss(OOF)={ll_best:.4f}")

# ------------------ Embeddings HF (DeBERTa/Roberta) si dispo ------------------
use_hf = os.environ.get("HF_READY","0") == "1"
HF_PATH = os.environ.get("HF_PATH")

oof_hf = None; pte_hf = None
if use_hf and HF_PATH and os.path.isdir(HF_PATH):
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(HF_PATH, use_fast=True)
    mdl = AutoModel.from_pretrained(HF_PATH).to(device)
    mdl.eval()

    def encode(texts, batch=32, max_len=192):
        # auto-ajuste le batch si CPU
        if device == "cpu": batch = 8
        outs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                enc = tok(list(texts[i:i+batch]), padding="max_length",
                          truncation=True, max_length=max_len, return_tensors="pt")
                enc = {k:v.to(device) for k,v in enc.items()}
                with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                    out = mdl(**enc)  # last_hidden_state
                    attn = enc["attention_mask"].unsqueeze(-1)
                    x = (out.last_hidden_state * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1)
                outs.append(x.detach().cpu().numpy())
        return np.vstack(outs).astype(np.float32)

    print(f"\n== Embeddings HF depuis: {HF_PATH} ==")
    X_tr_hf = encode(Xtr_text)
    X_te_hf = encode(Xte_text)
    print("Embeddings:", X_tr_hf.shape, X_te_hf.shape)

    # OOF sur embeddings (LogReg)
    est_hf = LogisticRegression(C=2.0, solver="liblinear", max_iter=2000, random_state=123)
    oof_hf, pte_hf = fit_oof_and_test("HF_LogReg", est_hf, X_tr_hf, y, X_te_hf)

    # Blend TF (oof_tf/pte_tf) + HF (oof_hf/pte_hf)
    a_best, ll2_best, oof_blend, pte_blend = 0.0, 1e9, None, None
    for a in np.linspace(0.2, 0.8, 25):
        o = a*oof_hf + (1-a)*oof_tf
        ll = log_loss(y, np.clip(o,1e-7,1-1e-7))
        if ll < ll2_best:
            a_best, ll2_best = float(a), float(ll)
            oof_blend, pte_blend = o, a*pte_hf + (1-a)*pte_tf
    # raffine à pas 0.01
    for a in np.linspace(max(0,a_best-0.1), min(1,a_best+0.1), 21):
        o = a*oof_hf + (1-a)*oof_tf
        ll = log_loss(y, np.clip(o,1e-7,1-1e-7))
        if ll < ll2_best:
            a_best, ll2_best = float(a), float(ll)
            oof_blend, pte_blend = o, a*pte_hf + (1-a)*pte_tf

    print(f"\n⇒ Ajout HF: a={a_best:.2f} | LogLoss(OOF)={ll2_best:.4f}  (TF-only={ll_best:.4f})")
    p_test_final = pte_blend
else:
    print("\n== Pas de modèle HF utilisable (on reste TF-IDF only) ==")
    p_test_final = pte_tf

# ------------------ Soumission (probas) ------------------
p_test_final = np.clip(p_test_final, 1e-7, 1-1e-7)
sub = pd.DataFrame({"row_id": test["row_id"].astype(int), "rule_violation": p_test_final.astype(float)})

# vérifications strictes
assert sub.shape == sample.shape, f"Shape mismatch: sub={sub.shape}, sample={sample.shape}"
assert list(sub.columns) == list(sample.columns)
assert set(sub["row_id"]) == set(sample["row_id"])
# aligne l'ordre exact de row_id
sub = sample[["row_id"]].merge(sub, on="row_id", how="left")
assert sub["rule_violation"].notna().all()

out = "/kaggle/working/submission.csv"
sub.to_csv(out, index=False)
print(f"\n✅ Écrit: {out} {sub.shape}\n", sub.head())
# ========================================================================================


