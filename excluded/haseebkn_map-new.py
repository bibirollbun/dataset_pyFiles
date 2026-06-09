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





import os, gc, math, warnings, numpy as np, pandas as pd, torch, glob, shutil
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy import sparse
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch import nn

DATA_DIR    = "/kaggle/input/map-charting-student-math-misunderstandings"
OUT_DIR     = "/kaggle/working"

MODEL_DIR_HINT = ""   

FOLDS       = 5
EPOCHS      = 3
BATCH_SIZE  = 16
MAX_LEN     = 256
LR          = 2e-5
WEIGHT_DECAY= 0.01
SEED        = 42
MIS_WEIGHT  = 1.0   # weight for misconception loss
CAT_WEIGHT  = 0.5   # weight for category loss
PATIENCE    = 1     

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); np.random.seed(SEED)

# ------------------------ I/O CHECKS ------------------------
req = ["train.csv","test.csv","sample_submission.csv"]
missing = [f for f in req if not os.path.isfile(os.path.join(DATA_DIR,f))]
if missing:
    raise FileNotFoundError(f"Missing competition files: {missing} under {DATA_DIR}")

train_df = pd.read_csv(os.path.join(DATA_DIR,"train.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR,"test.csv"))
sub_df   = pd.read_csv(os.path.join(DATA_DIR,"sample_submission.csv"))

# ------------------------ UTILITIES ------------------------
def pick(opts, df):
    for c in opts:
        if c in df.columns: return c
    return None

def detect_pred_col(df):
    for c in ["Category:Misconception","prediction","predictions","labels","label"]:
        if c in df.columns: return c
    non_id = [c for c in df.columns if "id" not in c.lower()]
    return non_id[-1] if non_id else df.columns[-1]

def normalize_mis(s):
    return (s.astype(str).fillna("NA").replace({"": "NA","nan":"NA","None":"NA"}).str.strip())

def combine_text(df, q, a, e):
    qv = df[q].astype(str) if q else ""
    av = df[a].astype(str) if a else ""
    ev = df[e].astype(str) if e else ""
    return (("Question: "+qv+"\n") if q else "") + (("Answer: "+av+"\n") if a else "") + (("Explanation: "+ev) if e else "")

def map3_score(true_ids, top3):
    s = 0.0
    for t, preds in zip(true_ids, top3):
        preds = list(preds)
        if t in preds:
            r = preds.index(t) + 1
            if r <= 3: s += 1.0 / r
    return s / len(true_ids)

def find_local_hf_checkpoint(hint=""):
    """
    Return a directory that contains config.json and (pytorch_model.bin OR model.safetensors).
    Searches the hint first (if provided), then all of /kaggle/input recursively,
    preferring paths whose names contain common encoder keywords.
    """
    def is_ckpt(path):
        if not os.path.isdir(path): return False
        files = set(os.listdir(path))
        has_config = "config.json" in files
        has_weights = ("pytorch_model.bin" in files) or ("model.safetensors" in files)
        return has_config and has_weights

    # 1) Direct hint
    if hint:
        if is_ckpt(hint): return hint
        # Sometimes the actual files live under snapshots/<hash>
        for root, dirs, files in os.walk(hint):
            if is_ckpt(root): return root

    # 2) Search entire /kaggle/input
    candidates = []
    for root, dirs, files in os.walk("/kaggle/input"):
        files = set(files)
        if "config.json" in files and ("pytorch_model.bin" in files or "model.safetensors" in files):
            candidates.append(root)

    if not candidates:
        return None

    # Prefer names with these keywords and shorter paths (less nesting)
    prefer = ("deberta", "roberta", "bert", "electra", "xlm")
    def score(p):
        low = p.lower()
        matches = sum(1 for kw in prefer if kw in low)
        return (matches, -len(low))  # more matches, shorter path
    candidates.sort(key=score, reverse=True)
    return candidates[0]

# ------------------------ BUILD TEXT & LABELS ------------------------
Q_tr = pick(['QuestionText','question','question_text','prompt','prompt_text','problem'], train_df)
A_tr = pick(['MC_Answer','answer','student_answer','student_response','response'], train_df)
E_tr = pick(['StudentExplanation','student_explanation','explanation','rationale','student_response'], train_df)
C_tr = pick(['Category','category','label_category'], train_df)
M_tr = pick(['Misconception','misconception','misconception_type','misconception_label'], train_df)

Q_te = pick(['QuestionText','question','question_text','prompt','prompt_text','problem'], test_df) or Q_tr
A_te = pick(['MC_Answer','answer','student_answer','student_response','response'], test_df) or A_tr
E_te = pick(['StudentExplanation','student_explanation','explanation','rationale','student_response'], test_df) or E_tr

if Q_tr is None or C_tr is None:
    raise ValueError("Required columns not found in train.csv (need at least QuestionText & Category).")

train_df["text"] = combine_text(train_df, Q_tr, A_tr, E_tr).str.lower()
test_df["text"]  = combine_text(test_df,  Q_te,   A_te,   E_te).str.lower()

if M_tr is None:
    train_df["Misconception"] = "NA"
else:
    train_df["Misconception"] = normalize_mis(train_df[M_tr])
train_df["Category"] = train_df[C_tr].astype(str).str.strip()
train_df["CombinedLabel"] = train_df["Category"] + ":" + train_df["Misconception"]

# Combined-label mapping (submission space)
unique_combined = sorted(train_df["CombinedLabel"].unique().tolist())
label_to_id = {lab:i for i,lab in enumerate(unique_combined)}
id_to_label = {i:lab for lab,i in label_to_id.items()}

# Heads label spaces
categories = sorted(train_df["Category"].unique().tolist())
cat_to_id  = {c:i for i,c in enumerate(categories)}
misconceptions = sorted(x for x in train_df["Misconception"].unique() if x != "NA")
mis_to_id  = {m:i for i,m in enumerate(misconceptions)}  # NA excluded

# precompute combo mapping -> (cat_idx, mis_idx or -1 for NA)
combo_meta = []
for lab in unique_combined:
    c, m = lab.split(":", 1)
    ci = cat_to_id[c]
    if m == "NA": combo_meta.append((ci, -1))
    else:         combo_meta.append((ci, mis_to_id.get(m, -2)))  # -2 shouldn't occur in train

def combined_probs(cat_probs, mis_probs):
    """cat_probs [B, C], mis_probs [B, M] -> combined [B, len(unique_combined)]"""
    B = cat_probs.shape[0]
    C = np.zeros((B, len(combo_meta)), dtype=np.float32)
    for j,(ci, mi) in enumerate(combo_meta):
        if mi == -1:      # NA
            C[:,j] = cat_probs[:,ci]
        elif mi >= 0:
            C[:,j] = cat_probs[:,ci] * mis_probs[:,mi]
        else:
            C[:,j] = 0.0
    return C

# ------------------------ TRY TO FIND A LOCAL MODEL ------------------------
MODEL_DIR = find_local_hf_checkpoint(MODEL_DIR_HINT)
print("Resolved MODEL_DIR:", MODEL_DIR)

# ------------------------ IF MODEL FOUND: MULTI-TASK 5-FOLD ------------------------
proba_transformer = None
if MODEL_DIR is not None:
    print("Using local HF checkpoint at:", MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)

    def tok(texts, max_len=256):
        enc = tokenizer(list(texts), padding="max_length", truncation=True, max_length=max_len)
        return torch.tensor(enc["input_ids"], dtype=torch.long), torch.tensor(enc["attention_mask"], dtype=torch.long)

    train_ids, train_mask = tok(train_df["text"].values, MAX_LEN)
    test_ids,  test_mask  = tok(test_df["text"].values,  MAX_LEN)
    y_cat = torch.tensor(train_df["Category"].map(cat_to_id).values, dtype=torch.long)

    # misconception labels: -100 for NA -> ignored by loss
    y_mis_list = []
    for m in train_df["Misconception"].values:
        y_mis_list.append(-100 if m=="NA" else mis_to_id[m])
    y_mis = torch.tensor(y_mis_list, dtype=torch.long)

    y_combined = torch.tensor(train_df["CombinedLabel"].map(label_to_id).values, dtype=torch.long)

    class MultiTaskModel(nn.Module):
        def __init__(self, base_dir, num_cat, num_mis):
            super().__init__()
            self.base = AutoModel.from_pretrained(base_dir, local_files_only=True)
            hidden = self.base.config.hidden_size
            self.cat_head = nn.Linear(hidden, num_cat)
            self.mis_head = nn.Linear(hidden, num_mis)
            nn.init.xavier_uniform_(self.cat_head.weight); nn.init.zeros_(self.cat_head.bias)
            nn.init.xavier_uniform_(self.mis_head.weight); nn.init.zeros_(self.mis_head.bias)
            self.ce_cat = nn.CrossEntropyLoss()
            self.ce_mis = nn.CrossEntropyLoss(ignore_index=-100)

        def forward(self, input_ids, attention_mask, labels_cat=None, labels_mis=None):
            out = self.base(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:,0,:]
            lc = self.cat_head(cls)
            lm = self.mis_head(cls)
            if labels_cat is not None and labels_mis is not None:
                loss = CAT_WEIGHT * self.ce_cat(lc, labels_cat) + MIS_WEIGHT * self.ce_mis(lm, labels_mis)
                return loss, lc, lm
            return lc, lm

    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    all_fold_preds = []
    fold_id = 0

    for tr_idx, va_idx in skf.split(train_df, y_combined.numpy()):
        fold_id += 1
        print(f"\n===== Fold {fold_id} / {FOLDS} =====")
        Xtr_ids, Xtr_mask = train_ids[tr_idx], train_mask[tr_idx]
        Xva_ids, Xva_mask = train_ids[va_idx], train_mask[va_idx]
        ytr_cat, ytr_mis  = y_cat[tr_idx], y_mis[tr_idx]
        yva_cat, yva_mis  = y_cat[va_idx], y_mis[va_idx]
        yva_comb          = y_combined[va_idx].numpy()

        model = MultiTaskModel(MODEL_DIR, len(cat_to_id), len(mis_to_id)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        dl_tr = DataLoader(TensorDataset(Xtr_ids, Xtr_mask, ytr_cat, ytr_mis), batch_size=BATCH_SIZE, shuffle=True)
        dl_va = DataLoader(TensorDataset(Xva_ids, Xva_mask, yva_cat, yva_mis), batch_size=BATCH_SIZE, shuffle=False)

        best_map3 = -1.0; best_sd=None; no_improve=0
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        for ep in range(1, EPOCHS+1):
            model.train()
            running=0.0; count=0
            for batch in dl_tr:
                ids, mask, c, m = [b.to(device) for b in batch]
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    loss, lc, lm = model(ids, mask, c, m)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                running += loss.item() * ids.size(0); count += ids.size(0)

            # validate
            model.eval()
            preds_va = []
            with torch.no_grad():
                for batch in dl_va:
                    ids, mask, c, m = [b.to(device) for b in batch]
                    lc, lm = model(ids, mask)
                    cat_p = torch.softmax(lc, dim=1).cpu().numpy()
                    mis_p = torch.softmax(lm, dim=1).cpu().numpy()
                    preds_va.append(combined_probs(cat_p, mis_p))
            v_pred = np.vstack(preds_va)
            top3 = np.argsort(-v_pred, axis=1)[:, :3]
            map3 = map3_score(yva_comb, top3)
            print(f"Epoch {ep}/{EPOCHS} | train loss {running/max(count,1):.4f} | val MAP@3 {map3:.4f}")

            if map3 > best_map3:
                best_map3 = map3
                best_sd = {k:v.cpu() for k,v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print("Early stopping.")
                    break

        if best_sd is not None:
            model.load_state_dict({k:v.to(device) for k,v in best_sd.items()})
        print(f"[Fold {fold_id}] Best val MAP@3 = {best_map3:.4f}")

        # test inference
        dl_te = DataLoader(TensorDataset(test_ids, test_mask), batch_size=BATCH_SIZE, shuffle=False)
        te_preds = []
        model.eval()
        with torch.no_grad():
            for batch in dl_te:
                ids, mask = [b.to(device) for b in batch]
                lc, lm = model(ids, mask)
                cat_p = torch.softmax(lc, dim=1).cpu().numpy()
                mis_p = torch.softmax(lm, dim=1).cpu().numpy()
                te_preds.append(combined_probs(cat_p, mis_p))
        te_pred = np.vstack(te_preds)
        all_fold_preds.append(te_pred)

        del model, optimizer, dl_tr, dl_va, dl_te, best_sd
        torch.cuda.empty_cache(); gc.collect()

    proba_transformer = np.mean(all_fold_preds, axis=0)  # [n_test, n_combined]
else:
    print("[INFO] No local HF checkpoint found under /kaggle/input. Will use baseline TF-IDF + LR.")

# ------------------------ BASELINE ------------------------
# Build text (already done) and combined labels mapping (already done).
# We fit baseline on train_df["text"] and output proba over combined labels.
WORD_NGRAM=(1,3); CHAR_NGRAM=(4,6); WORD_MAXF=12000; CHAR_MAXF=6000
stop_words="english"
word_vec = TfidfVectorizer(stop_words=stop_words, ngram_range=WORD_NGRAM, max_features=WORD_MAXF)
char_vec = TfidfVectorizer(analyzer="char", ngram_range=CHAR_NGRAM, max_features=CHAR_MAXF)

X_tr_text = train_df["text"].values
X_te_text = test_df["text"].values
y_full    = train_df["CombinedLabel"].map(label_to_id).values

word_vec.fit(X_tr_text)
char_vec.fit(X_tr_text)
X_tr = sparse.hstack([word_vec.transform(X_tr_text), char_vec.transform(X_tr_text)])
X_te = sparse.hstack([word_vec.transform(X_te_text),  char_vec.transform(X_te_text)])

lr = LogisticRegression(solver="lbfgs", multi_class="multinomial", max_iter=1000)
lr.fit(X_tr, y_full)
proba_baseline = lr.predict_proba(X_te)  # [n_test, n_combined]

# ------------------------ ENSEMBLE ------------------------
if proba_transformer is not None and proba_transformer.shape == proba_baseline.shape:
    proba = 0.5 * proba_baseline + 0.5 * proba_transformer
    used = "Ensemble (Transformer + TFIDF-LR)"
elif proba_transformer is not None:
    proba = proba_transformer
    used = "Transformer only"
else:
    proba = proba_baseline
    used = "Baseline TFIDF-LR only"

top3_idx = np.argsort(-proba, axis=1)[:, :3]
pred_strings = [" ".join(id_to_label[j] for j in row) for row in top3_idx]

# ------------------------ WRITE SUBMISSION ------------------------
pred_col = detect_pred_col(sub_df)
submission = sub_df.copy()
submission[pred_col] = pred_strings[:len(submission)]
final_csv = os.path.join(OUT_DIR, "submission.csv")
submission.to_csv(final_csv, index=False)
print(f"\n✅ Saved {used} to:", final_csv)
print(submission.head(5))

# ------------------------ FINALIZE ------------------------
target = os.path.join(OUT_DIR, "submission.csv")
if os.path.exists(target):
    print("✅ submission.csv already present:", target)
    print(pd.read_csv(target).head())
else:
    for name in ["submission_roberta.csv","submission_baseline.csv","submission_tfidf.csv","submission_lr.csv"]:
        p = os.path.join(OUT_DIR, name)
        if os.path.exists(p):
            shutil.copy(p, target)
            print(f"✅ Copied {name} -> submission.csv")
            print(pd.read_csv(target).head())
            break
    else:
        g = globals()
        df = None
        for var in ["submission","sub_out","submission_df","sub_df","sub"]:
            if var in g and isinstance(g[var], pd.DataFrame):
                df = g[var].copy()
                print(f"Using in-memory DataFrame: {var}")
                break
        if df is None:
            sample = os.path.join(DATA_DIR, "sample_submission.csv")
            if not os.path.exists(sample):
                raise FileNotFoundError("sample_submission.csv not found in input dir.")
            df = pd.read_csv(sample)
            preds = g.get("pred_strings", None)
            if preds is None:
                raise RuntimeError("No predictions found to finalize submission.")
            pc = detect_pred_col(df)
            df[pc] = preds[:len(df)]
        pc = detect_pred_col(df)
        if df[pc].isna().any() or (df[pc].astype(str).str.strip() == "").any():
            preds = g.get("pred_strings", None)
            if preds is None:
                raise RuntimeError("Predictions column is empty and 'pred_strings' not found.")
            df[pc] = preds[:len(df)]
        df.to_csv(target, index=False)
        print("✅ Wrote submission.csv from in-memory DataFrame.")
        print(df.head())

assert os.path.exists(target), "submission.csv was not created."
print("\nAll set! submission.csv saved at:", target)





