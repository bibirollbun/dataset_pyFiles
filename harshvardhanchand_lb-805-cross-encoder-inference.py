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


#Training 
https://www.kaggle.com/code/harshvardhanchand/lb-805-cross-encoder-train




import os, json, numpy as np, pandas as pd, torch
from tqdm.auto import tqdm
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer

# Paths (EDIT the two lines below to your datasets)
COMP_DATA      = "/kaggle/input/jigsaw-agile-community-rules"           
MODEL_DATA     = "/kaggle/input/cross-encoder-folds-flattened/ce_deberta_v3_20250825_162658"     

TEST_PATH      = os.path.join(COMP_DATA, "test.csv")            
SUB_PATH       = "/kaggle/working/submission.csv"

# Inference options
MAX_LENGTH            = 512
TOP_K                 = 5       # pick best-3 folds by val AUC
PRED_BATCH_SIZE_TEST  = 32       # for 3 models loaded at once; reduce to 16 if OOM
USE_FAST_TOKENIZER    = False    # avoid byte-fallback warning, keep parity with training

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "disabled"  # silence W&B



# ---------- Pick top-3 folds ----------
with open(os.path.join(MODEL_DATA, "fold_scores.json")) as f:
    fs = json.load(f)["fold_scores"]   # list of [fold_id, auc]
top = sorted(fs, key=lambda kv: kv[1], reverse=True)[:TOP_K]
top_paths = [os.path.join(MODEL_DATA, f"model_fold{fold}") for (fold, _) in top]
print("Using top folds:", top)

# ---------- Build prompt (same as training; no dropout) ----------
import re
URL_TOKEN, EMAIL_TOKEN, PHONE_TOKEN = "<URL>", "<EMAIL>", "<PHONE>"
_URL_RE   = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"(?:(?:\+?\d{1,3}[ -]?)?(?:\(?\d{2,4}\)?[ -]?)?\d{3,4}[ -]?\d{4})")
def _collapse_ws(s): 
    s = s.replace("\r\n","\n"); s = re.sub(r"[ \t]+"," ",s); s = re.sub(r"\n{3,}","\n\n",s); return s.strip()
def clean_post(text):
    s = text or ""
    s = _URL_RE.sub(URL_TOKEN, s)
    s = _EMAIL_RE.sub(EMAIL_TOKEN, s)
    s = _PHONE_RE.sub(PHONE_TOKEN, s)
    return _collapse_ws(s)
def clean_rule(text): return _collapse_ws(text or "")
def bullets(title, items):
    items = [i for i in items if isinstance(i,str) and i.strip()]
    return (title + "\n" + "\n".join(f"- {i}" for i in items) + "\n") if items else ""
def build_prompt(row):
    rule_text = clean_rule(row["rule"])
    c = lambda x: clean_post(x)
    pos1, pos2 = c(row.get("positive_example_1","")), c(row.get("positive_example_2",""))
    neg1, neg2 = c(row.get("negative_example_1","")), c(row.get("negative_example_2",""))
    body       = c(row["body"])
    subreddit  = (row.get("subreddit","") or "").strip()
    ctx = f"Rule:\n{rule_text}\n\n{bullets('Positive examples:', [pos1, pos2])}{bullets('Negative examples:', [neg1, neg2])}"
    sub = f"[subreddit: {subreddit}]\n" if subreddit else ""
    post = f"[QUERY POST]\n{sub}{body}"
    return ctx + "\n" + post + "\n\nTask: Does the post violate the rule? Only respond Yes/No"

# ---------- Read test & ensure columns ----------
test = pd.read_csv(TEST_PATH)
for col in ["positive_example_1","positive_example_2","negative_example_1","negative_example_2","subreddit"]:
    if col not in test.columns: test[col] = ""

# Build prompts once (keep pair format, second string empty to match training)
test_prompts = [[build_prompt({
    "rule": r["rule"],
    "positive_example_1": r.get("positive_example_1",""),
    "positive_example_2": r.get("positive_example_2",""),
    "negative_example_1": r.get("negative_example_1",""),
    "negative_example_2": r.get("negative_example_2",""),
    "subreddit": r.get("subreddit",""),
    "body": r["body"],
}), ""] for _, r in test.iterrows()]

# ---------- Load models (all 3 at once) ----------
device_count = torch.cuda.device_count()
devices = [f"cuda:{i}" for i in range(device_count)] if device_count>0 else ["cpu"]
models = []
for i, p in enumerate(top_paths):
    print(f"Attempting to load model from path: '{p}'") # <-- DEBUG LINE
    dev = devices[min(i, len(devices)-1)]
    m = CrossEncoder(
        p,
        num_labels=1,
        max_length=512,
        device=dev,
        tokenizer_args={"use_fast": False}
    )
    
    
    try:
        m.model.half()
    except Exception:
        pass
    models.append(m)

# ---------- Predict (ensemble average) ----------
def batched_predict_probs_multi(models, prompts, bs=PRED_BATCH_SIZE_TEST, desc="Scoring (top-3 ensemble)"):
    out = np.zeros(len(prompts), dtype=np.float32)
    for i in tqdm(range(0, len(prompts), bs), desc=desc):
        sl = slice(i, i+bs); batch = prompts[sl]
        probs_stack = []
        for m in models:
            logits = m.predict(batch)
            probs  = torch.sigmoid(torch.tensor(logits)).numpy()
            probs_stack.append(probs)
        out[sl] = np.mean(np.vstack(probs_stack), axis=0)
    return out

preds = batched_predict_probs_multi(models, test_prompts, bs=PRED_BATCH_SIZE_TEST)
test["rule_violation"] = preds
sub_cols = ["row_id","rule_violation"] if "row_id" in test.columns else ["rule_violation"]
test[sub_cols].to_csv(SUB_PATH, index=False)
print("Saved:", SUB_PATH)


