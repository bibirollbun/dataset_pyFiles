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


# =========================== 1. CONFIG =======================================
import os, random, math, gc
import numpy as np, pandas as pd
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
SEED       = 42
MODEL_DIR  = "/kaggle/input/huggingfacedebertav3variants/deberta-v3-base"  # change to -small/-xsmall if OOM
EPOCHS     = 6   # 10 for -small/-xsmall
MAX_LEN    = 256
BATCH_T    = 16  # per‑device train
BATCH_E    = 32  # per‑device eval
LR         = 5e-5

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================== 2. LOAD DATA ====================================
train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
train.Misconception = train.Misconception.fillna("NA")
train["target"] = train.Category + ":" + train.Misconception

# correctness feature (how often MC_Answer is chosen when Category starts with True)
idx_true = train.Category.str.startswith("True")
correct_counts = (
    train[idx_true]
    .groupby(["QuestionId", "MC_Answer"])
    .MC_Answer.agg("count")
    .reset_index(name="c")
    .sort_values("c", ascending=False)
    .drop_duplicates(["QuestionId"])
)
correct_counts["is_correct"] = 1
train = train.merge(correct_counts[["QuestionId", "MC_Answer", "is_correct"]],
                    on=["QuestionId", "MC_Answer"], how="left")
train.is_correct = train.is_correct.fillna(0)

le = LabelEncoder()
train["label"] = le.fit_transform(train["target"])
NUM_CLASSES = len(le.classes_)
print("Unique target classes:", NUM_CLASSES)

# =========================== 3. PROMPT ENGINEERING ===========================

def build_prompt(row):
    correctness = "correct." if row.is_correct else "incorrect."
    return (
        f"Question: {row.QuestionText}\n"
        f"Answer: {row.MC_Answer}\n"
        f"This answer is {correctness}\n"
        f"Student Explanation: {row.StudentExplanation}"
    )

train["text"] = train.apply(build_prompt, axis=1)

# =========================== 4. TOKENIZER & DATASETS =========================
print("Loading tokenizer…")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
collator  = DataCollatorWithPadding(tokenizer)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

# robust stratified split (handles singleton classes)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
train_idx, val_idx = next(skf.split(train, train["label"]))

df_train = train.iloc[train_idx].reset_index(drop=True)
df_val   = train.iloc[val_idx].reset_index(drop=True)

train_ds = Dataset.from_pandas(df_train[["text", "label"]]).map(tokenize, batched=True)
val_ds   = Dataset.from_pandas(df_val[["text", "label"]]).map(tokenize, batched=True)

cols = ["input_ids", "attention_mask", "label"]
train_ds.set_format("torch", columns=cols)
val_ds.set_format("torch", columns=cols)

# =========================== 5. MODEL & TRAINER ==============================
print("Loading model…")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR, num_labels=NUM_CLASSES
).to(device)

# MAP@3 metric
def map3_metric(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    top3 = np.argsort(-probs, axis=1)[:, :3]
    match = (top3 == labels[:, None])
    return {"map@3": (match[:,0] + match[:,1]/2 + match[:,2]/3).mean()}

args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_T,
    per_device_eval_batch_size=BATCH_E,
   
    load_best_model_at_end=True,
    metric_for_best_model="map@3",
    greater_is_better=True,
    seed=SEED,
    report_to="none",
    save_strategy="no",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=map3_metric,
)

trainer.train()
trainer.save_model("./best_model")
import joblib; joblib.dump(le, "./label_encoder.joblib")

# =========================== 6. INFERENCE ON TEST ===========================
print("\nInference on test set…")

test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
# correctness lookup map
dict_corr = correct_counts.set_index(["QuestionId", "MC_Answer"])["is_correct"].to_dict()

test["is_correct"] = test.apply(lambda r: int(dict_corr.get((r.QuestionId, r.MC_Answer), 0)), axis=1)

test["text"] = test.apply(build_prompt, axis=1)

test_ds = Dataset.from_pandas(test[["text"]]).map(lambda b: tokenizer(b["text"], truncation=True, padding="max_length", max_length=MAX_LEN), batched=True)

test_ds.set_format("torch", columns=["input_ids", "attention_mask"])

test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_E)
model.eval()
all_probs = []
for batch in test_loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        logits = model(**batch).logits
    all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
probs = np.vstack(all_probs)

# top‑3 predictions
top3 = np.argsort(-probs, axis=1)[:, :3]
labels_flat = le.inverse_transform(top3.flatten()).reshape(top3.shape)
joined = [" ".join(row) for row in labels_flat]

sub = pd.DataFrame({"row_id": test.row_id, "Category:Misconception": joined})
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv ✅")

