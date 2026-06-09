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


# --- DATA PREPARATION ---------------------------------------------------
import pandas as pd, numpy as np, re, random, os, torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = "/kaggle/input/map-charting-student-math-misunderstandings"
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")

# Kaggle sample_submission uses `row_id`; keep that name everywhere
if "row_id" not in test.columns:
    test = test.rename(columns={"id": "row_id"})

# Concatenate the relevant text fields exactly once per row
def clean(txt: str) -> str:
    # keep math symbols and LaTeX; just normalise whitespace
    return re.sub(r"\s+", " ", str(txt)).strip()

def make_sentence(row):
    return (
        f"Question: {row.QuestionText} Answer: {row.MC_Answer} "
        f"Student explanation: {row.StudentExplanation}"
    )

train["sentence"] = train.apply(make_sentence, axis=1).map(clean)
test ["sentence"] = test .apply(make_sentence, axis=1).map(clean)

train["Misconception"] = train["Misconception"].fillna("NA")
train["label_str"] = train["Category"] + ":" + train["Misconception"]
le = LabelEncoder().fit(train["label_str"])
train["label"]   = le.transform(train["label_str"])
NUM_LABELS      = len(le.classes_)
print("Num classes:", NUM_LABELS)  # should be 65




from transformers import AutoTokenizer
MODEL_NAME = "/kaggle/input/deberta-v3-base/deberta-v3-base"   # or any other
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

MAX_LEN = 256  # fits v3â€‘base in 16â€¯GB

def tok_fn(batch):
    return tok(
        batch["sentence"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )


train_df, val_df = train_test_split(
    train,
    test_size=0.1,
    stratify=train["Category"],
    random_state=SEED,
)


ds = DatasetDict({
    "train": Dataset.from_pandas(train_df[["sentence", "label"]]),
    "val"  : Dataset.from_pandas(val_df  [["sentence", "label"]]),
    "test" : Dataset.from_pandas(test[["sentence"]])
}).map(tok_fn, batched=True).with_format("torch")

print(ds)



from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score

def metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {"accuracy": accuracy_score(labels, preds)}

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
)



import numpy as np
from sklearn.metrics import accuracy_score

def mapk(preds, targets, k=3):
    """
    Computes the mean average precision at k (MAP@k)
    """
    assert len(preds) == len(targets)
    score = 0.0
    for pred, target in zip(preds, targets):
        if target in pred[:k]:
            score += 1.0 / (pred[:k].index(target) + 1)
    return score / len(preds)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Get top 3 predicted class indices
    top3_preds = np.argsort(logits, axis=1)[:, ::-1][:, :3]

    # Compute MAP@3
    map3 = mapk(top3_preds.tolist(), labels.tolist(), k=3)

    return {
        "eval_map@3": map3,
        "eval_accuracy": accuracy_score(labels, np.argmax(logits, axis=1)),
    }


args = TrainingArguments(
    output_dir="./chk",
    num_train_epochs=10,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,

    # match eval and save strategies
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=200,
    eval_steps=200,
    save_total_limit=3,

    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    fp16=True,
    max_grad_norm=1.0,

    metric_for_best_model="map@3",
    greater_is_better=True,
    load_best_model_at_end=True,

    logging_steps=200,
    report_to="none",
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["val"],
    tokenizer=tok,
    data_collator=DataCollatorWithPadding(tok),
    compute_metrics=compute_metrics,
)


trainer.train()


print("ğŸ”�  Evaluating on validation set â€¦")
eval_results = trainer.evaluate()        # uses eval_dataset you gave in Trainer()

# 2)  Display metrics (handles oldâ€‘vsâ€‘new key names gracefully)
if "eval_accuracy" in eval_results:
    print(f"Validation accuracy: {eval_results['eval_accuracy']:.4f}")
elif "accuracy" in eval_results:
    print(f"Validation accuracy: {eval_results['accuracy']:.4f}")
else:
    print("Eval metrics:", json.dumps(eval_results, indent=2))



trainer.save_model("./best")           # explicit final save
tok.save_pretrained("./best")


# --- PREDICT -------------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained("./best").to("cuda")
model.eval()

probs = []
B = 64
for i in range(0, len(ds["test"]), B):
    batch = ds["test"][i : i+B]
    with torch.no_grad():
        out = model(
            input_ids=batch["input_ids"].to("cuda"),
            attention_mask=batch["attention_mask"].to("cuda"),
        )
    probs.append(torch.softmax(out.logits, dim=-1).cpu().numpy())

probs = np.vstack(probs)     # shape [N,]
top3 = np.argsort(-probs, axis=1)[:, :3]
# decode back to full strings
labels_top3 = le.inverse_transform(top3.flatten()).reshape(top3.shape)
joined      = [" ".join(row) for row in labels_top3]

sub = pd.DataFrame({
    "row_id": test["row_id"],
    "Category:Misconception": joined
})
sub.to_csv("submission.csv", index=False)
print(sub.head())

