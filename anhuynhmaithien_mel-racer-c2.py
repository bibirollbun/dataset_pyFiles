import os, re, numpy as np, pandas as pd, torch
from math import log2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

SEED = 42 
np.random.seed(SEED)
torch.manual_seed(SEED)

TRAIN_PATH = "/kaggle/input/rmit-hackathon-2025/train.csv"
TEST_PATH  = "/kaggle/input/rmit-hackathon-2025/test.csv"

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
print("Train shape:", train.shape, "| Test shape:", test.shape)
assert {'Id','text','label'}.issubset(train.columns)
assert {'Id','text'}.issubset(test.columns)


def clean_text(t: str) -> str:
    t = str(t).lower()
    t = re.sub(r"http\S+|www\S+", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

train["clean_text"] = train["text"].apply(clean_text)
test["clean_text"]  = test["text"].apply(clean_text)


def shannon_entropy(s):
    if not s: return 0
    from collections import Counter
    p = np.array(list(Counter(s).values()))/len(s)
    return -(p*np.log2(p)).sum()


JB_TYPES = {
    "ignore_instruction": ["ignore", "disregard", "override previous"],
    "roleplay": ["as dan", "pretend to be", "roleplay"],
    "policy_bypass": ["bypass", "policy", "unfiltered", "uncensored"],
    "ethical_override": ["ethics", "morality", "for research"],
    "sudo_prompt": ["sudo", "command mode", "root access"],
}


def detect_jb_types(text):
    t = str(text).lower()
    return pd.Series({k: int(any(kw in t for kw in kws)) for k, kws in JB_TYPES.items()})

for df in [train, test]:
    df["len_chars"] = df["clean_text"].str.len()
    df["len_words"] = df["clean_text"].str.split().str.len()
    df["num_punct"] = df["text"].str.count(r"[^\w\s]")
    df["num_upper_ratio"] = df["text"].apply(lambda s: sum(c.isupper() for c in s)/max(1,len(s)))
    df["num_digit_ratio"] = df["text"].apply(lambda s: sum(c.isdigit() for c in s)/max(1,len(s)))
    df["entropy"] = df["text"].apply(shannon_entropy)
    df[list(JB_TYPES.keys())] = df["text"].apply(detect_jb_types)

label_map = {"benign": 0, "jailbreak": 1}
train["target"] = train["label"].map(label_map)


X_tr, X_val, y_tr, y_val = train_test_split(
    train["clean_text"], train["target"],
    test_size=0.10, stratify=train["target"], random_state=SEED
)
print("Train:", len(X_tr), "| Val:", len(X_val))


from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from scipy.special import softmax

MODEL_PATH = "/kaggle/input/deberta/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH, local_files_only=True, num_labels=2
)


if hasattr(model, "classifier") and hasattr(model.classifier, "dropout"):
    model.classifier.dropout.p = 0.25

train_df = pd.DataFrame({"text": X_tr, "label": y_tr})
val_df   = pd.DataFrame({"text": X_val, "label": y_val})
train_ds = Dataset.from_pandas(train_df)
val_ds   = Dataset.from_pandas(val_df)


def tok_fn(batch):
    enc = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=384)
    enc["labels"] = batch["label"]
    return enc

train_tok = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
val_tok   = val_ds.map(tok_fn, batched=True, remove_columns=["text"])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = softmax(logits, axis=1)[:,1]
    return {"roc_auc": roc_auc_score(labels, probs)}


args = TrainingArguments(
    output_dir="./deberta_base_runs",
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="roc_auc",
    greater_is_better=True,
    num_train_epochs=4,
    learning_rate=1.5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    warmup_ratio=0.1,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    logging_strategy="steps",
    logging_steps=50,
    report_to="none",
    fp16=torch.cuda.is_available(),
    save_total_limit=1,
)


trainer = Trainer(
    model=model, args=args,
    train_dataset=train_tok, eval_dataset=val_tok,
    tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)
print("Fine-tuning DeBERTa...")


_ = trainer.train()
eval_out = trainer.evaluate()
print("DeBERTa eval:", eval_out)

val_prob_deb = softmax(trainer.predict(val_tok).predictions, axis=1)[:,1]
print(f"Validation AUC: {roc_auc_score(y_val, val_prob_deb):.4f}")

np.save(f"deberta_val_seed{SEED}.npy", val_prob_deb)

test_ds = Dataset.from_pandas(test[["text"]])
test_tok = test_ds.map(lambda b: tokenizer(b["text"], truncation=True, padding="max_length", max_length=384),
                       batched=True, remove_columns=["text"])
test_prob_deb = softmax(trainer.predict(test_tok).predictions, axis=1)[:,1]


np.save(f"deberta_test_seed{SEED}.npy", test_prob_deb)

# Submission
submission = pd.DataFrame({
    'Id': test['Id'],
    'label': test_prob_deb
})
submission.to_csv('submission.csv', index=False)
print(f"\n Submission saved! Validation AUC: {roc_auc_score(y_val, val_prob_deb):.4f}")

