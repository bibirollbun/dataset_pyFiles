import os, re, random
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

TRAIN_CSV = "/kaggle/input/jigsaw-agile-community-rules/train.csv"
TEST_CSV = "/kaggle/input/jigsaw-agile-community-rules/test.csv"

BASE_MODEL = "/kaggle/input/bert_base_uncased/transformers/default/6/saved_deberta"

BATCH_SIZE = 4
GRAD_ACC = 2
BASE_LR = 3e-5
WARMUP_RATIO = 0.1
STAGE2_EPOCHS = 100
OUTPUT_DIR = "./bert-rule-violation-ft"
os.makedirs(OUTPUT_DIR, exist_ok=True)

os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def random_deletion(text, p=0.1):
    words = text.split()
    if len(words) == 1:
        return text
    remaining = [word for word in words if random.random() > p]
    if len(remaining) == 0:
        if len(words) > 0:
            return random.choice(words)
        else:
            return ""
    return " ".join(remaining)

def random_swap(text, n=1):
    words = text.split()
    length = len(words)
    for _ in range(n):
        if length < 2:
            break
        idx1, idx2 = random.sample(range(length), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return " ".join(words)

def augment_row(row, p_aug=0.5):
    if random.random() < p_aug:
        aug_body = random_deletion(row['body'], p=0.1)
        aug_body = random_swap(aug_body, n=1)
        row['body'] = aug_body
    return row


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def format_prompt(row):
    return (
        f"Does the following comment violate the rule?\n\n"
        f"Rule: {row['rule']}\n"
        f"Comment: {row['body']}\n\n"
        f"Here are some examples for guidance:\n"
        f"Violates: {row['positive_example_1']} | {row['positive_example_2']}\n"
        f"Does not violate: {row['negative_example_1']} | {row['negative_example_2']}"
    )


df = pd.read_csv(TRAIN_CSV)
for col in ["rule", "body", "positive_example_1", "positive_example_2",
            "negative_example_1", "negative_example_2"]:
    df[col] = df[col].apply(clean_text)

df["label"] = df["rule_violation"].astype(float)

train_df, val_df = train_test_split(
    df,
    test_size=0.10,
    stratify=df["label"],
    random_state=SEED
)

print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")


train_df = train_df.apply(augment_row, axis=1)

train_df["prompt"] = train_df.apply(format_prompt, axis=1)
val_df["prompt"] = val_df.apply(format_prompt, axis=1)

train_df['comment_len'] = train_df['body'].apply(lambda x: len(x.split()))


LENGTH_THRESHOLD = 100
easy_df = train_df[train_df['comment_len'] <= LENGTH_THRESHOLD].copy()
hard_df = train_df[train_df['comment_len'] > LENGTH_THRESHOLD].copy()

print(f"Easy examples: {len(easy_df)}, Hard examples: {len(hard_df)}")


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

all_prompts = pd.concat([train_df["prompt"], val_df["prompt"]])
lengths = [len(tokenizer.encode(p, truncation=False)) for p in tqdm(all_prompts, desc="Tokenizing for length")]
p90 = int(np.percentile(lengths, 90))
MAX_LEN = min(512, max(128, p90))
print(f"90th percentile token length: {p90}, Using max_length = {MAX_LEN}")

def tokenize_function(examples):
    return tokenizer(
        examples["prompt"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN
    )

def prepare_dataset(df):
    ds = Dataset.from_pandas(df.reset_index(drop=True))
    ds = ds.map(tokenize_function, batched=True)

    cols_to_remove = [
        "prompt", "rule", "body",
        "positive_example_1", "positive_example_2",
        "negative_example_1", "negative_example_2",
        "rule_violation", "comment_len"
    ]
    existing_remove = [c for c in cols_to_remove if c in ds.column_names]
    ds = ds.remove_columns(existing_remove)

    if "label" in ds.column_names:
        ds = ds.rename_column("label", "labels")

    ds.set_format("torch")
    return ds

easy_ds = prepare_dataset(easy_df)
hard_ds = prepare_dataset(train_df)
val_ds = prepare_dataset(val_df)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy().flatten()
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "roc_auc": roc_auc_score(labels, probs)
    }

def freeze_layers(model):
    for name, param in model.named_parameters():
        if ("encoder.layer.11" not in name) and ("classifier" not in name):
            param.requires_grad = False

def train_one_model(seed, output_subdir):
    print(f"Training model with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)
    freeze_layers(model)

    training_args = TrainingArguments(
        output_dir=output_subdir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=BASE_LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC,
        num_train_epochs=100,
        weight_decay=0.01,
        logging_dir=os.path.join(output_subdir, 'logs'),
        logging_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        seed=seed,
        lr_scheduler_type="reduce_lr_on_plateau",
        warmup_ratio=WARMUP_RATIO
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=easy_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("Stage 1: Training on easy examples (frozen layers)")
    trainer.train()

    for param in model.parameters():
        param.requires_grad = True

    trainer.train_dataset = hard_ds

    print("Stage 2: Training on full data (all layers unfrozen)")
    trainer.train()

    trainer.save_model(output_subdir)
    return trainer


seeds = [SEED, SEED+1, SEED+2]
trainers = []
for s in seeds:
    outdir = os.path.join(OUTPUT_DIR, f"model_seed_{s}")
    trainer = train_one_model(s, outdir)
    trainers.append(trainer)

test_df = pd.read_csv(TEST_CSV)
for col in ["rule", "body", "positive_example_1", "positive_example_2",
            "negative_example_1", "negative_example_2"]:
    test_df[col] = test_df[col].apply(clean_text)
test_df["prompt"] = test_df.apply(format_prompt, axis=1)


test_ds = Dataset.from_pandas(test_df[["prompt"]]).map(tokenize_function, batched=True).remove_columns(["prompt"]).with_format("torch")

def tta_predict(trainers, dataset, rounds=3):
    all_probs = []
    for trainer in trainers:
        model_probs = []
        for _ in range(rounds):
            preds = trainer.predict(dataset)
            logits = preds.predictions
            probs = torch.sigmoid(torch.tensor(logits)).numpy().flatten()
            model_probs.append(probs)
        avg_model_probs = np.mean(model_probs, axis=0)
        all_probs.append(avg_model_probs)
    ensemble_probs = np.mean(all_probs, axis=0)
    return ensemble_probs


final_probs = tta_predict(trainers, test_ds, rounds=3)

submission = pd.DataFrame({"row_id": test_df["row_id"], "rule_violation": final_probs.astype(np.float32)})
submission.to_csv("submission.csv", index=False)

