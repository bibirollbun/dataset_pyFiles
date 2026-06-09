# !pip uninstall -y transformers accelerate peft protobuf torch torchvision torchaudio

# !pip install -q protobuf==3.20.*
# !pip install -q transformers==4.36.2
# !pip install -q accelerate==0.25.0
# !pip install -q peft==0.6.2

# !pip install -q torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
#     --index-url https://download.pytorch.org/whl/cu118


import torch, transformers, accelerate, peft

print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Accelerate:", accelerate.__version__)
print("PEFT:", peft.__version__)
     


import os
import random
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import roc_auc_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# Disable tokenizer parallel annoyances
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CFG:
    model_name = "microsoft/deberta-v3-base"   # DEBERTA v3 BASE
    max_len = 256                              # you can try 320 later
    train_batch_size = 16
    eval_batch_size = 16
    epochs = 2                                  # try 3 if you want even higher
    lr = 2e-5
    seed = 42
    target_cols = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CFG.seed)

print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")



train_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip"
test_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip"
sample_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip"

train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

train_df["comment_text"] = train_df["comment_text"].fillna("")
test_df["comment_text"]  = test_df["comment_text"].fillna("")

# Ensure labels are floats
train_df[CFG.target_cols] = train_df[CFG.target_cols].astype("float32")

print(train_df.shape, test_df.shape)
train_df.head()



from sklearn.model_selection import train_test_split

train_df_split, valid_df_split = train_test_split(
    train_df,
    test_size=0.1,
    random_state=CFG.seed
)

# Add "labels" column as list of 6 floats
train_df_split = train_df_split.copy()
valid_df_split = valid_df_split.copy()

train_df_split["labels"] = train_df_split[CFG.target_cols].values.tolist()
valid_df_split["labels"] = valid_df_split[CFG.target_cols].values.tolist()

# Keep only text + labels for HF Datasets
train_hf = Dataset.from_pandas(train_df_split[["comment_text", "labels"]])
valid_hf = Dataset.from_pandas(valid_df_split[["comment_text", "labels"]])

train_hf, valid_hf



tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

def tokenize_batch(batch):
    return tokenizer(
        batch["comment_text"],
        padding="max_length",
        truncation=True,
        max_length=CFG.max_len,
    )

train_hf = train_hf.map(tokenize_batch, batched=True)
valid_hf = valid_hf.map(tokenize_batch, batched=True)

# Remove raw text, keep tokenized inputs + labels
train_hf = train_hf.remove_columns(["comment_text"])
valid_hf = valid_hf.remove_columns(["comment_text"])

# Set format for PyTorch
train_hf.set_format(type="torch")
valid_hf.set_format(type="torch")

train_hf[0]



model = AutoModelForSequenceClassification.from_pretrained(
    CFG.model_name,
    num_labels=len(CFG.target_cols),
    problem_type="multi_label_classification",   # IMPORTANT
)

model.to("cuda" if torch.cuda.is_available() else "cpu")



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # logits: (batch, 6), labels: (batch, 6)
    probs = 1 / (1 + np.exp(-logits))  # sigmoid

    roc_list = []
    for i in range(labels.shape[1]):
        try:
            roc = roc_auc_score(labels[:, i], probs[:, i])
        except ValueError:
            roc = np.nan
        roc_list.append(roc)

    mean_roc = np.nanmean(roc_list)
    return {"mean_roc_auc": mean_roc}



data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="./deberta-v3-base-jigsaw",
    num_train_epochs=CFG.epochs,
    per_device_train_batch_size=CFG.train_batch_size,
    per_device_eval_batch_size=CFG.eval_batch_size,
    learning_rate=CFG.lr,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="mean_roc_auc",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=2,
    report_to=[],  # no wandb
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_hf,
    eval_dataset=valid_hf,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)



trainer.train()



metrics = trainer.evaluate()
metrics



test_hf = Dataset.from_pandas(test_df[["comment_text"]])

test_hf = test_hf.map(tokenize_batch, batched=True)
test_hf = test_hf.remove_columns(["comment_text"])
test_hf.set_format(type="torch")

test_hf[0]



preds = trainer.predict(test_hf)
logits = preds.predictions
probs = 1 / (1 + np.exp(-logits))  # sigmoid → probabilities
probs.shape



submission = sample_sub.copy()
submission[CFG.target_cols] = probs
submission.to_csv("submission.csv", index=False)
submission.head()


