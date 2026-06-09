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


TRAIN_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
ORIGIN_DATA_PATH = "/kaggle/input/train2-0/train.csv"
TEST_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
MODEL_PATH = "/kaggle/input/microsoftdeberta-v3-large/transformers/default/1"
OUTPUT_PATH = "/kaggle/working/"

MAX_LEN = 512
OVERLAP = 64
MIN_CHUNK_RATIO = 0.375 #Skip if this chunk < RATIO * MAX_LEN
BATCH_SIZE = 2
TEST_SIZE = 0.2
EPOCH = 3
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42


import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from transformers import TrainerCallback, EarlyStoppingCallback
from transformers import Trainer, TrainingArguments
from transformers import set_seed

from tqdm import tqdm
import random
import matplotlib.pyplot as plt

from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
set_seed(SEED)



df = pd.read_csv(TRAIN_PATH)
origin_df = pd.read_csv(ORIGIN_DATA_PATH)

intersection = pd.merge(df, origin_df, on="full_text", how="inner")[["essay_id", "full_text", "score", "prompt_name"]].reset_index(drop=True)
difference = df[~df["essay_id"].isin(intersection["essay_id"])].reset_index(drop=True)
print("Length of persuade data:", len(intersection))
print("Length of non-persuade data:", len(difference))


df["is_persuade"] = df["essay_id"].isin(intersection["essay_id"]).astype(int)
num_labels = df["score"].nunique()
print("Total samples: ",len(df))
print("Number of labels:", num_labels)
print("Sample: ")
print(df.head(5))


df["text_length"] = df["full_text"].apply(len)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

df["score"].value_counts().sort_index().plot(
    kind="bar", ax=axes[0], color="#007acc"
)
axes[0].set_title("Score Distribution")
axes[0].set_xlabel("Score")
axes[0].set_ylabel("Count")

axes[1].hist(df["text_length"], bins=10, color="#00bfae")
axes[1].set_title("Essay Length Distribution")
axes[1].set_xlabel("Essay Length (chars)")
axes[1].set_ylabel("Count")

axes[2].scatter(df["text_length"], df["score"], color="#ff7043")
axes[2].set_title("Essay Length vs Score")
axes[2].set_xlabel("Essay Length")
axes[2].set_ylabel("Score")

plt.tight_layout()
plt.show()


tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_PATH)


sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(tokens)


import torch
from torch.utils.data import Dataset

class EssayDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512, overlap=128, min_chunk_ratio=0.3):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.overlap = overlap
        self.min_chunk_ratio = min_chunk_ratio
        self.num_special = tokenizer.num_special_tokens_to_add(pair=False)

        for _, row in df.iterrows():
            tag_str = "[A]" if row["is_persuade"] else "[B]"
            tag_tokens = tokenizer(tag_str, add_special_tokens=False)["input_ids"]
            len_tag = len(tag_tokens)

            max_content_len = self.max_len - self.num_special - len_tag

            label = int(row["score"]) - 1
            base_tokens = tokenizer(row["full_text"], add_special_tokens=False)["input_ids"]

            step = max_content_len - overlap
            start = 0

            while start < len(base_tokens):
                end = start + max_content_len
                chunk = base_tokens[start:end]

                if len(chunk) < max_content_len * self.min_chunk_ratio and start != 0:
                    break

                chunk_with_tag = tag_tokens + chunk
                processed = tokenizer.build_inputs_with_special_tokens(chunk_with_tag)
                if len(processed) > self.max_len:
                    processed = processed[:self.max_len]
                
                pad_len = self.max_len - len(processed)
                if pad_len > 0:
                    processed += [tokenizer.pad_token_id] * pad_len

                self.samples.append((processed, label))

                start += step
                if end >= len(base_tokens):
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, label = self.samples[idx]
        ids = torch.tensor(ids, dtype=torch.long)
        # Mask: 1 cho token thật, 0 cho padding
        mask = (ids != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": ids,
            "attention_mask": mask,
            "labels": torch.tensor(label, dtype=torch.long)
        }


train_df, val_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    stratify=df[["score", "is_persuade"]],
    random_state=SEED
)



train_dataset = EssayDataset(train_df, tokenizer, MAX_LEN, OVERLAP, MIN_CHUNK_RATIO)

val_dataset = EssayDataset(val_df, tokenizer, MAX_LEN, OVERLAP, MIN_CHUNK_RATIO)

print("Train size:", len(train_dataset))
print("Validation size:", len(val_dataset))


model = DebertaV2ForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=num_labels)
print(model)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if len(logits.shape) > 1:
        preds = np.argmax(logits, axis=-1)
    else:
        preds = logits

    qwk_score = cohen_kappa_score(labels, preds, weights="quadratic")
    acc_score = accuracy_score(labels, preds)

    return {
        "quadratic_weighted_kappa": qwk_score
    }



class TrainValEvalCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        current_epoch = int(state.epoch) if state.epoch is not None else 0
        total_epochs = args.num_train_epochs

        train_metrics = self.trainer.evaluate(eval_dataset=self.trainer.train_dataset)
        val_metrics   = self.trainer.evaluate(eval_dataset=self.trainer.eval_dataset)

        self.trainer.save_metrics("train", train_metrics)
        self.trainer.save_metrics("eval", val_metrics)

        train_str = " | ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
        val_str   = " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
        tqdm.write(f"Epoch {current_epoch}/{total_epochs} || TRAIN -> {train_str} || EVAL -> {val_str}")

        return control

training_args = TrainingArguments(
    output_dir=f"{OUTPUT_PATH}/models/",
    overwrite_output_dir=True,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=2,

    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    greater_is_better=True,

    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO, 
    lr_scheduler_type="cosine",
    weight_decay=WEIGHT_DECAY,
    num_train_epochs=EPOCH,

    logging_strategy="steps",
    logging_steps=50,
    report_to="none",

    max_grad_norm=1.0, 
    dataloader_num_workers=2,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))
trainer.add_callback(TrainValEvalCallback(trainer))



train_result = trainer.train()


train_pred = trainer.predict(train_dataset)
y_true_train = train_pred.label_ids
y_pred_train = np.argmax(train_pred.predictions, axis=1)

cm_train = confusion_matrix(y_true_train, y_pred_train)
disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train)
disp_train.plot(cmap="Greens", values_format='d')
plt.title("Confusion Matrix - Training")
plt.show()

val_pred = trainer.predict(val_dataset)
y_true_val = val_pred.label_ids
y_pred_val = np.argmax(val_pred.predictions, axis=1)

cm_val = confusion_matrix(y_true_val, y_pred_val)
disp_val = ConfusionMatrixDisplay(confusion_matrix=cm_val)
disp_val.plot(cmap="Blues", values_format='d')
plt.title("Confusion Matrix - Validation")
plt.show()


trainer.save_model(f"{OUTPUT_PATH}/kaggle/working/my_best_model")


tokenizer.save_pretrained(f"{OUTPUT_PATH}/kaggle/working/my_best_model")

