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


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
# Load model directly
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datasets import Dataset

import torch
# import torch.nn as nn

from transformers import AutoModelForSequenceClassification, AutoModel, AutoTokenizer, TrainingArguments, Trainer
# from torch.utils.data import Dataset, DataLoader
from torchvision.ops import sigmoid_focal_loss


import numpy as np
import random
import os

def set_seed(seed=42):
    """Set all random seeds for reproducibility across PyTorch, numpy, and Python."""
    # PyTorch seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Numpy and Python random seeds
    np.random.seed(seed)
    random.seed(seed)

    # Environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For CUDA >= 10.2

set_seed(42)
VER = 1
DIR = f"ver_{VER}"
os.makedirs(DIR, exist_ok=True)


df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
df.head(2)


# Group the data
df['pair_id'] = df['QuestionText'] + '||' + df['MC_Answer']

# Split on QA_id so no leakage
from sklearn.model_selection import GroupShuffleSplit
splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, val_idx = next(splitter.split(df, groups=df['pair_id']))
train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]


train_df = train_df.drop_duplicates(subset=["pair_id"], keep='first')
val_df = val_df.drop_duplicates(subset=["pair_id"], keep='first')
print(train_df.shape, val_df.shape)


import re

def mask_question(text):
    # Lowercase for normalization
    text = text.lower()
    
    # Replace numbers with a single placeholder
    text = re.sub(r'\d+(\.\d+)?', '<NUM>', text)
    
    # Replace variables (single letters surrounded by spaces or punctuation)
    text = re.sub(r'\b[a-z]\b', '<VAR>', text)
    
    # Collapse multiple placeholders into one
    text = re.sub(r'(<VAR>\s*){2,}', '<VAR> ', text)
    
    # Strip extra spaces
    return re.sub(r'\s+', ' ', text).strip()

train_df['MaskedQuestion'] = train_df['QuestionText'].apply(mask_question)
val_df['MaskedQuestion'] = val_df['QuestionText'].apply(mask_question)



print(train_df.iloc[0].values)


import random

def augment_masked_question(q):
    # Replace placeholder tokens with random values
    num = str(random.randint(1, 99))
    var = random.choice(["x", "y", "n", "m"])
    q = q.replace("<NUM>", num).replace("<VAR>", var)
    return q

augmented = []
for _, row in train_df.iterrows():
    for _ in range(5):  # 5 variants
        aug_q = augment_masked_question(row['MaskedQuestion'])
        augmented.append({
            'MaskedQuestion': aug_q,
            'MC_Answer': row['MC_Answer'],
            'StudentExplanation': row['StudentExplanation']
        })
aug_train_df = pd.DataFrame(augmented)



print(aug_train_df.head(2))


train_df["Correct"] = (train_df.apply(lambda row: row.Category.split('_')[0],axis=1)=='True').astype(int)
train_df["Misconception"] = train_df["Misconception"].fillna("NA")


classifier_base_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(classifier_base_name, use_fast=True)


# Combine features
def format_input(row):
    return (
        "Act as a Math Teacher and decide whether the answer is correct or not.\n"
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
    )
train_df["classifier_text"] = train_df.apply(format_input, axis=1)


MAX_LEN = 256
lengths = [len(tokenizer.encode(t, truncation=False)) for t in train_df.classifier_text]
import matplotlib.pyplot as plt
L = (np.array(lengths)>MAX_LEN).sum()
print(f"There are {L} train sample(s) with more than {MAX_LEN} tokens")
print(np.sort( lengths ))

plt.hist(lengths, bins=50)
plt.title("Token Length Distribution")
plt.xlabel("Number of tokens")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


# Split into train and validation sets
train_df_1, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Convert to Hugging Face Dataset
COLS = ['classifier_text','Correct']
train_ds = Dataset.from_pandas(train_df_1[COLS])
val_ds = Dataset.from_pandas(train_df[COLS])


# Tokenization function
def tokenize(batch, prompt_col="classifier_text", max_len=256):
    return tokenizer(batch[prompt_col], padding="max_length", truncation=True, max_length=max_len)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)
train_ds = train_ds.rename_column("Correct", "labels")
val_ds = val_ds.rename_column("Correct", "labels")


# Set format for PyTorch
columns = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
train_ds.set_format(type='torch', columns=columns)
val_ds.set_format(type='torch', columns=columns)


n_classification_classes = 2
classifier = AutoModelForSequenceClassification.from_pretrained(
            classifier_base_name,
            num_labels=n_classification_classes)


from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"precision": precision_score(labels, preds, average="binary")}


from transformers import EarlyStoppingCallback
from transformers.trainer_callback import TrainerControl, TrainerState
# 1. Custom callback that stops when precision hits or exceeds a threshold
class PrecisionThresholdCallback(EarlyStoppingCallback):
    def __init__(self, threshold: float = 0.95):
        super().__init__(early_stopping_patience=9999)  # never stop via patience
        self.threshold = threshold

    def on_evaluate(self, args: TrainingArguments,
                    state: TrainerState,
                    control: TrainerControl, **kwargs):
        # state.log_history[-1] contains the last evaluation results
        logs = state.log_history[-1] if state.log_history else {}
        precision = logs.get("eval_precision")
        if precision is not None and precision >= self.threshold:
            control.should_training_stop = True   # stop right now


class_args = TrainingArguments(
        output_dir = f"./{DIR}",
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        save_strategy="steps", #no for no saving 
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        logging_dir="./logs",
        logging_steps=300,
        save_steps=300,
        eval_steps=300,
        save_total_limit=1,
        metric_for_best_model="precision",
        greater_is_better=True,
        load_best_model_at_end=True,
        label_names=["labels"],
        report_to="none",
        bf16=False, # TRAIN WITH BF16 IF LOCAL GPU IS NEWER GPU          
        fp16=True, # INFER WITH FP16 BECAUSE KAGGLE IS T4 GPU
)
trainer = Trainer(
        model=classifier,
        args=class_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[PrecisionThresholdCallback(threshold=0.97)],
)


trainer.train()


trainer.save_model(f"ver_{VER}")      
tokenizer.save_pretrained(f"ver_{VER}")




