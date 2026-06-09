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


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import wandb
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import json

from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, DebertaV2PreTrainedModel
from transformers import Trainer, TrainingArguments, get_linear_schedule_with_warmup
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch, gc

from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.model_selection import train_test_split
from collections import defaultdict
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold


MODEL_NAME="microsoft/deberta-v3-large"
TRAIN_PATH="/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
TEST_PATH="/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
TOKENIZER="/kaggle/input/deberta-v3-small-tokenizer/pytorch/default/1"
MODEL="/kaggle/input/deberta-v3-small-model/pytorch/default/1"
MODE = "regression"  # "regression" hoáº·c "classification"

FOLDS = [
    "/kaggle/input/train-deberta-v1/pytorch/default/1/checkpoint_fold1_epoch4.pth",
    "/kaggle/input/train-deberta-v1/pytorch/default/1/checkpoint_fold2_epoch3.pth",
    "/kaggle/input/train-deberta-v1/pytorch/default/1/checkpoint_fold3_epoch4.pth",
    "/kaggle/input/ml-test/checkpoint_fold4_epoch4.pth",
    "/kaggle/input/ml-test/checkpoint_fold5_epoch4.pth"
]

MAX_LEN = 1024
OVERLAP = 152
MIN_CHUNK_RATIO = 0.3  # bá»� Ä‘oáº¡n náº¿u < 0.3 * MAX_LEN
BATCH_SIZE = 1
TEST_SIZE = 0.2
EPOCHS = 5
LR = 2e-5
NUM_CLASSES=6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = DebertaV2Tokenizer.from_pretrained(TOKENIZER)


sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(tokens)


class CustomAES2Model_classification(nn.Module):
    def __init__(self, model_name, num_classes=6, model_dir=MODEL, dropout_rate=0.2, freeze_layers=False):
        super().__init__()
        if os.path.exists(model_dir) and any(os.scandir(model_dir)):
            print(f"ğŸ”¹ Loading backbone from local directory: {model_dir}")
            self.backbone = AutoModel.from_pretrained(model_dir)
        else:
            print(f"ğŸ”¸ Downloading backbone from HuggingFace Hub: {model_name}")
            self.backbone = AutoModel.from_pretrained(model_name)
            os.makedirs(model_dir, exist_ok=True)
            self.backbone.save_pretrained(model_dir)

        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)

        # Classification head: tá»« hidden_size â†’ hidden_size/2 â†’ num_classes
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )

        if freeze_layers:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits  # shape [batch_size, num_classes]



df = pd.read_csv(TRAIN_PATH)
num_labels = df["score"].nunique()
print("Total samples: ",len(df))
print("Number of labels:", num_labels)
print("Sample: ")
print(df.head(5))


from torch.utils.data import Dataset
import torch

class EssayDataset(Dataset):
    def __init__(self, df, tokenizer, text_col="full_text",
                 max_length=1024, overlap=152, min_chunk_ratio=0.1):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.overlap = overlap
        self.min_chunk_ratio = min_chunk_ratio
        self.samples = []

        for _, row in df.iterrows():
            essay_id = row["essay_id"]
            text = row[text_col]
            tokens = tokenizer.encode(text, add_special_tokens=False)
            n = len(tokens)
            stride = max_length - overlap - 2  # chá»«a chá»— cho [CLS] vÃ  [SEP]

            # Náº¿u ngáº¯n hÆ¡n max_length â†’ 1 chunk
            if n <= max_length - 2:
                self.samples.append((tokens, essay_id))
            else:
                for start in range(0, n, stride):
                    end = min(start + max_length - 2, n)
                    chunk = tokens[start:end]
                    # Bá»� Ä‘oáº¡n quÃ¡ ngáº¯n
                    if len(chunk) < (max_length - 2) * min_chunk_ratio:
                        continue
                    self.samples.append((chunk, essay_id))
                    if end == n:
                        break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, essay_id = self.samples[idx]
        # Táº¡o input ids vá»›i [CLS], [SEP], vÃ  pad
        input_ids = [self.tokenizer.cls_token_id] + tokens + [self.tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids += [self.tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length

        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "essay_id": essay_id
        }
        return item



from collections import defaultdict
import numpy as np
test_df = pd.read_csv(TEST_PATH)


# Khá»Ÿi táº¡o dataset
dataset = EssayDataset(test_df, tokenizer, max_length=MAX_LEN, overlap=OVERLAP, min_chunk_ratio=MIN_CHUNK_RATIO)

# Sá»­ dá»¥ng dict Ä‘á»ƒ lÆ°u danh sÃ¡ch Ä‘á»™ dÃ i chunks theo essay_id
lengths_per_essay = defaultdict(list)

for tokens, essay_id in dataset.samples:
    chunk_len = len(tokens)  # sá»‘ token khÃ´ng tÃ­nh CLS vÃ  SEP
    lengths_per_essay[essay_id].append(chunk_len)

# PhÃ¢n tÃ­ch thá»‘ng kÃª
for eid, lengths in list(lengths_per_essay.items())[:10]:
    print(f"Essay ID {eid} â†’ {len(lengths)} chunks, lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.2f}")

# Náº¿u muá»‘n thá»‘ng kÃª toÃ n bá»™
all_lengths = [l for lengths in lengths_per_essay.values() for l in lengths]
print(f"Overall chunk lengths: min={min(all_lengths)}, max={max(all_lengths)}, mean={np.mean(all_lengths):.2f}")



test_df = pd.read_csv(TEST_PATH)
test_dataset = EssayDataset(test_df, tokenizer, max_length=MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


len(test_df["full_text"][2])


models = []
for ckpt in FOLDS:
    model = CustomAES2Model_classification(
        model_name=MODEL_NAME,
        model_dir=MODEL,
        num_classes=NUM_CLASSES
    )
    state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state, strict=False)
    model.to(DEVICE)
    model.eval()
    models.append(model)
print(f"âœ… Loaded {len(models)} fold models")



all_chunk_probs = []  # lÆ°u xÃ¡c suáº¥t dá»± Ä‘oÃ¡n má»—i chunk
essay_ids = []        # lÆ°u essay_id tÆ°Æ¡ng á»©ng má»—i chunk

for batch in tqdm(test_loader, desc="Inference"):
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    essay_batch_ids = batch["essay_id"]

    with torch.no_grad():
        fold_probs = []
        for model in models:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs, dim=1)  # xÃ¡c suáº¥t cho má»—i class
            fold_probs.append(probs.cpu().numpy())

        # shape [num_folds, batch_size, num_classes]
        fold_probs = np.stack(fold_probs, axis=0)

        # Trung bÃ¬nh xÃ¡c suáº¥t giá»¯a cÃ¡c model (folds)
        mean_probs = fold_probs.mean(axis=0)  # shape [batch_size, num_classes]

    # lÆ°u láº¡i káº¿t quáº£ cho má»—i chunk
    all_chunk_probs.extend(mean_probs)
    essay_ids.extend(essay_batch_ids)


from collections import Counter

# Táº¡o DataFrame
df_probs = pd.DataFrame({"essay_id": essay_ids, "probs": all_chunk_probs})

# Group theo essay_id â†’ má»—i essay cÃ³ list of probs tá»« cÃ¡c chunk
grouped = df_probs.groupby("essay_id")["probs"].apply(list).reset_index()

# HÃ m tÃ­nh trung bÃ¬nh xÃ¡c suáº¥t giá»¯a cÃ¡c chunk vÃ  láº¥y lá»›p dá»± Ä‘oÃ¡n cuá»‘i cÃ¹ng
def avg_and_round(probs_list):
    arr = np.stack(probs_list, axis=0)      # shape [num_chunks, num_classes]
    mean_arr = arr.mean(axis=0)             # trung bÃ¬nh xÃ¡c suáº¥t giá»¯a cÃ¡c chunk
    pred_class = int(np.argmax(mean_arr))   # lá»›p cÃ³ xÃ¡c suáº¥t cao nháº¥t
    return pred_class

grouped["pred_class"] = grouped["probs"].apply(avg_and_round)

# Náº¿u score = class + 1
grouped["score"] = grouped["pred_class"] + 1

# Merge vá»›i test_df Ä‘á»ƒ táº¡o submission
submission = test_df.merge(grouped[["essay_id", "score"]], on="essay_id", how="left")[["essay_id", "score"]]
# submission.to_csv("submission.csv", index=False)

# print(submission.head())
# print("âœ… Saved submission.csv successfully!")


# LÆ°u file **trong thÆ° má»¥c Kaggle nháº­n diá»‡n**
submission.to_csv("/kaggle/working/submission.csv", index=False)




print(submission.head())



import os
print(os.listdir("/kaggle/working/"))































