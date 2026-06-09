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


import os, gc, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import transformers
import torch
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")



from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


model_name = "microsoft/deberta-v3-base"
max_len = 384
epochs = 3
batch =16
lr = 2e-5
n_folds = 5
seed = 42


train = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
test = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")


y = train["rule_violation"].astype(int).values


train.head()


train.info()


train.isna().sum()


train.isnull().sum()


train.describe()


train.shape


test.head()


test.info()


test.isna().sum()


test.isnull().sum()


test.describe()


test.shape


def make_pair(row):
    comment = str(row["body"])
    rulepart = " | ".join([
        f"rule:{row.get('rule','')}",
        f"pos1:{row.get('positive_example_1','')}",
        f"pos2:{row.get('positive_example_2','')}",
        f"neg1:{row.get('negative_example_1','')}",
        f"neg2:{row.get('negative_example_2','')}",
        f"subreddit:r/{row.get('subreddit','')}"
    ])
    return comment, rulepart


train_pairs = [make_pair(r) for _,r in train.iterrows()]
test_pairs = [make_pair(r) for _,r in test.iterrows()]


# Load tokenizer from your local dataset
tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/roberta-base-local1/roberta_base_local/fold2", use_fast=True)


class RuleDataset(Dataset):
    def __init__(self,pairs,labels = None):
        self.pairs = pairs
        self.labels = labels
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        a,b = self.pairs[idx]
        enc = tokenizer(
            a,b,max_length = max_len,
            truncation = True,
            padding = "max_length",
            return_tensors = "pt"
        )
        item = {k:v.squeeze(0) for k,v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    logits,labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    return {"roc_auc": roc_auc_score(labels,probs)}


skf = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=seed)


oof = np.zeros(len(train),dtype=float)
pred = np.zeros(len(test),dtype=float)



TRAIN = False


for fold, (tr, va) in enumerate(skf.split(train_pairs, y), 1):
    print(f"Processing fold {fold}...")
    tr_ds = RuleDataset([train_pairs[i] for i in tr], y[tr])
    va_ds = RuleDataset([train_pairs[i] for i in va], y[va])
    te_ds = RuleDataset(test_pairs, None)

    fold_path = f"/kaggle/input/roberta-base-local1/roberta_base_local/fold{fold}"  # load from dataset version
    if TRAIN:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, problem_type="single_label_classification"
        )
        args = TrainingArguments(
            output_dir=f"./fold{fold}",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=batch,
            per_device_eval_batch_size=batch,
            num_train_epochs=epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="roc_auc",
            save_total_limit=1,
            fp16=torch.cuda.is_available(),
            report_to="none",
            seed=seed
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tr_ds,
            eval_dataset=va_ds,
            compute_metrics=compute_metrics
        )
        trainer.train()
        model.save_pretrained(f"./roberta_base_local/fold{fold}")
        tokenizer.save_pretrained(f"./roberta_base_local/fold{fold}")
    else:
        print(f"Loading model for fold {fold} from {fold_path}")
        model = AutoModelForSequenceClassification.from_pretrained(fold_path)
        tokenizer = AutoTokenizer.from_pretrained(fold_path)

        trainer = Trainer(
            model=model,
            args=TrainingArguments(output_dir="./dummy", report_to="none"),
        )

    # VALID
    va_logits = trainer.predict(va_ds).predictions
    va_probs = torch.softmax(torch.tensor(va_logits), dim=-1).numpy()[:, 1]
    oof[va] = va_probs
    print(f"Fold{fold} AUC:{roc_auc_score(y[va], va_probs):.4f}")

    # TEST
    te_logits = trainer.predict(te_ds).predictions
    te_probs = torch.softmax(torch.tensor(te_logits), dim=-1).numpy()[:, 1]
    pred += te_probs / n_folds

    del model
    gc.collect()
    torch.cuda.empty_cache()



print("OOF AUC:", roc_auc_score(y, oof))


sub = pd.DataFrame({
    "row_id": test["row_id"] if "row_id" in test.columns else sample_sub["row_id"],
    "rule_violation": np.clip(pred, 0, 1)
})
sub.to_csv("/kaggle/working/submission.csv", index=False)
print(sub.head())


#!zip -r roberta_base_local.zip roberta_base_local

