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


import os
import random
import numpy as np
import torch
import numpy as np # linear algebra
import pandas as pd 
try:
    # Hugging Face convenience fn; sets Python/Rand, NumPy, Torch seeds
    from transformers import set_seed  
except ImportError:
    set_seed = None

def seed_everything(seed):
    # 1. Python built‑ins
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)

    # 2. NumPy
    np.random.seed(seed)

    # 3. PyTorch (CPU & GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 4. CuDNN: make deterministic, but may slow you down
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 5. Transformers (if installed)
    if set_seed is not None:
        set_seed(seed)

# call it!
seed_everything(257)


import torch 
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score


train=pd.read_csv('/kaggle/input/quora-insincere-questions-classification/train.csv')
test=pd.read_csv('/kaggle/input/quora-insincere-questions-classification/test.csv')
sample=pd.read_csv('/kaggle/input/quora-insincere-questions-classification/sample_submission.csv')



train


zeros = train[train['target'] == 0]
ones  = train[train['target'] == 1]


zeros_sampled = zeros.sample(n=len(ones), random_state=42)
gol = pd.concat([zeros_sampled, ones]).reset_index(drop=True)


gol


small_train = gol.sample(n=100000).reset_index(drop=True)


train_text=small_train['question_text']
test_text=test['question_text']
target=small_train['target']


train_texts, val_texts, train_targets, val_targets=train_test_split(train_text, target, test_size=0.1, stratify=target)


class QuoraDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        # Force lists to avoid pandas indexing quirks
        self.texts  = list(texts)
        self.labels = None if labels is None else list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item



name='roberta-base'
model=AutoModelForSequenceClassification.from_pretrained(name)


tokenizer=AutoTokenizer.from_pretrained(name)


train_dataset=QuoraDataset(train_texts, train_targets, tokenizer, max_len=384)
val_dataset=QuoraDataset(val_texts, val_targets, tokenizer, max_len=384)
test_dataset=QuoraDataset(test_text, [0]*len(test_text), tokenizer, max_len=384)


def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=1)
    return {'f1': f1_score(eval_pred.label_ids, preds)}


args=TrainingArguments(
    output_dir='gol',
    num_train_epochs=2,
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    eval_strategy='steps',
    eval_steps=500,
    lr_scheduler_type='cosine',
    warmup_ratio=0.1,
    metric_for_best_model='f1',
    load_best_model_at_end=True,
    save_strategy='steps',
    greater_is_better=True,
    report_to='none',
    fp16=True

)


trainer=Trainer(
    args=args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    
)


trainer.train()


SAVE_DIR = "./quora_ckpt"   # will hold config.json, pytorch_model.bin, vocab, etc.
trainer.save_model(SAVE_DIR)              # saves model weights + config
tokenizer.save_pretrained(SAVE_DIR)       # saves tokenizer files

# ---------- 4. (optional) zip it for upload ----------
import shutil, os, zipfile
ZIP_NAME = "quora_ckpt.zip"
shutil.make_archive(base_name=SAVE_DIR, format='zip', root_dir=SAVE_DIR)

print(f"Checkpoint written to {SAVE_DIR}  and zipped as {ZIP_NAME}")


#output = trainer.predict(test_dataset)
#logits = output.predictions
#preds = np.argmax(logits, axis=1).astype(int)

#sample['prediction'] = preds
#sample.to_csv('submission.csv', index=False)








