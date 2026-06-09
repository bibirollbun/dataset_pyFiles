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
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split 
import re


train=pd.read_csv('/kaggle/input/commonlitreadabilityprize/train.csv')
test=pd.read_csv('/kaggle/input/commonlitreadabilityprize/test.csv')
sample=pd.read_csv('/kaggle/input/commonlitreadabilityprize/sample_submission.csv')


train


test


sample


import re, html, unicodedata

RE_URL   = re.compile(r'https?://\S+|www\.\S+')
RE_EMAIL = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')
RE_PHONE = re.compile(r'\+?\d[\d\-\s()]{7,}\d')

# Юникодные кавычки/тире/многоточия → ASCII
PUNCT_MAP = {
    '\u2018':'\'', '\u2019':'\'', '\u201C':'"', '\u201D':'"',
    '\u2013':'-', '\u2014':'-', '\u2212':'-', '\u2026':'...',
}

def _normalize_punct(s: str) -> str:
    return s.translate(str.maketrans(PUNCT_MAP))

def clean_for_readability(text: str, keep_case: bool = True) -> str:
    """Бережная чистка: сохраняем пунктуацию/цифры/дефисы и границы предложений."""
    if not isinstance(text, str):
        return ""
    # Юникод + HTML
    s = unicodedata.normalize('NFKC', text)
    s = html.unescape(s)
    s = _normalize_punct(s)

    # Плейсхолдеры сущностей (оставляем их как маркеры сложности)
    s = RE_URL.sub(' <URL> ', s)
    s = RE_EMAIL.sub(' <EMAIL> ', s)
    s = RE_PHONE.sub(' <PHONE> ', s)

    # Убираем управляющие символы, но сохраняем . , ; : ! ? - ' "
    # Заменяем прочие символы на пробел
    s = re.sub(r"[^A-Za-z0-9\s\.\,\;\:\!\?\'\"\-\(\)]", " ", s)

    # Схлопываем повторяющиеся знаки, но оставляем максимум 2 (например ??, --)
    s = re.sub(r'([\.!\?,;:\-])\1{2,}', r'\1\1', s)

    # Правим пробелы вокруг пунктуации
    s = re.sub(r'\s+([.,;:!?])', r'\1', s)
    s = re.sub(r'([(\-])\s+', r'\1', s)     # после ( и -
    s = re.sub(r'\s{2,}', ' ', s).strip()

    if not keep_case:
        s = s.lower()
    return s



#texts_train=train['excerpt'].apply(clean_for_readability)


texts_train=train['excerpt']


texts_train


labels=train['target'].tolist()


#texts_test=test['excerpt'].apply(clean_for_readability)


texts_test=test['excerpt']


texts_test


train_texts, eval_texts, train_labels, eval_labels=train_test_split(texts_train, labels, )





class ReadDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, ):
        self.texts=texts
        self.labels=labels
        self.tokenizer=tokenizer
        self.max_len=max_len
        
    def __len__(self):
        return len(self.texts)
    def __getitem__(self,idx):
        text=self.texts.iloc[idx]
        encoding=self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt',
        )
        return{
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32)
        }
    


MODEL_DIR = "/kaggle/input/huggingface-roberta/"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR + "roberta-base")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR + "roberta-base", problem_type='regression', num_labels=1)


#name='bert-base-uncased'
#model=AutoModelForSequenceClassification.from_pretrained(name,problem_type='regression', num_labels=1)


#tokenizer=AutoTokenizer.from_pretrained(name)


train_dataset=ReadDataset(train_texts, train_labels, tokenizer, max_len=512)
eval_dataset=ReadDataset(eval_texts, eval_labels, tokenizer, max_len=512)
test_dataset=ReadDataset(texts_test, [0]*len(texts_test), tokenizer, max_len=512)



import numpy as np
from sklearn.metrics import mean_squared_error

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.asarray(preds)
    labels = np.asarray(labels)

    # Частые формы (N, 1) → (N,)
    if preds.ndim == 2 and preds.shape[1] == 1:
        preds = preds.squeeze(1)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels.squeeze(1)

    # Мультиаутпут: усредняем RMSE по таргетам
    if preds.ndim == 2 and labels.ndim == 2 and preds.shape[1] > 1:
        rmses = [
            mean_squared_error(labels[:, j], preds[:, j], squared=False)
            for j in range(preds.shape[1])
        ]
        return {"rmse": float(np.mean(rmses))}

    rmse = mean_squared_error(labels, preds, squared=False)
    return {"rmse": float(rmse)}



args=TrainingArguments(
    output_dir='goal',
    learning_rate=3e-5,
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    lr_scheduler_type='cosine',
    warmup_ratio=0.05,
    eval_strategy='epoch',
    metric_for_best_model="rmse", 
    greater_is_better=False,
    report_to='none',
    weight_decay=0.01,
)


trainer=Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    args=args
)


trainer.train()



preds = trainer.predict(test_dataset).predictions.reshape(-1)
sub_simple = pd.DataFrame({"id": test["id"], "target": preds})
sub_simple.to_csv("submission.csv", index=False)


SAVE_DIR = "./roberta_readability_ckpt"   # will hold config.json, pytorch_model.bin, vocab, etc.
trainer.save_model(SAVE_DIR)              # saves model weights + config
tokenizer.save_pretrained(SAVE_DIR)       # saves tokenizer files

# ---------- 4. (optional) zip it for upload ----------
import shutil, os, zipfile
ZIP_NAME = "roberta_readability_ckpt.zip"
shutil.make_archive(base_name=SAVE_DIR, format='zip', root_dir=SAVE_DIR)

print(f"Checkpoint written to {SAVE_DIR}  and zipped as {ZIP_NAME}")




