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

try:
    # Hugging Face convenience fn; sets Python/Rand, NumPy, Torch seeds
    from transformers import set_seed  
except ImportError:
    set_seed = None

def seed_everything(seed: int = 42):
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
seed_everything(42)



device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')


!pip install --quiet evaluate


import torch 
import spacy
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, PreTrainedTokenizer, BertTokenizer, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, EvalPrediction 
import re
import evaluate 
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


train=pd.read_csv("/kaggle/input/toxic-comment-classification/train_preproccesed.csv")
test=pd.read_csv("/kaggle/input/toxic-comment-classification/test_preproccesed.csv")


train=pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip")
test=pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip")
test_labels=pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip")
sample=pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip")


'''
убрать большие буквы
убрать кавычки 
убрать обратные слеши и n (\n)
до базовых слов сделать 
леммитизация (стемминг на крайний случай)
'''


#train.info()


nlp=spacy.load('en_core_web_sm')


def clean(text):
    #text=str(text)
    text=text.lower()
    text=text.replace('\n', ' ')
    text=re.sub(r'[^a-zA-Z0-9]', ' ', text)
    text=re.sub(r"\s+", " ", text).strip()
    doc=nlp(text)
    lemmas=[token.lemma_ for token in doc if not token.is_space]
    return " ".join(lemmas)


train['comment_text']=train['comment_text'].apply(clean)


test['comment_text']=test['comment_text'].apply(clean)


train.to_csv("train_preproccesed.csv", index=False)


test.to_csv("test_preproccesed.csv", index=False)


фффффффф


class ToxicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts=texts
        self.labels=labels
        self.tokenizer=tokenizer
        self.max_len=max_len
        
    def __len__(self):
        return len(self.texts)

        
    def __getitem__(self, idx):
        text  = str(self.texts[idx])
        labels_for_one=self.labels[idx]
        encoding=self.tokenizer(
            text, 
            max_length=self.max_len,
            truncation   = True,
            padding='max_length',
            add_special_tokens=True,
            return_tensors='pt',
            return_attention_mask=True,
            
            
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels_for_one, dtype=torch.float),
        }
        


full_train_texts=train['comment_text'].tolist()
test_texts=test['comment_text'].tolist()

train_labels=train[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].values



train_texts, eval_texts, train_targets, eval_targets=train_test_split(full_train_texts, train_labels, test_size=0.1)


tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased')


train_dataset=ToxicDataset(train_texts, train_targets,tokenizer, max_len=256 )
eval_dataset=ToxicDataset(eval_texts, eval_targets,tokenizer, max_len=256 )



config=AutoConfig.from_pretrained(
    'bert-base-uncased', 
    num_labels=6,
    problem_type="multi_label_classification"
)


model=AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', config=config)


model.to(device)


roc_auc = evaluate.load("roc_auc", "multilabel")

# 2) Define a compute_metrics fn
def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    # Convert logits → probabilities with sigmoid
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    # Pass them in as "prediction_scores"
    result = roc_auc.compute(prediction_scores=probs, references=labels)
    return {"roc_auc": result["roc_auc"]}


import wandb


wandb.login(key='09a943df432f7799fb447e658f9b2149dee74ee0')


import os

# chose a name for your W&B project
os.environ["WANDB_PROJECT"] = "toxic-comment-classification"



args= TrainingArguments(
    output_dir='gol',
    #eval_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=3e-5,
    num_train_epochs=1,
    lr_scheduler_type='cosine',
    #warmup_ratio=0.1,
    warmup_steps=500,
    #save_strategy='best',
    save_total_limit=2,
    seed=42,
    #fp16=True,
    greater_is_better=True,
    #label_smoothing_factor=0.1,
    logging_strategy="steps",             # log every N steps (needed for WandB)
    logging_steps=50,                     # change this to your desired frequency
    report_to="wandb",                    # enable WandB logging
    run_name="toxic-comment-classifier2", 
    load_best_model_at_end=True,
    metric_for_best_model="roc_auc",
    eval_strategy="steps",
    eval_steps=1000,

    # save a checkpoint every 1000 steps (and keep only the last 2)
    save_strategy="steps",
    save_steps=1000,

    #project bame?
)


trainer=Trainer(
    args=args,
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)


trainer.train()





from scipy.special import expit  # vectorized sigmoid

# 1) Prepare your test dataset
#    If your ToxicDataset requires labels, just pass dummy zeros — they won’t be used at predict time.
test_labels_dummy = np.zeros((len(test_texts), 6), dtype=float)
test_dataset = ToxicDataset(
    texts    = test_texts,
    labels   = test_labels_dummy,
    tokenizer=tokenizer,
    max_len  = 256
)



trainer.compute_metrics = None


pred_out = trainer.predict(test_dataset)



probs = expit(pred_out.predictions)


submission = pd.DataFrame(
    probs,
    columns=[
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate"
    ]
)
# Don’t forget the test IDs
submission.insert(0, "id", test["id"].values)

# 5) Save to CSV
submission.to_csv("submission.csv", index=False)


pred_out = trainer.predict(test_dataset)

# 3) Convert logits → probabilities with sigmoid
#    If you’d rather use PyTorch: probs = torch.sigmoid(torch.from_numpy(pred_out.predictions)).numpy()
probs = expit(pred_out.predictions)

# 4) Build your submission DataFrame
submission = pd.DataFrame(
    probs,
    columns=[
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate"
    ]
)
# Don’t forget the test IDs
submission.insert(0, "id", test["id"].values)

# 5) Save to CSV
submission.to_csv("submission.csv", index=False)




