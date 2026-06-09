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


# Installation (one-time run)
!pip install torch


# Installation (one-time run)
!pip install transformers==4.38.2 peft==0.8.2 accelerate==0.29.3



# Hide unnecessary warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
import torch


train = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')

train = train.fillna("")
test = test.fillna("")
print("Train shape:", train.shape)
print("Test shape:", test.shape)


def create_transformer_text(row):
    return f"""
Rule: {row['rule']}
Post: {row['body']}
Positive: {row['positive_example_1']}; {row['positive_example_2']}
Negative: {row['negative_example_1']}; {row['negative_example_2']}
Violation? 0=No, 1=Yes
""".strip()

train['transformer_text'] = train.apply(create_transformer_text, axis=1)
test['transformer_text'] = test.apply(create_transformer_text, axis=1)



class TransformerDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors=None
        )
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': label
        }



def train_transformer_model(model_name, train_texts, val_texts, train_labels, val_labels, test_texts):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset = TransformerDataset(train_texts, train_labels, tokenizer)
    val_dataset = TransformerDataset(val_texts, val_labels, tokenizer)
    test_dataset = TransformerDataset(test_texts, [0]*len(test), tokenizer)

    training_args = TrainingArguments(
        output_dir='./temp',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        warmup_steps=100,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=torch.cuda.is_available(),
        logging_dir='./log',
        report_to="none",
        disable_tqdm=True,
        seed=42
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()
    outputs = trainer.predict(test_dataset).predictions
    return torch.nn.functional.softmax(torch.tensor(outputs), dim=-1)[:, 1].numpy()



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds_deberta = np.zeros(len(test))
test_preds_roberta = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(skf.split(train, train['rule_violation'])):
    print(f"\n Fold {fold+1}/5")

    X_train_t = train['transformer_text'].iloc[train_idx].values
    X_val_t = train['transformer_text'].iloc[val_idx].values
    y_train = train['rule_violation'].iloc[train_idx].values
    y_val = train['rule_violation'].iloc[val_idx].values

    # DeBERTa-v3-small
    deberta_proba = train_transformer_model(
        "microsoft/deberta-v3-small", X_train_t, X_val_t, y_train, y_val, test['transformer_text']
    )
    test_preds_deberta += deberta_proba / 5

    # RoBERTa-base
    roberta_proba = train_transformer_model(
        "roberta-base", X_train_t, X_val_t, y_train, y_val, test['transformer_text']
    )
    test_preds_roberta += roberta_proba / 5



def create_tfidf_text(row):
    return f"{row['body']} {row['rule']}"

train['tfidf_text'] = train.apply(create_tfidf_text, axis=1)
test['tfidf_text'] = test.apply(create_tfidf_text, axis=1)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')
X_train_tfidf = vectorizer.fit_transform(train['tfidf_text'])
X_test_tfidf = vectorizer.transform(test['tfidf_text'])

# Model training
lr_model = LogisticRegression(C=1.0, max_iter=1000)
lr_model.fit(X_train_tfidf, train['rule_violation'])

# Expectations
lr_proba = lr_model.predict_proba(X_test_tfidf)[:, 1]



final_proba = (
    0.50 * test_preds_deberta +   # The strongest
    0.35 * test_preds_roberta +   # strong
    0.15 * lr_proba               # Adds variety
)


# Save the final result
submission = pd.DataFrame({
    'row_id': test['row_id'],
    'rule_violation': final_proba
})

submission.to_csv('submission_ensemble_final.csv', index=False)
print("\n Saved: submission_ensemble_final.csv")
print(submission.head())

# Save individual forecasts (optional)
pd.DataFrame({
    'row_id': test['row_id'],
    'deberta': test_preds_deberta,
    'roberta': test_preds_roberta,
    'tfidf_lr': lr_proba,
    'final': final_proba
}).to_csv('ensemble_breakdown.csv', index=False)





