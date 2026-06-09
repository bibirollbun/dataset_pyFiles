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


train_df=pd.read_csv("/kaggle/input/aiquest-bangla-sentiment-analysis-competition/train.csv")
test_df=pd.read_csv("/kaggle/input/aiquest-bangla-sentiment-analysis-competition/sample_submission.csv")


train_df
test_df


for i in range(1, 3):
   globals()[f"train{i}"] = train_df.copy()


import os
import re
import gc
import numpy as np
import pandas as pd
import torch
import unicodedata
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import nltk



# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')


# Bangla text cleaner
def bangla_cleaner(text):
    stop_words = set(stopwords.words('bengali'))
    text = re.sub(r'\|?See Translation', '', text)
    text = text.lower()
    text = re.sub(r'[a-z0-9_]+', '', text)
    text = re.sub(r'[।॥!?.,;:\"\'‘’—…()\[\]{}<>@#$%^&*_+=|\\/~`]', '', text)
    text = re.sub(r'[^ঀ-৿\s]', '', text)
    text = unicodedata.normalize('NFKC', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


# Clean text
train_df['clean_text'] = train_df['text'].apply(bangla_cleaner)
test_df['clean_text'] = test_df['sentiment'].apply(bangla_cleaner)


# Label encoding
le = LabelEncoder()
train_df['label'] = le.fit_transform(train_df['sentiment'])


# Tokenizer
model_name = "csebuetnlp/banglabert"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Dataset class
class BanglaDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=128)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


# Cross-validation setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df['clean_text'], train_df['label'])):
    print(f"\nFold {fold+1}")
    train_texts = train_df.iloc[train_idx]['clean_text'].tolist()
    val_texts = train_df.iloc[val_idx]['clean_text'].tolist()
    train_labels = train_df.iloc[train_idx]['label'].tolist()
    val_labels = train_df.iloc[val_idx]['label'].tolist()

    train_dataset = BanglaDataset(train_texts, train_labels)
    val_dataset = BanglaDataset(val_texts, val_labels)

    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)



!pip install transformers --upgrade
training_args = TrainingArguments(
    output_dir=f"./results_fold_{fold+1}",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    logging_dir=f"./logs_fold_{fold+1}",
    logging_steps=10,
    save_strategy="epoch",
    # The 'evaluation_strategy' argument was introduced in a later version
    # of Transformers. For older versions, you might need to specify it
    # differently or omit it altogether.
    # evaluation_strategy="epoch", 
    report_to="none"
)




    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            'accuracy': accuracy_score(labels, preds),
            'f1': f1_score(labels, preds, average='macro')
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()

    # Predict on test
    test_dataset = BanglaDataset(test_df['clean_text'].tolist())
    preds = trainer.predict(test_dataset).predictions
    test_preds.append(torch.softmax(torch.tensor(preds), dim=1).numpy())

    del model
    torch.cuda.empty_cache()
    gc.collect()

# Average predictions
final_preds = np.mean(test_preds, axis=0)
pred_labels = np.argmax(final_preds, axis=1)
decoded_preds = le.inverse_transform(pred_labels)

# Create submission
submission = pd.DataFrame({
    "id": test_df["id"],
    "sentiment": decoded_preds
})

submission.to_csv("submission.csv", index=False)
print("submission.csv created!")





