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
import numpy as np
import re
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_scheduler
from tqdm import tqdm


def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame):
    # Create target label
    train_df['target'] = train_df['Category'] + ':' + train_df['Misconception']

    # Filter rare classes
    class_counts = train_df['target'].value_counts()
    keep = class_counts[class_counts > 1].index
    train_df = train_df[train_df['target'].isin(keep)].copy()

    # Label Encoding
    label_encoder = LabelEncoder()
    train_df['encoded_label'] = label_encoder.fit_transform(train_df['target'])

    # Clean text
    def clean(row):
        answer = re.sub(r'\\[\(\)a-zA-Z{}]', '', row['MC_Answer']).strip()
        return f"Question: {row['QuestionText']} Correct Answer: {answer} Student Explanation: {row['StudentExplanation']}"

    train_df['input_text'] = train_df.apply(clean, axis=1)
    test_df['input_text'] = test_df.apply(clean, axis=1)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        train_df['input_text'].values, 
        train_df['encoded_label'].values,
        test_size=0.2, 
        stratify=train_df['encoded_label'], 
        random_state=42
    )

    return X_train, X_val, y_train, y_val, label_encoder, test_df



class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in enc.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item



def train_model(model, train_loader, val_loader, epochs=3, lr=2e-5):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_training_steps = len(train_loader) * epochs
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            pbar.set_postfix({"loss": loss.item()})
    return model



from torch.optim import AdamW  # ✅ correct and recommended now


# Load your CSVs
train_df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test_df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")


# Preprocess
X_train, X_val, y_train, y_val, label_encoder, test_df = preprocess(train_df, test_df)

# Tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Datasets
train_dataset = TextDataset(X_train, y_train, tokenizer)
val_dataset = TextDataset(X_val, y_val, tokenizer)

# Loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Model
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=len(label_encoder.classes_))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train
model = train_model(model, train_loader, val_loader, epochs=10)



class TestDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in enc.items()}
        return item

test_dataset = TestDataset(test_df['input_text'].tolist(), tokenizer)
test_loader = DataLoader(test_dataset, batch_size=16)

model.eval()
predictions = []

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())

# Decode labels
decoded_preds = label_encoder.inverse_transform(predictions)

# Add predictions to test_df
test_df['Predicted'] = decoded_preds
test_df[['ID', 'Predicted']].to_csv("submission.csv", index=False)



model_name = "google/gemma-2-9b-it"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_encoder.classes_),
    device_map="auto",
    quantization_config=bnb
)
model.eval()

