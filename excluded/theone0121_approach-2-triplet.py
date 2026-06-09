# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/equinox-triplet/ideology_predictions_with_context.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import ast

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = torch.cuda.amp.GradScaler()

df = pd.read_csv("/kaggle/input/equinox-triplet/ideology_predictions_with_context.csv")
data = []

for _, row in df.iterrows():
    if pd.isna(row['scores']):
        continue
    data.append({
        'combined_text': row['combined_text'],
        'scores': ast.literal_eval(row['scores']),
        'top_label': row.get('top_label', None)
    })

def parse_bias_score(score_dict):
    keys = ['Far Left', 'Left', 'Center', 'Right', 'Far Right']
    return np.array([score_dict[k] for k in keys], dtype=np.float32)

class TripletDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.data = data
        self.texts = [d['combined_text'] for d in data]
        self.labels = [np.argmax(parse_bias_score(d['scores'])) for d in data]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return {
            'input': self.tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length'),
            'label': torch.tensor(label, dtype=torch.long)
        }

class BiasEncoder(nn.Module):
    def __init__(self, model_name):
        super(BiasEncoder, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.encoder.config.hidden_size, 256)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        return self.proj(pooled)

def cosine_distance(x, y):
    return 1.0 - nn.functional.cosine_similarity(x, y)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BiasEncoder("bert-base-uncased")


if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model = model.to(device)

optimizer = AdamW(model.parameters(), lr=3e-5)
loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=cosine_distance, margin=0.5)

dataset = TripletDataset(data, tokenizer)
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)

num_epochs = 3
total_steps = len(loader) * num_epochs
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)

def get_triplets(embeddings, labels):
    triplets = []
    labels = labels.cpu().numpy()
    for i in range(len(embeddings)):
        anchor = embeddings[i]
        anchor_label = labels[i]
        pos_idx = np.where(labels == anchor_label)[0]
        neg_idx = np.where(labels != anchor_label)[0]
        if len(pos_idx) < 2 or len(neg_idx) < 1:
            continue
        pos = embeddings[random.choice(pos_idx[pos_idx != i])]
        neg = embeddings[random.choice(neg_idx)]
        triplets.append((anchor, pos, neg))
    return triplets

model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
        inputs = batch['input']
        labels = batch['label'].to(device)
        input_ids = inputs['input_ids'].squeeze(1).to(device)
        attention_mask = inputs['attention_mask'].squeeze(1).to(device)

        with torch.cuda.amp.autocast():
            embeddings = model(input_ids, attention_mask)
            triplets = get_triplets(embeddings, labels)
            if not triplets:
                continue
            anchors, positives, negatives = zip(*triplets)
            anchors = torch.stack(anchors).to(device)
            positives = torch.stack(positives).to(device)
            negatives = torch.stack(negatives).to(device)
            loss = loss_fn(anchors, positives, negatives)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")





