# libraries
!pip install torch==2.0.1 torchvision==0.15.2 --quiet
!pip install transformers==4.30.2 pandas scikit-learn tqdm --quiet

import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from transformers.optimization import AdamW
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

# data loading and preprocessing
train_df = pd.read_csv("/kaggle/input/aiquest-bangla-sentiment-analysis-competition/train.csv")

def clean_text(text):
    text = str(text)
    text = text.replace("।", ".").replace("؟", "?").replace("!", "!")
    return " ".join(text.split())

train_df['text'] = train_df['text'].apply(clean_text)

# labels and tokenization
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
train_df['label'] = train_df['sentiment'].map(label_map)  

tokenizer = AutoTokenizer.from_pretrained("sagorsarker/bangla-bert-base")

encoded_data = tokenizer(
    train_df['text'].tolist(),
    padding='max_length',
    truncation=True,
    max_length=128,
    return_tensors='pt'
)

labels = torch.tensor(train_df['label'].values, dtype=torch.long)

# tensor dataset
dataset = TensorDataset(
    encoded_data['input_ids'],
    encoded_data['attention_mask'],
    labels
)

# train and val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# model
model = AutoModelForSequenceClassification.from_pretrained(
    "sagorsarker/bangla-bert-base",
    num_labels=3
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


batch_size = 8
train_loader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=batch_size
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size
)

# imbalance handling
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_df['label'].values),
    y=train_df['label'].values
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

# train loop
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        optimizer.zero_grad()
        
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        batch_labels = batch[2].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=batch_labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    # eval
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            batch_labels = batch[2].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, batch_labels)
            val_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_labels).sum().item()
    
    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {total_loss/len(train_loader):.4f}")
    print(f"Val Loss: {val_loss/len(val_loader):.4f}")
    print(f"Val Accuracy: {correct/val_size:.4f}")

# submission template
sample_sub = pd.read_csv("/kaggle/input/aiquest-bangla-sentiment-analysis-competition/sample_submission.csv")
sample_sub['sentiment'] = 'neutral'  
sample_sub.to_csv('submission.csv', index=False)




