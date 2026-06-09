import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('/kaggle/input/depi-r-2-competition-1/xy_train.csv')
df.head()


df['label'].unique()


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='label', palette='viridis')
plt.title('Distribution of Labels')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()


df_train = pd.read_csv('/kaggle/input/depi-r-2-competition-1/xy_train.csv')
df_test = pd.read_csv('/kaggle/input/depi-r-2-competition-1/x_test.csv')


df_train = df_train[df_train['label'] != 2]


df_train.shape


from transformers import DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


from tqdm.auto import tqdm

tqdm.pandas()


import torch
from torch.utils.data import DataLoader, TensorDataset


device = torch.device('cuda' if torch.cuda.is_available() == True else 'cpu')


from transformers import DistilBertTokenizerFast
import torch
from torch.utils.data import Dataset, DataLoader

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

inputs = tokenizer(
    df_train["text"].tolist(),
    max_length=128,
    truncation=True,
    padding="max_length",
    return_tensors="pt",
    return_attention_mask=True,
    add_special_tokens=True
)


class BertDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': torch.tensor(self.labels[idx])
        }
        
    def __len__(self):
        return len(self.labels)

dataset = BertDataset(inputs, df_train["label"].tolist())


dataloader = DataLoader(
    dataset,
    batch_size=64, 
    shuffle=True,
    num_workers=4,  
    pin_memory=True  
)


from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(df_train["label"].unique())  # Auto-detects num classes
).to("cuda" if torch.cuda.is_available() else "cpu")


from sklearn.metrics import accuracy_score

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

for epoch in range(2): #Define epochs here...
    print(f"\nâœ¨ Epoch {epoch + 1}/3")
    model.train()

    train_loop = tqdm(dataloader, desc="Training", leave=False)
    all_preds = []
    all_labels = []
    
    for batch in train_loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(**batch)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch['labels'].cpu().numpy())
        
        current_acc = accuracy_score(all_labels, all_preds)
        train_loop.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{current_acc:.2%}"
        })
    
    epoch_acc = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch + 1} Results:")
    print(f"â€¢ Training Accuracy: {epoch_acc:.2%}")
    print(f"â€¢ Average Loss: {loss.item():.4f}")

print("\nğŸ�¯ Training complete!")


tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
encodings = tokenizer(list(df_test['text']), truncation=True, padding=True, max_length=256)

class ReviewDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return len(self.encodings['input_ids'])
    def __getitem__(self, idx):
        return {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }

test_dataset = ReviewDataset(encodings)
test_loader = DataLoader(test_dataset, batch_size=16)

model.eval()
predictions = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())

df_test['predicted_label'] = predictions
df_test[['ID', 'predicted_label']].to_csv("submissions.csv", index=False)

