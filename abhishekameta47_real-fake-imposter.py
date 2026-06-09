import pandas as pd
import os


train_df = pd.read_csv("/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv")

Folder_path = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train"

text_data = []
labels = []

for _, txt_id in train_df.iterrows():
    folder_id = int(txt_id["id"])
    real_txt_number = txt_id["real_text_id"]  # 1 or 2

    folder_name = f"article_{folder_id:04d}"  
    folder_path = os.path.join(Folder_path, folder_name)

    # File names
    real_file = f"file_{real_txt_number}.txt"
    fake_file = f"file_{1 if real_txt_number == 2 else 2}.txt"

    real_path = os.path.join(folder_path, real_file)
    fake_path = os.path.join(folder_path, fake_file)

    # Read and append
    with open(real_path, "r", encoding="utf-8") as f:
        text_data.append(f.read())
        labels.append(1)

    with open(fake_path, "r", encoding="utf-8") as f:
        text_data.append(f.read())
        labels.append(0)




from sklearn.model_selection import train_test_split
from transformers import BigBirdTokenizerFast, BigBirdForSequenceClassification
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class ArticleDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        # Ensure labels are float tensors of shape [1]
        item["labels"] = torch.tensor([self.labels[idx]], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)





train_texts, val_texts, train_labels, val_labels = train_test_split(
    text_data, labels, test_size=0.2, stratify=labels, random_state=42
)

print(len(train_texts))
tokenizer = BigBirdTokenizerFast.from_pretrained("google/bigbird-roberta-base")

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=1024, return_tensors="pt")
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=1024, return_tensors="pt")

train_dataset = ArticleDataset(train_encodings, train_labels)
val_dataset = ArticleDataset(val_encodings, val_labels)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BigBirdForSequenceClassification.from_pretrained(
    "google/bigbird-roberta-base",
    num_labels=1
)

model = nn.DataParallel(model)

model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=2)

optimizer = Adam(model.parameters(), lr=5e-6)
loss_fn = nn.BCEWithLogitsLoss()
for epoch in range(5):
    model.train()
    for batch in train_loader:
        inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
        labels = batch["labels"].to(device).squeeze()
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze()
        loss = loss_fn(outputs, labels)
        loss.backward()
        torch.cuda.empty_cache()
        optimizer.step()

    # Validation
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for batch in val_loader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != "labels"}
            labels = batch["labels"].to(device).squeeze()
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze()
            preds.extend(outputs.cpu().numpy())
            actuals.extend(labels.cpu().numpy())

    sigmoid_preds = F.sigmoid(torch.tensor(preds))
    binary_preds = [1 if p > 0.5 else 0 for p in sigmoid_preds]
    acc = accuracy_score(actuals, binary_preds)
    print(f"Epoch {epoch+1}: Val Accuracy = {acc:.4f}")

    # Create confusion matrix
    cm = confusion_matrix(actuals, binary_preds)
    tn, fp, fn, tp = confusion_matrix(actuals, binary_preds).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Display it
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - Epoch {epoch+1}  || f1 score {f1_score:.2f}")
    plt.show()



class TestDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])


test_dir = "/kaggle/input/fake-or-real-the-impostor-hunt/data/test"
test_ids = sorted(os.listdir(test_dir))

test_texts = []
test_file_ids = []

for folder in test_ids:
    folder_path = os.path.join(test_dir, folder)
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])
    
    for file in files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            test_texts.append(text)
            
            folder_id = int(folder.replace("article_", ""))
            test_file_ids.append(f"{folder_id}")

test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=2048, return_tensors="pt")

test_dataset = TestDataset(test_encodings)
test_loader = DataLoader(test_dataset, batch_size=2)

model.eval()
predictions = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze()
        probs = torch.sigmoid(outputs)
        binary_preds = [1 if p > 0.5 else 0 for p in probs]
        predictions.extend(binary_preds)


submission_df = pd.DataFrame({
    "id": test_file_ids,
    "real_text_id": predictions
})

print(submission_df[:5])

ids=[]
data=[]
for i in range(0,len(submission_df),2):
    ids.append(submission_df['id'][i])
    if submission_df['real_text_id'][i] == 0:
        data.append(1)
    else:
        data.append(2)

submission = pd.DataFrame({
    "id": ids,
    "real_text_id": data
})

print(submission[:5])

submission.to_csv("/kaggle/working/submission.csv", index=False)


