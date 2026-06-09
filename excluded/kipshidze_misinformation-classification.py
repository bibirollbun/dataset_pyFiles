import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
import torch.nn as nn


dataframe = pd.read_csv("/kaggle/input/depi-r-2-competition-1/xy_train.csv")
dataframe["label"] = dataframe["label"].replace(2, 1)

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token


class MisinformationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]["text"]
        label = self.data.iloc[idx]["label"]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="np"
        )
        
        return {
            "input_ids": torch.tensor(encoding["input_ids"]).squeeze(0),
            "attention_mask": torch.tensor(encoding["attention_mask"]).squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


dataset = MisinformationDataset(dataframe, tokenizer)

dataloader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True
)


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, nhead=4, num_layers=2, num_classes=2, max_length=128):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_length, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids) + self.positional_encoding[:, :input_ids.size(1), :]
        x = self.transformer_encoder(x)
        x = x[:, 0, :]
        x = self.dropout(x)
        return self.fc(x)


model = TransformerClassifier(
    vocab_size=tokenizer.vocab_size,
    embed_dim=512,
    nhead=8,
    num_layers=8,
    num_classes=2,
    max_length=128
)


for batch in dataloader:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["label"]
    logits = model.forward(input_ids, attention_mask)
    break

labels, logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(
    params=model.parameters(),
    lr=2e-5
)

criterion = nn.CrossEntropyLoss()

num_epochs = 4


for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for n, batch in enumerate(dataloader, 1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model.forward(input_ids=input_ids, attention_mask=attention_mask)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(f"Batch {n}/{len(dataloader)} - Loss: {total_loss/n:.4f} | {loss:.4f}", end="\r", flush=True)
    print(end="\n")

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")


class TestDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.texts = dataframe["text"].tolist()
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

df_test = pd.read_csv("/kaggle/input/depi-r-2-competition-1/x_test.csv")

test_dataset = TestDataset(df_test, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


model.eval()

all_preds = []
with torch.no_grad():
    for n, batch in enumerate(test_loader, 1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.tolist())
        print(f"Batch {n}/{len(test_loader)} - {len(all_preds)}", end="\r")


submission = pd.DataFrame({
    "ID": df_test["ID"],
    "label": all_preds
})
submission.to_csv("./submission.csv", index=False)
submission

