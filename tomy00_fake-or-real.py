import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup



# 0. PREPROCESSING DATA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv("/kaggle/input/fake-or-real-tomydataset/data/supertrain_merged.csv") 
df["label"] = df["real_text_id"].apply(lambda x: 1 if x == 1 else 0)
df["text"] = df.apply(lambda row: row["text_1"] + " [SEP] " + row["text_2"], axis=1)

# Tokenizer laden
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)

dataset = Dataset.from_pandas(df[["text", "label"]])
dataset = dataset.map(tokenize, batched=True)
dataset = dataset.train_test_split(test_size=0.1)



# 1. BUILDING
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch",
)



# 2. TRAINING
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)
dataset["train"].set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
dataset["test"].set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_loader = DataLoader(dataset["train"], batch_size=8, shuffle=True)
eval_loader = DataLoader(dataset["test"], batch_size=8)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()
num_epochs = 10

num_training_steps = num_epochs * len(train_loader)
num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)


for epoch in tqdm(range(num_epochs), desc="Training läuft..."):
    model.train()
    loop = tqdm(train_loader, leave=True)
    total_loss = 0

    for batch in loop:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()  # <-- Lernrate anpassen

        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"Train loss Epoch {epoch+1}: {avg_loss:.4f}")


model.save_pretrained("./bert-fake-vs-real")
tokenizer.save_pretrained("./bert-fake-vs-real")




# 3. SUBMISSION
from datasets import Dataset

df_test = pd.read_csv("/kaggle/input/test-real-or-fake-tomy/test_merged.csv")
df_test["text"] = df_test["text_1"].astype(str) + " [SEP] " + df_test["text_2"].astype(str)
df_test = df_test.dropna(subset=["text"])

test_dataset = Dataset.from_pandas(df_test)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

test_dataset = test_dataset.map(tokenize_function, batched=True)

test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

test_loader = DataLoader(test_dataset, batch_size=8)


model = BertForSequenceClassification.from_pretrained("./bert-fake-vs-real")
model.to(device)
model.eval()


predictions = []
with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        predictions.extend(preds.cpu().numpy())




# 4. Submission CSV erstellen
submission = pd.DataFrame({
    "pair_id": df_test["pair_id"],
    "prediction": predictions
})

submission.to_csv("test_submission.csv", index=False)
print("✅ submission.csv erfolgreich gespeichert.")


