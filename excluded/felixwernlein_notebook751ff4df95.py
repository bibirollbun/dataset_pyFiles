import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report


df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
df.head()


df.info()


df.isnull().sum()


def create_input_text(row):
    return f"[RULE] {row['rule']} [SUBREDDIT] {row['subreddit']} [BODY] {row['body']} [POS1] {row['positive_example_1']} [POS2] {row['positive_example_2']} [NEG1] {row['negative_example_1']} [NEG2] {row['negative_example_2']}"


df['text'] = df.apply(create_input_text, axis=1)


train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'].tolist(), df['rule_violation'].tolist(), test_size=0.2, random_state=42)


tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/bert-base/bert-base-uncased-tokenizer")
model = AutoModelForSequenceClassification.from_pretrained("/kaggle/input/bert-base/bert/bert-base-uncased-model", num_labels=2)


class CustomDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        self.labels = labels

    def __getitems(self, idx):
        item = {kex: torch.tensor(val[idx] for key, val in self.encodings.items())}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

    def __len__(self):
        return len(self.labels)


train_dataset = CustomDataset(train_texts, train_labels)
val_dataset = CustomDataset(val_texts, val_labels)


training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="no",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    load_best_model_at_end=False
)


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    auc = roc_auc_score(p.label_ids, p.predictions[:, 1])
    return {"accuracy": (preds == p.labels_ids).mean(), "auc": auc}


train = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


train


test_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
test_df.head()


test_df['text'] = test_df.apply(create_input_text, axis=1)


# Tokenize
test_encodings = tokenizer(
    test_df["text"].tolist(),
    truncation=True,
    padding=True,
    max_length=512,
    return_tensors="pt"
)


import torch
# Ensure model is in eval mode and on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Run predictions in batches
batch_size = 16
probs = []

with torch.no_grad():
    for i in range(0, len(test_df), batch_size):
        batch = {key: val[i:i+batch_size].to(device) for key, val in test_encodings.items()}
        outputs = model(**batch)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)[:, 1]  # class 1 = rule_violation
        probs.extend(probabilities.cpu().numpy())



# Create submission DataFrame
submission = pd.DataFrame({
    "row_id": test_df["row_id"],
    "rule_violation": probs
})

# Save to CSV
submission.to_csv("submission.csv", index=False)


pd.read_csv('submission.csv')




