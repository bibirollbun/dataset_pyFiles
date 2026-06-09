import pandas as pd
import torch
import re
import matplotlib.pyplot as plt
import numpy as np
from random import randint
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix


print(torch.cuda.is_available())
print(torch.cuda.device_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current device name:", torch.cuda.get_device_name(0))


# Load datasets
train_df = pd.read_csv('/kaggle/input/bert-classification-ioai/train.tsv', sep=',')
test_df = pd.read_csv('/kaggle/input/bert-classification-ioai/test.tsv', sep=',')

print("Training set size:", train_df.shape)
print("Test set size:", test_df.shape)
train_df.head(5)


class_counts = train_df['class'].value_counts()
print("Class distribution in training data:")
for label, count in class_counts.items():
    print(f"  Class {label}: {count} tweets ({count/len(train_df)*100:.2f}%)")


plt.figure(figsize=(4,3))
plt.bar(class_counts.index.astype(str), class_counts.values, color=['skyblue','salmon'])
plt.xlabel('Class'); plt.ylabel('Number of Tweets')
plt.title('Training Set Class Distribution')
plt.xticks([0,1], labels=['Not ADR (0)', 'ADR (1)'])
plt.show()


class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.encodings['input_ids'])
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item


class_counts = train_df['class'].value_counts()
print("Class distribution in training data:")
for label, count in class_counts.items():
    print(f"  Class {label}: {count} tweets ({count/len(train_df)*100:.2f}%)")


def preprocess_tweet(text: str) -> str:
    text = text.lower()
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^\w\s,.!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

train_df['clean_tweet'] = train_df['tweet'].apply(preprocess_tweet)
test_df['clean_tweet'] = test_df['tweet'].apply(preprocess_tweet)

for i in range(3):
    print("Original:", train_df.loc[i, 'tweet'])
    print("Cleaned: ", train_df.loc[i, 'clean_tweet'])
    print("---")


class_weights = torch.tensor([0.3, 0.7])

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get('logits')
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


model_name = "ai-forever/ruBert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)


MAX_LEN = 128
train_encodings = tokenizer(list(train_df['clean_tweet']), truncation=True, padding=True, max_length=MAX_LEN)
test_encodings = tokenizer(list(test_df['clean_tweet']), truncation=True, padding=True, max_length=MAX_LEN)


example_idx = 1
print("Tweet:", train_df['clean_tweet'].iloc[example_idx])
print("Tokens:", tokenizer.tokenize(train_df['clean_tweet'].iloc[example_idx]))
print("Token IDs:", train_encodings['input_ids'][example_idx][:10], "...")
print("Attention Mask:", train_encodings['attention_mask'][example_idx][:10], "...")


train_indices, val_indices = train_test_split(
    range(len(train_df)), test_size=0.1,
    stratify=train_df['class'], random_state=42
)
train_indices = list(train_indices); val_indices = list(val_indices)


train_inputs = {key: [val[i] for i in train_indices] for key, val in train_encodings.items()}
val_inputs   = {key: [val[i] for i in val_indices]   for key, val in train_encodings.items()}
train_labels = train_df['class'].iloc[train_indices].tolist()
val_labels   = train_df['class'].iloc[val_indices].tolist()

print(f"Training on {len(train_labels)} tweets, validating on {len(val_labels)} tweets.")
print("ADR class frequency in train split: {:.2f}%".format(sum(train_labels)/len(train_labels)*100))
print("ADR class frequency in val split: {:.2f}%".format(sum(val_labels)/len(val_labels)*100))


train_dataset = TweetDataset(train_inputs, train_labels)
val_dataset   = TweetDataset(val_inputs, val_labels)

class_weights = torch.tensor([0.3, 0.7])

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get('logits')
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model = model.to(device)


training_args = TrainingArguments(
    output_dir='./model_checkpoints',
    eval_strategy='epoch',
    save_strategy='no',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=1e-05,
    weight_decay=0.01,
    logging_strategy='epoch',
    seed=42,
    warmup_steps=100,
    disable_tqdm=False,
    logging_steps=10,
    report_to=[]
)


trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
)

trainer.train()


val_outputs = trainer.predict(val_dataset)
val_logits = val_outputs.predictions
val_probs = torch.nn.functional.softmax(torch.tensor(val_logits), dim=1).numpy()[:, 1]


thresholds = np.linspace(0.3, 0.7, 41)
best_thresh = 0.5
best_f1 = 0.0
for t in thresholds:
    val_pred_adjusted = (val_probs >= t).astype(int)
    f1 = f1_score(np.array(val_labels), val_pred_adjusted, pos_label=1)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t
print(f"Best threshold: {best_thresh:.2f} with F1: {best_f1:.5f}")


val_pred_final = (val_probs >= best_thresh).astype(int)
print("Classification Report on validation:\n", classification_report(np.array(val_labels), val_pred_final, target_names=['Not ADR (0)', 'ADR (1)']))
print("Confusion Matrix:\n", confusion_matrix(np.array(val_labels), val_pred_final))


test_dataset = TweetDataset(test_encodings, labels=None)


test_outputs = trainer.predict(test_dataset)
test_logits = test_outputs.predictions
test_probs = torch.nn.functional.softmax(torch.tensor(test_logits), dim=1).numpy()[:, 1]
test_pred_labels = (test_probs >= best_thresh).astype(int)

submission_df = pd.DataFrame({
    'id': test_df['id'],
    'class': test_pred_labels
})

submit_id = randint(1000, 10000)

submission_df.to_csv(f'test_predictions_{submit_id}.csv', index=False)
print(f"Сохранены предсказания в файл test_predictions_{submit_id}.csv")

