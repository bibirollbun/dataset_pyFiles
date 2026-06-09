# !python -v


!pip -V


!pip show datasets


SEED = 2025


import pandas as pd
from datasets import Dataset,  load_dataset
import json
import os
import re
from collections import Counter
import random
from transformers import AutoTokenizer, AutoModelForPreTraining
import torch
import gc
import time
import numpy as np
# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seed for reproducibility
def set_seed(seed=2025):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
print(device)
# print(torch.cuda.is_available())  # should be True
# print(torch.cuda.get_device_name(0))  # prints GPU name
print(f"Seed: {SEED}")


def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)



# Load the dataset
dataset = load_dataset("KFUPM-JRCAI/arabic-generated-abstracts")

# Convert each split (if needed) to pandas
df = dataset['from_title_and_content'].to_pandas()
# If there are other splits, you can load them the same way:
# test_df = dataset['test'].to_pandas()

# Display the first few rows




print(df.info())


o = pd.DataFrame(data = {"content": df["original_abstract"].to_list(), "Class": ["human"] * len(df["original_abstract"]) })

text_machines = []
for i in range(1,5):
    text_machines.append(pd.DataFrame(data = {"content":df.iloc[:,i].to_list(), "Class": ["machine"] * len(df.iloc[:,i])}))



df = pd.concat([o]+text_machines)


df.info()


train_df = pd.read_csv("/kaggle/input/arageneval-subtask-3/ground_truth.csv")
val_df = pd.read_csv("/kaggle/input/arageneval-subtask-3/dev_unlabeled.csv")
test_df = pd.read_csv("/kaggle/input/arageneval-subtask-3/test_unlabeled.csv")


train_df = pd.concat([train_df, df])


train_df.info()
val_df.info()
test_df.info()


train_df = train_df.drop("ID", axis=1)


display(train_df.head())
display(val_df.head())
display(test_df.head())


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def eda_summary(df, text=None, label=None, name='DataFrame'):
    print(f'ğŸ“‹ Summary of {name}')
    print(df.info())
    print('\nğŸ“Œ Basic stats:')
    print(df.describe(include='object'))

    print('\nğŸ”� Null values:')
    print(df.isnull().sum())
    if(label):
        print('\nğŸ‘¥ Number of unique labels:', df[label].nunique())
        print('ğŸ§¾ Sample labels:', df[label].unique()[:10])

        # Distribution of samples per author
        label_counts = df[label].value_counts()
        print('\nğŸ“Š Top 10 labels by number of samples:')
        print(label_counts.head(10))

        plt.figure(figsize=(12, 4))
        label_counts.plot(kind="bar")
        # sns.histplot(author_counts, bins=len(author_counts), kde=True)
        plt.title(f'{name}: Samples per Label')
        plt.xlabel('Number of Samples')
        plt.ylabel('Author Count')
        plt.show()
    if text:
        # Text length analysis
        df['text_length'] = df[text].fillna("").dropna().apply(len)
        df['word_count'] = df[text].fillna("").dropna().apply(lambda x: len(str(x).split()))

        print('\nğŸ“� Text Length Statistics:')
        print(df[['text_length', 'word_count']].describe())

        # Histograms
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        sns.histplot(df['text_length'], bins=50, ax=axes[0])
        axes[0].set_title(f'{name}: Text Length (chars)')
        sns.histplot(df['word_count'], bins=50, ax=axes[1])
        axes[1].set_title(f'{name}: Word Count')
        plt.tight_layout()
        plt.show()

    return df

train_df = eda_summary(train_df, text='content', label='Class', name='Train')
val_df = eda_summary(val_df, text='content', name='Validation')
test_df = eda_summary(test_df, text='content', name='Test')


train_df = train_df.dropna()


id2label = train_df["Class"].unique().tolist()
label2id = {name: i for i, name in enumerate(id2label)}
num_classes = len(id2label)
print(list(zip(id2label, label2id.values())))


train_df["Class_encoded"] = train_df["Class"].apply(lambda x: label2id[x])


import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import torch.optim as optim
from tqdm.auto import tqdm


# Define a custom dataset for our text data
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels  # Can be None for test data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            # padding='longest',
            max_length=self.max_length,
            return_tensors="pt"
        )
        # Remove batch dimension
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item


from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

model_name = "NAMAA-Space/AraModernBert-Base-V1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes, 
                                                           label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True, 
                                                           # pad_token_id=tokenizer.pad_token_id

)
model.to(device)
print(model)


import torch
import torch.nn.functional as F

data = "Ù†Øµ ØªØ¬Ø±ÙŠØ¨ÙŠ"
input_tokenized = tokenizer([data], return_tensors="pt").to(device)
outputs = model(**input_tokenized)

logits = outputs.logits  # shape: (batch_size, num_classes)

probs = F.softmax(logits, dim=-1)


confidence, predicted_class_id = torch.max(probs, dim=-1)
confidence = confidence.item()
predicted_class_id = predicted_class_id.item()

predicted_label = model.config.id2label[predicted_class_id]
print("Input:", data)
print("Tokenized text:", input_tokenized)
print("Logits shape:", logits.shape)
print("Logits:", logits)
print(f"Predicted label: {predicted_label}")
print(f"Confidence: {confidence:.4f}")  # Ù…Ù† 0 Ø¥Ù„Ù‰ 1


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        # Move all tensor data in the batch to the device
        for key in batch:
            batch[key] = batch[key].to(device)
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    preds = []
    true_labels = []
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        for key in batch:
            batch[key] = batch[key].to(device)
        outputs = model(**batch)
        logits = outputs.logits
        batch_preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        preds.extend(batch_preds)
        if "labels" in batch:
            true_labels.extend(batch["labels"].detach().cpu().numpy())
    return np.array(preds), np.array(true_labels)

def predict_probas(model, dataloader, device):
    model.eval()
    all_probs = []
    for batch in tqdm(dataloader, desc="Predicting", leave=False):
        for key in batch:
            batch[key] = batch[key].to(device)
        outputs = model(**batch)
        probs = torch.softmax(outputs.logits, dim=1).detach().cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

def show_confusion_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix {title}")
    plt.show()


from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
MAX_LENGTH = 256
train_batch_size = 32
eval_batch_size = 4
data_collator = DataCollatorWithPadding(tokenizer)
train_texts = train_df['content'].tolist()
train_labels = train_df['Class_encoded'].tolist()

# train_texts = train_df['text_in_author_style'].tolist()
# train_labels = train_df['author_encoded'].tolist()

# val_texts = val_df['text_in_author_style'].tolist()
# val_labels = val_df['author_encoded'].tolist()
test_texts = test_df['content'].tolist()


train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_length=MAX_LENGTH)
# val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_length=MAX_LENGTH)
test_dataset = TextDataset(test_texts, None, tokenizer, max_length=MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=data_collator)
# val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=data_collator)
# test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=data_collator)

lr = 2e-5


optimizer = optim.AdamW(model.parameters(), lr=lr)
num_epochs = 4
for epoch in range(num_epochs):
    # Evaluate after each epoch
    train_loss = train_epoch(model, train_loader, optimizer, device)
    # val_preds, val_true = evaluate(model, val_loader, device)
    # epoch_f1 = f1_score(val_true, val_preds, average="micro")
    # print("Validation classification report")
    # print(classification_report(val_true, val_preds))
    # show_confusion_matrix(val_true, val_preds, labels=range(num_classes), title="Validation")
    # print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}, F1 (micro): {epoch_f1:.4f}")
    print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}")


test_texts = test_df['content'].tolist()


test_dataset = TextDataset(test_texts, None, tokenizer, max_length=MAX_LENGTH)
test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=data_collator)
test_probs = predict_probas(model, test_loader, device)
final_test_preds = np.argmax(test_probs, axis=1)
final_test_preds = list(map(lambda x:id2label[x], final_test_preds))


print(final_test_preds[:10])
len(final_test_preds)


submission = pd.DataFrame({
    "id": test_df["id"],
    "label": final_test_preds
})
submission.to_csv("predictions.csv", index=False)
print("Submission saved to predictions.csv")


!zip predictions.zip predictions.csv

