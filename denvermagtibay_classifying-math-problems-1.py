import pandas as pd
train_df = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv")
test_df = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv")


# Basic shape
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Preview data
train_df.head()

# Check for nulls
print(train_df.isnull().sum())
print(test_df.isnull().sum())


import matplotlib.pyplot as plt

# Count samples per label
label_counts = train_df['label'].value_counts().sort_index()

# Print actual counts
print("Label Distribution (Counts):")
for label, count in label_counts.items():
    print(f"Label {label}: {count}")

# Plot the bar chart
label_counts.plot(kind='bar')
plt.title("Label Distribution")
plt.xlabel("Topic Label")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.grid(True)
plt.show()


import matplotlib.pyplot as plt

# Character and word length
train_df['char_len'] = train_df['Question'].str.len()
train_df['word_len'] = train_df['Question'].str.split().apply(len)

# Character length stats
print("Character Length Statistics:")
print(train_df['char_len'].describe())
print("\n")

# Word length stats
print("Word Length Statistics:")
print(train_df['word_len'].describe())
print("\n")

# Character length histogram
train_df['char_len'].hist(bins=30)
plt.title("Character Length Distribution")
plt.xlabel("Number of Characters")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Word length histogram
train_df['word_len'].hist(bins=30)
plt.title("Word Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


for label in sorted(train_df['label'].unique()):
    sample = train_df[train_df['label'] == label].sample(1, random_state=42)
    print(f"\nLabel {label}:\n{sample['Question'].values[0]}")


import re

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\+\-\*/=^()., ]", " ", text)  # keep math symbols
    text = re.sub(r"\s+", " ", text)
    return text.strip()

train_df['cleaned'] = train_df['Question'].apply(preprocess)
test_df['cleaned'] = test_df['Question'].apply(preprocess)


from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import numpy as np

X = train_df['cleaned']
y = train_df['label']

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(X)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_index, val_index in skf.split(X_tfidf, y):
    X_train, X_val = X_tfidf[train_index], X_tfidf[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")  # class imbalance handling
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)

    score = f1_score(y_val, preds, average="micro")
    scores.append(score)

print(f"Average F1-micro: {np.mean(scores):.4f}")


from transformers import AutoTokenizer
import matplotlib.pyplot as plt

# Load BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Compute token lengths
token_lens = train_df['cleaned'].apply(lambda x: len(tokenizer.tokenize(x)))

# Print statistics
print("Token Length Statistics:")
print(token_lens.describe())

# Optional: Print specific percentiles
percentiles = token_lens.quantile([0.9, 0.95, 0.99])
print("\nToken Length Percentiles:")
print(percentiles)

# Plot histogram
token_lens.hist(bins=30)
plt.title("Distribution of Tokenized Input Lengths (BERT tokenizer)")
plt.xlabel("Number of Tokens")
plt.ylabel("Number of Questions")
plt.grid(True)
plt.show()


# !pip install transformers datasets accelerate -q
# !pip install --upgrade transformers


import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import torch
from datasets import Dataset
import random


MODEL_NAME = "bert-base-uncased"  # Replace with other models later
NUM_LABELS = 8
MAX_LENGTH = 256  # Based on your 99th percentile analysis
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(example):
    return tokenizer(
        example["cleaned"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )


# Convert to Hugging Face Dataset:

train_dataset = Dataset.from_pandas(train_df[['cleaned', 'label']])
train_dataset = train_dataset.map(tokenize_function, batched=True)
train_dataset = train_dataset.rename_column("label", "labels")
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from datasets import Dataset
import numpy as np
import os

# Disable WandB if not needed
os.environ["WANDB_DISABLED"] = "true"

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Basic config
MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 8
MAX_LENGTH = 256
SEED = 42

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(example):
    return tokenizer(
        example["cleaned"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

# Convert to HuggingFace dataset
train_df_subset = train_df[['cleaned', 'label']]
hf_dataset = Dataset.from_pandas(train_df_subset)
hf_dataset = hf_dataset.map(tokenize_function, batched=True)
hf_dataset = hf_dataset.rename_column("label", "labels")
hf_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Stratified K-Fold (run one fold as example)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
labels = train_df["label"].values
all_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, labels)):
    print(f"\nðŸŸ¦ Fold {fold + 1} â€” Using GPU: {torch.cuda.is_available()}")

    train_split = hf_dataset.select(train_idx.tolist())
    val_split = hf_dataset.select(val_idx.tolist())

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    ).to(device)  # move to GPU

    args = TrainingArguments(
        output_dir=f"./bert-fold-{fold}",
        do_train=True,
        do_eval=True,
        learning_rate=2e-5,
        per_device_train_batch_size=8,  # P100 safe size
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=f"./logs-{fold}",
        seed=SEED,
        logging_steps=50,
        save_strategy="no"
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {"f1": f1_score(labels, preds, average="micro")}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_split,
        eval_dataset=val_split,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer)
    )

    trainer.train()
    eval_result = trainer.evaluate()
    print(f"Fold {fold + 1} F1-micro: {eval_result['eval_f1']:.4f}")

    all_scores.append(eval_result['eval_f1'])

    break  # Just run one fold for testing


import pandas as pd
import re

# Load test data
test_df = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv")

# Apply same preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\+\-\*/=^()., ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

test_df["cleaned"] = test_df["Question"].apply(preprocess)



from datasets import Dataset

# Convert to Hugging Face Dataset and tokenize
test_dataset = Dataset.from_pandas(test_df[["cleaned"]])
test_dataset = test_dataset.map(tokenize_function, batched=True)
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])


# Ensure model is in eval mode
model.eval()

# Predict
predictions = trainer.predict(test_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=1)


submission = pd.DataFrame({
    "id": test_df.index,
    "label": predicted_labels
})

submission.to_csv("submission.csv", index=False)
print("âœ… submission.csv has been saved!")







