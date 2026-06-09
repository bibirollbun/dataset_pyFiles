# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

# ================================================================
# 1. CONFIGURATION
# ================================================================
MODEL_NAME = "/kaggle/input/databerta/kaggle/working/deberta-v3-base-local"
KAGGLE_DATA_PATH = "/kaggle/input/map-charting-student-math-misunderstandings/"

MAX_LEN = 512
SEED = 42
TRAIN_BATCH_SIZE = 8
EPOCHS = 3
TOP_K = 3 # This is the 'k' for MAP@k

torch.manual_seed(SEED)
np.random.seed(SEED)
USE_GPU = torch.cuda.is_available()

# ================================================================
# 2. DATA LOADING & PREPROCESSING
# ================================================================
try:
    train_df = pd.read_csv(os.path.join(KAGGLE_DATA_PATH, "train.csv"))
    test_df = pd.read_csv(os.path.join(KAGGLE_DATA_PATH, "test.csv"))
except FileNotFoundError:
    print("Dataset files not found. Please check the KAGGLE_DATA_PATH.")
    exit()

# Handle missing columns safely
for col in ["Category", "Misconception", "QuestionText", "MC_Answer", "StudentExplanation"]:
    if col not in train_df.columns:
        train_df[col] = ""
    if col not in test_df.columns:
        test_df[col] = ""

train_df["Misconception"] = train_df["Misconception"].fillna("NA")
train_df["Category"] = train_df["Category"].fillna("Unknown")

train_df["Target"] = train_df["Category"] + ":" + train_df["Misconception"]
# Note: The original code assumes a single label per row. We keep this
# as the train.csv data format also seems to have a single ground truth.
train_targets_lists = [[label] for label in train_df["Target"].tolist()]

mlb = MultiLabelBinarizer()
Y_train = mlb.fit_transform(train_targets_lists).astype(np.float32)
all_labels = mlb.classes_
num_labels = len(all_labels)

# Define safe text formatter
def format_input(row):
    q = str(row.get("QuestionText", ""))
    a = str(row.get("MC_Answer", ""))
    e = str(row.get("StudentExplanation", ""))
    return f"Question: {q}\nAnswer: {a}\nExplanation: {e}"

train_df["input_text"] = train_df.apply(format_input, axis=1)
test_df["input_text"] = test_df.apply(format_input, axis=1)

X_train = train_df["input_text"].tolist()
X_test = test_df["input_text"].tolist()

# ================================================================
# 3. TOKENIZATION
# ================================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def safe_tokenize(texts):
    texts = [str(t) if pd.notna(t) else "" for t in texts]
    return tokenizer(
        texts, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt"
    )

raw_train_encodings = safe_tokenize(X_train)
raw_test_encodings = safe_tokenize(X_test)

# ================================================================
# 4. DATASET CLASS & METRICS
# ================================================================
class MisconceptionDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: v[idx].clone().detach() for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

train_dataset = MisconceptionDataset(raw_train_encodings, Y_train)
test_dataset = MisconceptionDataset(raw_test_encodings)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def map_at_k(y_true, y_pred, k=3):
    """
    Computes the Mean Average Precision at k (MAP@k) for a single sample.
    Assumes one ground-truth label per sample.
    """
    y_true_indices = np.where(y_true == 1)[0]
    if not len(y_true_indices):
        return 0.0

    relevant_labels = set(y_true_indices)
    score = 0.0
    num_hits = 0.0
    
    for i, p_idx in enumerate(y_pred):
        if p_idx in relevant_labels:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
            relevant_labels.remove(p_idx) # Only score each correct label once
    
    return score / min(len(y_true_indices), k)

def compute_metrics(p):
    logits = p.predictions
    probas = torch.sigmoid(torch.tensor(logits)).numpy()
    
    # Get the top K predictions for each sample
    predictions_indices = np.argsort(probas, axis=1)[:, ::-1]
    
    # Calculate MAP@3 for each sample and then average
    map_scores = [map_at_k(y_true, y_pred, k=TOP_K) for y_true, y_pred in zip(p.label_ids, predictions_indices)]
    
    return {"map_at_3": np.mean(map_scores)}

# ================================================================
# 5. MODEL & TRAINER
# ================================================================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    problem_type="multi_label_classification"
)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=TRAIN_BATCH_SIZE * 2,
    warmup_steps=100,
    weight_decay=0.01,
    learning_rate=2e-5,
    logging_dir="./logs",
    logging_steps=100,
    save_strategy="epoch",
    report_to="none",
    fp16=USE_GPU
    # evaluation_strategy="steps",
    # eval_steps=500
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ================================================================
# 6. TRAINING & SUBMISSION
# ================================================================
print("\n--- Starting Model Training ---")
trainer.train()
print("✅ Training Complete.")

# Prediction
pred_output = trainer.predict(test_dataset)
raw_predictions = pred_output.predictions

probabilities = torch.sigmoid(torch.tensor(raw_predictions)).numpy()
final_predictions = []

for row_probs in probabilities:
    top_indices = np.argsort(row_probs)[::-1][:TOP_K]
    top_labels = [all_labels[i] for i in top_indices]
    final_predictions.append(" ".join(top_labels))

if "row_id" not in test_df.columns:
    test_df["row_id"] = range(len(test_df))

submission_df = pd.DataFrame({
    "row_id": test_df["row_id"],
    "Category:Misconception": final_predictions
})

submission_df.to_csv("submission.csv", index=False)
print("\n✅ Submission saved as submission.csv")
print(submission_df.head())

