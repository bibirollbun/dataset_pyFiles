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
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import average_precision_score

# --- FIX FOR P100 GPU (CUDA Capability < 7.0) ---
import torch._dynamo
torch._dynamo.config.suppress_errors = True
# --- END FIX ---

# --- Configuration ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
VER = 1
MODEL_NAME = "jhu-clsp/ettin-encoder-400m"
EPOCHS = 3
DIR = f"ver_{VER}_ettin"
os.makedirs(DIR, exist_ok=True)
MAX_LEN = 256

# --- Data Loading and Initial Preprocessing ---
train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')

# Handle 'Misconception' NA values
train.Misconception = train.Misconception.fillna('NA')

# Create target and label
train['target'] = train.Category + ":" + train.Misconception
le = LabelEncoder()
train['label'] = le.fit_transform(train['target'])
n_classes = len(le.classes_)

print(f"Train shape: {train.shape} with {n_classes} target classes")
print("\nTrain Head:")
print(train.head())

# --- Determine Correct Answers ---
idx = train.apply(lambda row: row.Category.split('_')[0], axis=1) == 'True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId', 'MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c', ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId', 'MC_Answer']]
correct['is_correct'] = 1
train = train.merge(correct, on=['QuestionId', 'MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)
print("\nTrain Head after is_correct merge:")
print(train.head())

# --- Exploratory Data Analysis (EDA) ---
print("\n--- Exploratory Data Analysis ---")

# 1. Distribution of Categories
print("\nDistribution of Categories:")
print(train['Category'].value_counts())
plt.figure(figsize=(10, 6))
train['Category'].value_counts().plot(kind='bar')
plt.title('Distribution of Categories')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 2. Distribution of Misconceptions (for non-NA)
print("\nDistribution of Misconceptions (excluding NA):")
misconceptions_counts = train[train['Misconception'] != 'NA']['Misconception'].value_counts()
print(misconceptions_counts)
plt.figure(figsize=(12, 7))
misconceptions_counts.head(20).plot(kind='bar')
plt.title('Top 20 Misconceptions (excluding NA)')
plt.xlabel('Misconception')
plt.ylabel('Count')
plt.xticks(rotation=90, ha='right')
plt.tight_layout()
plt.show()

# 3. Relationship between Category and is_correct
print("\nRelationship between Category and is_correct:")
print(pd.crosstab(train['Category'], train['is_correct'], normalize='index'))

# 4. Length of QuestionText and StudentExplanation
train['QuestionText_len'] = train['QuestionText'].apply(len)
train['StudentExplanation_len'] = train['StudentExplanation'].apply(len)

print("\nQuestionText Length Statistics:")
print(train['QuestionText_len'].describe())
print("\nStudentExplanation Length Statistics:")
print(train['StudentExplanation_len'].describe())

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.hist(train['QuestionText_len'], bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of QuestionText Length')
plt.xlabel('Length')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(train['StudentExplanation_len'], bins=50, color='lightcoral', edgecolor='black')
plt.title('Distribution of StudentExplanation Length')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# --- Text Formatting and Tokenization ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def format_input(row):
    x = "This answer is correct."
    if not row['is_correct']:
        x = "This is answer is incorrect."
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"{x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )

train['text'] = train.apply(format_input, axis=1)
print("\nExample prompt for our LLM:")
print(train.text.values[0])

# Token Length Distribution (re-evaluate with new tokenizer if it's different)
lengths = [len(tokenizer.encode(t, truncation=False)) for t in train["text"]]
plt.figure(figsize=(10, 6))
plt.hist(lengths, bins=50)
plt.title("Token Length Distribution (after new tokenizer)")
plt.xlabel("Number of tokens")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

L = (np.array(lengths) > MAX_LEN).sum()
print(f"There are {L} train sample(s) with more than {MAX_LEN} tokens after tokenizing with {MODEL_NAME}")
print("Sorted token lengths (last 50):")
print(np.sort(lengths)[-50:])

# --- Dataset Preparation ---
train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)

COLS = ['text', 'label']
train_ds = Dataset.from_pandas(train_df[COLS])
val_ds = Dataset.from_pandas(val_df[COLS])

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LEN)

print("\nTokenizing training dataset...")
train_ds = train_ds.map(tokenize, batched=True, num_proc=os.cpu_count())
print("Tokenizing validation dataset...")
val_ds = val_ds.map(tokenize, batched=True, num_proc=os.cpu_count())

columns = ['input_ids', 'attention_mask', 'label']
train_ds.set_format(type='torch', columns=columns)
val_ds.set_format(type='torch', columns=columns)

# --- Model Loading ---
print(f"\nLoading model: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=n_classes,
    ignore_mismatched_sizes=True
)
print("Model loaded. New classification head initialized.")

# --- Custom Metric for MAP@3 ---
def compute_map3(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

    top3 = np.argsort(-probs, axis=1)[:, :3]
    match = (top3 == labels[:, None])

    map3 = 0
    for i in range(len(labels)):
        if match[i, 0]:
            map3 += 1.0
        elif match[i, 1]:
            map3 += 1.0 / 2
        elif match[i, 2]:
            map3 += 1.0 / 3
    return {"map@3": map3 / len(labels)}

# --- Training Arguments ---
training_args = TrainingArguments(
    output_dir=f"./{DIR}",
    do_train=True,
    do_eval=True,
    eval_strategy="steps",
    save_strategy="steps",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    logging_dir="./logs",
    logging_steps=50,
    save_steps=200,
    eval_steps=200,
    save_total_limit=1,
    metric_for_best_model="map@3",
    greater_is_better=True,
    load_best_model_at_end=True,
    report_to="none",
    fp16=True,
)

# --- Trainer Initialization and Training ---
print("\nInitializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_map3,
)

print("\nStarting training...")
trainer.train()
print("\nTraining complete.")

# --- Save Model and Tokenizer ---
print("\nSaving model and tokenizer...")
trainer.save_model(f"./{DIR}/final_model")
tokenizer.save_pretrained(f"./{DIR}/final_model")
print(f"Model and tokenizer saved to ./{DIR}/final_model")




