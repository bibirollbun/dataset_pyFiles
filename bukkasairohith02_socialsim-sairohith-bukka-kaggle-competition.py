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


import gc
gc.collect()



DATA_DIR = "/kaggle/input/social-sim-challenge-social-media-based-personas/train"
OUTPUT_PATH = "/kaggle/working/train_dataset.csv"


import os
os.listdir(DATA_DIR)


import os
import json
import ujson
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict


import os, ujson, pandas as pd
from tqdm import tqdm
import gc

DATA_DIR = "/kaggle/input/social-sim-challenge-social-media-based-personas/train"
SAVE_DIR = "./parsed_clusters"
os.makedirs(SAVE_DIR, exist_ok=True)

label_cols = ['like', 'unlike', 'repost', 'unrepost', 'follow', 'unfollow', 'block',
              'unblock', 'post_update', 'post_delete', 'quote', 'post', 'reply']

def extract_from_cluster(filepath):
    rows = []
    with open(filepath, "r") as f:
        for line in f:
            obj = ujson.loads(line)
            cluster_id = obj["cluster_id"]
            thread_text = []
            row = {"cluster_id": cluster_id, "id": obj["id"]}
            for entry in obj["thread"]:
                if "text" in entry:
                    thread_text.append(entry["text"])
                if "actions" in entry:
                    for action in label_cols:
                        row[f"label_{action}"] = entry["actions"].get(action, False)
            row["history_text"] = "\n".join(thread_text)
            rows.append(row)
    return rows

# âœ… Stream and save per-cluster safely
for i in tqdm(range(25)):
    file_path = os.path.join(DATA_DIR, f"cluster_{i}.jsonl")
    cluster_data = extract_from_cluster(file_path)
    df = pd.DataFrame(cluster_data)
    df.to_csv(f"{SAVE_DIR}/cluster_{i}_parsed.csv", index=False)
    del df, cluster_data
    gc.collect()

print("âœ… Parsed and saved all clusters to ./parsed_clusters/")



import glob
dfs = [pd.read_csv(file) for file in glob.glob("./parsed_clusters/*.csv")]
df_all = pd.concat(dfs, ignore_index=True)
df_all.to_csv("final_labels_with_history.csv", index=False)
print("âœ… final_labels_with_history.csv saved. Shape:", df_all.shape)



import glob
import pandas as pd

csv_files = glob.glob("./parsed_clusters/*.csv")
dfs = [pd.read_csv(f) for f in csv_files]
df_all = pd.concat(dfs, ignore_index=True)
print(df_all.shape)
print(df_all.head())



import matplotlib.pyplot as plt
import seaborn as sns

# 1. Distribution of all 13 actions
label_cols = [col for col in df_all.columns if col.startswith("label_")]
action_counts = df_all[label_cols].sum().sort_values(ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(x=action_counts.values, y=action_counts.index, palette="Blues_r")
plt.title("Action Frequency Distribution")
plt.xlabel("Count")
plt.ylabel("Action")
plt.grid(axis="x")
plt.show()


from sklearn.metrics import multilabel_confusion_matrix
import numpy as np

# Compute correlation matrix
co_matrix = df_all[label_cols].T.dot(df_all[label_cols])
np.fill_diagonal(co_matrix.values, 0)  # remove self-cooccurrence

plt.figure(figsize=(10, 8))
sns.heatmap(co_matrix, annot=True, fmt="d", cmap="coolwarm", square=True)
plt.title("Action Co-occurrence Heatmap")
plt.show()



df_all["history_len"] = df_all["history_text"].str.split().apply(len)

plt.figure(figsize=(10, 4))
sns.histplot(df_all["history_len"], bins=100, kde=True, color="teal")
plt.title("Distribution of History Text Length (in words)")
plt.xlabel("Number of words")
plt.ylabel("Count")
plt.show()



df_all["label_sum"] = df_all[label_cols].sum(axis=1)

plt.figure(figsize=(8,4))
sns.countplot(x="label_sum", data=df_all, palette="viridis")
plt.title("How Many Actions Co-occur?")
plt.xlabel("Number of Actions (per row)")
plt.ylabel("Count")
plt.show()

df_all["label_sum"].value_counts(normalize=True)



action_avg_lengths = {
    action: df_all[df_all[action]]["history_len"].mean()
    for action in label_cols
}
pd.Series(action_avg_lengths).sort_values(ascending=False).plot(kind="barh", figsize=(10,6), color='orange')
plt.title("Average History Length by Action Type")
plt.xlabel("Avg # of Words in History")
plt.ylabel("Action Type")
plt.show()



# Count labels by cluster
cluster_action_counts = df_all.groupby("cluster_id")[label_cols].sum()

plt.figure(figsize=(14, 8))
sns.heatmap(cluster_action_counts, annot=False, cmap="YlGnBu", linewidths=0.3)
plt.title("Cluster-wise Action Frequency")
plt.xlabel("Action")
plt.ylabel("Persona Cluster")
plt.show()



import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import time


X = df_all["history_text"]
y = df_all[label_cols]

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



# STEP 1: Keep only samples with at least one positive label
mask = y_train[label_cols].sum(axis=1) > 0
X_pos = X_train[mask]
y_pos = y_train[mask]

# STEP 2: Take first 20000 rows
X_train_sub = X_pos[:20000]
y_train_sub = y_pos.iloc[:20000]

# STEP 3: Drop label columns that still have only one class (e.g., all 0s)
label_cols_sub = [col for col in label_cols if len(y_train_sub[col].unique()) > 1]
y_train_sub = y_train_sub[label_cols_sub]

# STEP 4: Vectorize text
tfidf = TfidfVectorizer(max_features=20000, stop_words="english")
X_train_tfidf_sub = tfidf.fit_transform(X_train_sub)
X_val_tfidf = tfidf.transform(X_val)

# STEP 5: Fit model
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
import time

base_model = LogisticRegression(class_weight="balanced", solver='liblinear', max_iter=1000)
multi_model = MultiOutputClassifier(base_model)

start = time.time()
multi_model.fit(X_train_tfidf_sub, y_train_sub)
print(f"âœ… Subset training completed in {(time.time() - start)/60:.2f} minutes")



from sklearn.metrics import f1_score

# Predict on validation set
y_pred = multi_model.predict(X_val_tfidf)

# Restrict y_val to the same label columns used in training
y_val_filtered = y_val[label_cols_sub]

# Compute F1 scores
print("ğŸ“Š Per-label F1 Scores:")
for i, col in enumerate(label_cols_sub):
    f1 = f1_score(y_val_filtered[col], y_pred[:, i], average='binary', zero_division=0)
    print(f"{col:20s}: F1 = {f1:.4f}")

# Overall macro F1
overall_macro_f1 = f1_score(y_val_filtered, y_pred, average='macro', zero_division=0)
print(f"\nğŸ”¥ Overall Macro F1 Score: {overall_macro_f1:.4f}")



!pip install -q sentence-transformers

from sentence_transformers import SentenceTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time


df_all = pd.read_csv("final_labels_with_history.csv")  # or whatever file you saved
label_cols = [col for col in df_all.columns if col.startswith("label_")]

X = df_all["history_text"]
y = df_all[label_cols]

# Train/Val Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # Initialize model



# Filter samples with sufficient history length
min_length = 300
mask = df_all['history_text'].str.len() > min_length

X = df_all.loc[mask, 'history_text']
y = df_all.loc[mask, label_cols]

# Split train/val
# (Assuming you have a split, else do train_test_split here)

# Subset training data to 20k samples for faster prototyping
X_train_sub = X_train[:20000]
y_train_sub = y_train.iloc[:20000]

# Filter labels with >=2 classes in train subset
label_cols_sub = [col for col in y_train_sub.columns if len(y_train_sub[col].unique()) > 1]
y_train_sub_filtered = y_train_sub[label_cols_sub]

# Use filtered labels for training and evaluation



MAX_WORDS = 300
df_all["history_text"] = df_all["history_text"].apply(lambda x: " ".join(x.split()[:MAX_WORDS]))


!pip install -q sentence-transformers

from sentence_transformers import SentenceTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time



from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer

# âœ‚ï¸� Step 1: Filter samples with long enough history (length > 300)
df_filtered = df_all[df_all["history_text"].str.len() > 300].reset_index(drop=True)

# ğŸ§¾ Step 2: Split X and y
X = df_filtered["history_text"]
y = df_filtered[[col for col in df_filtered.columns if col.startswith("label_")]]

# âœ‚ï¸� Step 3: Subset first 20k samples for train and 4k for val
X_train = X[:20000]
y_train = y.iloc[:20000]
X_val = X[20000:24000]
y_val = y.iloc[20000:24000]

# ğŸ§¹ Step 4: Remove labels that are all one class (e.g., all False) in train
label_cols_sub = [col for col in y_train.columns if len(y_train[col].unique()) > 1]
y_train = y_train[label_cols_sub]
y_val = y_val[label_cols_sub]

# ğŸ’  Step 5: Embed text
model = SentenceTransformer('all-MiniLM-L6-v2')
X_train_embeds = model.encode(X_train.tolist(), batch_size=64, show_progress_bar=True)
X_val_embeds = model.encode(X_val.tolist(), batch_size=64, show_progress_bar=True)

# ğŸ¤– Step 6: Train and predict
base_model = LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=1000)
multi_model = MultiOutputClassifier(base_model)

multi_model.fit(X_train_embeds, y_train)
y_pred = multi_model.predict(X_val_embeds)

print("ğŸ“Š Per-label F1 Scores:")
for i, col in enumerate(label_cols_sub):  # Use only trained labels
    f1 = f1_score(y_val[col], y_pred[:, i], average='binary', zero_division=0)
    print(f"{col:20s}: F1 = {f1:.4f}")


macro_f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
print(f"\nğŸ”¥ Macro F1 Score (subset): {macro_f1:.4f}")



!pip install -q transformers datasets

import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset



# df_filtered is already loaded if you're continuing; if not, reload it:
# df_filtered = pd.read_csv("final_labels_with_history.csv")
# df_filtered = df_filtered[df_filtered["history_text"].str.len() > 300].reset_index(drop=True)

df_gen = df_filtered[df_filtered["label_post"] == True][["history_text"]].copy()
df_gen["prompt"] = "Given the following social history, generate a response:\n" + df_gen["history_text"]

def format_prompt(example):
    return {
        "input_text": f"[Persona: {example['persona']}] [Context: {example['context']}] =>",
        "label_text": example['post_text']
    }



from transformers import DataCollatorForLanguageModeling
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False  # Important: we are NOT doing masked language modeling (like BERT)
)



tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(
        examples["prompt"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

dataset = Dataset.from_pandas(df_gen[["prompt"]])
tokenized_dataset = dataset.map(tokenize_function, batched=True)


from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("distilgpt2")
model.resize_token_embeddings(len(tokenizer))

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)



import re

def clean_text(text):
    text = re.sub(r"<.*?>", "", text)  # remove HTML
    text = re.sub(r"http\S+", "", text)  # remove URLs
    return text.strip()



print(type(tokenized_dataset))  # Should print: <class 'datasets.arrow_dataset.Dataset'>
print(tokenized_dataset.column_names)  # Should include 'input_ids', 'attention_mask', etc.



from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./distilgpt2-socialsim",     
    overwrite_output_dir=True,               
    per_device_train_batch_size=4,          
    num_train_epochs=3,                     
    max_steps=1,                          
    logging_steps=10,                       
    save_steps=50,                         
    save_total_limit=2,                      
    fp16=True,                               
    disable_tqdm=False,                      
    report_to=[],                        
    run_name="test-socialsim-gpt2"           
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,  # âœ… fix here
    tokenizer=tokenizer,
    data_collator=data_collator,
)



trainer.train()


def generate_post(persona, context, max_length=100):
    input_prompt = f"[Persona: {persona}] [Context: {context}] =>"
    inputs = tokenizer(input_prompt, return_tensors="pt").input_ids
    output = model.generate(
        inputs, 
        max_length=max_length, 
        num_return_sequences=1, 
        do_sample=True, 
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Remove the prompt from output to get only the generated post
    return generated.replace(input_prompt, "").strip()



print(generate_post("Psychologist", "Feeling lonely during lockdown"))



# ğŸ“¦ Install if needed
# !pip install datasets transformers --quiet

# ğŸ“š Imports
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import re

# ğŸ§¼ Clean text function
def clean_text(text):
    if text is None:
        return ""
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# âœ�ï¸� Format prompt function
def format_prompt(example):
    cleaned_post = clean_text(example.get("post_text", ""))
    return {
        "prompt": f"Persona: {example.get('persona_label', '')}\nContext: {example.get('history_text', '')}\nPost:",
        "post_text": cleaned_post
    }

# ğŸ”¢ Load and sample dataset (dev mode = 50k)
raw_dataset = load_dataset("csv", data_files="final_labels_with_history.csv")["train"]
raw_dataset = raw_dataset.shuffle(seed=42).select(range(500))  # ğŸ‘ˆ Sample for dev mode

# ğŸ§¼ Format prompts (multi-processing)
formatted_dataset = raw_dataset.map(format_prompt, num_proc=4, desc="Formatting")

# ğŸ”  Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

# âœ‚ï¸� Tokenization function
def tokenize_fn(examples):
    full_texts = [p + " " + t for p, t in zip(examples["prompt"], examples["post_text"])]
    tokenized = tokenizer(
        full_texts,
        truncation=True,
        padding="max_length",
        max_length=64  # ğŸ‘ˆ shorter seq length for speed
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# ğŸ”„ Tokenize dataset (batched + parallel)
tokenized_dataset = formatted_dataset.map(tokenize_fn, batched=True, num_proc=4, desc="Tokenizing")

# ğŸ§¹ Keep only necessary columns
columns_to_keep = ["input_ids", "attention_mask", "labels"]
tokenized_dataset = tokenized_dataset.remove_columns([col for col in tokenized_dataset.column_names if col not in columns_to_keep])

# ğŸ¤– Load model
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# ğŸ“¦ Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# âš™ï¸� Training arguments
training_args = TrainingArguments(
    output_dir="./distilgpt2-socialsim",
    overwrite_output_dir=True,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    fp16=True,
    report_to=[],
    run_name="socialsim-gpt2-dev"
)

# ğŸš‚ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# ğŸš€ Train the model
trainer.train()

# âœ¨ Inference function (optimized)
def generate_post(persona, context, max_length=100):
    input_prompt = f"Persona: {persona}\nContext: {context}\nPost:"
    inputs = tokenizer(input_prompt, return_tensors="pt").input_ids
    output = model.generate(
        inputs,
        max_length=max_length,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text.replace(input_prompt, "").strip()

# ğŸ”� Try a generation
print(generate_post("Psychologist", "Feeling lonely during lockdown"))



import os
print(os.listdir("/kaggle/input/social-sim-challenge-social-media-based-personas/test"))



import os
import json
import pandas as pd

# âœ… 1. Define the test directory path
test_dir = '/kaggle/input/social-sim-challenge-social-media-based-personas/test/'

# âœ… 2. Collect all .jsonl file paths
jsonl_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.jsonl')]

# âœ… 3. Load all threads into a list
rows = []
for file in jsonl_files:
    with open(file, 'r') as f:
        for line in f:
            row = json.loads(line)
            rows.append({
                "id": row.get("id"),
                "cluster_id": row.get("cluster_id"),
                "persona_label": row.get("persona", {}).get("persona_label", "User"),
                "history_text": row.get("thread", [])
            })

# âœ… 4. Convert to DataFrame
test_df = pd.DataFrame(rows)

# âœ… 5. Flatten thread history to plain text (used for generation/classification)
def flatten_thread(thread):
    return " ".join([t["text"] for t in thread if "text" in t])

test_df["history_text"] = test_df["history_text"].apply(flatten_thread)

# âœ… 6. Final Preview
test_df.head()



import os
print(os.listdir("/kaggle/working/"))



from sentence_transformers import SentenceTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np

# 1. Extract texts and labels
texts = df_all['history_text'].tolist()
label_columns = [
    'label_like', 'label_unlike', 'label_repost', 'label_unrepost',
    'label_follow', 'label_unfollow', 'label_block', 'label_unblock',
    'label_post_update', 'label_post_delete', 'label_quote', 'label_post',
    'label_reply'
]
labels = df_all[label_columns].values

# 2. Subsample first 5000 examples
texts_small = texts[:5000]
labels_small = labels[:5000]

# 3. Filter label columns with only one class
label_columns_filtered = [
    label_columns[i]
    for i in range(labels_small.shape[1])
    if len(np.unique(labels_small[:, i])) > 1
]
labels_small_filtered = df_all[label_columns_filtered].values[:5000]

# 4. Embed texts
embedder = SentenceTransformer('all-MiniLM-L6-v2')
X_small = embedder.encode(
    texts_small,
    batch_size=512,
    convert_to_numpy=True,
    show_progress_bar=True
)

# 5. Train classifier
clf = MultiOutputClassifier(LogisticRegression(max_iter=1000))
clf.fit(X_small, labels_small_filtered)

# 6. Save model
joblib.dump(clf, "classifier_small.pkl")
print("âœ… Model trained and saved as classifier_small.pkl")
print("âœ… Used labels:", label_columns_filtered)



import joblib
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# âœ… Auto-detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load classifier
clf = joblib.load("/kaggle/working/classifier_small.pkl")  # your local path

# Load GPT2 generator
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("distilgpt2")
model_gpt2 = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)
model_gpt2.eval()



import pandas as pd
import os
import json

# Path to the test folder
test_dir = "/kaggle/input/social-sim-challenge-social-media-based-personas/test/"

# List all .jsonl files in the folder
files = [f for f in os.listdir(test_dir) if f.endswith(".jsonl")]

# Parse them all into one DataFrame
test_df_list = []
for file in files:
    with open(os.path.join(test_dir, file), 'r') as f:
        for line in f:
            test_df_list.append(json.loads(line))

test_df = pd.DataFrame(test_df_list)
print("âœ… Combined test rows:", test_df.shape[0])
print("âœ… Columns:", test_df.columns.tolist())



import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
from tqdm import tqdm
import os

# === Setup ===
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token  # Fix padding issue
tokenizer.padding_side = "left"
model = GPT2LMHeadModel.from_pretrained("distilgpt2")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# === Load your test data ===
test_df = pd.read_csv("/kaggle/working/final_labels_with_history.csv")

# === Select action ===
actions = ["label_post"]  # You can extend this list later
batch_size = 32
max_rows = 10000  # Use top 10k rows for quick testing

for act in actions:
    print(f"\nProcessing: {act}")

    # Create output column with default 'EMPTY'
    test_df[act] = "EMPTY"

    # Filter only True rows
    active_rows = test_df[test_df[act] == True].head(max_rows).copy()

    # Store generated results
    generated_texts = []
    indices = active_rows.index.tolist()
    texts_to_generate = active_rows["history_text"].tolist()

    # === Generation Function ===
    def generate_texts(texts, max_length=130):
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Generate in batches
    for i in tqdm(range(0, len(texts_to_generate), batch_size), desc=f"Generating for {act}"):
        batch_texts = texts_to_generate[i:i + batch_size]
        batch_outputs = generate_texts(batch_texts)
        generated_texts.extend(batch_outputs)

    # Update test_df with generated texts
    for idx, gen_text in zip(indices, generated_texts):
        test_df.at[idx, act] = gen_text

    # Save partial result
    part_file = f"submission_{act}_part.csv"
    test_df[["id", "cluster_id", act]].to_csv(part_file, index=False)
    print(f"Saved partial result to: {part_file}")

# === Final Merge ===
print("\nMerging final submission file...")
final_df = test_df[["id", "cluster_id"] + actions]
final_df.to_csv("final_submission.csv", index=False)
print("Final submission saved as 'final_submission.csv'")



import pandas as pd

df = pd.read_csv('final_submission.csv')

print(df.shape)  # Should match test dataset rows

print(df.columns)  # Check columns

print(df.head())  # Sample data

# Check for missing or null values
print(df.isnull().sum())



# Read the file line by line to inspect its raw content
submission_filename = 'submission.csv'

print(f"\n--- Raw content of '{submission_filename}' (first 10 lines) ---")
try:
    with open(submission_filename, 'r') as f:
        for i, line in enumerate(f):
            print(line.strip()) # .strip() removes newline characters
            if i >= 9: # Print only first 10 lines (header + 9 data rows)
                break
except FileNotFoundError:
    print(f"Error: The file '{submission_filename}' was not found.")
    print("Please ensure you have run the code to generate the submission.csv file in your notebook environment first.")
except Exception as e:
    print(f"An error occurred while reading the file: {e}")


import csv # Import the csv module at the top of your script

# ... (your existing code to create final_submission_df,
# including the .apply(lambda x: "True" if x else "False") for action columns) ...

# Save the DataFrame to a CSV file for submission
output_filename = 'submission.csv'
final_submission_df.to_csv(output_filename, index=False, quoting=csv.QUOTE_NONNUMERIC)
print(f"\nFinal submission saved as '{output_filename}' with proper quoting.")


import csv # Make sure this is at the top of your script

# ... (your code to create final_submission_df) ...

final_submission_df.to_csv(output_filename, index=False, quoting=csv.QUOTE_NONNUMERIC)




