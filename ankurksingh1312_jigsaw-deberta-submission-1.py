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


pd.set_option('display.max_colwidth', None)
df_test = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
df_test


# for training 

# -----------------------------
# 3. Augment function (with subreddit prefix in rules)
# -----------------------------
def format_rule_with_subreddit(row):
    """Helper to create rule string with subreddit context"""
    return f"Discussion on subreddit topic {row['subreddit']} and rules are {row['rule']}"

def augment_test_rows(row):
    rows = []
    rule_with_subreddit = format_rule_with_subreddit(row)

    for col, label in [
        ('positive_example_1', 1), 
        ('positive_example_2', 1),
        ('negative_example_1', 0), 
        ('negative_example_2', 0)
    ]:
        if pd.notna(row[col]):
            rows.append({"rule": rule_with_subreddit, "body": row[col], "rule_violation": label})

    return rows


# -----------------------------
# Apply augmentation
# -----------------------------
augmented_test = [r for _, row in df_test.iterrows() for r in augment_test_rows(row)]

augmented_df = pd.DataFrame(augmented_test)
print("Augmented dataset size:", augmented_df.shape)


augmented_df


# -----------------------------
# 2. URL keyword extraction (with domain/path semantics)
# -----------------------------
import re

def keywords_from_url(text: str) -> str:
    if not isinstance(text, str):
        return ""

    url_pattern = r'https?://[^\s/$.?#].[^\s]*'
    urls = re.findall(url_pattern, text)
    
    if not urls:
        return "" 

    all_semantics = []
    seen_semantics = set()

    for url in urls:
        url_lower = url.lower()
        
        # 1. Extract domain
        domain_match = re.search(r"(?:https?://)?([a-z0-9\-\.]+)\.[a-z]{2,}", url_lower)
        if domain_match:
            full_domain = domain_match.group(1)
            parts = full_domain.split('.')
            for part in parts:
                if part and part not in seen_semantics and len(part) > 3:  # Avoid 'www'
                    all_semantics.append(f"domain:{part}")
                    seen_semantics.add(part)

        # 2. Extract path parts
        path = re.sub(r"^(?:https?://)?[a-z0-9\.-]+\.[a-z]{2,}/?", "", url_lower)
        path_parts = [p for p in re.split(r'[/_.-]+', path) if p and p.isalnum()]

        for part in path_parts:
            # Clean file extensions / queries / fragments
            part_clean = re.sub(r"\.(html?|php|asp|jsp)$|#.*|\?.*", "", part)
            if part_clean and part_clean not in seen_semantics and len(part_clean) > 3:
                all_semantics.append(f"path:{part_clean}")
                seen_semantics.add(part_clean)

    return " ".join(all_semantics) if all_semantics else ""


def append_url_keywords(row):
    kw = keywords_from_url(row["body"])  # Extract URL keywords
    if kw:                             # If there are keywords
        return row["body"] + " " + kw  # Append them
    else:                               # If no keywords found
        return row["body"]  


# Apply to train & test
augmented_df["body"] = augmented_df.apply(append_url_keywords, axis=1)
augmented_df


import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification


tokenizer = AutoTokenizer.from_pretrained('/kaggle/input/deberta_modified_url_cleaned/tensorflow2/default/1/jigsaw_modified_DeBERT_url_cleaning_added')
model = TFAutoModelForSequenceClassification.from_pretrained('/kaggle/input/deberta_modified_url_cleaned/tensorflow2/default/1/jigsaw_modified_DeBERT_url_cleaning_added')


MAX_LEN = 512

# -----------------------------
# 3. Tokenize data
# -----------------------------
encodings = tokenizer(
    list(augmented_df['rule']),
    list(augmented_df['body']),
    truncation=True,
    padding="max_length",
    max_length=MAX_LEN,
    return_tensors="tf"
)

train_dataset = tf.data.Dataset.from_tensor_slices((
    {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"]
    },
    tf.convert_to_tensor(augmented_df["rule_violation"].values, dtype=tf.float32)
)).batch(2)

# -----------------------------
# 4. Compile model
# -----------------------------
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = [tf.keras.metrics.BinaryAccuracy(name="accuracy")]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


h = model.fit(train_dataset, epochs=2)





#predition preparation

# no training block


def format_rule_with_subreddit(row):
    """Helper to create rule string with subreddit context"""
    return f"Discussion on subreddit topic {row['subreddit']} and rules are {row['rule']}"


def augment_test_rows(row):
    rows = []
    rule_with_subreddit = format_rule_with_subreddit(row)
    rows.append({"rule": rule_with_subreddit, "body": row["body"]})
    return rows



augmented_test = [r for _, row in df_test.iterrows() for r in augment_test_rows(row)]
df_test_predition = pd.DataFrame( augmented_test)

df_test_predition["body"] = df_test_predition.apply(append_url_keywords, axis=1)
df_test_predition


MAX_LEN=512
test_encodings = tokenizer(
    list(df_test_predition['rule']),
    list(df_test_predition['body']),
    truncation=True,
    padding="max_length",
    max_length=MAX_LEN,
    return_tensors="tf"
)

preds = model.predict({
    "input_ids": test_encodings["input_ids"],
    "attention_mask": test_encodings["attention_mask"]
})

# Convert logits to probabilities
probs = tf.sigmoid(preds.logits).numpy().flatten()
print("Predicted probabilities:", probs)


submission = pd.DataFrame({
    "row_id": df_test["row_id"],
    "rule_violation": probs
})

submission.to_csv("submission.csv", index=False)
print("Saved predictions to submissions.csv")

