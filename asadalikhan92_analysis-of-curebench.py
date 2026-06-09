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


import json
import pandas as pd
import matplotlib.pyplot as plt

# ======================
# File paths
# ======================
val_file = "/kaggle/input/cure-bench-internal-reasoning/curebench_valset_pharse1.jsonl"
test_file = "/kaggle/input/cure-bench-internal-reasoning/curebench_testset_phase1.jsonl"

# ======================
# Load function
# ======================
def load_jsonl(file_path):
    """Load a .jsonl file into a list of dictionaries"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

# ======================
# Load datasets
# ======================
val_data = load_jsonl(val_file)
test_data = load_jsonl(test_file)

val_df = pd.DataFrame(val_data)
test_df = pd.DataFrame(test_data)

# ======================
# Dataset summary
# ======================
def dataset_summary(df, name="Dataset"):
    print(f"\n--- {name} Summary ---")
    
    # Shape
    print(f"Shape: {df.shape}")
    
    # Missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    # Unique values per column (skip dicts)
    print("\nUnique values per column:")
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, dict)).any():
            print(f"{col}: contains dict objects (skipping detailed count)")
        else:
            print(f"{col}: {df[col].nunique()} unique values")
    
    # Data types
    print("\nData types:")
    print(df.dtypes)
    
    # Sample preview
    print(f"\n{name} sample:")
    print(df.head(3))
    
    # Sample text columns
    text_cols = [col for col in df.columns if df[col].dtype == 'object']
    for col in text_cols[:2]:
        print(f"\nSample values from '{col}':")
        print(df[col].head(5).tolist())

# ======================
# Run summaries
# ======================
dataset_summary(val_df, "Validation Set")
dataset_summary(test_df, "Test Set")

# ======================
# Optional: Flatten options dict into separate columns
# ======================
def flatten_options(df):
    """Flatten 'options' column into separate option_A, option_B..."""
    if "options" in df.columns:
        options_expanded = df["options"].apply(pd.Series)
        options_expanded.columns = [f"option_{col}" for col in options_expanded.columns]
        df = pd.concat([df.drop(columns=["options"]), options_expanded], axis=1)
    return df

val_df_flat = flatten_options(val_df)
test_df_flat = flatten_options(test_df)

# ======================
# Save previews
# ======================
val_df_flat.to_csv("curebench_valset_preview.csv", index=False)
test_df_flat.to_csv("curebench_testset_preview.csv", index=False)

print("\nCSV previews saved: 'curebench_valset_preview.csv', 'curebench_testset_preview.csv'")

# ======================
# Visualization Section
# ======================

# 1. Distribution of question types
plt.figure(figsize=(6,4))
val_df["question_type"].value_counts().plot(kind="bar", title="Validation Set - Question Types")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6,4))
test_df["question_type"].value_counts().plot(kind="bar", title="Test Set - Question Types")
plt.ylabel("Count")
plt.show()

# 2. Distribution of correct answers (only validation set)
if "correct_answer" in val_df.columns:
    plt.figure(figsize=(6,4))
    val_df["correct_answer"].value_counts().plot(kind="bar", title="Validation Set - Correct Answers")
    plt.ylabel("Count")
    plt.show()

# 3. Number of options per question
val_df["num_options"] = val_df["options"].apply(lambda x: len(x))
test_df["num_options"] = test_df["options"].apply(lambda x: len(x))

plt.figure(figsize=(6,4))
val_df["num_options"].value_counts().sort_index().plot(kind="bar", title="Validation Set - Options Count")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6,4))
test_df["num_options"].value_counts().sort_index().plot(kind="bar", title="Test Set - Options Count")
plt.ylabel("Count")
plt.show()

# 4. Question length distribution
val_df["question_length"] = val_df["question"].apply(lambda x: len(x.split()))
test_df["question_length"] = test_df["question"].apply(lambda x: len(x.split()))

plt.figure(figsize=(6,4))
val_df["question_length"].hist(bins=30)
plt.title("Validation Set - Question Length Distribution")
plt.xlabel("Words per question")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(6,4))
test_df["question_length"].hist(bins=30)
plt.title("Test Set - Question Length Distribution")
plt.xlabel("Words per question")
plt.ylabel("Frequency")
plt.show()


