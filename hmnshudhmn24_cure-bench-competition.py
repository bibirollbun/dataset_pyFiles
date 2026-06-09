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


import pandas as pd
import json
import os
import re

# ===============================
# File Paths (Kaggle Input)
# ===============================
val_file = "/kaggle/input/cure-bench/curebench_valset_pharse1.jsonl"
test_file = "/kaggle/input/cure-bench/curebench_testset_phase1.jsonl"

# ===============================
# Load JSONL into DataFrame
# ===============================
def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except Exception as e:
                # skip malformed lines but report
                print(f"Warning: skipping malformed line: {e}")
    return pd.DataFrame(records)

val_df = load_jsonl(val_file)
test_df = load_jsonl(test_file)

print("Validation set shape:", val_df.shape)
print("Test set shape:", test_df.shape)
print("Validation columns:", list(val_df.columns))
print("Test columns:", list(test_df.columns))

# ===============================
# Helper: find label column in validation set (if present)
# ===============================
possible_label_names = [
    "answer", "label", "golden_answer", "correct_answer",
    "ground_truth", "target", "answer_key", "solution"
]
label_col = None
for name in possible_label_names:
    if name in val_df.columns:
        label_col = name
        break

if label_col:
    print("Found label column in validation set:", label_col)
else:
    print("No obvious label column found in validation set. Proceeding without eval.")

# ===============================
# Rule-Based Predictor (robust & deterministic)
# ===============================

def rule_based_prediction(row):
    q = str(row.get("question", "")).lower()
    options = row.get("options", {})

    # Normalize options to a dict if needed
    if isinstance(options, list):
        # convert list to dict with letters A,B,C...
        letters = [chr(ord('A') + i) for i in range(len(options))]
        options = {letters[i]: options[i] for i in range(len(options))}
    if not isinstance(options, dict):
        return "D" if "D" in options else (next(iter(options.keys())) if options else "D")

    excluded = set()

    # ---- Rule: Avoid celecoxib in poor CYP2C9 metabolizers ----
    if "cyp2c9" in q and "poor" in q:
        for key, val in options.items():
            if isinstance(val, str) and "celecoxib" in val.lower():
                excluded.add(key)

    # ---- Rule: Avoid aspirin for children when possible ----
    if any(tok in q for tok in ["child", "children", "pediatric", "paediatric", "kid", "childhood"]):
        for key, val in options.items():
            if isinstance(val, str) and "aspirin" in val.lower():
                excluded.add(key)

    # ---- Rule: If question explicitly asks for 'none' or mentions contraindication, prefer 'None of the above' ----
    if any(token in q for token in ["contraindicated", "contraindication", "not recommended", "avoid"]):
        for key, val in options.items():
            if isinstance(val, str) and "none" in val.lower():
                return key

    # ---- Rule: If an option's main drug token appears in the question, pick that option ----
    for key, val in options.items():
        if not isinstance(val, str):
            continue
        val_clean = val.strip().lower()
        if not val_clean:
            continue
        # Split option text into alphanumeric tokens (drop short tokens)
        tokens = [t for t in re.split(r"\W+", val_clean) if len(t) >= 3]
        for t in tokens:
            if t in q and key not in excluded:
                return key

    # ---- Fallback: pick first non-excluded, non-empty option deterministically ----
    for key, val in options.items():
        if key in excluded:
            continue
        if isinstance(val, str) and val.strip():
            return key

    # ---- If all options excluded or empty, pick 'D' if exists else first option key ----
    if "D" in options:
        return "D"
    if len(options) > 0:
        return next(iter(options.keys()))
    return "D"

# Apply predictions
val_df["prediction"] = val_df.apply(rule_based_prediction, axis=1)
test_df["prediction"] = test_df.apply(rule_based_prediction, axis=1)

# ===============================
# Evaluation on validation set (if label available)
# ===============================
if label_col:
    # normalize label values to strings
    val_labels = val_df[label_col].astype(str).str.strip().str.upper()
    val_preds = val_df["prediction"].astype(str).str.strip().str.upper()
    acc = (val_labels == val_preds).mean()
    print(f"Validation accuracy: {acc:.4f} ({int((val_labels == val_preds).sum())}/{len(val_labels)})")
    # Show a few mismatch examples for manual inspection
    mismatches = val_df[val_labels != val_preds][["id", "question", label_col, "prediction"]].head(10)
    if not mismatches.empty:
        print("Sample mismatches (for inspection):")
        print(mismatches.to_string(index=False))

# ===============================
# Prepare Submission File (Test Set Only)
# ===============================
submission = test_df[["id", "prediction"]]

# Ensure output directory
os.makedirs("/kaggle/working", exist_ok=True)

submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)

print("Submission file saved at:", submission_path)
print(submission.head(10).to_string(index=False))

