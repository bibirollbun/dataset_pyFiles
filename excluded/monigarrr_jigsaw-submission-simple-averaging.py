# --- Imports and Setup ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata
from datetime import datetime

# --- DEBUGGING CELL: List available files ---
import os

print("--- Files available in the input directory: ---")
# This will recursively list every file in every input directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
print("---------------------------------------------")

# --- Configuration ---
# 1. add the outputs of my 6 training notebooks as a data source
INPUT_DIR = "/kaggle/input/jigsaw-monigarr-strategy-1-ft-csvfiles/"
OUTPUT_DIR = "/kaggle/working/"

# List of MoniGarr’s submission files and their CV scores

# OOP AUC Scores for each model I fine-tuned
# each model has it's own training notebook
# each model has it's own training params in configs.py
MODELS = {
    "deberta-v2-xlarge": 0.81955,
    "deberta-v3-large": 0.82763,
    "deberta-v3-base": 0.83356,
    "roberta-large": 0.80837,
    "xlm-roberta-large": 0.70696,
    "electra-large-discriminator": 0.80272,
}

# Define a consistent suffix for your filenames
FILENAME_SUFFIX = "-finetuned"


# Load each submission file
subs = []
for model_name in MODELS.keys():
    
    file_path = f"{INPUT_DIR}submission_{model_name}{FILENAME_SUFFIX}.csv"
    sub_df = pd.read_csv(file_path)

    # --- DATA SANITY CHECK ---
    assert sub_df['rule_violation'].min() >= 0.0, f"{model_name} has negative predictions!"
    assert sub_df['rule_violation'].max() <= 1.0, f"{model_name} has predictions > 1!"
    
    # Rename the prediction column to be model-specific
    sub_df.rename(columns={'rule_violation': model_name}, inplace=True)
    subs.append(sub_df)

# Merge all predictions into a single DataFrame
ensemble_df = pd.DataFrame({'row_id': subs[0]['row_id']})
for sub in subs:
    ensemble_df = pd.merge(ensemble_df, sub, on='row_id')

print("Ensemble DataFrame created and validated successfully!")
ensemble_df.head()


# Calculate the correlation matrix for the model predictions
corr = ensemble_df[MODELS.keys()].corr()

# Plot a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.4f')
plt.title('Prediction Correlation Heatmap')
plt.show()


# Calculate the simple average
ensemble_df['simple_avg'] = ensemble_df[MODELS.keys()].mean(axis=1)

# Create the submission file
submission_simple = ensemble_df[['row_id', 'simple_avg']].copy()
submission_simple.rename(columns={'simple_avg': 'rule_violation'}, inplace=True)

# Save to the writable output directory
submission_simple.to_csv('/kaggle/working/submission.csv', index=False)

print("Simple Average submission file created.")
submission_simple.head()

