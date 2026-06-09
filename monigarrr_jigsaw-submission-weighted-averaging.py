# --- Imports and Setup ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata
from datetime import datetime

# --- Configuration ---
# 1. add the outputs of my 6 training notebooks as a data source
INPUT_DIR = "/kaggle/input/strategy-1-final-submissions/"

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


# Define weights based on CV scores (you can experiment with these)
weights = np.array([score for score in MODELS.values()]) # Squaring scores gives more weight to top performers
# Normalize weights to sum to 1
normalized_weights = weights / np.sum(weights)

# Calculate the weighted average
ensemble_df['weighted_avg'] = np.average(ensemble_df[MODELS.keys()], weights=normalized_weights, axis=1)

# Create the submission file
submission_weighted = ensemble_df[['row_id', 'weighted_avg']].copy()
submission_weighted.rename(columns={'weighted_avg': 'rule_violation'}, inplace=True)

# Save to the writable output directory
submission_weighted.to_csv('/kaggle/working/submission.csv', index=False)

print("Weighted Average submission file created.")
submission_weighted.head()


# --- PRE-SUBMISSION CHECKLIST ---
print("--- Running Final Sanity Checks ---")
# Create a copy to avoid modifying the original
final_submission_df = submission_weighted.copy()

# 1. Check for Missing Values (NaNs)
if final_submission_df.isnull().values.any():
    print("ERROR: Submission file contains NaN values! Filling with 0.5")
    final_submission_df.fillna(0.5, inplace=True) # A common way to handle NaNs
else:
    print("No missing values found.")

# 2. Check Data Types
print("\nVerifying data types:")
print(final_submission_df.dtypes)
assert 'int' in str(final_submission_df['row_id'].dtype), "row_id column is not an integer!"
assert 'float' in str(final_submission_df['rule_violation'].dtype), "rule_violation column is not a float!"
print("Data types are correct.")

# 3. Check Prediction Range
min_pred = final_submission_df['rule_violation'].min()
max_pred = final_submission_df['rule_violation'].max()
print(f"\nPrediction range: [{min_pred:.4f}, {max_pred:.4f}]")
assert min_pred >= 0.0, "Found predictions less than 0!"
assert max_pred <= 1.0, "Found predictions greater than 1!"
print("Prediction range is valid.")

# 4. Check Row and Column Count
try:
    sample_sub = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv')
    print(f"\nSubmission shape: {final_submission_df.shape}")
    print(f"Sample shape:     {sample_sub.shape}")
    assert final_submission_df.shape == sample_sub.shape, "Submission shape does not match sample!"
    print("Shape is correct.")
except FileNotFoundError:
    print("\nCould not find sample_submission.csv to check shape. Skipping.")

print("\n--- All checks passed! Saving file. ---")

# Save the SANITIZED DataFrame
final_submission_df.to_csv('submission.csv', index=False)

