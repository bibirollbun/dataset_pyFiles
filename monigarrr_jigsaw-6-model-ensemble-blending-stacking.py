# --- Imports and Setup ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata

# --- Configuration ---
# This notebook uses a private dataset of pre-computed model predictions.
# will be public after the competition is complete.
INPUT_DIR = "/kaggle/input/jigsaw-monigarr-strategy-1-ft-csvfiles/"

# The models included in the dataset and their OOF CV scores
MODELS = {
    "deberta-v2-xlarge": 0.81955,
    "deberta-v3-large": 0.82763,
    "deberta-v3-base": 0.83356,
    "roberta-large": 0.80837,
    "xlm-roberta-large": 0.70696,
    "electra-large-discriminator": 0.80272,
}


# Load each submission file
subs = []
for model_name in MODELS.keys():
    file_path = f"{INPUT_DIR}submission_{model_name}-finetuned.csv"
    sub_df = pd.read_csv(file_path)
    
    # --- Data Sanity Check ---
    assert sub_df['rule_violation'].min() >= 0.0, f"{model_name} has negative predictions!"
    assert sub_df['rule_violation'].max() <= 1.0, f"{model_name} has predictions > 1!"
    
    sub_df.rename(columns={'rule_violation': model_name}, inplace=True)
    subs.append(sub_df)

# Merge all predictions into a single DataFrame
ensemble_df = pd.DataFrame({'row_id': subs[0]['row_id']})
for sub in subs:
    ensemble_df = pd.merge(ensemble_df, sub, on='row_id')

print("Ensemble DataFrame created and validated successfully!")
ensemble_df.head()


# Calculate the correlation matrix
corr = ensemble_df[list(MODELS.keys())].corr()

# Plot the heatmap
plt.figure(figsize=(12, 9))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.4f', linewidths=.5)
plt.title('Prediction Correlation Heatmap', size=16)
plt.show()


# --- Simple Average ---
ensemble_df['simple_avg'] = ensemble_df[list(MODELS.keys())].mean(axis=1)
submission_simple = ensemble_df[['row_id', 'simple_avg']].rename(columns={'simple_avg': 'rule_violation'})
submission_simple.to_csv('submission_simple_avg.csv', index=False)
print("Simple Average submission created.")

# --- Weighted Average ---
weights = np.array([score**2 for score in MODELS.values()]) # Squaring scores gives more weight to top performers
normalized_weights = weights / np.sum(weights)
ensemble_df['weighted_avg'] = np.average(ensemble_df[list(MODELS.keys())], weights=normalized_weights, axis=1)
submission_weighted = ensemble_df[['row_id', 'weighted_avg']].rename(columns={'weighted_avg': 'rule_violation'})
submission_weighted.to_csv('submission_weighted_avg.csv', index=False)
print("Weighted Average submission created.")

# --- Rank Average ---
for model_name in MODELS.keys():
    ensemble_df[f'rank_{model_name}'] = rankdata(ensemble_df[model_name])
rank_cols = [f'rank_{model_name}' for model_name in MODELS.keys()]
ensemble_df['rank_avg'] = ensemble_df[rank_cols].mean(axis=1)
ensemble_df['rank_avg_norm'] = (ensemble_df['rank_avg'] - ensemble_df['rank_avg'].min()) / (ensemble_df['rank_avg'].max() - ensemble_df['rank_avg'].min())
submission_rank = ensemble_df[['row_id', 'rank_avg_norm']].rename(columns={'rank_avg_norm': 'rule_violation'})
submission_rank.to_csv('submission_rank_avg.csv', index=False)
print("Rank Average submission created.")

print("\n--- Final Blended Predictions ---")
ensemble_df[['row_id', 'simple_avg', 'weighted_avg', 'rank_avg_norm']].head()


import pandas as pd

# Define the path to the file created by your previous notebook
input_filename = "/kaggle/working/submission_simple_avg.csv"
#input_filename = "/kaggle/working/submission_rank_avg.csv"
#input_filename = "/kaggle/working/submission_weighted_avg.csv"

# Define the required final submission filename
output_filename = "/kaggle/working/submission.csv"

# Read your generated file
submission_df = pd.read_csv(input_filename)

# Save it with the correct name for the competition
submission_df.to_csv(output_filename, index=False)

print(f"Successfully created final submission file: {output_filename}")
submission_df.head()

