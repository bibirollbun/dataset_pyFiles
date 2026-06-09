%load_ext autoreload
%autoreload 2


import os
import glob
import warnings

# --- Third-party
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from experiment_setup import ExperimentSetup

# --- Notebook settings
warnings.filterwarnings('ignore')

%matplotlib inline


helper = ExperimentSetup()

# Get the seed, and apply it to all the internals like pandas and numpy
seed = helper.set_seeds()

helper.configure_pandas()
helper.suppress_warnings()

# The target feature
TARGET = 'diagnosed_diabetes'

# A split in the data has been described here: https://www.kaggle.com/code/masayakawamata/s5e12-xgb-bridging-the-cv-lb-gap
# This is the index identified above; we'll use it to split the data
ORIGINAL_START_INDEX = 678260

# Use manual weights (See explaination below)
USE_MANUAL_WEIGHTS = True


# Load Ground Truth
training_df = helper.read_training_dataset()
y_true = training_df[TARGET].values

# Train the blender only on post-split OOF rows
post_mask = training_df['id'] >= ORIGINAL_START_INDEX


submission_df = helper.read_sample_submission_dataset()


oof_files = []
test_files = []

if helper.running_in_kaggle():
    pred_dir = '/kaggle/input/ps-s5e12-*/predictions'
else:
    pred_dir = 'predictions'

oof_files = sorted(glob.glob(f'{pred_dir}/*_oof_preds.csv'))
test_files = sorted(glob.glob(f'{pred_dir}/*_test_preds.csv'))

print(f"Found {len(oof_files)} OOF files and {len(test_files)} Test files.")


# Helper to load and merge predictions
def load_preds(file_list, index_col='id'):
    df_list = []
    for file in file_list:
        model_name = os.path.basename(file).replace('_oof_preds.csv', '').replace('_test_preds.csv', '')
        # Read file
        df = pd.read_csv(file)
        
        # If files contain 'id' and 'pred', index by id. 
        # Assuming the column naming convention is standard (e.g. 'pred_xgb', 'diagnosed_diabetes', or similar)
        # We will dynamically find the prediction column (not 'id' or 'target')
        pred_col = [c for c in df.columns if c not in ['id', TARGET]][0]
        
        df = df.rename(columns={pred_col: model_name})
        df = df.set_index(index_col)[model_name]
        df_list.append(df)
        
    return pd.concat(df_list, axis=1)


# Create DataFrames
oof_df = load_preds(oof_files)
test_df = load_preds(test_files)

# Train the blender only on post-split OOF rows
oof_df = oof_df[post_mask]
y_true = y_true[post_mask]

# Ensure OOF aligns with y_true (sort by index if needed)
oof_df = oof_df.sort_index()
print(f"OOF Shape: {oof_df.shape}")
print(f"Test Shape: {test_df.shape}")


# CDF (Rank) Transformation
# This normalizes predictions to [0, 1] based on their rank, 
# making them comparable distributions (uniform) before blending.
print("\nApplying CDF (Rank) Transformation...")

def get_cdf(df):
    return df.apply(lambda x: rankdata(x) / len(x))

oof_cdf = get_cdf(oof_df)
test_cdf = get_cdf(test_df)

# Check individual scores after transformation
print("\nIndividual Model AUC (Ranked):")
best_single_model = None
best_single_score = 0

for col in oof_cdf.columns:
    score = roc_auc_score(y_true, oof_cdf[col])
    print(f"{col}: {score:.6f}")
    if score > best_single_score:
        best_single_score = score
        best_single_model = col


# Hill Climbing Algorithm (Caruana et al.)
# Iteratively adds the model that maximizes the ensemble AUC
def hill_climbing(oof_pred_df, y, iterations=100, verbose=False):
    # Initialize with the best single model
    current_preds = oof_pred_df[best_single_model].copy()
    
    # Store counts of each model in the ensemble (represents weights)
    # Start with 1 count for the best model
    model_counts = {col: 0 for col in oof_pred_df.columns}
    model_counts[best_single_model] += 1
    
    history = [best_single_score]
    
    print(f"\nStarting Hill Climbing for {iterations} iterations...")
    
    for i in range(iterations):
        best_step_score = -1
        best_step_col = None
        
        # Try adding each model to the current ensemble
        for col in oof_pred_df.columns:
            # We don't need to divide by (i+2) for AUC calculation as it's rank-invariant,
            # but we sum them to keep the magnitude increasing linearly.
            # Simulates: new_ensemble = (current_sum + new_model_preds)
            temp_preds = current_preds + oof_pred_df[col]
            
            score = roc_auc_score(y, temp_preds)
            
            if score > best_step_score:
                best_step_score = score
                best_step_col = col
        
        # Update the ensemble with the winner of this round
        current_preds += oof_pred_df[best_step_col]
        model_counts[best_step_col] += 1
        history.append(best_step_score)
        
        if verbose and i % 10 == 0:
            print(f"Iter {i+1}: Added {best_step_col} -> AUC: {best_step_score:.6f}")
            
    # Calculate final weights
    total_counts = sum(model_counts.values())
    weights = {k: v / total_counts for k, v in model_counts.items()}
    
    return weights, history


# Run Optimization
weights, history = hill_climbing(oof_cdf, y_true, iterations=200, verbose=True)

# Results & Submission
print("\nOptimal Blending Weights:")
for model, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
    if weight > 0:
        print(f"{model}: {weight:.4f}")

# Plot history
plt.figure(figsize=(10, 5))
plt.plot(history)
plt.title("Hill Climbing Optimization")
plt.xlabel("Iteration")
plt.ylabel("OOF AUC")
plt.grid(True, alpha=0.3)
plt.show()


if USE_MANUAL_WEIGHTS:
    # Manually override weight as discussed above
    weights = {
        'cb': 0.45,
        'lgb': 0.40,
        'xgb': 0.15
    }
    print('Overriding using manual weights:\n', weight)


# Apply weights to Test Predictions
# Note: We apply weights to the CDF (Ranked) test predictions
final_test_preds = np.zeros(len(test_cdf))
for model, weight in weights.items():
    final_test_preds += test_cdf[model] * weight

# Save Submission
submission_df[TARGET] = final_test_preds

submission_df.to_csv('submission.csv', index=False)
print('Saved: submission.csv')

print('SUBMISSION')
print('==========')

print(submission_df.head(10))




