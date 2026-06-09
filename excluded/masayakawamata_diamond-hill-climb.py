import pandas as pd
import numpy as np
import glob
import os
from functools import reduce
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

# --- 0. Load OOF and Test Predictions ---
oof_files = sorted(glob.glob('/kaggle/input/**/oof_*.csv'))
test_files = sorted(glob.glob('/kaggle/input/**/test_*.csv'))

print("Found OOF files:")
if not oof_files:
    print(" - None")
else:
    for f in oof_files:
        print(f" - {f}")

print("\nFound Test prediction files:")
if not test_files:
    print(" - None")
else:
    for f in test_files:
        print(f" - {f}")

oof_df = None
if oof_files:
    oof_dfs = []
    for file_path in oof_files:
        df = pd.read_csv(file_path)
        model_name = os.path.basename(file_path).replace('oof_', '').replace('.csv', '')
        df.rename(columns={'price': model_name}, inplace=True)
        oof_dfs.append(df)

    if oof_dfs:
        oof_df = reduce(lambda left, right: pd.merge(left, right, on='id'), oof_dfs)

    print("\n--- Combined OOF DataFrame ---")
    if oof_df is not None:
        print('OOF Shape:', oof_df.shape)
    else:
        print("Could not create a combined OOF dataframe.")

test_pred_df = None
if test_files:
    test_dfs = []
    for file_path in test_files:
        df = pd.read_csv(file_path, index_col='id')
        model_name = os.path.basename(file_path).replace('test_', '').replace('.csv', '')
        df.rename(columns={'price': model_name}, inplace=True)
        test_dfs.append(df)
    
    if test_dfs:
        test_pred_df = pd.concat(test_dfs, axis=1)

    print("\n--- Combined Test Prediction DataFrame ---")
    if test_pred_df is not None:
        print('Test Prediction Shape:', test_pred_df.shape)
    else:
        print("Could not create a combined test prediction dataframe.")

train = pd.read_csv('/kaggle/input/predicting-the-price-of-diamond/train.csv')
oof_df = pd.merge(oof_df, train[['id', 'price']], on='id')

y_true = oof_df['price']
model_cols = [col for col in oof_df.columns if col not in ['id', 'price']]
test_pred_df = test_pred_df[model_cols]


# --- 1. Hill Climbing Algorithm for R2 Score Maximization ---
print("\n--- Starting Hill Climbing to Maximize R2 Score ---")

best_score = -np.inf  
best_weights = {}
best_ensemble_pred_oof = None
history = []

print("Step 0: Finding the best single model to initialize the ensemble...")
for model in model_cols:
    score = r2_score(y_true, oof_df[model])
    if score > best_score: 
        best_score = score
        best_weights = {model: 1.0}
        best_ensemble_pred_oof = oof_df[model].copy()

print(f"Best initial model: '{list(best_weights.keys())[0]}' with R2 Score: {best_score:.5f}\n")
history.append({'step': 0, 'model_added': list(best_weights.keys())[0], 'score': best_score, 'weights': best_weights.copy()})


for i in range(1, len(model_cols) * 2): 
    print(f"Step {i}: Searching for the best model to blend...")
    
    potential_next_model = None
    potential_best_weight = 0
    score_to_beat = best_score 
    
    for model in model_cols:
        for w in np.arange(0.01, 1.00, 0.01):
            current_pred = (1 - w) * best_ensemble_pred_oof + w * oof_df[model]
            current_score = r2_score(y_true, current_pred)
        
            if current_score > score_to_beat:
                score_to_beat = current_score
                potential_next_model = model
                potential_best_weight = w
    
    if potential_next_model:
        best_score = score_to_beat
        best_ensemble_pred_oof = (1 - potential_best_weight) * best_ensemble_pred_oof + potential_best_weight * oof_df[potential_next_model]
        
        for m in best_weights:
            best_weights[m] *= (1 - potential_best_weight)
        if potential_next_model in best_weights:
            best_weights[potential_next_model] += potential_best_weight
        else:
            best_weights[potential_next_model] = potential_best_weight
        
        print(f"  -> Blended '{potential_next_model}' with weight {potential_best_weight:.2f}. New Ensemble R2 Score: {best_score:.5f}")
        history.append({'step': i, 'model_blended': potential_next_model, 'score': best_score, 'weights': best_weights.copy()})
    else:
        print("No further improvement found. Stopping.")
        break


print("\n--- Hill Climbing Finished ---")
print(f"Final OOF R2 Score: {best_score:.5f}")

final_weights_sorted = sorted(best_weights.items(), key=lambda item: item[1], reverse=True)
print("Final model weights:")
for model, weight in final_weights_sorted:
    print(f"  - {model}: {weight:.4f}")

final_test_pred = np.zeros(len(test_pred_df))
for model, weight in best_weights.items():
    final_test_pred += test_pred_df[model] * weight
    
submission_df = pd.DataFrame({'id': test_pred_df.index, 'price': final_test_pred})
submission_df.to_csv('submission_r2_hc.csv', index=False)
print("\nSubmission file 'submission_r2_hc.csv' created successfully.")
print(submission_df.head())




