import pandas as pd
import lightgbm as lgb
import optuna
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from typing import Dict, Any


train_df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")


train_df.head()


train_df.columns



# LB scores from your files
lb_scores = {
    'submission_26.38218': 26.38218,
    'submission_26.38229': 26.38229,
    'submission_26.38299': 26.38299,
    'submission_26.38304': 26.38304,
    'submission_26.38305': 26.38305
}

def weighted_ensemble():
    base_path = "/kaggle/input/beats-per-minute-prediction/"
    
    # Glob all files
    files = glob.glob(os.path.join(base_path, "submission_26.38*.csv"))
    
    if len(files) != 5:
        print(f"Found {len(files)} files, expected 5")
        # Fallback manual list
        files = [
            os.path.join(base_path, 'submission_26.38218.csv'),
            os.path.join(base_path, 'submission_26.38229.csv'), 
            os.path.join(base_path, 'submission_26.38299.csv'),
            os.path.join(base_path, 'submission_26.38304.csv'),
            os.path.join(base_path, 'submission_26.38305.csv')
        ]
    
    submissions = []
    weights = []
    
    print("Loading submissions:")
    for file_path in files:
        df = pd.read_csv(file_path)
        
        # Get filename without extension + path
        filename = os.path.basename(file_path).replace(".csv", "")
        
        # Get LB score
        lb_score = lb_scores.get(filename, 26.38250)
        
        # Calculate weight (lower LB = higher weight)
        weight = 1.0 / lb_score
        
        submissions.append(df['BeatsPerMinute'].values)
        weights.append(weight)
        
        print(f"  {filename}: LB {lb_score:.5f}")
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    print(f"\nWeights assigned:")
    for i, file_path in enumerate(files):
        filename = os.path.basename(file_path).replace(".csv", "")
        lb_score = lb_scores.get(filename, 26.38250)
        print(f"  {filename} (LB: {lb_score:.5f}): {weights[i]:.3f}")
    
    # Calculate weighted average
    predictions_array = np.array(submissions)
    weighted_predictions = np.average(predictions_array, axis=0, weights=weights)
    
    # Use first file's IDs for consistency
    base_df = pd.read_csv(files[0])
    result_df = pd.DataFrame({
        'id': base_df['id'],
        'BeatsPerMinute': weighted_predictions
    })
    
    # Save output
    result_df.to_csv("submission.csv", index=False)
    
    print(f"\nâœ… Weighted ensemble saved as 'submission.csv'")
    print(f"ðŸ“Š Statistics:")
    print(f"   Mean BPM: {weighted_predictions.mean():.4f}")
    print(f"   Std BPM:  {weighted_predictions.std():.4f}")
    print(f"   Range:    {weighted_predictions.min():.2f} - {weighted_predictions.max():.2f}")
    print(f"   Samples:  {len(weighted_predictions)}")
    
    return result_df

# Run
if __name__ == "__main__":
    weighted_ensemble()


