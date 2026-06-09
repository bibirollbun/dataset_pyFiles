import pandas as pd
import numpy as np
from pathlib import Path

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


def blend_submissions(weight_dict, output_path, blend_method='weighted_average'):
    """
    Blend multiple submission files with specified weights.
    
    Parameters:
    -----------
    weight_dict : dict
        Dictionary with file paths as keys and weights as values
    output_path : str
        Path where the blended submission will be saved
    blend_method : str
        Method for blending: 'weighted_average', 'geometric_mean', or 'rank_average'
    
    Returns:
    --------
    pd.DataFrame
        The blended submission dataframe
    """
    
    # Input validation
    if not weight_dict:
        raise ValueError("weight_dict cannot be empty")
    
    if not all(weight > 0 for weight in weight_dict.values()):
        raise ValueError("All weights must be positive")
    
    # Read and process all submissions
    submissions = []
    for path, weight in weight_dict.items():
        try:
            df = pd.read_csv(path)
            # Validate required columns
            if 'id' not in df.columns or 'loan_paid_back' not in df.columns:
                raise ValueError(f"File {path} must contain 'id' and 'loan_paid_back' columns")
            
            # Validate prediction range
            if not ((df['loan_paid_back'] >= 0) & (df['loan_paid_back'] <= 1)).all():
                raise ValueError(f"Predictions in {path} must be between 0 and 1")
                
            submissions.append({
                'df': df,
                'weight': weight,
                'name': Path(path).stem
            })
        except FileNotFoundError:
            raise FileNotFoundError(f"Submission file not found: {path}")
    
    # Use the first submission as base
    base_df = submissions[0]['df'][['id']].copy()
    
    # Apply different blending methods
    if blend_method == 'weighted_average':
        blended_pred = _weighted_average_blend(submissions, base_df)
    elif blend_method == 'geometric_mean':
        blended_pred = _geometric_mean_blend(submissions, base_df)
    elif blend_method == 'rank_average':
        blended_pred = _rank_average_blend(submissions, base_df)
    else:
        raise ValueError(f"Unknown blend method: {blend_method}")
    
    # Create final submission
    blended_df = base_df.copy()
    blended_df['loan_paid_back'] = blended_pred
    
    # Validate final predictions
    if not ((blended_df['loan_paid_back'] >= 0) & (blended_df['loan_paid_back'] <= 1)).all():
        print("âš ï¸�  Warning: Some blended predictions are outside [0,1] range. Clipping...")
        blended_df['loan_paid_back'] = blended_df['loan_paid_back'].clip(0, 1)
    
    # Save results
    blended_df.to_csv(output_path, index=False)
    
    # Print summary statistics
    _print_blend_summary(blended_df, submissions, blend_method)
    
    print(f"âœ… Blended submission saved to {output_path}")
    return blended_df


def _weighted_average_blend(submissions, base_df):
    """Calculate weighted average of predictions."""
    total_weight = sum(sub['weight'] for sub in submissions)
    weighted_sum = 0
    
    for sub in submissions:
        # Ensure proper alignment by merging on id
        merged = base_df.merge(sub['df'][['id', 'loan_paid_back']], on='id', how='left')
        weighted_sum += merged['loan_paid_back'] * sub['weight']
    
    return weighted_sum / total_weight


def _geometric_mean_blend(submissions, base_df):
    """Calculate weighted geometric mean of predictions."""
    log_sum = 0
    total_weight = sum(sub['weight'] for sub in submissions)
    
    for sub in submissions:
        merged = base_df.merge(sub['df'][['id', 'loan_paid_back']], on='id', how='left')
        # Add small epsilon to avoid log(0)
        log_sum += np.log(merged['loan_paid_back'] + 1e-8) * sub['weight']
    
    return np.exp(log_sum / total_weight)


def _rank_average_blend(submissions, base_df):
    """Calculate weighted average of ranks."""
    total_weight = sum(sub['weight'] for sub in submissions)
    rank_sum = 0
    
    for sub in submissions:
        merged = base_df.merge(sub['df'][['id', 'loan_paid_back']], on='id', how='left')
        rank_sum += merged['loan_paid_back'].rank(pct=True) * sub['weight']
    
    return rank_sum / total_weight


def _print_blend_summary(blended_df, submissions, blend_method):
    """Print summary statistics of the blending operation."""
    print(f"\nğŸ“Š Blending Summary ({blend_method})")
    print("-" * 40)
    
    # Print individual submission weights and stats
    for i, sub in enumerate(submissions, 1):
        preds = sub['df']['loan_paid_back']
        print(f"{i}. {sub['name']}: "
              f"weight={sub['weight']:.2f}, "
              f"mean={preds.mean():.4f}, "
              f"std={preds.std():.4f}")
    
    # Print final blended statistics
    final_preds = blended_df['loan_paid_back']
    print(f"\nğŸ“ˆ Blended Result:")
    print(f"   Mean: {final_preds.mean():.4f}")
    print(f"   Std:  {final_preds.std():.4f}")
    print(f"   Min:  {final_preds.min():.4f}")
    print(f"   Max:  {final_preds.max():.4f}")
    print(f"   Samples: {len(blended_df):,}")




def main():
    weight_dict = {
        "/kaggle/input/predicting-loan-payback-vault/submission.csv": 0.66,
        "/kaggle/input/predicting-loan-payback-vault/submission (1).csv": 0.33,
    }
    blend_submissions(weight_dict, output_path="submission.csv")


if __name__ == "__main__":
    main()




