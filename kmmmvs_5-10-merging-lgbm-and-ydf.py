# Code inspired by and adapted from @anthonytherrien and @mikhailnaumov.
# Special thanks for their valuable open-source contributions and insights!


import pandas as pd

# 1. Load the new submission (current code result) and the previous best submission (private leaderboard data)
current_df = pd.read_csv('/kaggle/input/ps-s5e10-lightgbm-cb-ensemble/submission.csv')
previous_df = pd.read_csv('/kaggle/input/road-risk-single-ydf/submission.csv')

# 2. Merge predictions by id (accurate row-wise alignment)
merge_df = current_df.merge(previous_df, on='id', how='inner', suffixes=('_current', '_previous'))

# 3. Auto-weight calculation based on public leaderboard scores (LB)
current_lb = 0.05541    # Example - current code leaderboard score (lower is better)
previous_lb = 0.05541   # Example - previous submission leaderboard score
W_current = 1/current_lb / (1/current_lb + 1/previous_lb)
W_previous = 1/previous_lb / (1/current_lb + 1/previous_lb)
print(f"Auto Ensemble Weights: current={W_current:.5f}, previous={W_previous:.5f}")

# 4. Weighted blending of the two prediction values
merge_df['accident_risk'] = (
    W_current * merge_df['accident_risk_current'] +
    W_previous * merge_df['accident_risk_previous']
)

# 5. Clipping to [0, 1] range (for safety)
merge_df['accident_risk'] = merge_df['accident_risk'].clip(0, 1)

# 6. Save with submission format (id, accident_risk)
merge_df[['id', 'accident_risk']].to_csv('submission.csv', index=False)


