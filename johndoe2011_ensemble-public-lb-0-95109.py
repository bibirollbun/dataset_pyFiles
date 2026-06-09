import pandas as pd

# Load the predictions
ensemble_only = pd.read_csv('/kaggle/input/drw-lgbm/ensemble_only_prediction.csv')
prophet_alt = pd.read_csv('/kaggle/input/drw-lgbm/prophet_alt_prediction.csv')
prophet_default = pd.read_csv('/kaggle/input/drw-lgbm/prophet_default_prediction.csv')
submission_adv = pd.read_csv('/kaggle/input/drw-lgbm/submission_advanced_blend.csv')

# Assign weights (customize these)
weights = {
    'ensemble_only': 0.5,
    'prophet_alt': 0.3,
    'prophet_default': 0.25,
    'submission_adv': 0.55
}

# Rename prediction columns to avoid collision
ensemble_only.rename(columns={'prediction': 'ensemble_only_pred'}, inplace=True)
prophet_alt.rename(columns={'prediction': 'prophet_alt_pred'}, inplace=True)
prophet_default.rename(columns={'prediction': 'prophet_default_pred'}, inplace=True)
submission_adv.rename(columns={'prediction': 'submission_adv_pred'}, inplace=True)

# Merge all on 'ID'
df = ensemble_only.merge(prophet_alt, on='ID') \
                  .merge(prophet_default, on='ID') \
                  .merge(submission_adv, on='ID')

# Compute weighted average
df['prediction'] = (
    df['ensemble_only_pred'] * weights['ensemble_only'] +
    df['prophet_alt_pred'] * weights['prophet_alt'] +
    df['prophet_default_pred'] * weights['prophet_default'] +
    df['submission_adv_pred'] * weights['submission_adv']
)

# Keep only ID and final prediction
submission = df[['ID', 'prediction']]

# Save to CSV
submission.to_csv('final_ensemble_submission.csv', index=False)

