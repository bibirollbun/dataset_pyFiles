import pandas as pd

# Load the predictions
df1 = pd.read_csv('/kaggle/input/drw0719/submission_95109.csv')
df2 = pd.read_csv('/kaggle/input/drw0719/submission_95004.csv')
df3 = pd.read_csv('/kaggle/input/drw0719/submission_95002.csv')
df4 = pd.read_csv('/kaggle/input/drw0719/submission_94857.csv')

# Assign weights (customize these)
weights = {
    'df1': 0.85,
    'df2': 0.05,
    'df3': 0.05,
    'df4': 0.05
}

# Rename prediction columns to avoid collision
df1.rename(columns={'prediction': 'df1'}, inplace=True)
df2.rename(columns={'prediction': 'df2'}, inplace=True)
df3.rename(columns={'prediction': 'df3'}, inplace=True)
df4.rename(columns={'prediction': 'df4'}, inplace=True)

# Merge all on 'ID'
df = df1.merge(df2, on='ID') \
        .merge(df3, on='ID') \
        .merge(df4, on='ID')

# Compute weighted average
df['prediction'] = (
    df['df1'] * weights['df1'] +
    df['df2'] * weights['df2'] +
    df['df3'] * weights['df3'] +
    df['df4'] * weights['df4']
)

# Keep only ID and final prediction
submission = df[['ID', 'prediction']]

# Save to CSV
submission.to_csv('final_ensemble_submission.csv', index=False)

