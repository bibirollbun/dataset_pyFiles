import pandas as pd

# List of your submission files (update filenames if needed)
files = [
    "/kaggle/input/example-notebook-evaluate-othello-boards/submission.csv",
    "/kaggle/input/starter-notebook-v1/submission_enhanced.csv",
    "/kaggle/input/starter-notebook-v1/submission_single.csv",
    "/kaggle/input/v2-starter-notebook/submission.csv",
    "/kaggle/input/notebook8e7795cff8/submission_pytorch.csv"
]

# Load all CSVs
dfs = [pd.read_csv(f) for f in files]

# Merge them on 'id'
merged_df = dfs[0]
for i in range(1, len(dfs)):
    merged_df = pd.merge(merged_df, dfs[i], on="id", suffixes=('', f'_{i}'))

# Collect all 'turn_player_advantage' columns
adv_cols = [col for col in merged_df.columns if 'turn_player_advantage' in col]

# ====== Weights ======
# Equal weights by default, but you can edit this list
weights = [0.3,0.1,0.2,0.1,0.3] #[1/len(adv_cols)] * len(adv_cols)   # e.g., [0.2,0.2,0.2,0.2,0.2]

# Weighted ensemble
merged_df['turn_player_advantage'] = sum(
    merged_df[adv_cols[i]] * weights[i] for i in range(len(weights))
)

# Keep only final submission columns
final_df = merged_df[['id', 'turn_player_advantage']]

# Save ensemble submission
final_df.to_csv("/kaggle/working/submission.csv", index=False)

print("Ensemble submission created successfully!")




