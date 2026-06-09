import pandas as pd
from pathlib import Path

# --- 1. Configuration ---
# ❗️ IMPORTANT: Add the outputs of your base notebooks as data sources to this one
INPUT_DIR = Path('/kaggle/input/blendmy/')

# List your submission files
sub_files = {
    'sub_A': INPUT_DIR / 'submission (19).csv',
    'sub_B': INPUT_DIR / 'submission (21).csv',
    'sub_C': INPUT_DIR / 'submission (23).csv'
}

# --- 2. Helper Functions ---
def rank_to_score(sr):
    """Converts ranks (lower is better) to scores (higher is better)."""
    return 1 / sr

def score_to_rank(s):
    """Converts scores back to ranks."""
    return s.rank(method='first', ascending=False).astype(int)

# --- 3. Blending Logic ---
print("Loading submission files...")
# Load all submission files into a list
dfs = [pd.read_csv(path) for path in sub_files.values()]

# Convert ranks to scores
score_frames = []
for i, df in enumerate(dfs):
    tmp = df[['Id', 'ranker_id', 'selected']].copy()
    tmp['score'] = tmp.groupby('ranker_id')['selected'].transform(rank_to_score)
    score_frames.append(tmp[['Id', 'ranker_id', 'score']].rename(columns={'score': f'score_{i}'}))

# Merge all score dataframes together
merged = score_frames[0]
for frame in score_frames[1:]:
    merged = merged.merge(frame, on=['Id', 'ranker_id'], how='left')

# --- 4. Weighted Averaging ---
# Define weights for each submission. Give more weight to your best-performing models.
weights = [0.70, 0.25, 0.10] # Corresponds to sub_A, sub_B, sub_C
score_cols = [f'score_{i}' for i in range(len(dfs))]
w = pd.Series(weights, index=score_cols)

# Compute the weighted average score
merged['final_score'] = (merged[score_cols] * w).sum(axis=1) / w.sum()

# Convert the final blended scores back to ranks
merged['selected'] = merged.groupby('ranker_id')['final_score'].transform(score_to_rank)

# --- 5. Save Final Submission ---
final_submission = merged[['Id', 'ranker_id', 'selected']]
final_submission.to_csv("submission.csv", index=False)
print("✅ Final blended submission created successfully!")

